"""
Persistent HTTP service for the strif string-template engine.

Endpoints:
  POST /render                 Render a template (strict by default).
  POST /compile                Precompile and hash-cache a template.
  GET  /trace/{id}/stream      SSE stream of ordered decision-trace events.
  GET  /trace                  Browser page that renders live and records.

Run with:  uv run python -m strif.server
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .string_template import (
    MissingVariableError,
    RecursionLimitError,
    TemplateSyntaxError,
    compile_template,
    template_hash,
)
from .string_template import (
    render as render_template,
)

__all__ = ["CompileCache", "TraceManager", "build_server", "main"]


class CompileCache:
    """Hash-keyed compile cache with singleflight dedup.

    Concurrent requests for identical template content share one compilation;
    waiters reuse the first caller's future. The key is a content hash, so an
    edited template gets a new key and transparently misses the old entry.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, Any] = {}
        self._inflight: dict[str, Future[Any]] = {}
        self.compile_count = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "compile_count": self.compile_count,
                "cache_size": len(self._entries),
                "inflight": len(self._inflight),
            }

    def reset_stats(self) -> None:
        with self._lock:
            self.compile_count = 0

    def digest_of(self, template: str) -> str:
        return template_hash(template)

    def get_by_digest(self, digest: str) -> Any:
        with self._lock:
            return self._entries.get(digest)

    def get_or_compile(self, template: str) -> tuple[Any, bool]:
        """Return (compiled, freshly_compiled). Identical content compiles once
        even under concurrency; later callers await the leader's future."""
        digest = template_hash(template)
        leader = False
        new_future: Future[Any] | None = None
        with self._lock:
            cached = self._entries.get(digest)
            if cached is not None:
                return cached, False
            future = self._inflight.get(digest)
            if future is None:
                # The leader compiles inline (outside the lock); the future
                # is registered synchronously so all concurrent callers join
                # the same in-flight result.
                new_future = Future()
                future = new_future
                self._inflight[digest] = future
                leader = True
        if leader:
            assert new_future is not None
            try:
                compiled = compile_template(template)
            except BaseException as exc:
                with self._lock:
                    self._inflight.pop(digest, None)
                new_future.set_exception(exc)
                raise
            with self._lock:
                self.compile_count += 1
                self._entries[digest] = compiled
                self._inflight.pop(digest, None)
            new_future.set_result(compiled)
            return compiled, True
        return future.result(), False


@dataclass
class _Trace:
    trace_id: str
    digest: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    output: str | None = None
    error: dict[str, Any] | None = None
    subscribers: set[Any] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


class TraceManager:
    """Runs traced renders independently of SSE subscribers.

    Events are appended in token order and buffered per trace, so a client
    disconnect never interrupts the render and clients can reconnect with
    Last-Event-ID to resume from the breakpoint.
    """

    def __init__(self, cache: CompileCache, max_workers: int = 16) -> None:
        self.cache = cache
        self._traces: dict[str, _Trace] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="strif-render"
        )

    def get(self, trace_id: str) -> _Trace | None:
        with self._lock:
            return self._traces.get(trace_id)

    def start(
        self,
        *,
        template: str | None,
        digest: str | None,
        variables: dict[str, Any],
        recurse: bool,
        strict: bool,
    ) -> str:
        trace_id = uuid.uuid4().hex
        trace = _Trace(trace_id=trace_id, digest=digest)
        with self._lock:
            self._traces[trace_id] = trace
        self._executor.submit(self._run, trace, template, digest, variables, recurse, strict)
        return trace_id

    def _run(
        self,
        trace: _Trace,
        template: str | None,
        digest: str | None,
        variables: dict[str, Any],
        recurse: bool,
        strict: bool,
    ) -> None:
        def sink(event: dict[str, Any]) -> None:
            self._append(trace, event)

        try:
            compiled = self._resolve(template, digest)
            output = render_template(
                compiled,
                variables,
                strict=strict,
                recurse=recurse,
                sink=sink,
                compiler=lambda text: self.cache.get_or_compile(text)[0],
            )
            with trace.lock:
                trace.status = "done"
                trace.output = output
            self._append(trace, {"type": "result", "output": output, "digest": trace.digest})
            self._finish(trace, {"type": "done", "digest": trace.digest})
        except MissingVariableError as exc:
            self._fail(trace, "missing_variable", f"Missing variable: {exc}", 400)
        except RecursionLimitError as exc:
            self._fail(trace, "recursion_limit", str(exc), 409)
        except TemplateSyntaxError as exc:
            self._fail(trace, "syntax_error", str(exc), 422)
        except KeyError as exc:
            self._fail(trace, "unknown_digest", str(exc), 404)
        except Exception as exc:  # noqa: BLE001 - surfaced, never kills worker
            self._fail(trace, "internal_error", str(exc), 500, traceback.format_exc())

    def _resolve(self, template: str | None, digest: str | None) -> Any:
        if template is not None:
            compiled, _ = self.cache.get_or_compile(template)
            return compiled
        if not digest:
            raise TemplateSyntaxError("Either template or digest must be provided")
        compiled = self.cache.get_by_digest(digest)
        if compiled is None:
            raise KeyError(f"Unknown template digest: {digest}")
        return compiled

    def _fail(
        self, trace: _Trace, code: str, message: str, http_status: int, detail: str | None = None
    ) -> None:
        with trace.lock:
            trace.status = "error"
            trace.error = {"code": code, "message": message, "status": http_status}
        event: dict[str, Any] = {
            "type": "error",
            "code": code,
            "message": message,
            "status": http_status,
        }
        if detail:
            event["detail"] = detail
        self._append(trace, event)
        self._finish(trace, {"type": "done", "digest": trace.digest, "error": code})

    def _finish(self, trace: _Trace, event: dict[str, Any]) -> None:
        subscribers = self._append(trace, event)
        for queue_ref in subscribers:
            queue_ref.put(None)

    def _append(self, trace: _Trace, event: dict[str, Any]) -> set[Any]:
        with trace.lock:
            seq = len(trace.events) + 1
            stored = {"seq": seq, **event}
            trace.events.append(stored)
            subscribers = set(trace.subscribers)
        for queue_ref in subscribers:
            queue_ref.put(stored)
        return subscribers

    def subscribe(self, trace: _Trace) -> Any:
        q: Any = queue.Queue()
        with trace.lock:
            trace.subscribers.add(q)
        return q

    def unsubscribe(self, trace: _Trace, q: Any) -> None:
        with trace.lock:
            trace.subscribers.discard(q)


class _HttpError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        data = json.loads(raw.decode("utf-8") or "{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _HttpError(400, "invalid_json", str(exc)) from exc
    if not isinstance(data, dict):
        raise _HttpError(400, "invalid_request", "JSON body must be an object")
    return data


TRACE_PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>strif 模板判定轨迹</title>
<style>
body{font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;margin:24px;background:#0f1115;color:#d7dae0}
textarea,input{width:100%;box-sizing:border-box;background:#1a1d23;color:#d7dae0;border:1px solid #333a45;border-radius:6px;padding:8px}
textarea{height:80px}
button{background:#3b82f6;color:#fff;border:0;border-radius:6px;padding:8px 16px;margin:8px 8px 0 0;cursor:pointer}
button.secondary{background:#374151}
label{display:block;margin:10px 0 4px;color:#9aa4b2}
#log{margin-top:18px;border:1px solid #333a45;border-radius:6px;padding:12px;min-height:220px}
.ev{display:block;padding:1px 4px;border-radius:4px;white-space:pre-wrap}
.escape{background:#3f2d14;color:#fbbf24}.condition{background:#14331f;color:#4ade80}
.variable{background:#1d2742;color:#93c5fd}.error{background:#3d1518;color:#f87171}.done{color:#6b7280}
#status{margin-left:10px;color:#9aa4b2}
</style>
</head>
<body>
<h2>strif 模板渲染 · SSE 判定轨迹录屏</h2>
<label>模板（${name} 变量 · $${ 转义 · $if{x}…$else…$end 条件，可嵌套）</label>
<textarea id="tpl">$${user}=${user} $if{user}hello ${user}$else anonymous$end $if{admin}[$if{role}role=${role}$end]$end</textarea>
<label>变量（JSON 对象）</label>
<textarea id="vars">{"user":"alice","admin":"yes","role":"root"}</textarea>
<label><input type="checkbox" id="rec" checked style="width:auto"> 录屏（浏览器会询问共享哪个标签页，结束自动下载 webm）</label>
<label><input type="checkbox" id="recurse" style="width:auto"> recurse=true（变量值递归展开；自引用返回 409）</label>
<div>
<button id="go">渲染并录屏</button>
<button class="secondary" id="replay">按 Trace ID 重连</button>
<input id="traceid" placeholder="trace id（断点续推）" style="width:300px">
</div>
<span id="status"></span>
<div id="log"></div>
<script>
let es=null, recorder=null;
const $=id=>document.getElementById(id);
function setStatus(s){$('status').textContent=s;}
function addEvent(ev,id){
  const span=document.createElement('span');
  span.className='ev '+(ev.type==='escape'?'escape':ev.type.indexOf('condition')===0?'condition'
    :ev.type==='variable'?'variable':ev.type==='error'?'error':'done');
  span.textContent='#'+id+' ['+ev.type+'] '+JSON.stringify(ev);
  $('log').appendChild(span);
  window.scrollTo(0,document.body.scrollHeight);
}
async function startRecording(){
  if(!$('rec').checked)return null;
  const stream=await navigator.mediaDevices.getDisplayMedia({video:true,audio:false});
  const rec=new MediaRecorder(stream,{mimeType:'video/webm'});
  const chunks=[];
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
  rec.onstop=()=>{
    const blob=new Blob(chunks,{type:'video/webm'});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='strif-trace-'+Date.now()+'.webm';
    a.click();
    stream.getTracks().forEach(t=>t.stop());
  };
  rec.start();
  return rec;
}
function stopRecording(){if(recorder&&recorder.state!=='inactive')recorder.stop();recorder=null;}
async function watch(id){
  if(es)es.close();
  $('log').textContent='';
  es=new EventSource('/trace/'+id+'/stream');
  const names=['render_start','escape','variable','condition','condition_end',
    'recursion_limit','result','error','render_end','done','render_aborted'];
  let finished=false;
  for(const name of names){
    es.addEventListener(name,e=>{
      let ev={};
      try{ev=JSON.parse(e.data);}catch(_){}
      addEvent(ev,e.lastEventId);
      if(name==='done'||name==='error'){finished=true;setStatus('finished: '+name);es.close();stopRecording();}
    });
  }
  es.onerror=()=>{if(!finished)setStatus('连接断开，可用 Trace ID 重连续推…');};
  setStatus('watching trace '+id);
}
$('go').onclick=async()=>{
  let vars={};
  try{vars=JSON.parse($('vars').value||'{}');}catch(e){setStatus('变量 JSON 解析失败: '+e);return;}
  stopRecording();
  try{recorder=await startRecording();}catch(e){setStatus('录屏启动失败（仍可看轨迹）: '+e);}
  const resp=await fetch('/render',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({template:$('tpl').value,variables:vars,recurse:$('recurse').checked,trace:true})});
  const data=await resp.json();
  if(!resp.ok){setStatus('render 请求失败: '+JSON.stringify(data));stopRecording();return;}
  $('traceid').value=data.trace_id;
  watch(data.trace_id);
};
$('replay').onclick=()=>{const id=$('traceid').value.trim();if(id)watch(id);};
</script>
</body>
</html>
"""


def _make_handler(cache: CompileCache, traces: TraceManager) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "strif/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, exc: _HttpError) -> None:
            self._send_json(exc.status, {"error": exc.code, "message": exc.message})

        def do_GET(self) -> None:  # noqa: N802
            parts = [p for p in urlsplit(self.path).path.split("/") if p]
            try:
                if parts == ["health"]:
                    self._send_json(200, {"ok": True})
                elif parts == ["trace"]:
                    self._serve_page()
                elif len(parts) == 3 and parts[0] == "trace" and parts[2] == "stream":
                    self._serve_stream(parts[1])
                elif parts == ["debug", "stats"]:
                    self._send_json(200, cache.stats())
                else:
                    self._send_json(404, {"error": "not_found", "message": self.path})
            except _HttpError as exc:
                self._send_error(exc)

        def do_POST(self) -> None:  # noqa: N802
            parts = [p for p in urlsplit(self.path).path.split("/") if p]
            try:
                if parts == ["compile"]:
                    self._handle_compile()
                elif parts == ["render"]:
                    self._handle_render()
                elif parts == ["debug", "reset"]:
                    cache.reset_stats()
                    self._send_json(200, {"reset": True})
                else:
                    self._send_json(404, {"error": "not_found", "message": self.path})
            except _HttpError as exc:
                self._send_error(exc)

        def _handle_compile(self) -> None:
            data = _read_json(self)
            template = data.get("template")
            if not isinstance(template, str):
                raise _HttpError(400, "invalid_request", "'template' must be a string")
            try:
                compiled, fresh = cache.get_or_compile(template)
            except TemplateSyntaxError as exc:
                raise _HttpError(422, "syntax_error", str(exc)) from exc
            self._send_json(
                200,
                {
                    "digest": compiled.digest,
                    "compiled": fresh,
                    "cached": not fresh,
                    "compile_count": cache.compile_count,
                },
            )

        def _handle_render(self) -> None:
            data = _read_json(self)
            template = data.get("template")
            digest = data.get("digest")
            variables = data.get("variables", {})
            recurse = bool(data.get("recurse", False))
            strict_mode = data.get("mode", "render") != "replace"
            trace_requested = bool(data.get("trace", False))

            if template is not None and not isinstance(template, str):
                raise _HttpError(400, "invalid_request", "'template' must be a string")
            if digest is not None and not isinstance(digest, str):
                raise _HttpError(400, "invalid_request", "'digest' must be a string")
            if template is None and not digest:
                raise _HttpError(
                    400, "invalid_request", "Provide 'template' or a compiled 'digest'"
                )
            if not isinstance(variables, dict):
                raise _HttpError(400, "invalid_request", "'variables' must be an object")

            if digest is None:
                assert template is not None
                digest = cache.digest_of(template)

            if trace_requested:
                trace_id = traces.start(
                    template=template,
                    digest=digest,
                    variables=variables,
                    recurse=recurse,
                    strict=strict_mode,
                )
                self._send_json(
                    202,
                    {"trace_id": trace_id, "digest": digest, "stream": f"/trace/{trace_id}/stream"},
                )
                return

            try:
                if template is not None:
                    compiled, fresh = cache.get_or_compile(template)
                else:
                    compiled = cache.get_by_digest(digest)
                    fresh = False
                    if compiled is None:
                        raise _HttpError(404, "unknown_digest", f"Unknown digest: {digest}")
            except TemplateSyntaxError as exc:
                raise _HttpError(422, "syntax_error", str(exc)) from exc

            try:
                output = render_template(
                    compiled,
                    variables,
                    strict=strict_mode,
                    recurse=recurse,
                    compiler=lambda text: cache.get_or_compile(text)[0],
                )
            except MissingVariableError as exc:
                raise _HttpError(400, "missing_variable", f"Missing variable: {exc}") from exc
            except RecursionLimitError as exc:
                raise _HttpError(409, "recursion_limit", str(exc)) from exc
            self._send_json(200, {"output": output, "digest": digest, "compiled": fresh})

        def _serve_stream(self, trace_id: str) -> None:
            trace = traces.get(trace_id)
            if trace is None:
                self._send_json(404, {"error": "not_found", "message": f"trace {trace_id}"})
                return

            last_id_raw = self.headers.get("Last-Event-ID", "")
            try:
                last_seq = int(last_id_raw) if last_id_raw else 0
            except ValueError:
                last_seq = 0

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            q = traces.subscribe(trace)
            try:
                while True:
                    # Single source of truth: the ordered event buffer. The
                    # queue is only used as a "something changed" signal, so
                    # events appended between subscribe() and the first read
                    # are never lost nor duplicated.
                    with trace.lock:
                        pending = [e for e in trace.events if e["seq"] > last_seq]
                        is_finished = trace.status in ("done", "error")
                    for event in pending:
                        last_seq = event["seq"]
                        self._write_sse(event)
                    if pending:
                        self.wfile.flush()
                    if is_finished:
                        return
                    try:
                        q.get(timeout=15)
                    except queue.Empty:
                        # Heartbeat keeps proxies alive; a failed write means
                        # the client went away, but the detached render lives on.
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Client disconnected: the background render keeps running and
                # the buffered trace remains available for Last-Event-ID resume.
                return
            finally:
                traces.unsubscribe(trace, q)

        def _write_sse(self, event: dict[str, Any]) -> None:
            seq = event["seq"]
            name = event.get("type", "message")
            data = json.dumps(event, ensure_ascii=False)
            payload = f"id: {seq}\nevent: {name}\ndata: {data}\n\n".encode()
            self.wfile.write(payload)

        def _serve_page(self) -> None:
            body = TRACE_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def build_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    cache = CompileCache()
    traces = TraceManager(cache)
    server = _QuietThreadingHTTPServer((host, port), _make_handler(cache, traces))
    server.compile_cache = cache  # type: ignore[attr-defined]
    server.trace_manager = traces  # type: ignore[attr-defined]
    return server


class _QuietThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server with Nagle disabled (avoids loopback delayed-ACK stalls)
    and quiet handling of client-side disconnects (expected for SSE)."""

    daemon_threads = True
    compile_cache: Any = None
    trace_manager: Any = None

    def server_bind(self) -> None:
        super().server_bind()
        try:
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def get_request(self) -> Any:
        request, client_address = super().get_request()
        try:
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return request, client_address

    def handle_error(self, request: Any, client_address: Any) -> None:
        import sys

        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        except ConnectionError:
            # Client went away mid-request (normal for SSE disconnects).
            pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="strif template server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = build_server(args.host, args.port)
    print(f"strif server listening on http://{args.host}:{args.port}")
    print(f"trace page: http://{args.host}:{args.port}/trace")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
