"""
Resident HTTP service around the strif string template engine.

Endpoints:
  POST /render                    render a template (strict by default)
  POST /compile                   parse + hash-cache a template (single-flight)
  GET  /trace/{id}/stream         Server-Sent Events stream of the decision trace
  GET  /trace/{id}                trace snapshot as JSON
  GET  /trace                     browser page for live traces (screen recording)
  GET  /stats                     compile cache counters
  GET  /health                    liveness probe

Run with:  uv run python -m strif.server
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from strif.string_template import (
    DEFAULT_MAX_DEPTH,
    MissingVariableError,
    RecursionLimitError,
    TemplateSyntaxError,
    compile_template,
    render_template,
    replace_template,
    template_hash,
)

DEFAULT_HOST = os.environ.get("STRIF_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("STRIF_PORT", "8765"))
MAX_TRACES = 256
MAX_CACHE_ENTRIES = 4096
SSE_HEARTBEAT_SECONDS = 15

CompiledTemplateLike = Any


class CompileCache:
    """
    Hash-addressed compiled-template cache with single-flight merging.

    The first request for a template content inserts a placeholder future and
    compiles outside the lock; concurrent requests for the same hash await that
    same future, so a template is compiled exactly once. A changed template has
    a different hash and therefore misses (automatic invalidation). All locks
    are reentrant and never held while compiling, so render/recursion paths that
    re-enter the cache cannot deadlock.
    """

    def __init__(self, compiler: Any = compile_template, max_entries: int = MAX_CACHE_ENTRIES):
        self._lock = threading.RLock()
        self._compiler = compiler
        self._entries: OrderedDict[str, Future[CompiledTemplateLike]] = OrderedDict()
        self._max_entries = max_entries
        self.total_compiles = 0

    def get(self, template: str) -> tuple[Future[CompiledTemplateLike], bool, str]:
        h = template_hash(template)
        with self._lock:
            future = self._entries.get(h)
            if future is not None:
                self._entries.move_to_end(h)
                return future, True, h
            future: Future[CompiledTemplateLike] = Future()
            self._entries[h] = future
            self.total_compiles += 1

        try:
            compiled = self._compiler(template)
        except BaseException as exc:  # propagate to every waiter, do not poison cache
            with self._lock:
                self._entries.pop(h, None)
            future.set_exception(exc)
            return future, False, h

        future.set_result(compiled)
        with self._lock:
            while len(self._entries) > self._max_entries:
                evict_key = None
                for key, old_future in self._entries.items():
                    # Never evict an in-flight compile: waiters and recursion
                    # paths re-entering the cache must keep resolving it.
                    if old_future.done():
                        evict_key = key
                        break
                if evict_key is None:
                    break
                self._entries.pop(evict_key)
        return future, False, h

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"total_compiles": self.total_compiles, "entry_count": len(self._entries)}


class Trace:
    """
    Ordered, replayable decision trace for one render.

    Events are buffered for late SSE subscribers and Last-Event-ID resume.
    Subscribers may disconnect at any time: the render worker keeps emitting
    into this object; only the per-connection copy loop stops.
    """

    def __init__(self, trace_id: str):
        self.id = trace_id
        self._condition = threading.Condition()
        self.completed = threading.Event()
        self._events: list[dict[str, Any]] = []
        self.finished = False
        self.error: str | None = None
        self.result: str | None = None

    def emit(self, payload: dict[str, Any]) -> None:
        with self._condition:
            sequence = len(self._events)
            event = {"seq": sequence, "data": payload}
            self._events.append(event)
            self._condition.notify_all()

    def finish(self, result: str) -> None:
        with self._condition:
            self.result = result
            self.finished = True
            self.emit_terminal({"type": "done", "result": result})
            self._condition.notify_all()
        self.completed.set()

    def fail(self, message: str, error_type: str) -> None:
        with self._condition:
            self.error = message
            self.finished = True
            self.emit_terminal({"type": "error", "error": error_type, "message": message})
            self._condition.notify_all()
        self.completed.set()

    def emit_terminal(self, data: dict[str, Any]) -> None:
        sequence = len(self._events)
        self._events.append({"seq": sequence, "data": data})

    def snapshot(self, after_seq: int = -1) -> list[dict[str, Any]]:
        with self._condition:
            return list(self._events[after_seq + 1 :])

    def wait_for(self, next_seq: int, timeout: float | None = None) -> bool:
        with self._condition:
            if next_seq < len(self._events) or self.finished:
                return True
            return self._condition.wait(timeout)

    def to_dict(self) -> dict[str, Any]:
        with self._condition:
            return {
                "id": self.id,
                "finished": self.finished,
                "error": self.error,
                "result": self.result,
                "events": list(self._events),
            }


class TraceStore:
    """Bounded registry of recent traces keyed by render id."""

    def __init__(self, max_traces: int = MAX_TRACES):
        self._lock = threading.Lock()
        self._traces: OrderedDict[str, Trace] = OrderedDict()
        self._max = max_traces

    def create(self, trace_id: str) -> Trace:
        with self._lock:
            trace = Trace(trace_id)
            self._traces[trace_id] = trace
            self._traces.move_to_end(trace_id)
            while len(self._traces) > self._max:
                self._traces.popitem(last=False)
            return trace

    def get(self, trace_id: str) -> Trace | None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                self._traces.move_to_end(trace_id)
            return trace


compile_cache = CompileCache()
trace_store = TraceStore()


def start_render(
    trace_id: str,
    template: str,
    variables: dict[str, Any] | None,
    *,
    strict: bool,
    recurse: bool,
    max_depth: int,
    wait: bool,
) -> Trace:
    """Kick off a render on a daemon worker and return its trace."""
    trace = trace_store.create(trace_id)
    trace.emit(
        {
            "type": "render_start",
            "trace_id": trace_id,
            "strict": strict,
            "recurse": recurse,
            "max_depth": max_depth,
            "wait": wait,
        }
    )
    worker = threading.Thread(
        target=_render_worker,
        args=(trace_id, template, variables, strict, recurse, max_depth),
        name=f"render-{trace_id[:8]}",
        daemon=True,
    )
    worker.start()
    return trace


def _render_worker(
    trace_id: str,
    template: str,
    variables: dict[str, Any] | None,
    strict: bool,
    recurse: bool,
    max_depth: int,
) -> None:
    # Rendering is detached from any HTTP connection: a dropped SSE client only
    # stops that connection; the worker keeps going and the trace stays replayable.
    trace = trace_store.get(trace_id)
    assert trace is not None
    render = render_template if strict else replace_template
    try:
        future, _cached, h = compile_cache.get(template)
        compiled = future.result()
        trace.emit({"type": "compile_ready", "hash": h, "pos": 0, "level": 0})
        result = render(
            compiled,
            variables,
            recurse=recurse,
            max_depth=max_depth,
            emit=trace.emit,
            compile_fn=lambda text: compile_cache.get(text)[0].result(),
        )
        trace.finish(result)
    except MissingVariableError as exc:
        trace.emit({"type": "missing_variable", "name": exc.name, "pos": 0, "level": 0})
        trace.fail(str(exc), "missing_variable")
    except RecursionLimitError as exc:
        trace.emit(
            {"type": "recursion_limit", "name": exc.name, "depth": exc.depth, "pos": 0, "level": 0}
        )
        trace.fail(str(exc), "recursion_limit")
    except TemplateSyntaxError as exc:
        trace.fail(str(exc), "syntax_error")
    except Exception as exc:  # pragma: no cover - defensive
        trace.fail(str(exc) or exc.__class__.__name__, "internal_error")


class Handler(BaseHTTPRequestHandler):
    server_version = "StrifTemplateServer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            # SSE clients and tests drop connections on purpose; this must
            # never look like a server crash.
            self.close_connection = True

    def _send_json(self, status: int, payload: Any, extra_headers: dict[str, str] | None = None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error_json(self, status: int, error_type: str, message: str):
        self._send_json(status, {"error": error_type, "message": message})

    def _read_json(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_error_json(400, "invalid_json", f"request body is not valid JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            self._send_error_json(400, "invalid_request", "request body must be a JSON object")
            return None
        return payload

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Last-Event-ID")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/health":
            self._send_json(200, {"ok": True})
        elif path == "/stats":
            self._send_json(200, compile_cache.stats())
        elif path == "/trace" or path == "/trace/":
            self._serve_trace_page()
        elif path.startswith("/trace/"):
            rest = path[len("/trace/") :]
            if rest.endswith("/stream"):
                self._serve_stream(rest[: -len("/stream")])
            else:
                self._serve_trace(rest)
        elif path == "/":
            self._send_json(
                200,
                {
                    "service": "strif template server",
                    "endpoints": [
                        "POST /render",
                        "POST /compile",
                        "GET /trace/{id}/stream",
                        "GET /trace/{id}",
                        "GET /trace",
                        "GET /stats",
                        "GET /health",
                    ],
                },
            )
        else:
            self._send_error_json(404, "not_found", f"no route for {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/compile":
            self._handle_compile()
        elif parsed.path == "/render":
            self._handle_render()
        else:
            self._send_error_json(404, "not_found", f"no route for {parsed.path}")

    def _handle_compile(self):
        payload = self._read_json()
        if payload is None:
            return
        template = payload.get("template")
        if not isinstance(template, str):
            self._send_error_json(400, "invalid_request", "'template' must be a string")
            return
        try:
            future, cached, h = compile_cache.get(template)
            compiled = future.result()
        except TemplateSyntaxError as exc:
            self._send_error_json(400, "syntax_error", str(exc))
            return
        self._send_json(
            200,
            {
                "hash": h,
                "cached": cached,
                "nodes": len(compiled.nodes),
                "stats": compile_cache.stats(),
            },
        )

    def _handle_render(self):
        payload = self._read_json()
        if payload is None:
            return
        template = payload.get("template")
        if not isinstance(template, str):
            self._send_error_json(400, "invalid_request", "'template' must be a string")
            return
        variables = payload.get("variables", {})
        if variables is None:
            variables = {}
        if not isinstance(variables, dict):
            self._send_error_json(400, "invalid_request", "'variables' must be an object")
            return
        mode = payload.get("mode", "render")
        if mode not in ("render", "replace"):
            self._send_error_json(400, "invalid_request", "'mode' must be 'render' or 'replace'")
            return
        strict = mode == "render"
        recurse = bool(payload.get("recurse", False))
        wait = bool(payload.get("wait", True))
        max_depth = payload.get("max_depth", DEFAULT_MAX_DEPTH)
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
            self._send_error_json(400, "invalid_request", "'max_depth' must be a non-negative int")
            return

        trace_id = uuid.uuid4().hex
        trace = start_render(
            trace_id,
            template,
            variables,
            strict=strict,
            recurse=recurse,
            max_depth=max_depth,
            wait=wait,
        )

        if not wait:
            self._send_json(
                202,
                {
                    "trace_id": trace_id,
                    "status": "running",
                    "stream": f"/trace/{trace_id}/stream",
                },
            )
            return

        if not trace.completed.wait(timeout=120):
            self._send_error_json(504, "render_timeout", "render did not finish in 120s")
            return

        status = 200
        error = None
        if trace.error:
            terminal = trace.snapshot()[-1]["data"]
            error = terminal.get("error", "error")
            status = {
                "missing_variable": 422,
                "recursion_limit": 409,
                "syntax_error": 400,
            }.get(error, 500)
        response: dict[str, Any] = {
            "trace_id": trace_id,
            "output": trace.result if status == 200 else None,
            "hash": template_hash(template),
            "stats": compile_cache.stats(),
        }
        if error:
            response["error"] = error
            response["message"] = trace.error
        self._send_json(status, response)

    def _serve_trace(self, trace_id: str):
        trace = trace_store.get(trace_id)
        if trace is None:
            self._send_error_json(404, "not_found", f"unknown trace id {trace_id!r}")
            return
        self._send_json(200, trace.to_dict())

    def _serve_stream(self, trace_id: str):
        trace = trace_store.get(trace_id)
        if trace is None:
            self._send_error_json(404, "not_found", f"unknown trace id {trace_id!r}")
            return

        last_header = self.headers.get("Last-Event-ID", "")
        try:
            after_seq = int(last_header) if last_header else -1
        except ValueError:
            after_seq = -1

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "Last-Event-ID")
        self.end_headers()

        next_seq = after_seq + 1
        try:
            while True:
                events = trace.snapshot(next_seq - 1)
                for event in events:
                    data = json.dumps(event["data"], ensure_ascii=False)
                    frame = f"id: {event['seq']}\nevent: trace\ndata: {data}\n\n"
                    self.wfile.write(frame.encode("utf-8"))
                    next_seq = event["seq"] + 1
                self.wfile.flush()
                if trace.finished:
                    return
                notified = trace.wait_for(next_seq, timeout=SSE_HEARTBEAT_SECONDS)
                if not notified:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            # Client went away; server-side rendering is unaffected and the
            # client may resume with Last-Event-ID from where it stopped.
            return

    def _serve_trace_page(self):
        body = TRACE_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass


def make_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def main() -> None:
    httpd = make_server()
    host, port = httpd.server_address[:2]
    print(f"strif template server listening on http://{host}:{port}")
    print(f"trace page: http://{host}:{port}/trace")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


TRACE_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>strif template trace</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    margin: 0; font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: #0f1218; color: #d6deeb; display: grid;
    grid-template-columns: 420px 1fr; grid-template-rows: 100vh;
  }
  #panel { padding: 16px; overflow: auto; border-right: 1px solid #242b38; }
  #view { padding: 16px; overflow: auto; }
  h1 { font-size: 15px; margin: 0 0 12px; color: #82aaff; }
  label { display: block; margin: 10px 0 4px; color: #8a94a6; font-size: 12px; }
  textarea, input[type=text] {
    width: 100%; background: #161b25; color: #d6deeb; border: 1px solid #2a3242;
    border-radius: 6px; padding: 8px; font: inherit;
  }
  textarea { resize: vertical; }
  .row { display: flex; gap: 12px; align-items: center; margin-top: 12px; }
  button {
    background: #2c5fff; color: white; border: 0; border-radius: 6px;
    padding: 8px 14px; font: inherit; cursor: pointer;
  }
  button.secondary { background: #2a3242; }
  button:disabled { opacity: .5; cursor: default; }
  .check { display: flex; align-items: center; gap: 6px; color: #d6deeb; font-size: 13px; }
  #status { margin-top: 12px; font-size: 12px; color: #8a94a6; }
  #output { margin-top: 12px; padding: 10px; background: #161b25; border-radius: 6px;
    white-space: pre-wrap; word-break: break-all; border: 1px solid #2a3242; min-height: 20px; }
  .ev { display: grid; grid-template-columns: 46px 130px 1fr; gap: 8px;
    padding: 4px 8px; border-left: 3px solid #3a4254; border-radius: 3px;
    animation: flash .6s ease; align-items: baseline;
  }
  @keyframes flash { from { background: #1d3a5f; } to { background: transparent; } }
  .ev .seq { color: #5c6778; font-size: 12px; text-align: right; }
  .ev .kind { font-weight: 700; font-size: 12px; }
  .ev .body { color: #b7c2d8; white-space: pre-wrap; word-break: break-all; }
  .t-text { border-left-color: #546e7a; color: #78909c; }
  .t-escape { border-left-color: #ffcb6b; color: #ffcb6b; }
  .t-variable { border-left-color: #82aaff; color: #82aaff; }
  .t-condition_open { border-left-color: #c792ea; color: #c792ea; }
  .t-condition_else { border-left-color: #a882dd; color: #a882dd; }
  .t-condition_close { border-left-color: #7e57c2; color: #a882dd; }
  .t-recurse_enter, .t-recurse_exit { border-left-color: #89ddff; color: #89ddff; }
  .t-render_start, .t-compile_ready { border-left-color: #546e7a; color: #8ea; }
  .t-done { border-left-color: #2ecc71; color: #2ecc71; }
  .t-error, .t-missing_variable, .t-recursion_limit { border-left-color: #ff5370; color: #ff5370; }
</style>
</head>
<body>
<div id="panel">
  <h1>strif template trace</h1>
  <label for="tpl">template</label>
  <textarea id="tpl" rows="4">Hi ${name} $${name} ${?active}[${name}: ${role}]${|}inactive${/}</textarea>
  <label for="vars">variables (JSON)</label>
  <textarea id="vars" rows="5">{"name": "Ada", "role": "admin", "active": true}</textarea>
  <label for="mode">missing variables</label>
  <select id="mode" style="width:100%;background:#161b25;color:#d6deeb;border:1px solid #2a3242;border-radius:6px;padding:8px">
    <option value="render">render → MissingVariableError (422)</option>
    <option value="replace">replace → empty string</option>
  </select>
  <div class="row">
    <span class="check"><input type="checkbox" id="recurse"> recurse=true</span>
    <span class="check"><input type="checkbox" id="reconnect"> resume with Last-Event-ID</span>
  </div>
  <div class="row">
    <button id="go">render + stream trace</button>
    <button id="compile" class="secondary">compile</button>
    <button id="stop" class="secondary" disabled>disconnect SSE</button>
  </div>
  <div id="status">idle</div>
  <label>output</label>
  <div id="output"></div>
</div>
<div id="view"></div>
<script>
const $ = (id) => document.getElementById(id);
let es = null;

function setStatus(text) { $("status").textContent = text; }
function colorClass(type) { return "ev t-" + type; }
function addEvent(evt) {
  const row = document.createElement("div");
  row.className = colorClass(evt.type);
  const seq = document.createElement("span"); seq.className = "seq"; seq.textContent = evt.seq;
  const kind = document.createElement("span"); kind.className = "kind";
  kind.textContent = evt.type + "  L" + (evt.level ?? "-");
  const body = document.createElement("span"); body.className = "body";
  body.textContent = evt.type === "text" ? JSON.stringify(evt.text)
    : JSON.stringify(Object.fromEntries(Object.entries(evt).filter(([k]) => k !== "type")));
  row.append(seq, kind, body);
  $("view").appendChild(row);
  $("view").scrollTop = $("view").scrollHeight;
}

function openStream(traceId) {
  es = new EventSource("/trace/" + traceId + "/stream");
  es.addEventListener("trace", (msg) => {
    const data = JSON.parse(msg.data);
    addEvent({seq: msg.lastEventId, ...data});
    if (data.type === "done") {
      $("output").textContent = data.result ?? "";
      setStatus("done");
      es.close(); es = null; $("stop").disabled = true; $("go").disabled = false;
    } else if (data.type === "error" || data.type === "missing_variable" ||
               data.type === "recursion_limit") {
      if (data.message) $("output").textContent = data.message;
      if (data.type === "error") {
        setStatus("failed: " + (data.message || ""));
        es.close(); es = null; $("stop").disabled = true; $("go").disabled = false;
      }
    }
  });
  es.onerror = () => {
    // EventSource auto-reconnects (and re-sends Last-Event-ID). When the
    // resume checkbox is off, keep it simple and surface the disconnect.
    if (!$("reconnect").checked && es && es.readyState === EventSource.CLOSED) setStatus("disconnected");
  };
  $("stop").disabled = false;
}

$("stop").onclick = () => { if (es) { es.close(); es = null; setStatus("SSE disconnected; render continues server-side"); $("stop").disabled = true; } };

$("go").onclick = async () => {
  let variables;
  try { variables = JSON.parse($("vars").value || "{}"); }
  catch (e) { setStatus("variables are not valid JSON: " + e); return; }
  $("view").innerHTML = ""; $("output").textContent = "";
  $("go").disabled = true; setStatus("starting render…");
  const resp = await fetch("/render", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({template: $("tpl").value, variables, mode: $("mode").value,
      recurse: $("recurse").checked, wait: false})});
  if (resp.status !== 202) { setStatus("render rejected: " + resp.status + " " + await resp.text()); $("go").disabled = false; return; }
  const info = await resp.json();
  setStatus("streaming trace " + info.trace_id.slice(0, 8) + "…");
  openStream(info.trace_id);
};

$("compile").onclick = async () => {
  const resp = await fetch("/compile", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({template: $("tpl").value})});
  const info = await resp.json();
  setStatus(resp.ok ? "compiled " + info.hash.slice(0, 12) + " cached=" + info.cached
    + " total_compiles=" + info.stats.total_compiles : "compile failed: " + JSON.stringify(info));
};
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
