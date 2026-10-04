"""
Acceptance checks for the strif template server.

  uv run python -m strif.server        # terminal 1
  uv run python -m strif.selfcheck     # terminal 2

If no server is reachable on STRIF_HOST:STRIF_PORT (127.0.0.1:8765), a
temporary in-process server is started on an ephemeral port.

Checks cover: singleflight compile counts, hash-based invalidation, ordered
trace events, SSE disconnect + Last-Event-ID resume, strict/replace modes,
409 on recursive self-reference, and reentrant cache/recursion concurrency.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .server import build_server

HOST = os.environ.get("STRIF_HOST", "127.0.0.1")
PORT = int(os.environ.get("STRIF_PORT", "8765"))


class CheckFailure(AssertionError):
    pass


def _request(
    host: str, port: int, method: str, path: str, body: dict[str, Any] | None = None
) -> tuple[int, dict[str, Any]]:
    conn = _get_conn(host, port)
    try:
        return _do_request(conn, method, path, body)
    except (ConnectionError, http.client.HTTPException, OSError):
        conn.close()
        conn = http.client.HTTPConnection(host, port, timeout=15)
        return _do_request(conn, method, path, body)


_CONN_TLS = threading.local()


def _get_conn(host: str, port: int) -> http.client.HTTPConnection:
    conn = getattr(_CONN_TLS, "conn", None)
    if conn is None or getattr(_CONN_TLS, "key", None) != (host, port):
        conn = http.client.HTTPConnection(host, port, timeout=15)
        conn.connect()
        _CONN_TLS.conn = conn
        _CONN_TLS.key = (host, port)
    return conn


def _do_request(
    conn: http.client.HTTPConnection,
    method: str,
    path: str,
    body: dict[str, Any] | None,
) -> tuple[int, dict[str, Any]]:
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8")
    data = json.loads(raw) if raw else {}
    return resp.status, data


def _open_stream(host: str, port: int, trace_id: str, last_id: str | None = None) -> Any:
    sock = socket.create_connection((host, port), timeout=15)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass
    req = f"GET /trace/{trace_id}/stream HTTP/1.1\r\nHost: {host}:{port}\r\n"
    if last_id:
        req += f"Last-Event-ID: {last_id}\r\n"
    req += "Accept: text/event-stream\r\nConnection: close\r\n\r\n"
    sock.sendall(req.encode("utf-8"))
    return sock


def _parse_sse(raw: bytes) -> dict[str, Any] | None:
    event_id: str | None = None
    event_type: str | None = None
    data_lines: list[str] = []
    for line in raw.decode("utf-8").split("\n"):
        if line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines or event_type is None:
        return None
    event = json.loads("".join(data_lines))
    event["_id"] = int(event_id) if event_id else None
    event["_event"] = event_type
    return event


def _read_sse_events(sock: Any, max_events: int, timeout: float = 8.0) -> list[dict[str, Any]]:
    sock.settimeout(timeout)
    buf = b""
    events: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    while len(events) < max_events and time.monotonic() < deadline:
        try:
            chunk = sock.recv(4096)
        except TimeoutError:
            break
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            raw_event, buf = buf.split(b"\n\n", 1)
            event = _parse_sse(raw_event)
            if event is not None:
                events.append(event)
            if len(events) >= max_events:
                break
    return events


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def _start_traced(
    host: str, port: int, template: str, variables: dict[str, Any], recurse: bool = False
) -> str:
    status, data = _request(
        host,
        port,
        "POST",
        "/render",
        {
            "template": template,
            "variables": variables,
            "trace": True,
            "recurse": recurse,
        },
    )
    _check(status == 202, f"expected 202 for traced render, got {status}: {data}")
    return str(data["trace_id"])


def run_checks(host: str, port: int) -> list[str]:
    results: list[str] = []

    _request(host, port, "POST", "/debug/reset")

    # 1. Concurrent identical templates compile exactly once.
    template = "$${keep} Hello ${name}! $if{name}($if{vip}vip$else plain$end)$else nobody$end"
    barrier = threading.Barrier(12)

    def hit_compile() -> tuple[int, dict[str, Any]]:
        _request(host, port, "GET", "/health")  # pre-warm keep-alive connection
        barrier.wait()
        return _request(host, port, "POST", "/compile", {"template": template})

    with ThreadPoolExecutor(max_workers=12) as pool:
        responses = list(pool.map(lambda _: hit_compile(), range(12)))
    _check({s for s, _ in responses} == {200}, f"compile statuses: {responses}")
    digests = {d["digest"] for _, d in responses}
    _check(len(digests) == 1, f"digests diverged: {digests}")
    fresh = [d for _, d in responses if bool(d["compiled"])]
    _check(len(fresh) == 1, f"expected exactly 1 fresh compile, got {len(fresh)}")
    _, stats = _request(host, port, "GET", "/debug/stats")
    _check(stats["compile_count"] == 1, f"compile_count={stats['compile_count']}, want 1")
    results.append("1. 同一模板 12 并发只编译 1 次（单飞合并）")

    # 2. Content change => new hash => recompile; old content still cached.
    edited = template + " "
    status, data = _request(host, port, "POST", "/compile", {"template": edited})
    _check(status == 200 and data["compiled"] is True, f"edit should recompile: {data}")
    _check(data["digest"] not in digests, "edited template must hash differently")
    status, data = _request(host, port, "POST", "/compile", {"template": template})
    _check(data["compiled"] is False, "original content must still hit cache")
    results.append("2. 模板内容变更即按新哈希失效重编译")

    # 3. Trace order: escape, then variable, then outer -> inner condition
    #    decisions, with inner condition_end before outer condition_end.
    tid = _start_traced(host, port, template, {"name": "Alice", "vip": "yes"})
    sock = _open_stream(host, port, tid)
    events = _read_sse_events(sock, 100, timeout=10)
    sock.close()
    types = [e["_event"] for e in events]
    _check(bool(types) and types[0] == "render_start", f"first event: {types}")
    _check(bool(types) and types[-1] == "done", f"last event: {types}")
    body = types[1:-1]
    expected_prefix = ["escape", "variable", "condition", "condition", "condition_end"]
    _check(body[: len(expected_prefix)] == expected_prefix, f"trace order wrong: {body}")
    _check(body.index("escape") < body.index("condition"), "escape precedes conditions")
    conds = [e for e in events if e["_event"] == "condition"]
    ends = [e for e in events if e["_event"] == "condition_end"]
    _check(
        bool([c["name"] for c in conds] == ["name", "vip"]),
        f"outer condition decided first: {[c['name'] for c in conds]}",
    )
    _check(
        conds[0]["_id"] < conds[1]["_id"] < ends[0]["_id"] < ends[1]["_id"],
        "nested condition must close inner-before-outer",
    )
    ids = [e["_id"] for e in events]
    _check(ids == list(range(1, len(ids) + 1)), f"seq must be gapless: {ids}")
    result = next(e for e in events if e["_event"] == "result")
    _check(result["output"] == "${keep} Hello Alice! (vip)", f"output: {result['output']!r}")
    results.append("3. SSE 轨迹按 token 顺序（转义→变量→外层条件→内层条件→内闭→外闭）")

    # 4. Mid-stream disconnect does not stop the render; Last-Event-ID resume.
    tid = _start_traced(host, port, "${a}-${b}-${c}", {"a": "1", "b": "2", "c": "3"})
    first = _open_stream(host, port, tid)
    first_events = _read_sse_events(first, 3, timeout=5)
    _check(len(first_events) >= 1, "should receive at least one event")
    last_seen = first_events[-1]["_id"]
    first.close()
    time.sleep(0.5)  # detached render continues server-side
    second = _open_stream(host, port, tid, last_id=str(last_seen))
    rest = _read_sse_events(second, 100, timeout=5)
    second.close()
    _check(bool(rest), "resume must return buffered events")
    joined = [e["_id"] for e in first_events] + [e["_id"] for e in rest]
    _check(joined == list(range(1, len(joined) + 1)), f"gap/duplicate on resume: {joined}")
    _check(rest[-1]["_event"] == "done", "resumed stream must reach done")
    results.append("4. 断开不中断渲染，Last-Event-ID 续推无重复无缺口")

    # 5. Strict render: missing variable -> 400.
    status, data = _request(host, port, "POST", "/render", {"template": "${missing}"})
    _check(status == 400 and data["error"] == "missing_variable", f"missing: {status} {data}")
    results.append("5. render 缺变量 → MissingVariableError / 400")

    # 6. replace mode: missing values become "".
    status, data = _request(
        host,
        port,
        "POST",
        "/render",
        {"template": "x=${x}|y=${y}", "variables": {"x": "1"}, "mode": "replace"},
    )
    _check(status == 200 and data["output"] == "x=1|y=", f"replace: {status} {data}")
    results.append("6. replace 缺值换空串")

    # 7. recurse=True self-reference -> 409, promptly.
    started = time.monotonic()
    status, data = _request(
        host,
        port,
        "POST",
        "/render",
        {"template": "${loop}", "variables": {"loop": "${loop}"}, "recurse": True},
    )
    elapsed = time.monotonic() - started
    _check(status == 409 and data["error"] == "recursion_limit", f"self-ref: {status} {data}")
    _check(elapsed < 10, f"self-reference took {elapsed:.1f}s — looks like a hang")
    results.append("7. recurse=True 自引用撞深度上限返回 409（不挂死）")

    # 7b. The 409 is also observable through the trace stream.
    tid = _start_traced(host, port, "${loop}", {"loop": "${loop}"}, recurse=True)
    sock = _open_stream(host, port, tid)
    evs = _read_sse_events(sock, 100, timeout=10)
    sock.close()
    errs = [e for e in evs if e["_event"] == "error"]
    _check(len(errs) == 1 and errs[0]["status"] == 409, f"trace error: {evs}")
    _check(
        any(e["_event"] == "recursion_limit" for e in evs),
        "recursion_limit trace event expected",
    )
    results.append("7b. SSE 轨迹同样给出 409 与 recursion_limit 事件")

    # 8. recurse=False keeps the reference literal.
    status, data = _request(
        host,
        port,
        "POST",
        "/render",
        {"template": "${v}", "variables": {"v": "${v}"}, "recurse": False},
    )
    _check(status == 200 and data["output"] == "${v}", f"non-recursive: {data}")
    results.append("8. recurse=False 不递归展开")

    # 9. Reentrancy: concurrent recursive renders sharing the compile cache.
    templ = "[$if{a}${b}$else no$end]"

    def recursive_hit(i: int) -> tuple[int, dict[str, Any]]:
        return _request(
            host,
            port,
            "POST",
            "/render",
            {
                "template": templ,
                "variables": {"a": "x", "b": "${p}", "p": f"v{i}"},
                "recurse": True,
            },
        )

    with ThreadPoolExecutor(max_workers=10) as pool:
        rrs = list(pool.map(recursive_hit, range(20)))
    for i, (status, data) in enumerate(rrs):
        _check(
            status == 200 and data["output"] == f"[v{i}]",
            f"reentrant recursive render {i}: {status} {data}",
        )
    results.append("9. 缓存与递归路径并发可重入")

    return results


def _ensure_server() -> tuple[str, int, Any]:
    """Use a running server if reachable; otherwise start a temp one."""
    try:
        status, _ = _request(HOST, PORT, "GET", "/health")
        if status == 200:
            return HOST, PORT, None
    except OSError:
        pass
    server = build_server("127.0.0.1", 0)
    bound = server.server_address
    host, port = str(bound[0]), int(bound[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return host, port, server


def main() -> int:
    host, port, server = _ensure_server()
    where = f"http://{host}:{port}"
    print(f"strif selfcheck against {where}")
    try:
        results = run_checks(host, port)
    except CheckFailure as exc:
        print(f"FAIL: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 2
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
    for line in results:
        print(f"PASS  {line}")
    _, stats = _request(host, port, "GET", "/debug/stats") if server is None else (0, {})
    if server is None:
        print(f"server stats: {stats}")
    print(f"\nAll {len(results)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
