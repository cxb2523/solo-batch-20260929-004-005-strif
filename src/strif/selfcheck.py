"""
Acceptance checks for the strif template server.

  uv run python -m strif.selfcheck

Talks to the server at $STRIF_BASE_URL if set, otherwise the default local
server if one is already running, otherwise starts an ephemeral server on a
free port (with a slowed compiler, so the single-flight window is observable).

Exits non-zero if any check fails.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

from strif import server as server_mod
from strif.string_template import compile_template


class CheckFailure(AssertionError):
    pass


def check(name):
    def decorator(fn):
        fn.check_name = name
        return fn

    return decorator


class Client:
    def __init__(self, base_url: str, slow: bool = False):
        self.base_url = base_url.rstrip("/")
        self.slow = slow
        parsed = base_url.split("://", 1)[1]
        host_port, _, _prefix = parsed.partition("/")
        self.host, _, port = host_port.partition(":")
        self.port = int(port or "80")

    def request(self, method: str, path: str, payload=None, timeout: float = 30.0):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def render(self, template, variables=None, **opts):
        body = {"template": template, "variables": variables or {}}
        body.update(opts)
        return self.request("POST", "/render", body)

    def compile(self, template):
        return self.request("POST", "/compile", {"template": template})

    def trace(self, trace_id):
        status, body = self.request("GET", f"/trace/{trace_id}")
        return status, body

    def sse_events(self, trace_id, last_event_id=None, stop_after=None, deadline=30.0):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-store"}
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)
        conn.request("GET", f"/trace/{trace_id}/stream", headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            conn.close()
            raise CheckFailure(f"SSE status {resp.status}")
        events = []
        expiry = time.monotonic() + deadline
        event_id = None
        data_lines = []
        try:
            while time.monotonic() < expiry:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8").rstrip("\r\n")
                if line.startswith("id:"):
                    event_id = int(line[3:].strip())
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif line == "":
                    if event_id is not None and data_lines:
                        parsed = {"seq": event_id, **json.loads("".join(data_lines))}
                        events.append(parsed)
                        if stop_after is not None and len(events) >= stop_after:
                            break
                        if parsed.get("type") in ("done", "error"):
                            break
                    event_id = None
                    data_lines = []
        finally:
            conn.close()
        return events


def _probe(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=1.0) as resp:
            return resp.status == 200
    except OSError:
        return False


def start_ephemeral() -> str:
    httpd = server_mod.make_server("127.0.0.1", 0)
    port = httpd.server_address[1]

    # Force a measurable compile window so N concurrent requests observably
    # merge onto a single compile instead of racing. Different template
    # contents may still compile at the same time; identical content must not.
    guard_lock = threading.Lock()
    in_flight: dict[str, int] = {}

    def slow_compiler(text, _guard=guard_lock, _in_flight=in_flight):
        with _guard:
            if text in _in_flight:
                raise CheckFailure(
                    "identical template was compiled concurrently (no single-flight)"
                )
            _in_flight[text] = _in_flight.get(text, 0) + 1
        time.sleep(0.35)
        with _guard:
            _in_flight.pop(text, None)
        return compile_template(text)

    server_mod.compile_cache = server_mod.CompileCache(compiler=slow_compiler)
    server_mod.trace_store = server_mod.TraceStore()
    # Worker threads captured these names at call time through the module.
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _probe(base_url):
            return base_url
    raise CheckFailure("ephemeral server did not start")


def get_client() -> Client:
    base_url = os.environ.get("STRIF_BASE_URL")
    if base_url:
        if not _probe(base_url):
            raise CheckFailure(f"STRIF_BASE_URL {base_url} is not reachable")
        return Client(base_url)
    default_url = f"http://{server_mod.DEFAULT_HOST}:{server_mod.DEFAULT_PORT}"
    if _probe(default_url):
        return Client(default_url)
    return Client(start_ephemeral(), slow=True)


TPL = (
    "user=${name}; escaped=$${name}; "
    "${?active}[on: ${name} (${?admin}admin${|}user${/})]${|}off${/}; "
    "tail=${tail}"
)
VARS = {"name": "Ada", "active": "yes", "admin": True, "tail": "end"}


@check("render basics (variables, escape precedence, nesting)")
def c_render_basics(c: Client):
    status, body = c.render(TPL, VARS)
    if status != 200:
        raise CheckFailure(f"render status {status}: {body}")
    out = body["output"]
    expected = "user=Ada; escaped=${name}; [on: Ada (admin)]; tail=end"
    if out != expected:
        raise CheckFailure(f"output mismatch:\n got={out!r}\nwant={expected!r}")


@check("replace mode blanks missing variables; render mode errors")
def c_missing(c: Client):
    status, body = c.render(TPL, {k: v for k, v in VARS.items() if k != "tail"}, mode="replace")
    if status != 200:
        raise CheckFailure(f"replace status {status}: {body}")
    if not body["output"].endswith("tail="):
        raise CheckFailure(f"missing variable should be blank, got {body['output']!r}")

    status, body = c.render("need ${x}", {})
    if status != 422 or body.get("error") != "missing_variable":
        raise CheckFailure(f"expected 422 missing_variable, got {status} {body}")


@check("condition truthiness = present and non-empty; nesting evaluates correctly")
def c_conditions(c: Client):
    tpl = "${?a}A${?b}B${|}bno${/}${|}C${/}"
    cases = [
        ({"a": "x", "b": 0}, "Abno"),
        ({"a": False, "b": "x"}, "C"),
        ({"a": [1], "b": "z"}, "AB"),
        ({}, "C"),
        ({"a": "", "b": "z"}, "C"),
    ]
    for variables, expected in cases:
        status, body = c.render(tpl, variables, mode="replace")
        if status != 200 or body["output"] != expected:
            raise CheckFailure(f"{variables}: got {status} {body!r}, want {expected!r}")


@check("compile cache: N concurrent identical templates compile once (single-flight)")
def c_single_flight(c: Client):
    tpl = "single-${x}-${?y}a${|}b${/}-" + uuid.uuid4().hex
    before = c.request("GET", "/stats")[1]["total_compiles"]
    n = 8
    barrier = threading.Barrier(n)

    def one(_):
        barrier.wait()
        return c.compile(tpl)

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(one, range(n)))

    statuses = [status for status, _ in results]
    if set(statuses) != {200}:
        raise CheckFailure(f"compile statuses: {statuses}")
    hashes = {body["hash"] for _, body in results}
    if len(hashes) != 1:
        raise CheckFailure(f"mixed hashes: {hashes}")
    cached_flags = [body["cached"] for _, body in results]
    if cached_flags.count(False) != 1 or cached_flags.count(True) != n - 1:
        raise CheckFailure(f"expected 1 fresh + {n - 1} merged, got {cached_flags}")
    after = c.request("GET", "/stats")[1]["total_compiles"]
    if after - before != 1:
        raise CheckFailure(f"total_compiles delta {after - before}, expected 1")
    status, body = c.compile(tpl)
    if not body["cached"]:
        raise CheckFailure("repeat compile was not served from cache")


@check("template content change invalidates cache (new hash)")
def c_invalidation(c: Client):
    tag = uuid.uuid4().hex
    v1 = "v1-${v" + tag + "}"
    v2 = "v2-${v" + tag + "}"
    c.compile(v1)
    s1, b1 = c.compile(v1)
    if s1 != 200:
        raise CheckFailure(f"repeat compile v1 failed: {s1} {b1}")
    body1 = b1
    c.compile(v2)
    s2, b2 = c.compile(v2)
    if s2 != 200:
        raise CheckFailure(f"repeat compile v2 failed: {s2} {b2}")
    body2 = b2
    if body1["hash"] == body2["hash"]:
        raise CheckFailure("changed template produced same hash")
    if not body1["cached"] or not body2["cached"]:
        raise CheckFailure("second compile of each template should hit cache")


@check("trace: tokens stream in order; escape decided before condition/block tokens")
def c_trace_order(c: Client):
    status, body = c.request(
        "POST",
        "/render",
        {"template": TPL, "variables": VARS, "wait": False},
    )
    if status != 202:
        raise CheckFailure(f"async render status {status}: {body}")
    trace_id = body["trace_id"]
    events = c.sse_events(trace_id)
    if not events or events[-1]["type"] != "done":
        raise CheckFailure(f"trace did not finish: {[e['type'] for e in events[-3:]]}")

    seqs = [e["seq"] for e in events]
    if seqs != list(range(len(events))):
        raise CheckFailure("trace sequence ids are not contiguous from 0")

    # Within every depth, pos must be non-decreasing (token parse order).
    for depth in {e.get("depth", 0) for e in events}:
        level_events = [e for e in events if e.get("depth", 0) == depth and "pos" in e]
        positions = [e["pos"] for e in level_events]
        if positions != sorted(positions):
            raise CheckFailure(f"trace events out of parse order at depth {depth}: {positions}")

    escape = next(e for e in events if e["type"] == "escape")
    if escape["literal"] != "${" or escape["raw"] != "$${":
        raise CheckFailure(f"escape event malformed: {escape}")
    open_evt = next(e for e in events if e["type"] == "condition_open" and e["name"] == "active")
    if escape["pos"] < open_evt["pos"] and not escape["seq"] < open_evt["seq"]:
        raise CheckFailure("escape token must be decided before later condition block")
    if not open_evt["truthy"] or open_evt["branch"] != "then":
        raise CheckFailure(f"condition decision wrong: {open_evt}")

    opens = [e for e in events if e["type"] == "condition_open"]
    closes = [e for e in events if e["type"] == "condition_close"]
    if [e["level"] for e in opens] != [1, 2] or [e["level"] for e in closes] != [2, 1]:
        raise CheckFailure("nested condition levels not opened/closed in matching order")
    for open_e, close_e in zip([e["pos"] for e in opens], [e["pos"] for e in closes], strict=False):
        if close_e <= open_e:
            raise CheckFailure("condition close must follow its open")

    # Snapshot endpoint agrees with the stream.
    status, snap = c.trace(trace_id)
    if status != 200 or [e["seq"] for e in snap["events"]] != seqs:
        raise CheckFailure("GET /trace/{id} snapshot disagrees with SSE stream")


@check("SSE disconnect does not stop render; Last-Event-ID resumes without repeats")
def c_sse_resume(c: Client):
    tpl = "${?a}${name} ${?b}[${tail}]${/}${|}no${/}-${x}"
    variables = {"a": 1, "b": 1, "name": "Ada", "tail": "T", "x": "X"}
    status, body = c.request(
        "POST", "/render", {"template": tpl, "variables": variables, "wait": False}
    )
    trace_id = body["trace_id"]

    first = c.sse_events(trace_id, stop_after=4)
    if len(first) != 4:
        raise CheckFailure(f"expected 4 buffered events, got {len(first)}")
    time.sleep(0.5)  # worker keeps running after the client dropped

    status, snap = c.trace(trace_id)
    if not snap["finished"]:
        # Give a fast machine a moment, but it should already be done.
        time.sleep(0.5)
        status, snap = c.trace(trace_id)
    if not snap["finished"]:
        raise CheckFailure("render did not finish after SSE client disconnected")

    last_id = first[-1]["seq"]
    rest = c.sse_events(trace_id, last_event_id=last_id)
    resumed_seqs = [e["seq"] for e in rest]
    if not resumed_seqs or resumed_seqs[0] != last_id + 1:
        raise CheckFailure(f"resume must start at {last_id + 1}, got {resumed_seqs[:3]}")

    full = c.sse_events(trace_id)
    if full == first:
        raise CheckFailure("full replay after completion should include all events")
    all_from_start = first + rest
    if [e["seq"] for e in all_from_start] != list(range(len(snap["events"]))):
        raise CheckFailure("resumed stream has gaps or duplicates")


@check("recursion only with recurse=True; self reference -> 409, never hangs")
def c_recursion(c: Client):
    status, body = c.render("${x}", {"x": "${x}"})
    if status != 200 or body["output"] != "${x}":
        raise CheckFailure(f"without recurse, value stays literal: {status} {body}")

    status, body = c.render("${x}", {"x": "${x}"}, recurse=True)
    if status != 409 or body.get("error") != "recursion_limit":
        raise CheckFailure(f"self reference must yield 409, got {status} {body}")

    # Mutual reference cycle must also hit the depth cap quickly.
    started = time.monotonic()
    status, body = c.render("${a}", {"a": "${b}", "b": "${a}"}, recurse=True)
    elapsed = time.monotonic() - started
    if status != 409 or elapsed > 20:
        raise CheckFailure(f"cycle must yield 409 promptly, got {status} {body} after {elapsed}s")

    status, body = c.render("${x}", {"x": "${y}", "y": "deep"}, recurse=True)
    if status != 200 or body["output"] != "deep":
        raise CheckFailure(f"bounded recursion should expand, got {status} {body}")


@check("cache + recursion paths are reentrant under concurrent mixed load")
def c_reentrant(c: Client):
    tag = uuid.uuid4().hex
    outer = f"[${{a}}|${{b}}]-{tag[:8]}"
    inner_name = f"inner_{tag.replace('-', '_')}"
    variables = {"a": f"${{{inner_name}}}", "b": "B", inner_name: f"deep-{tag[:8]}"}
    c.compile(f"${{{inner_name}}}")

    n = 6
    barrier = threading.Barrier(n)
    errors = []

    def one(i):
        barrier.wait()
        if i % 2:
            status, body = c.render(outer, variables, recurse=True, mode="replace")
            if status != 200 or body["output"] != f"[deep-{tag[:8]}|B]-{tag[:8]}":
                errors.append(f"render {i}: {status} {body}")
        else:
            status, body = c.compile(outer)
            if status != 200:
                errors.append(f"compile {i}: {status} {body}")

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(one, range(n)))
    if errors:
        raise CheckFailure("; ".join(errors))


CHECKS = [
    c_render_basics,
    c_missing,
    c_conditions,
    c_single_flight,
    c_invalidation,
    c_trace_order,
    c_sse_resume,
    c_recursion,
    c_reentrant,
]


def main() -> int:
    print("strif template server selfcheck")
    client = get_client()
    print(f"target: {client.base_url}" + (" (ephemeral, slowed compiler)" if client.slow else ""))
    passed = 0
    for fn in CHECKS:
        name = fn.check_name
        try:
            fn(client)
        except Exception as exc:
            print(f"  FAIL  {name}\n        {exc}")
            return 1
        passed += 1
        print(f"  pass  {name}")
    stats = client.request("GET", "/stats")[1]
    print(f"\n{passed}/{len(CHECKS)} checks passed; compile stats: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
