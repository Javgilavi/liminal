"""HMI server: routes, localhost and cross-site protections, e-stop wiring (fake session, no MuJoCo)."""

import http.client
import json
import threading

import pytest
from orchestrator.hmi.server import PAGE, make_server


class FakeSession:
    def __init__(self, vocab):
        self.vocab = vocab
        self.calls = []
        self.planning = False
        self.executor = None

    def snapshot(self):
        return {"ok": True, "calls": list(self.calls)}

    def frame(self, name):
        return b"\xff\xd8jpeg" if name == "head" else None

    def estop(self, reason):
        self.calls.append(("estop", reason))

    def release(self):
        self.calls.append(("release",))

    def resume(self):
        self.calls.append(("resume",))

    def abort(self):
        self.calls.append(("abort",))

    def plan(self, goal):
        self.calls.append(("plan", goal))

    def approve(self, version):
        self.calls.append(("approve", version))
        return "ok" if version == 1 else "no valid plan with that version"

    def override(self, step, done):
        self.calls.append(("override", step, done))

    def reset_scene(self, seed, lid):
        self.calls.append(("reset", seed, lid))
        return "ok"

    def load_plan_json(self, plan, goal):
        class P:
            def to_json(self):
                return {"ok": True}

        return P()


@pytest.fixture
def hmi(vocab):
    session = FakeSession(vocab)
    server = make_server(session, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield session, server.server_address[1]
    server.shutdown()
    server.server_close()


def request(port, method, path, body=None, headers=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    hdrs = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    data = None if body is None else json.dumps(body)
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    return resp.status, resp.read()


POST_HEADERS = {"Content-Type": "application/json", "X-HMI": "1"}


def test_page_state_frames_schema(hmi):
    session, port = hmi
    status, body = request(port, "GET", "/")
    assert status == 200 and body == PAGE.read_bytes() and b"E-STOP" in body
    assert request(port, "GET", "/api/state")[0] == 200
    assert request(port, "GET", "/api/frame/head.jpg")[1].startswith(b"\xff\xd8")
    assert request(port, "GET", "/api/frame/nope.jpg")[0] == 404
    schema = json.loads(request(port, "GET", "/api/schema")[1])
    assert schema["properties"]["steps"]["items"]["properties"]["skill"]["enum"] == session.vocab.names()


def test_estop_and_controls(hmi):
    session, port = hmi
    for path in ("/api/estop", "/api/release", "/api/resume", "/api/abort"):
        assert request(port, "POST", path, {}, POST_HEADERS)[0] == 200
    assert [c[0] for c in session.calls] == ["estop", "release", "resume", "abort"]
    assert request(port, "POST", "/api/approve", {"version": 2}, POST_HEADERS)[0] == 409
    assert request(port, "POST", "/api/approve", {"version": 1}, POST_HEADERS)[0] == 200
    assert request(port, "POST", "/api/override", {"step": "a", "done": True}, POST_HEADERS)[0] == 200
    assert ("override", "a", True) in session.calls
    assert request(port, "POST", "/api/plan", {"goal": ""}, POST_HEADERS)[0] == 400


def test_cross_site_and_rebinding_are_refused(hmi):
    session, port = hmi
    # No custom header: a plain cross-site form or fetch cannot set it without a CORS preflight.
    assert request(port, "POST", "/api/approve", {"version": 1}, {"Content-Type": "application/json"})[0] == 403
    assert request(port, "POST", "/api/estop", {}, {"X-HMI": "1", "Content-Type": "text/plain"})[0] == 403
    # DNS rebinding: a foreign Host header is refused even for reads.
    assert request(port, "GET", "/api/state", host="evil.example:80")[0] == 403
    assert request(port, "POST", "/api/approve", {"version": 1}, POST_HEADERS, host="evil.example")[0] == 403
    assert request(port, "GET", "/api/state", host=f"localhost:{port}")[0] == 200
    assert not any(c[0] == "approve" for c in session.calls)


def test_server_refuses_non_local_bind(vocab):
    with pytest.raises(ValueError):
        make_server(FakeSession(vocab), "0.0.0.0", 0)


def test_page_has_spacebar_estop():
    page = PAGE.read_text()
    assert 'e.code !== "Space"' in page or "Space" in page
    assert "/api/estop" in page
