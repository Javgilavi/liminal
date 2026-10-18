"""Companion contracts use fake telemetry/models; no network, GPU, serial or MuJoCo."""

import asyncio
import math
import sys
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from liminal.core.types import SO101_JOINTS
from liminal.ui.twin import JointMirror, parse_telemetry, telemetry_source_age


@pytest.mark.parametrize("key,scale", [("t", 1), ("t_ms", 1000)])
def test_haller_timestamps_reject_old_future_and_nonfinite_samples(monkeypatch, key, scale):
    monkeypatch.setattr("liminal.ui.twin.time.time", lambda: 1000.0)
    assert telemetry_source_age({key: 999.8 * scale}) == pytest.approx(0.2)
    for stamp in (999.0, 1001.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="timestamp"):
            telemetry_source_age({key: stamp * scale})
    assert telemetry_source_age({}) == 0


def test_measured_joint_mirror_preserves_native_coordinates():
    mirror = JointMirror.__new__(JointMirror)
    mirror.model = SimpleNamespace(
        joint=lambda name: SimpleNamespace(range=(-0.2, 1.8), qposadr=[SO101_JOINTS.index(name.removeprefix("right_"))])
    )
    mirror.data = SimpleNamespace(qpos=[0.0] * 6)
    mirror.mujoco = SimpleNamespace(mj_forward=MagicMock())
    mirror.apply({"right": [10, -20, 30, -40, 50, 25]})
    assert mirror.data.qpos[:5] == pytest.approx([math.radians(x) for x in [10, -20, 30, -40, 50]])
    assert mirror.data.qpos[5] == pytest.approx(0.3)
    mirror.mujoco.mj_forward.assert_called_once()
    with pytest.raises(ValueError):
        mirror.apply({"right": [float("nan")] * 6})


def test_haller_telemetry_requires_complete_finite_arms():
    message = {
        "arms": {
            "right": {"joints": {j: {"pos": i} for i, j in enumerate(SO101_JOINTS)}, "torque": True, "mode": "manual"}
        }
    }
    assert parse_telemetry(message)["right"]["positions"] == list(range(6))
    message["arms"]["right"]["joints"]["gripper"]["pos"] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        parse_telemetry(message)


@pytest.fixture
def companion(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from liminal.ui.research import create_research_app

    class FakeMirror:
        def __init__(self, path):
            pass

        def geometry(self):
            return {"poses": self.poses(), "geometries": [], "meshes": {}}

        def poses(self):
            return [[0, 0, 0, 1, 0, 0, 0]]

        def apply(self, arms):
            pass

    class NoNetwork:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(1000)

    @asynccontextmanager
    async def fake_connect(*args, **kwargs):
        yield NoNetwork()

    monkeypatch.setattr("liminal.ui.twin.JointMirror", FakeMirror)
    monkeypatch.setattr("websockets.asyncio.client.connect", fake_connect)
    monkeypatch.setattr("pathlib.Path.mkdir", lambda *args, **kwargs: None)
    runner = AsyncMock()
    monkeypatch.setitem(sys.modules, "liminal.policy.live", SimpleNamespace(run=runner))
    profile = tmp_path / "profile.yaml"
    profile.write_text("arm: right\ncamera_verified: true\nunits_verified: true\n")
    app = create_research_app(str(profile), site_url="https://javgilavi.github.io/liminal/")
    with TestClient(app) as client:
        yield client, runner


def fresh(client):
    client.app.state.telemetry.update(
        received_at=time.monotonic(),
        error=None,
        arms={"right": {"positions": [0] * 6, "mode": "manual", "torque": True}},
    )


def test_disconnected_and_stale_telemetry_refuse_trials(companion):
    client, runner = companion
    assert client.get("/").status_code == 200
    assert not client.get("/api/state").json()["connected"]
    assert client.post("/api/run", json={"prompt": "Pick up the marker"}).status_code == 409
    fresh(client)
    client.app.state.telemetry["received_at"] -= 1
    assert client.post("/api/run", json={"prompt": "Pick up the marker"}).status_code == 409
    runner.assert_not_called()


def test_source_age_is_included_in_trial_freshness(companion):
    client, runner = companion
    fresh(client)
    client.app.state.telemetry.update(received_at=time.monotonic() - 0.3, source_age=0.25)
    state = client.get("/api/state").json()
    assert not state["connected"] and state["sample_age_ms"] >= 550
    assert client.post("/api/run", json={"prompt": "Pick pen"}).status_code == 409
    runner.assert_not_called()


def test_viewing_and_pairing_need_no_policy_profile(companion):
    from fastapi.testclient import TestClient

    from liminal.ui.research import create_research_app

    _, runner = companion  # reuse the isolated fake mirror and telemetry transport
    app = create_research_app(site_url="https://javgilavi.github.io/liminal/")
    with TestClient(app) as client:
        fresh(client)
        state = client.get("/api/state").json()
        assert state["connected"] and not state["profile_configured"]
        assert state["camera_labels"]["wrist_a"] == "Wrist A"
        pair = client.get("/pair")
        assert pair.status_code == 200 and "Fresh measured joints" in pair.text
        assert client.get("/api/scene").status_code == 200
        assert client.post("/api/run", json={"prompt": "Pick pen"}).status_code == 409
    runner.assert_not_called()


def test_trial_refuses_missing_policy_arm_even_when_another_arm_is_fresh(companion):
    client, runner = companion
    fresh(client)
    arms = client.app.state.telemetry["arms"]
    arms["left"] = arms.pop("right")
    response = client.post("/api/run", json={"prompt": "Pick pen"})
    assert response.status_code == 409 and "right arm" in response.json()["detail"]
    runner.assert_not_called()


def test_observation_mode_uses_existing_runner_and_prompt(companion):
    client, runner = companion
    fresh(client)
    response = client.post("/api/run", json={"prompt": "Pick up the marker", "duration_s": 12})
    assert response.status_code == 200
    client.portal.call(asyncio.sleep, 0.01)
    runner.assert_awaited_once()
    assert runner.call_args.args[0]["prompt"] == "Pick up the marker"
    assert runner.call_args.args[0]["duration_s"] == 12
    assert runner.call_args.args[2] is False


def test_execution_requires_manual_torque_and_ack(companion):
    client, runner = companion
    fresh(client)
    payload = {"prompt": "Pick up the marker", "execute": True}
    assert client.post("/api/run", json=payload).status_code == 409
    payload["engage_acknowledged"] = True
    client.app.state.telemetry["arms"]["right"]["torque"] = False
    assert client.post("/api/run", json=payload).status_code == 409
    runner.assert_not_called()


def test_cross_origin_and_invalid_camera_rejected(companion):
    client, _ = companion
    assert client.post("/api/stop", headers={"Origin": "https://other.example"}).status_code == 403
    assert client.get("/api/cameras/missing").status_code == 404
    assert client.get("/api/cameras/top").status_code == 503
    assert client.post("/api/run", json={"prompt": "test", "duration_s": 10000}).status_code == 422


def test_public_lab_pairing_requires_token_and_keeps_hardware_gates(companion):
    import html
    import re
    from urllib.parse import parse_qs, urlsplit

    client, runner = companion
    pair = client.get("/pair")
    assert pair.status_code == 200
    assert pair.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in pair.headers["content-security-policy"]
    link = html.unescape(re.search(r'<a href="([^"]+)"', pair.text).group(1))
    token = parse_qs(urlsplit(link).fragment)["token"][0]
    headers = {"Origin": "https://javgilavi.github.io"}
    assert client.get("/api/state", headers=headers).status_code == 401
    assert client.get("/pair", headers=headers).status_code == 401
    assert client.get("/api/state", headers={"Origin": "https://untrusted.example"}).status_code == 403
    headers["Authorization"] = "Bearer " + token
    state = client.get("/api/state", headers=headers)
    assert state.status_code == 200
    assert state.headers["access-control-allow-origin"] == headers["Origin"]
    assert state.json()["camera_labels"]["wrist_a"] == "Wrist A"
    assert client.get("/api/scene", headers=headers).json()["poses"] == [[0, 0, 0, 1, 0, 0, 0]]
    assert client.post("/api/run", headers=headers, json={"prompt": "Pick pen"}).status_code == 409
    fresh(client)
    assert client.post("/api/run", headers=headers, json={"prompt": "Pick pen", "execute": True}).status_code == 409
    runner.assert_not_called()
    preflight = client.options(
        "/api/run",
        headers={
            "Origin": headers["Origin"],
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert preflight.status_code == 200
    with client.websocket_connect(
        "/ws/twin", headers={"Origin": headers["Origin"]}, subprotocols=["liminal", "token." + token]
    ) as ws:
        assert ws.accepted_subprotocol == "liminal"
        assert ws.receive_json()["connected"]


def test_public_lab_rejects_unpaired_websocket(companion):
    from starlette.websockets import WebSocketDisconnect

    client, _ = companion
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/twin", headers={"Origin": "https://javgilavi.github.io"}):
            pass


def test_companion_rejects_rebound_hosts_and_cross_site_camera_embedding(companion):
    client, runner = companion
    assert client.get("/api/state", headers={"Host": "attacker.example"}).status_code == 400
    assert (
        client.post(
            "/api/run",
            headers={"Host": "attacker.example", "Origin": "http://attacker.example"},
            json={"prompt": "Pick pen"},
        ).status_code
        == 400
    )
    assert client.get("/api/cameras/top", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert (
        client.get("/pair", headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}).status_code == 200
    )
    runner.assert_not_called()


@pytest.fixture
def orchestrator_bridge(monkeypatch):
    """A fake ws4 orchestrator HMI on loopback: joints of both arms, a head frame, plan / approve / estop."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from fastapi.testclient import TestClient

    from liminal.ui.research import create_research_app

    seen = {"posts": [], "planned": None}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/api/twin/state":
                started = any(p == "/api/approve" for p, _ in seen["posts"])
                return self._send(200, json.dumps({
                    "t": time.time(), "joints": {"left": [1, 2, 3, 4, 5, 6], "right": [0, 0.8, 2, 0.4, 0, 99.7]},
                    "modes": {"left": "policy" if started else "idle", "right": "holding"},
                    "planning": False, "estopped": False, "run": "running" if started else "idle"}).encode())
            if self.path == "/api/state":
                goal = seen["planned"]
                proposal = {"goal": goal, "version": 3, "ok": True, "errors": []} if goal else None
                return self._send(200, json.dumps({"planning": False, "proposal": proposal}).encode())
            if self.path == "/api/frame/head.jpg":
                return self._send(200, b"\xff\xd8jpeg", "image/jpeg")
            self._send(404, b"{}")

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            assert self.headers["X-HMI"] == "1"
            seen["posts"].append((self.path, body))
            if self.path == "/api/plan":
                seen["planned"] = body["goal"]
                return self._send(202, b'{"ok": true}')
            self._send(200, b'{"ok": true, "message": "ok"}')

        def log_message(self, *args):
            pass

    class FakeMirror:
        applied = []

        def __init__(self, path):
            pass

        def geometry(self):
            return {"poses": self.poses(), "geometries": [], "meshes": {}}

        def poses(self):
            return [[0, 0, 0, 1, 0, 0, 0]]

        def apply(self, arms):
            FakeMirror.applied.append(arms)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr("liminal.ui.twin.JointMirror", FakeMirror)
    app = create_research_app(orchestrator=f"http://127.0.0.1:{server.server_address[1]}",
                              site_url="https://javgilavi.github.io/liminal/")
    try:
        with TestClient(app) as client:
            deadline = time.monotonic() + 5
            while not client.get("/api/state").json()["connected"] and time.monotonic() < deadline:
                time.sleep(0.05)
            yield client, seen, FakeMirror
    finally:
        server.shutdown()
        server.server_close()


def test_orchestrator_bridge_mirrors_both_arms_and_runs_the_plan(orchestrator_bridge):
    client, seen, mirror = orchestrator_bridge
    state = client.get("/api/state").json()
    assert state["connected"] and state["profile_configured"] and state["active_arm"] == "right"
    # robot left (the pen arm) is Haller's and the scene's "right"
    assert state["arms"]["right"]["positions"] == [1, 2, 3, 4, 5, 6] and state["arms"]["left"]["mode"] == "holding"
    assert mirror.applied[-1]["right"] == [1, 2, 3, 4, 5, 6]
    assert client.get("/api/cameras/top").content == b"\xff\xd8jpeg"
    goal = "open the box, put the green pen in the green slot"
    planned = client.post("/api/run", json={"prompt": goal, "execute": False}).json()
    assert planned["started"] is False and [p for p, _ in seen["posts"]] == ["/api/plan"]
    assert client.post("/api/run", json={"prompt": goal, "execute": True}).status_code == 409  # no acknowledgement
    started = client.post("/api/run", json={"prompt": goal, "execute": True, "engage_acknowledged": True}).json()
    assert started["started"] and seen["posts"][-1] == ("/api/approve", {"version": 3})
    deadline = time.monotonic() + 5
    while client.get("/api/state").json()["job"]["state"] != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert client.get("/api/state").json()["job"]["state"] == "running"
    assert client.post("/api/stop").json()["stopped"] and seen["posts"][-1][0] == "/api/estop"


def test_orchestrator_bridge_rejects_a_remote_orchestrator_and_a_profile(tmp_path):
    pytest.importorskip("mujoco")  # create_research_app builds the twin mirror before it checks the arguments
    from liminal.ui.research import create_research_app

    with pytest.raises(ValueError, match="local orchestrator"):
        create_research_app(orchestrator="http://10.0.0.5:8080")
    profile = tmp_path / "profile.yaml"
    profile.write_text("arm: right\n")
    with pytest.raises(ValueError, match="do not pass --profile"):
        create_research_app(str(profile), orchestrator="http://127.0.0.1:8080")
