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
from liminal.ui.twin import JointMirror, parse_telemetry


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
