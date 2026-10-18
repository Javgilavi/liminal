import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from liminal.ui.server import create_app

    app = create_app("configs/robot/mock_dual.yaml")
    with TestClient(app) as client:
        yield client
    assert all(not arm.connected for arm in app.state.context.arms.values())


def test_ui_and_telemetry(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    state = client.get("/state").json()
    assert set(state["arms"]) == {"left", "right"}
    assert len(state["joints"]) == 6
    assert state["sample_age_s"] < 0.5
    assert client.get("/tasks").json()["tasks"]


def test_wrist_feed_labels_jpeg_and_capture_failure(client, monkeypatch):
    camera = SimpleNamespace(label="Wrist A (arm unverified)", read=MagicMock(return_value="rgb"))
    client.app.state.context.cameras["wrist_a"] = camera
    cv = SimpleNamespace(COLOR_RGB2BGR=4, cvtColor=MagicMock(return_value="bgr"),
                         imencode=MagicMock(return_value=(True, SimpleNamespace(tobytes=lambda: b"jpeg"))))
    monkeypatch.setitem(sys.modules, "cv2", cv)
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(asarray=lambda frame, **kw: frame, uint8=int))
    state = client.get("/state").json()
    assert "wrist_a" in state["cameras"]
    assert state["camera_labels"]["wrist_a"] == camera.label
    response = client.get("/cameras/wrist_a.jpg")
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == b"jpeg"
    cv.cvtColor.assert_called_once_with("rgb", cv.COLOR_RGB2BGR)
    camera.read.side_effect = RuntimeError("USB disconnected")
    assert client.get("/cameras/wrist_a.jpg").status_code == 503
    assert client.get("/cameras/missing.jpg").status_code == 404


def test_manual_command_uses_supervisor_and_estop_requires_reset(client):
    from liminal.safety import JointLimits

    ctx = client.app.state.context
    ctx.safety.limits["left"] = JointLimits([-10] * 6, [10] * 6, 1)
    assert client.post("/arms/left/goal", json={"positions": [50] * 6}).json()["positions"] == [1] * 6
    assert client.post("/estop").json()["estopped"]
    assert client.post("/arms/left/goal", json={"positions": [2] * 6}).status_code == 409
    assert client.post("/run", json={"task": "single_pick_place"}).status_code == 409
    assert client.post("/reset").status_code == 200
    assert client.post("/arms/left/goal", json={"positions": [2] * 6}).status_code == 200


def test_malformed_command_and_unknown_arm(client):
    assert client.post("/arms/left/goal", json={"positions": [1]}).status_code == 422
    assert client.post("/arms/missing/goal", json={"positions": [0] * 6}).status_code == 404
    assert client.post("/arms/left/goal", json={"positions": [1e300] * 6}).status_code == 200


def test_owned_arm_and_fault_refuse_controls(client):
    ctx = client.app.state.context
    ctx.safety._owners["left"] = "policy"
    assert client.post("/arms/left/goal", json={"positions": [1] * 6}).status_code == 409
    assert client.post("/arms/left/torque", json={"enabled": True}).status_code == 409
    ctx.arms["left"].fault = "overheated"
    assert client.post("/reset").status_code == 409


def test_cross_origin_commands_are_rejected(client):
    assert client.post("/reset", headers={"Origin": "https://unrelated.example"}).status_code == 403


def test_mock_task_runs_and_reports_success(client):
    response = client.post("/run", json={"task": "single_pick_place"})
    assert response.status_code == 200
    # Wait within the server's loop, without network/hardware or external polling.
    async def done():
        await asyncio.sleep(0.3)
    client.portal.call(done)
    assert client.get("/state").json()["last_report"]["goal_verified"]


def test_partial_startup_closes_all_opened_resources():
    from liminal.hardware.runtime import connected

    first, second = MagicMock(), MagicMock()
    first.joint_limits = None
    second.connect.side_effect = RuntimeError("unplugged")
    ctx = SimpleNamespace(arms={"left": first, "right": second}, cameras={},
                          safety=SimpleNamespace(limits={}, estop=MagicMock()))
    with pytest.raises(RuntimeError, match="unplugged"), connected(ctx):
        pytest.fail("must not reach running state")
    first.disconnect.assert_called_once()
    second.disconnect.assert_called_once()


def test_calibration_and_config_limits_are_intersected():
    from liminal.hardware.runtime import connected
    from liminal.safety import JointLimits

    arm = MagicMock()
    arm.joint_limits = ([-100] * 6, [100] * 6)
    ctx = SimpleNamespace(arms={"left": arm}, cameras={}, safety=SimpleNamespace(
        limits={"left": JointLimits([-90] * 6, [80] * 6, 2)}, estop=MagicMock()))
    with connected(ctx):
        assert ctx.safety.limits["left"].lower == [-90] * 6
        assert ctx.safety.limits["left"].upper == [80] * 6
        assert arm.joint_limits == ([-90] * 6, [80] * 6)


def test_real_home_requires_a_measured_pose(make_ctx):
    from liminal.core.types import Skill

    ctx = make_ctx()
    ctx.arms["left"].joint_limits = ([-100] * 6, [100] * 6)
    ctx.skills.primitives.home_poses = {}
    with pytest.raises(ValueError, match="no measured home"):
        asyncio.run(ctx.skills.primitives.execute(Skill("left", "home"), ctx))
    assert ctx.arms["left"].history == []
