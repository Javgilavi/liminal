"""The HMI's digital twin: measured joints of both fake Haller arms on main's MuJoCo scene, robot sides mapped onto the
scene's Haller (operator-frame) arm names, and the HMI routes the page uses. No hardware; skipped without mujoco."""

import gzip
import http.client
import json
import math
import threading

import pytest

mujoco = pytest.importorskip("mujoco")

from orchestrator.hmi.server import make_server  # noqa: E402
from orchestrator.session import Session, SessionConfig  # noqa: E402
from orchestrator.twin import SCENE, Twin  # noqa: E402

pytestmark = pytest.mark.skipif(not SCENE.exists(), reason="main's research_replay scene is not in this checkout")


def qpos(twin, joint):
    return float(twin.data.qpos[twin.model.joint(joint).qposadr[0]])


def test_robot_left_arm_drives_the_scenes_right_arm():
    t = Twin()
    t.apply({"left": [10, -20, 30, 40, 5, 50]})
    assert qpos(t, "right_shoulder_pan") == pytest.approx(math.radians(10))
    assert qpos(t, "right_elbow_flex") == pytest.approx(math.radians(30))
    lo, hi = t.model.joint("right_gripper").range
    assert qpos(t, "right_gripper") == pytest.approx((lo + hi) / 2)
    assert qpos(t, "left_shoulder_pan") == 0.0  # the box arm was not given
    before = t.poses()
    t.apply({"right": [0, 30, 0, 0, 0, 0], "left": None, "x": [1] * 6})  # unknown and missing arms are skipped
    assert qpos(t, "left_shoulder_lift") == pytest.approx(math.radians(30)) and t.poses() != before
    geo = t.geometry()
    assert geo["geometries"] and geo["meshes"] and len(geo["poses"]) == t.model.nbody


def get(port, path, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", path, headers=headers or {})
    r = conn.getresponse()
    return r.status, dict(r.getheaders()), r.read()


def test_hmi_serves_the_twin_at_the_measured_joints(tmp_path):
    cfg = SessionConfig(env="haller", robot="fake", policy="fake", planner="stub", verifier="manual",
                        fake_start=(5, -90, 90, 70, 0, 2), right_fake_start=(0, 0.8, 2.0, 0.4, 0, 99.7),
                        trial_log=str(tmp_path / "trials.jsonl"), runs_dir=str(tmp_path / "runs"))
    s = Session(cfg)
    server = make_server(s, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        code, _, body = get(port, "/api/twin/state")
        state = json.loads(body)
        assert code == 200 and state["run"] == "idle" and not state["planning"]
        assert state["joints"]["left"][:3] == pytest.approx([5, -90, 90], abs=0.5)
        assert state["joints"]["right"][5] == pytest.approx(99.7, abs=0.5)
        assert state["modes"] == {"left": "idle", "right": "idle"}
        assert qpos(s._twin, "right_shoulder_pan") == pytest.approx(math.radians(state["joints"]["left"][0]))
        code, headers, body = get(port, "/api/twin/scene", {"Accept-Encoding": "gzip"})
        assert code == 200 and headers.get("Content-Encoding") == "gzip"
        scene = json.loads(gzip.decompress(body))
        assert len(scene["poses"]) == len(state["poses"]) and scene["geometries"]
        for path in ("/twin/viewer.js", "/twin/assets/vendor/three.module.js", "/twin/assets/vendor/OrbitControls.js"):
            assert get(port, path)[0] == 200, path
        assert get(port, "/twin/../session.py")[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        s.close()
