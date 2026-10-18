"""collision_monitor: slot -> arm -> robot side mapping, e-stop latch, dry run, offline replay.

The fake HMI is a real HTTP server on 127.0.0.1 serving the two routes the monitor uses, in haller's shapes
(human_teleop.status(), server.py /estop). Trajectories are synthetic: both TCPs driven to one point in front of
the torso (colliding) or kept 20 cm apart (clear), solved with ws4/kinematics IK on the current station geometry.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import collision_monitor as cm
import numpy as np
import pytest

TELEOP = Path(__file__).resolve().parents[1]
WS4 = TELEOP.parent
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# stance -> session slot -> haller arm id (haller frontend armPairing, vrTeleopProtocol.test.ts:351-358)
PAIRING = {"behind": {"left": "right", "right": "left"}, "front": {"left": "left", "right": "right"}}


def status(robot_q: dict[str, np.ndarray] | None, stance: str = "behind", started_at: float = 1.0) -> dict:
    """A running /teleop/human body whose committed goals put the ROBOT sides at `robot_q` (LeRobot degrees)."""
    if robot_q is None:
        return {"running": False, "state": "idle", "left_arm": None, "right_arm": None, "goal_deg": {}}
    pair = PAIRING[stance]
    goal = {slot: dict(zip(ARM, map(float, robot_q[cm.HALLER_TO_ROBOT[arm]]), strict=True))
            for slot, arm in pair.items()}
    return {"running": True, "state": "running", "left_arm": pair["left"], "right_arm": pair["right"],
            "started_at": started_at, "goal_deg": goal}


class FakeHMI:
    """GET /teleop/human serves `script` one entry per request (the last repeats); POST /estop stops the session."""

    def __init__(self, script: list[dict]):
        self.script, self.i, self.estops = script, 0, []
        hmi = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path != "/teleop/human":
                    self.send_error(404)
                    return
                body = hmi.script[min(hmi.i, len(hmi.script) - 1)]
                hmi.i += 1
                self._send(hmi.stopped if hmi.estops else body)

            def do_POST(self):
                if self.path != "/estop":
                    self.send_error(404)
                    return
                n = int(self.headers.get("content-length") or 0)
                hmi.estops.append({"at_poll": hmi.i, "body": json.loads(self.rfile.read(n) or b"{}")})
                self._send({"ok": True})

        self.stopped = {**status(None), "stopped_reason": "estop"}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def hmi_factory():
    made = []

    def make(script):
        made.append(FakeHMI(script))
        return made[-1]

    yield make
    for h in made:
        h.close()


# ---- no MuJoCo needed ------------------------------------------------------------------------------------------

def test_import_does_not_load_mujoco():
    code = (f"import sys; sys.path.insert(0, {str(TELEOP)!r}); import collision_monitor; "
            "print('mujoco' in sys.modules, 'pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_haller_ids_map_to_opposite_robot_sides():
    assert cm.HALLER_TO_ROBOT == {"right": "left", "left": "right"}


@pytest.mark.parametrize("stance", ["behind", "front"])
def test_goals_follow_the_session_pairing(stance):
    q = {"left": np.arange(6.0), "right": 100 + np.arange(6.0)}
    out, why = cm.goals_from_status(status(q, stance))
    assert why == [] and set(out) == {"left", "right"}
    for side in out:
        np.testing.assert_allclose(out[side], q[side])


def test_one_arm_session_and_incomplete_goals_are_reported_not_guessed():
    body = status({"left": np.zeros(6), "right": np.zeros(6)})
    body["right_arm"] = None
    out, why = cm.goals_from_status(body)
    assert set(out) == {"left"} and why == ["slot right drives no arm"]
    body = status({"left": np.zeros(6), "right": np.zeros(6)})
    del body["goal_deg"]["left"]["elbow_flex"]
    body["goal_deg"]["right"].pop("gripper")
    out, why = cm.goals_from_status(body)
    assert set(out) == {"right"} and "elbow_flex" in why[0]
    assert out["right"][5] == 100.0  # missing gripper is checked fully open


def test_to_model_units():
    q, g = cm.to_model({"left": np.array([90.0, -45, 0, 0, 180, 25])})
    np.testing.assert_allclose(q["left"], [np.pi / 2, -np.pi / 4, 0, 0, np.pi])
    assert g["left"] == 0.25


def _pan_gap(q_deg):
    """Stand-in clearance for logic tests: metres = 1 mm per degree of pan between the two arms."""
    return (q_deg["left"][0] - q_deg["right"][0]) / 1000, ("left_gripper", "right_gripper")


def test_monitor_latches_once_per_session_and_rearms_on_a_new_one(hmi_factory, tmp_path):
    far = {"left": np.r_[80.0, np.zeros(5)], "right": np.zeros(6)}
    near = {"left": np.r_[10.0, np.zeros(5)], "right": np.zeros(6)}
    hmi = hmi_factory([status(None), status(far), status(near)])
    log = tmp_path / "events.jsonl"
    said = []
    mon = cm.Monitor(cm.Backend(hmi.url), _pan_gap, margin_m=0.030, log_path=log, say=said.append)
    states = [mon.step().state for _ in range(5)]
    assert states == ["idle", "ok", "estop", "idle", "idle"]
    assert len(hmi.estops) == 1 and hmi.estops[0]["body"] == {}
    ev = json.loads(log.read_text().splitlines()[0])
    assert ev["clearance_mm"] == 10.0 and ev["margin_mm"] == 30.0 and ev["pairing"] == {"left": "right",
                                                                                       "right": "left"}
    assert "E-STOP SENT" in said[0] and "10.0 mm < 30 mm" in said[0]
    # the same session still below the margin is not e-stopped twice; a new session is
    hmi.estops.clear()
    hmi.script, hmi.i = [status(near), status(near, started_at=2.0)], 0
    mon.tripped_session = 1.0
    assert mon.step().state == "below"
    assert hmi.estops == []
    assert mon.step().state == "estop" and len(hmi.estops) == 1


def test_dry_run_never_posts(hmi_factory):
    hmi = hmi_factory([status({"left": np.zeros(6), "right": np.zeros(6)})])
    mon = cm.Monitor(cm.Backend(hmi.url), _pan_gap, dry_run=True, log_path=None, say=lambda s: None)
    p = mon.step()
    assert p.state == "below" and p.clearance_m == 0.0 and hmi.estops == []


def test_unreachable_backend_is_a_state_not_a_crash():
    mon = cm.Monitor(cm.Backend("http://127.0.0.1:9", timeout=0.2), _pan_gap, log_path=None, say=lambda s: None)
    assert mon.step().state == "unreachable"


# ---- MuJoCo station model --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clearance():
    pytest.importorskip("mujoco")
    try:
        return cm.Clearance()
    except ValueError as exc:  # ws4/sim regenerating its model (another workstream): not our failure
        pytest.skip(f"sim scene does not load right now: {exc}")


def _trajectory(half_gap_m: float, n: int = 30) -> dict[str, np.ndarray]:
    """(n, 6) robot-side LeRobot degrees: from all-zero to TCPs at (0.2, +-half_gap, 0.08), fins down, half open."""
    sys.path.insert(0, str(WS4))
    from kinematics.ik import AnalyticIK
    from kinematics.se3 import make_pose
    from kinematics.so101_kinematics import Station

    st = Station.load()
    ik = AnalyticIK(st.arm)
    R = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    s = np.linspace(0, 1, n)[:, None]
    out = {}
    for side, y in (("left", half_gap_m), ("right", -half_gap_m)):
        goal = np.rad2deg(ik.solve(st.to_base(side, make_pose(R, np.array([0.2, y, 0.08]))), None).q)
        out[side] = np.c_[s * goal, np.full(n, 50.0)]
    return out


COLLIDING = 0.0  # both TCPs at the same point
CLEAR = 0.10  # TCPs 20 cm apart


def test_synthetic_trajectories_on_the_station_model(clearance):
    hit = [clearance({k: v[t] for k, v in _trajectory(COLLIDING).items()})[0] for t in range(30)]
    ok = [clearance({k: v[t] for k, v in _trajectory(CLEAR).items()})[0] for t in range(30)]
    assert hit[0] > 0.1 and hit[-1] < 0  # starts 16 cm apart at all-zero, ends interpenetrating
    assert min(ok) > 0.05


def test_side_swap_matters(clearance):
    """Feeding robot-left joints to the robot-right arm turns the colliding pose into a clear one."""
    q = {k: v[-1] for k, v in _trajectory(COLLIDING).items()}
    assert clearance(q)[0] < 0
    assert clearance({"left": q["right"], "right": q["left"]})[0] > 0.05


@pytest.mark.parametrize("stance", ["behind", "front"])
def test_live_monitor_estops_the_colliding_trajectory_only(clearance, hmi_factory, stance, tmp_path):
    for gap, expect_stop in ((CLEAR, False), (COLLIDING, True)):
        traj = _trajectory(gap)
        hmi = hmi_factory([status({k: v[t] for k, v in traj.items()}, stance) for t in range(30)])
        mon = cm.Monitor(cm.Backend(hmi.url), clearance, log_path=tmp_path / "ev.jsonl", say=lambda s: None)
        polls = [mon.step() for _ in range(32)]
        assert bool(hmi.estops) is expect_stop
        if expect_stop:
            first = next(i for i, p in enumerate(polls) if p.state == "estop")
            assert polls[first].clearance_m < 0.030 <= polls[first - 1].clearance_m
            assert len(hmi.estops) == 1 and all(p.state == "idle" for p in polls[first + 1:])
        else:
            assert all(p.state == "ok" for p in polls)


def _haller_take(robot: dict[str, np.ndarray], mode: str = "ok", lag: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """(action, observation.state) in haller column order [arm left | arm right] for robot-side goals `robot`.

    State trails the goals by `lag` frames. Modes, as haller's recorder writes them (side session note
    recorder_pairing_findings.md): "ok"; "crossed" = dual session with a crossed pairing, action halves swapped;
    "solo_crossed" = one-arm crossed session driving arm left: its goals land in arm right's action columns and its
    own action columns repeat its state; "solo_ok" = one-arm session driving arm left, idle arm right's action
    repeats its state.
    """
    goal = np.c_[robot["right"], robot["left"]]  # haller left = robot right, haller right = robot left
    state = goal[np.maximum(np.arange(len(goal)) - lag, 0)]
    if mode.startswith("solo"):
        state[:, 6:] = goal[0, 6:]  # arm right never moves
    action = {"ok": goal, "crossed": np.c_[goal[:, 6:], goal[:, :6]],
              "solo_crossed": np.c_[state[:, :6], goal[:, :6]], "solo_ok": np.c_[goal[:, :6], state[:, 6:]]}[mode]
    return action.astype(np.float32), state.astype(np.float32)


def _write_recording(root: Path, episodes: list[dict[str, np.ndarray]], modes: list[str] | None = None) -> Path:
    """Minimal haller recording: info.json + one parquet, action/state columns named by HALLER arm id."""
    import pandas as pd

    names = [f"{h}_{j}" for h in ("left", "right") for j in ARM]
    rows = []
    for ep, robot in enumerate(episodes):
        act, state = _haller_take(robot, (modes or ["ok"] * len(episodes))[ep])
        for t in range(len(act)):
            rows.append({"action": act[t], "observation.state": state[t], "episode_index": ep, "frame_index": t})
    (root / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    joints = {n: {"source": "feetech_calibration"} for n in names}
    info = {"fps": 30, "features": {"action": {"dtype": "float32", "shape": [12], "names": names},
                                    "observation.state": {"dtype": "float32", "shape": [12], "names": names}},
            "haller_joint_calibration": {"state_unit": "deg", "joints": joints}}
    (root / "meta").mkdir()
    (root / "meta/info.json").write_text(json.dumps(info))
    return root


@pytest.mark.parametrize("mode, expect", [("ok", "as recorded"), ("crossed", "crossed"),
                                          ("solo_crossed", "solo crossed"), ("solo_ok", "as recorded")])
@pytest.mark.parametrize("gap", [CLEAR, COLLIDING])
def test_action_sides_per_take(mode, expect, gap):
    act, state = _haller_take(_trajectory(gap), mode)
    L, R = list(range(5)), list(range(6, 11))
    assert cm.action_sides(act.astype(float), state.astype(float), L, R) == expect


def test_replay_reports_min_clearance_per_episode(clearance, tmp_path):
    pytest.importorskip("pandas")
    ds = _write_recording(tmp_path / "rec", [_trajectory(CLEAR), _trajectory(COLLIDING)])
    rows, warn = cm.replay(ds, clearance)
    assert warn == []
    assert [r["episode"] for r in rows] == [0, 1] and all(r["frames"] == 30 for r in rows)
    assert rows[0]["min_clearance_mm"] > 50 and rows[0]["frames_below_margin"] == 0
    assert rows[1]["min_clearance_mm"] < 0 and rows[1]["frames_touching"] > 0 and rows[1]["min_frame"] > 20
    assert {r["action_sides"] for r in rows} == {"as recorded"}


def test_replay_cli_exit_code_json(clearance, tmp_path, capsys, monkeypatch):
    pytest.importorskip("pandas")
    monkeypatch.setattr(cm, "Clearance", lambda scene=None: clearance)
    ds = _write_recording(tmp_path / "rec", [_trajectory(CLEAR), _trajectory(COLLIDING)])
    out = tmp_path / "qa.json"
    assert cm.main(["replay", str(ds), "--json", str(out)]) == 1
    report = json.loads(out.read_text())
    assert [e["episode"] for e in report["episodes"]] == [0, 1] and "1 below 30 mm: [1]" in capsys.readouterr().out
    clear_only = _write_recording(tmp_path / "clear", [_trajectory(CLEAR)])
    assert cm.main(["replay", str(clear_only)]) == 0


def test_replay_flags_crossed_episodes_in_a_mixed_dataset(clearance, tmp_path, capsys, monkeypatch):
    """haller appends front and behind takes to one dataset: the check must name episodes, not average them away."""
    pytest.importorskip("pandas")
    monkeypatch.setattr(cm, "Clearance", lambda scene=None: clearance)
    modes = ["ok", "crossed", "solo_crossed", "solo_ok", "ok"]
    ds = _write_recording(tmp_path / "mixed", [_trajectory(CLEAR)] * len(modes), modes)
    rows, warn = cm.replay(ds, clearance)
    assert [r["action_sides"] for r in rows] == ["as recorded", "crossed", "solo crossed", "as recorded",
                                                 "as recorded"]
    assert warn[0].startswith("episodes [1]:") and "sides-as-recorded" not in warn[0]
    assert warn[1].startswith("episodes [2]:")
    assert cm.main(["replay", str(ds)]) == 1  # every clearance is fine; the crossed sides fail the QA
    assert "2 with crossed action sides: [1, 2]" in capsys.readouterr().out
    all_crossed = _write_recording(tmp_path / "behind", [_trajectory(CLEAR)] * 2, ["crossed"] * 2)
    assert "--sides-as-recorded" in cm.replay(all_crossed, clearance)[1][0]
