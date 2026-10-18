"""The final demo chain on the Haller (skills.yaml): robot-right arm opens the box, then the robot-left arm puts the
green pen, the yellow cutter and the red pen in their slots. Both arms are run_single_arm's fake arms; each skill has
its own package on flux.remote's real HTTP server (localhost, token), served by a fake policy. No hardware, no pod."""

import threading
import time
from collections import defaultdict

import numpy as np
import pytest
import yaml
from orchestrator.session import Session, SessionConfig, load_trial_log
from orchestrator.skills import DEFAULT_VOCABULARY, load_vocabulary
from orchestrator.verifier import Verdict

GOAL = ("open the box, put the green pen in the green slot, put the yellow cutter in the yellow slot, "
        "put the red pen in its slot")
CHAIN = ["open_box", "put_green_pen_in_slot", "put_yellow_cutter_in_slot", "put_pen_in_slot"]
PACKAGES = {"open_box": "lora_open_box", "put_green_pen_in_slot": "lora_green_pen",
            "put_yellow_cutter_in_slot": "lora_yellow_cutter", "put_pen_in_slot": "lora_red_pen"}
STRINGS = {"open_box": "open the box by pulling its handle",
           "put_green_pen_in_slot": "put the green pen in the green slot",
           "put_yellow_cutter_in_slot": "put the yellow cutter in the yellow slot",
           "put_pen_in_slot": "put the red pen in its slot in the cardboard box"}


def test_vocabulary_is_the_final_chain():
    v = load_vocabulary(DEFAULT_VOCABULARY)
    assert v.station.arms_available == ("left", "right")
    assert {name: v[name].prompt for name in CHAIN} == STRINGS
    box = v["open_box"]
    assert box.arms == ("right",) and box.other_arm_pose == "tucked" and not box.park_after  # it holds the lid
    assert all(v[name].arms == ("left",) for name in CHAIN[1:])
    assert all(v[name].start_pose == "home" for name in CHAIN)
    assert all(v[n].precheck == "open_box" and v[n].park_after and not v[n].other_arm_pose for n in CHAIN[1:])
    assert v.station.home_for("left") == (0, 0, 0, 0, 0, 100)
    assert v.station.home_for("right") == (0, 0.8, 2.0, 0.4, 0, 99.7)
    assert set(v.station.poses_for("left")) == set(v.station.poses_for("right")) == {"home", "tucked"}


class ScriptedChecks:
    """Verifier stand-in: per skill, a list of verdicts consumed in order (then always True)."""

    name, fast = "scripted-test", True

    def __init__(self, script=None):
        self.script = defaultdict(list, {k: list(v) for k, v in (script or {}).items()})
        self.calls = []

    def check(self, step, skill, obs):
        self.calls.append(skill.name)
        ok = self.script[skill.name].pop(0) if self.script[skill.name] else True
        return Verdict(ok, 0.9, "as scripted", self.name)


@pytest.fixture
def chain_server(tmp_path):
    from flux.backends import FakeBackend
    from flux.contract import SO101_CONTRACT
    from flux.remote import make_server

    backends = {pkg: FakeBackend(SO101_CONTRACT, velocity=0.0, delay_s=0.005) for pkg in PACKAGES.values()}
    server = make_server(backends, "127.0.0.1", 0, token="chain-token")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = tmp_path / "flux_token"
    token.write_text("chain-token\n")
    yield f"http://127.0.0.1:{server.server_address[1]}", str(token), backends
    server.shutdown()
    server.server_close()


def short_chain_vocab(tmp_path, timeout=3.0):
    data = yaml.safe_load(DEFAULT_VOCABULARY.read_text())
    for name in CHAIN:
        data["skills"][name]["timeout_s"] = timeout
        data["skills"][name]["verify_every_chunks"] = 1
    path = tmp_path / "skills.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def chain_session(tmp_path, server, token, **kw):
    cfg = SessionConfig(env="haller", robot="fake", policy="remote", server=server, token_file=token,
                        skill_packages=dict(PACKAGES), planner="stub", verifier="manual", realtime=False,
                        task="final_chain", vocabulary=short_chain_vocab(tmp_path), max_run_s=120,
                        condition="post_training", variant="lora_chain", checkpoint="ws4/chain-loras@test",
                        trial_log=str(tmp_path / "trials.jsonl"), runs_dir=str(tmp_path / "runs"), **kw)
    return Session(cfg)


def test_final_chain_end_to_end_over_the_policy_server(tmp_path, chain_server):
    server, token, backends = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        assert s.rig.both and sorted(s.rig.arms) == ["left", "right"]
        assert sorted(s.target_backend) == ["left:lora_green_pen", "left:lora_red_pen", "left:lora_yellow_cutter",
                                            "left:prim", "right:lora_open_box", "right:prim"]
        s.verifier = checks = ScriptedChecks()
        plan = s.plan(GOAL).result.plan
        assert [(st.skill, st.arm) for st in plan.steps] == [
            ("open_box", "right"), ("put_green_pen_in_slot", "left"), ("put_yellow_cutter_in_slot", "left"),
            ("put_pen_in_slot", "left")]
        assert [st.depends_on for st in plan.steps] == [[]] + [[plan.steps[i].id] for i in range(3)]
        starts = []
        original = s.runner.start

        def spy(target, task):
            joints = {a: x.robot.joints().round(1) for a, x in s.rig.arms.items()}
            starts.append((target, task, joints, list(s.runner.active_targets())))
            return original(target, task)

        s.runner.start = spy
        result = s.execute(plan)
        assert result.status == "succeeded" and (result.steps_done, result.steps_total) == (4, 4)
        assert result.interventions == 0
        # Box: pen arm tucked, box arm home, open, then the box arm HOLDS (no ramp). Each pen skill: lid precheck, pen
        # arm home, skill, park at home. One arm moving at a time.
        expected = [("left:prim", "tucked"), ("right:prim", "home"), ("right:lora_open_box", STRINGS["open_box"])]
        for skill in CHAIN[1:]:
            expected += [("left:prim", "home"), (f"left:{PACKAGES[skill]}", STRINGS[skill]), ("left:prim", "home")]
        assert [(t, task) for t, task, _, _ in starts] == expected
        assert all(active == [] for _, _, _, active in starts)  # nothing else moves when a ramp or skill starts
        poses = s.vocab.station.poses
        box_q = starts[3][2]["right"]  # the box arm when the first pen ramp starts: holding the lid
        for target, _, joints, _ in starts[3:]:
            assert np.abs(joints["right"] - box_q).max() <= 0.5  # it never moves again
            if not target.endswith(":prim"):
                assert np.abs(joints["left"] - poses["left"]["home"]).max() <= 3.0, (target, joints)
        assert np.abs(starts[2][2]["left"] - poses["left"]["tucked"]).max() <= 3.0  # pen arm tucked for the box
        # each package served only its own skill (after one warm-up request per prompt)
        for skill, pkg in PACKAGES.items():
            assert {r.task for r in backends[pkg].requests} == {STRINGS[skill]}
        assert checks.calls.count("open_box") == 4  # its own check, then the lid precheck before each pen skill
        assert all(checks.calls.count(skill) == 2 for skill in CHAIN[1:])  # passed, then again after parking
        trials = load_trial_log().load(tmp_path / "trials.jsonl")
        assert [t.task for t in trials if t.task != "final_chain"] == CHAIN
        (chain,) = [t for t in trials if t.task == "final_chain"]
        assert chain.success and chain.steps_done == chain.steps_total == 4
        snap = s.executor.snapshot()
        assert [st["status"] for st in snap["steps"]] == ["done"] * 4
        assert [st["arm"] for st in snap["steps"]] == ["right", "left", "left", "left"]
    finally:
        s.close()


def test_pen_skill_holds_when_the_lid_is_closed(tmp_path, chain_server):
    """Before each pen skill the lid check runs again; a closed lid stops everything in a safe hold before the pen arm
    moves (parent 2026-10-18). Resume checks again."""
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        s.verifier = checks = ScriptedChecks({"open_box": [True, False]})  # opened, then found closed
        holds, starts = [], []
        s.rig.hold = lambda reason="": holds.append(reason)
        original = s.runner.start
        s.runner.start = lambda target, task: starts.append((target, task)) or original(target, task)
        plan = s.plan("open the box, put the green pen in the green slot").result.plan
        out = {}
        thread = threading.Thread(target=lambda: out.setdefault("r", s.execute(plan)))
        thread.start()
        deadline = time.time() + 30
        while (s.executor is None or s.executor.state != "estopped") and time.time() < deadline:
            time.sleep(0.01)
        assert s.executor.state == "estopped" and holds and "open_box no longer holds" in holds[0]
        assert starts == [("left:prim", "tucked"), ("right:prim", "home"),
                          ("right:lora_open_box", STRINGS["open_box"])]  # the pen arm never moved
        green = s.executor.runs[plan.steps[1].id]
        assert green.status == "interrupted" and not green.policy_started
        s.release()
        while s.executor.state != "paused" and time.time() < deadline:
            time.sleep(0.01)
        s.resume()  # the lid check runs again (and passes now)
        thread.join(timeout=30)
        assert out["r"].status == "succeeded" and checks.calls.count("open_box") == 3
    finally:
        s.close()


def test_left_arm_keepout_sees_the_right_arm_live(tmp_path, chain_server):
    """Both arms connected: each arm's other_arm zone follows the other arm's measured pose every tick, no bus read."""
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        s.runner.step()
        right_q = tuple(float(v) for v in s.rig.arms["right"].robot.last_q)
        left_q = tuple(float(v) for v in s.rig.arms["left"].robot.last_q)
        assert s.rig.guard.keepouts["left"].other_q == right_q
        assert s.rig.guard.keepouts["right"].other_q == left_q
        assert s.rig.refresh_other_arm().note == "live"
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------- imagination panel
def _pack_jpegs(frames):
    import io

    from PIL import Image

    chunks = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=95)
        chunks.append(buf.getvalue())
    return np.frombuffer(b"".join(chunks), dtype=np.uint8), [len(c) for c in chunks]


def imagining_backend():
    """FakeBackend that also 'imagines' like ws4/imagine/serving.py in lazy mode: an id in the chunk's response, the
    frames (41 canvases of scene | wrist, offsets +2..+42, frame k filled with grey level 5 * offset) on fetch."""
    import dataclasses

    from flux.backends import FakeBackend
    from flux.contract import SO101_CONTRACT

    class Imagining(FakeBackend):
        store: dict = {}

        def predict(self, request):
            resp = super().predict(request)
            if not request.imagine:
                return resp
            iid = f"im{len(self.store)}"
            offsets = list(range(2, 43))
            blob, lengths = _pack_jpegs([np.full((256, 512, 3), 5 * o, np.uint8) for o in offsets])
            meta = {"id": iid, "mode": request.imagine, "frame_offsets": offsets}
            self.store[iid] = ({**meta, "lengths": {"frames": lengths}}, {"frames": blob})
            return dataclasses.replace(resp, imagination={"meta": meta, "arrays": {}})

        def fetch_imagination(self, iid, wait_s):
            return self.store[iid]

    return Imagining(SO101_CONTRACT, velocity=0.0)


def test_imagination_panel_pairs_imagined_and_actual(tmp_path):
    import io
    import time
    from types import SimpleNamespace

    from flux.backends import PolicyRequest
    from flux.contract import IMG_HEAD, WRIST_KEY
    from flux.remote import RemoteBackend, make_server
    from orchestrator.imagination_panel import ImaginationPanel
    from PIL import Image

    server = make_server({"lora_open_box": imagining_backend()}, "127.0.0.1", 0, token="t")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        backend = RemoteBackend(url, "lora_open_box", token="t", imagine="lazy")
        from flux.contract import SO101_CONTRACT

        c = SO101_CONTRACT
        images = {k: np.zeros((c.history_snapshots, *c.image_hw, 3), np.uint8) for k in c.camera_keys}
        resp = backend.predict(PolicyRequest("open the box by pulling its handle", np.zeros((8, 6)), np.zeros((8, 6)),
                                             images, c.snapshots, target="right:lora_open_box"))
        assert resp.info["imagination_id"] == "im0"
        runner = SimpleNamespace(_targets={"right:lora_open_box": SimpleNamespace(last_response=resp)},
                                 active_targets=lambda: ["right:lora_open_box"])
        panel = ImaginationPanel(runner, {"right:lora_open_box": backend})
        assert panel.state() == {"enabled": True, "ready": False, "fetched": 0, "error": ""}
        frame = np.ones((480, 640, 3), np.uint8)
        obs = {IMG_HEAD: 200 * frame, WRIST_KEY["right"]: 50 * frame, WRIST_KEY["left"]: 0 * frame}
        panel.on_frame(obs)
        deadline = time.time() + 10
        while not panel.state()["ready"] and time.time() < deadline:
            time.sleep(0.02)
        st = panel.state()
        assert st["ready"] and st["arm"] == "right" and st["offset"] == 32 and st["ahead_s"] == pytest.approx(1.07)
        imagined = np.asarray(Image.open(io.BytesIO(panel.frame("imagined"))))
        assert imagined.shape == (256, 512, 3) and abs(int(imagined.mean()) - 160) <= 3  # the +32 frame
        actual = np.asarray(Image.open(io.BytesIO(panel.frame("actual"))))
        assert actual.shape == (256, 512, 3)  # head | the right arm's wrist
        assert abs(int(actual[:, :256].mean()) - 200) <= 3 and abs(int(actual[:, 256:].mean()) - 50) <= 3
        panel.on_frame(obs)  # same chunk: no second fetch
        time.sleep(0.1)
        assert panel.state()["fetched"] == 1
        panel.close()
    finally:
        server.shutdown()
        server.server_close()


def test_session_wires_imagination(tmp_path, chain_server):
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token, imagine="lazy")
    try:
        assert s.imagination is not None and all(be.imagine == "lazy" for t, be in s.target_backend.items()
                                                 if not t.endswith(":prim"))
        assert s.snapshot()["imagination"]["enabled"] and s.snapshot()["arms"] == ["left", "right"]
        assert s.frame("imagined") is None  # nothing fetched yet
    finally:
        s.close()


def test_cli_builds_the_chain_config():
    import argparse

    from orchestrator.__main__ import _config, _session_args

    p = argparse.ArgumentParser()
    _session_args(p)
    a = p.parse_args(["--env", "haller", "--skill-package", "open_box=lora_open_box", "--skill-package",
                      "put_pen_in_slot = lora_red_pen", "--imagine", "lazy", "--server", "http://127.0.0.1:8790"])
    cfg = _config(a)
    assert cfg.skill_packages == {"open_box": "lora_open_box", "put_pen_in_slot": "lora_red_pen"}
    assert cfg.imagine == "lazy" and cfg.policy == "remote" and cfg.task == "haller_chain"
    assert cfg.right_arm_port == "/dev/haller_arm_left" and cfg.right_robot_id == "haller_left"
    with pytest.raises(SystemExit):
        _config(p.parse_args(["--env", "haller", "--skill-package", "open_box"]))


def test_tuck_at_the_end_when_asked(tmp_path, chain_server):
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        s.verifier = ScriptedChecks()
        plan = s.plan("put the red pen in its slot, then tuck the arm").result.plan
        assert [(st.skill, st.arm) for st in plan.steps] == [("put_pen_in_slot", "left"), ("tuck", "left")]
        assert s.execute(plan).status == "succeeded"
        tucked = s.vocab.station.poses["left"]["tucked"]
        assert np.abs(s.rig.arms["left"].robot.joints() - tucked).max() <= 3.0
    finally:
        s.close()


def test_hold_keeps_the_grip_and_holding_arms_do_not_stall():
    from types import SimpleNamespace

    from orchestrator.haller import freeze_keep_grip
    from orchestrator.monitors import StallMonitor

    sent = []
    robot = SimpleNamespace(slot="right", last_sent=np.array([0, 0, 0, 0, 0, 5.0]),
                            joints=lambda: np.array([1.0, 2, 3, 4, 5, 30.0]),
                            send_joints=lambda q, measure=False: sent.append(np.array(q)))
    freeze_keep_grip(robot)  # joints frozen where they are, the gripper keeps squeezing the lid's tab
    assert np.allclose(sent[0], [1, 2, 3, 4, 5, 5.0])
    driven = {"right": False}
    stall = StallMonitor("right", stall_deg=8.0, stall_ticks=2, active=lambda arm: driven[arm])
    obs = {"observation.state": np.zeros(12)}
    obs["observation.state"][6 + 1] = 20.0  # lift lags 20 deg under the lid's weight
    cmd = np.zeros(12)
    assert all(stall(SimpleNamespace(tick=t, command=cmd), obs) is None for t in range(1, 6))  # holding: exempt
    driven["right"] = True
    assert stall(SimpleNamespace(tick=6, command=cmd), obs) is None and "shoulder_lift" in stall(
        SimpleNamespace(tick=7, command=cmd), obs)
