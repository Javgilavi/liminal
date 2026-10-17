"""Wire and motion gates; no cameras, serial devices, network, or GPU."""

import asyncio
import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from liminal.core.types import SO101_JOINTS
from liminal.policy.live import (
    WireArm,
    execution_gate,
    legacy_shoulder_elbow,
    pose_limits,
    run,
    selected_state,
    transform,
    vector,
)
from liminal.policy.live_tuning import inference_settings, rollout_duration, worker_arguments
from liminal.safety.supervisor import JointLimits, SafetySupervisor


def profile():
    return {"arm": "right", "joint_scale": [1] * 6, "joint_offset": [0] * 6,
            "camera_verified": True, "units_verified": True, "expected_port": "test-right-port",
            "duration_s": 10}


@pytest.mark.parametrize("launcher", ["infer_robot.sh", "infer_robot_hf_space.sh"])
def test_default_launchers_select_hf_normalization(tmp_path, launcher):
    """Exercise real shell dispatch with a fake interpreter, without hardware or SSH."""
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("infer_robot.sh", "infer_robot_hf_space.sh"):
        target = scripts / name
        target.write_text((root / "scripts" / name).read_text())
        target.chmod(0o755)
    interpreter = tmp_path / ".venv-robot/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@"\n')
    interpreter.chmod(0o755)
    result = subprocess.run([str(scripts / launcher), "--prompt", "Grasp the white cup."],
                            check=True, capture_output=True, text=True)
    arguments = result.stdout.splitlines()
    assert arguments[:2] == ["-m", "liminal.policy.live"]
    assert arguments[-2:] == ["--prompt", "Grasp the white cup."]
    cfg = yaml.safe_load((root / arguments[arguments.index("--config") + 1]).read_text())
    assert cfg["normalization_profile"] == "hf_space_native"
    assert cfg["checkpoint"] == "/liminal/models/so101-hf-space-native-20261017"
    assert cfg["official_lerobot"] is True
    assert cfg["joint_scale"] == [1] * 6
    assert cfg["joint_offset"] == [0] * 6
    assert cfg["duration_s"] == 120


def test_cli_defaults_to_hf_normalization(tmp_path, monkeypatch):
    from liminal.policy import live

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    monkeypatch.setattr("sys.argv", ["live", "--output", str(tmp_path / "run")])
    calls = []

    async def fake_run(cfg, output, execute):
        calls.append(cfg)
        assert execute is False

    monkeypatch.setattr(live, "run", fake_run)
    live.main()
    assert calls[0]["normalization_profile"] == "hf_space_native"
    assert calls[0]["checkpoint"] == "/liminal/models/so101-hf-space-native-20261017"


def test_explicit_right_selection_independent_of_wire_order():
    items = [(f"{side}_{joint}", i + offset) for side, offset in (("left", 100), ("right", 0))
             for i, joint in enumerate(SO101_JOINTS)]
    items.reverse()
    message = {"state_names": [k for k, _ in items], "state": [v for _, v in items]}
    assert selected_state(message, "right") == list(range(6))
    with pytest.raises(KeyError):
        selected_state({"state_names": list(SO101_JOINTS), "state": [0] * 6}, "right")


def test_unprefixed_state_requires_confirmed_single_arm_identity():
    message = {"state_names": list(SO101_JOINTS), "state": list(range(6))}
    assert selected_state(message, "right", single_arm="right") == list(range(6))
    with pytest.raises(KeyError):
        selected_state(message, "right", single_arm="left")


def test_duplicate_or_incomplete_joint_names_refused():
    for names in (["right_gripper"] * 6, ["right_gripper"] * 5):
        with pytest.raises(ValueError):
            selected_state({"state_names": names, "state": [0] * 6}, "right")


@pytest.mark.parametrize("bad", [[0] * 5, [0] * 7, [float("nan")] * 6, [float("inf")] * 6])
def test_nonfinite_and_wrong_dimension_rejected(bad):
    with pytest.raises(ValueError):
        vector(bad)


def test_joint_mapping_is_invertible_and_cannot_rescale_degrees():
    cfg = profile()
    cfg["joint_offset"] = [5, 100, 30, 0, -5, 0]
    cfg["joint_scale"][2] = -1
    q = [1, 2, 3, 4, 5, 6]
    assert transform(transform(q, cfg), cfg, inverse=True) == q
    cfg["joint_scale"][0] = 2
    with pytest.raises(ValueError):
        transform(q, cfg)


def test_legacy_mapping_is_opt_in_and_does_not_mutate_profile():
    original = profile()
    cfg = legacy_shoulder_elbow(original)
    assert original == profile()
    local = [5, -40, 50, 20, 6, 10]
    assert transform(local, cfg) == [5, 130, 140, 20, 6, 10]
    assert transform([5, 130, 140, 20, 6, 10], cfg, inverse=True) == local
    with pytest.raises(ValueError, match="refusing a second mapping"):
        legacy_shoulder_elbow(cfg)


def test_hf_normalization_cannot_be_combined_with_legacy_mapping():
    cfg = {**profile(), "normalization_profile": "hf_space_native"}
    with pytest.raises(ValueError, match="native coordinates"):
        legacy_shoulder_elbow(cfg)


@pytest.mark.parametrize("change", [
    {"joint_offset": [0, 90, 90, 0, -90, 0]},
    {"joint_scale": [1, -1, 1, 1, 1, 1]},
    {"checkpoint": None},
    {"official_lerobot": False},
])
def test_hf_profile_rejects_incompatible_recipe_before_io(tmp_path, monkeypatch, change):
    cfg = {**profile(), "normalization_profile": "hf_space_native", "official_lerobot": True,
           "checkpoint": "/test/hf-package", **change}
    monkeypatch.setattr("liminal.policy.live.http", lambda *args: pytest.fail("unexpected hardware request"))
    with pytest.raises(ValueError, match="HF Space"):
        asyncio.run(run(cfg, tmp_path, execute=False))


def test_official_worker_receives_explicit_checkpoint_and_normalization(monkeypatch):
    from liminal.policy import live

    calls = []

    async def spawn(*args, **kwargs):
        calls.append(args)
        return "worker"

    monkeypatch.setattr(live.asyncio, "create_subprocess_exec", spawn)
    cfg = {"official_lerobot": True, "checkpoint": "/test/package with spaces",
           "normalization_profile": "hf_space_native", "python": "/test/python", "worker": "/test/worker.py",
           "vm": "test", "zone": "test-zone", "project": "test-project"}
    assert asyncio.run(live.remote_worker(cfg, None)) == "worker"
    import shlex

    command = calls[0][-1].removeprefix("--command=").split("; exec ")[1]
    assert shlex.split(command) == ["/test/python", "-u", "/test/worker.py", "--checkpoint",
                                     "/test/package with spaces", "--normalization-profile", "hf_space_native"]


def test_legacy_mapping_transforms_real_observation_and_command_history_parser(tmp_path):
    from liminal.policy.live import LiveRig

    async def exercise():
        cfg = legacy_shoulder_elbow({**profile(), "single_arm": "right", "scene_key": "top"})
        rig = LiveRig(cfg, tmp_path)
        rig.run_id = "mapped-run"
        rig.wrist, rig.last_wrist = "wrist-image", time.monotonic()
        reader = asyncio.StreamReader()
        for index in range(8):
            measured = [index, -40, 50, 20, 6, 10]
            command = [index, -35, 55, 22, 7, 12]
            reader.feed_data((json.dumps({
                "type": "policy_observation", "t_ms": int(time.time() * 1000), "run_id": rig.run_id,
                "state_names": list(SO101_JOINTS), "state": measured,
                "command": command, "command_seq": -1, "images": {"top": "scene-image"},
            }) + "\n").encode())
        reader.feed_eof()
        await rig.observations(reader)
        assert len(rig.history) == 8
        for index, frame in enumerate(rig.history):
            assert frame["state"] == [index, 130, 140, 20, 6, 10]
            assert frame["command_history"] == [index, 125, 145, 22, 7, 12]
            assert frame["scene"] == "scene-image" and frame["wrist"] == "wrist-image"
        assert rig.latest["state"] == [7, -40, 50, 20, 6, 10]

    asyncio.run(exercise())


def test_invalid_coordinate_mapping_fails_before_io(tmp_path, monkeypatch):
    cfg = {**profile(), "joint_scale": [1, 0, 1, 1, 1, 1]}
    monkeypatch.setattr("liminal.policy.live.http", lambda *args: pytest.fail("unexpected hardware request"))
    with pytest.raises(ValueError, match="joint scales"):
        asyncio.run(run(cfg, tmp_path, execute=True))


def test_execution_requires_both_verifications_and_reports_distribution():
    cfg = profile()
    stats = {"q01": [-10] * 6, "q99": [10] * 6}
    execution_gate(cfg, [0] * 6, stats)
    for key in ("camera_verified", "units_verified"):
        unverified = deepcopy(cfg)
        unverified[key] = False
        with pytest.raises(RuntimeError, match="motion blocked"):
            execution_gate(unverified, [0] * 6, stats)
    stats = {"q01": [-10, 85, -10, -10, -10, -10], "q99": [10, 184, 10, 10, 10, 10]}
    assert execution_gate(cfg, [0, -94, 0, 0, 0, 0], stats) == ["shoulder_lift"]


def test_unverified_execution_fails_before_any_io(tmp_path, monkeypatch):
    cfg = profile()
    cfg["camera_verified"] = False
    monkeypatch.setattr("liminal.policy.live.http", lambda *args: pytest.fail("hardware request before verification"))
    with pytest.raises(RuntimeError, match="motion blocked"):
        asyncio.run(run(cfg, tmp_path, execute=True))


def test_stopped_arm_is_refused_before_worker_loading_without_rearming(tmp_path, monkeypatch):
    calls = []

    def read_only_http(route, body=None):
        assert body is None, "must never clear STOP automatically"
        calls.append(route)
        if route == "/config":
            return {"arms": [{"id": "right", "source": "real", "port": "test-right-port"}]}
        return {"ok": True, "mode": "stop"}

    async def no_worker(*args):
        pytest.fail("worker loaded while arm is stopped")

    monkeypatch.setattr("liminal.policy.live.http", read_only_http)
    monkeypatch.setattr("liminal.policy.live.remote_worker", no_worker)
    with pytest.raises(RuntimeError, match="Select MANUAL"):
        asyncio.run(run(profile(), tmp_path, execute=True))
    assert calls == ["/config", "/arm/right/preflight"]


def test_proxy_commands_are_supervised_and_only_selected_arm_exists():
    arm = WireArm("right")
    arm.current = [0] * 6
    safety = SafetySupervisor({"right": arm}, {"right": JointLimits([-10] * 6, [10] * 6, 5 / 30)})
    result = safety.send("right", [100] * 6)
    assert result == pytest.approx([5 / 30] * 6)
    assert arm.pending == result
    assert set(safety.arms) == {"right"}
    safety.estop()
    assert arm.pending is None
    with pytest.raises(RuntimeError):
        safety.send("right", [0] * 6)


def test_live_limits_allow_task_travel_without_start_relative_or_lead_caps():
    cells = {j: {"min": -180, "max": 180} for j in SO101_JOINTS}
    lower, upper = pose_limits(cells, [0] * 6)
    arm = WireArm("right")
    arm.current = [0] * 6
    full_travel = max(hi - lo for lo, hi in zip(lower, upper, strict=True))
    safety = SafetySupervisor({"right": arm}, {"right": JointLimits(lower, upper, full_travel)})
    # A >5-degree request must reach Haller, which owns physical slew limiting.
    assert safety.send("right", [40] * 6) == [40] * 6
    assert safety.send("right", [120] * 6) == [120] * 6
    assert safety.send("right", [999] * 6) == upper
    assert safety.send("right", [-999] * 6) == lower
    safety.estop()
    assert arm.pending is None
    with pytest.raises(RuntimeError):
        safety.send("right", [40] * 6)


@pytest.mark.parametrize("key", ["max_speed", "max_tracking_error", "max_excursion", "coordinated_slowdown"])
def test_obsolete_caps_are_not_silently_ignored(tmp_path, monkeypatch, key):
    cfg = {**profile(), key: 1}
    monkeypatch.setattr("liminal.policy.live.http", lambda *args: pytest.fail("unexpected hardware request"))
    with pytest.raises(ValueError, match="obsolete live bridge caps"):
        asyncio.run(run(cfg, tmp_path, execute=True))


def test_chosen_start_does_not_narrow_calibrated_travel():
    cells = {j: {"min": -100, "max": 100} for j in SO101_JOINTS}
    lower, upper = pose_limits(cells, [0] * 6)
    assert lower == [-100] * 6 and upper == [100] * 6
    for index, value in [(2, 97), (5, 99), (0, -99)]:
        q = [0] * 6
        q[index] = value
        lower, upper = pose_limits(cells, q)
        assert lower == [-100] * 6 and upper == [100] * 6
    for index, value in [(2, 101), (4, 101), (5, -101)]:
        q = [0] * 6
        q[index] = value
        with pytest.raises(RuntimeError, match="outside allowed limits"):
            pose_limits(cells, q)


@pytest.mark.parametrize("scenario", [
    "duration", "inference_failure", "inference_failure_cleanup_failure", "unlimited_cancel", "old_controller",
    "official_native", "official_mapped", "official_mapped_invalid", "official_mapped_observe",
    "official_hf_native", "official_hf_wrong_normalization",
])
def test_full_mocked_lifecycle_preserves_other_arm_and_never_engages(tmp_path, monkeypatch, scenario):
    from liminal.policy import live

    official = scenario.startswith("official_")
    mapped = scenario.startswith("official_mapped")
    cfg = {**profile(), "duration_s": None if scenario == "unlimited_cancel" or official else 0.01,
           "prompt": "stack the cups"}
    if official:
        cfg["official_lerobot"] = True
    if scenario.startswith("official_hf_"):
        cfg.update(normalization_profile="hf_space_native", checkpoint="/test/hf-package")
    if mapped:
        cfg = legacy_shoulder_elbow(cfg)
    calls = []
    sent = []

    def fake_http(route, body=None):
        calls.append((route, body))
        if route == "/config":
            return {"arms": [{"id": "right", "source": "real", "port": "test-right-port"}]}
        return {"ok": True}

    async def park(self):
        await asyncio.Event().wait()

    async def fake_observations(self, reader):
        self.latest = {"state": [0, -40, 50, 20, 5, 10] if mapped else [0] * 6,
                       "scene": "eA==", "wrist": "eA=="}
        self.last_observation = self.last_telemetry = self.last_wrist = time.monotonic()
        cells = {j: {"min": -100, "max": 100} for j in SO101_JOINTS}
        self.telemetry = {"arms": {"right": {"mode": "manual", "torque": True, "joints": cells}}}
        initial_model = transform(self.latest["state"], cfg)
        self.history.extend([{**self.latest, "state": initial_model, "command_history": initial_model}] * 8)
        await asyncio.Event().wait()

    class Writer:
        def write(self, raw):
            message = json.loads(raw)
            sent.append(message)
            if scenario == "unlimited_cancel" and message.get("type") == "policy_action" and message["seq"] == 1:
                asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)

        async def drain(self):
            pass

        def close(self):
            pass

        async def wait_closed(self):
            pass

    class Reader:
        def __init__(self, messages):
            self.messages = iter(messages)

        async def readline(self):
            return (json.dumps(next(self.messages)) + "\n").encode()

    class Process:
        stdin = Writer()

        def __init__(self):
            result = {"type": "error", "detail": "GPU failed"} if scenario.startswith("inference_failure") else {
                "type": "actions", "actions": [[40] * 6] * 42, "inference_seconds": 0.1,
            }
            ready = {"type": "ready", "checkpoint": "mock", "inference": {
                **inference_settings(cfg), "control_hz": 30, "chunk_size": 42,
            }, "state_statistics": {
                "q01": [-10] * 6, "q99": [10] * 6,
            }}
            if official:
                ready.update(integration="lerobot.Flux3Policy + saved pre/post processors", source_revision="mock")
                ready["inference"].update(fps=30, inference_seed=42)
                result["actions"] = [[i / 10] * 6 for i in range(42)]
                if mapped:
                    result["actions"] = [[i / 10, 130 + i / 10, 140 + i / 10, 20, 5, 10] for i in range(42)]
                if scenario in ("official_mapped_invalid", "official_mapped_observe"):
                    result["actions"][31][2] = -20  # Model-valid number, but robot elbow = -110.
                if scenario == "official_hf_native":
                    ready["normalization_profile"] = "hf_space_native"
            self.stdout = Reader([ready, result, {"type": "error", "detail": "mock rollout complete"}])

        async def wait(self):
            if scenario == "inference_failure_cleanup_failure":
                raise RuntimeError("worker cleanup failed")
            return 0

    async def worker(*args):
        return Process()

    async def connection(*args, **kwargs):
        hello = {"type": "policy_hello_ack", "ok": True}
        if scenario != "old_controller":
            hello["command_feedback"] = 1
        return Reader([hello]), Writer()

    async def fake_send_command(self, writer, seq, target):
        writer.write((json.dumps({"type": "policy_action", "seq": seq,
                                 "action": {"right": dict(zip(SO101_JOINTS, target, strict=True))}}) + "\n").encode())
        self.command = list(target)
        self.command_seq = self.observed_command_seq = seq
        if official:
            self.last_observation = self.last_telemetry = self.last_wrist = time.monotonic()
        await asyncio.sleep(0)
        return {}

    monkeypatch.setattr(live.LiveRig, "send_command", fake_send_command)
    monkeypatch.setattr(live, "http", fake_http)
    monkeypatch.setattr(live, "remote_worker", worker)
    monkeypatch.setattr(live.asyncio, "open_connection", connection)
    monkeypatch.setattr(live.LiveRig, "telemetry_loop", park)
    monkeypatch.setattr(live.LiveRig, "cameras", park)
    monkeypatch.setattr(live.LiveRig, "observations", fake_observations)
    if scenario == "official_hf_wrong_normalization":
        with pytest.raises(RuntimeError, match="different normalization statistics"):
            asyncio.run(live.run(cfg, tmp_path, execute=True))
        assert not any(m.get("type") in ("policy_hello", "policy_action") for m in sent)
        assert not any(body is not None for _, body in calls)
    elif scenario == "official_mapped_observe":
        asyncio.run(live.run(cfg, tmp_path, execute=False))
        report = json.loads((tmp_path / "report.json").read_text())
        assert "elbow_flex=-110" in report["execution_blocker"]
        assert not any(m.get("type") == "policy_action" for m in sent)
        assert not any(body is not None for _, body in calls)
    elif scenario == "official_mapped_invalid":
        with pytest.raises(RuntimeError, match="elbow_flex=-110"):
            asyncio.run(live.run(cfg, tmp_path, execute=True))
        assert not any(m.get("type") == "policy_action" for m in sent)
        assert [(r, b) for r, b in calls if b is not None] == [("/arm/right/mode", {"mode": "stop"})]
    elif official:
        with pytest.raises(RuntimeError, match="mock rollout complete"):
            asyncio.run(live.run(cfg, tmp_path, execute=True))
        actions = [m for m in sent if m.get("type") == "policy_action"]
        # Exactly the first 32 original targets, followed by another prediction:
        # no interpolation, rebasing, duplication or execution of the final ten.
        expected = ([[i / 10, -40 - i / 10, 50 + i / 10, 20, 5, 10] for i in range(32)]
                    if mapped else [[i / 10] * 6 for i in range(32)])
        for message, target in zip(actions, expected, strict=True):
            assert list(message["action"]["right"].values()) == pytest.approx(target)
        assert [(r, b) for r, b in calls if b is not None] == [("/arm/right/mode", {"mode": "stop"})]
    elif scenario.startswith("inference_failure") or scenario == "old_controller":
        error = "feedback update" if scenario == "old_controller" else "GPU failed"
        with pytest.raises(RuntimeError, match=error):
            asyncio.run(live.run(cfg, tmp_path, execute=True))
        assert not any(body is not None for _, body in calls)
        assert not any(m.get("type") == "policy_action" for m in sent)
    else:
        if scenario == "unlimited_cancel":
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(live.run(cfg, tmp_path, execute=True))
        else:
            asyncio.run(live.run(cfg, tmp_path, execute=True))
        actions = [m for m in sent if m.get("type") == "policy_action"]
        assert actions and all(set(m["action"]) == {"right"} for m in actions)
        assert all(list(m["action"]["right"].values()) == [40] * 6 for m in actions)
        assert [(r, b) for r, b in calls if b is not None] == [("/arm/right/mode", {"mode": "stop"})]
    assert not any("left" in route for route, _ in calls)
    if scenario not in ("old_controller", "official_hf_wrong_normalization"):
        saved = json.loads((tmp_path / "input-0000.json").read_text())
        assert saved["prompt"] == cfg["prompt"]
        assert len(saved["history"]) == 8
        assert all({"state", "command_history", "scene", "wrist"} <= frame.keys()
                   for frame in saved["history"])
        if mapped:
            assert saved["history"][0]["state"] == [0, 130, 140, 20, 5, 10]
            assert saved["history"][0]["command_history"] == [0, 130, 140, 20, 5, 10]
            report = json.loads((tmp_path / "report.json").read_text())
            assert report["coordinate_mapping"]["name"] == "experimental_legacy_shoulder_elbow"
            assert report["first_robot_actions"][0] == [0, -40, 50, 20, 5, 10]
            plan = json.loads((tmp_path / "plans.jsonl").read_text().splitlines()[0])
            assert plan["actions"][0] == [0, 130, 140, 20, 5, 10]
            assert plan["robot_actions"][0] == [0, -40, 50, 20, 5, 10]
    if scenario == "inference_failure_cleanup_failure":
        report = json.loads((tmp_path / "report.json").read_text())
        assert report["error"] == "GPU failed"
        assert report["cleanup_errors"] == ["GPU worker cleanup: worker cleanup failed"]


def test_inference_overrides_and_unlimited_duration():
    cfg = {"inference": {"n_action_steps": 16, "num_inference_steps": 8, "seed": 7}}
    settings = inference_settings(cfg)
    assert settings["n_action_steps"] == 16 and settings["num_inference_steps"] == 8
    assert settings["guidance_scale"] == 3 and settings["sampler_shift"] == 6.93
    arguments = worker_arguments(settings)
    assert arguments[arguments.index("--num-inference-steps") + 1] == "8"
    assert "--guidance-scale-action" not in arguments
    assert rollout_duration({}) is None
    assert rollout_duration({"duration_s": None}) is None
    assert rollout_duration({"duration_s": 3600}) == 3600


@pytest.mark.parametrize("inference", [
    {"n_action_steps": 0}, {"n_action_steps": 43}, {"num_inference_steps": 0},
    {"n_action_steps": True}, {"guidance_scale": float("nan")},
    {"sampler_shift": 0}, {"seed": -1}, {"unknown": 4},
])
def test_invalid_inference_overrides_are_rejected(inference):
    with pytest.raises(ValueError):
        inference_settings({"inference": inference})
