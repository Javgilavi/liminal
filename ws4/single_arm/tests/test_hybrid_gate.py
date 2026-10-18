"""Opus gate on run_single_arm (hybrid_gate.py), fake arm and fake policy, a fake reasoner in place of the API."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO / "src"))

pytest.importorskip("cv2")

import run_single_arm as rsa  # noqa: E402

import liminal.policy.hybrid as hybrid  # noqa: E402


def decision(payload, mode):
    pose = payload["current_eef"]
    return {"request_id": payload["request_id"], "mode": mode, "steps": 2, "reason": f"test {mode}",
            "edit": {"delta_position": [0, 0, .001], "delta_rotation_vector": [0, 0, 0], "gripper": "keep"},
            "target": {"position": pose["position"], "quaternion_wxyz": pose["quaternion_wxyz"],
                       "gripper_closed": False},
            "assessment": {"task_progress": {"verified_completed": [], "currently_attempting": "pick",
                                              "remaining": ["place"]},
                           "current_subgoal": "pick", "execution_status": "failed",
                           "execution_evidence": "empty gripper", "expected_next_intent": "pick",
                           "predicted_next_intent": "place", "intent_status": "misaligned",
                           "intent_evidence": "proposal moves away"}}


def fake_reasoner(mode, calls):
    class Reasoner:
        def __init__(self, config):
            self.usage = {}

        async def decide(self, payload, images, schema):
            calls.append((payload, [image["label"] for image in images]))
            if mode == "error":
                raise TimeoutError("no answer")
            return decision(payload, mode)

    return Reasoner


def gate_config(tmp_path, **changes):
    raw = yaml.safe_load((REPO / "configs/hybrid/claude_opus_haller.yaml").read_text())
    raw.update(model_path=str(REPO / "sim/single_arm/so101.xml"), **changes)
    path = tmp_path / "gate.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def attempt(tmp_path, monkeypatch, mode, *extra):
    calls = []
    monkeypatch.setattr(hybrid, "ClaudeReasoner", fake_reasoner(mode, calls))
    monkeypatch.setattr(rsa, "RUNS", tmp_path)
    monkeypatch.setattr(rsa, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(rsa, "KEY_FILE", tmp_path / "KEY")
    rsa.main(["--robot", "fake", "--seconds", "2", "--yes", "--no-video", "--no-log", "--fake-delay", "0.01",
              "--keepout-ticks", "100000", "--hybrid-config", str(gate_config(tmp_path)), *extra])
    meta = json.loads(next(tmp_path.glob("*/attempt01/meta.json")).read_text())
    trace = np.load(next(tmp_path.glob("*/attempt01/trace.npz")))
    events = [json.loads(line) for line in next(tmp_path.glob("hybrid_*.jsonl")).read_text().splitlines()]
    return meta, trace, events, calls


def test_continue_lets_flux_chunks_through_the_guard(tmp_path, monkeypatch):
    meta, trace, events, calls = attempt(tmp_path, monkeypatch, "student")
    assert meta["end_reason"] == "timeout" and meta["chunks"] >= 2
    assert len(calls) >= 2 and calls[0][1] == ["current_scene", "current_wrist"]
    assert calls[1][1] == ["previous_before_scene", "previous_before_wrist", "current_scene", "current_wrist"]
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "hybrid_attempt" and "hybrid_decision" in kinds and "hybrid_executed" in kinds
    assert all("data" not in str(e.get("images", "")) for e in events)
    assert np.abs(np.diff(trace["command"], axis=0)).max() > 0  # FLUX's commands reached the arm (from rest: gripper)


@pytest.mark.parametrize("mode", ["stop", "eef", "edit", "error"])
def test_stop_correction_or_error_is_the_runner_freeze(tmp_path, monkeypatch, mode):
    meta, trace, events, calls = attempt(tmp_path, monkeypatch, mode)
    assert meta["end_reason"].startswith("stop (error HybridStop")
    assert len(calls) == 1  # frozen at the first chunk boundary: nothing more is asked or sent
    commands = trace["command"]
    assert np.abs(commands - commands[0]).max() == 0  # the arm only ever held its start pose


def test_gate_refuses_verified_mapping_and_prefetch(tmp_path, monkeypatch):
    monkeypatch.setattr(rsa, "RUNS", tmp_path)
    base = ["--robot", "fake", "--seconds", "1", "--yes", "--no-video", "--no-log"]
    with pytest.raises(SystemExit, match="continue/stop only"):
        rsa.main([*base, "--hybrid-config", str(gate_config(tmp_path, kinematics_verified=True))])
    with pytest.raises(SystemExit, match="no prefetch"):
        rsa.main([*base, "--hybrid-config", str(gate_config(tmp_path)), "--prefetch-ticks", "4"])


def test_every_nth_chunk_is_reviewed_and_the_rest_pass_unchanged(tmp_path, monkeypatch):
    meta, trace, events, calls = attempt(tmp_path, monkeypatch, "student", "--hybrid-every", "3", "--action-steps", "8")
    assert meta["end_reason"] == "timeout" and meta["chunks"] >= 4
    # chunk 1, 4, 7, ... (a request still in flight at the timeout may be one more than the accepted chunks)
    assert len(calls) in {-(-meta["chunks"] // 3), -(-(meta["chunks"] + 1) // 3)} and len(calls) < meta["chunks"]
    executed = [e["executed_joints"] for e in events if e["kind"] == "hybrid_executed"]
    assert executed and all(len(x) <= 33 for x in executed)  # 3 chunks of commands, subsampled for the payload
    assert next(e for e in events if e["kind"] == "hybrid_attempt")["every"] == 3
