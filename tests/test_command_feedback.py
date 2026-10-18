"""Acknowledged targets, intervention handling, and fail-closed wire behavior."""

import asyncio
import json
import time
from copy import deepcopy

import pytest

from liminal.core.types import SO101_JOINTS
from liminal.policy.command_feedback import committed_target, rebase_action
from liminal.policy.live import LiveRig


def ack(seq=0, value=48.0):
    return {"type": "policy_action_ack", "run_id": "run", "seq": seq, "t_ms": int(time.time() * 1000),
            "result": {"run_id": "run", "seq": seq, "sides": {"right": {
                "arm": "right", "committed": dict.fromkeys(SO101_JOINTS, value),
                "altered": dict.fromkeys(SO101_JOINTS, "rate_cap")}}}}


def test_rebase_does_not_accumulate_unexecuted_motion_or_delta_encode_gripper():
    # FLUX's next planned elbow target is 75 after 74, but Haller issued 48.
    result = rebase_action([75] * 5 + [30], [74] * 5 + [20], [48] * 5 + [10])
    assert result == [49] * 5 + [30]
    assert committed_target(ack(), "run", 0, "right") == [48] * 6


@pytest.mark.parametrize("problem", ["run", "sequence", "missing", "nonfinite", "refused", "dropped", "arm"])
def test_untrustworthy_commit_is_rejected(problem):
    message = deepcopy(ack())
    side = message["result"]["sides"]["right"]
    if problem == "run":
        message["run_id"] = "other"
    elif problem == "sequence":
        message["seq"] = 1
    elif problem == "missing":
        side["committed"].pop("elbow_flex")
    elif problem == "nonfinite":
        side["committed"]["elbow_flex"] = float("nan")
    elif problem in ("refused", "dropped"):
        side[problem] = "elbow_flex"
    else:
        side["arm"] = "left"
    with pytest.raises(RuntimeError):
        committed_target(message, "run", 0, "right")


def test_ack_and_observation_use_committed_target_not_requested_or_measured(tmp_path):
    async def exercise():
        rig = LiveRig({"arm": "right", "single_arm": "right", "scene_key": "top",
                       "joint_scale": [1] * 6, "joint_offset": [0] * 6}, tmp_path)
        rig.run_id = "run"
        rig.wrist, rig.last_wrist = "eA==", time.monotonic()
        reader = asyncio.StreamReader()

        class Writer:
            def write(self, raw):
                assert list(json.loads(raw)["action"]["right"].values()) == [74] * 6
                reader.feed_data((json.dumps(ack()) + "\n").encode())
                # Measured 47, actually committed 48, requested 74: all differ.
                observation = {"type": "policy_observation", "run_id": "run", "command_seq": 0,
                               "t_ms": int(time.time() * 1000), "state_names": list(SO101_JOINTS),
                               "state": [47] * 6, "command": [48] * 6, "images": {"top": "eA=="}}
                reader.feed_data((json.dumps(observation) + "\n").encode())

            async def drain(self):
                pass

        task = asyncio.create_task(rig.observations(reader))
        try:
            await rig.send_command(Writer(), 0, [74] * 6)
            await rig.wait_observed(0)
            assert rig.command == [48] * 6
            assert rig.history[-1]["state"] == [47] * 6
            assert rig.history[-1]["command_history"] == [48] * 6
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["disconnect", "stale", "wrong_sequence", "timeout"])
def test_wire_fault_never_promotes_requested_target_to_committed(tmp_path, failure):
    async def exercise():
        rig = LiveRig({"arm": "right"}, tmp_path)
        rig.run_id = "run"
        reader = asyncio.StreamReader()

        class Writer:
            def write(self, raw):
                if failure == "disconnect":
                    reader.feed_eof()
                elif failure != "timeout":
                    message = ack(seq=2 if failure == "wrong_sequence" else 0)
                    if failure == "stale":
                        message["t_ms"] -= 2000
                    reader.feed_data((json.dumps(message) + "\n").encode())

            async def drain(self):
                pass

        task = asyncio.create_task(rig.observations(reader))
        try:
            with pytest.raises(RuntimeError):
                await rig.send_command(Writer(), 0, [74] * 6)
            assert rig.command is None
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
