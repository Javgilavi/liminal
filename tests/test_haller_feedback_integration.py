"""Real bridge/client over in-memory streams; mock servos, no network or GPU.

Run with Haller installed (or its backend on PYTHONPATH) to exercise both repos.
"""

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("haller_hmi")

from haller_hmi.arm import ArmHandle
from haller_hmi.config import ArmConfig, MotionConfig
from haller_hmi.policy_bridge import FiniteActionIngest, PolicyBridge
from haller_hmi.safety import Mode
from haller_hmi.tick import TickBus

from liminal.core.types import SO101_JOINTS
from liminal.policy.live import LiveRig


def test_real_feedback_preserves_targets_and_stops_on_disconnect(tmp_path):
    async def exercise():
        handle = ArmHandle(ArmConfig(id="right", model="so101_follower", port="unused", calibration_id="mock"),
                           motion=MotionConfig(max_speed_deg_s=20, ramp_hz=50))
        handle.robot = MagicMock()
        handle.torque_enabled = True
        handle.joint_limits_deg = {j: (-100, 100) for j in SO101_JOINTS}
        handle._last_commanded = dict.fromkeys(SO101_JOINTS, 0.0)
        handle.state_snapshot = lambda: {"joints": {j: {"pos": 0.0} for j in SO101_JOINTS}}
        handle.disable_torque = lambda: setattr(handle, "torque_enabled", False)
        bridge = PolicyBridge(arms={"right": handle}, tick_bus=TickBus(),
                              get_cameras=lambda: None, get_recorder=lambda: None,
                              calibration=SimpleNamespace(current=None))

        def observe():
            payload = bridge.observe()
            payload["images"] = {"top": "eA=="}
            return payload

        server = FiniteActionIngest(bus_conflict=lambda: None, submit=bridge.submit,
                                    observe=observe, on_session_end=bridge.release)
        bridge.attach_ingest(server)
        client_reader, server_reader = asyncio.StreamReader(), asyncio.StreamReader()

        class Writer:
            def __init__(self, peer):
                self.peer = peer

            def get_extra_info(self, name):
                return "in-memory"

            def write(self, raw):
                self.peer.feed_data(raw)

            async def drain(self):
                await asyncio.sleep(0)

            def close(self):
                self.peer.feed_eof()

        client_writer = Writer(server_reader)
        task = asyncio.create_task(server._handle_client(server_reader, Writer(client_reader)))
        rig = LiveRig({"arm": "right", "single_arm": "right", "scene_key": "top",
                       "joint_scale": [1] * 6, "joint_offset": [0] * 6}, tmp_path)
        rig.run_id = "run"
        rig.wrist = "eA=="
        rig.last_wrist = rig.last_telemetry = time.monotonic()
        rig.telemetry = {"arms": {"right": {"mode": "manual", "torque": True}}}
        reader_task = None
        try:
            client_writer.write((json.dumps({"type": "policy_hello", "run_id": "run", "unit": "deg",
                                             "rig": "solo", "control_hz_declared": 30,
                                             "command_feedback": 1}) + "\n").encode())
            hello = json.loads(await asyncio.wait_for(client_reader.readline(), 1))
            assert hello["command_feedback"] == 1
            reader_task = asyncio.create_task(rig.observations(client_reader))
            for seq in range(4):
                # Pretend inference paused for a second: no accumulated step credit.
                handle._last_command_at = time.monotonic() - 1.0
                prediction = [40.0 + seq] * 5 + [20.0]
                await rig.send_command(client_writer, seq, prediction)
                await rig.wait_observed(seq)
                assert rig.command == pytest.approx(prediction)
                assert rig.history[-1]["command_history"] == pytest.approx(rig.command)
                assert rig.history[-1]["state"] == [0.0] * 6
                actual = handle.robot.send_action.call_args.args[0]
                assert list(actual.values()) == pytest.approx(rig.command)
            assert server.actions_committed == 4
        finally:
            client_writer.close()
            await asyncio.wait_for(task, 1)
            if reader_task:
                await asyncio.wait_for(reader_task, 1)
        assert not handle.torque_enabled
        assert handle.guard.mode == Mode.STOP

    asyncio.run(exercise())
