"""Inter-arm collision gate on the ws4/kinematics MuJoCo model (no rendering)."""

import dataclasses

import numpy as np
import pytest
from flux.backends import FakeBackend
from flux.contract import SO101_CONTRACT
from flux.dual_arm_runner import per_arm_target
from orchestrator.gate import CollisionGate, GatedDualArmRunner, TickGuardRobot
from orchestrator.primitives import HOME_DEG


@pytest.fixture(scope="module")
def gate():
    return CollisionGate()


@pytest.fixture(scope="module")
def meeting_poses():
    """Both grippers top-down at the same point between the arms: they must collide."""
    from orchestrator.sim_backend import SCENE
    from sim.env import XLeRobotStationEnv
    from sim.ik import ArmIK, top_down_rotation

    env = XLeRobotStationEnv(scene=SCENE, randomize=False, render_images=False)
    env.reset(seed=0)
    poses = {}
    for arm in ("left", "right"):
        q5, err, _ = ArmIK(env.model, arm).solve(env.data.qpos.copy(), np.array([0.25, 0.0, 0.10]),
                                                 top_down_rotation(np.pi / 2))
        assert err < 0.005
        q6 = env.units.arms[arm].from_radians(np.r_[q5, 0.0])
        q6[5] = 40.0
        poses[arm] = q6
    return poses


def test_home_is_clear_and_meeting_collides(gate, meeting_poses):
    home = np.r_[HOME_DEG, HOME_DEG]
    assert gate.command_penetration(home) == 0.0
    assert gate.command_penetration(np.r_[meeting_poses["left"], meeting_poses["right"]]) > 0.0
    assert gate.command_penetration(np.r_[meeting_poses["left"], HOME_DEG]) == 0.0


def _runner(robot, gate, chunk):
    backend = FakeBackend(SO101_CONTRACT, output=chunk)
    target = dataclasses.replace(per_arm_target("right", "fake"), name="right:x")
    return GatedDualArmRunner(robot, [target], {"fake": backend}, gate=gate)


def test_chunk_into_the_other_arm_is_truncated(gate, meeting_poses, follow_robot):
    robot = follow_robot(np.r_[meeting_poses["left"], HOME_DEG])
    n = SO101_CONTRACT.chunk_size
    chunk = HOME_DEG[None] + (meeting_poses["right"] - HOME_DEG)[None] * np.linspace(0, 1, n)[:, None]
    runner = _runner(robot, gate, chunk)
    blocked = []
    runner.on_blocked = lambda arm, other: blocked.append((arm, other))
    before = gate.stats.truncated + gate.stats.refused
    runner.start("right:x", "go")
    for _ in range(SO101_CONTRACT.n_action_steps):
        report = runner.step()
        assert gate.command_penetration(report.command) == 0.0
    assert gate.stats.truncated + gate.stats.refused == before + 1
    assert blocked == [("right", "left")]
    assert np.abs(robot.state[6:] - meeting_poses["right"]).max() > 1.0  # it stopped short


def test_moving_apart_is_allowed(gate, meeting_poses, follow_robot):
    """Arms already touching (as after an e-stop mid-motion) may back out along the way they came in."""
    path = HOME_DEG[None] + (meeting_poses["right"] - HOME_DEG)[None] * np.linspace(0, 1, 200)[:, None]
    depth = gate.penetration({"left": np.repeat(meeting_poses["left"][None], 200, 0), "right": path})
    first = int(np.flatnonzero(depth > 0)[0])
    touching = path[first]
    robot = follow_robot(np.r_[meeting_poses["left"], touching])
    assert gate.command_penetration(robot.state) > 0
    n = SO101_CONTRACT.chunk_size
    chunk = touching[None] + (HOME_DEG - touching)[None] * np.linspace(0, 1, n)[:, None]
    runner = _runner(robot, gate, chunk)
    before = gate.stats.truncated + gate.stats.refused
    runner.start("right:x", "back off")
    runner.step()
    assert gate.stats.truncated + gate.stats.refused == before


def test_tick_guard_latches_estop(gate, meeting_poses, follow_robot):
    robot = follow_robot(np.r_[meeting_poses["left"], HOME_DEG])
    reasons = []
    guarded = TickGuardRobot(robot, gate, reasons.append)
    guarded.get_observation()
    guarded.send_action(np.r_[meeting_poses["left"], HOME_DEG])
    assert robot.sent == 1
    guarded.send_action(np.r_[meeting_poses["left"], meeting_poses["right"]])
    assert robot.sent == 1 and reasons and "e-stop latched" in reasons[0]


def test_gate_is_fast(gate, meeting_poses):
    frames = np.repeat(meeting_poses["left"][None], 32, 0)
    gate.penetration({"left": frames, "right": frames})
    assert gate.stats.as_dict()["ms_per_frame"] < 1.0
