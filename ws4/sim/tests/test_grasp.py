"""The soft fin gripper grasps and lifts a 4 cm cube and a small cup (scripted IK pick, no rendering)."""

import pytest

from ws4.sim.env import XLeRobotStationEnv
from ws4.sim.scripted_pick import run_pick

from .conftest import SIM


@pytest.mark.parametrize("obj,side", [("cube_red", "left"), ("cube_blue", "right"), ("cup", "left")])
def test_pick_and_lift(obj, side):
    env = XLeRobotStationEnv(scene=SIM / "scenes" / "grasp_test.xml", randomize=False, render_images=False)
    env.reset(seed=0)
    log = run_pick(env, obj, side)
    env.close()
    assert log["success"], log
    assert log["lift_height"] > 0.08
