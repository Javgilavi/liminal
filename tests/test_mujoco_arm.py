from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from liminal.app import _load, build_arms  # noqa: E402
from liminal.sim.mujoco_arm import MujocoArm, MujocoWorld  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCENE = str(ROOT / "examples/dual_so101_sim/scene.xml")


def _settle(world, seconds=1.5):
    for _ in range(int(seconds / world.model.opt.timestep)):
        world.mj.mj_step(world.model, world.data)


def test_mujoco_arm_uses_real_so101_units():
    world = MujocoWorld(SCENE)
    arm = MujocoArm("left", world, "left_")
    assert arm.num_joints == 6 and list(arm.grip_range) == [5]
    assert arm.read_state() == pytest.approx([0, 0, 0, 0, 0, arm.read_state()[5]], abs=1e-3)
    target = [20.0, -30.0, 40.0, 15.0, -10.0, 50.0]  # degrees, gripper %
    arm.send_action(target)
    lo, hi = arm.grip_range[5]
    assert world.data.ctrl[arm.ids[0]] == pytest.approx(0.349066, abs=1e-5)  # 20 deg in radians
    assert world.data.ctrl[arm.ids[5]] == pytest.approx((lo + hi) / 2)
    _settle(world)
    assert arm.read_state() == pytest.approx(target, abs=1.0)
    arm.send_action([0, 0, 0, 0, 0, 150])  # gripper clamps to 100 % = fully open
    assert world.data.ctrl[arm.ids[5]] == pytest.approx(hi)
    with pytest.raises(ValueError):
        arm.send_action([0.0] * 5)


def test_mujoco_dual_config_matches_real_format(monkeypatch):
    monkeypatch.chdir(ROOT)
    cfg, real = _load("configs/robot/mujoco_dual.yaml"), _load("configs/robot/so101_dual.yaml")
    for key in ("arms", "arm_groups", "cameras", "limits", "home", "bases"):
        assert set(cfg[key]) == set(real[key]), key
    arms, cameras = build_arms(cfg)
    try:
        assert sorted(cameras) == ["left_wrist", "right_wrist", "top"]
        for name, arm in arms.items():
            lower, upper = cfg["limits"][name]["lower"], cfg["limits"][name]["upper"]
            assert all(lo <= h <= hi for lo, h, hi in zip(lower, cfg["home"][name], upper, strict=True))
            assert all(lo <= s <= hi for lo, s, hi in zip(lower, arm.read_state(), upper, strict=True))
    finally:
        arms["left"].world.stop()
