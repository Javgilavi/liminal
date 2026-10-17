"""XLeRobotStationEnv API, observation format, determinism, rendering speed."""

import time

import numpy as np
import pytest

from ws4.sim.env import XLeRobotStationEnv


@pytest.fixture(scope="module")
def env_noimg():
    env = XLeRobotStationEnv(render_images=False)
    yield env
    env.close()


def test_reset_step_state(env_noimg):
    obs, info = env_noimg.reset(seed=3)
    s = obs["observation.state"]
    assert s.shape == (12,) and s.dtype == np.float32
    np.testing.assert_allclose(s[:5], env_noimg.home[:5], atol=1.0)
    obs, r, term, trunc, info = env_noimg.step(s)
    assert set(obs["privileged"]) >= {"object_poses", "ee_poses", "qpos_rad"}
    assert obs["privileged"]["ee_poses"]["left"].shape == (7,)


def test_hold_still(env_noimg):
    obs, _ = env_noimg.reset(seed=0)
    a = obs["observation.state"].copy()
    for _ in range(30):
        obs, *_ = env_noimg.step(a)
    assert np.abs(obs["observation.state"][:5] - a[:5]).max() < 1.0  # degrees
    assert np.abs(obs["observation.state"][6:11] - a[6:11]).max() < 1.0


def test_tracks_a_joint_command(env_noimg):
    obs, _ = env_noimg.reset(seed=0)
    a = obs["observation.state"].copy()
    a[0] += 20.0   # left shoulder_pan +20 deg
    a[8] -= 15.0   # right elbow_flex -15 deg
    for _ in range(45):
        obs, *_ = env_noimg.step(a)
    s = obs["observation.state"]
    assert abs(s[0] - a[0]) < 1.5 and abs(s[8] - a[8]) < 1.5


def test_seeded_randomisation_is_deterministic():
    e1 = XLeRobotStationEnv(render_images=False)
    e2 = XLeRobotStationEnv(render_images=False)
    p1 = e1.reset(seed=11)[0]["privileged"]["object_poses"]
    p2 = e2.reset(seed=11)[0]["privileged"]["object_poses"]
    p3 = e2.reset(seed=12)[0]["privileged"]["object_poses"]
    for k in p1:
        np.testing.assert_allclose(p1[k], p2[k], atol=1e-9)
    assert any(np.linalg.norm(p1[k][:2] - p3[k][:2]) > 1e-3 for k in p1)
    # props rest on the table after reset
    for k, v in p1.items():
        assert -0.01 < v[2] < 0.2, (k, v)


def test_images_and_arm_view(can_render):
    env = XLeRobotStationEnv(depth=True)
    obs, _ = env.reset(seed=0)
    for cam in ("head", "left_wrist", "right_wrist"):
        img = obs[f"observation.images.{cam}"]
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8 and img.std() > 5
    depth = obs["observation.images.head_depth"]
    assert depth.shape == (480, 640, 1) and depth.dtype == np.uint16
    # table is 0.3 to 1.5 m from the head camera
    assert 300 < np.median(depth) < 1500
    v = env.arm_view("right", obs)
    assert set(v) == {"scene", "wrist", "state"} and v["state"].shape == (6,)
    np.testing.assert_array_equal(v["wrist"], obs["observation.images.right_wrist"])
    t = time.time()
    for _ in range(10):
        env.step(obs["observation.state"])
    per_step = (time.time() - t) / 10
    env.close()
    print(f"step with 3 cameras + depth: {per_step * 1000:.1f} ms")
    assert per_step < 0.5  # loose bound for CPU/osmesa machines; ~20 ms with EGL on this laptop
