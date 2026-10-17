"""DualArmRunner with fake policies and a fake robot (numpy only, no torch)."""

import threading
import time

import numpy as np
import pytest
from flux.backends import FakeBackend, PolicyRequest
from flux.contract import (
    BIMANUAL_GRID_CONTRACT,
    IMG_HEAD,
    IMG_LEFT_WRIST,
    IMG_RIGHT_WRIST,
    OBS_STATE,
    SO101_CONTRACT,
    SO101_SCENE,
    SO101_WRIST,
    ModelContract,
)
from flux.dual_arm_runner import (
    ArmCalibration,
    DualArmRunner,
    GymEnvRobot,
    bimanual_target,
    per_arm_target,
)
from flux.ee50_hook import EE50Adapter
from flux.remote import RemoteBackend, make_server
from flux.safety import SafetyConfig, SafetyGuard

SMALL = ModelContract(
    name="small", action_dim=6, camera_keys=(SO101_SCENE, SO101_WRIST), n_obs_steps=4, chunk_size=6,
    n_action_steps=4, image_hw=(16, 16),
)  # fmt: skip
WIDE = SafetyConfig(lower=-np.full(12, 1e6), upper=np.full(12, 1e6), max_step=np.full(12, 1e6))


class FakeRobot:
    """State follows commands exactly; each camera is a constant image so mappings are checkable."""

    def __init__(self, state=None, follow=True):
        self.state = np.zeros(12) if state is None else np.asarray(state, float)
        self.follow = follow
        self.sent = []
        self.images = {IMG_HEAD: 10, IMG_LEFT_WRIST: 20, IMG_RIGHT_WRIST: 30}

    def get_observation(self):
        obs = {OBS_STATE: self.state.copy()}
        obs.update({k: np.full((48, 64, 3), v, np.uint8) for k, v in self.images.items()})
        return obs

    def send_action(self, action):
        self.sent.append(np.array(action))
        if self.follow:
            self.state = np.array(action, float)


def make_runner(backends, targets=None, robot=None, guard=None, async_inference=False):
    robot = robot or FakeRobot()
    targets = targets or [per_arm_target("left", "m"), per_arm_target("right", "m")]
    runner = DualArmRunner(robot, targets, backends, guard or SafetyGuard(WIDE), async_inference=async_inference)
    return runner, robot


def test_one_arm_moves_other_holds_and_replans_after_n_action_steps():
    backend = FakeBackend(SMALL, velocity=1.0)
    runner, robot = make_runner({"m": backend})
    runner.start("left", "push the box")
    r = runner.step()
    assert r.status == {"left": "replanned"}
    assert np.allclose(robot.sent[-1][:5], 1.0) and np.allclose(robot.sent[-1][6:], 0.0)
    for _ in range(SMALL.n_action_steps - 1):
        runner.step()
    assert len(backend.requests) == 1
    runner.step()  # queue empty -> second chunk anchored on the last issued command (4.0)
    assert len(backend.requests) == 2
    req = backend.requests[-1]
    assert np.allclose(req.commands[-1][:5], 4.0) and np.allclose(req.states[-1][:5], 4.0)
    assert np.allclose(robot.sent[-1][:5], 5.0)
    assert req.task == "push the box" and req.target == "left"


def test_cameras_are_mapped_per_arm_on_a_shared_model():
    backend = FakeBackend(SMALL)
    runner, _ = make_runner({"m": backend})
    runner.start("left", "a")
    runner.start("right", "b")
    runner.step()
    by_target = {r.target: r for r in backend.requests}
    assert by_target["left"].images[SO101_SCENE].max() == 10
    assert by_target["left"].images[SO101_WRIST].max() == 20
    assert by_target["right"].images[SO101_WRIST].max() == 30
    assert by_target["right"].task == "b"
    assert by_target["left"].images[SO101_SCENE].shape == (2, 16, 16, 3)  # only the encoded snapshots travel


def test_async_other_arm_keeps_moving_while_one_waits_and_history_freezes():
    slow = FakeBackend(SMALL, velocity=1.0, delay_s=0.3)
    fast = FakeBackend(SMALL, velocity=-1.0)
    targets = [per_arm_target("left", "slow"), per_arm_target("right", "fast")]
    runner, robot = make_runner({"slow": slow, "fast": fast}, targets, async_inference=True)
    runner.start("left", "a")
    runner.start("right", "b")
    statuses = []
    for _ in range(4):
        statuses.append(runner.step().status)
        time.sleep(0.01)
    assert statuses[0] == {"left": "waiting", "right": "waiting"}
    assert any(s["right"] == "executing" for s in statuses[1:]) and all(s["left"] == "waiting" for s in statuses)
    assert np.allclose(robot.state[:5], 0.0) and robot.state[6] < 0
    deadline = time.time() + 2
    while runner.step().status.get("left") != "replanned":
        assert time.time() < deadline
    assert np.allclose(robot.state[:5], 1.0)
    assert len(slow.requests) == 1  # no new left request while waiting
    runner.close()


def test_arm_ownership_conflict():
    runner, _ = make_runner(
        {"m": FakeBackend(SMALL), "bi": FakeBackend(BIMANUAL_GRID_CONTRACT)},
        [per_arm_target("left", "m"), bimanual_target("bi")],
    )
    runner.start("left", "a")
    with pytest.raises(RuntimeError, match="owned"):
        runner.start("both", "b")
    runner.stop("left")
    runner.start("both", "b")


def test_bimanual_target_drives_both_arms_and_checks_dims():
    contract = ModelContract(
        name="bi", action_dim=12, camera_keys=(IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST), n_obs_steps=3,
        chunk_size=5, n_action_steps=2, image_hw=(12, 16), absolute_dims=(5, 11),
    )  # fmt: skip
    backend = FakeBackend(contract, velocity=0.5, absolute=np.full(12, 40.0))
    runner, robot = make_runner({"bi": backend}, [bimanual_target("bi")])
    runner.start("both", "lift the box together")
    runner.step()
    assert np.allclose(robot.sent[-1][[0, 6]], 0.5) and np.allclose(robot.sent[-1][[5, 11]], 40.0)
    assert backend.requests[0].images[IMG_RIGHT_WRIST].shape == (2, 12, 16, 3)
    with pytest.raises(ValueError, match="outputs 12-D"):
        make_runner({"bi": backend}, [per_arm_target("left", "bi")])


def test_guard_intervention_is_what_history_records():
    backend = FakeBackend(SMALL, velocity=10.0)
    guard = SafetyGuard(SafetyConfig(lower=-np.full(12, 100), upper=np.full(12, 100), max_step=np.full(12, 2.0)))
    runner, robot = make_runner({"m": backend}, guard=guard)
    runner.start("left", "a")
    for _ in range(SMALL.n_action_steps + 1):
        report = runner.step()
    assert report.guard_intervened
    assert np.allclose(backend.requests[-1].commands[-1][:5], 2.0 * SMALL.n_action_steps)


def test_mirrored_arm_flips_pan_and_roll_and_images():
    backend = FakeBackend(SMALL, velocity=np.array([1.0, 0, 0, 0, 1.0, 0]))
    robot = FakeRobot(state=np.r_[np.array([5.0, 1, 2, 3, 7, 50]), np.zeros(6)])
    runner, robot = make_runner({"m": backend}, [per_arm_target("left", "m", mirror=True)], robot=robot)
    runner.start("left", "a")
    runner.step()
    assert np.allclose(backend.requests[0].states[-1], [-5.0, 1, 2, 3, -7, 50])
    assert np.allclose(robot.sent[-1][:6], [4.0, 1, 2, 3, 6, 50])  # +1 in model units is -1 on the robot
    cal = ArmCalibration(scale=np.full(6, 2.0), offset=np.ones(6))
    assert np.allclose(cal.to_robot(cal.to_model(np.arange(6.0))), np.arange(6.0))


def test_estop_freezes_and_release_restarts_episode():
    backend = FakeBackend(SMALL, velocity=1.0)
    runner, robot = make_runner({"m": backend})
    runner.start("left", "a")
    runner.step()
    runner.guard.estop()
    n_sent = len(robot.sent)
    assert runner.step().status == {"left": "estopped"} and len(robot.sent) == n_sent
    runner.release_estop()
    runner.step()
    assert len(backend.requests) == 2 and len(robot.sent) == n_sent + 1


def test_gym_env_robot_adapter():
    class Env:
        def reset(self, seed=None):
            return {OBS_STATE: np.zeros(12)}, {}

        def step(self, a):
            return {OBS_STATE: np.asarray(a, float)}, 0.0, False, False, {}

    robot = GymEnvRobot(Env())
    robot.send_action(np.ones(12))
    assert np.allclose(robot.get_observation()[OBS_STATE], 1.0)


def test_remote_backend_roundtrip():
    local = FakeBackend(SO101_CONTRACT, velocity=0.25)
    server = make_server({"so101": local}, port=0, token="s3cret")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        remote = RemoteBackend(url, "so101", token="s3cret")
        assert remote.contract == SO101_CONTRACT
        n = SO101_CONTRACT.n_obs_steps
        images = {k: np.full((2, 256, 256, 3), 7, np.uint8) for k in SO101_CONTRACT.camera_keys}
        req = PolicyRequest("pick", np.ones((n, 6)), np.ones((n, 6)) * 2, images, SO101_CONTRACT.snapshots, "left")
        out = remote.predict(req)
        assert out.commands.shape == (42, 6) and np.allclose(out.commands[0], 2.25)
        assert local.requests[-1].task == "pick" and local.requests[-1].images[SO101_SCENE].max() == 7
        assert out.info["request_bytes"] < 60_000  # four JPEG snapshots, not 786 KB of raw pixels
        raw = RemoteBackend(url, "so101", token="s3cret", codec="raw").predict(req)
        assert raw.info["request_bytes"] > 700_000 and np.allclose(raw.commands, out.commands)
        with pytest.raises(RuntimeError, match="401"):
            RemoteBackend(url, "so101", token="wrong")
        bad = PolicyRequest("pick", np.ones((3, 6)), np.ones((3, 6)), images, SO101_CONTRACT.snapshots)
        with pytest.raises(RuntimeError, match="400"):
            remote.predict(bad)
    finally:
        server.shutdown()


def test_ee50_adapter_hook_with_toy_kinematics():
    # Toy "kinematics": the pose translation is the first three joints / 100; IK inverts it.
    def fk(arm, q):
        T = np.eye(4)
        T[:3, 3] = np.asarray(q[:3]) / 100.0
        return T

    def ik(arm, T, seed):
        q = seed.copy()
        q[:3] = T[:3, 3] * 100.0
        return q

    def encode(poses, grippers, anchor):
        out = np.zeros((1, 50))
        for i, arm in enumerate(("left", "right")):
            if arm in poses:
                out[0, i * 25 : i * 25 + 3] = poses[arm][0][:3, 3]
                out[0, i * 25 + 9] = grippers[arm][0]
        return out

    def decode(x, anchor):
        poses, grippers = {}, {}
        for i, arm in enumerate(anchor):
            T = np.tile(np.eye(4), (len(x), 1, 1))
            T[:, :3, 3] = x[:, i * 25 : i * 25 + 3]
            poses[arm], grippers[arm] = T, x[:, i * 25 + 9]
        return poses, grippers

    adapter = EE50Adapter(arms=("left", "right"), fk=fk, ik=ik, encode=encode, decode=decode)
    joints = np.r_[np.array([10.0, 20, 30, 0, 0, 80]), np.array([1.0, 2, 3, 0, 0, 20])]
    vec = adapter.joints_to_model(joints)
    assert vec.shape == (50,) and np.isclose(vec[1], 0.2) and np.isclose(vec[25 + 9], 0.2)
    back = adapter.model_to_joints(np.stack([vec, vec]), joints)
    assert back.shape == (2, 12) and np.allclose(back[0], joints)


def test_lerobot_bimanual_robot_adapter_maps_bi_so_follower_keys():
    from flux.run_xlerobot import STATE_KEYS, LeRobotBimanualRobot

    class FakeLeRobot:
        def __init__(self):
            self.sent = None

        def get_observation(self):
            obs = {k: float(i) for i, k in enumerate(STATE_KEYS)}
            obs.update(
                {
                    name: np.full((4, 4, 3), v, np.uint8)
                    for name, v in (("head", 1), ("left_wrist", 2), ("right_wrist", 3))
                }
            )
            return obs

        def send_action(self, action):
            self.sent = action

    fake = FakeLeRobot()
    robot = LeRobotBimanualRobot(fake)
    obs = robot.get_observation()
    assert np.array_equal(obs[OBS_STATE], np.arange(12.0))
    assert STATE_KEYS[0] == "left_shoulder_pan.pos" and STATE_KEYS[11] == "right_gripper.pos"
    assert obs[IMG_LEFT_WRIST].max() == 2 and obs[IMG_HEAD].max() == 1
    robot.send_action(np.arange(12.0) * 2)
    assert fake.sent["right_shoulder_pan.pos"] == 12.0
