"""Integration with LeRobot's real ``flux3`` code on CPU, using a tiny random DiT and fake frozen encoders.

Skipped unless ``lerobot.policies.flux3`` imports (LeRobot main, not 0.5.1). Run with the CPU venv from
GPU_RUNBOOK.md ("CPU checks"). The fakes mirror lerobot@146d847:tests/policies/flux3/helpers.py.
"""

import numpy as np
import pytest

flux3 = pytest.importorskip("lerobot.policies.flux3")
torch = pytest.importorskip("torch")

from flux.backends import LeRobotFlux3Backend  # noqa: E402
from flux.contract import IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST, OBS_STATE, SO101_SCENE, SO101_WRIST  # noqa: E402
from flux.dual_arm_runner import DualArmRunner, bimanual_target, per_arm_target  # noqa: E402
from flux.make_bimanual_base import build_bimanual_policy, save_base  # noqa: E402
from flux.package_tools import read_history_quantiles, renormalize_package  # noqa: E402
from flux.safety import SafetyConfig, SafetyGuard  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.flux3 import Flux3Config, Flux3Policy, make_flux3_pre_post_processors  # noqa: E402
from torch import nn  # noqa: E402

TINY_DIT = {"hidden_size": 64, "num_heads": 2, "depth": 1, "depth_single_blocks": 1, "axes_dim": [8, 8, 8, 8],
            "context_in_dim": 32}  # fmt: skip
HW = 64
WIDE = SafetyConfig(lower=-np.full(12, 1e6), upper=np.full(12, 1e6), max_step=np.full(12, 1e6))


class FakeVideoVAE:
    def __init__(self):
        self.proj = torch.randn(96, 3, generator=torch.Generator().manual_seed(0)) * 0.1
        self.module = nn.Identity()

    def encode(self, video):
        b, c, t, h, w = video.shape
        x = video.float()[:, :, ::4]
        tp = x.shape[2]
        x = torch.nn.functional.avg_pool2d(x.transpose(1, 2).reshape(-1, c, h, w), 32)
        x = torch.einsum("nchw,oc->nohw", x, self.proj)
        return x.reshape(b, tp, 96, h // 32, w // 32).transpose(1, 2).to(torch.bfloat16)

    def encode_task(self, video):
        return self.encode(video[:, :, : 1 + 4 * ((video.shape[2] - 1) // 4)])


class MockTextEncoder(nn.Module):
    def forward_bucketed(self, text, *, fixed_length=None):
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % (2**31)
        return torch.randn(1, fixed_length or 64, 32, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


@pytest.fixture(autouse=True)
def fake_encoders(monkeypatch):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, config: FakeVideoVAE())
    monkeypatch.setattr(Flux3Policy, "_build_text_encoder", lambda self, config: MockTextEncoder())


def so101_like_config(**overrides):
    """The SO-101 package's history contract, shrunk: 4 ticks, 2 snapshots, past commands, 20 / 8 actions."""
    feats = {SO101_SCENE: (3, HW, HW), SO101_WRIST: (3, HW, HW)}
    kwargs = {
        "input_features": {
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=s) for k, s in feats.items()},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        "output_features": {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        "camera_layout": "side_by_side",
        "camera_keys": [SO101_SCENE, SO101_WRIST],
        "canvas_hw": (HW, 2 * HW),
        "n_obs_steps": 4,
        "history_snapshots": 2,
        "condition_on_past_actions": True,
        "chunk_size": 20,
        "n_action_steps": 8,
        "delta_absolute_dims": [-1],
        "action_channel_weights": [1, 1, 1, 1, 1, 2],
        "sampler": "euler",
        "guidance_scale": 3.0,
        "guidance_scale_action": None,
        "dit_config": TINY_DIT,
        "dtype": torch.float32,
        "video_vae_id": None,
        "device": "cpu",
        "normalization_stats": {
            "state": {"q01": [-90.0] * 5 + [0.0], "q99": [90.0] * 5 + [100.0]},
            "action": {"q01": [-2.0] * 5 + [0.0], "q99": [2.0] * 5 + [100.0]},
        },
    }
    kwargs.update(overrides)
    return Flux3Config(**kwargs)


@pytest.fixture
def tiny_package(tmp_path):
    torch.manual_seed(0)
    cfg = so101_like_config()
    policy = Flux3Policy(cfg)
    with torch.no_grad():  # zero-initialized output heads would predict exactly zero; make outputs informative
        for p in policy.dit.final_layer.parameters():
            p.normal_(0, 0.05)
    pre, post = make_flux3_pre_post_processors(cfg)
    path = save_base(policy, pre, post, tmp_path / "so101_tiny")
    return path


def load_backend(path):
    policy = Flux3Policy.from_pretrained(str(path))
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(path))
    return LeRobotFlux3Backend(policy, pre, post, name="tiny"), policy, pre, post


class ReplayRobot:
    """Plays a fixed observation sequence regardless of commands (so both loops see identical inputs)."""

    def __init__(self, n_ticks, seed=0):
        rng = np.random.default_rng(seed)
        self.states = np.cumsum(rng.normal(0, 1.0, (n_ticks, 12)), axis=0)
        self.states[:, [5, 11]] = 50.0
        self.images = rng.integers(0, 255, (n_ticks, 3, HW, HW, 3), dtype=np.uint8)
        self.t = 0
        self.sent = []

    def get_observation(self):
        imgs = self.images[self.t]
        return {OBS_STATE: self.states[self.t], IMG_HEAD: imgs[0], IMG_LEFT_WRIST: imgs[1], IMG_RIGHT_WRIST: imgs[2]}

    def send_action(self, action):
        self.sent.append(np.array(action))
        self.t += 1


def test_runner_matches_lerobot_synchronous_loop(tiny_package):
    """Our stateless, history-on-the-robot path gives the commands of LeRobot's documented sync loop."""
    n_ticks = 18  # three chunks of 8: replans at ticks 0, 8 and 16
    backend, _, _, _ = load_backend(tiny_package)
    robot = ReplayRobot(n_ticks)
    runner = DualArmRunner(robot, [per_arm_target("left", "tiny")], {"tiny": backend}, SafetyGuard(WIDE))
    runner.start("left", "push the green box")
    for _ in range(n_ticks):
        runner.step()
    ours = np.stack(robot.sent)[:, :6]

    # Reference: lerobot docs flux3.mdx:206-218, pre -> select_action -> post on every tick.
    _, policy, pre, post = load_backend(tiny_package)
    ref_robot = ReplayRobot(n_ticks)
    reference = []
    for _ in range(n_ticks):
        obs = ref_robot.get_observation()
        batch = {
            OBS_STATE: torch.as_tensor(obs[OBS_STATE][:6], dtype=torch.float32),
            SO101_SCENE: torch.from_numpy(obs[IMG_HEAD]).permute(2, 0, 1)[None],
            SO101_WRIST: torch.from_numpy(obs[IMG_LEFT_WRIST]).permute(2, 0, 1)[None],
            "task": ["push the green box"],
        }
        batch[OBS_STATE] = batch[OBS_STATE][None]
        action = post(policy.select_action(pre(batch)))
        reference.append(action[0].double().numpy())
        ref_robot.send_action(np.zeros(12))
    reference = np.stack(reference)
    assert np.isfinite(reference).all() and np.abs(np.diff(reference[:, 0])).max() > 1e-4
    np.testing.assert_allclose(ours, reference, rtol=1e-4, atol=1e-3)


def test_one_model_serves_both_arms_with_separate_histories(tiny_package):
    backend, _, _, _ = load_backend(tiny_package)
    robot = ReplayRobot(3, seed=1)
    runner = DualArmRunner(
        robot, [per_arm_target("left", "tiny"), per_arm_target("right", "tiny")], {"tiny": backend}, SafetyGuard(WIDE)
    )
    runner.start("left", "push the box aside")
    runner.start("right", "pick up the cup")
    for _ in range(3):
        runner.step()
    sent = np.stack(robot.sent)
    assert np.isfinite(sent).all()
    assert not np.allclose(sent[:, :6], sent[:, 6:])  # different arms, states, cameras and prompts


def test_renormalized_package_keeps_weights_and_swaps_quantiles(tiny_package, tmp_path):
    stats = {
        "state": {"q01": [-50.0] * 5 + [5.0], "q99": [50.0] * 5 + [95.0]},
        "action": {"q01": [-1.0] * 5 + [5.0], "q99": [1.0] * 5 + [95.0]},
    }
    out = renormalize_package(tiny_package, tmp_path / "left_arm", stats, device="cpu")
    assert (out / "model.safetensors").is_symlink()
    backend, policy, pre, _ = load_backend(out)
    got = read_history_quantiles(pre)
    assert np.allclose(got["state"]["q99"], stats["state"]["q99"])
    assert np.allclose(got["action"]["q01"], stats["action"]["q01"])
    original = Flux3Policy.from_pretrained(str(tiny_package))
    a, b = original.state_dict(), policy.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_bimanual_base_from_so101_package(tiny_package, tmp_path):
    feats_in = {
        IMG_HEAD: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        IMG_LEFT_WRIST: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        IMG_RIGHT_WRIST: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 48, 64)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(12,)),
    }
    feats_out = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(12,))}
    stats = {
        "state": {"q01": ([-90.0] * 5 + [0.0]) * 2, "q99": ([90.0] * 5 + [100.0]) * 2},
        "action": {"q01": ([-2.0] * 5 + [0.0]) * 2, "q99": ([2.0] * 5 + [100.0]) * 2},
    }
    policy, pre, post = build_bimanual_policy(
        str(tiny_package), feats_in, feats_out, stats, source_kind="so101", canvas_hw=(64, 64)
    )
    sd = policy.dit.state_dict()
    assert sd["emb_in.action.weight"].shape[1] == 12 and sd["emb_in.action_cond.weight"].shape[1] == 24
    assert torch.count_nonzero(sd["final_layer.action.linear.weight"]) == 0  # fresh head predicts zero first
    source = Flux3Policy.from_pretrained(str(tiny_package)).dit.state_dict()
    trunk = [k for k in sd if not k.startswith(("emb_in.action", "final_layer.action"))]
    assert trunk and all(torch.equal(sd[k], source[k]) for k in trunk)  # SO-101 trunk and action branch kept
    assert policy.config.delta_absolute_dims == [5, 11] and policy.config.camera_layout == "grid"

    out = save_base(policy, pre, post, tmp_path / "bimanual_base")
    backend, _, _, _ = load_backend(out)
    assert backend.contract.action_dim == 12 and backend.contract.camera_keys == (
        IMG_HEAD,
        IMG_LEFT_WRIST,
        IMG_RIGHT_WRIST,
    )
    robot = ReplayRobot(3, seed=2)
    runner = DualArmRunner(robot, [bimanual_target("bi")], {"bi": backend}, SafetyGuard(WIDE))
    runner.start("both", "move the box away and grab the cup")
    for _ in range(3):
        runner.step()
    assert np.isfinite(np.stack(robot.sent)).all()
