"""The GPU scripts' code paths on LeRobot's real ``flux3`` policy code, with a tiny random DiT on CPU.

Skipped unless ``lerobot.policies.flux3`` imports (LeRobot main). Run with the flux agent's CPU venv:

    ws4/.cache/flux/venv-lerobot-cpu/bin/python -m pytest ws4/ttt/tests/test_lerobot_integration.py -q

The fake encoders and the shrunk SO-101 config mirror ``ws4/flux/tests/test_lerobot_flux3.py``.
"""

import pytest

flux3 = pytest.importorskip("lerobot.policies.flux3")
torch = pytest.importorskip("torch")

from flux.backends import LeRobotFlux3Backend  # noqa: E402
from flux.contract import OBS_STATE, SO101_SCENE, SO101_WRIST  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.flux3 import Flux3Config, Flux3Policy, make_flux3_pre_post_processors  # noqa: E402
from torch import nn  # noqa: E402
from ttt.flux_hook import FluxFeatureTap  # noqa: E402
from ttt.flux_scale_check import run_scale_check  # noqa: E402

TINY_DIT = {"hidden_size": 64, "num_heads": 2, "depth": 1, "depth_single_blocks": 3, "axes_dim": [8, 8, 8, 8],
            "context_in_dim": 32}  # fmt: skip
HW = 64


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


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, config: FakeVideoVAE())
    monkeypatch.setattr(Flux3Policy, "_build_text_encoder", lambda self, config: MockTextEncoder())
    cfg = Flux3Config(
        input_features={
            SO101_SCENE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, HW, HW)),
            SO101_WRIST: PolicyFeature(type=FeatureType.VISUAL, shape=(3, HW, HW)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        camera_layout="side_by_side",
        camera_keys=[SO101_SCENE, SO101_WRIST],
        canvas_hw=(HW, 2 * HW),
        n_obs_steps=4,
        history_snapshots=2,
        condition_on_past_actions=True,
        chunk_size=20,
        n_action_steps=8,
        delta_absolute_dims=[-1],
        action_channel_weights=[1, 1, 1, 1, 1, 2],
        sampler="euler",
        guidance_scale=3.0,
        guidance_scale_action=None,
        dit_config=TINY_DIT,
        dtype=torch.float32,
        video_vae_id=None,
        device="cpu",
        normalization_stats={
            "state": {"q01": [-90.0] * 5 + [0.0], "q99": [90.0] * 5 + [100.0]},
            "action": {"q01": [-2.0] * 5 + [0.0], "q99": [2.0] * 5 + [100.0]},
        },
    )
    torch.manual_seed(0)
    policy = Flux3Policy(cfg)
    with torch.no_grad():
        for p in policy.dit.final_layer.parameters():
            p.normal_(0, 0.05)
    pre, post = make_flux3_pre_post_processors(cfg)
    return LeRobotFlux3Backend(policy, pre, post, name="tiny")


def _request(backend):
    from flux.measure_latency import synthetic_request

    return synthetic_request(backend.contract, "Close the box lid")


def test_feature_tap_on_lerobot_policy_gives_one_vector_per_replan(backend):
    dit, m = backend.policy.dit, backend.policy.modality
    taps = [FluxFeatureTap(2, ("video", m)).attach(dit), FluxFeatureTap(1, ("video_cond", f"{m}_cond")).attach(dit)]
    backend.predict(_request(backend))
    feats = [t.pop() for t in taps]
    assert all(f.shape == (1, 1, 64) for f in feats)
    assert all(torch.isfinite(f).all() for f in feats)


def test_scale_check_gate0_is_identical_on_lerobot_policy(backend):
    report = run_scale_check(backend, _request(backend), layers=[0, 2], inner_dim=16, runs=2)
    assert report["repeatability_max_abs"] == 0.0
    assert report["gate0_bit_identical"]
    assert 0 < report["gate1e-3_max_abs_vs_plain"]
    assert report["adapter_params"] > 0
