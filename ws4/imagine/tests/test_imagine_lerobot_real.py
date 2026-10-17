"""Integration against the real LeRobot ``Flux3Policy`` (tiny random DiT, CPU, fake encoders).

Skipped unless ``lerobot.policies.flux3`` is importable (LeRobot main, not the 0.5.1 release on this laptop).
Run it with a LeRobot-main environment, see GPU_RUNBOOK.md step 1 or:
``uv pip install "lerobot @ git+https://github.com/huggingface/lerobot@146d847"``.
The config mirrors the released SO-101 package schema (flux-action tests/fixtures/so101_history/config.json):
history conditioning, 8 observations, 2 snapshots, past commands, side_by_side, predict 42 / execute 32.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot.policies.flux3")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.flux3 import Flux3Config, Flux3Policy  # noqa: E402

from ws4.imagine.fakes import FakeVideoVAE  # noqa: E402
from ws4.imagine.predict_with_video import Imaginer, VideoSpec, alternatives_batch  # noqa: E402

TINY_DIT = {
    "hidden_size": 64,
    "num_heads": 2,
    "depth": 1,
    "depth_single_blocks": 1,
    "axes_dim": [8, 8, 8, 8],
    "context_in_dim": 32,
}
SCENE, WRIST = "observation.images.scene", "observation.images.wrist"


class MockTextEncoder(torch.nn.Module):
    def forward_bucketed(self, text, *, fixed_length=None):
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return torch.randn(1, fixed_length or 80, 32, generator=gen).to(torch.bfloat16)


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(Flux3Policy, "_build_text_encoder", lambda self, config: MockTextEncoder())
    monkeypatch.setattr(Flux3Policy, "_build_video_vae", lambda self, config: FakeVideoVAE())
    cfg = Flux3Config(
        input_features={
            SCENE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            WRIST: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        camera_layout="side_by_side",
        camera_keys=[SCENE, WRIST],
        canvas_hw=(64, 128),
        dit_config=TINY_DIT,
        dtype=torch.float32,
        video_vae_id=None,
        device="cpu",
        n_obs_steps=8,
        history_snapshots=2,
        condition_on_past_actions=True,
        chunk_size=42,
        n_action_steps=32,
        delta_absolute_dims=[-1],
        sampler="euler",
        num_inference_steps=4,
        guidance_scale=3.0,
        guidance_scale_action=None,
        sampler_shift=6.93,
        inference_seed=42,
        normalization_stats={name: {"q01": [-2.0] * 6, "q99": [2.0] * 6} for name in ("action", "state")},
    )
    return Flux3Policy(cfg).eval()


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(0)
    return {
        SCENE: torch.rand(1, 8, 3, 64, 64, generator=g),
        WRIST: torch.rand(1, 8, 3, 64, 64, generator=g),
        "observation.state": torch.randn(1, 8, 6, generator=g),
        "observation.past_actions": torch.randn(1, 8, 6, generator=g),
        "task": "pick up the cube",
    }


def test_spec_matches_real_config(policy):
    spec = VideoSpec.from_config(policy.config)
    assert spec.conditioning == "history" and spec.latent_hw == (2, 4)
    assert spec.n_pred_latents == policy.packer.predicted_latent_frames(policy.config) == 11
    assert spec.frame_offsets == tuple(range(2, 43))


def test_imagine_returns_stock_actions_and_video(policy, batch):
    stock = policy.predict_action_chunk(batch)
    imaginer = Imaginer(policy)
    (rec,) = imaginer.imagine(batch)
    assert torch.equal(rec.action, stock[0])
    assert tuple(rec.latents.shape) == (96, 11, 2, 4)
    assert tuple(rec.cond_latents.shape) == (2, 96, 1, 2, 4)  # two history snapshots
    assert rec.frames.shape == (41, 64, 128, 3)
    assert rec.cond_frames.shape == (2, 64, 128, 3)
    assert "_sample" not in policy.__dict__ and "predict_action_chunk" not in policy.__dict__


def test_installed_select_action_records_every_chunk(policy, batch):
    imaginer = Imaginer(policy).install()
    policy.reset()
    for _ in range(33):  # 32 executed steps, then a second chunk
        policy.select_action(batch)
    assert [r.step for r in imaginer.history] == [0, 32]
    imaginer.uninstall()


def test_alternative_instructions_match_single_runs(policy, batch):
    recs = Imaginer(policy).imagine(alternatives_batch(batch, ["push the box left", "lift the box"]))
    assert [r.task for r in recs] == ["push the box left", "lift the box"]
    single = dict(batch, task="lift the box")
    assert torch.equal(recs[1].action, policy.predict_action_chunk(single)[0])
    assert not torch.equal(recs[0].latents, recs[1].latents)  # the caption changes the imagined future


def test_best_of_n_executes_the_chosen_seed(policy, batch):
    imaginer = Imaginer(policy, n_candidates=3, scorer=lambda clips, **kw: [0.0, 5.0, 1.0])
    (rec,) = imaginer.imagine(batch)
    assert [c.seed for c in rec.candidates] == [42, 43, 44]
    assert rec.chosen == 1 and rec.seed == 43
    policy.config.inference_seed = 43
    assert torch.equal(rec.action, policy.predict_action_chunk(batch)[0])
