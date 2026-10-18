import pytest

torch = pytest.importorskip("torch")

from ws4.imagine.fakes import FakeVideoVAE  # noqa: E402
from ws4.imagine.predict_with_video import (  # noqa: E402
    VideoSpec,
    condition_latents,
    decoded_frame_sources,
    decoded_slots,
    encoder_slots,
    latent_frames,
    tokens_to_latents,
)


@pytest.mark.parametrize("n_frames", range(1, 100))
def test_encoder_slot_count_matches_latent_formula(n_frames):
    slots = encoder_slots(n_frames)
    assert len(slots) == latent_frames(n_frames)
    assert all(len(s) == 4 for s in slots)


@pytest.mark.parametrize("n_latents", range(1, 20))
def test_decoder_yields_4n_minus_3_frames(n_latents):
    assert len(decoded_slots(n_latents)) == 4 * n_latents - 3


def test_causal_clip_decodes_to_itself():
    # 4k+1 clips (the VAE's native grid) map decoded frame i to clip frame i.
    for n in (1, 5, 33, 45):
        assert [c for _, c in decoded_frame_sources(n)] == list(range(n))


def test_so101_offsets():
    spec = VideoSpec("history", 42, 32, 30.0, (8, 16), (256, 512), "side_by_side")
    assert spec.n_pred_latents == 11
    assert spec.frame_offsets == tuple(range(2, 43))
    assert spec.decoded_hw == (256, 512)
    assert spec.seconds(42) == pytest.approx(1.4)


def test_droid_frame_offsets():
    spec = VideoSpec("frame", 32, 32, 15.0, (17, 20), (544, 736), "droid")
    assert spec.n_pred_latents == 8
    assert spec.decode_with_condition
    assert spec.frame_offsets == tuple(range(1, 33))


def test_lerobot_default_history_chunk_32():
    spec = VideoSpec("history", 32, 32, 30.0, (8, 16), (256, 512), "side_by_side")
    assert spec.frame_offsets == tuple(range(4, 33))


@pytest.mark.parametrize("n_frames", [1, 32, 33, 42])
def test_fake_vae_round_trip_follows_slot_mapping(n_frames):
    video = torch.linspace(-0.9, 0.9, n_frames).view(1, 1, n_frames, 1, 1).expand(1, 3, n_frames, 32, 64)
    decoded = FakeVideoVAE().decode(FakeVideoVAE().encode(video))
    expected = torch.stack([video[:, :, c] for _, c in decoded_frame_sources(n_frames)], dim=2)
    assert torch.allclose(decoded.float(), expected.float(), atol=1e-2)


def test_tokens_to_latents_inverts_t_h_w_packing():
    lat = torch.randn(1, 96, 3, 2, 4)
    tokens = lat.permute(0, 2, 3, 4, 1).reshape(1, 3 * 2 * 4, 96)  # prc_vid: "c t h w -> (t h w) c"
    assert torch.equal(tokens_to_latents(tokens, 3, (2, 4)), lat)
    with pytest.raises(ValueError):
        tokens_to_latents(tokens, 4, (2, 4))


def test_condition_latents_splits_snapshots():
    lat = torch.randn(1, 96, 2, 2, 4)
    tokens = lat.permute(0, 2, 3, 4, 1).reshape(1, 16, 96)
    cond = condition_latents(tokens, (2, 4))
    assert cond.shape == (2, 96, 1, 2, 4)
    assert torch.equal(cond[1, :, 0], lat[0, :, 1])
