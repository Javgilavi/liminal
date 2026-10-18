"""The adapter against flux-action's own ``JointSingleSeq``, built tiny on CPU (no weights, no GPU).

Skipped when the clone is not in ``ws4/.cache/flux-action`` (it is gitignored).
"""

import sys
from pathlib import Path

import pytest
import torch
from ttt.flux_hook import FluxFeatureTap, FluxTTTAdapter, stream_layout

FLUX_SRC = Path(__file__).resolve().parents[2] / ".cache" / "flux-action" / "src"
STREAMS = ("action", "action_cond")
LENS = {"ctx": 5, "x_video": 6, "x_video_cond": 4, "x_action": 3, "x_action_cond": 2}
CHANNELS = {"x_video": 4, "x_video_cond": 4, "x_action": 3, "x_action_cond": 6}


@pytest.fixture(scope="module")
def transformer():
    if not (FLUX_SRC / "flux_action" / "models" / "transformer.py").exists():
        pytest.skip("flux-action clone not found in ws4/.cache")
    pytest.importorskip("einops")
    sys.path.insert(0, str(FLUX_SRC))
    from flux_action.models import transformer

    return transformer


def tiny_dit(transformer):
    params = transformer.JointSingleSeqParams(
        in_channels={"video": 4, "video_cond": 4, "action": 3, "action_cond": 6},
        sequence={
            "x_video": "video",
            "x_video_cond": "video_cond",
            "x_action": "action",
            "x_action_cond": "action_cond",
        },
        vec_in_dim=8,
        context_in_dim=16,
        hidden_size=32,
        num_heads=2,
        depth=1,
        depth_single_blocks=3,
        axes_dim=[4, 4, 4, 4],
        mlp_ratio=2.0,
        attn_mode="torch",
    )
    torch.manual_seed(0)
    dit = transformer.JointSingleSeq(params).eval()
    for p in dit.parameters():
        p.requires_grad_(False)
    return dit


def window(seed, t_noise=0.5):
    """One window's keyword arguments (batch 1), the way policy._sample calls the DiT."""
    g = torch.Generator().manual_seed(seed)

    def ids(n, t0=0):
        return torch.stack([torch.arange(n) + t0, *(torch.zeros(n, dtype=torch.long),) * 3], -1)[None]

    kw = {
        "ctx": torch.randn(1, LENS["ctx"], 16, generator=g),
        "ctx_ids": ids(LENS["ctx"]),
        "timesteps_ctx": torch.zeros(1, LENS["ctx"]),
    }
    for key, ch in CHANNELS.items():
        n = LENS[key]
        kw[key] = torch.randn(1, n, ch, generator=g)
        kw[f"{key}_ids"] = ids(n, 3)
        cond = key.endswith("_cond")
        kw[f"{key}_timesteps"] = torch.full((1, n), 0.0 if cond else t_noise)
    return kw


VECTOR = torch.randn(1, 8, generator=torch.Generator().manual_seed(99))


def call(dit, kw):
    return dit(vector=VECTOR, **kw)


def pack(windows):
    """Concatenate windows the way packing.pack_windows does (one sequence, per-window token counts)."""
    packed = {k: torch.cat([w[k] for w in windows], 1) for k in windows[0]}
    seqlens = {"ctx": [LENS["ctx"]] * len(windows), **{k: [LENS[k]] * len(windows) for k in CHANNELS}}
    return packed, seqlens


def test_layout_selects_only_action_tokens(transformer):
    dit = tiny_dit(transformer)
    lay = stream_layout(dit.sequence, window(0), STREAMS)
    # content order video(6), video_cond(4), action(3), action_cond(2): action tokens sit at 10..14
    assert lay.index.tolist() == [[10, 11, 12, 13, 14]] and not lay.packed
    packed, seqlens = pack([window(0), window(1)])
    lay = stream_layout(dit.sequence, {**packed, "seqlens": seqlens}, STREAMS)
    # window-major: [ctx 5, video 6, video_cond 4, action 3, action_cond 2] = 20 tokens per window
    assert lay.index.tolist() == [[15, 16, 17, 18, 19], [35, 36, 37, 38, 39]]


def test_gate_zero_leaves_flux_exactly_unchanged(transformer):
    dit = tiny_dit(transformer)
    kw = window(0)
    ref = call(dit, kw)
    adapter = FluxTTTAdapter(32, layers=[0, 2], streams=STREAMS, gate_init=0.0, inner_dim=16).attach(dit)
    out = call(dit, kw)
    for k in ref:
        torch.testing.assert_close(out[k], ref[k], rtol=0, atol=0)
    adapter.detach_hooks()


def test_paper_gate_init_is_near_identity(transformer):
    dit = tiny_dit(transformer)
    kw = window(0)
    ref = call(dit, kw)
    FluxTTTAdapter(32, layers=[0, 1, 2], streams=STREAMS, gate_init=1e-3, inner_dim=16).attach(dit)
    out = call(dit, kw)
    for k in ref:
        rel = (out[k] - ref[k]).abs().max() / ref[k].abs().max()
        assert 0 < rel < 1e-2, (k, float(rel))


def test_hook_after_last_block_touches_only_the_action_streams(transformer):
    dit = tiny_dit(transformer)
    kw = window(0)
    ref = call(dit, kw)
    FluxTTTAdapter(32, layers=[2], streams=STREAMS, gate_init=1.0, inner_dim=16).attach(dit)
    out = call(dit, kw)
    torch.testing.assert_close(out["x_video"], ref["x_video"], rtol=0, atol=0)
    torch.testing.assert_close(out["x_video_cond"], ref["x_video_cond"], rtol=0, atol=0)
    assert not torch.allclose(out["x_action"], ref["x_action"])


def test_memory_advances_only_on_commit(transformer):
    dit = tiny_dit(transformer)
    adapter = FluxTTTAdapter(32, layers=[1], streams=STREAMS, gate_init=1.0, inner_dim=16).attach(dit)
    adapter.reset()
    a = call(dit, window(0))["x_action"]
    b = call(dit, window(0))["x_action"]  # e.g. the next denoising step: still reads W0
    torch.testing.assert_close(a, b)
    adapter.commit()
    c = call(dit, window(0))["x_action"]  # next re-plan: reads W_1
    assert not torch.allclose(a, c)
    adapter.reset()
    torch.testing.assert_close(call(dit, window(0))["x_action"], a)


def test_packed_sequence_training_equals_streaming_inference(transformer):
    """Training packs consecutive re-plans into one call; inference runs them one by one with commits."""
    dit = tiny_dit(transformer)
    adapter = FluxTTTAdapter(32, layers=[0, 2], streams=STREAMS, gate_init=0.7, inner_dim=16).attach(dit)
    windows = [window(s, t_noise=0.2 + 0.15 * s) for s in range(4)]  # one noise level per window
    adapter.reset()
    streamed = []
    for w in windows:
        streamed.append(call(dit, w)["x_action"])
        adapter.commit()
    adapter.reset()
    packed, seqlens = pack(windows)
    out = dit(vector=VECTOR, seqlens=seqlens, **packed)["x_action"]
    torch.testing.assert_close(out, torch.cat(streamed, 1), rtol=1e-4, atol=1e-5)


def test_gradients_reach_only_the_adapter(transformer):
    dit = tiny_dit(transformer)
    adapter = FluxTTTAdapter(32, layers=[1], streams=STREAMS, gate_init=1e-3, inner_dim=16).attach(dit)
    packed, seqlens = pack([window(s) for s in range(3)])
    out = dit(vector=VECTOR, seqlens=seqlens, **packed)
    out["x_action"].pow(2).mean().backward()
    assert all(p.grad is None for p in dit.parameters())
    assert adapter.layers["1"].gate.alpha.grad.abs().sum() > 0
    assert adapter.layers["1"].ttt.w0["w1"].grad.abs().sum() > 0


def test_feature_tap_pools_one_vector_per_window(transformer):
    dit = tiny_dit(transformer)
    tap = FluxFeatureTap(layer=2, streams=("video", "action")).attach(dit)
    call(dit, window(0))
    call(dit, window(1))
    feat = tap.pop()
    assert feat.shape == (1, 1, 32)
    packed, seqlens = pack([window(0), window(1), window(2)])
    dit(vector=VECTOR, seqlens=seqlens, **packed)
    assert tap.pop().shape == (1, 3, 32)
    with pytest.raises(RuntimeError):
        tap.pop()
