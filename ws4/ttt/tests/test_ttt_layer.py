import pytest
import torch
from torch import nn
from ttt import GatedTTT, TemporalTTTBlock, TTTLayer, detach_state, fast_forward, fast_grad
from ttt.temporal import masked_sequence_mean, sequence_action_forcing_timesteps, tbptt_slices


def _state(kind, b=2, h=3, dh=5, dm=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "linear":
        return {"w": torch.randn(b, h, dh, dh, generator=g), "b": torch.randn(b, h, 1, dh, generator=g)}
    return {
        "w1": torch.randn(b, h, dh, dm, generator=g),
        "b1": torch.randn(b, h, 1, dm, generator=g),
        "w2": torch.randn(b, h, dm, dh, generator=g),
        "b2": torch.randn(b, h, 1, dh, generator=g),
    }


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("residual", [False, True])
def test_analytic_inner_gradient_matches_autograd(kind, residual):
    torch.manual_seed(0)
    state = {k: v.double().requires_grad_() for k, v in _state(kind).items()}
    k, v = torch.randn(2, 3, 4, 5).double(), torch.randn(2, 3, 4, 5).double()
    eta = torch.rand(2, 3, 4, 1).double()
    grads, _ = fast_grad(kind, state, k, v, eta, residual)
    pred = fast_forward(kind, state, k, residual)
    loss = (0.5 * eta * (pred - v).pow(2)).sum() / k.shape[-2]
    auto = torch.autograd.grad(loss, list(state.values()))
    for name, g in zip(state, auto, strict=True):
        torch.testing.assert_close(grads[name], g)


def test_linear_memory_writes_and_recalls_associations_without_training():
    """Delta rule with orthonormal keys and eta = 1: one inner step stores every (k, v) exactly."""
    d = 8
    keys = torch.eye(d)[None, None]  # (1, 1, 8, 8): 8 orthonormal keys
    values = torch.randn(1, 1, d, d)
    state = {"w": torch.zeros(1, 1, d, d), "b": torch.zeros(1, 1, 1, d)}
    for i in range(d):  # one token per mini-batch, eta = 1
        grads, _ = fast_grad(
            "linear", state, keys[:, :, i : i + 1], values[:, :, i : i + 1], torch.ones(1, 1, 1, 1), False
        )
        state = {"w": state["w"] - grads["w"], "b": state["b"]}  # keep the bias fixed for an exact readout
    recalled = fast_forward("linear", state, keys, residual=False)
    torch.testing.assert_close(recalled, values)


@pytest.mark.parametrize("fast", ["linear", "mlp"])
def test_streaming_equals_one_pass(fast):
    torch.manual_seed(0)
    layer = TTTLayer(16, num_heads=2, fast=fast, mini_batch=3)
    x = torch.randn(2, 12, 16)
    y_full, s_full = layer(x)
    state, ys = None, []
    for start in range(0, 12, 3):  # chunk boundaries aligned with the inner mini-batches
        y, state = layer(x[:, start : start + 3], state)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, 1), y_full)
    for name in s_full:
        torch.testing.assert_close(state[name], s_full[name])


@pytest.mark.parametrize("order", ["update_then_apply", "apply_then_update"])
def test_causal_across_mini_batches(order):
    torch.manual_seed(0)
    layer = TTTLayer(16, num_heads=2, fast="mlp", mini_batch=4, order=order)
    x = torch.randn(1, 16, 16)
    x2 = x.clone()
    x2[:, 8:] += torch.randn(1, 8, 16)  # change the future (mini-batches 3 and 4)
    y1, _ = layer(x)
    y2, _ = layer(x2)
    torch.testing.assert_close(y1[:, :8], y2[:, :8])
    assert not torch.allclose(y1[:, 8:], y2[:, 8:])


def test_apply_then_update_first_mini_batch_reads_w0_only():
    torch.manual_seed(0)
    layer = TTTLayer(16, num_heads=2, fast="linear", mini_batch=4, order="apply_then_update")
    x = torch.randn(1, 8, 16)
    x2 = x.clone()
    x2[:, 1:4] += 1.0  # same queries at position 0, different keys/values inside the mini-batch
    y1, _ = layer(x)
    y2, _ = layer(x2)
    torch.testing.assert_close(y1[:, 0], y2[:, 0])


@pytest.mark.parametrize("fast", ["linear", "mlp"])
def test_outer_loss_reaches_w0_lr_head_and_projections(fast):
    torch.manual_seed(0)
    layer = TTTLayer(16, num_heads=2, fast=fast, mini_batch=2)
    y, _ = layer(torch.randn(3, 8, 16))
    y.pow(2).mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_gate_zero_is_exact_identity_and_paper_init_is_near_identity():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 32)
    exact = GatedTTT(32, gate_init=0.0, fast="mlp", mini_batch=5)
    y, _ = exact(x)
    torch.testing.assert_close(y, x, rtol=0, atol=0)
    near = GatedTTT(32, gate_init=1e-3, fast="mlp", mini_batch=5)
    y, _ = near(x)
    o, _ = near.ttt(near.norm(x))
    assert (y - x).abs().max() <= 1.01e-3 * o.abs().max()
    assert (y - x).abs().max() > 0


def test_gate_gets_gradient_at_init_so_the_memory_can_switch_on():
    torch.manual_seed(0)
    g = GatedTTT(16, gate_init=1e-3, fast="linear", mini_batch=4)
    y, _ = g(torch.randn(2, 8, 16))
    (y * torch.randn_like(y)).sum().backward()
    assert g.gate.alpha.grad.abs().sum() > 0
    assert g.ttt.w0["w"].grad.abs().sum() > 0


def test_tbptt_carries_values_but_cuts_gradients():
    torch.manual_seed(0)
    layer = TTTLayer(16, num_heads=2, fast="mlp", mini_batch=2)
    x = torch.randn(1, 8, 16, requires_grad=True)
    y_full, _ = layer(x)
    y1, s = layer(x[:, :4])
    y2, _ = layer(x[:, 4:], detach_state(s))
    torch.testing.assert_close(torch.cat([y1, y2], 1), y_full)  # same forward values
    (gx,) = torch.autograd.grad(y2.sum(), x, allow_unused=True)
    assert gx is not None and gx[:, :4].abs().sum() == 0  # nothing flows back into segment 1
    assert gx[:, 4:].abs().sum() > 0
    grads = torch.autograd.grad(y1.sum(), list(layer.w0.values()))
    assert all(g.abs().sum() > 0 for g in grads)  # W0 still learns through the first segment


def test_temporal_block_leaves_unselected_tokens_alone_and_is_causal_in_time():
    torch.manual_seed(0)
    block = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True, norm_first=True)
    tb = TemporalTTTBlock(block, 16, token_index=[0, 2], gate_init=0.5, num_heads=2, fast="linear").eval()
    x = torch.randn(2, 6, 3, 16)
    y, _ = tb(x)
    per_step = block(x.reshape(12, 3, 16)).reshape(2, 6, 3, 16)
    torch.testing.assert_close(y[:, :, 1], per_step[:, :, 1])  # token 1 skips the TTT layer
    assert not torch.allclose(y[:, :, 0], per_step[:, :, 0])
    x2 = x.clone()
    x2[:, 4:] += 1.0
    y2, _ = tb(x2)
    torch.testing.assert_close(y[:, :4], y2[:, :4])  # earlier timesteps cannot see later ones


def test_temporal_block_streaming_matches_sequence():
    torch.manual_seed(0)
    block = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.0, batch_first=True, norm_first=True)
    tb = TemporalTTTBlock(block, 16, gate_init=0.3, num_heads=2, fast="mlp").eval()
    x = torch.randn(1, 5, 3, 16)
    y_seq, _ = tb(x)
    state, ys = None, []
    for t in range(5):  # one control step at a time, as at inference
        y, state = tb(x[:, t : t + 1], state)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, 1), y_seq)


def test_sequence_action_forcing_draws_one_level_per_chunk():
    torch.manual_seed(0)
    for convention in ("flux", "groot"):
        t = sequence_action_forcing_timesteps(64, 32, convention=convention)
        assert t.shape == (64, 32)
        assert ((t >= 0) & (t <= 1)).all()
        assert t.std(1).mean() > 0.05  # independent within a sequence, not one shared level


def test_tbptt_slices_and_masked_mean():
    assert tbptt_slices(10, 4) == [slice(0, 4), slice(4, 8), slice(8, 10)]
    loss = torch.tensor([[1.0, 2.0, 3.0]])
    assert float(masked_sequence_mean(loss, torch.tensor([[False, True, True]]))) == pytest.approx(2.5)
    assert float(masked_sequence_mean(loss, torch.zeros(1, 3, dtype=torch.bool))) == 0.0
