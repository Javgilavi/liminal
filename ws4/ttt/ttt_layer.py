"""Test-time-training (TTT) layer with a key-value-binding (KVB) inner loss and a tanh gate.

Follows the RoboTTT recipe (arXiv 2607.15275, Sec. 2-3, App. A.1) on top of the TTT lineage:

* fast model ``f_W``: linear (``x W + b``) or a two-layer GELU MLP, one per head, optionally residual
  (``f_W(x) = x + g_W(x)``, as in Sun et al. 2024, arXiv 2407.04620);
* inner (KVB) loss per token ``l_i(W) = eta_i / 2 * ||f_W(k_i) - v_i||^2`` (RoboTTT Eq. 1);
* mini-batch inner updates: the tokens of one mini-batch share one gradient step taken at the fast weights
  that entered the mini-batch (Sun et al. 2024; LaCT, arXiv 2505.23884, Eq. 4);
* "update then apply" (RoboTTT Eq. 1-2): ``W_i = W_{i-1} - grad``, then ``o = f_{W_i}(q)``;
* learnable initial fast weights ``W0`` and a learnable per-token inner learning rate
  ``eta_i = base_lr * sigmoid(theta_lr . x_i + b)`` (Sun et al. 2024; RoboTTT App. A.1: base rate 0.1);
* gated residual ``O = O_attn + tanh(alpha) * TTT(O_attn)`` with ``alpha`` initialized to 1e-3
  (RoboTTT Eq. 3), so a pretrained network starts out (almost) unchanged.

The inner gradients are written out analytically (no ``torch.autograd.grad`` in the inner loop), as in the
LaCT pseudo-code. They are ordinary differentiable tensor expressions, so the outer loss reaches ``W0``,
the projections and the learning-rate head through "gradients of gradients" with plain ``backward()``.

This module needs PyTorch: it is the PyTorch prototype itself, so torch is imported at module level.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

State = dict[str, Tensor]
FastKind = Literal["linear", "mlp"]
Order = Literal["update_then_apply", "apply_then_update"]

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def gelu_grad(h: Tensor) -> Tensor:
    """Derivative of the exact (erf) GELU: ``Phi(h) + h * phi(h)``."""
    cdf = 0.5 * (1.0 + torch.erf(h * _INV_SQRT_2))
    pdf = torch.exp(-0.5 * h * h) * _INV_SQRT_2PI
    return cdf + h * pdf


def fast_forward(kind: FastKind, state: Mapping[str, Tensor], x: Tensor, residual: bool) -> Tensor:
    """Apply the fast model to ``x`` ``(B, H, L, dh)`` with per-sample, per-head fast weights."""
    if kind == "linear":
        y = x @ state["w"] + state["b"]
    else:
        y = F.gelu(x @ state["w1"] + state["b1"]) @ state["w2"] + state["b2"]
    return x + y if residual else y


def fast_grad(
    kind: FastKind, state: Mapping[str, Tensor], k: Tensor, v: Tensor, eta: Tensor, residual: bool
) -> tuple[State, Tensor]:
    """Gradient of ``sum_i eta_i / (2 n) * ||f_W(k_i) - v_i||^2`` over one mini-batch of ``n`` tokens.

    ``k``, ``v``: ``(B, H, n, dh)``; ``eta``: ``(B, H, n, 1)``. Returns the gradient per fast-weight tensor
    and the per-token inner loss (for logging). Written out by hand; ``tests/test_ttt_layer.py`` checks it
    against autograd.
    """
    n = k.shape[-2]
    if kind == "linear":
        pred = k @ state["w"] + state["b"]
    else:
        h = k @ state["w1"] + state["b1"]
        a = F.gelu(h)
        pred = a @ state["w2"] + state["b2"]
    if residual:
        pred = pred + k
    err = pred - v
    loss = 0.5 * (eta * err.pow(2)).sum(-1)  # (B, H, n)
    e = err * eta / n  # dL/dpred
    if kind == "linear":
        grads = {"w": k.transpose(-1, -2) @ e, "b": e.sum(-2, keepdim=True)}
    else:
        dh = (e @ state["w2"].transpose(-1, -2)) * gelu_grad(h)
        grads = {
            "w1": k.transpose(-1, -2) @ dh,
            "b1": dh.sum(-2, keepdim=True),
            "w2": a.transpose(-1, -2) @ e,
            "b2": e.sum(-2, keepdim=True),
        }
    return grads, loss


class TTTLayer(nn.Module):
    """Multi-head TTT layer: ``(B, L, D) -> (B, L, D)`` plus the fast-weight state after the last token.

    The sequence is processed in mini-batches of ``mini_batch`` tokens. Pass the returned state back in to
    continue the same episode (streaming inference or truncated BPTT); ``state=None`` starts from ``W0``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        fast: FastKind = "mlp",
        mlp_ratio: float = 2.0,
        base_lr: float | None = None,
        mini_batch: int = 16,
        order: Order = "update_then_apply",
        residual_fast: bool = True,
        qk_norm: bool = True,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        if fast not in ("linear", "mlp"):
            raise ValueError(f"unknown fast model {fast!r}")
        if order not in ("update_then_apply", "apply_then_update"):
            raise ValueError(f"unknown order {order!r}")
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.fast, self.order, self.mini_batch = fast, order, mini_batch
        self.residual_fast, self.qk_norm = residual_fast, qk_norm
        # Sun et al. 2024 and RoboTTT App. A.1: base inner rate 1.0 for linear, 0.1 for MLP fast weights.
        self.base_lr = (1.0 if fast == "linear" else 0.1) if base_lr is None else base_lr

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.lr_proj = nn.Linear(dim, num_heads)
        self.out_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.lr_proj.weight, std=0.02)
        nn.init.zeros_(self.lr_proj.bias)  # sigmoid(0) = 0.5: half the base rate at init

        h, dh = num_heads, self.head_dim
        if fast == "linear":
            self.w0 = nn.ParameterDict(
                {"w": nn.Parameter(torch.randn(h, dh, dh) * 0.02), "b": nn.Parameter(torch.zeros(h, 1, dh))}
            )
        else:
            dm = max(1, int(round(dh * mlp_ratio)))
            # LaCT App. A: initial fast weights with std 1/sqrt(fan_in).
            self.w0 = nn.ParameterDict(
                {
                    "w1": nn.Parameter(torch.randn(h, dh, dm) / math.sqrt(dh)),
                    "b1": nn.Parameter(torch.zeros(h, 1, dm)),
                    "w2": nn.Parameter(torch.randn(h, dm, dh) / math.sqrt(dm)),
                    "b2": nn.Parameter(torch.zeros(h, 1, dh)),
                }
            )

    def init_state(self, batch: int) -> State:
        """``W0`` broadcast to the batch (a view: the outer gradient reaches the shared ``W0``)."""
        return {name: p.unsqueeze(0).expand(batch, *p.shape) for name, p in self.w0.items()}

    def _heads(self, x: Tensor) -> Tensor:
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, state: State | None = None, mini_batch: int | None = None) -> tuple[Tensor, State]:
        b, length, _ = x.shape
        size = mini_batch or self.mini_batch
        q, k, v = (self._heads(t) for t in self.qkv(x).chunk(3, dim=-1))
        if self.qk_norm:
            # L2-normalized keys keep one inner step with eta <= 1 from overshooting (LaCT Alg. 1).
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        eta = self.base_lr * torch.sigmoid(self.lr_proj(x)).transpose(1, 2).unsqueeze(-1)  # (B, H, L, 1)
        if state is None:
            state = self.init_state(b)

        outputs = []
        for start in range(0, length, size):
            sl = slice(start, start + size)
            if self.order == "apply_then_update":
                outputs.append(fast_forward(self.fast, state, q[:, :, sl], self.residual_fast))
            grads, _ = fast_grad(self.fast, state, k[:, :, sl], v[:, :, sl], eta[:, :, sl], self.residual_fast)
            state = {name: w - grads[name] for name, w in state.items()}
            if self.order == "update_then_apply":
                outputs.append(fast_forward(self.fast, state, q[:, :, sl], self.residual_fast))
        o = self.out_norm(torch.cat(outputs, dim=2))  # per-head norm
        o = o.transpose(1, 2).reshape(b, length, self.dim)
        return self.out_proj(o), state


class TanhGate(nn.Module):
    """``tanh(alpha) * x`` with a per-channel ``alpha`` (RoboTTT Eq. 3, initialized near zero)."""

    def __init__(self, dim: int, init: float = 1e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh(self.alpha) * x


class GatedTTT(nn.Module):
    """``y = x + tanh(alpha) * TTT(norm(x))``: the residual RoboTTT adds after a block's attention.

    With ``gate_init=0`` the module is exactly the identity at initialization; with the paper's 1e-3 it is
    the identity up to a 1e-3-scaled perturbation, so wrapping a pretrained block keeps its behaviour.
    """

    def __init__(
        self,
        dim: int,
        gate_init: float = 1e-3,
        pre_norm: bool = True,
        inner_dim: int | None = None,
        **ttt_kwargs,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        # ``inner_dim`` runs the memory in a narrower space (FLUX's 3072-wide stream: see flux_hook.py).
        width = dim if inner_dim is None else inner_dim
        self.down = nn.Identity() if inner_dim is None else nn.Linear(dim, inner_dim, bias=False)
        self.ttt = TTTLayer(width, **ttt_kwargs)
        self.up = nn.Identity() if inner_dim is None else nn.Linear(inner_dim, dim, bias=False)
        self.gate = TanhGate(dim, gate_init)

    def init_state(self, batch: int) -> State:
        return self.ttt.init_state(batch)

    def forward(self, x: Tensor, state: State | None = None, mini_batch: int | None = None) -> tuple[Tensor, State]:
        o, state = self.ttt(self.down(self.norm(x)), state, mini_batch)
        return x + self.gate(self.up(o)), state


def detach_state(state):
    """Cut the gradient path at a truncated-BPTT boundary; the fast-weight values carry over unchanged."""
    if isinstance(state, Tensor):
        return state.detach()
    if isinstance(state, Mapping):
        return {key: detach_state(value) for key, value in state.items()}
    if isinstance(state, (list, tuple)):
        return type(state)(detach_state(value) for value in state)
    if state is None:
        return None
    raise TypeError(f"cannot detach state of type {type(state).__name__}")
