"""How a TTT layer hooks into a per-timestep transformer block, plus the sequence-training helpers.

RoboTTT's layout (arXiv 2607.15275, Sec. 3.1, Fig. 2-3): attention runs *within* one timestep (registers,
state token, noised action tokens; cross-attention to that timestep's vision-language tokens), the TTT layer
runs *across* timesteps over the flattened per-timestep outputs, one timestep per inner mini-batch, and its
output is added back through the tanh gate. :class:`TemporalTTTBlock` is that wiring around any block that
maps ``(batch, tokens, dim) -> (batch, tokens, dim)``.

Helpers for the training recipe (Sec. 3.2-3.3):

* :func:`tbptt_slices` / :func:`detach_state` (in ``ttt_layer``): truncated BPTT, fast weights carried over
  segment boundaries with their gradient cut;
* :func:`sequence_action_forcing_timesteps`: one independent flow-matching noise level per action chunk;
* :func:`masked_sequence_mean`: loss masking, so context-only timesteps (an in-context human video, the
  robot's own failed actions in DAgger Distillation) update the fast weights without being imitation targets.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .ttt_layer import GatedTTT, State


class TemporalTTTBlock(nn.Module):
    """``block`` per timestep, then a gated TTT layer across timesteps on the selected token positions.

    Input ``(B, T, N, D)``: ``T`` timesteps of ``N`` tokens. ``token_index`` picks the positions that enter
    the TTT layer (RoboTTT: registers, state and action tokens; not the vision-language tokens). The other
    positions leave unchanged. One inner mini-batch = the selected tokens of one timestep, so a timestep sees
    the fast weights updated on itself and on every earlier timestep, never on a later one.
    """

    def __init__(self, block: nn.Module, dim: int, token_index: Sequence[int] | None = None, **gated_kwargs):
        super().__init__()
        self.block = block
        self.ttt = GatedTTT(dim, **gated_kwargs)
        self.register_buffer(
            "token_index",
            None if token_index is None else torch.as_tensor(list(token_index), dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        b, t, n, d = x.shape
        h = self.block(x.reshape(b * t, n, d)).reshape(b, t, n, d)
        idx = self.token_index
        sel = h if idx is None else h.index_select(2, idx)
        k = sel.shape[2]
        out, state = self.ttt(sel.reshape(b, t * k, d), state, mini_batch=k)
        out = out.reshape(b, t, k, d)
        return (out if idx is None else h.index_copy(2, idx, out)), state


def tbptt_slices(length: int, segment: int) -> list[slice]:
    """Consecutive segments of at most ``segment`` timesteps covering ``range(length)``."""
    if segment < 1:
        raise ValueError("segment length must be positive")
    return [slice(s, min(s + segment, length)) for s in range(0, length, segment)]


def sequence_action_forcing_timesteps(
    batch: int,
    steps: int,
    generator: torch.Generator | None = None,
    convention: str = "flux",
    width: float = 0.75,
    shift: float = 42.0,
) -> Tensor:
    """``(batch, steps)`` flow-matching noise levels, drawn independently for every action chunk.

    RoboTTT found that one shared level per sequence ("full-sequence diffusion") makes training unstable and
    samples one per chunk instead (Sec. 3.2, Eq. 5). ``convention="groot"``: RoboTTT's draw
    ``tau = 0.999 * (1 - u)``, ``u ~ Beta(1.5, 1)``, with ``tau = 1`` the clean action. ``convention="flux"``:
    flux-action's training draw (``processing/packing.py:397-410``: ``logit(t) ~ Logistic(0, width)`` then
    the rational shift), with ``t = 1`` pure noise. Only the per-chunk independence is RoboTTT's idea.
    """
    if convention == "groot":
        u = torch.distributions.Beta(1.5, 1.0).sample((batch, steps))
        return 0.999 * (1.0 - u)
    if convention == "flux":
        eps = torch.finfo(torch.float32).eps
        u = torch.rand(batch, steps, generator=generator).clamp(eps, 1 - eps)
        t = torch.sigmoid(width * (torch.log(u) - torch.log1p(-u)))
        return shift * t / (1.0 + (shift - 1.0) * t)
    raise ValueError(f"unknown convention {convention!r}")


def masked_sequence_mean(loss: Tensor, mask: Tensor) -> Tensor:
    """Mean of a per-timestep loss over the timesteps whose ``mask`` is true (0 when none is)."""
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)
