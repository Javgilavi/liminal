"""Where a RoboTTT-style memory attaches in FLUX 3 Action, as working hooks on flux-action's reference DiT.

Source pinned: black-forest-labs/flux-action@830f1ce178d7aea52f18fde31d8e8e36ce6bae21 (cloned into
``ws4/.cache/flux-action``, read on 2026-10-17). Line numbers below are in that commit.

FLUX's DiT, ``JointSingleSeq`` (``src/flux_action/models/transformer.py:553``): hidden 3072, 24 heads,
5 per-stream ``ModeBlock`` layers per stream (``content_mode_blocks[m]``, ``txt_mode_blocks``) and 28 joint
``SingleStreamBlock`` layers over all streams (``single_blocks``); defaults at ``transformer.py:145-157``.
Streams of one SO-101 "history" window (``configs/so101/train.json``, ``processing/history.py``):

=====================  ====================================================  ===========================
token group            stream and content                                    tokens (SO-101 recipe)
=====================  ====================================================  ===========================
task                   ``ctx``: Qwen3-VL caption features                    320 (``text_fixed_length``)
history (visual)       ``x_video_cond``: VAE latents of 2 snapshots          2 x 8 x 16 = 256
state + past actions   ``x_action_cond``: 8 ticks of [past cmd delta|state]  8 (``history.py:120-154``)
future                 ``x_video``: noised future latents                    11 x 8 x 16 = 1408
action                 ``x_action``: noised action chunk                     42 (``chunk_size``)
=====================  ====================================================  ===========================

Forward (``transformer.py:720-906``): each stream runs its own 5 mode blocks (``844-849``), the content
streams are concatenated in ``self.sequence`` order (``851``) and the 28 joint blocks attend over
``[ctx | content]`` (``856-866``), or, with ``seqlens``, over each packed window separately in window-major
order (``867-880``). So FLUX's whole memory is the 8-tick (0.27 s at 30 Hz) state/command history and 2
snapshots. ``select_action`` re-plans only when the 32-step queue is empty (``policy.py:1069-1077``): one DiT
"timestep" per re-plan, i.e. per executed chunk of 32 ticks (about 1.07 s).

Attachment (RoboTTT Sec. 3.1, Fig. 2-3, mapped onto FLUX):

* after joint block ``i`` (the block output already holds its parallel attention + MLP residual,
  ``transformer.py:541-550``; FLUX has no "between attention and MLP" point), a :class:`GatedTTT` reads the
  tokens of the action streams (``x_action`` + ``x_action_cond``: 50 tokens for SO-101). They play the role
  of RoboTTT's noised action tokens and state token. Like RoboTTT, the 1984 visual and text tokens skip the
  TTT layer; they reach it through joint attention. Learned register tokens (RoboTTT's ``R_t``) would need a
  new stream or extra ``x_action_cond`` tokens and a fine-tune: left out here;
* sequence unit = one window = one re-plan; one inner mini-batch = that window's 50 tokens;
* inference: every denoising call reads the committed state ``W_{t-1}`` and stages ``W_t``; after the chunk is
  sampled, :meth:`FluxTTTAdapter.commit` keeps the state staged by the last call, which is the conditional
  pass of the final step (``inference/sampling.py:61-63`` runs unconditional first);
* sequence training: flux-action already packs the windows of a micro-batch into one sequence with
  attention confined to each window (``processing/packing.py:478-511``) and draws one noise level per window
  (``policy.py:662``, ``packing.py:397-410``). With the windows of a micro-batch taken as consecutive re-plans
  of one episode, that *is* RoboTTT's per-timestep attention plus sequence action forcing; the hook only has
  to run the TTT layer across the packed windows. Truncated BPTT = :meth:`commit` between micro-batches.

The adapter is a sidecar ``nn.Module``: FLUX's own module tree and state-dict keys stay untouched, so
``load_action_checkpoint`` (``models/wiring.py:182``) keeps working and the adapter saves separately.
Attach it after the FLUX weights are loaded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .ttt_layer import GatedTTT, State, detach_state

FLUX_ACTION_COMMIT = "830f1ce178d7aea52f18fde31d8e8e36ce6bae21"


@dataclass(frozen=True)
class StreamLayout:
    """Positions of the memory's tokens in a joint block's output for one DiT call.

    Unpacked call: ``index`` is ``(1, n)`` positions in the content sequence ``img`` (one window). Packed
    call: ``index`` is ``(W, n)`` positions in the window-major joint sequence, one row per window.
    """

    packed: bool
    index: Tensor

    @property
    def tokens_per_step(self) -> int:
        return int(self.index.shape[1])

    @property
    def steps(self) -> int:
        return int(self.index.shape[0])


def stream_layout(sequence: Mapping[str, str], kwargs: Mapping[str, Any], streams: Sequence[str]) -> StreamLayout:
    """Mirror ``JointSingleSeq.forward``'s token bookkeeping (``transformer.py:748-781``, ``851``, ``871-875``)."""
    lens: dict[str, int] = {}
    first_key: dict[str, str] = {}
    for key, mod in sequence.items():  # stream order = insertion order of ``params.sequence``
        if key in kwargs:
            lens[mod] = lens.get(mod, 0) + int(kwargs[key].shape[1])
            first_key.setdefault(mod, key)
    missing = [s for s in streams if s not in lens]
    if missing:
        raise ValueError(f"streams {missing} are not in this DiT call (active: {list(lens)})")
    active = list(lens)
    seqlens = kwargs.get("seqlens")
    if seqlens is None:
        offsets, pos = {}, 0
        for mod in active:
            offsets[mod] = pos
            pos += lens[mod]
        index = torch.cat([torch.arange(offsets[m], offsets[m] + lens[m]) for m in streams])
        return StreamLayout(False, index[None])
    ctx_lens = [int(n) for n in seqlens["ctx"]]
    per_window = {m: [int(n) for n in seqlens[first_key[m]]] for m in active}
    rows, base = [], 0
    for i, n_ctx in enumerate(ctx_lens):  # window i: ctx_i, then each active stream's window-i tokens
        pos, offsets = base + n_ctx, {}
        for mod in active:
            offsets[mod] = pos
            pos += per_window[mod][i]
        rows.append(torch.cat([torch.arange(offsets[m], offsets[m] + per_window[m][i]) for m in streams]))
        base = pos
    if len({r.numel() for r in rows}) != 1:
        raise ValueError("every packed window must carry the same number of memory tokens")
    return StreamLayout(True, torch.stack(rows))


class FluxTTTAdapter(nn.Module):
    """Gated TTT layers after selected joint blocks of a flux-action ``JointSingleSeq``.

    ``streams``: the action modality and its conditioning stream, e.g. ``("action", "action_cond")`` for the
    SO-101 recipe (``action_modality="action"``). ``inner_dim`` narrows the memory (3072 -> e.g. 512) to keep
    the added parameters small: about ``2 * 3072 * inner_dim + 4 * inner_dim**2`` per layer.
    """

    def __init__(
        self,
        hidden_size: int,
        layers: Sequence[int],
        streams: Sequence[str],
        gate_init: float = 1e-3,
        inner_dim: int | None = 512,
        **ttt_kwargs: Any,
    ):
        super().__init__()
        self.layer_ids = [int(i) for i in layers]
        self.streams = tuple(streams)
        ttt_kwargs.setdefault("fast", "mlp")
        ttt_kwargs.setdefault("num_heads", 4)
        self.layers = nn.ModuleDict(
            {str(i): GatedTTT(hidden_size, gate_init=gate_init, inner_dim=inner_dim, **ttt_kwargs) for i in layers}
        )
        self.committed: dict[int, State] = {}
        self.pending: dict[int, State] = {}
        self.layout: StreamLayout | None = None
        self.enabled = True
        self._handles: list[Any] = []

    # ---- episode state -------------------------------------------------------------------------
    def reset(self) -> None:
        """Start a new episode: the next call reads the learned ``W0``."""
        self.committed, self.pending = {}, {}

    def commit(self, detach: bool = True) -> None:
        """Keep the fast weights staged by the last DiT call; ``detach`` cuts the gradient (truncated BPTT)."""
        if self.pending:
            self.committed = {k: (detach_state(v) if detach else v) for k, v in self.pending.items()}
        self.pending = {}

    # ---- hooks ---------------------------------------------------------------------------------
    def attach(self, dit: nn.Module) -> FluxTTTAdapter:
        if self._handles:
            raise RuntimeError("adapter already attached")
        sequence = dict(dit.sequence)

        def record_layout(_module, args, kwargs):
            if args:
                raise TypeError("FluxTTTAdapter expects JointSingleSeq to be called with keyword arguments")
            self.layout = stream_layout(sequence, kwargs, self.streams)

        self._handles.append(dit.register_forward_pre_hook(record_layout, with_kwargs=True))
        for i in self.layer_ids:
            self._handles.append(dit.single_blocks[i].register_forward_hook(self._block_hook(i), with_kwargs=True))
        return self

    def detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _block_hook(self, key: int):
        ttt = self.layers[str(key)]

        def hook(_module, _args, _kwargs, output):
            layout = self.layout
            if layout is None or not self.enabled:
                return output
            ctx_out, x_out = output
            flat = layout.index.to(x_out.device).reshape(-1)
            sel = x_out.index_select(1, flat).float()  # (B, steps * n, D); the memory runs in fp32
            state = self.committed.get(key)
            if state is None or next(iter(state.values())).shape[0] != x_out.shape[0]:
                state = ttt.init_state(x_out.shape[0])
            y, self.pending[key] = ttt(sel, state, mini_batch=layout.tokens_per_step)
            return ctx_out, x_out.index_copy(1, flat, y.to(x_out.dtype))

        return hook


class FluxFeatureTap:
    """Pipeline P2 (FLUX frozen): pool a joint block's hidden states of chosen streams at every DiT call.

    ``streams=("video", "action")`` at a late block, read at the final denoising call, summarizes what FLUX
    imagines (future video latents) and intends (action chunk) for this re-plan; ``("video_cond",
    "action_cond")`` summarizes what it observed. :meth:`pop` returns the last call's ``(B, D)`` feature.
    """

    def __init__(self, layer: int, streams: Sequence[str]):
        self.layer, self.streams = int(layer), tuple(streams)
        self.layout: StreamLayout | None = None
        self.features: list[Tensor] = []
        self._handles: list[Any] = []

    def attach(self, dit: nn.Module) -> FluxFeatureTap:
        sequence = dict(dit.sequence)

        def record_layout(_module, _args, kwargs):
            self.layout = stream_layout(sequence, kwargs, self.streams)

        def capture(_module, _args, _kwargs, output):
            _, x_out = output
            flat = self.layout.index.to(x_out.device)
            tokens = x_out[:, flat.reshape(-1)].unflatten(1, flat.shape)  # (B, steps, n, D)
            self.features.append(tokens.float().mean(2).detach())  # (B, steps, D)

        self._handles = [
            dit.register_forward_pre_hook(record_layout, with_kwargs=True),
            dit.single_blocks[self.layer].register_forward_hook(capture, with_kwargs=True),
        ]
        return self

    def pop(self) -> Tensor:
        """Feature of the last DiT call, ``(B, steps, D)``; clears the buffer (call once per re-plan)."""
        if not self.features:
            raise RuntimeError("no DiT call captured since the last pop")
        last = self.features[-1]
        self.features = []
        return last

    def detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


# ---- how the pieces plug into flux-action's policy and trainer (pseudo-code, needs the weights) --------
#
# Inference, SO-101 history profile (policy.py:1060-1077), one memory step per re-plan:
#
#     adapter = FluxTTTAdapter(3072, layers=[24, 25, 26, 27], streams=("action", "action_cond")).attach(policy.dit)
#     adapter.load_state_dict(torch.load("ttt_adapter.pt")); adapter.to(policy.device).eval(); adapter.reset()
#     ...
#     if not policy._action_queue:                       # inside select_action, before predict
#         targets = policy.predict_normalized_targets(window)   # 4 solver steps x (1 or 2 CFG passes)
#         adapter.commit()                               # W_t from the last call (conditional, final step)
#
# Sequence training with the flux-action trainer (training/trainer.py), FLUX frozen, adapter trainable:
#
#     for episode in episodes:                           # consecutive windows, stride = n_action_steps (32)
#         adapter.reset()
#         for windows in split(episode.windows, 16):     # one micro-batch = one TBPTT segment, time-ordered
#             loss, _ = policy(collate(windows))         # packed: per-window attention + per-window noise level
#             loss.backward()                            # grads reach only the adapter (FLUX requires_grad=False)
#             adapter.commit()                           # fast weights carry over, gradient cut
#         optimizer.step(); optimizer.zero_grad()
#
# Loss masking (RoboTTT Sec. 3.3): zero ``action_mask`` rows (packing.py:519-572) for windows that should only
# be context (the robot's own failed actions in DAgger Distillation, a human demo clip); they still update the
# fast weights because the hook runs on every window.
