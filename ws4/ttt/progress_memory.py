"""Pipeline P2: an episode memory downstream of a frozen FLUX 3 Action, for the orchestrator.

One feature vector per FLUX re-plan (``flux_hook.FluxFeatureTap``: pooled hidden states of a late joint block)
goes into a small memory (gated TTT layer, or a GRU) that runs over the whole episode, one step per re-plan, and
predicts the current subtask and a "subtask done" flag. On the synthetic check below a GRU was as good or better
than the TTT memory at up to about 190 re-plans (3 seeds), so ``memory="gru"`` is the practical choice for
demo-length episodes. The output is advisory: the orchestrator (WS3) decides, perception still checks success,
and nothing here reaches the motors.

``python ws4/ttt/progress_memory.py`` runs a synthetic check of the failure mode it is for: a task made of
repeated, identical-looking stages (pick/place object A, then the same motions for object B), where a
per-re-plan classifier cannot tell which repetition it is in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ttt.temporal import tbptt_slices
    from ttt.ttt_layer import GatedTTT, State, detach_state
else:
    from .temporal import tbptt_slices
    from .ttt_layer import GatedTTT, State, detach_state


class EpisodeProgressMemory(nn.Module):
    """``(B, T, feat_dim)`` re-plan features -> subtask logits ``(B, T, n_subtasks)`` and done logits ``(B, T)``.

    ``memory``: ``"ttt"`` (gated TTT layer), ``"gru"`` (same width GRU, the obvious cheap alternative),
    ``"lastk"`` (the last ``k`` re-plans side by side, a fixed window like FLUX's own short history) or ``"none"``
    (per-re-plan baseline). Everything else is shared. Optional heads: ``n_occurrences`` (how many times the
    current skill already ran, the part that needs memory when stages repeat) and ``progress_head`` (progress
    inside the current stage, 0..1). Input features are standardized with ``feat_mean``/``feat_std`` buffers
    (:meth:`set_normalization`), so a saved state dict is self-contained.
    """

    def __init__(
        self,
        feat_dim: int,
        n_subtasks: int,
        dim: int = 128,
        heads: int = 4,
        fast: str = "linear",
        memory: str = "ttt",
        gate_init: float = 1.0,
        n_occurrences: int = 0,
        progress_head: bool = False,
        k: int = 4,
    ):
        super().__init__()
        self.register_buffer("feat_mean", torch.zeros(feat_dim))
        self.register_buffer("feat_std", torch.ones(feat_dim))
        self.inp = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim))
        self.local = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        if memory not in ("ttt", "gru", "lastk", "none"):
            raise ValueError(f"unknown memory {memory!r}")
        self.kind, self.k = memory, k
        self.window = nn.Linear(k * dim, dim) if memory == "lastk" else None
        # No pretrained model to protect here, so the gate starts open (1.0) unlike inside FLUX.
        self.memory = (
            GatedTTT(dim, gate_init=gate_init, num_heads=heads, fast=fast, mini_batch=1)
            if memory == "ttt"
            else nn.GRU(dim, dim, batch_first=True)
            if memory == "gru"
            else None
        )
        self.post = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.subtask = nn.Linear(dim, n_subtasks)
        self.done = nn.Linear(dim, 1)
        self.occurrence = nn.Linear(dim, n_occurrences) if n_occurrences else None
        self.stage_progress = nn.Linear(dim, 1) if progress_head else None

    @torch.no_grad()
    def set_normalization(self, mean: Tensor, std: Tensor) -> None:
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(1e-4))

    def forward(
        self, feats: Tensor, state: State | Tensor | None = None
    ) -> tuple[dict[str, Tensor], State | Tensor | None]:
        h = self.inp((feats - self.feat_mean) / self.feat_std)
        h = h + self.local(h)
        if self.kind == "ttt":
            h, state = self.memory(h, state, mini_batch=1)
        elif self.kind == "gru":
            out, state = self.memory(h, state)
            h = h + out
        elif self.kind == "lastk":
            b, t, d = h.shape
            past = h.new_zeros(b, self.k - 1, d) if state is None else state
            seq = torch.cat([past, h], 1)  # (B, k - 1 + T, D)
            win = seq.unfold(1, self.k, 1)  # (B, T, D, k): window ending at each step
            h = h + self.window(win.permute(0, 1, 3, 2).reshape(b, t, self.k * d))
            state = seq[:, -(self.k - 1) :] if self.k > 1 else seq[:, :0]
        h = self.norm(h + self.post(h))
        out = {"subtask": self.subtask(h), "done": self.done(h).squeeze(-1)}
        if self.occurrence is not None:
            out["occurrence"] = self.occurrence(h)
        if self.stage_progress is not None:
            out["stage_progress"] = torch.sigmoid(self.stage_progress(h).squeeze(-1))
        return out, state

    @torch.no_grad()
    def step(
        self, feat: Tensor, state: State | Tensor | None = None
    ) -> tuple[dict[str, Tensor], State | Tensor | None]:
        """Streaming use, one call per re-plan: ``feat (B, feat_dim)``; keep the returned state per episode."""
        out, state = self(feat[:, None], state)
        return {k: v[:, 0] for k, v in out.items()}, state


def fit(
    model: EpisodeProgressMemory,
    feats: Tensor,
    subtask: Tensor,
    done: Tensor,
    steps: int = 400,
    batch: int = 32,
    segment: int = 32,
    lr: float = 3e-3,
    seed: int = 0,
) -> list[float]:
    """Train on padded episodes ``feats (N, T, F)``, ``subtask (N, T)`` (-1 = padding), ``done (N, T)``."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, feats.shape[0], (batch,), generator=gen)
        x, y, d = feats[idx], subtask[idx], done[idx]
        valid = y >= 0
        denom = valid.sum().clamp_min(1)
        opt.zero_grad()
        state, total = None, 0.0
        for sl in tbptt_slices(x.shape[1], segment):
            out, state = model(x[:, sl], state)
            ce = F.cross_entropy(out["subtask"].flatten(0, 1), y[:, sl].clamp_min(0).flatten(), reduction="none")
            bce = F.binary_cross_entropy_with_logits(out["done"], d[:, sl].float(), reduction="none").flatten()
            loss = ((ce + bce) * valid[:, sl].flatten().float()).sum() / denom
            loss.backward()
            total += float(loss.detach())
            state = detach_state(state)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(total)
    return losses


def synthetic_aliased_episodes(
    n: int, feat_dim: int = 32, objects: int = 2, noise: float = 0.3, seed: int = 0, proto_seed: int = 1234
) -> dict[str, Tensor]:
    """Episodes of ``objects`` x [reach, grasp, move, place] stages, 2-6 re-plans each.

    The feature of a re-plan depends only on the stage *type* (plus noise), so the second object's stages look
    exactly like the first's. Label = stage index over the whole episode (``4 * objects`` classes); ``done`` = 1
    on the last re-plan of a stage.
    """
    # The stage-type prototypes stand for "what FLUX sees": fixed across splits, only episodes vary with seed.
    protos = torch.randn(4, feat_dim, generator=torch.Generator().manual_seed(proto_seed))
    gen = torch.Generator().manual_seed(seed)
    episodes = []
    for _ in range(n):
        labels, done = [], []
        for stage in range(4 * objects):
            length = int(torch.randint(2, 7, (1,), generator=gen))
            labels += [stage] * length
            done += [0] * (length - 1) + [1]
        episodes.append((torch.tensor(labels), torch.tensor(done)))
    t_max = max(len(lab) for lab, _ in episodes)
    feats = torch.zeros(n, t_max, feat_dim)
    subtask = torch.full((n, t_max), -1, dtype=torch.long)
    done_t = torch.zeros(n, t_max, dtype=torch.long)
    for i, (lab, dn) in enumerate(episodes):
        t = len(lab)
        feats[i, :t] = protos[lab % 4] + noise * torch.randn(t, feat_dim, generator=gen)
        subtask[i, :t], done_t[i, :t] = lab, dn
    return {"feats": feats, "subtask": subtask, "done": done_t}


@torch.no_grad()
def accuracy(model: EpisodeProgressMemory, data: dict[str, Tensor]) -> float:
    model.eval()
    out, _ = model(data["feats"])
    valid = data["subtask"] >= 0
    return float((out["subtask"].argmax(-1) == data["subtask"])[valid].float().mean())


def main(argv: list[str] | None = None) -> dict[str, float]:
    from ttt.throttle import throttle

    throttle()  # shares the laptop with 30 Hz robot loops
    parser = argparse.ArgumentParser(description="synthetic check of the P2 episode memory")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--objects", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None, help="write the accuracies as JSON")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    train = synthetic_aliased_episodes(512, objects=args.objects, seed=2 * args.seed)
    test = synthetic_aliased_episodes(256, objects=args.objects, seed=2 * args.seed + 1)
    results = {"chance_upper_bound_without_memory": 1.0 / args.objects}
    for name, kind in (("per_replan_no_memory", "none"), ("gru_memory", "gru"), ("ttt_memory", "ttt")):
        torch.manual_seed(args.seed)
        model = EpisodeProgressMemory(32, 4 * args.objects, memory=kind)
        fit(model, train["feats"], train["subtask"], train["done"], steps=args.steps, seed=args.seed)
        results[name] = accuracy(model, test)
        print(f"{name:<22} subtask accuracy {results[name]:.3f}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"objects": args.objects, "steps": args.steps, "seed": args.seed, **results}, indent=2)
        )
    return results


if __name__ == "__main__":
    main()
