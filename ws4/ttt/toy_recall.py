"""Toy long-horizon task: does the gated TTT memory work, and does the gate leave a pretrained policy intact?

Episode (one token of features per timestep, plus learned register tokens):

1. demo phase (context only, loss masked): ``n_pairs`` timesteps, each showing an (object, bin) pair, the
   analogue of RoboTTT's in-context demonstration;
2. work phase (``gap`` timesteps): an object id is shown and the target is a fixed reactive skill
   ``bin = (3 * object + 1) mod n_bins`` that needs no memory (what a pretrained short-context policy knows);
3. query phase (``n_queries`` timesteps): an object from the demo is shown again and the target is the bin
   the demo paired it with, which only a policy that remembers the demo can produce.

Protocol:

* stage 1: pretrain a per-timestep policy without memory ("base", stands in for pretrained FLUX);
* stage 2a ("base+ttt_*"): wrap its first block with a gated TTT layer (``tanh(1e-3)`` gate) initialized from
  the base weights, check that the outputs at initialization match the base, fine-tune with truncated BPTT on
  short episodes (gap <= ``train_gap``), then evaluate on gaps up to 64x longer with constant memory;
* stage 2b ("scratch+ttt_*"): the same memory policy trained from scratch on the same total step budget;
* baseline: the same policy with an 8-timestep frame stack (the size of FLUX 3 Action's SO-101 history,
  ``n_obs_steps=8``), trained from scratch on the same data.

Run: ``python ws4/ttt/toy_recall.py`` (several minutes on a laptop CPU), ``--quick`` for a smoke run.
Results go to ``ws4/ttt/results/toy_recall.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ttt.temporal import TemporalTTTBlock, tbptt_slices
    from ttt.ttt_layer import detach_state
else:
    from .temporal import TemporalTTTBlock, tbptt_slices
    from .ttt_layer import detach_state

DEMO, WORK, QUERY = 0, 1, 2


@dataclass
class RecallTask:
    n_objects: int = 16
    n_bins: int = 16
    n_pairs: int = 6
    n_queries: int = 6
    obs_noise: float = 0.1

    @property
    def n_features(self) -> int:
        return self.n_objects + self.n_bins + 3

    def reactive_bin(self, obj: Tensor) -> Tensor:
        return (3 * obj + 1) % self.n_bins

    def sample(self, batch: int, gap: int, generator: torch.Generator) -> dict[str, Tensor]:
        """``obs (B, T, F)``, ``target (B, T)``, ``phase (B, T)`` with ``T = n_pairs + gap + n_queries``."""
        p, q = self.n_pairs, self.n_queries
        t = p + gap + q
        objs = torch.stack([torch.randperm(self.n_objects, generator=generator)[:p] for _ in range(batch)])
        bins = torch.randint(0, self.n_bins, (batch, p), generator=generator)
        work_objs = torch.randint(0, self.n_objects, (batch, gap), generator=generator)
        pick = torch.randint(0, p, (batch, q), generator=generator)
        query_objs, query_bins = objs.gather(1, pick), bins.gather(1, pick)

        shown_obj = torch.cat([objs, work_objs, query_objs], 1)  # (B, T)
        obs = torch.zeros(batch, t, self.n_features)
        obs.scatter_(2, shown_obj[..., None], 1.0)
        obs[:, :p].scatter_(2, (self.n_objects + bins)[..., None], 1.0)
        phase = torch.cat(
            [torch.full((batch, p), DEMO), torch.full((batch, gap), WORK), torch.full((batch, q), QUERY)], 1
        )
        obs.scatter_(2, (self.n_objects + self.n_bins + phase)[..., None], 1.0)
        obs += self.obs_noise * torch.randn(obs.shape, generator=generator)
        target = torch.cat([bins, self.reactive_bin(work_objs), query_bins], 1)
        return {"obs": obs, "target": target, "phase": phase}


class ToyPolicy(nn.Module):
    """Per-timestep transformer (observation token + registers); optional memory across timesteps."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        dim: int = 64,
        heads: int = 4,
        registers: int = 2,
        memory: str = "none",
        frame_stack: int = 1,
        gate_init: float = 1e-3,
        ttt_tokens: list[int] | None = None,
        ttt_kwargs: dict | None = None,
    ):
        super().__init__()
        self.memory, self.frame_stack = memory, frame_stack
        self.embed = nn.Linear(n_in * frame_stack, dim)
        self.registers = nn.Parameter(torch.randn(registers, dim) * 0.02)
        self.block1 = self._block(dim, heads)
        self.block2 = self._block(dim, heads)
        self.readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_out))
        self.temporal = None
        if memory in ("ttt_linear", "ttt_mlp"):
            self.temporal = TemporalTTTBlock(
                self.block1,
                dim,
                token_index=ttt_tokens,
                gate_init=gate_init,
                num_heads=heads,
                fast=memory.removeprefix("ttt_"),
                **(ttt_kwargs or {}),
            )

    @staticmethod
    def _block(dim: int, heads: int) -> nn.Module:
        return nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True)

    def _stack(self, obs: Tensor) -> Tensor:
        if self.frame_stack == 1:
            return obs
        b, t, f = obs.shape
        padded = torch.cat([obs.new_zeros(b, self.frame_stack - 1, f), obs], 1)
        return torch.cat([padded[:, i : i + t] for i in range(self.frame_stack)], -1)

    def forward(self, obs: Tensor, state=None) -> tuple[Tensor, object]:
        b, t, _ = obs.shape
        tok = self.embed(self._stack(obs))[:, :, None]
        x = torch.cat([tok, self.registers.expand(b, t, -1, -1)], 2)  # (B, T, N, D)
        n, d = x.shape[2:]
        if self.temporal is not None:
            x, state = self.temporal(x, state)
        else:
            x = self.block1(x.reshape(b * t, n, d)).reshape(b, t, n, d)
        x = self.block2(x.reshape(b * t, n, d)).reshape(b, t, n, d)
        return self.readout(x[:, :, 0]), state


def masked_ce(logits: Tensor, target: Tensor, mask: Tensor, denom: Tensor) -> Tensor:
    ce = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="none")
    return (ce * mask.flatten().float()).sum() / denom


def train(
    model: ToyPolicy,
    task: RecallTask,
    steps: int,
    train_gap: int,
    segment: int | None,
    seed: int,
    lr: float = 3e-3,
    batch: int = 64,
    gate_lr_mult: float = 1.0,
    demo_loss: bool = False,
) -> list[float]:
    """Adam with truncated BPTT over ``segment``-timestep pieces (``None``: one piece per episode).

    ``gate_lr_mult`` scales the learning rate of the tanh-gate parameters only. Adam moves ``alpha`` by about
    ``lr`` per step, so from 1e-3 the gate needs hundreds of steps to open; RoboTTT trains for 30K steps,
    this toy for a few hundred, hence the option.
    """
    gen = torch.Generator().manual_seed(seed)
    gates = [p for n, p in model.named_parameters() if n.endswith("gate.alpha")]
    rest = [p for n, p in model.named_parameters() if not n.endswith("gate.alpha")]
    opt = torch.optim.Adam([{"params": rest}, {"params": gates, "lr": lr * gate_lr_mult}], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))
    losses = []
    model.train()
    for _ in range(steps):
        ep = task.sample(batch, int(torch.randint(0, train_gap + 1, (1,), generator=gen)), gen)
        # With a memory the demo is context only (it updates fast weights, it is never a target), as in RoboTTT's
        # in-context video imitation. The memory-less pretraining instead learns to read every observation,
        # demo pairs included ("copy the bin you are shown"), like a pretrained policy that perceives well.
        mask = torch.ones_like(ep["phase"], dtype=torch.bool) if demo_loss else ep["phase"] != DEMO
        denom = mask.sum().clamp_min(1)
        length = ep["obs"].shape[1]
        stateful = model.temporal is not None and segment is not None
        opt.zero_grad()
        state, total = None, 0.0
        for sl in tbptt_slices(length, segment if stateful else length):
            logits, state = model(ep["obs"][:, sl], state)
            loss = masked_ce(logits, ep["target"][:, sl], mask[:, sl], denom)
            loss.backward()
            total += float(loss.detach())
            state = detach_state(state)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(total)
    return losses


@torch.no_grad()
def evaluate(model: ToyPolicy, task: RecallTask, gap: int, seed: int, batch: int = 256) -> dict[str, float]:
    model.eval()
    ep = task.sample(batch, gap, torch.Generator().manual_seed(seed))
    logits, _ = model(ep["obs"])
    correct = logits.argmax(-1) == ep["target"]
    out = {}
    for name, ph in (("reactive_acc", WORK), ("recall_acc", QUERY)):
        m = ep["phase"] == ph
        out[name] = float(correct[m].float().mean()) if m.any() else float("nan")
    return out


@torch.no_grad()
def inner_lr_by_phase(model: ToyPolicy, task: RecallTask, gap: int, seed: int) -> dict[str, float]:
    """Mean learned inner learning rate of the observation token, per phase: what the memory chooses to keep."""
    captured = {}
    lr_proj = model.temporal.ttt.ttt.lr_proj
    handle = lr_proj.register_forward_hook(lambda _m, _i, out: captured.setdefault("eta", out))
    ep = task.sample(128, gap, torch.Generator().manual_seed(seed))
    model.eval()
    model(ep["obs"])
    handle.remove()
    idx = model.temporal.token_index
    n_tok = 1 + model.registers.shape[0] if idx is None else idx.numel()
    eta = torch.sigmoid(captured["eta"]).mean(-1).reshape(128, -1, n_tok)[:, :, 0]  # obs token, mean over heads
    eta = eta * model.temporal.ttt.ttt.base_lr
    return {
        name: float(eta[ep["phase"] == ph].mean()) for name, ph in (("demo", DEMO), ("work", WORK), ("query", QUERY))
    }


@torch.no_grad()
def init_deviation(wrapped: ToyPolicy, base: ToyPolicy, task: RecallTask, seed: int) -> dict[str, float]:
    """How far the freshly wrapped policy is from the pretrained base (gate at initialization)."""
    wrapped.eval()
    base.eval()
    ep = task.sample(128, 24, torch.Generator().manual_seed(seed))
    a, _ = wrapped(ep["obs"])
    b, _ = base(ep["obs"])
    return {
        "max_abs_logit_diff": float((a - b).abs().max()),
        "max_abs_base_logit": float(b.abs().max()),
        "same_argmax_frac": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
    }


def main(argv: list[str] | None = None) -> dict:
    from ttt.throttle import throttle

    throttle()  # shares the laptop with 30 Hz robot loops
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="tiny smoke run (seconds)")
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--train-gap", type=int, default=16)
    parser.add_argument("--segment", type=int, default=24, help="TBPTT segment, in timesteps")
    parser.add_argument("--gate-lr-mult", type=float, default=10.0, help="1 = the paper's single learning rate")
    parser.add_argument("--eval-gaps", type=int, nargs="+", default=[0, 16, 64, 256, 1024])
    parser.add_argument("--kinds", nargs="+", default=["ttt_linear", "ttt_mlp"], choices=["ttt_linear", "ttt_mlp"])
    parser.add_argument("--inits", nargs="+", default=["base", "scratch"], choices=["base", "scratch"])
    parser.add_argument("--skip-frame-stack", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "toy_recall.json")
    args = parser.parse_args(argv)
    if args.quick:
        args.pretrain_steps, args.steps, args.eval_gaps = 5, 5, [0, 32]
    torch.manual_seed(args.seed)
    task = RecallTask()
    results: dict = {"task": asdict(task), "args": {k: str(v) for k, v in vars(args).items()}, "models": {}}
    t0 = time.time()

    def report(name: str, model: ToyPolicy, extra: dict | None = None) -> None:
        evals = {str(g): evaluate(model, task, g, seed=1000 + g) for g in args.eval_gaps}
        results["models"][name] = {"eval": evals, **(extra or {})}
        row = "  ".join(
            f"gap {g}: react {e['reactive_acc']:.2f} recall {e['recall_acc']:.2f}" for g, e in evals.items()
        )
        print(f"[{time.time() - t0:6.1f}s] {name:<22} {row}", flush=True)

    # Stage 1: the pretrained short-context policy (no memory).
    base = ToyPolicy(task.n_features, task.n_bins)
    train(base, task, args.pretrain_steps, args.train_gap, None, seed=args.seed, demo_loss=True)
    report("base_no_memory", base)

    # Baseline: FLUX-sized 8-timestep frame stack, trained from scratch on the same budget.
    if not args.skip_frame_stack:
        window = ToyPolicy(task.n_features, task.n_bins, frame_stack=8)
        train(window, task, args.pretrain_steps + args.steps, args.train_gap, None, seed=args.seed + 1)
        report("frame_stack_8", window)

    # Stage 2: a gated TTT memory, either wrapped around the pretrained base (the FLUX situation: identity at
    # initialization, then fine-tuned with truncated BPTT) or trained from scratch on the same total budget.
    for kind in args.kinds:
        for init in args.inits:
            # Only the observation token enters the memory: with the two registers in each inner mini-batch the
            # toy learned several times slower (ablation in ws4/research/robottt_flux.md).
            model = ToyPolicy(task.n_features, task.n_bins, memory=kind, ttt_tokens=[0])
            extra: dict = {}
            steps = args.steps
            if init == "base":
                missing = model.load_state_dict(base.state_dict(), strict=False)
                assert not missing.unexpected_keys, missing.unexpected_keys
                extra["init_vs_base"] = init_deviation(model, base, task, seed=7)
                extra["eval_at_init"] = {str(g): evaluate(model, task, g, seed=1000 + g) for g in args.eval_gaps[:2]}
            else:
                steps = args.pretrain_steps + args.steps
            losses = train(
                model, task, steps, args.train_gap, args.segment, seed=args.seed + 2, gate_lr_mult=args.gate_lr_mult
            )
            extra.update(
                {
                    "train_steps": steps,
                    "final_train_loss": sum(losses[-20:]) / max(1, len(losses[-20:])),
                    "gate_tanh_alpha_mean_abs": float(torch.tanh(model.temporal.ttt.gate.alpha.detach()).abs().mean()),
                    "inner_lr_by_phase": inner_lr_by_phase(model, task, 64, seed=11),
                }
            )
            report(f"{init}+{kind}", model, extra)
            print(f"         { ({k: v for k, v in extra.items() if k != 'eval_at_init'}) }", flush=True)

    results["seconds"] = time.time() - t0
    results["chance_recall"] = 1.0 / task.n_bins
    if not args.quick:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return results


if __name__ == "__main__":
    main()
