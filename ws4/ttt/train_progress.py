"""Train and evaluate the P2 progress memory on labelled episodes (CPU).

    python ws4/ttt/train_progress.py --data ws4/.cache/ttt/sim --features frame \\
        --memories none lastk gru ttt --out ws4/ttt/results/progress_sim_frame.json --save-dir ws4/.cache/ttt/models

``--data`` takes one or more directories of episode directories (e.g. the run-to-end ``sim`` set plus the
check-ended sets written by ``flux_sim_closed_loop.py --stand-in-stops-on-check``); they are pooled. Splits
episodes 80/20 by a hash of their name (stable across runs), trains each memory type with the same network,
budget and seed, and reports on the held-out episodes:

* ``stage_acc``: the plan step the model puts first, using the known plan (``features.stage_posterior``);
* ``stage_acc_repeat``: the same on re-plans whose skill already ran earlier in the episode (aliased stages);
* ``occurrence_acc``, ``skill_acc``; ``progress_mae`` (overall progress, 0..1); ``stage_progress_mae``;
* ``done_auroc`` and ``done_f1`` at 0.5 for the "current stage finishes before the next re-plan" flag.

Each trained model is saved with everything ``progress_api.ProgressMonitor`` needs (config, skills, feature
source and featurizer settings, normalization stats, weights).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ttt.features import MAX_OCCURRENCE, FrameFeaturizer, Sequence, episode_sequence, stage_posterior  # noqa: E402
from ttt.progress_memory import EpisodeProgressMemory  # noqa: E402
from ttt.temporal import tbptt_slices  # noqa: E402
from ttt.ttt_layer import detach_state  # noqa: E402

MEMORIES = ("none", "lastk", "gru", "ttt")


def is_test(name: str, frac: float = 0.2) -> bool:
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % 1000 < frac * 1000


def load_sequences(
    data: Path | list[Path], source: str, skills: list[str] | None = None, flux_subset: bool = False, flux_parts=None
) -> tuple[list[Sequence], list[str]]:
    roots = [data] if isinstance(data, Path) else list(data)
    dirs = sorted(
        {p.parent for root in roots for p in root.glob("*/episode.json") if (p.parent / "frames.npz").exists()}
    )
    if source != "frame" or flux_subset:
        dirs = [d for d in dirs if (d / "features_flux.npz").exists()]
    if not dirs:
        raise FileNotFoundError(f"no episodes with {source} features under {[str(r) for r in roots]}")
    if skills is None:
        found = set()
        for d in dirs:
            found.update(s["skill"] for s in json.loads((d / "episode.json").read_text())["steps"])
        skills = sorted(found)
    feat = FrameFeaturizer()
    return [episode_sequence(d, skills, source, feat, flux_parts) for d in dirs], skills


def pad(seqs: list[Sequence]) -> dict[str, torch.Tensor]:
    n, t, d = len(seqs), max(len(s.feat) for s in seqs), seqs[0].feat.shape[1]
    out = {
        "feat": torch.zeros(n, t, d),
        "skill": torch.full((n, t), -1, dtype=torch.long),
        "occurrence": torch.zeros(n, t, dtype=torch.long),
        "stage_progress": torch.zeros(n, t),
        "done": torch.zeros(n, t),
        "valid": torch.zeros(n, t, dtype=torch.bool),
    }
    for i, s in enumerate(seqs):
        r = len(s.feat)
        out["feat"][i, :r] = torch.from_numpy(s.feat)
        out["skill"][i, :r] = torch.from_numpy(s.skill)
        out["occurrence"][i, :r] = torch.from_numpy(s.occurrence)
        out["stage_progress"][i, :r] = torch.from_numpy(s.stage_progress).float()
        out["done"][i, :r] = torch.from_numpy(s.done).float()
        out["valid"][i, :r] = torch.from_numpy(s.skill >= 0)
    return out


def fit_progress(
    model: EpisodeProgressMemory,
    data: dict[str, torch.Tensor],
    steps: int = 600,
    batch: int = 16,
    segment: int = 32,
    lr: float = 2e-3,
    seed: int = 0,
) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    losses = []
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, data["feat"].shape[0], (batch,), generator=gen)
        b = {k: v[idx] for k, v in data.items()}
        denom = b["valid"].sum().clamp_min(1)
        opt.zero_grad()
        state, total = None, 0.0
        for sl in tbptt_slices(b["feat"].shape[1], segment):
            out, state = model(b["feat"][:, sl], state)
            v = b["valid"][:, sl].flatten().float()
            loss = F.cross_entropy(
                out["subtask"].flatten(0, 1), b["skill"][:, sl].clamp_min(0).flatten(), reduction="none"
            )
            loss = loss + F.binary_cross_entropy_with_logits(
                out["done"].flatten(), b["done"][:, sl].flatten(), reduction="none"
            )
            if "occurrence" in out:
                loss = loss + F.cross_entropy(
                    out["occurrence"].flatten(0, 1), b["occurrence"][:, sl].flatten(), reduction="none"
                )
            if "stage_progress" in out:
                loss = loss + 4.0 * (out["stage_progress"].flatten() - b["stage_progress"][:, sl].flatten()) ** 2
            loss = (loss * v).sum() / denom
            loss.backward()
            total += float(loss.detach())
            state = detach_state(state)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(total)
    return losses


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def evaluate(model: EpisodeProgressMemory, seqs: list[Sequence], skills: list[str]) -> dict:
    model.eval()
    rows = {
        k: []
        for k in (
            "stage",
            "stage_hat",
            "occ",
            "occ_hat",
            "skill",
            "skill_hat",
            "rep",
            "prog",
            "prog_hat",
            "sp",
            "sp_hat",
            "done",
            "done_p",
            "task",
        )
    }
    for s in seqs:
        out, _ = model(torch.from_numpy(s.feat)[None])
        sk = torch.softmax(out["subtask"][0], -1).numpy()
        oc = torch.softmax(out["occurrence"][0], -1).numpy()
        sp = out["stage_progress"][0].numpy()
        post = stage_posterior(sk, oc, s.plan_skills, skills)
        stage_hat = post.argmax(-1)
        valid = s.stage >= 0
        n = len(s.plan_skills)
        rows["stage"].append(s.stage[valid])
        rows["stage_hat"].append(stage_hat[valid])
        rows["occ"].append(s.occurrence[valid])
        rows["occ_hat"].append(oc.argmax(-1)[valid])
        rows["skill"].append(s.skill[valid])
        rows["skill_hat"].append(sk.argmax(-1)[valid])
        rows["rep"].append(s.occurrence[valid] > 0)
        rows["prog"].append(s.progress[valid])
        rows["prog_hat"].append(((stage_hat + sp) / n)[valid])
        rows["sp"].append(s.stage_progress[valid])
        rows["sp_hat"].append(sp[valid])
        rows["done"].append(s.done[valid])
        rows["done_p"].append(torch.sigmoid(out["done"][0]).numpy()[valid])
        rows["task"].append(np.array([s.task] * int(valid.sum())))
    r = {k: np.concatenate(v) for k, v in rows.items()}

    def metrics(m: np.ndarray) -> dict:
        d_hat = r["done_p"][m] > 0.5
        d = r["done"][m] == 1
        tp = float((d_hat & d).sum())
        f1 = 2 * tp / max(1.0, float(d_hat.sum() + d.sum()))
        rep = m & r["rep"]
        return {
            "n": int(m.sum()),
            "stage_acc": float((r["stage"][m] == r["stage_hat"][m]).mean()),
            "stage_acc_repeat": float((r["stage"][rep] == r["stage_hat"][rep]).mean()) if rep.any() else float("nan"),
            "occurrence_acc": float((r["occ"][m] == r["occ_hat"][m]).mean()),
            "skill_acc": float((r["skill"][m] == r["skill_hat"][m]).mean()),
            "progress_mae": float(np.abs(r["prog"][m] - r["prog_hat"][m]).mean()),
            "stage_progress_mae": float(np.abs(r["sp"][m] - r["sp_hat"][m]).mean()),
            "done_auroc": _auroc(r["done_p"][m], r["done"][m]),
            "done_f1": f1,
        }

    out = {"all": metrics(np.ones(len(r["stage"]), dtype=bool))}
    for task in sorted(set(r["task"])):
        out[task] = metrics(r["task"] == task)
    return out


def main(argv: list[str] | None = None) -> dict:
    from ttt.throttle import throttle

    throttle()  # shares the laptop with 30 Hz robot loops
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--data", type=Path, nargs="+", default=[Path(__file__).resolve().parents[1] / ".cache" / "ttt" / "sim"]
    )
    ap.add_argument("--features", default="frame", choices=["frame", "flux", "flux+frame"])
    ap.add_argument("--flux-subset", action="store_true", help="only episodes that have FLUX features (any source)")
    ap.add_argument("--flux-parts", nargs="+", default=None, help='FLUX taps to keep, e.g. "27:imagined"')
    ap.add_argument("--tag", default="", help="suffix for saved model names")
    ap.add_argument("--memories", nargs="+", default=list(MEMORIES), choices=MEMORIES)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    seqs, skills = load_sequences(args.data, args.features, flux_subset=args.flux_subset, flux_parts=args.flux_parts)
    train = [s for s in seqs if not is_test(s.episode)]
    test = [s for s in seqs if is_test(s.episode)]
    data = pad(train)
    feats = torch.cat([torch.from_numpy(s.feat) for s in train])
    mean, std = feats.mean(0), feats.std(0)
    print(
        f"{len(train)} train / {len(test)} test episodes, {feats.shape[0]} train re-plans, "
        f"feature dim {feats.shape[1]}, skills {skills} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    results = {
        "data": [str(d) for d in args.data],
        "features": args.features,
        "skills": skills,
        "n_train": len(train),
        "n_test": len(test),
        "train_replans": int(feats.shape[0]),
        "steps": args.steps,
        "models": {},
    }
    for memory in args.memories:
        for seed in args.seeds:
            torch.manual_seed(seed)
            cfg = {
                "feat_dim": int(feats.shape[1]),
                "n_subtasks": len(skills),
                "dim": args.dim,
                "memory": memory,
                "n_occurrences": MAX_OCCURRENCE,
                "progress_head": True,
            }
            model = EpisodeProgressMemory(**cfg)
            model.set_normalization(mean, std)
            fit_progress(model, data, steps=args.steps, seed=seed)
            ev = evaluate(model, test, skills)
            results["models"].setdefault(memory, {})[str(seed)] = ev
            a = ev["all"]
            print(
                f"[{time.time() - t0:5.0f}s] {memory:<6} seed {seed}: stage {a['stage_acc']:.3f} "
                f"(repeat {a['stage_acc_repeat']:.3f}) occ {a['occurrence_acc']:.3f} prog_mae {a['progress_mae']:.3f} "
                f"done_auroc {a['done_auroc']:.3f}",
                flush=True,
            )
            if args.save_dir is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "config": cfg,
                        "skills": skills,
                        "features": args.features,
                        "featurizer": FrameFeaturizer().config() if "frame" in args.features else None,
                        "flux_parts": args.flux_parts,
                        "state_dict": model.state_dict(),
                        "eval": ev,
                        "max_occurrence": MAX_OCCURRENCE,
                    },
                    args.save_dir / f"progress_{args.features.replace('+', '_')}{args.tag}_{memory}_s{seed}.pt",
                )
    results["seconds"] = round(time.time() - t0, 1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
