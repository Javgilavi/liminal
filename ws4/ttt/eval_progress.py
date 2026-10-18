"""Evaluate saved progress monitors on held-out episodes of several sets, per task family (CPU).

    python ws4/ttt/eval_progress.py \\
        --models ws4/ttt/models/progress_frame_gru_s0.pt ws4/ttt/models/progress_frame_checkend_gru_s0.pt \\
        --set run_to_end=ws4/.cache/ttt/sim --set check_ended='ws4/.cache/ttt/sim_stepend/check/*' \\
        --out ws4/ttt/results/progress_step_end.json

Each ``--set name=glob`` pools the episode directories under the matching data directories. Only held-out
episodes count (``train_progress.is_test``, the same hash split the training used), unless ``--all``. Episodes
written by ``flux_sim_closed_loop.py`` carry the task ``flux_<skill>+<skill>...``; they are reported under the
``sim_episodes.py`` family with the same plan (``lid_cycles_1`` is the closed loop's default open/close plan).
Metrics are ``train_progress.evaluate``'s.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ttt.sim_episodes import _task_steps  # noqa: E402

FAMILIES = ("lid_cycles_1", "lid_cycles_2", "lid_cycles_3", "lid_cycles_4", "pack_lidded", "pack")


def family(plan_skills: list[str], task: str) -> str:
    """The ``sim_episodes.py`` task family with this plan, else the episode's own task name."""
    for name in FAMILIES:
        if [s for s, _ in _task_steps(name)[1]] == list(plan_skills):
            return name
    return task


def load_set(pattern: str, skills: list[str], held_out: bool = True) -> list:
    from ttt.train_progress import is_test, load_sequences

    roots = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
    if not roots:
        raise FileNotFoundError(f"no data directories match {pattern!r}")
    seqs, _ = load_sequences(roots, "frame", skills=skills)
    for s in seqs:
        s.task = family(s.plan_skills, s.task)
    return [s for s in seqs if is_test(s.episode)] if held_out else seqs


def load_model(path: Path):
    import torch

    from ttt.progress_memory import EpisodeProgressMemory

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["features"] != "frame":
        raise ValueError(f"{path}: only frame-feature models are supported here")
    model = EpisodeProgressMemory(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["skills"]


def main(argv: list[str] | None = None) -> dict:
    from ttt.throttle import throttle

    throttle()  # shares the laptop with 30 Hz robot loops
    from ttt.train_progress import evaluate

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--set", dest="sets", action="append", required=True, help="name=glob of data directories")
    ap.add_argument("--all", action="store_true", help="every episode, not only the held-out split")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    sets = dict(s.split("=", 1) for s in args.sets)
    results: dict = {"held_out_only": not args.all, "sets": {}, "models": {}}
    cache: dict = {}
    for path in args.models:
        model, skills = load_model(path)
        res = results["models"].setdefault(path.name, {})
        for name, pattern in sets.items():
            key = (name, tuple(skills))
            if key not in cache:
                cache[key] = load_set(pattern, skills, held_out=not args.all)
                results["sets"][name] = {"pattern": pattern, "episodes": len(cache[key])}
            res[name] = evaluate(model, cache[key], skills)
            a = res[name]["all"]
            print(
                f"[{time.time() - t0:5.0f}s] {path.name:<40} {name:<14} n {a['n']:5d}  stage {a['stage_acc']:.3f} "
                f"(repeat {a['stage_acc_repeat']:.3f}) prog_mae {a['progress_mae']:.3f} "
                f"done_auroc {a['done_auroc']:.3f}",
                flush=True,
            )
    results["seconds"] = round(time.time() - t0, 1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    main()
