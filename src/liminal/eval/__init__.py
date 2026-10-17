"""V6 evaluation: compare harness configurations on the same scenarios under matched budgets.

    liminal eval configs/eval/smoke.yaml [--out runs/eval/smoke]

Replay/mock results only; every report is labeled "REPLAY/MOCK, not hardware".
"""

from __future__ import annotations

import sys
from pathlib import Path

from liminal.eval.report import render_report
from liminal.eval.runner import ExperimentResult, run_experiment
from liminal.eval.spec import Experiment, ExperimentError, UnmatchedBudgets, load_experiment

__all__ = ["Experiment", "ExperimentError", "ExperimentResult", "UnmatchedBudgets", "eval_main",
           "load_experiment", "render_report", "run_experiment"]


def eval_main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="liminal eval", description="Compare harness configurations (replay/mock).")
    p.add_argument("experiment", help="experiment YAML, e.g. configs/eval/smoke.yaml")
    p.add_argument("--out", help="output directory (default runs/eval/<experiment name>)")
    args = p.parse_args(argv)
    try:
        exp = load_experiment(args.experiment)
        out = Path(args.out) if args.out else Path("runs/eval") / exp.name
        result = run_experiment(args.experiment, out)
    except (ExperimentError, OSError) as e:
        print(f"cannot run experiment: {e}", file=sys.stderr)
        return 2
    print(Path(result.report_path).read_text(encoding="utf-8"))
    print(f"results: {result.results_path}\nreport: {result.report_path}")
    return 0
