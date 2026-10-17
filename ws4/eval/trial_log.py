"""Trial log for the graded evaluation: zero-shot vs post-training on the same task.

The BFL track brief grades both the released checkpoint zero-shot and the fine-tuned model, and asks teams to
keep "successful attempts, total attempts and completed task steps", what failed, and whether a human had to
intervene. One JSON line per attempt keeps that record honest and turns into the slide table in one command.

    python ws4/eval/trial_log.py add --task pack_box --condition zero_shot --variant A0 \\
        --checkpoint black-forest-labs/flux-3-action-so101@<rev> --layout L1 --steps-total 5 --steps-done 2 \\
        --failure "grasp miss on apple" --interventions 0
    python ws4/eval/trial_log.py summary --markdown
    python ws4/eval/trial_log.py list -n 10
    python ws4/eval/trial_log.py undo --yes      # drop the last line (kept in trials.removed.jsonl)

Stdlib only, so it runs on the robot PC, the laptop and the GPU box alike.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parent / "trials.jsonl"
CONDITIONS = ("zero_shot", "post_training")


@dataclass
class Trial:
    task: str
    condition: str  # zero_shot | post_training
    variant: str  # A0, A1, lora_A, lora_B, lora_P, ...
    checkpoint: str  # repo id or path, with revision
    steps_total: int
    steps_done: int
    success: bool
    interventions: int = 0
    layout: str = ""
    failure: str = ""
    notes: str = ""
    video: str = ""
    operator: str = ""
    attempt: int = 0  # 1-based per (task, variant), assigned on add
    time: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}, got {self.condition!r}")
        if not 0 <= self.steps_done <= self.steps_total:
            raise ValueError(f"steps_done must be in 0..steps_total, got {self.steps_done}/{self.steps_total}")
        if self.interventions < 0:
            raise ValueError("interventions must be >= 0")


def load(path: Path) -> list[Trial]:
    if not path.exists():
        return []
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                out.append(Trial(**json.loads(line)))
            except (TypeError, ValueError) as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    return out


def undo_last(path: Path) -> Trial | None:
    """Remove the last attempt and keep it in `<log>.removed.jsonl`, so a mistake never vanishes silently."""
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()] if path.exists() else []
    if not lines:
        return None
    last = Trial(**json.loads(lines[-1]))
    removed = path.with_name(path.stem + ".removed.jsonl")
    with removed.open("a") as f:
        f.write(lines[-1] + "\n")
    path.write_text("".join(ln + "\n" for ln in lines[:-1]))
    return last


def append(path: Path, trial: Trial) -> Trial:
    prior = [t for t in load(path) if (t.task, t.variant) == (trial.task, trial.variant)]
    trial.attempt = len(prior) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(trial)) + "\n")
    return trial


@dataclass
class Row:
    task: str
    condition: str
    variant: str
    checkpoints: list[str]
    attempts: int
    successes: int
    mean_steps_done: float
    steps_total: int
    with_intervention: int
    top_failures: list[tuple[str, int]]

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts else 0.0


def summarize(trials: list[Trial]) -> list[Row]:
    groups: dict[tuple[str, str, str], list[Trial]] = defaultdict(list)
    for t in trials:
        groups[(t.task, t.condition, t.variant)].append(t)
    rows = []
    # zero_shot before post_training within a task, so the table reads as before/after.
    def order(kv: tuple[tuple[str, str, str], list[Trial]]) -> tuple[str, int, str]:
        (task, cond, var), _ = kv
        return task, CONDITIONS.index(cond), var

    for (task, cond, var), ts in sorted(groups.items(), key=order):
        fails = Counter(t.failure for t in ts if t.failure and not t.success)
        rows.append(
            Row(
                task=task,
                condition=cond,
                variant=var,
                checkpoints=sorted({t.checkpoint for t in ts}),
                attempts=len(ts),
                successes=sum(t.success for t in ts),
                mean_steps_done=sum(t.steps_done for t in ts) / len(ts),
                steps_total=max(t.steps_total for t in ts),
                with_intervention=sum(t.interventions > 0 for t in ts),
                top_failures=fails.most_common(3),
            )
        )
    return rows


def render(rows: list[Row], markdown: bool) -> str:
    head = ["task", "condition", "variant", "success", "steps done (mean)", "human help", "top failures", "checkpoint"]
    body = [
        [
            r.task,
            r.condition,
            r.variant,
            f"{r.successes}/{r.attempts} ({r.success_rate:.0%})",
            f"{r.mean_steps_done:.1f}/{r.steps_total}",
            f"{r.with_intervention}/{r.attempts}",
            "; ".join(f"{k} x{v}" for k, v in r.top_failures) or "-",
            ", ".join(r.checkpoints),
        ]
        for r in rows
    ]
    if markdown:
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(c.replace("|", "/") for c in row) + " |" for row in body]
        return "\n".join(lines)
    widths = [max(len(x) for x in col) for col in zip(head, *body, strict=True)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join(fmt.format(*row) for row in [head, *body])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG, help=f"JSONL file (default {DEFAULT_LOG})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="record one attempt")
    a.add_argument("--task", required=True)
    a.add_argument("--condition", required=True, choices=CONDITIONS)
    a.add_argument("--variant", required=True, help="A0, A1, lora_A, lora_B, lora_P, ...")
    a.add_argument("--checkpoint", required=True, help="repo id or path, include the revision")
    a.add_argument("--steps-total", type=int, required=True)
    a.add_argument("--steps-done", type=int, required=True)
    a.add_argument("--success", choices=("auto", "yes", "no"), default="auto",
                   help="auto = all steps done and no intervention")
    a.add_argument("--interventions", type=int, default=0, help="times a human had to step in")
    a.add_argument("--layout", default="", help="start layout id, e.g. L1")
    a.add_argument("--failure", default="", help="short failure reason, reused verbatim so they group")
    a.add_argument("--notes", default="")
    a.add_argument("--video", default="")
    a.add_argument("--operator", default="")

    s = sub.add_parser("summary", help="per task / condition / variant table")
    s.add_argument("--markdown", action="store_true")
    s.add_argument("--task", default=None)

    lst = sub.add_parser("list", help="last N attempts")
    lst.add_argument("-n", type=int, default=10)

    u = sub.add_parser("undo", help="remove the last attempt (dry run unless --yes)")
    u.add_argument("--yes", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "add":
        success = {"yes": True, "no": False}.get(
            args.success, args.steps_done == args.steps_total and args.interventions == 0
        )
        t = append(
            args.log,
            Trial(
                task=args.task, condition=args.condition, variant=args.variant, checkpoint=args.checkpoint,
                steps_total=args.steps_total, steps_done=args.steps_done, success=success,
                interventions=args.interventions, layout=args.layout, failure=args.failure, notes=args.notes,
                video=args.video, operator=args.operator,
            ),
        )
        print(f"logged {t.task}/{t.variant} attempt {t.attempt}: {t.steps_done}/{t.steps_total} steps, "
              f"{'success' if t.success else 'fail'} -> {args.log}")
    elif args.cmd == "summary":
        trials = [t for t in load(args.log) if args.task in (None, t.task)]
        if not trials:
            print(f"no attempts in {args.log}")
            return 1
        print(render(summarize(trials), args.markdown))
    elif args.cmd == "undo":
        trials = load(args.log)
        if not trials:
            print(f"no attempts in {args.log}")
            return 1
        t = trials[-1]
        desc = f"{t.time} {t.task}/{t.variant} #{t.attempt} {t.steps_done}/{t.steps_total} {t.failure}".rstrip()
        if not args.yes:
            print(f"would remove: {desc}  (re-run with --yes)")
            return 0
        undo_last(args.log)
        print(f"removed: {desc}  (kept in {args.log.with_name(args.log.stem + '.removed.jsonl')})")
    else:
        for t in load(args.log)[-args.n:]:
            print(f"{t.time}  {t.task}/{t.variant} #{t.attempt} [{t.layout}] {t.steps_done}/{t.steps_total} "
                  f"{'OK' if t.success else 'FAIL'} help={t.interventions} {t.failure}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
