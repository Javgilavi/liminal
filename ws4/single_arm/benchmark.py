"""Red-pen benchmark of 18 Oct: one row per block from ws4/eval/trials.jsonl plus each attempt's meta.json and
trace.npz (the run folders under ws4/single_arm/runs/). Writes benchmarks/red_pen_2026-10-18.{md,json}.

    python ws4/single_arm/benchmark.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "benchmarks"
STEP_NAMES = ("reach", "grasp", "insert")


def _notes(t: dict) -> str:
    return t.get("notes", "")


def _is(t: dict, variant: str, task: str = "put_pen_in_slot") -> bool:
    return t.get("variant") == variant and t.get("task") == task


def _pen84(t: dict) -> bool:  # the graded session on the rebuilt pen84 table (run folders from 04:48 on)
    return "/runs/20261018_0" in t.get("video", "") and t["video"].split("/runs/")[1][:13] >= "20261018_0448"


BLOCKS = [
    ("FLUX RAW 10k, lid-held scene (P3), horizon 32", "lid held up by the box arm; pen84 LoRA never saw it",
     lambda t: _is(t, "LoRA-10k-raw", "put_pen_in_slot_lid_held")),
    ("ACT pen84 100k, pen84 table", "ACT session, box-arm tolerance 15",
     lambda t: _is(t, "ACT-pen84-100k")),
    ("FLUX EMA 10k, pen84 table, horizon 32", "stall 8 deg on every joint",
     lambda t: _is(t, "LoRA-10k-ema") and _pen84(t)),
    ("FLUX RAW 10k, pen84 table, horizon 32", "stall 8 deg on every joint",
     lambda t: _is(t, "LoRA-10k-raw") and _pen84(t)),
    ("FLUX RAW 5k, pen84 table, horizon 32", "stall 8 deg on every joint",
     lambda t: _is(t, "LoRA-5k-raw") and _pen84(t) and "lift stall" not in _notes(t)),
    ("FLUX RAW 5k, horizon 32, lift stall 25", "one attempt (load freeze), then horizon 16",
     lambda t: _is(t, "LoRA-5k-raw") and "lift stall 25" in _notes(t) and "action horizon" not in _notes(t)),
    ("FLUX RAW 5k, horizon 16, lift stall 25",
     "A B C in one session, D A resumed (START_AT=4); the successes ended with the stop key after the insert, "
     "D and A were stopped by Oscar at 26 s and 43 s without a reach",
     lambda t: _is(t, "LoRA-5k-raw") and "action horizon 16 per" in _notes(t)),
    ("FLUX RAW 5k, horizon 8, lift stall 25", "layouts A B (operator stopped both, session ended)",
     lambda t: _is(t, "LoRA-5k-raw") and "action horizon 8 per" in _notes(t)),
]


def end_kind(reason: str) -> str:
    if reason.startswith("timeout"):
        return "timeout"
    for key, kind in (("stall", "stall freeze"), ("load", "load freeze"), ("keep-out", "keep-out stop"),
                      ("operator: success", "success key")):  # fmt: skip
        if key in reason:
            return kind
    return "operator stop" if reason.startswith("stop (") else reason


def attempt_row(t: dict) -> dict:
    folder = REPO / Path(t["video"]).parent
    meta = json.loads((folder / "meta.json").read_text())
    tr = np.load(folder / "trace.npz", allow_pickle=True)
    s, ts = tr["state"], tr["t"]
    early = s[ts <= 3.0]
    shut = next((float(ts[k]) for k in range(len(s)) if s[k, 5] < 20), None)
    return {"attempt": t.get("attempt"), "layout": meta.get("layout", ""), "steps_done": int(t["steps_done"]),
            "interventions": int(t.get("interventions") or 0), "end": meta["end_reason"], "end_kind":
            end_kind(meta["end_reason"]), "duration_s": round(float(meta["duration_s"]), 1),
            "moved_5deg_in_3s": bool(len(early) and np.abs(early[:, :5] - s[0, :5]).max() > 5),
            "gripper_shut_s": None if shut is None else round(shut, 1),
            "keepout_refusals": {k: v for k, v in meta.get("keepout_refusals", {}).items() if v},
            "chunks": meta.get("chunks"), "run": str(folder.relative_to(REPO))}  # fmt: skip


def summary(name: str, note: str, rows: list[dict]) -> str:
    n = len(rows)
    steps = [r["steps_done"] for r in rows]
    fails = Counter(STEP_NAMES[s] for s in steps if s < 3)
    ends = Counter(r["end_kind"] for r in rows)
    wins = [r["duration_s"] for r in rows if r["steps_done"] == 3]
    shut = [r["gripper_shut_s"] for r in rows if r["gripper_shut_s"] is not None]
    cells = [name, str(n), f"**{sum(s == 3 for s in steps)}/{n}**",
             f"{sum(steps)}/{3 * n} ({', '.join(map(str, steps))})",
             f"{sum(s >= 1 for s in steps)} / {sum(s >= 2 for s in steps)} / {sum(s >= 3 for s in steps)}",
             ", ".join(f"{k} {v}" for k, v in fails.most_common()) or "-",
             str(sum(r["interventions"] for r in rows)),
             ", ".join(f"{k} {v}" for k, v in ends.most_common()),
             f"{sum(r['moved_5deg_in_3s'] for r in rows)}/{n}, gripper shut "
             f"{round(float(np.median(shut)), 1) if shut else '-'} s",
             f"{round(float(np.median(wins)))} s" if wins else "-", note]
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    trials = [json.loads(line) for line in (REPO / "ws4/eval/trials.jsonl").read_text().splitlines() if line.strip()]
    out = {}
    lines = ["| Block | Attempts | Success | Steps done | Reach / grasp / insert | Failure point | Interventions "
             "(as scored) | How attempts ended | Moved > 5 deg in 3 s; median gripper close | Median time to success "
             "| Notes |", "|" + "---|" * 11]
    for name, note, pred in BLOCKS:
        rows = [attempt_row(t) for t in trials if pred(t)]
        out[name] = {"note": note, "attempts": rows}
        if rows:
            lines.append(summary(name, note, rows))
    OUT.mkdir(exist_ok=True)
    (OUT / "red_pen_2026-10-18.json").write_text(json.dumps(out, indent=1))
    table = "\n".join(lines)
    (OUT / "red_pen_table_2026-10-18.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
