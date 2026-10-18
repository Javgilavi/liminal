"""Markdown rows for pen_lora.md from offline_eval.py JSONs (one per checkpoint), in the FLUX curve's columns.

python ws4/act/curve_table.py /workspace/act/eval/act_pen84_*.json [--loss /workspace/act/logs/act_pen84.log]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def pct(values) -> str:
    return ", ".join(f"{100 * v:.0f}" for v in values) + " %"


def train_l1(log: Path) -> dict[int, float]:
    """Train L1 near each checkpoint: mean ``l1_loss`` of lerobot_train's log lines whose step label (rounded to
    1K, e.g. "20K") equals the checkpoint's."""
    out: dict[int, list[float]] = {}
    steps = re.findall(r"step:([\d.]+)(K?) smpl.*?l1_loss:([\d.]+)", log.read_text(errors="replace"))
    for value, k, l1 in steps:
        out.setdefault(round(float(value) * (1000 if k else 1)), []).append(float(l1))
    return {s: float(np.mean(v)) for s, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("json", nargs="+")
    p.add_argument("--loss", default=None, help="lerobot_train log, for the train L1 column")
    args = p.parse_args()
    losses = train_l1(Path(args.loss)) if args.loss else {}
    rows = []
    for path in sorted(args.json):
        result = json.loads(Path(path).read_text())
        for m in result["variants"].values():
            step = int(re.search(r"/(\d{6})/", m["checkpoint"]).group(1))
            mae = m["moving_mae"]
            loss = f"{losses[step]:.3f}" if step in losses else "-"
            arm = f"{', '.join(f'{v:.1f}' for v in mae[:5])} ({np.mean(mae[:5]):.1f})"
            rows.append((step, f"| ACT | {step // 1000}k | {arm}; {mae[5]:.1f} | "
                               f"{pct(m['moving_direction_agreement'])} | {pct(m['moving_rel_disp_error_median'])} | "
                               f"{loss} | {1e3 * m['latency_s']:.0f} ms |"))  # fmt: skip
    print("\n".join(r for _, r in sorted(rows)))


if __name__ == "__main__":
    main()
