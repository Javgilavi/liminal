"""Held-out action error per layout (A, B, C, D) from offline_diag.py / offline_eval.py ``.npz`` outputs.

Works on both: every ``pred_<name>`` array of each file is scored against its ``true`` / ``anchor`` on the moving
frames (offline_diag.moving_frames), grouped by the layout of each frame's episode (ws4_split.json "layouts").

    python ws4/act/by_layout.py --root /workspace/real_lora/pen_in_right_final \\
        --split /workspace/real_lora/ws4_split_final.json /workspace/act/eval/act_pen84_100000.npz LABEL=FILE.npz ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WS4 = Path(__file__).resolve().parents[1]
for path in (WS4, WS4 / "single_arm"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("npz", nargs="+", help="FILE.npz or LABEL=FILE.npz")
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default="pen")
    p.add_argument("--split", required=True)
    args = p.parse_args()
    from flux.package_tools import dataset_arrays
    from offline_diag import moving_frames

    episode_index = dataset_arrays(args.repo_id, root=args.root)["episode_index"]
    split = json.loads(Path(args.split).read_text())
    groups = split.get("layouts") or split["heldout"]  # skill splits may only name the held-out groups
    layout_of = {int(e): name for name, eps in groups.items() for e in eps}
    print("| model | layout | moving frames | arm MAE pan, lift, elbow, wflex, roll (deg), mean | gripper MAE (%) |")
    print("|---|---|---|---|---|")
    for item in args.npz:
        label, _, path = item.rpartition("=")
        data = np.load(path)
        true, anchor = data["true"], data["anchor"]
        layouts = np.array([layout_of[int(episode_index[f])] for f in data["frames"]])
        moving = moving_frames(true, anchor)
        for key in sorted(k for k in data.files if k.startswith("pred_")):
            name = label or f"{Path(path).stem}:{key.removeprefix('pred_')}"
            for lay in sorted(set(layouts)):
                sel = moving & (layouts == lay)
                mae = np.abs(data[key][sel] - true[sel]).mean(axis=(0, 1))
                print(f"| {name} | {lay} | {int(sel.sum())} | {', '.join(f'{v:.1f}' for v in mae[:5])} "
                      f"({mae[:5].mean():.1f}) | {mae[5]:.1f} |")  # fmt: skip


if __name__ == "__main__":
    main()
