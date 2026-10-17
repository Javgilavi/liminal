"""Start frames of the held-out takes as placement references for the post-training eval.

For each layout in the dataset's ``ws4_split.json``, writes the first frame of every held-out take (head and wrist
camera) as ``<layout>_ep<NN>_head.jpg`` / ``_wrist.jpg``, an ``overview.jpg`` contact sheet (one row per layout) and
``index.json`` (take, layout, length, start joints of the task arm). Reads a snapshot copy of the finalised
dataset, never the live recording root.

    python ws4/single_arm/make_layout_refs.py --root ws4/.cache/single_arm_diag/pen_final_raw \\
        --out ws4/single_arm/layouts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

KEYS = {"head": "observation.images.head", "wrist": "observation.images.right_wrist"}


def first_frames(repo_id: str, root: str, episode: int) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
    item = ds[0]
    if int(item["episode_index"]) != episode or int(item["frame_index"]) != 0:
        raise SystemExit(f"take {episode}: got episode {int(item['episode_index'])} frame {int(item['frame_index'])}")
    frames = {name: (item[key].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
              for name, key in KEYS.items()}  # fmt: skip
    return frames, item["observation.state"].numpy(), len(ds)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True, help="snapshot of the finalised dataset (with ws4_split.json)")
    p.add_argument("--repo-id", default="local/pen_final")
    p.add_argument("--out", default=str(HERE / "layouts"))
    args = p.parse_args()
    import cv2
    from flux.package_tools import arm_columns

    split = json.loads((Path(args.root) / "ws4_split.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, rows = {"dataset": split["dataset"], "task": split["task"], "split_updated": split["updated"],
                   "takes": []}, []  # fmt: skip
    for layout, takes in split["heldout"].items():
        row = []
        for ep in takes:
            frames, state, length = first_frames(args.repo_id, args.root, ep)
            for name, img in frames.items():
                cv2.imwrite(str(out / f"{layout}_ep{ep:03d}_{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])  # fmt: skip
            task_arm = arm_columns(state[None], "right")[0] if state.shape[-1] == 12 else state
            index["takes"].append({"layout": layout, "episode": ep, "frames": length,
                                   "start_joints_task_arm": [round(float(v), 1) for v in task_arm]})  # fmt: skip
            thumb = cv2.resize(frames["head"], (320, 240))
            cv2.putText(thumb, f"{layout}  take {ep}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            row.append(thumb)
            print(f"{layout} take {ep}: {length} frames, task arm start {np.round(task_arm, 1)}", flush=True)
        rows.append(np.concatenate(row, axis=1))
    width = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows], axis=0)
    cv2.imwrite(str(out / "overview.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index['takes'])} takes to {out}")


if __name__ == "__main__":
    main()
