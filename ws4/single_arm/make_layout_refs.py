"""Start frames of the held-out takes as placement and home-check references for the real-arm evals.

For each layout of the held-out split, writes the first frame of every held-out take (head and the task arm's wrist
camera) as ``<layout>_ep<NNN>_head.jpg`` / ``_wrist.jpg``, an ``overview.jpg`` contact sheet (one row per layout) and
``index.json`` (take, layout, length, start joints of the task arm). Reads a snapshot copy of a finalised dataset,
never a live recording root. ``build_skill.py`` calls ``write_refs`` for each skill.

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


def first_frames(repo_id: str, root: str, episode: int, keys: dict[str, str]) -> tuple[dict, np.ndarray, int]:
    """Frame 0 of one take for each camera in ``keys`` (name -> dataset key; a key the dataset lacks is skipped)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
    item = ds[0]
    if int(item["episode_index"]) != episode or int(item["frame_index"]) != 0:
        raise SystemExit(f"take {episode}: got episode {int(item['episode_index'])} frame {int(item['frame_index'])}")
    frames = {name: (item[key].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
              for name, key in keys.items() if key in item}  # fmt: skip
    return frames, item["observation.state"].numpy(), len(ds)


def write_refs(root: str, heldout: dict[str, list[int]], out: Path, *, repo_id: str = "local/refs",
               wrist_key: str = "observation.images.right_wrist", columns: str = "right",
               meta: dict | None = None) -> dict:  # fmt: skip
    """Reference frames of the held-out takes into ``out``; returns the index (also written as index.json)."""
    import cv2
    from flux.package_tools import arm_columns

    out.mkdir(parents=True, exist_ok=True)
    keys = {"head": "observation.images.head", "wrist": wrist_key}
    index, rows = {**(meta or {}), "wrist_key": wrist_key, "columns": columns, "takes": []}, []
    for layout, takes in heldout.items():
        row = []
        for ep in takes:
            frames, state, length = first_frames(repo_id, root, ep, keys)
            for name, img in frames.items():
                cv2.imwrite(str(out / f"{layout}_ep{ep:03d}_{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])  # fmt: skip
            task_arm = arm_columns(state[None], columns)[0] if state.shape[-1] == 12 else state
            index["takes"].append({"layout": layout, "episode": ep, "frames": length, "cameras": sorted(frames),
                                   "start_joints_task_arm": [round(float(v), 1) for v in task_arm]})  # fmt: skip
            thumb = cv2.resize(frames["head"], (320, 240))
            cv2.putText(thumb, f"{layout} take {ep}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            row.append(thumb)
            print(f"{layout} take {ep}: {length} frames, task arm start {np.round(task_arm, 1)}, "
                  f"cameras {sorted(frames)}", flush=True)  # fmt: skip
        rows.append(np.concatenate(row, axis=1))
    width = max(r.shape[1] for r in rows)
    sheet = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows], axis=0)
    cv2.imwrite(str(out / "overview.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index['takes'])} takes to {out}")
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True, help="snapshot of the finalised dataset (with ws4_split.json)")
    p.add_argument("--repo-id", default="local/pen_final")
    p.add_argument("--out", default=str(HERE / "layouts"))
    p.add_argument("--wrist-key", default="observation.images.right_wrist")
    p.add_argument("--columns", default="right", help="task arm columns of a 12-D dataset (haller naming)")
    args = p.parse_args()
    split = json.loads((Path(args.root) / "ws4_split.json").read_text())
    write_refs(args.root, split["heldout"], Path(args.out), repo_id=args.repo_id, wrist_key=args.wrist_key,
               columns=args.columns, meta={"dataset": split["dataset"], "task": split["task"],
                                           "split_updated": split["updated"]})  # fmt: skip


if __name__ == "__main__":
    main()
