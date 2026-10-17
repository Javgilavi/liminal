"""Open recorded or converted datasets with the LeRobot that trains FLUX and decode every episode's ends.

Run in the LeRobot main env (~/ws4/lerobot/.venv, the env `lerobot-train --policy.type=flux3` uses):

    ~/ws4/lerobot/.venv/bin/python -I ws4/teleop/check_load.py DATASET_ROOT [DATASET_ROOT ...]

For every episode it fetches the first and the last frame through `LeRobotDataset.__getitem__`, which decodes
each camera at the parquet timestamp and raises if the video does not cover it (LeRobot's timestamp tolerance
check). That is the failure a dropped encoder frame or a truncated mp4 produces at training time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check(root: Path) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    info = json.loads((root / "meta" / "info.json").read_text())
    ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
    cams = list(ds.meta.video_keys)
    out = {"root": str(root), "episodes": ds.meta.total_episodes, "frames": len(ds), "fps": ds.fps,
           "cameras": cams, "state_dim": info["features"]["observation.state"]["shape"][0], "bad": []}
    for ep in range(ds.meta.total_episodes):
        rec = ds.meta.episodes[ep]
        lo, hi = int(rec["dataset_from_index"]), int(rec["dataset_to_index"]) - 1
        for idx in (lo, hi):
            item = ds[idx]
            for k in cams:
                img = item[k]
                if tuple(img.shape[-2:]) != tuple(info["features"][k]["shape"][:2]):
                    out["bad"].append(f"ep {ep} idx {idx} {k}: shape {tuple(img.shape)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", type=Path, nargs="+")
    args = ap.parse_args(argv)
    rc = 0
    for r in args.roots:
        try:
            res = check(r.resolve())
        except Exception as e:  # report and keep going: one bad dataset should not hide the others
            print(f"FAIL {r}: {type(e).__name__}: {e}")
            rc = 1
            continue
        ok = not res["bad"]
        rc |= 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {r}: {res['episodes']} episodes, {res['frames']} frames, fps {res['fps']}, "
              f"state {res['state_dim']}-D, cameras {[c.split('.')[-1] for c in res['cameras']]}")
        for b in res["bad"]:
            print("     ", b)
    return rc


if __name__ == "__main__":
    sys.exit(main())
