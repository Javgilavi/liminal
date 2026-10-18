"""Grab a few frames from a Hub LeRobot v3.0 dataset without downloading its videos, and tile them.

Usage:
    python -I ws4/data/sample_frames.py MrC4t/bi_so_bag --root ws4/.cache/datasets/MrC4t__bi_so_bag \
        --episodes 0 7 --offset 3 --out ws4/.cache/frames

`--root` must hold the dataset's meta/ (from inspect_dataset.py or a data+meta download). ffmpeg reads the
mp4 over HTTPS with range requests, so only the bytes around each requested frame are fetched. The output is one
JPEG per (episode, camera) plus a contact sheet `<id>.jpg` (rows = episodes, columns = cameras).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def frame_jobs(root: Path, episodes: list[int], offset_s: float) -> list[tuple[int, str, str, float]]:
    """(episode, camera key, video path inside repo, timestamp in that file) for each requested frame."""
    import pandas as pd

    info = json.loads((root / "meta/info.json").read_text())
    cams = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    ep = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))])
    ep = ep.set_index("episode_index")
    jobs = []
    for e in episodes:
        row = ep.loc[e]
        for cam in cams:
            chunk = int(row[f"videos/{cam}/chunk_index"])
            fidx = int(row[f"videos/{cam}/file_index"])
            t0 = float(row[f"videos/{cam}/from_timestamp"])
            t1 = float(row[f"videos/{cam}/to_timestamp"])
            path = info["video_path"].format(video_key=cam, chunk_index=chunk, file_index=fidx)
            jobs.append((e, cam, path, min(t0 + offset_s, t1 - 0.05)))
    return jobs


def grab(url: str, t: float, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}", "-i", url,
           "-frames:v", "1", "-q:v", "3", str(dst)]
    return subprocess.run(cmd, timeout=180).returncode == 0 and dst.exists()


def contact_sheet(paths: list[list[Path]], dst: Path, cell_w: int = 320) -> None:
    from PIL import Image

    rows = [[Image.open(p) for p in row if p.exists()] for row in paths]
    cell_h = max(int(im.height * cell_w / im.width) for row in rows for im in row)
    sheet = Image.new("RGB", (cell_w * max(len(r) for r in rows), cell_h * len(rows)), "black")
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            im = im.convert("RGB").resize((cell_w, int(im.height * cell_w / im.width)))
            sheet.paste(im, (j * cell_w, i * cell_h))
    sheet.save(dst, quality=85)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_id")
    ap.add_argument("--root", type=Path, required=True, help="local folder with the dataset's meta/")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--offset", type=float, default=2.0, help="seconds into each episode")
    ap.add_argument("--out", type=Path, default=Path("ws4/.cache/frames"))
    args = ap.parse_args(argv)

    base = f"https://huggingface.co/datasets/{args.repo_id}/resolve/{args.revision}/"
    out = args.out / args.repo_id.replace("/", "__")
    grid: dict[int, list[Path]] = {}
    for e, cam, path, t in frame_jobs(args.root, args.episodes, args.offset):
        dst = out / f"ep{e:04d}_{cam.split('.')[-1]}.jpg"
        if not dst.exists():
            grab(base + path, t, dst)
        grid.setdefault(e, []).append(dst)
    contact_sheet(list(grid.values()), args.out / f"{args.repo_id.replace('/', '__')}.jpg")
    print(args.out / f"{args.repo_id.replace('/', '__')}.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
