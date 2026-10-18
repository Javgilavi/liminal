"""Merge local LeRobot v3.0 datasets into one, for a single-dataset LoRA run (LeRobot main has no multi-dataset).

Usage (GPU box, LeRobot env):
    python -I ws4/data/merge_datasets.py OUT_DIR SRC1 SRC2 ... [--check-only]

LeRobot's own rule (lerobot@146d847 datasets/feature_utils.py:143-171, aggregate.py:123-157): same fps, same
robot_type and the same features, where for videos only the encoder settings (g, crf, preset, fast_decode,
extra_options, video_backend) and the depth-marker spelling may differ. Codec and frame size must match, so
h264 and AV1 sources, or 240x320 and 480x640 wrist cameras, cannot be merged without re-encoding.
`--check-only` runs that rule here, without LeRobot, and says which inputs are compatible with the first one.
Videos are kept one file per source (`concatenate_videos=False`) so no stream-copy concatenation is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ENCODER_KEYS = {f"video.{k}" for k in ("g", "crf", "preset", "fast_decode", "extra_options", "video_backend")}
DEPTH_KEYS = ("is_depth_map", "video.is_depth_map")


def _normalized(feature: dict) -> dict:
    f = json.loads(json.dumps(feature))
    info = f.get("info") or {}
    depth = any(bool(info.pop(k, False)) for k in DEPTH_KEYS)
    f["info"] = {k: v for k, v in info.items() if k not in ENCODER_KEYS}
    if f.get("dtype") == "video":
        f["info"]["depth"] = depth
    return f


def compatibility(infos: list[dict]) -> list[list[str]]:
    """Reasons each info.json cannot be merged with the first one (empty list = compatible)."""
    ref = infos[0]
    out = []
    for info in infos:
        why = []
        if info.get("fps") != ref.get("fps"):
            why.append(f"fps {info.get('fps')} != {ref.get('fps')}")
        if info.get("robot_type") != ref.get("robot_type"):
            why.append(f"robot_type {info.get('robot_type')!r} != {ref.get('robot_type')!r}")
        fa, fb = ref["features"], info["features"]
        if set(fa) != set(fb):
            why.append(f"feature keys differ: {sorted(set(fa) ^ set(fb))}")
        for k in sorted(set(fa) & set(fb)):
            if _normalized(fa[k]) != _normalized(fb[k]):
                a, b = _normalized(fa[k]), _normalized(fb[k])
                diff = {x for x in set(a) | set(b) if a.get(x) != b.get(x)}
                if "info" in diff:
                    ia, ib = a.get("info", {}), b.get("info", {})
                    diff = (diff - {"info"}) | {f"info.{x}" for x in set(ia) | set(ib) if ia.get(x) != ib.get(x)}
                why.append(f"{k}: {sorted(diff)}")
        out.append(why)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("srcs", type=Path, nargs="+")
    ap.add_argument("--repo-id", default="local/merged", help="name stored in the merged dataset")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    infos = [json.loads((s / "meta/info.json").read_text()) for s in args.srcs]
    bad = False
    for src, why in zip(args.srcs, compatibility(infos), strict=True):
        print(f"{'ok ' if not why else 'NO '} {src}" + ("" if not why else "\n      " + "\n      ".join(why)))
        bad |= bool(why)
    if bad:
        print("incompatible inputs: drop them or re-encode their videos first", file=sys.stderr)
        return 1
    if args.check_only:
        return 0
    from lerobot.datasets.aggregate import aggregate_datasets  # LeRobot main (box env), not needed for --check-only

    aggregate_datasets(
        repo_ids=[f"local/{s.name}" for s in args.srcs],
        aggr_repo_id=args.repo_id,
        roots=list(args.srcs),
        aggr_root=args.out,
        concatenate_videos=False,
    )
    merged = json.loads((args.out / "meta/info.json").read_text())
    print(f"merged {merged['total_episodes']} episodes, {merged['total_frames']} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)  # skip interpreter teardown: pyarrow deadlocked there in 2 of ~12 laptop runs
