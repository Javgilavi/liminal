"""Re-encode a LeRobot v3.0 dataset's videos to AV1 (and optionally 640x480) so it merges with the public core set.

Fallback path only: ws4/teleop records AV1 640x480 directly. Use this for takes recorded with h264
(`WS4_VCODEC=h264`, or haller's own up.sh) or with 1280x720 wrist cameras.

    python ws4/teleop/transcode.py SRC DST [--size 640x480] [--crf 30] [--preset 12]

Writes a full copy at DST (parquet and metadata copied, every mp4 re-encoded frame by frame, so frame counts and
timestamps are unchanged) and updates info.json (codec, size, feature shape). Per-episode image statistics are
copied unchanged: they are pixel means over a downsampled frame and move by well under 1% with a re-encode or
resize. LeRobot's own AV1 settings are the defaults (datasets/video_utils.py `_get_codec_options`: g 2, crf 30,
preset 12, yuv420p). Needs PyAV with libsvtav1 (the haller-hmi and LeRobot venvs both have it).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from fractions import Fraction
from pathlib import Path


def transcode_file(src: Path, dst: Path, size: tuple[int, int] | None, crf: int, preset: int, g: int = 2) -> int:
    import av

    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with av.open(str(src)) as inp, av.open(str(dst), "w") as out:
        vin = inp.streams.video[0]
        fps = vin.average_rate or vin.guessed_rate or Fraction(30)
        h, w = size or (vin.codec_context.height, vin.codec_context.width)
        vout = out.add_stream("libsvtav1", rate=fps, options={"g": str(g), "crf": str(crf), "preset": str(preset)})
        vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
        vout.time_base = Fraction(1, int(round(float(fps))))
        for frame in inp.decode(vin):
            if (frame.height, frame.width) != (h, w):
                frame = frame.reformat(width=w, height=h)
            frame = frame.reformat(format="yuv420p")
            frame.pts = n
            frame.time_base = vout.time_base
            for pkt in vout.encode(frame):
                out.mux(pkt)
            n += 1
        for pkt in vout.encode():
            out.mux(pkt)
    return n


def transcode(src: Path, dst: Path, size: tuple[int, int] | None = None, crf: int = 30, preset: int = 12) -> dict:
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.mp4"))
    info = json.loads((src / "meta" / "info.json").read_text())
    counts = {}
    for mp4 in sorted((src / "videos").rglob("*.mp4")):
        rel = mp4.relative_to(src)
        counts[str(rel)] = transcode_file(mp4, dst / rel, size, crf, preset)
    for feat in info["features"].values():
        if feat.get("dtype") != "video":
            continue
        vi = feat.setdefault("info", {})
        vi["video.codec"] = "av1"
        vi["video.pix_fmt"] = "yuv420p"
        if size:
            vi["video.height"], vi["video.width"] = size
            feat["shape"] = [size[0], size[1], feat["shape"][2]]
    info["ws4_transcode"] = {"source": str(src), "codec": "libsvtav1", "crf": crf, "preset": preset,
                             "size": list(size) if size else None}
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--size", default=None, help="WxH, e.g. 640x480")
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--preset", type=int, default=12)
    args = ap.parse_args(argv)
    size = None
    if args.size:
        w, h = (int(x) for x in args.size.lower().split("x"))
        size = (h, w)
    counts = transcode(args.src, args.dst, size, args.crf, args.preset)
    print(f"re-encoded {len(counts)} files, {sum(counts.values())} frames -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
