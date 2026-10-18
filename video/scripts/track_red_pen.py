"""Find the red pen cap in each frame of a camera crop and write per-frame boxes as JSON.

The HUD tracking box is drawn from this data, so it follows what the camera really saw.
Usage: python track_red_pen.py <video.mp4> <out.json> [min_pixels]
"""

import json
import subprocess
import sys

import numpy as np


def frames(path: str):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         path], capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE)
    size = w * h * 3
    while chunk := proc.stdout.read(size):
        if len(chunk) < size:
            break
        yield np.frombuffer(chunk, np.uint8).reshape(h, w, 3), w, h


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    min_px = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    boxes = []
    for img, w, h in frames(src):
        rgb = img.astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        mx, mn = rgb.max(-1), rgb.min(-1)
        sat = (mx - mn) / (mx + 1e-6)
        # Hue of red-dominant pixels in degrees: the cap sits near 350-10, the orange slot holder near 25-40.
        hue = np.where(r >= mx, (60 * (g - b) / (mx - mn + 1e-6)) % 360, 180)
        mask = (r > 0.55) & (sat > 0.55) & ((hue < 12) | (hue > 340))
        if mask.sum() < min_px:
            boxes.append(None)
            continue
        # Keep the widest red column band (the cap), not the marker label or slot edges beside it.
        cols = mask.sum(0) >= 3
        best, start = (0, 0, 0), None
        for x, on in enumerate(np.append(cols, False)):
            if on and start is None:
                start = x
            elif not on and start is not None:
                weight = int(mask[:, start:x].sum())
                if weight > best[0]:
                    best = (weight, start, x)
                start = None
        weight, x0, x1 = best
        if weight < min_px:
            boxes.append(None)
            continue
        ys = np.nonzero(mask[:, x0:x1].any(1))[0]
        y0, y1 = np.percentile(ys, [2, 98])
        boxes.append([round(x0 / w, 4), round(y0 / h, 4), round(x1 / w, 4), round(y1 / h, 4)])
    with open(out, "w") as f:
        json.dump({"fps": 30, "boxes": boxes}, f)
    found = sum(b is not None for b in boxes)
    print(f"{out}: {found}/{len(boxes)} frames with a red cap")


if __name__ == "__main__":
    main()
