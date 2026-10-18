"""Render the station's cameras to PNGs and put the third-person render next to the real photo.

    MUJOCO_GL=egl python ws4/sim/render_compare.py [--scene ws4/sim/scenes/tabletop_props.xml] [--out ws4/sim/outputs]

Writes <out>/{head,left_wrist,right_wrist,photo_view,overview}.png, head_depth.png and compare.png
(photo | photo_view render | head | wrists).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")

from ws4.sim.geometry import SIM_DIR  # noqa: E402

PHOTO = SIM_DIR.parent / "assets" / "photos" / "xlerobot_station.jpg"
# Folded rest pose close to the photo: upper arm back, forearm folded forward, gripper tilted up.
PHOTO_POSE_DEG = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 97.0, "wrist_flex": -45.0,
                  "wrist_roll": 0.0, "gripper": 10.0}


def main() -> None:
    from PIL import Image, ImageDraw

    from ws4.sim.env import XLeRobotStationEnv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(SIM_DIR / "scenes" / "tabletop_props.xml"))
    ap.add_argument("--out", default=str(SIM_DIR / "outputs"))
    ap.add_argument("--pose", choices=["photo", "zero"], default="photo")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    env = XLeRobotStationEnv(scene=a.scene, randomize=False, depth=True)
    env.reset(seed=0)
    if a.pose == "photo":
        arm = [PHOTO_POSE_DEG[j] for j in env.JOINTS]
        env.set_joint_positions(np.array(arm + arm))
    else:
        env.set_joint_positions(np.array([0.0] * 5 + [0.0] + [0.0] * 5 + [0.0]))
    imgs = {}
    for cam in ("head", "left_wrist", "right_wrist", "photo_view", "overview"):
        imgs[cam] = env.render(cam)
        Image.fromarray(imgs[cam]).save(out / f"{cam}.png")
    depth = env.render("head", depth=True)
    d8 = (255 * np.clip(depth / 1.5, 0, 1)).astype(np.uint8)
    Image.fromarray(d8).save(out / "head_depth.png")

    photo = Image.open(PHOTO).convert("RGB").resize((640, 480))
    tiles = [(photo, "photo"), (Image.fromarray(imgs["photo_view"]), "sim photo_view"),
             (Image.fromarray(imgs["head"]), "sim head"), (Image.fromarray(imgs["overview"]), "sim overview"),
             (Image.fromarray(imgs["left_wrist"]), "sim left_wrist"),
             (Image.fromarray(imgs["right_wrist"]), "sim right_wrist")]
    W = Image.new("RGB", (1280, 1440), "black")
    for i, (im, label) in enumerate(tiles):
        im = im.resize((640, 480))
        ImageDraw.Draw(im).text((8, 8), label, fill=(255, 255, 0))
        W.paste(im, ((i % 2) * 640, (i // 2) * 480))
    W.save(out / "compare.png")
    env.close()
    print("wrote", ", ".join(sorted(p.name for p in out.glob("*.png"))), "to", out)


if __name__ == "__main__":
    main()
