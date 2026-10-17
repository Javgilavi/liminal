"""Smoke test: load the Haller dual SO-101 scene, hold both arms at a pose, step, render cameras.

    MUJOCO_GL=egl .venv/bin/python check.py          # headless; writes renders/*.png

Joint values are radians with zero at the middle of each joint's range, i.e. LeRobot SO-101
"new calibration" in degree mode converted to radians.
"""

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

HERE = Path(__file__).parent
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# Reaching down toward the bench with the gripper half open (degrees, LeRobot order).
POSE_DEG = (0, 30, -10, 70, 0, 30)
CAMERAS = {"top": (640, 480), "front": (960, 720), "left_wrist": (1280, 720), "right_wrist": (1280, 720)}

model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
data = mujoco.MjData(model)

for arm in ("left", "right"):
    for j, deg in zip(JOINTS, POSE_DEG, strict=True):
        q = np.deg2rad(deg)
        data.qpos[model.joint(f"{arm}_{j}").qposadr[0]] = q
        data.ctrl[model.actuator(f"{arm}_{j}").id] = q
mujoco.mj_forward(model, data)

for _ in range(int(2.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)

for arm in ("left", "right"):
    q = [data.qpos[model.joint(f"{arm}_{j}").qposadr[0]] for j in JOINTS]
    err = np.rad2deg(np.array(q) - np.deg2rad(POSE_DEG))
    print(f"{arm:5s} tip at {np.round(data.site(f'{arm}_gripperframe').xpos, 3)}, "
          f"tracking error after 2 s (deg): {np.round(err, 1)}")
for cube in ("cube_red", "cube_green", "cube_amber"):
    print(f"{cube:10s} at {np.round(data.body(cube).xpos, 3)}")

out = HERE / "renders"
out.mkdir(exist_ok=True)
for cam, (w, h) in CAMERAS.items():
    renderer = mujoco.Renderer(model, height=h, width=w)
    renderer.update_scene(data, camera=cam)
    Image.fromarray(renderer.render()).save(out / f"{cam}.png")
    renderer.close()
print(f"renders in {out}")
