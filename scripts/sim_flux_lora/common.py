"""Shared, simulation-only scene and coordinate contract for collection and rollout."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = Path(os.environ.get("LIMINAL_SIM_ROOT", str(REPO_ROOT / "sim/single_arm")))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SIM_ROOT))

from liminal.safety.supervisor import JointLimits, SafetySupervisor  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TASK = "Pick up the red block and place it in the blue container."
SIGNS = np.array([1, -1, 1, 1, 1], dtype=float)
OFFSETS = np.array([0, 90, 90, 0, 0], dtype=float)


class FrameArm:
    name = "sim"
    num_joints = 6

    def __init__(self, model, data, frame="dataset90"):
        self.model, self.data = model, data
        if frame not in ("dataset90", "native"):
            raise ValueError(frame)
        self.signs = SIGNS if frame == "dataset90" else np.ones(5)
        self.offsets = OFFSETS if frame == "dataset90" else np.zeros(5)
        self.jids = np.array([model.joint(n).id for n in JOINTS])
        self.qadr = model.jnt_qposadr[self.jids]
        self.aids = np.array([model.actuator(n).id for n in JOINTS])
        self.ranges = model.jnt_range[self.jids].copy()

    def to_policy(self, q):
        result = np.rad2deg(np.asarray(q, float))
        result[:5] = result[:5] * self.signs + self.offsets
        lo, hi = self.ranges[-1]
        result[-1] = (q[-1] - lo) / (hi - lo) * 100
        return result.astype(np.float32)

    def to_sim(self, values):
        values = np.asarray(values, float).copy()
        values[:5] = (values[:5] - self.offsets) / self.signs
        result = np.deg2rad(values)
        lo, hi = self.ranges[-1]
        result[-1] = lo + values[-1] / 100 * (hi - lo)
        return result

    def read_state(self):
        return self.to_policy(self.data.qpos[self.qadr]).tolist()

    def send_action(self, values):
        self.data.ctrl[self.aids] = self.to_sim(values)

    def hold(self, positions=None):
        self.send_action(self.read_state() if positions is None else positions)

    def stop(self):
        self.hold()


class Supervisor(SafetySupervisor):
    def __init__(self, arm):
        a, b = arm.to_policy(arm.ranges[:, 0]), arm.to_policy(arm.ranges[:, 1])
        limits = JointLimits(np.minimum(a, b).tolist(), np.maximum(a, b).tolist(), max_step=math.inf)
        super().__init__({"sim": arm}, {"sim": limits})
        self.max_steps = np.array([3, 3, 3, 3, 3, 6], float)

    def validate(self, arm, target, current=None):
        bounded = np.asarray(super().validate(arm, target, None))
        if current is not None:
            bounded = np.asarray(current) + np.clip(bounded - current, -self.max_steps, self.max_steps)
        return bounded.tolist()


def make_episode(seed, frame="dataset90"):
    import mujoco
    import official_camera_setup

    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(str(SIM_ROOT / "scene.xml"))
    data = mujoco.MjData(model)
    arm = FrameArm(model, data, frame=frame)
    q = np.deg2rad(np.array([0, -94, 85, 70, 0, -4.5], float))
    q[:5] += np.deg2rad(rng.uniform(-1, 1, 5))
    data.qpos[arm.qadr] = q
    data.ctrl[arm.aids] = q
    block = model.jnt_qposadr[model.joint("block_free").id]
    block_xy = np.array([0.23, -0.04]) + rng.uniform(-0.012, 0.012, 2)
    bowl_xy = np.array([0.23, 0.13]) + rng.uniform(-0.006, 0.006, 2)
    yaw = float(rng.uniform(-0.10, 0.10))
    data.qpos[block : block + 2] = block_xy
    data.qpos[block + 3 : block + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    model.body_pos[model.body("bowl").id, :2] = bowl_xy
    official_camera_setup.configure(model, data)
    scene = model.camera("scene").id
    camera_jitter = rng.uniform(-0.004, 0.004, 3)
    model.cam_pos[scene] += camera_jitter
    model.cam_fovy[scene] += rng.uniform(-1, 1)
    model.light_diffuse[:] *= rng.uniform(0.85, 1.15)
    model.geom_rgba[model.geom("table").id, :3] += rng.uniform(-0.025, 0.025, 3)
    model.geom_rgba[model.geom("block_geom").id, :3] = [rng.uniform(0.78, 0.95), 0.08, 0.06]
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == model.body("bowl").id:
            model.geom_rgba[gid, :3] = [0.10, rng.uniform(0.25, 0.35), rng.uniform(0.78, 0.95)]
    mujoco.mj_forward(model, data)
    for _ in range(90):
        mujoco.mj_step(model, data)
    metadata = {
        "seed": seed,
        "block_xy": block_xy.tolist(),
        "bowl_xy": bowl_xy.tolist(),
        "block_yaw_rad": yaw,
        "camera_jitter_m": camera_jitter.tolist(),
        "initial_native_degrees": np.rad2deg(data.qpos[arm.qadr]).tolist(),
    }
    return model, data, arm, Supervisor(arm), metadata


def images(renderer, data):
    from PIL import Image

    frames = {}
    for role, camera in [("scene", "scene"), ("wrist", "wrist_cam")]:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render().copy()
        frames[role] = np.asarray(Image.fromarray(pixels).resize((256, 256), Image.Resampling.BILINEAR))
    return frames


def contact_flags(model, data):
    block = model.geom("block_geom").id
    moving = model.body("moving_jaw_so101_v1").id
    fixed = model.body("gripper").id
    touched = set()
    for contact in data.contact:
        if contact.geom1 == block:
            touched.add(int(model.geom_bodyid[contact.geom2]))
        elif contact.geom2 == block:
            touched.add(int(model.geom_bodyid[contact.geom1]))
    return bool(touched & {moving, fixed}), moving in touched and fixed in touched


def outcome(model, data):
    block = data.body("block").xpos.copy()
    bowl = data.body("bowl").xpos.copy()
    velocity = model.jnt_dofadr[model.joint("block_free").id]
    slow = np.linalg.norm(data.qvel[velocity : velocity + 3]) < 0.05
    inside = np.all(np.abs(block[:2] - bowl[:2]) < 0.033) and 0 < block[2] - bowl[2] < 0.06
    return {"block_xyz": block.tolist(), "block_in_bowl": bool(inside), "slow": bool(slow)}
