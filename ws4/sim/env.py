"""XLeRobotStationEnv: MuJoCo sim of Oscar's bimanual XLeRobot station with LeRobot-style I/O.

    from ws4.sim.env import XLeRobotStationEnv
    env = XLeRobotStationEnv()                      # scenes/tabletop_props.xml, degrees, randomised
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(obs["observation.state"])  # hold still

Actions and `observation.state` are 12-D absolute joint positions [left 6 | right 6] in LeRobot SO-101
calibrated units (joint order shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper):
degrees for the five body joints and 0..100 for the gripper (see units.py). Images are uint8 HxWx3;
`observation.images.head_depth` is uint16 millimetres HxWx1 like LeRobot's RealSense camera. Privileged
ground truth (object poses, EE poses, raw qpos) is under `privileged` and must not be fed to a policy.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .geometry import ARM_JOINTS, SIDES, SIM_DIR, load_geometry, val
from .units import StationUnits

DEFAULT_SCENE = SIM_DIR / "scenes" / "tabletop_props.xml"
CAMERAS = ("head", "left_wrist", "right_wrist")
# Ready pose in LeRobot degrees: upper arm up, forearm forward, wrist pitched down, gripper half open.
HOME_DEG = np.array([0.0, -10.0, 10.0, 75.0, 0.0, 30.0])

# Domain randomisation ranges (uniform, +-). Prop regions are in world x/y (metres).
DR = {
    "prop_regions": {
        "cube_red": ((0.16, 0.36), (0.02, 0.26)), "cube_green": ((0.16, 0.36), (-0.12, 0.12)),
        "cube_blue": ((0.16, 0.36), (-0.26, -0.02)), "cup": ((0.18, 0.38), (0.08, 0.30)),
        "bowl": ((0.22, 0.42), (-0.30, -0.10)), "lidbox": ((0.38, 0.46), (-0.08, 0.10)),
        "target": ((0.36, 0.44), (-0.10, 0.06)),
    },
    "min_separation": 0.075,
    "yaw": np.pi,
    "rgb_jitter": 0.12,
    "light_dir_jitter": 0.25,
    "light_diffuse_jitter": 0.2,
    "ambient_jitter": 0.1,
    "head_cam_pos": 0.01, "head_cam_rot_deg": 2.0, "head_fovy_deg": 2.0,
    "wrist_cam_pos": 0.004, "wrist_cam_rot_deg": 2.0, "wrist_fovy_deg": 3.0,
}


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _small_rot_quat(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg))
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])


class XLeRobotStationEnv:
    JOINTS = ARM_JOINTS

    def __init__(self, scene: str | Path | None = None, units: str = "degrees", calibration: dict | None = None,
                 cameras=CAMERAS, image_size: tuple[int, int] = (480, 640), depth: bool = False,
                 randomize: bool = True, control_hz: float = 30.0, render_images: bool = True,
                 home_deg=None) -> None:
        os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self._mj = mujoco
        self.scene = Path(scene) if scene else DEFAULT_SCENE
        self.model = mujoco.MjModel.from_xml_path(str(self.scene))
        self.data = mujoco.MjData(self.model)
        self.units = StationUnits(units, calibration)
        self.cameras = tuple(cameras)
        self.height, self.width = image_size
        self.depth = depth
        self.randomize = randomize
        self.render_images = render_images
        self.control_hz = control_hz
        self.n_substeps = max(1, int(round(1.0 / (control_hz * self.model.opt.timestep))))
        self.home = np.asarray(HOME_DEG if home_deg is None else home_deg, dtype=float)
        self.rng = np.random.default_rng()
        self._renderer = None
        m = self.model
        self.qadr = np.array([m.jnt_qposadr[m.joint(f"{s}_{j}").id] for s in SIDES for j in ARM_JOINTS])
        self.dadr = np.array([m.jnt_dofadr[m.joint(f"{s}_{j}").id] for s in SIDES for j in ARM_JOINTS])
        self.act = np.array([m.actuator(f"{s}_{j}").id for s in SIDES for j in ARM_JOINTS])
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[self.act].T
        free = mujoco.mjtJoint.mjJNT_FREE
        self.free_bodies = [m.body(m.jnt_bodyid[j]).name for j in range(m.njnt) if m.jnt_type[j] == free]
        self.prop_bodies = self.free_bodies + [n for n in ("lidbox",) if self._has_body(n)]
        self._nominal = {
            "geom_rgba": m.geom_rgba.copy(), "body_pos": m.body_pos.copy(), "body_quat": m.body_quat.copy(),
            "light_dir": m.light_dir.copy(), "light_diffuse": m.light_diffuse.copy(),
            "cam_pos": m.cam_pos.copy(), "cam_quat": m.cam_quat.copy(), "cam_fovy": m.cam_fovy.copy(),
            "headlight_ambient": m.vis.headlight.ambient.copy(),
        }
        self.step_count = 0

    # ------------------------------------------------------------------------------------ basics
    def _has_body(self, name: str) -> bool:
        return self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_BODY, name) >= 0

    @property
    def dt(self) -> float:
        return self.n_substeps * self.model.opt.timestep

    def reset(self, seed: int | None = None, options: dict | None = None):
        mj = self._mj
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        for k, v in self._nominal.items():
            if k == "headlight_ambient":
                m.vis.headlight.ambient[:] = v
            else:
                getattr(m, k)[:] = v
        mj.mj_resetData(m, d)
        if self.randomize:
            self._randomize_scene()
        home = np.asarray((options or {}).get("home_deg", self.home), dtype=float)
        self.set_joint_positions(np.concatenate([home, home]))
        # Let props settle on the table with the arms held at home.
        for _ in range(int(0.3 / m.opt.timestep)):
            mj.mj_step(m, d)
        self.step_count = 0
        return self.get_obs(), {"seed": seed}

    def set_joint_positions(self, values12) -> None:
        """Teleport both arms to a 12-D LeRobot-unit pose (and hold it with the actuators)."""
        q = self.units.to_radians(values12)
        self.data.qpos[self.qadr] = q
        self.data.qvel[self.dadr] = 0.0
        self.data.ctrl[self.act] = np.clip(q, self.ctrl_lo, self.ctrl_hi)
        self._mj.mj_forward(self.model, self.data)

    def step(self, action):
        q = self.units.to_radians(np.asarray(action, dtype=float).reshape(12))
        self.data.ctrl[self.act] = np.clip(q, self.ctrl_lo, self.ctrl_hi)
        self._mj.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.step_count += 1
        return self.get_obs(), 0.0, False, False, {}

    # ------------------------------------------------------------------------------ observations
    def joint_state(self) -> np.ndarray:
        """12-D state in LeRobot units (what `observation.state` holds)."""
        return self.units.from_radians(self.data.qpos[self.qadr]).astype(np.float32)

    def get_obs(self) -> dict:
        obs = {"observation.state": self.joint_state()}
        if self.render_images:
            for cam in self.cameras:
                obs[f"observation.images.{cam}"] = self.render(cam)
            if self.depth and "head" in self.cameras:
                dm = self.render("head", depth=True)
                obs["observation.images.head_depth"] = np.clip(dm * 1000.0, 0, 65535).astype(np.uint16)[..., None]
        obs["privileged"] = {
            "object_poses": self.object_poses(),
            "ee_poses": {s: np.concatenate(self.ee_pose(s)) for s in SIDES},
            "qpos_rad": self.data.qpos[self.qadr].copy(),
            "lid_angle": self._lid_angle(),
        }
        return obs

    def arm_view(self, side: str, obs: dict | None = None) -> dict:
        """Inputs for the single-arm FLUX SO-101 checkpoint: head image as `scene`, this arm's wrist image, 6-D state.

        Pass `obs` to reuse already rendered images; otherwise renders now."""
        i = SIDES.index(side)
        if obs is not None and f"observation.images.{side}_wrist" in obs:
            scene, wrist = obs["observation.images.head"], obs[f"observation.images.{side}_wrist"]
            state = obs["observation.state"][6 * i: 6 * i + 6]
        else:
            scene, wrist = self.render("head"), self.render(f"{side}_wrist")
            state = self.joint_state()[6 * i: 6 * i + 6]
        return {"scene": scene, "wrist": wrist, "state": np.asarray(state, dtype=np.float32)}

    def ee_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """World pose of `<side>_ee` (TCP between the fingertips): position and quaternion [w, x, y, z]."""
        sid = self.model.site(f"{side}_ee").id
        q = np.empty(4)
        self._mj.mju_mat2Quat(q, self.data.site_xmat[sid])
        return self.data.site_xpos[sid].copy(), q

    def object_poses(self) -> dict:
        out = {}
        for name in self.prop_bodies:
            b = self.model.body(name).id
            out[name] = np.concatenate([self.data.xpos[b], self.data.xquat[b]])
        return out

    def _lid_angle(self):
        j = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, "lidbox_hinge")
        return float(self.data.qpos[self.model.jnt_qposadr[j]]) if j >= 0 else None

    # ----------------------------------------------------------------------------------- render
    def render(self, camera: str, depth: bool = False) -> np.ndarray:
        if self._renderer is None:
            self._renderer = self._mj.Renderer(self.model, self.height, self.width)
        r = self._renderer
        if depth:
            r.enable_depth_rendering()
        r.update_scene(self.data, camera=camera)
        img = r.render()
        if depth:
            r.disable_depth_rendering()
            return img.astype(np.float32)
        return img

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------------ randomisation
    def _randomize_scene(self) -> None:
        m, rng = self.model, self.rng
        placed: list[np.ndarray] = []
        for name, ((x0, x1), (y0, y1)) in DR["prop_regions"].items():
            if not self._has_body(name):
                continue
            for _ in range(50):
                xy = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
                if all(np.linalg.norm(xy - p) > DR["min_separation"] for p in placed):
                    break
            placed.append(xy)
            yaw = rng.uniform(-DR["yaw"], DR["yaw"]) if name != "lidbox" else rng.uniform(-0.3, 0.3)
            qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            b = m.body(name).id
            jnt = m.body_jntadr[b]
            if jnt >= 0 and m.jnt_type[jnt] == self._mj.mjtJoint.mjJNT_FREE:
                a = m.jnt_qposadr[jnt]
                self.data.qpos[a:a + 2] = xy
                self.data.qpos[a + 2] = m.body_pos[b][2] + 0.002
                self.data.qpos[a + 3:a + 7] = qz
            else:
                m.body_pos[b][:2] = xy
                m.body_quat[b] = qz
        # Prop and table colours
        for g in range(m.ngeom):
            bname = m.body(m.geom_bodyid[g]).name
            if bname in self.prop_bodies or bname.startswith("lidbox") or bname == "table":
                rgb = self._nominal["geom_rgba"][g, :3] + rng.uniform(-DR["rgb_jitter"], DR["rgb_jitter"], 3)
                m.geom_rgba[g, :3] = np.clip(rgb, 0, 1)
        # Lighting
        for i in range(m.nlight):
            d = self._nominal["light_dir"][i] + rng.uniform(-DR["light_dir_jitter"], DR["light_dir_jitter"], 3)
            m.light_dir[i] = d / np.linalg.norm(d)
            gain = 1 + rng.uniform(-1, 1) * DR["light_diffuse_jitter"]
            m.light_diffuse[i] = np.clip(self._nominal["light_diffuse"][i] * gain, 0, 1)
        amb = self._nominal["headlight_ambient"] + rng.uniform(-1, 1) * DR["ambient_jitter"]
        m.vis.headlight.ambient[:] = np.clip(amb, 0, 1)
        # Camera extrinsics / fov jitter (models mount tolerance and calibration error)
        for cam in CAMERAS:
            c = m.camera(cam).id
            head = cam == "head"
            dp = DR["head_cam_pos"] if head else DR["wrist_cam_pos"]
            dr = DR["head_cam_rot_deg"] if head else DR["wrist_cam_rot_deg"]
            df = DR["head_fovy_deg"] if head else DR["wrist_fovy_deg"]
            m.cam_pos[c] = self._nominal["cam_pos"][c] + rng.uniform(-dp, dp, 3)
            m.cam_quat[c] = _quat_mul(self._nominal["cam_quat"][c], _small_rot_quat(rng, dr))
            m.cam_fovy[c] = self._nominal["cam_fovy"][c] + rng.uniform(-df, df)


def geometry_summary() -> dict:
    """Small dict of the key station numbers (for logging alongside episodes)."""
    g = load_geometry()
    return {s: {"base_pos": val(g["arms"][s]["base_pos"]), "base_quat": val(g["arms"][s]["base_quat"])} for s in SIDES}
