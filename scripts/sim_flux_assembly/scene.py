"""Shared native-coordinate assembly environment and independent physics predicates."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim_flux_lora"))

from common import FrameArm, Supervisor, images  # noqa: E402

FPS = 20
SUBSTEPS = 25
PROMPTS = {
    "pick_gear": "Pick up the blue gear and hold it upright above the tray.",
    "seat_gear": "Lower the blue gear over the metal shaft, release it onto the base, and move the gripper clear.",
    "pick_collar": "Pick up the brass spacer collar and hold it upright above the tray.",
    "seat_collar": "Lower the brass spacer collar onto the shaft above the blue gear, release it, and move clear.",
}
SUCCESS = {
    "pick_gear": "gear_held",
    "seat_gear": "gear_seated",
    "pick_collar": "collar_held",
    "seat_collar": "assembly_complete",
}
PRECONDITIONS = {
    "pick_gear": (),
    "seat_gear": ("gear_held",),
    "pick_collar": ("gear_seated",),
    "seat_collar": ("gear_seated", "collar_held"),
}


class AssemblyScene:
    def __init__(self, root: Path, seed: int, render: bool = False):
        import mujoco
        import numpy as np
        import official_camera_setup

        self.root, self.seed = root, seed
        self.model = mujoco.MjModel.from_xml_path(str(root / "scene.xml"))
        self.model.opt.timestep = 1 / (FPS * SUBSTEPS)
        self.model.opt.iterations = 80
        self.data = mujoco.MjData(self.model)
        self.arm = FrameArm(self.model, self.data, frame="native")
        self.safety = Supervisor(self.arm)
        self.rng = np.random.default_rng(seed)
        q = np.deg2rad([0, -94, 85, 70, 0, -4.5])
        q[:5] += np.deg2rad(self.rng.uniform(-0.5, 0.5, 5))
        self.data.qpos[self.arm.qadr] = q
        self.data.ctrl[self.arm.aids] = q
        fixture = np.array([0.23, 0.125]) + self.rng.uniform(-0.003, 0.003, 2)
        self.model.body_pos[self.model.body("fixture").id, :2] = fixture
        placements = {}
        for name, home in [("gear", [0.23, -0.045]), ("collar", [0.23, -0.13])]:
            xy = np.array(home) + self.rng.uniform(-0.004, 0.004, 2)
            address = self.model.jnt_qposadr[self.model.joint(f"{name}_free").id]
            self.data.qpos[address : address + 2] = xy
            yaw = self.rng.uniform(-0.12, 0.12)
            self.data.qpos[address + 3 : address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.model.body_pos[self.model.body(f"{name}_tray").id, :2] = xy
            placements[name] = xy.tolist()
        official_camera_setup.configure(self.model, self.data)
        self.model.light_diffuse[:] *= self.rng.uniform(0.9, 1.1)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640) if render else None
        self.part_ids = {name: self.model.body(name).id for name in ["gear", "collar"]}
        self.jaw_ids = {self.model.body(name).id for name in ["gripper", "moving_jaw_so101_v1"]}
        self.environment_ids = {0} | {
            self.model.body(name).id for name in ["fixture", "shaft", "gear_tray", "collar_tray", "gear", "collar"]
        }
        self.start_heights = {name: float(self.data.body(name).xpos[2]) for name in self.part_ids}
        self.counters = {key: 0 for key in ["gear_held", "gear_seated", "collar_held", "collar_seated"]}
        self.facts = {key: False for key in [*self.counters, "assembly_complete"]}
        self.metadata = {
            "seed": seed,
            "fixture_xy": fixture.tolist(),
            "part_xy": placements,
            "fps": FPS,
            "coordinate_frame": "native",
            "substeps": SUBSTEPS,
            "physics_timestep": self.model.opt.timestep,
        }
        self.tick = 0
        self.max_lift = {name: 0.0 for name in self.part_ids}
        self.ever_bilateral = {name: False for name in self.part_ids}
        self.update_predicates()

    def frames(self):
        if self.renderer is None:
            raise RuntimeError("Rendering was not enabled")
        return images(self.renderer, self.data)

    def contacts(self, name):
        body = self.part_ids[name]
        touching = set()
        for contact in self.data.contact:
            first, second = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if first == body:
                touching.add(int(second))
            elif second == body:
                touching.add(int(first))
        return touching

    def update_predicates(self):
        import numpy as np

        target = self.data.body("shaft").xpos[:2]
        state = {}
        for name in self.part_ids:
            body = self.data.body(name)
            contacts = self.contacts(name)
            bilateral = self.jaw_ids <= contacts
            arm_contact = bool(contacts - self.environment_ids)
            dof = self.model.jnt_dofadr[self.model.joint(f"{name}_free").id]
            speed = float(np.linalg.norm(self.data.qvel[dof : dof + 3]))
            angular = float(np.linalg.norm(self.data.qvel[dof + 3 : dof + 6]))
            lift = float(body.xpos[2] - self.start_heights[name])
            self.max_lift[name] = max(self.max_lift[name], lift)
            self.ever_bilateral[name] |= bilateral
            tilt = float(np.arccos(np.clip(body.xmat.reshape(3, 3)[2, 2], -1, 1)))
            xy_error = float(np.linalg.norm(body.xpos[:2] - target))
            bottom = 0.010 if name == "gear" else 0.042
            supported = (
                self.model.body("fixture").id in contacts if name == "gear" else self.part_ids["gear"] in contacts
            )
            held = bilateral and lift > 0.04 and speed < 0.04 and tilt < 0.35
            seated = (
                xy_error < 0.003
                and abs(body.xpos[2] - bottom) < 0.0035
                and tilt < 0.15
                and supported
                and not arm_contact
                and speed < 0.02
                and angular < 0.3
            )
            for suffix, value, threshold in [("held", held, 3), ("seated", seated, 10)]:
                key = f"{name}_{suffix}"
                self.counters[key] = self.counters[key] + 1 if value else 0
                self.facts[key] = self.counters[key] >= threshold
            state[name] = {
                "position": body.xpos.tolist(),
                "xy_error": xy_error,
                "tilt": tilt,
                "speed": speed,
                "bilateral": bilateral,
                "robot_contact": arm_contact,
                "supported": supported,
                "max_lift": self.max_lift[name],
            }
        self.facts["assembly_complete"] = self.facts["gear_seated"] and self.facts["collar_seated"]
        self.state = state
        return self.facts.copy()

    def step(self, target):
        import mujoco
        import numpy as np

        issued = self.safety.send("sim", np.asarray(target, float).tolist())
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.tick += 1
        self.update_predicates()
        return issued

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


__all__ = ["AssemblyScene", "FPS", "PROMPTS", "SUCCESS", "PRECONDITIONS"]
