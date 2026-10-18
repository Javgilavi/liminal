"""Contact-only IK demonstrations, segmented at the same atomic success predicates."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scene import FPS, PROMPTS, SUCCESS, AssemblyScene


class Teacher:
    def __init__(self, scene: AssemblyScene):
        import mujoco
        import numpy as np

        self.scene, self.mujoco, self.np = scene, mujoco, np
        self.scratch = mujoco.MjData(scene.model)
        self.q = scene.data.qpos[scene.arm.qadr].copy()
        self.scratch.qpos[:] = scene.data.qpos
        tips = [scene.model.geom(name).id for name in ["fixed_jaw_sph_tip1", "moving_jaw_sph_tip1"]]
        self.gripper = scene.model.body("gripper").id
        self.scratch.qpos[scene.arm.qadr[-1]] = scene.arm.to_sim([0, 0, 0, 0, 0, 45])[-1]
        mujoco.mj_forward(scene.model, self.scratch)
        rotation = self.scratch.xmat[self.gripper].reshape(3, 3)
        self.anchor = rotation.T @ (self.scratch.geom_xpos[tips].mean(0) - self.scratch.xpos[self.gripper])
        self.jaw_anchor = self.anchor.copy()
        self.axis = np.array([0.0, 0.0, 1.0])
        self.grip_offsets = {}
        self.records = {key: [] for key in ["states", "actions", "qpos", "skill", "phase", "gear_xyz", "collar_xyz"]}
        self.pixels = {key: [] for key in ["scene", "wrist"]}
        self.segments = []

    def ik(self, target, opening, orientation_weight=0.08):
        from scipy.optimize import least_squares

        np, scene, mj = self.np, self.scene, self.mujoco
        q = scene.data.qpos[scene.arm.qadr].copy()
        q[-1] = scene.arm.to_sim([0, 0, 0, 0, 0, opening])[-1]

        def residual(values):
            self.scratch.qpos[:] = scene.data.qpos
            self.scratch.qpos[scene.arm.qadr] = values
            mj.mj_forward(scene.model, self.scratch)
            rotation = self.scratch.xmat[self.gripper].reshape(3, 3)
            return np.r_[
                self.scratch.xpos[self.gripper] + rotation @ self.anchor - target,
                orientation_weight * (rotation @ self.axis - [0, 0, 1]),
            ]

        lower, upper = scene.arm.ranges[:5, 0] + 0.01, scene.arm.ranges[:5, 1] - 0.01

        def solve(start):
            return least_squares(
                lambda joints: residual(np.r_[joints, q[-1]]),
                np.clip(start, lower, upper),
                bounds=(lower, upper),
                max_nfev=90,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )

        best = solve(q[:5])
        if np.linalg.norm(best.fun) > 0.002:
            for roll in [-90, 0, 90]:
                candidate = solve(np.deg2rad([-25, -10, 25, 75, roll]))
                if np.linalg.norm(candidate.fun) < np.linalg.norm(best.fun):
                    best = candidate
        q[:5] = best.x
        return scene.arm.to_policy(q)

    def move_part(self, part, phase, xyz, ticks, skill=None):
        np, scene = self.np, self.scene
        start = scene.data.body(part).xpos.copy()
        target = np.asarray(xyz)
        for tick in range(ticks + 35):
            # The teacher observes the part pose. This compensates grasp slip and
            # actuator deflection; the learned policy later receives only cameras/state.
            rotation = scene.data.xmat[self.gripper].reshape(3, 3)
            self.anchor = rotation.T @ (scene.data.body(part).xpos - scene.data.xpos[self.gripper])
            self.axis = rotation.T @ scene.data.body(part).xmat.reshape(3, 3)[:, 2]
            u = min(1.0, (tick + 1) / ticks)
            desired = start + (u * u * (3 - 2 * u)) * (target - start)
            command = self.ik(desired, 0)
            self.record_step(skill or f"seat_{part}", phase, command)
            error = np.linalg.norm(scene.data.body(part).xpos - target)
            if u >= 1 and error < 0.0015 and scene.state[part]["tilt"] < 0.10:
                return True
        return False

    def record_step(self, skill, phase, command):
        scene = self.scene
        self.records["states"].append(scene.arm.read_state())
        self.records["qpos"].append(scene.data.qpos.copy())
        self.records["skill"].append(skill)
        self.records["phase"].append(phase)
        if scene.renderer is not None:
            for role, frame in scene.frames().items():
                self.pixels[role].append(frame)
        issued = scene.step(command)
        self.records["actions"].append(issued)
        for part in ["gear", "collar"]:
            self.records[f"{part}_xyz"].append(scene.data.body(part).xpos.copy())

    def move(self, skill, phase, xyz, opening, speed=1.2, max_ticks=100, until=None):
        np, scene = self.np, self.scene
        start = np.asarray(scene.arm.read_state())
        goal = self.ik(np.asarray(xyz), opening)
        # Smooth joint trajectory, bounded by the shared supervisor. Avoid the old
        # teacher's fixed pause after every transit waypoint.
        duration = max(12, int(np.ceil(np.max(np.abs(goal[:5] - start[:5])) / speed)))
        for tick in range(max_ticks):
            u = min(1.0, (tick + 1) / duration)
            blend = u * u * (3 - 2 * u)
            command = start + blend * (goal - start)
            command[-1] = opening
            self.record_step(skill, phase, command)
            if until is not None and until():
                return True
            error = np.abs(np.asarray(scene.arm.read_state()) - goal)
            if u >= 1 and max(error[:5]) < 1.0 and error[5] < 2:
                return True
        return False

    def hold(self, skill, phase, ticks, until=None):
        command = self.scene.arm.to_policy(self.scene.data.ctrl[self.scene.arm.aids])
        for _ in range(ticks):
            self.record_step(skill, phase, command)
            if until is not None and until():
                return True
        return False

    def pick(self, part):
        np, scene = self.np, self.scene
        skill = f"pick_{part}"
        xy = scene.data.body(part).xpos[:2].copy()
        grip_z = float(scene.data.body(part).xpos[2]) + (0.024 if part == "gear" else 0.020)
        self.move(skill, "above", np.r_[xy, 0.105], 55)
        self.move(skill, "approach", np.r_[xy, grip_z], 55, speed=0.7)
        self.move(
            skill, "close", np.r_[xy, grip_z], 0, max_ticks=55, until=lambda: scene.jaw_ids <= scene.contacts(part)
        )
        if not scene.jaw_ids <= scene.contacts(part):
            return False
        self.hold(skill, "grasp_settle", 6)
        # Use the actual grasp-to-part transform, never an attachment constraint.
        self.mujoco.mj_forward(scene.model, scene.data)
        rotation = scene.data.xmat[self.gripper].reshape(3, 3)
        grip = scene.data.xpos[self.gripper] + rotation @ self.anchor
        self.grip_offsets[part] = {
            "position": rotation.T @ (scene.data.body(part).xpos - grip),
            "axis": rotation.T @ scene.data.body(part).xmat.reshape(3, 3)[:, 2],
        }
        self.axis = self.grip_offsets[part]["axis"]
        lift_z = float(scene.rng.uniform(0.070, 0.077))
        self.move_part(part, "lift", np.r_[scene.data.body(part).xpos[:2].copy(), lift_z], 60, skill=skill)
        if not scene.facts[SUCCESS[skill]]:
            self.hold(skill, "lift_settle", 25, until=lambda: scene.facts[SUCCESS[skill]])
        return bool(scene.facts[SUCCESS[skill]])

    def seat(self, part):
        np, scene = self.np, self.scene
        skill = f"seat_{part}"
        xy = scene.data.body("shaft").xpos[:2].copy()
        bottom = 0.010 if part == "gear" else 0.042
        rotation = scene.data.xmat[self.gripper].reshape(3, 3)
        self.anchor = rotation.T @ (scene.data.body(part).xpos - scene.data.xpos[self.gripper])
        self.axis = rotation.T @ scene.data.body(part).xmat.reshape(3, 3)[:, 2]
        # Keep the bore vertical above the shaft tip before lowering it.
        above = np.r_[xy, 0.070]
        self.move_part(part, "carry", above, 110)
        self.move_part(part, "insert", np.r_[xy, bottom + 0.0005], 90)
        self.hold(skill, "insert_settle", 8)
        self.anchor = self.jaw_anchor.copy()
        self.axis = np.array([0.0, 0.0, 1.0])
        current = scene.data.xpos[self.gripper] + scene.data.xmat[self.gripper].reshape(3, 3) @ self.anchor
        self.move(skill, "release", current.copy(), 55, max_ticks=30)
        self.move(skill, "retract", np.r_[xy, 0.135], 55, speed=0.8, until=lambda: scene.facts[SUCCESS[skill]])
        if not scene.facts[SUCCESS[skill]]:
            self.hold(skill, "seat_settle", 25, until=lambda: scene.facts[SUCCESS[skill]])
        return bool(scene.facts[SUCCESS[skill]])

    async def run(self):
        scene = self.scene
        async with scene.safety.claim("sim", "assembly-demonstration-teacher"):
            for skill in PROMPTS:
                start = len(self.records["states"])
                part = skill.split("_")[1]
                ok = self.pick(part) if skill.startswith("pick") else self.seat(part)
                segment = {
                    "skill": skill,
                    "prompt": PROMPTS[skill],
                    "start": start,
                    "end": len(self.records["states"]),
                    "success": ok,
                    "facts": scene.facts.copy(),
                    "state": scene.state.copy(),
                }
                self.segments.append(segment)
                print(json.dumps({"seed": scene.seed, **segment}), flush=True)
                if not ok:
                    return False
        return bool(scene.facts["assembly_complete"])

    def save(self, output: Path, success):
        np, scene = self.np, self.scene
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "success": success,
            "metadata": scene.metadata,
            "segments": self.segments,
            "frames": len(self.records["states"]),
            "fps": FPS,
            "final_state": scene.state,
            "source": "scripted IK teacher; contact physics only; not learned inference",
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(
            output / "episode.npz",
            **{key: np.asarray(value) for key, value in self.records.items()},
            **{f"images.{key}": np.asarray(value) for key, value in self.pixels.items()},
        )


async def main(args):
    scene = AssemblyScene(args.root, args.seed, render=args.render)
    try:
        teacher = Teacher(scene)
        success = await teacher.run()
        teacher.save(args.output, success)
        print(json.dumps({"seed": args.seed, "assembly_success": success, "frames": len(teacher.records["states"])}))
    finally:
        scene.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    asyncio.run(main(parser.parse_args()))
