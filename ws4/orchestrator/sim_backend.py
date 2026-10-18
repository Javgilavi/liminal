"""Laptop backend: the ws4/sim station with the orchestrator's pack-box scene and scripted stand-ins for FLUX skills.

``ScriptedSimBackend`` implements the ``flux.backends.FluxBackend`` interface (``contract`` + ``predict``), so the
runner, the collision gate, the safety guard and the executor treat it exactly like a FLUX package: one per-arm
target, chunks of 42 commands of which 32 run, no self-termination. Each skill is planned once at start with the sim's
damped least-squares IK (``ws4/sim/ik.py``) and the grasp geometry of ``ws4/sim/scripted_pick.py``, then served
chunk by chunk. Progress is re-anchored to the last executed command at every chunk, so truncation by the collision
gate or the guard's step limit only delays the motion.

This is a stand-in, not FLUX: it reads privileged object poses. ``PackBoxSim.predicate`` is the ground-truth check
the sim verifier uses.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flux.backends import PolicyRequest, PolicyResponse
from flux.contract import ARMS, SO101_CONTRACT, arm_slice

from . import PACKAGE_DIR
from .skills import Skill

SCENE = PACKAGE_DIR / "scenes" / "pack_box.xml"
BOX_XY = np.array([0.25, 0.0])
BOX_INNER_HALF = 0.069  # inner wall faces at +-0.069 m (walls at +-0.072, 3 mm thick)
BOX_RIM_Z = 0.053
LID_HALF = 0.085
LID_KNOB_Z = 0.022  # knob centre above the lid body origin
NOMINAL_XY = {"apple": (0.21, 0.16), "yellow_block": (0.21, -0.16), "lid": (0.08, -0.30)}
LID_PARK_XY = {"right": (0.08, -0.30), "left": (0.08, 0.30)}
PLACE_DY = {"left": 0.045, "right": -0.045}  # each arm drops on its own half of the box
DROP_TCP_Z = 0.10
MAX_JOINT_STEP_DEG = 2.5  # per 30 Hz tick; the runner's guard allows 4
MAX_GRIPPER_STEP = 6.0


class SimSkillError(RuntimeError):
    """The scripted stand-in cannot run this skill here (no sim binding, unreachable target)."""


def _yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _tilt(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    zz = 1 - 2 * (x * x + y * y)  # body z axis . world z
    return float(np.arccos(np.clip(zz, -1.0, 1.0)))


def _wrap(a: float) -> float:
    return float(np.angle(np.exp(1j * a)))


class PackBoxSim:
    """The env plus scene facts: layouts, ground-truth predicates, object positions."""

    def __init__(self, image_size: tuple[int, int] = (240, 320), randomize: bool = True, render: bool = True,
                 cameras: tuple[str, ...] = ("head", "left_wrist", "right_wrist")):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from sim.env import XLeRobotStationEnv

        self.env = XLeRobotStationEnv(scene=SCENE, randomize=randomize, render_images=render, cameras=cameras,
                                      image_size=image_size)
        self.lock = threading.RLock()  # MuJoCo state and the GL context belong to the control thread
        self.layout: dict[str, Any] = {}

    def reset(self, seed: int | None = 0, lid_on_box: bool = False, jitter: float = 0.02) -> dict:
        import mujoco

        with self.lock:
            env = self.env
            env.reset(seed=seed)
            m, d = env.model, env.data
            rng = np.random.default_rng(None if seed is None else seed + 1000)
            placed = {}
            for name, (x, y) in NOMINAL_XY.items():
                xy = np.array([x, y]) + (rng.uniform(-jitter, jitter, 2) if jitter else 0.0)
                z = {"apple": 0.019, "yellow_block": 0.0201, "lid": 0.004}[name]
                yaw = rng.uniform(-0.4, 0.4) if jitter else 0.0
                if name == "lid" and lid_on_box:
                    xy, z, yaw = BOX_XY.copy(), BOX_RIM_Z + 0.0045, 0.0
                adr = m.jnt_qposadr[m.joint(f"{name}_joint").id]
                d.qpos[adr:adr + 3] = [xy[0], xy[1], z]
                d.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                dof = m.jnt_dofadr[m.joint(f"{name}_joint").id]
                d.qvel[dof:dof + 6] = 0
                placed[name] = [round(float(xy[0]), 3), round(float(xy[1]), 3), round(yaw, 3)]
            mujoco.mj_forward(m, d)
            for _ in range(int(0.3 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            env.step_count = 0
            self.layout = {"seed": seed, "lid_on_box": lid_on_box, "objects": placed}
            return env.get_obs()

    # ------------------------------------------------------------------ facts
    def poses(self) -> dict[str, np.ndarray]:
        with self.lock:
            return self.env.object_poses()

    def predicate(self, check: dict, obs: dict, arm: str | None = None) -> tuple[bool, str]:
        poses = obs["privileged"]["object_poses"]
        if "in_box" in check:
            name = check["in_box"]
            p = poses[name][:3]
            rel = np.abs(p[:2] - BOX_XY)
            ok = bool((rel < BOX_INNER_HALF).all() and p[2] < BOX_RIM_Z + 0.01)
            return ok, f"{name} at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}), {'inside' if ok else 'not inside'} the box"
        if "lid_on_box" in check or "lid_off_box" in check:
            pose = poses["lid"]
            off = float(np.linalg.norm(pose[:2] - BOX_XY))
            on = off < 0.025 and BOX_RIM_Z < pose[2] < BOX_RIM_Z + 0.02 and _tilt(pose[3:]) < 0.25
            want_on = "lid_on_box" in check
            ok = on if want_on else (not on and pose[2] < 0.03)
            return ok, (f"lid centre {off * 100:.1f} cm from the box centre at z {pose[2]:.3f}, "
                        f"tilt {_tilt(pose[3:]):.2f} rad")
        raise ValueError(f"unknown sim check {check}")

    def side_of(self, obj: str) -> str:
        return ARMS[0] if self.poses()[obj][1] >= 0 else ARMS[1]


class SimRobot:
    """``RobotInterface`` over the sim (same contract as ``flux.dual_arm_runner.GymEnvRobot``), with a custom reset."""

    def __init__(self, sim: PackBoxSim, seed: int | None = 0, lid_on_box: bool = False):
        self.sim = sim
        self._obs = sim.reset(seed, lid_on_box)
        self.steps = 0

    def reset(self, seed: int | None = 0, lid_on_box: bool = False) -> dict:
        self._obs = self.sim.reset(seed, lid_on_box)
        self.steps = 0
        return self._obs

    def get_observation(self) -> dict:
        return self._obs

    def send_action(self, action: np.ndarray) -> None:
        with self.sim.lock:
            out = self.sim.env.step(np.asarray(action, dtype=np.float32))
        self._obs = out[0]
        self.steps += 1


@dataclass
class _Plan:
    skill: str
    dense: np.ndarray  # (N, 6) robot units
    labels: list[str]
    last_start: int | None = None
    cursor: int = 0
    info: dict = field(default_factory=dict)


class ScriptedSimBackend:
    """Per-arm scripted skills in the sim behind the FLUX backend interface."""

    def __init__(self, sim: PackBoxSim, home_deg: np.ndarray | None = None):
        from sim.env import HOME_DEG

        self.sim = sim
        self.contract = SO101_CONTRACT
        self.home = np.asarray(HOME_DEG if home_deg is None else home_deg, dtype=float)
        self._plans: dict[str, _Plan] = {}
        self._iks: dict[str, Any] = {}

    # ------------------------------------------------------------------ planning
    def _ik(self, arm: str):
        if arm not in self._iks:
            from sim.ik import ArmIK

            self._iks[arm] = ArmIK(self.sim.env.model, arm)
        return self._iks[arm]

    def begin(self, target: str, arm: str, skill: Skill) -> dict:
        if not skill.sim:
            raise SimSkillError(f"{skill.name} has no scripted sim binding")
        with self.sim.lock:
            plan = self._plan_skill(arm, skill)
        self._plans[target] = plan
        return plan.info

    def end(self, target: str) -> None:
        self._plans.pop(target, None)

    def finished(self, target: str, state6: np.ndarray | None = None) -> bool:
        plan = self._plans.get(target)
        if plan is None or plan.cursor < len(plan.dense) - 1:
            return False
        if state6 is None:
            return True
        return bool(np.abs(np.asarray(state6)[:5] - plan.dense[-1][:5]).max() < 3.0)

    def progress(self, target: str) -> tuple[int, int, str]:
        plan = self._plans.get(target)
        if plan is None:
            return 0, 0, ""
        i = min(plan.cursor, len(plan.dense) - 1)
        return i, len(plan.dense), plan.labels[i]

    def _plan_skill(self, arm: str, skill: Skill) -> _Plan:
        from sim.ik import top_down_rotation

        from ws4.sim.scripted_pick import grasp_offset_x, gripper_opening_pct

        env = self.sim.env
        m, d = env.model, env.data
        spec = skill.sim
        action = spec["action"]
        units = env.units.arms[arm]
        state6 = env.joint_state()[arm_slice(arm)].astype(float)
        poses = env.object_poses()
        base = d.xpos[m.body(f"{arm}_base").id].copy()

        def heading_to(p) -> float:
            return float(np.arctan2(p[1] - base[1], p[0] - base[0]))

        def pick_yaw(obj_yaw: float, p, square: bool) -> float:
            want = heading_to(p) + np.pi / 2  # fingers open across the reach direction: small wrist roll
            cands = [obj_yaw + k * np.pi / 2 for k in range(4)] if square else [want]
            return min(cands, key=lambda a: abs(_wrap(a - want)))

        if action == "pick_place":
            obj = spec["object"]
            pose = poses[obj]
            grasp_point = pose[:3].copy()
            width = float(spec.get("width", 0.04))
            square = obj != "apple"
            g_yaw = pick_yaw(_yaw(pose[3:]), grasp_point, square)
            place_point = np.array([BOX_XY[0], BOX_XY[1] + PLACE_DY[arm], DROP_TCP_Z])
            p_yaw = heading_to(place_point) + np.pi / 2
        elif action in ("place_lid", "remove_lid"):
            obj = "lid"
            pose = poses["lid"]
            grasp_point = pose[:3] + np.array([0, 0, LID_KNOB_Z + 0.004])
            width = 0.036
            g_yaw = pick_yaw(_yaw(pose[3:]), grasp_point, True)
            rel = g_yaw - _yaw(pose[3:])  # gripper yaw relative to the lid, kept while carrying
            if action == "place_lid":
                target_lid = np.array([BOX_XY[0], BOX_XY[1], BOX_RIM_Z + 0.004 + 0.006])
                lid_yaw = 0.0
            else:
                px, py = LID_PARK_XY[arm]
                target_lid = np.array([px, py, 0.004 + 0.004])
                lid_yaw = _yaw(pose[3:])
            place_point = target_lid + np.array([0, 0, LID_KNOB_Z + 0.004])
            want = heading_to(place_point) + np.pi / 2
            p_yaw = min((lid_yaw + rel + k * np.pi / 2 for k in range(4)), key=lambda a: abs(_wrap(a - want)))
        else:
            raise SimSkillError(f"unknown sim action {action!r}")

        open_pct = gripper_opening_pct(width)
        Rg, Rp = top_down_rotation(g_yaw), top_down_rotation(p_yaw)
        # Object centre sits this far along the TCP x axis, just clear of the fixed fin (ws4/sim/scripted_pick.py);
        # round objects get more clearance so the descending fin does not push them.
        off = grasp_offset_x(width, 0.008 if obj == "apple" else 0.004)
        grasp_tcp = grasp_point - Rg[:, 0] * off
        place_tcp = place_point - Rp[:, 0] * off
        carry = max(grasp_tcp[2], place_tcp[2]) + 0.045
        # (label, tcp, rotation, gripper %, dwell ticks)
        waypoints = [
            ("approach", grasp_tcp + [0, 0, 0.08], Rg, open_pct, 0),
            ("descend", grasp_tcp + [0, 0, 0.03], Rg, open_pct, 0),
            ("descend", grasp_tcp, Rg, open_pct, 4),
            ("grasp", grasp_tcp, Rg, 0.0, 18),
            ("lift", np.r_[grasp_tcp[:2], carry], Rg, 0.0, 0),
            ("carry", np.r_[place_tcp[:2], carry], Rp, 0.0, 0),
            ("lower", place_tcp, Rp, 0.0, 6),
            ("release", place_tcp, Rp, open_pct, 12),
            ("clear", place_tcp + [0, 0, 0.045], Rp, open_pct, 0),
        ]
        ik = self._ik(arm)
        q_full = d.qpos.copy()
        q5 = units.to_radians(state6)[:5]
        targets: list[tuple[str, np.ndarray, int]] = []
        errs = []
        for label, tcp, R, grip, dwell in waypoints:
            q5, ep, er = ik.solve(q_full, np.asarray(tcp, float), R, q_init=q5)
            errs.append((label, round(ep, 4), round(er, 3)))
            if ep > 0.012 and label in ("descend", "grasp", "lower", "release"):
                raise SimSkillError(f"{skill.name}: {label} point unreachable for the {arm} arm "
                                    f"(IK error {ep * 100:.1f} cm)")
            q6 = units.from_radians(np.r_[q5, units.gripper_closed_rad])
            q6[5] = grip
            targets.append((label, q6, dwell))
        targets.append(("home", self.home.copy(), 0))
        dense, labels = self._densify(state6, targets)
        info = {"skill": skill.name, "arm": arm, "object": obj, "ik_errors": errs, "frames": len(dense)}
        return _Plan(skill.name, dense, labels, info=info)

    @staticmethod
    def _densify(start: np.ndarray, targets: list[tuple[str, np.ndarray, int]]) -> tuple[np.ndarray, list[str]]:
        frames = [np.asarray(start, float)]
        labels = ["start"]
        cur = frames[0]
        for label, goal, dwell in targets:
            body = np.abs(goal[:5] - cur[:5]).max()
            grip = abs(goal[5] - cur[5])
            # Cosine profile: peak step is pi/2 times the mean step.
            n = int(max(6, np.ceil(np.pi / 2 * body / MAX_JOINT_STEP_DEG),
                        np.ceil(np.pi / 2 * grip / MAX_GRIPPER_STEP)))
            s = (1 - np.cos(np.pi * np.arange(1, n + 1) / n)) / 2
            seg = cur[None] + (goal - cur)[None] * s[:, None]
            frames.extend(seg)
            labels.extend([label] * n)
            frames.extend([goal] * dwell)
            labels.extend([label] * dwell)
            cur = goal
        return np.asarray(frames), labels

    # ------------------------------------------------------------------ FluxBackend interface
    def predict(self, request: PolicyRequest) -> PolicyResponse:
        plan = self._plans.get(request.target)
        anchor = np.asarray(request.commands[-1], dtype=float)
        n_chunk, n_exec = self.contract.chunk_size, self.contract.n_action_steps
        if plan is None:  # no skill: hold
            return PolicyResponse(np.repeat(anchor[None], n_chunk, 0), 0.0, {"backend": "sim_scripted", "idle": True})
        if plan.last_start is not None:
            lo = min(plan.last_start, len(plan.dense) - 1)
            hi = min(lo + n_exec, len(plan.dense))
            dist = np.abs(plan.dense[lo:hi] - anchor[None]).max(axis=1)
            best = dist.min()
            # Latest frame among the closest ones: holds (dwells, gate refusals) repeat identical frames.
            j = lo + int(np.flatnonzero(dist <= best + 1e-6)[-1])
            plan.cursor = j + 1
        idx = np.minimum(np.arange(plan.cursor, plan.cursor + n_chunk), len(plan.dense) - 1)
        plan.last_start = plan.cursor
        chunk = plan.dense[idx].copy()
        return PolicyResponse(chunk, 0.0, {"backend": "sim_scripted", "phase": plan.labels[idx[0]]})


def arm_state(obs: dict, arm: str) -> np.ndarray:
    return np.asarray(obs["observation.state"], dtype=float)[arm_slice(arm)]


__all__ = ["PackBoxSim", "SimRobot", "ScriptedSimBackend", "SimSkillError", "SCENE", "arm_state"]
