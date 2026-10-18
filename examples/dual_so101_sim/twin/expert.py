"""Scripted expert demonstrations in the twin: pick-and-place into the box and opening the lid.

An expert is a list of Cartesian waypoints for the tool point (between the jaws). `run_episode`
interpolates them at 30 Hz, solves IK every tick, drives the arm through position actuators with the
same servo ramp as the simulator app, and records what the policy will see and do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from . import config as C
from .ik import ArmIK, frame
from .scene import JOINTS, object_extent

CONTROL_HZ = 30.0
OPEN, CLOSED = 30.0, 0.0  # gripper %, 30 % opens the fingertips ~45 mm
JAW_CENTRE = 0.022  # m from the fixed jaw face: middle of the 30 % opening
TASKS = {
    "open_box": ("left", None, "Open the box"),
    "place_green_marker": ("right", "green_marker", "Put the green marker in the box"),
    "place_red_marker": ("right", "red_marker", "Put the red marker in the box"),
    "place_cutter": ("right", "cutter", "Put the cutter in the box"),
}


@dataclass
class Waypoint:
    pos: np.ndarray        # tool point, world (m)
    R: np.ndarray          # gripper orientation (columns: approach, -, jaw axis)
    grip: float            # gripper command %
    seconds: float
    width: float = 0.017   # object thickness held between the jaws (tool offset)
    depth: float = 0.012   # how far the object sits behind the fingertips
    joint: bool = False    # interpolate in joint space (large transit moves)
    grip_force: float = 0.5  # gripper torque cap (N m) while executing this waypoint
    correct: bool = False  # afterwards, servo the measured tool point onto the target (closed loop)

    centre: float = 0.0    # jaw-axis offset of the object from the fixed jaw face; 0 = JAW_CENTRE

    @property
    def offset(self):
        # Centre the object in the open jaws (fixed face z=0, moving tip ~47 mm at 30 %), so neither
        # finger lands on it; closing then pushes it against the fixed jaw.
        return (-self.depth, 0.0, max(self.width / 2, self.centre or JAW_CENTRE))


def _smooth(s):
    return s * s * (3 - 2 * s)


class Arm:
    def __init__(self, model, data, arm):
        self.m, self.d, self.arm = model, data, arm
        self.ik = ArmIK(model, arm)
        self.act = [model.actuator(f"{arm}_{j}").id for j in JOINTS]
        self.qadr = [model.jnt_qposadr[model.joint(f"{arm}_{j}").id] for j in JOINTS]
        g = model.joint(f"{arm}_gripper").id
        self.glo, self.ghi = model.jnt_range[g]

    def state(self):
        q = self.d.qpos[self.qadr]
        return [math.degrees(v) for v in q[:5]] + [float((q[5] - self.glo) / (self.ghi - self.glo) * 100)]

    def command(self, q_arm_rad, grip_pct):
        ctrl = list(q_arm_rad) + [self.glo + np.clip(grip_pct, 0, 100) / 100 * (self.ghi - self.glo)]
        self.d.ctrl[self.act] = ctrl
        return [math.degrees(v) for v in q_arm_rad] + [float(grip_pct)]

    def tool_pose(self):
        site = self.m.site(f"{self.arm}_gripperframe").id
        return self.d.site_xpos[site].copy(), self.d.site_xmat[site].reshape(3, 3).copy()


def step_physics(model, data, start_ctrl, substeps):
    """One control tick with the commanded targets ramped across the substeps (servo-like)."""
    target = data.ctrl.copy()
    for i in range(substeps):
        data.ctrl[:] = start_ctrl + (target - start_ctrl) * (i + 1) / substeps
        mujoco.mj_step(model, data)
    return target


def rot_about(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def object_yaw(data, name):
    R = data.body(name).xmat.reshape(3, 3)
    return math.atan2(R[1, 0], R[0, 0])


def pick_place_waypoints(model, data, obj, rng, speed=1.0, miss=False):
    """Approach, grasp across the object's long axis, lift, carry to its slot in the box, release."""
    p = data.body(obj).xpos.copy()
    width = C.MARKER["radius"] * 2 if "marker" in obj else C.CUTTER["size"][1]  # jaws close across the cutter's width
    # The fingertip hull reaches 6.3 mm past the gripper site: keep it 4 mm above the mat.
    depth = float(np.clip(p[2] - C.MAT["size"][2] - 0.0063 - 0.004, -0.004, 0.012))
    jaw = object_yaw(data, obj) + math.pi / 2
    box = data.body("box")
    bR = box.xmat.reshape(3, 3)
    box_yaw = math.atan2(bR[1, 0], bR[0, 0])
    ox, oy, oz = C.BOX["outer"]
    slot = {"green_marker": (-ox * 0.18, 0.03), "red_marker": (-ox * 0.18, -0.03), "cutter": (ox * 0.18, 0.0)}[obj]
    drop = box.xpos + bR @ np.array([slot[0], slot[1], 0.0])
    drop_z = C.BOX["wall"] + 0.02 + width / 2
    noise = lambda s=0.004: rng.normal(0, s, 3) * np.array([1, 1, 0.3])  # noqa: E731
    down = lambda yaw: frame((0, 0, -1), (math.cos(yaw), math.sin(yaw), 0))  # noqa: E731
    s = 1.0 / speed
    wps = []
    if miss:  # a deliberately offset first attempt, then a correction (teaches recovery)
        off = rng.normal(0, 1, 2)
        off = 0.018 * off / np.linalg.norm(off)
        bad = p + np.array([off[0], off[1], 0.0])
        wps += [Waypoint(bad + [0, 0, 0.09], down(jaw), OPEN, 1.5 * s, width, depth),
                Waypoint(bad + [0, 0, 0.005], down(jaw), OPEN, 0.8 * s, width, depth),
                Waypoint(bad + [0, 0, 0.05], down(jaw), OPEN, 0.6 * s, width, depth)]
    hold, carry = 0.5, 0.2
    wps += [
        Waypoint(p + [0, 0, 0.09] + noise(), down(jaw), OPEN, (1.0 if miss else 1.6) * s, width, depth, joint=True),
        Waypoint(p + [0, 0, 0.025], down(jaw), OPEN, 0.7 * s, width, depth, correct=True),
        Waypoint(p + [0, 0, 0.0], down(jaw), OPEN, 0.9 * s, width, depth, correct=True),
        Waypoint(p + [0, 0, 0.0], down(jaw), CLOSED, 0.6 * s, width, depth, grip_force=hold),
        Waypoint(p + [0, 0, 0.0], down(jaw), CLOSED, 0.25 * s, width, depth, grip_force=hold),
        Waypoint(p + [0, 0, 0.12] + noise(), down(jaw), CLOSED, 0.9 * s, width, depth, grip_force=carry),
        Waypoint(np.array([drop[0], drop[1], oz + 0.10]) + noise(), down(box_yaw + math.pi / 2), CLOSED, 1.6 * s, width, depth, joint=True, grip_force=carry),
        Waypoint(np.array([drop[0], drop[1], drop_z]), down(box_yaw + math.pi / 2), CLOSED, 0.9 * s, width, depth, grip_force=carry, correct=True),
        Waypoint(np.array([drop[0], drop[1], drop_z]), down(box_yaw + math.pi / 2), OPEN, 0.5 * s, width, depth),
        Waypoint(np.array([drop[0], drop[1], oz + 0.12]), down(box_yaw + math.pi / 2), OPEN, 0.9 * s, width, depth),
    ]
    return wps


def open_box_waypoints(model, data, rng, speed=1.0):
    """Grip the lid's front flap, swing the lid up about its hinge to 75-85 deg, let go.

    The taped hinge's friction holds the lid open; beyond ~85 deg the left arm reaches its joint limits."""
    flap = data.site("lid_flap_site").xpos.copy()
    box = data.body("box")
    bR = box.xmat.reshape(3, 3)
    hinge_axis = bR @ np.array([-1.0, 0, 0])
    hinge = box.xpos + bR @ np.array([0, C.BOX["outer"][1] / 2, C.BOX["outer"][2]])
    R0 = frame((0, 0, -1), bR @ np.array([0, 1.0, 0]))  # jaws close across the flap
    width, depth = C.BOX["lid_thickness"], 0.008  # pinch the flap 8 mm behind the fingertips
    grip = flap + [0, 0, -0.002]
    s = 1.0 / speed
    # The flap cannot slide, so open the jaws only ~25 mm and centre the flap in that opening.
    wp = lambda pos, R, g, t, **kw: Waypoint(np.asarray(pos, float), R, g, t * s, width, depth, centre=0.012, **kw)  # noqa: E731
    lid_open = 15.0
    wps = [wp(flap + [0, 0, 0.09], R0, lid_open, 1.6, joint=True),
           wp(grip, R0, lid_open, 0.9, correct=True),
           wp(grip, R0, CLOSED, 0.6),
           wp(grip, R0, CLOSED, 0.25)]
    target_deg = rng.uniform(75, 85)
    for k in range(1, 7):  # swing in arcs so the interpolation follows the circle
        Ra = rot_about(hinge_axis, math.radians(target_deg * k / 6))
        wps.append(wp(hinge + Ra @ (grip - hinge), Ra @ R0, CLOSED, 0.45))
    Rend, end_pos = wps[-1].R, wps[-1].pos
    off = end_pos + Rend @ np.array([-0.025, 0, 0])  # slide the open jaws off the flap
    wps += [wp(end_pos, Rend, lid_open, 0.5), wp(off, Rend, lid_open, 0.6), wp(off + [0, 0, 0.04], Rend, lid_open, 0.6)]
    return wps


def plan_waypoints(model, data, arm, wps, tol=0.004):
    """Joint solution for every waypoint (robust multi-start, chained). None if any is unreachable."""
    q, sols = data.qpos[arm.ik.qadr].copy(), []
    for wp in wps:
        q, ep, er = arm.ik.solve_robust(data.qpos, wp.pos, wp.R, offset=wp.offset, q0=q)
        if ep > tol or er > 0.05:
            return None
        sols.append(q.copy())
    return sols


def make_waypoints(model, data, task, rng, speed, miss):
    _, obj, _ = TASKS[task]
    return (open_box_waypoints(model, data, rng, speed) if task == "open_box"
            else pick_place_waypoints(model, data, obj, rng, speed, miss))


def run_episode(model, data, task, rng, record=None, speed=None, miss=None):
    """Execute one scripted demonstration. `record(tick, arm, state, action)` is called every tick.

    Returns a dict with success and diagnostics, or {'feasible': False} if the layout is unreachable."""
    arm_name, obj, _ = TASKS[task]
    arm = Arm(model, data, arm_name)
    speed = speed if speed is not None else float(rng.uniform(0.85, 1.2))
    miss = miss if miss is not None else bool(rng.random() < 0.15)
    wps = make_waypoints(model, data, task, rng, speed, miss)
    sols = plan_waypoints(model, data, arm, wps)
    if sols is None:
        return {"task": task, "feasible": False, "success": False}
    substeps = round(1 / CONTROL_HZ / model.opt.timestep)
    ctrl = data.ctrl.copy()
    pos, R = arm.tool_pose()
    pos = pos + R @ np.array(wps[0].offset)
    grip = arm.state()[5]
    q = data.qpos[arm.ik.qadr].copy()
    tick, worst_ik = 0, 0.0
    grip_act = model.actuator(f"{arm_name}_gripper").id
    for wp, q_goal in zip(wps, sols):
        n = max(1, round(wp.seconds * CONTROL_HZ))
        R_from, q_from = R, q.copy()
        model.actuator_forcerange[grip_act] = [-wp.grip_force, wp.grip_force]
        for i in range(1, n + 1):
            s = _smooth(i / n)
            seed = q_from + (q_goal - q_from) * s
            if wp.joint:
                q = seed
            else:
                p_t = pos + (wp.pos - pos) * s
                # Orientation: interpolate the approach and jaw axes, then re-orthogonalise.
                R_t = frame(R_from[:, 0] * (1 - s) + wp.R[:, 0] * s, R_from[:, 2] * (1 - s) + wp.R[:, 2] * s)
                q_try, ep, _ = arm.ik.solve(data.qpos, p_t, R_t, offset=wp.offset, q0=q, iters=30)
                if ep > 0.005 or np.max(np.abs(q_try - seed)) > 0.6:  # drifting to another branch: follow the plan
                    q_try, ep, _ = arm.ik.solve(data.qpos, p_t, R_t, offset=wp.offset, q0=seed, iters=40)
                q = q_try
                worst_ik = max(worst_ik, ep)
            g = grip + (wp.grip - grip) * s
            state = arm.state()
            action = arm.command(q, g)
            if record is not None:
                record(tick, arm_name, state, action)
            ctrl = step_physics(model, data, ctrl, substeps)
            tick += 1
        if wp.correct:
            aim = wp.pos.copy()
            for _ in range(12):
                p_now, R_now = arm.tool_pose()
                err = wp.pos - (p_now + R_now @ np.array(wp.offset))
                err[2] = 0.0  # horizontal only: a vertical shortfall is usually contact, not lag
                if np.linalg.norm(err) < 0.0025:
                    break
                aim = aim + 0.7 * err
                shift = aim[:2] - wp.pos[:2]
                if np.linalg.norm(shift) > 0.015:
                    aim[:2] = wp.pos[:2] + 0.015 * shift / np.linalg.norm(shift)
                q, ep, _ = arm.ik.solve(data.qpos, aim, wp.R, offset=wp.offset, q0=q, iters=30)
                if record is not None:
                    record(tick, arm_name, arm.state(), arm.command(q, wp.grip))
                else:
                    arm.command(q, wp.grip)
                ctrl = step_physics(model, data, ctrl, substeps)
                tick += 1
        pos, R, grip = wp.pos, wp.R, wp.grip
    for _ in range(int(0.5 * CONTROL_HZ)):  # let things settle, still recorded
        if record is not None:
            record(tick, arm_name, arm.state(), arm.command(q, grip))
        ctrl = step_physics(model, data, ctrl, substeps)
        tick += 1
    return {"task": task, "ticks": tick, "speed": speed, "miss": miss, "worst_ik_m": worst_ik,
            **check_success(model, data, task)}


def check_success(model, data, task):
    if task == "open_box":
        lid = math.degrees(data.qpos[model.jnt_qposadr[model.joint("lid_hinge").id]])
        return {"success": lid > 65.0, "lid_deg": round(lid, 1)}
    _, obj, _ = TASKS[task]
    box = data.body("box")
    Rb = box.xmat.reshape(3, 3)
    local = Rb.T @ (data.body(obj).xpos - box.xpos)
    # Both ends of the object's long axis must be inside the inner walls and below the rim: the centre
    # alone passed a marker lying across a wall with one end outside the box.
    half, _ = object_extent(obj)
    axis = data.body(obj).xmat.reshape(3, 3)[:, 0] * half
    ends = [Rb.T @ (data.body(obj).xpos + s * axis - box.xpos) for s in (-1, 1)]
    ox, oy, oz = C.BOX["outer"]
    ix, iy = ox / 2 - C.BOX["wall"], oy / 2 - C.BOX["wall"]
    inside = all(abs(p[0]) < ix and abs(p[1]) < iy and p[2] < oz for p in ends)
    touching = any(model.body(model.geom_bodyid[c.geom1]).name.startswith(("left_", "right_")) and
                   model.geom_bodyid[c.geom2] == model.body(obj).id or
                   model.body(model.geom_bodyid[c.geom2]).name.startswith(("left_", "right_")) and
                   model.geom_bodyid[c.geom1] == model.body(obj).id for c in data.contact[:data.ncon])
    return {"success": bool(inside and not touching), "object_in_box_local": np.round(local, 3).tolist(),
            "object_ends_in_box_local": [np.round(p, 3).tolist() for p in ends]}
