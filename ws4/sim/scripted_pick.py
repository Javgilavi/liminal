"""Scripted top-down pick with IK, driven through the env's LeRobot-unit action interface.

    MUJOCO_GL=egl python ws4/sim/scripted_pick.py --object cube_red --side left [--video out.mp4]

Used by tests/test_grasp.py to check that the soft fin gripper can grasp and lift a cube and the cup.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ws4.sim.geometry import SIDES, load_geometry, val  # noqa: E402
from ws4.sim.ik import ArmIK, top_down_rotation  # noqa: E402

# Grasp widths (m) across the gripper opening direction, and grasp height above the object origin.
OBJECTS = {"cube": {"width": 0.040, "z": 0.0}, "cup": {"width": 0.072, "z": 0.035},
           "target": {"width": 0.036, "z": 0.0}}


def _kind(name: str) -> str:
    return "cube" if name.startswith("cube") else name


def grasp_offset_x(width: float, margin: float = 0.004) -> float:
    """Where the object centre must sit along the TCP x axis so the open moving fin closes it onto the fixed fin.

    The fixed fin does not move, so the object is placed just clear of it, not at the TCP.
    """
    g = load_geometry()
    fixed_contact_x = val(g["gripper"]["fixed_jaw_inner_x"]) + 0.0025  # the tip hook protrudes 2.5 mm
    return fixed_contact_x + width / 2 + margin - val(g["tcp"]["pos_in_gripper"])[0]


def gripper_opening_pct(width: float) -> float:
    """Open enough for `width` plus 25 mm clearance (linear in angle, rough)."""
    return float(np.clip(100.0 * (width + 0.025) / 0.12, 25, 100))


def run_pick(env, obj: str, side: str = "left", seed: int | None = None, frames: list | None = None,
             lift: float = 0.12, steps_per_segment: int = 40) -> dict:
    m, d = env.model, env.data
    ik = ArmIK(m, side)
    i0 = SIDES.index(side) * 6
    kind = OBJECTS[_kind(obj)]
    b = m.body(obj).id
    obj_pos = d.xpos[b].copy()
    # Pick the grasp yaw closest to the arm's current heading among the object's symmetric yaws (cubes: 90 deg).
    R_obj = d.xmat[b].reshape(3, 3)
    obj_yaw = np.arctan2(R_obj[1, 0], R_obj[0, 0])
    base = m.body(f"{side}_base").id
    heading = np.arctan2(obj_pos[1] - d.xpos[base][1], obj_pos[0] - d.xpos[base][0])
    cands = [obj_yaw + k * np.pi / 2 for k in range(4)] if _kind(obj) == "cube" else [heading + np.pi / 2]
    # Opening direction perpendicular-ish to the reach direction keeps the wrist roll small.
    yaw = min(cands, key=lambda a: abs(np.angle(np.exp(1j * (a - heading - np.pi / 2)))))
    Rg = top_down_rotation(yaw)
    off = grasp_offset_x(kind["width"])
    grasp_tcp = obj_pos + np.array([0, 0, kind["z"]]) - Rg[:, 0] * off
    waypoints = [("pre", grasp_tcp + [0, 0, 0.08], gripper_opening_pct(kind["width"])),
                 ("descend", grasp_tcp, gripper_opening_pct(kind["width"])),
                 ("close", grasp_tcp, 0.0),
                 ("lift", grasp_tcp + [0, 0, lift], 0.0)]
    state = env.joint_state().astype(float)
    q_full = d.qpos.copy()
    q5 = d.qpos[ik.qadr].copy()
    log = {"object": obj, "side": side, "ik_err": []}
    cur = state.copy()
    for name, tcp, grip in waypoints:
        q5, ep, er = ik.solve(q_full, tcp, Rg, q_init=q5)
        log["ik_err"].append((name, ep, er))
        q6 = np.concatenate([q5, [0.0]])
        target_units = env.units.arms[side].from_radians(q6)
        target_units[5] = grip
        goal = cur.copy()
        goal[i0:i0 + 6] = target_units
        n = steps_per_segment if name != "close" else steps_per_segment // 2
        for k in range(1, n + 1):
            a = cur + (goal - cur) * k / n
            obs, *_ = env.step(a)
            if frames is not None:
                frames.append(obs.get("observation.images.head", env.render("overview")))
        for _ in range(10):  # settle
            env.step(goal)
        cur = goal
    z_obj = d.xpos[b][2]
    tcp_now, _ = env.ee_pose(side)
    log.update(lift_height=float(z_obj - obj_pos[2]), obj_tcp_dist=float(np.linalg.norm(d.xpos[b] - tcp_now)),
               success=bool(z_obj - obj_pos[2] > 0.6 * lift and np.linalg.norm(d.xpos[b] - tcp_now) < 0.08))
    return log


def main() -> None:
    from ws4.sim.env import XLeRobotStationEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=None)
    ap.add_argument("--object", default="cube_red")
    ap.add_argument("--side", default="left", choices=SIDES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--video", default=None, help="write an mp4 of the overview camera")
    a = ap.parse_args()
    from ws4.sim.geometry import SIM_DIR

    scene = a.scene or str(SIM_DIR / "scenes" / "grasp_test.xml")
    env = XLeRobotStationEnv(scene=scene, randomize=a.randomize, render_images=a.video is not None,
                             cameras=("head",))
    env.reset(seed=a.seed)
    frames = [] if a.video else None
    log = run_pick(env, a.object, a.side, frames=frames)
    print(log)
    if a.video:
        import imageio

        imageio.mimsave(a.video, frames, fps=30)
    env.close()


if __name__ == "__main__":
    main()
