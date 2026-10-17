"""Generate independently verified, scripted simulation demonstrations for adaptation."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from common import contact_flags, images, make_episode, outcome


async def episode(seed, output, render):
    import mujoco

    model, data, arm, safety, randomization = make_episode(seed)
    scratch = mujoco.MjData(model)
    gids = [model.geom(n).id for n in ("fixed_jaw_sph_tip1", "moving_jaw_sph_tip1")]
    gripper = model.body("gripper").id
    scratch.qpos[:] = data.qpos
    best = None
    for opening in np.linspace(0, 60, 61):
        scratch.qpos[arm.qadr[-1]] = arm.to_sim([0, 0, 0, 0, 0, opening])[-1]
        mujoco.mj_forward(model, scratch)
        points = scratch.geom_xpos[gids]
        score = abs(np.linalg.norm(points[1] - points[0]) - 0.038)
        if best is None or score < best[0]:
            anchor = scratch.xmat[gripper].reshape(3, 3).T @ (points.mean(axis=0) - scratch.xpos[gripper])
            best = (score, anchor.copy())
    anchor = best[1]
    orientation_weight = 0.08

    def residual(values, target):
        scratch.qpos[:] = data.qpos
        scratch.qpos[arm.qadr] = values
        mujoco.mj_forward(model, scratch)
        rotation = scratch.xmat[gripper].reshape(3, 3)
        return np.r_[
            scratch.xpos[gripper] + rotation @ anchor - target, orientation_weight * (rotation[:, 2] - [0, 0, 1])
        ]

    def ik(target, opening, initial):
        values = initial.copy()
        values[-1] = arm.to_sim([0, 0, 0, 0, 0, opening])[-1]
        for _ in range(45):
            r = residual(values, target)
            jac = np.column_stack([(residual(values + np.eye(6)[i] * 1e-5, target) - r) / 1e-5 for i in range(5)])
            step = np.linalg.solve(jac.T @ jac + np.eye(5) * 1e-5, -jac.T @ r)
            values[:5] += np.clip(step, -0.15, 0.15)
            values[:5] = np.clip(values[:5], arm.ranges[:5, 0] + 0.01, arm.ranges[:5, 1] - 0.01)
            if np.linalg.norm(r) < 1e-4:
                break
        return values

    phases = [
        ("above", 90, 0.07, 55),
        ("approach", 90, 0.019, 55),
        ("close", 90, 0.019, 0),
        ("lift", 150, 0.10, 0),
        ("carry", 150, 0.09, 0),
        ("release", 90, 0.09, 55),
        ("settle", 90, 0.12, 55),
    ]
    start_xy = data.body("block").xpos[:2].copy()
    initial_height = float(data.body("block").xpos[2])
    q = data.qpos[arm.qadr].copy()
    renderer = mujoco.Renderer(model, height=480, width=640) if render else None
    records = {k: [] for k in ("states", "actions", "qpos", "block_xyz", "phase", "bilateral", "inside_bowl")}
    pixels = {"images.scene": [], "images.wrist": []}
    stable = 0
    try:
        async with safety.claim("sim", "synthetic-demonstration-teacher"):
            for phase, limit, z, opening in phases:
                if phase in ("carry", "release", "settle"):
                    xy = data.body("bowl").xpos[:2].copy()
                    orientation_weight = 0
                elif phase == "lift":
                    xy = start_xy * 0.8
                    orientation_weight = 0.015
                else:
                    xy = start_xy
                    orientation_weight = 0.08
                q = ik(np.r_[xy, z], opening, q)
                goal = arm.to_policy(q)
                contact_run = 0
                for tick in range(limit):
                    records["states"].append(arm.read_state())
                    records["qpos"].append(data.qpos.copy())
                    if renderer is not None:
                        cams = images(renderer, data)
                        for key in pixels:
                            pixels[key].append(cams[key.split(".")[1]])
                    issued = safety.send("sim", goal.tolist())
                    records["actions"].append(issued)
                    bilateral = False
                    for _ in range(10):
                        mujoco.mj_step(model, data)
                        _, both = contact_flags(model, data)
                        bilateral |= both
                    result = outcome(model, data)
                    records["block_xyz"].append(result["block_xyz"])
                    records["phase"].append(phase)
                    records["bilateral"].append(bilateral)
                    records["inside_bowl"].append(result["block_in_bowl"])
                    stable = (
                        stable + 1
                        if phase in ("release", "settle") and result["block_in_bowl"] and result["slow"]
                        else 0
                    )
                    contact_run = contact_run + 1 if bilateral else 0
                    error = np.abs(np.array(arm.read_state()) - goal)
                    near = bool(np.max(error[:5]) < 1.5 and error[5] < 2)
                    if tick >= 19:
                        if phase == "close" and contact_run >= 5:
                            break
                        if phase == "settle" and stable >= 15 and near:
                            break
                        if phase not in ("close", "settle") and near:
                            break
        max_lift = max(v[2] for v in records["block_xyz"]) - initial_height
        success = bool(stable >= 15 and max_lift > 0.04 and any(records["bilateral"]))
        report = {
            "seed": seed,
            "success": success,
            "frames": len(records["states"]),
            "max_lift_m": max_lift,
            "stable_inside_ticks": stable,
            "block_start_xy": start_xy.tolist(),
            "bowl_xy": model.body_pos[model.body("bowl").id, :2].tolist(),
            "final": outcome(model, data),
            "source": "scripted IK teacher; not FLUX output",
            "state_frame": "legacy policy degrees via dataset90, gripper open percent",
            "randomization": randomization,
            "fps": 30,
            "image_shape": [256, 256, 3] if render else None,
        }
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if success:
                np.savez_compressed(
                    output / "episode.npz",
                    **{k: np.asarray(v) for k, v in records.items()},
                    **{k: np.asarray(v, dtype=np.uint8) for k, v in pixels.items()},
                )
        print(json.dumps(report), flush=True)
        return report
    finally:
        if renderer is not None:
            renderer.close()


async def main(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    reports = []
    seed = args.start_seed
    while sum(r["success"] for r in reports) < args.episodes and seed < args.start_seed + args.episodes * 3:
        folder = root / f"episode-{seed:04d}"
        if (folder / "report.json").exists():
            reports.append(json.loads((folder / "report.json").read_text()))
            seed += 1
            continue
        reports.append(await episode(seed, folder, args.render))
        seed += 1
        (root / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    (root / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    if sum(r["success"] for r in reports) < args.episodes:
        raise RuntimeError("Insufficient verified demonstrations")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    asyncio.run(main(p.parse_args()))
