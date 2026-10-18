"""Generate randomized scripted demonstrations in the twin (CPU only, parallel).

Each episode: random layout -> scripted expert -> keep only successes. Saves per episode an .npz with
`rows` (tick, measured state[6], commanded action[6]; MJCF degrees and gripper %) and `qobs` (full
qpos per tick, for rendering later), plus a .json with the layout, task and result.

usage: python -m twin.generate --out DIR --tasks place_green_marker place_red_marker --episodes 100 --workers 32
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from . import config as C
from .expert import TASKS, run_episode
from .scene import Layout, build

MAT_X = (C.MAT["center"][0] - C.MAT["size"][0] / 2 + 0.035, C.MAT["center"][0] + C.MAT["size"][0] / 2 - 0.035)
MAT_Y = (C.MAT["center"][1] - C.MAT["size"][1] / 2 + 0.03, C.MAT["center"][1] + C.MAT["size"][1] / 2 - 0.03)


def random_layout(task: str, rng) -> Layout:
    objects, placed = {}, []
    for name in ("green_marker", "red_marker", "cutter"):
        for _ in range(200):
            xy = (rng.uniform(*MAT_X), rng.uniform(*MAT_Y))
            if all(math.dist(xy, other) > 0.065 for other in placed):
                break
        placed.append(xy)
        objects[name] = (xy[0], xy[1], float(rng.uniform(-math.pi, math.pi)))
    bx, by = C.BOX["center"]
    return Layout(
        box=(bx + rng.uniform(-0.02, 0.02), by + rng.uniform(-0.02, 0.02), math.radians(rng.uniform(-8, 8))),
        lid_deg=0.0 if task == "open_box" else float(rng.uniform(100, 115)),
        objects=objects,
        light_scale=float(rng.uniform(0.75, 1.25)),
        camera_jitter=(rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01),
                       math.radians(rng.uniform(-2, 2))),
    )


def one(args):
    task, seed, out = args
    rng = np.random.default_rng(seed)
    for attempt in range(8):  # resample unreachable layouts
        layout = random_layout(task, rng)
        model, data = build(layout)
        for _ in range(100):
            mujoco.mj_step(model, data)
        rows, qobs = [], []

        def record(tick, arm, state, action):
            rows.append([tick, *state, *action])
            qobs.append(data.qpos.copy())

        t0 = time.perf_counter()
        result = run_episode(model, data, task, rng, record=record)
        if result.get("feasible", True):
            break
    result.update(task=task, seed=seed, attempts=attempt + 1, seconds=round(time.perf_counter() - t0, 2),
                  prompt=TASKS[task][2], arm=TASKS[task][0])
    name = f"{task}-{seed:06d}"
    if result["success"]:
        np.savez_compressed(Path(out) / f"{name}.npz", rows=np.asarray(rows, np.float32), qobs=np.asarray(qobs, np.float64))
    lay = asdict(layout)
    (Path(out) / f"{name}.json").write_text(json.dumps({"layout": lay, "result": result}, default=float))
    return {k: result[k] for k in ("task", "seed", "success", "ticks", "attempts", "miss") if k in result}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=["place_green_marker", "place_red_marker"])
    ap.add_argument("--episodes", type=int, default=50, help="successful episodes wanted per task (attempts are capped)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, task in enumerate(args.tasks):
            ok, tried, seed = 0, 0, args.seed0 + 100000 * k
            while ok < args.episodes and tried < 3 * args.episodes:
                n = min(args.workers * 2, 3 * args.episodes - tried)
                for r in pool.imap_unordered(one, [(task, seed + i, str(args.out)) for i in range(n)]):
                    ok += r["success"]
                    tried += 1
                seed += n
                print(json.dumps({"task": task, "successes": ok, "attempted": tried}), flush=True)
            summary[task] = {"successes": ok, "attempted": tried}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("GENERATION_DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
