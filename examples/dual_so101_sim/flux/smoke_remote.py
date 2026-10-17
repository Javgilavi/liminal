"""Exercise real HTTP inference on one MuJoCo arm and save an inspectable trace.

MUJOCO_GL=egl ../.venv/bin/python smoke_remote.py --policy http://127.0.0.1:18100
This checks inference and simulation integration; it does not certify task success.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
from app import MAX_STEP_DEG, SCENE_WH, WRIST_WH, RemotePolicy, Sim, jpeg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="http://127.0.0.1:18100")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--scene-cam", choices=("front", "top"), default="front")
    parser.add_argument("--task", default="Pick up the green cube and place it on the blue square.")
    parser.add_argument("--ticks", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path(".runpod/smoke"))
    parser.add_argument("--allow-fake", action="store_true", help="protocol debugging only")
    args = parser.parse_args()
    if not 1 <= args.ticks <= 300:
        parser.error("ticks must be between 1 and 300")
    health_response = requests.get(args.policy.rstrip("/") + "/health", timeout=10)
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("fake", True) and not args.allow_fake:
        raise SystemExit("Refusing to label fake/unknown-backend responses as real FLUX inference.")
    args.output.mkdir(parents=True, exist_ok=True)
    policy, sim = RemotePolicy(args.policy), Sim()
    other = "left" if args.arm == "right" else "right"
    initial = {arm: sim.get_arm(arm) for arm in ("left", "right")}
    chunks = []
    started = time.monotonic()
    try:
        (args.output / "before.jpg").write_bytes(jpeg(sim.render("front", SCENE_WH)))
        policy.reset(args.task)
        with (args.output / "actions.jsonl").open("w") as trace:
            for tick in range(args.ticks):
                state = sim.get_arm(args.arm)
                scene = jpeg(sim.render(args.scene_cam, SCENE_WH))
                wrist = jpeg(sim.render(f"{args.arm}_wrist", WRIST_WH))
                action, infer_ms = policy.act(args.task, state, scene, wrist)
                safe = [s + float(np.clip(a - s, -MAX_STEP_DEG, MAX_STEP_DEG))
                        for s, a in zip(state[:5], action[:5], strict=True)]
                safe.append(float(np.clip(action[5], 0, 100)))
                sim.set_arm(args.arm, safe)
                sim.step()
                trace.write(json.dumps({"tick": tick, "arm": args.arm, "state": state,
                                        "predicted": action, "applied": safe, "infer_ms": infer_ms}) + "\n")
                trace.flush()
                if infer_ms is not None:
                    chunks.append(infer_ms)
                    print(f"tick {tick}: FLUX chunk inference {infer_ms:.1f} ms", flush=True)
        (args.output / "after.jpg").write_bytes(jpeg(sim.render("front", SCENE_WH)))
        report = {
            "status": "real_inference_smoke_passed" if not health.get("fake") else "fake_protocol_only",
            "task_success": "not evaluated", "health": health, "task": args.task,
            "arm": args.arm, "ticks": args.ticks, "scene_camera": args.scene_cam,
            "elapsed_s": round(time.monotonic() - started, 3), "chunk_inference_ms": chunks,
            "initial_arm_states": initial,
            "final_arm_states": {arm: sim.get_arm(arm) for arm in ("left", "right")},
            "control_targets_sent": {args.arm: args.ticks, other: 0},
            "artifacts": str(args.output),
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for renderer in sim.renderers.values():
            renderer.close()
        policy.http.close()


if __name__ == "__main__":
    main()
