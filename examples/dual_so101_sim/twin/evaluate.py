"""Closed-loop evaluation of FLUX (base or base + LoRA) in the twin, on held-out random layouts.

Same inputs as training: D455 `top` and the active arm's wrist camera rendered at 320 wide and resized to
256x256, joints converted to checkpoint coordinates (shoulder +180, elbow +90). Standard LeRobot loop:
saved processors + select_action (42 predicted, first 32 executed). Simulation time pauses during
inference (this measures policy quality, not latency). Success uses the expert's check.

usage (train-venv on the Pod): MUJOCO_GL=egl python -m twin.evaluate --label base --episodes 10
                               ... --adapter /opt/.../checkpoints/001200/pretrained_model
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .expert import CONTROL_HZ, TASKS, Arm, check_success, step_physics  # noqa: E402
from .export import to_checkpoint, write_mp4  # noqa: E402
from .generate import random_layout  # noqa: E402
from .scene import build  # noqa: E402

BASE = "/opt/flux3-action/hf-cache/hub/models--black-forest-labs--flux-3-action-so101/snapshots/c9e13b2aca6a0a472b3ea03fd90cff32d3e85849"
ENCODERS = "/opt/flux3-action/hf-cache/hub/models--black-forest-labs--flux-3-action-base/snapshots/62878e2925e59b7a89ec14463ce89932624c490d"
MAX_STEP_DEG = 6.0


def load_policy(adapter: str | None):
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.flux3 import Flux3Policy

    policy = Flux3Policy.from_pretrained(BASE)
    if adapter:
        from peft import PeftModel
        policy = PeftModel.from_pretrained(policy, adapter).merge_and_unload()
    policy = policy.to("cuda").eval()
    source = adapter or BASE
    pre, post = make_pre_post_processors(policy.config, pretrained_path=source)
    return torch, policy, pre, post


def run(torch, policy, pre, post, task, seed, seconds, media=None):
    rng = np.random.default_rng(seed)
    layout = random_layout(task, rng)
    model, data = build(layout)
    for _ in range(100):
        mujoco.mj_step(model, data)
    arm_name = TASKS[task][0]
    arm = Arm(model, data, arm_name)
    model.actuator_forcerange[model.actuator(f"{arm_name}_gripper").id] = [-0.5, 0.5]
    lo, hi = np.degrees(model.jnt_range[arm.ik.ids]).T
    ren = {"top": mujoco.Renderer(model, 240, 320), "wrist": mujoco.Renderer(model, 180, 320)}
    front = mujoco.Renderer(model, 360, 480) if media else None
    substeps = round(1 / CONTROL_HZ / model.opt.timestep)
    ctrl, frames, plans, t0 = data.ctrl.copy(), [], 0, time.perf_counter()
    policy.reset(); pre.reset(); post.reset()

    def image(r, cam):
        r.update_scene(data, camera=cam)
        a = np.asarray(Image.fromarray(r.render()).resize((256, 256), Image.Resampling.BILINEAR))
        return torch.from_numpy(a.copy()).permute(2, 0, 1).float().div(255.0)[None].to("cuda")

    prompt = TASKS[task][2]
    for tick in range(int(seconds * CONTROL_HZ)):
        state = arm.state()
        batch = {"observation.state": torch.tensor([to_checkpoint(state)], dtype=torch.float32),
                 "observation.images.scene": image(ren["top"], "top"),
                 "observation.images.wrist": image(ren["wrist"], f"{arm_name}_wrist"), "task": prompt}
        plans += len(policy._action_queue) == 0
        with torch.inference_mode():
            action = post(policy.select_action(pre(batch))).reshape(-1).float().cpu().numpy()
        action[1] -= 180.0
        action[2] -= 90.0
        joints = np.clip(np.clip(action[:5], np.array(state[:5]) - MAX_STEP_DEG, np.array(state[:5]) + MAX_STEP_DEG), lo, hi)
        arm.command(np.radians(joints), float(np.clip(action[5], 0, 100)))
        ctrl = step_physics(model, data, ctrl, substeps)
        if front is not None and tick % 2 == 0:
            front.update_scene(data, camera="front")
            frames.append(front.render().copy())
    result = check_success(model, data, task)
    obj = TASKS[task][1]
    if obj:
        tool = data.site(f"{arm_name}_gripperframe").xpos
        result["final_tool_to_object_cm"] = round(100 * float(np.linalg.norm(tool - data.body(obj).xpos)), 1)
        result["object_lift_cm_end"] = round(100 * float(data.body(obj).xpos[2] - 0.021), 1)
    for r in ren.values():
        r.close()
    if front is not None:
        front.close()
        write_mp4(Path(media) / f"{task}-{seed}.mp4", frames, fps=15)
    return {"task": task, "seed": seed, "plans": plans, "wall_s": round(time.perf_counter() - t0, 1), **result}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--tasks", nargs="+", default=["place_green_marker", "place_red_marker"])
    ap.add_argument("--episodes", type=int, default=10, help="per task")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--out", type=Path, default=Path("/opt/flux3-action/haller/evaluation"))
    ap.add_argument("--seed0", type=int, default=900000, help="held out: generation seeds are < 300000")
    args = ap.parse_args()
    out = args.out / args.label
    out.mkdir(parents=True, exist_ok=False)
    torch, policy, pre, post = load_policy(args.adapter)
    results = []
    for k, task in enumerate(args.tasks):
        for i in range(args.episodes):
            r = run(torch, policy, pre, post, task, args.seed0 + 1000 * k + i, args.seconds, media=out if i < 2 else None)
            results.append(r)
            print(json.dumps(r), flush=True)
    summary = {"label": args.label, "adapter": args.adapter, "results": results,
               "success_rate": {t: sum(r["success"] for r in results if r["task"] == t) / args.episodes for t in args.tasks}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("EVAL_DONE", json.dumps(summary["success_rate"]), flush=True)


if __name__ == "__main__":
    main()
