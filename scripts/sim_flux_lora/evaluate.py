"""Compare real base/raw/EMA policies in held-out simulation episodes."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
from common import TASK, contact_flags, images, make_episode, outcome


async def rollout(
    policy, pre, post, seed, root, ticks, frame="dataset90", normalization="checkpoint", checkpoint=None, fps=30
):
    import imageio.v2 as imageio
    import mujoco
    import torch

    folder = root / f"seed-{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    model, data, arm, safety, randomization = make_episode(seed, frame=frame)
    renderer = mujoco.Renderer(model, height=480, width=640)
    initial_z = float(data.body("block").xpos[2])
    initial_time = float(data.time)
    policy.reset()
    pre.reset()
    post.reset()
    history = [step for step in pre.steps if step.__class__.__name__ == "ObservationHistoryNormalizerProcessorStep"]
    assert len(history) == 1
    records = {name: [] for name in ["state", "raw", "issued", "block_xyz", "bilateral", "limited"]}
    stable = 0
    max_stable = 0
    touched = False
    max_lift = 0.0
    start = time.perf_counter()
    report = {
        "seed": seed,
        "randomization": randomization,
        "checkpoint": str(checkpoint) if checkpoint is not None else str(root),
        "task": TASK,
        "fps": fps,
        "physics_control_hz": 1 / (10 * model.opt.timestep),
        "timing_consistent": bool(np.isclose(fps, 1 / (10 * model.opt.timestep))),
        "requested_ticks": ticks,
        "execution_horizon": policy.config.n_action_steps,
        "joint_frame": frame,
        "normalization": normalization,
        "protocol": "Official select_action and both saved processors every tick; acknowledge issued commands; "
        "keep the official queued action horizon; reset all three between episodes.",
        "success_rule": "bilateral contact, lift > 3.5 cm, release and stable placement for 15 ticks",
    }
    try:
        with imageio.get_writer(
            folder / "rollout.mp4", fps=fps, codec="libx264", ffmpeg_params=["-threads", "2"]
        ) as video:
            async with safety.claim("sim", "official-lora-evaluation"):
                for tick in range(ticks):
                    frames = images(renderer, data)
                    state = np.asarray(arm.read_state(), np.float32)
                    obs = {"observation.state": torch.from_numpy(state.copy())[None].to("cuda"), "task": [TASK]}
                    for role, frame in frames.items():
                        obs["observation.images." + role] = (
                            torch.from_numpy(frame.copy()).permute(2, 0, 1)[None].to("cuda").float() / 255
                        )
                    with torch.inference_mode():
                        normalized = policy.select_action(pre(obs))
                        raw = post(normalized).float().cpu().numpy()[0]
                    assert raw.shape == (6,) and np.isfinite(raw).all()
                    issued = safety.send("sim", raw.tolist())
                    limited = bool(np.max(np.abs(raw - issued)) > 1e-4)
                    # The postprocessor shares this history with the preprocessor. Record
                    # the issued command rather than an unrealized target as the next delta anchor.
                    history[0].last_command = torch.as_tensor(issued, dtype=torch.float32, device="cuda")[None]
                    bilateral = False
                    for _ in range(10):
                        mujoco.mj_step(model, data)
                        contact, both = contact_flags(model, data)
                        touched |= contact
                        bilateral |= both
                    result = outcome(model, data)
                    max_lift = max(max_lift, result["block_xyz"][2] - initial_z)
                    released = issued[5] > 40
                    stable = stable + 1 if result["block_in_bowl"] and result["slow"] and released else 0
                    max_stable = max(max_stable, stable)
                    for key, value in [
                        ("state", state),
                        ("raw", raw),
                        ("issued", issued),
                        ("block_xyz", result["block_xyz"]),
                        ("bilateral", bilateral),
                        ("limited", limited),
                    ]:
                        records[key].append(value)
                    video.append_data(np.concatenate([frames["scene"], frames["wrist"]], axis=1))
                    if tick % 90 == 0:
                        print(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "tick": tick,
                                    "max_lift_m": max_lift,
                                    "inside": result["block_in_bowl"],
                                    "limited": limited,
                                }
                            ),
                            flush=True,
                        )
                    if max_stable >= 15 and max_lift > 0.035 and any(records["bilateral"]):
                        break
        report.update(
            status="completed",
            completed_ticks=len(records["raw"]),
            simulated_seconds=float(data.time - initial_time),
            wall_seconds=time.perf_counter() - start,
            touched_block=touched,
            bilateral_contact=bool(any(records["bilateral"])),
            max_lift_m=max_lift,
            stable_placement_ticks=max_stable,
            safety_limited_ticks=int(sum(records["limited"])),
            final=outcome(model, data),
            manipulation_success=bool(max_stable >= 15 and max_lift > 0.035 and any(records["bilateral"])),
        )
    except Exception as exc:
        report.update(status="error", error=repr(exc))
        raise
    finally:
        renderer.close()
        (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(folder / "trajectory.npz", **records)
    print(json.dumps(report), flush=True)
    return report


async def main(args):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.40)
    torch.manual_seed(42)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)
    (root / "run-config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = Path(args.checkpoint)
    cfg.device = "cuda"
    cfg.inference_seed = 42
    meta = LeRobotDatasetMetadata(args.repo_id, root=Path(args.dataset))
    policy = make_policy(cfg, ds_meta=meta).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
    if args.normalization_json:
        norms = json.loads(Path(args.normalization_json).read_text())
        policy.config.normalization_stats = norms
        updated = []
        for step in pre.steps + post.steps:
            if step.__class__.__name__ in (
                "ObservationHistoryNormalizerProcessorStep",
                "ActionTargetNormalizerProcessorStep",
                "ActionHistoryUnnormalizerProcessorStep",
            ):
                values = {
                    f"{stream}.{key}": torch.tensor(norms[stream][key], dtype=torch.float32)
                    for stream in step.streams
                    for key in ("q01", "q99")
                }
                step.load_state_dict(values)
                step.validate_statistics()
                updated.append(step.__class__.__name__)
        assert len(updated) == 3
        (root / "normalization-override.json").write_text(
            json.dumps(
                {
                    "statistics": norms,
                    "updated_processors": updated,
                    "source": args.normalization_json,
                    "weights_changed": False,
                },
                indent=2,
            )
            + "\n"
        )
    assert (policy.config.n_obs_steps, policy.config.chunk_size, policy.config.n_action_steps) == (8, 42, 32)
    assert policy.config.camera_keys == ["observation.images.scene", "observation.images.wrist"]
    reports = []
    for seed in range(args.start_seed, args.start_seed + args.episodes):
        folder = root / f"seed-{seed}"
        report_file = folder / "report.json"
        if args.resume and report_file.exists():
            saved = json.loads(report_file.read_text())
            if saved.get("status") == "completed":
                reports.append(saved)
                continue
        if args.resume and folder.exists():
            folder.rename(root / f"seed-{seed}-interrupted-{time.time_ns()}")
        reports.append(
            await rollout(
                policy,
                pre,
                post,
                seed,
                root,
                args.ticks,
                frame=args.frame,
                normalization="HF simulation override" if args.normalization_json else "checkpoint",
                checkpoint=args.checkpoint,
                fps=policy.config.fps,
            )
        )
        (root / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "episodes": len(reports),
                "successes": sum(r["manipulation_success"] for r in reports),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-id", default="local/liminal_so101_sim_pick_place")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--start-seed", type=int, default=20000)
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--frame", choices=["dataset90", "native"], default="dataset90")
    parser.add_argument("--normalization-json")
    parser.add_argument("--resume", action="store_true")
    asyncio.run(main(parser.parse_args()))
