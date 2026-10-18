"""Run learned atomic skills with predicate-gated sequencing and no teacher fallback."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harness import SKILLS, assembly_task  # noqa: E402
from scene import FPS, PROMPTS, SUCCESS, AssemblyScene  # noqa: E402


class LearnedBackend:
    def __init__(self, policy, pre, post, scene, video, budget, warm_history=None):
        self.policy, self.pre, self.post = policy, pre, post
        self.scene, self.video, self.budget = scene, video, budget
        self.warm_history = warm_history or []
        self.records = {key: [] for key in ["state", "raw", "issued", "skill", "gear_xyz", "collar_xyz"]}
        self.skills = []
        self.history = next(
            step for step in pre.steps if step.__class__.__name__ == "ObservationHistoryNormalizerProcessorStep"
        )

    async def execute(self, skill, ctx):
        import numpy as np
        import torch

        from liminal.core.types import SkillResult, Status

        if ctx.safety.owner_of("sim") is None:
            raise RuntimeError("The behavior tree must own the arm")
        scene = self.scene
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        # Diagnostic only: replay genuine held-out teacher observations into
        # the processor, then let the model act from that recorded state.
        for item in self.warm_history:
            self.history.last_command = torch.as_tensor(item["command"], device="cuda")[None].float()
            observation = {
                "observation.state": torch.as_tensor(item["state"], device="cuda")[None].float(),
                "task": [skill.prompt],
            }
            for role, pixels in item["frames"].items():
                observation[f"observation.images.{role}"] = (
                    torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].to("cuda").float() / 255
                )
            self.pre(observation)
            self.history.last_command = torch.as_tensor(item["issued"], device="cuda")[None].float()
        begin = len(self.records["state"])
        t0 = time.monotonic()
        for tick in range(self.budget):
            frames = scene.frames()
            state = np.asarray(scene.arm.read_state(), np.float32)
            observation = {"observation.state": torch.from_numpy(state)[None].to("cuda"), "task": [skill.prompt]}
            for role, pixels in frames.items():
                observation[f"observation.images.{role}"] = (
                    torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].to("cuda").float() / 255
                )
            with torch.inference_mode():
                raw = self.post(self.policy.select_action(self.pre(observation))).float().cpu().numpy()[0]
            assert raw.shape == (6,) and np.isfinite(raw).all()
            issued = scene.step(raw)
            self.history.last_command = torch.as_tensor(issued, dtype=torch.float32, device="cuda")[None]
            self.video.append_data(np.concatenate([frames["scene"], frames["wrist"]], axis=1))
            for key, value in [
                ("state", state),
                ("raw", raw),
                ("issued", issued),
                ("skill", skill.name),
                ("gear_xyz", scene.data.body("gear").xpos.copy()),
                ("collar_xyz", scene.data.body("collar").xpos.copy()),
            ]:
                self.records[key].append(value)
            ctx.world.update(scene.facts, source="mujoco_contact_pose_detector")
            if tick % 100 == 0:
                print(
                    json.dumps(
                        {
                            "seed": scene.seed,
                            "skill": skill.name,
                            "tick": tick,
                            "facts": scene.facts,
                            "state": scene.state,
                        }
                    ),
                    flush=True,
                )
            if scene.facts[skill.success]:
                status = Status.SUCCESS
                break
            await asyncio.sleep(0)
        else:
            status = Status.TIMEOUT
        result = {
            "skill": skill.name,
            "status": status.value,
            "start": begin,
            "end": len(self.records["state"]),
            "facts": scene.facts.copy(),
            "state": scene.state.copy(),
        }
        self.skills.append(result)
        print(json.dumps({"seed": scene.seed, "skill_end": result}), flush=True)
        if not status.ok:
            scene.arm.stop()
        return SkillResult(status, skill, "physics predicate checked", time.monotonic() - t0)


def prepare_atomic(scene, name, source, history_steps=0):
    """Replay held-out teacher start state only for isolated-skill diagnostics."""
    import mujoco
    import numpy as np

    report = json.loads((source / "report.json").read_text())
    segment = next(item for item in report["segments"] if item["skill"] == name)
    with np.load(source / "episode.npz") as arrays:
        start = segment["start"]
        index = start + history_steps
        assert index < segment["end"]
        scene.data.qpos[:] = arrays["qpos"][index]
        scene.data.qvel[:] = 0
        if index:
            scene.data.ctrl[scene.arm.aids] = scene.arm.to_sim(arrays["actions"][index - 1])
        else:
            scene.data.ctrl[scene.arm.aids] = scene.data.qpos[scene.arm.qadr]
        warm_history = []
        if history_steps:
            states, commands = arrays["states"], arrays["actions"]
            frames = {
                role: arrays[f"images.{role}"][start:index].copy() for role in ["scene", "wrist"]
            }
            for offset in range(history_steps):
                tick = start + offset
                warm_history.append(
                    {
                        "state": states[tick].copy(),
                        "command": commands[tick - 1].copy() if offset else states[tick].copy(),
                        "issued": commands[tick].copy(),
                        "frames": {role: pixels[offset] for role, pixels in frames.items()},
                    }
                )
    mujoco.mj_forward(scene.model, scene.data)
    scene.counters = dict.fromkeys(scene.counters, 0)
    scene.update_predicates()
    return warm_history


async def rollout(args, policy, pre, post, seed, source=None, atomic=None):
    import imageio.v2 as imageio
    import numpy as np

    from liminal.core.types import Skill, Status
    from liminal.orchestration.bt import SkillNode
    from liminal.orchestration.context import ExecutionContext
    from liminal.orchestration.executor import run_task
    from liminal.orchestration.skills import SkillRouter
    from liminal.perception.world_state import WorldState

    folder = args.output / (f"seed-{seed}" if atomic is None else f"atomic-{atomic}-seed-{seed}")
    folder.mkdir(parents=True, exist_ok=False)
    scene = AssemblyScene(args.root, seed, render=True)
    warm_history = []
    if atomic:
        warm_history = prepare_atomic(scene, atomic, source, policy.config.n_obs_steps if args.teacher_history else 0)
    try:
        with imageio.get_writer(
            folder / "rollout.mp4", fps=FPS, codec="libx264", ffmpeg_params=["-threads", "2"]
        ) as video:
            backend = LearnedBackend(policy, pre, post, scene, video, args.ticks_per_skill, warm_history)
            world = WorldState(log_path=folder / "events.jsonl")
            world.update(scene.facts, source="initial_detector")
            ctx = ExecutionContext({"sim": scene.arm}, world, scene.safety, SkillRouter(backend))
            if atomic:
                status = await SkillNode(
                    Skill("sim", atomic, PROMPTS[atomic], success=SUCCESS[atomic], timeout_s=180, verify_timeout_s=0.01)
                ).run(ctx)
                verified = status is Status.SUCCESS and scene.facts[SUCCESS[atomic]]
                task_report = {"status": status.value, "goal_verified": bool(verified)}
            else:
                task_report = asdict(await run_task(assembly_task(attempts=args.attempts), ctx))
            report = {
                "seed": seed,
                "checkpoint": args.checkpoint,
                "atomic_skill": atomic,
                "teacher_start_state_only": bool(atomic),
                "teacher_history_frames": len(warm_history),
                "learned_actions_only": True,
                "contract": {
                    "coordinates": "native",
                    "normalization": "saved HF statistics",
                    "fps": FPS,
                    "execution_steps": policy.config.n_action_steps,
                    "prediction_chunk_size": policy.config.chunk_size,
                    "guidance_scale": policy.config.guidance_scale,
                },
                "metadata": scene.metadata,
                "task_report": task_report,
                "skill_results": backend.skills,
                "final_facts": scene.facts,
                "final_state": scene.state,
                "assembly_success": bool(task_report["goal_verified"] and scene.facts["assembly_complete"]),
            }
            (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            np.savez_compressed(folder / "trajectory.npz", **backend.records)
            return report
    finally:
        scene.close()


async def main(args):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.40)
    torch.manual_seed(42)
    if args.teacher_history and not args.atomic:
        raise ValueError("Teacher-history replay is allowed only in isolated diagnostics")
    if args.skills and not args.atomic:
        raise ValueError("Skill selection is allowed only in isolated diagnostics")
    args.output.mkdir(parents=True, exist_ok=False)
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = Path(args.checkpoint)
    config.device = "cuda"
    assert config.fps == FPS and config.n_action_steps == 32
    if not 1 <= args.execution_steps <= config.chunk_size:
        raise ValueError("execution_steps must fit within the model's prediction chunk")
    config.n_action_steps = args.execution_steps
    config.guidance_scale = args.guidance
    metadata = LeRobotDatasetMetadata("local/liminal_nist_atomic_assembly", root=args.root / "dataset-v3")
    policy = make_policy(config, ds_meta=metadata).eval()
    assert policy.config.n_action_steps == args.execution_steps
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
    reports = []
    if args.atomic:
        splits = ["train", "validation"] if args.atomic_split == "both" else [args.atomic_split]
        sources = []
        for split in splits:
            accepted = [
                p.parent
                for p in sorted((args.root / "raw" / split).glob("episode-*/report.json"))
                if json.loads(p.read_text())["success"]
            ]
            sources.extend(accepted[: args.episodes])
        for source in sources:
            seed = json.loads((source / "report.json").read_text())["metadata"]["seed"]
            for name, _, _ in SKILLS:
                if args.skills and name not in args.skills:
                    continue
                reports.append(await rollout(args, policy, pre, post, seed, source, name))
                (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    else:
        for seed in range(args.start_seed, args.start_seed + args.episodes):
            reports.append(await rollout(args, policy, pre, post, seed))
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps({"episodes": len(reports), "assembly_success": sum(row["assembly_success"] for row in reports)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=20000)
    parser.add_argument("--ticks-per-skill", type=int, default=450)
    parser.add_argument("--execution-steps", type=int, default=32)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--atomic", action="store_true")
    parser.add_argument("--teacher-history", action="store_true")
    parser.add_argument("--atomic-split", choices=["train", "validation", "both"], default="validation")
    parser.add_argument("--skills", nargs="+", choices=list(PROMPTS))
    asyncio.run(main(parser.parse_args()))
