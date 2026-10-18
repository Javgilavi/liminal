"""Measure command errors on identical training/held-out histories before collecting more data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scene import PROMPTS


def main(args):
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.40)
    torch.manual_seed(42)
    args.output.mkdir(parents=True, exist_ok=False)
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = Path(args.checkpoint)
    config.device = "cuda"
    config.guidance_scale = args.guidance
    assert config.conditioning == "history" and config.action_representation == "delta"
    metadata = LeRobotDatasetMetadata("local/liminal_nist_atomic_assembly", root=args.root / "dataset-v3")
    policy = make_policy(config, ds_meta=metadata).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
    stats = json.loads((args.root / "hf-sim-normalization.json").read_text())["action"]
    span = np.asarray(stats["q99"]) - np.asarray(stats["q01"])
    manifest = json.loads((args.root / "dataset-manifest.json").read_text())
    rows, predictions, references = [], [], []
    for split in ["train", "validation"]:
        sources = sorted({item["source"] for item in manifest["episodes"] if item["split"] == split})
        for source in sources[: args.assemblies_per_split]:
            segments = [item for item in manifest["episodes"] if item["source"] == source]
            with np.load(source) as data:
                states, actions, phases = data["states"], data["actions"], data["phase"]
                cameras = {role: data[f"images.{role}"] for role in ["scene", "wrist"]}
                for segment in segments:
                    if args.skills and segment["skill"] not in args.skills:
                        continue
                    low = segment["start"] + config.n_obs_steps
                    high = segment["end"] - config.chunk_size - 1
                    assert high >= low, "Episode lacks a complete official training window"
                    anchors = sorted(set(np.linspace(low, high, args.anchors_per_skill).astype(int)))
                    for anchor in anchors:
                        target = actions[anchor : anchor + config.chunk_size].astype(np.float32)
                        previous = actions[anchor - 1].astype(np.float32)
                        for steps in args.denoise_steps:
                            policy.config.num_inference_steps = steps
                            policy.reset()
                            pre.reset()
                            post.reset()
                            observation = {
                                "observation.state": torch.from_numpy(
                                    states[anchor - config.n_obs_steps + 1 : anchor + 1].astype(np.float32)
                                )[None].to("cuda"),
                                "observation.command_history": torch.from_numpy(
                                    actions[anchor - config.n_obs_steps : anchor].astype(np.float32)
                                )[None].to("cuda"),
                                "task": [PROMPTS[segment["skill"]]],
                            }
                            for role, pixels in cameras.items():
                                image = pixels[anchor - config.n_obs_steps + 1 : anchor + 1].copy()
                                observation[f"observation.images.{role}"] = (
                                    torch.from_numpy(image).permute(0, 3, 1, 2)[None].to("cuda").float() / 255
                                )
                            with torch.inference_mode():
                                predicted = post(policy.predict_action_chunk(pre(observation))).float().cpu().numpy()[0]
                            assert predicted.shape == target.shape and np.isfinite(predicted).all()
                            delta_p = np.diff(np.vstack([previous, predicted]), axis=0)
                            delta_t = np.diff(np.vstack([previous, target]), axis=0)
                            delta_p[:, config.delta_absolute_dims] = predicted[:, config.delta_absolute_dims]
                            delta_t[:, config.delta_absolute_dims] = target[:, config.delta_absolute_dims]
                            error = predicted - target
                            horizons = {}
                            for horizon in [1, 8, 32, config.chunk_size]:
                                sample = error[horizon - 1]
                                horizons[str(horizon)] = {
                                    "joint_rmse_deg": float(np.sqrt(np.mean(sample[:5] ** 2))),
                                    "gripper_absolute_error_pp": float(abs(sample[-1])),
                                    "absolute_error_per_joint": np.abs(sample).tolist(),
                                }
                            record = {
                                "split": split,
                                "seed": segment["seed"],
                                "skill": segment["skill"],
                                "phase": str(phases[anchor]),
                                "anchor": int(anchor),
                                "denoise_steps": steps,
                                "normalized_delta_rmse": float(np.sqrt(np.mean((2 * (delta_p - delta_t) / span) ** 2))),
                                "horizons": horizons,
                            }
                            rows.append(record)
                            predictions.append(predicted)
                            references.append(target)
                            (args.output / "windows.json").write_text(json.dumps(rows, indent=2) + "\n")
                            print(json.dumps(record), flush=True)
    groups = []
    for split in ["train", "validation"]:
        for steps in args.denoise_steps:
            subset = [r for r in rows if r["split"] == split and r["denoise_steps"] == steps]
            groups.append(
                {
                    "split": split,
                    "denoise_steps": steps,
                    "windows": len(subset),
                    "mean_normalized_delta_rmse": float(np.mean([r["normalized_delta_rmse"] for r in subset])),
                    "mean_joint_rmse_deg_by_horizon": {
                        str(h): float(np.mean([r["horizons"][str(h)]["joint_rmse_deg"] for r in subset]))
                        for h in [1, 8, 32, config.chunk_size]
                    },
                }
            )
    report = {
        "checkpoint": args.checkpoint,
        "input": "Genuine recorded image/state/command history; no future observations or commands are inputs",
        "purpose": "Separate fitting error, train/validation gap, chunk drift and denoising-step sensitivity",
        "guidance": config.guidance_scale,
        "groups": groups,
        "windows": rows,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "predictions.npz", predicted=np.stack(predictions), target=np.stack(references))
    print(json.dumps({"summary": groups}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assemblies-per-split", type=int, default=2)
    parser.add_argument("--anchors-per-skill", type=int, default=4)
    parser.add_argument("--denoise-steps", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--skills", nargs="+", choices=list(PROMPTS))
    main(parser.parse_args())
