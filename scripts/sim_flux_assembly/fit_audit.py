"""Audit real decoded training windows and evaluate those exact inference histories."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def main(args):
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
    from torch.utils.data import default_collate

    torch.set_num_threads(8)
    frozen = json.loads(args.windows.read_text())
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path, cfg.device = Path(args.checkpoint), "cpu" if args.audit_only else "cuda"
    if cfg.fps != frozen["fps"] or cfg.n_obs_steps != frozen["history"] or cfg.chunk_size != frozen["chunk"]:
        raise ValueError("FIT_CONTRACT: checkpoint timing/history/chunk differs from frozen windows")
    if cfg.action_representation != "delta" or cfg.delta_absolute_dims != [-1]:
        raise ValueError("FIT_CONTRACT: require arm command deltas with absolute gripper")
    if cfg.camera_order != ["observation.images.scene", "observation.images.wrist"]:
        raise ValueError("FIT_CONTRACT: camera role/order mismatch")
    meta = LeRobotDatasetMetadata("local/liminal_nist_atomic_assembly", root=args.root / "dataset-v3")
    if meta.fps != cfg.fps:
        raise ValueError("FIT_TIMING: dataset and checkpoint fps disagree")
    dataset = LeRobotDataset(meta.repo_id, root=args.root / "dataset-v3",
                            delta_timestamps=resolve_delta_timestamps(cfg, meta), video_backend="pyav",
                            return_uint8=True)
    overrides = {"device_processor": {"device": cfg.device}}

    def processors():
        return make_pre_post_processors(cfg, pretrained_path=args.checkpoint, preprocessor_overrides=overrides)

    pre_train, _ = processors()
    pre_infer, post = processors()
    pre_sync, _ = processors()
    stats = json.loads((args.root / "hf-sim-normalization.json").read_text())
    for pipeline in [pre_train, pre_infer, post, pre_sync]:
        for step in pipeline.steps:
            for key, value in step.state_dict().items():
                if key.count(".") == 1 and key.split(".")[0] in stats:
                    stream, quantile = key.split(".")
                    if not np.allclose(value.cpu().numpy(), stats[stream][quantile], atol=1e-5, rtol=1e-6):
                        raise ValueError(f"FIT_NORMALIZATION: loaded processor mismatch at {key}")
    policy = None
    if not args.audit_only:
        torch.cuda.set_per_process_memory_fraction(0.40)
        torch.manual_seed(42)
        policy = make_policy(cfg, ds_meta=meta).eval()
    records, predicted_all, target_all = [], [], []
    source_path, source = None, None
    for window in frozen["windows"]:
        if source_path != window["source"]:
            if source is not None:
                source.close()
            source_path = window["source"]
            source = np.load(source_path)
            states, commands = source["states"], source["actions"]
        row = dataset[window["dataset_index"]]
        if any(bool(v.any()) for k, v in row.items() if k.endswith("_is_pad")):
            raise ValueError(f"FIT_PADDING: incomplete frozen window {window['dataset_index']}")
        anchor, history, chunk = window["source_anchor"], frozen["history"], frozen["chunk"]
        batch = default_collate([row])
        raw_actions = batch["action"].float()
        state_error = float(np.abs(batch["observation.state"].numpy()[0]
                                   - states[anchor - history + 1:anchor + 1]).max())
        action_error = float(np.abs(raw_actions.numpy()[0] - commands[anchor - history:anchor + chunk]).max())
        if max(state_error, action_error) > 1e-4:
            raise ValueError(
                f"FIT_ALIGNMENT: source/dataset timing mismatch: state={state_error}, action={action_error}"
            )
        pre_train.reset()
        training = _preprocess_dataset_batch(copy.deepcopy(batch), meta.camera_keys, {}, pre_train)
        offline = {"observation.state": batch["observation.state"],
                   "observation.command_history": raw_actions[:, :history], "task": batch["task"]}
        for key in meta.camera_keys:
            offline[key] = batch[key][:, :history].float() / 255
        pre_infer.reset()
        prepared = pre_infer(copy.deepcopy(offline))
        differences = {}
        for key in ["observation.state", "observation.past_actions", *meta.camera_keys]:
            target = training[key][:, :history] if key in meta.camera_keys else training[key]
            differences[key] = float((target - prepared[key]).abs().max())
        decoded = post(training["action"]).float().cpu()
        differences["command_roundtrip"] = float((decoded - raw_actions[:, history:]).abs().max())
        pre_sync.reset()
        sync_history = next(s for s in pre_sync.steps
                            if s.__class__.__name__ == "ObservationHistoryNormalizerProcessorStep")
        for tick in range(history):
            sync_history.last_command = raw_actions[:, tick].to(cfg.device)
            obs = {"observation.state": offline["observation.state"][:, tick], "task": batch["task"]}
            obs.update({k: offline[k][:, tick] for k in meta.camera_keys})
            synchronous = pre_sync(obs)
        for key in ["observation.state", "observation.past_actions", *meta.camera_keys]:
            differences[f"synchronous.{key}"] = float((synchronous[key] - prepared[key]).abs().max())
        if max(differences.values()) > 1e-4:
            raise ValueError(f"FIT_PROCESSORS: training/offline/synchronous contract mismatch: {differences}")
        record = {"split": window["split"], "seed": window["seed"], "dataset_index": window["dataset_index"],
                  "source_anchor": anchor, "source_state_error": state_error,
                  "source_command_error": action_error, "contract_errors": differences}
        if policy is not None:
            policy.reset()
            pre_infer.reset()
            post.reset()
            torch.manual_seed(42)
            with torch.inference_mode():
                predicted = post(policy.predict_action_chunk(pre_infer(copy.deepcopy(offline)))).float().cpu()
            error = predicted - raw_actions[:, history:]
            if not torch.isfinite(error).all():
                raise ValueError("FIT_PREDICTION: nonfinite predicted commands")
            record["joint_rmse"] = {str(h): float(error[0, h - 1, :5].square().mean().sqrt())
                                    for h in [1, 8, 32, chunk]}
            record["gripper_mae"] = {str(h): float(error[0, h - 1, -1].abs()) for h in [1, 8, 32, chunk]}
            record["per_joint_error_h32"] = error[0, 31].tolist()
            predicted_all.append(predicted.numpy()[0])
            target_all.append(raw_actions[:, history:].numpy()[0])
        records.append(record)
    if source is not None:
        source.close()
    groups = []
    if policy is not None:
        for split in ["train", "validation"]:
            subset = [r for r in records if r["split"] == split]
            groups.append({"split": split, "windows": len(subset), **{
                metric: {str(h): float(np.mean([r[metric][str(h)] for r in subset]))
                         for h in [1, 8, 32, frozen["chunk"]]}
                for metric in ["joint_rmse", "gripper_mae"]}})
        np.savez_compressed(args.output.with_suffix(".npz"), predicted=np.stack(predicted_all),
                            target=np.stack(target_all))
    report = {"checkpoint": args.checkpoint, "contract_passed": True, "groups": groups, "windows": records,
              "limitations": "Recorded-history prediction and processor agreement do not establish task success."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"contract_passed": True, "windows": len(records), "groups": groups}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    main(parser.parse_args())
