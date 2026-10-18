"""Compare fixed held-out windows through the official policy and saved processors."""

from __future__ import annotations

import argparse
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
    from torch.utils.data import DataLoader, Subset

    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.40)
    torch.manual_seed(42)
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = Path(args.checkpoint)
    cfg.device = "cuda"
    meta = LeRobotDatasetMetadata(args.repo_id, root=Path(args.dataset))
    policy = make_policy(cfg, ds_meta=meta).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
    dataset = LeRobotDataset(
        args.repo_id,
        root=Path(args.dataset),
        episodes=list(range(64, 80)),
        delta_timestamps=resolve_delta_timestamps(policy.config, meta),
        video_backend="pyav",
        return_uint8=True,
    )
    # Sample complete windows without image/state/action padding. Indices are fixed
    # independently of the adapter, so base/raw/EMA see exactly the same recordings.
    candidates = np.linspace(50, len(dataset) - 100, args.windows * 3).astype(int)
    selected = []
    for index in candidates:
        row = dataset[int(index)]
        if any(bool(value.any()) for key, value in row.items() if key.endswith("_is_pad")):
            continue
        selected.append(int(index))
    if len(selected) < args.windows:
        raise ValueError("Not enough complete held-out windows")
    selected = [selected[i] for i in np.linspace(0, len(selected) - 1, args.windows).astype(int)]
    loader = DataLoader(Subset(dataset, selected), batch_size=2, shuffle=False, num_workers=0)
    values = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, batch in enumerate(loader):
            torch.manual_seed(30000 + index)
            pre.reset()
            prepared = _preprocess_dataset_batch(batch, meta.camera_keys, {}, pre)
            loss, details = policy(prepared)
            result = {"loss": float(loss)}
            for key, value in details.items():
                if isinstance(value, (int, float)) or (torch.is_tensor(value) and value.numel() == 1):
                    result[key] = float(value)
            values.append(result)
    report = {
        "checkpoint": args.checkpoint,
        "windows": len(selected),
        "indices": selected,
        "episodes": list(range(64, 80)),
        "mean": {key: float(np.mean([row[key] for row in values])) for key in values[0]},
        "batches": values,
        "caveat": "Fixed-window stochastic denoising loss; not task success.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "batches"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repo-id", default="local/liminal_so101_sim_pick_place")
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", type=int, default=32)
    main(parser.parse_args())
