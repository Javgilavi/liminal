"""Option C via LeRobot: a 12-D bimanual ``flux3`` base for LoRA training with ``lerobot-train``.

Two sources, both supported by LeRobot source code:

* ``--source so101`` (recommended): start from the SO-101 package. ``Flux3Policy.from_pretrained(..., config=cfg,
  strict=False)`` keeps the SO-101 finetuned trunk and action branch and re-initializes only the embodiment heads
  whose shapes change (``emb_in.action`` 6->12, ``emb_in.action_cond`` 12->24, ``final_layer.*.linear``), with
  BFL's reference init: xavier inputs, zero output projection
  (lerobot@146d847:src/lerobot/policies/flux3/modeling_flux3.py:259-317, f3/wiring.py:159-181).
* ``--source base``: start from the action-pretrained trunk ``flux-3-action-base.safetensors`` exactly like
  lerobot@146d847:examples/flux3/export_base.py (shared ``action_prediction`` branch remapped to our modality,
  fresh heads; f3/wiring.py:131-156).

Either way the package gets new processors with 12-D q01/q99 from our dataset (command deltas, both grippers
absolute: ``delta_absolute_dims=[5, 11]``), three cameras on a ``grid`` canvas, and gripper channel weights 2.
``lerobot-train --policy.path=<output>`` with the top-level ``peft`` block then trains LoRA on the trunk and the
fresh heads in full (``modules_to_save``, modeling_flux3.py:243-249). BFL has not validated this path for a new
embodiment ("existing checks do not establish ... PEFT quality on a new embodiment", lerobot flux3.mdx:265-269).

Usage (GPU box, or CPU with enough RAM, about 2x 14 GB):
    python -m flux.make_bimanual_base --dataset ORG/xlerobot_bimanual --output outputs/xlerobot_bimanual_base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import ARM_DIM, IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST

SO101_REPO = "black-forest-labs/flux-3-action-so101"
BASE_TRUNK = "black-forest-labs/flux-3-action-base:flux-3-action-base.safetensors"
CAMERA_KEYS = [IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST]
# 2x2 grid of 192x256 cells: 4:3 cells for 640x480 cameras, 12x16 latent tokens per frame (1.5x the SO-101's 8x16).
GRID_CANVAS = (384, 512)
ABSOLUTE_DIMS = [ARM_DIM - 1, 2 * ARM_DIM - 1]
CHANNEL_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0] * 2
# History contract of the SO-101 package, kept for --source base so both bases are comparable
# (flux-action@830f1ce:tests/fixtures/so101_history/config.json).
SO101_HISTORY = {
    "conditioning": "history",
    "n_obs_steps": 8,
    "history_snapshots": 2,
    "condition_on_past_actions": True,
    "action_representation": "delta",
    "chunk_size": 42,
    "n_action_steps": 32,
    "fps": 30.0,
    "text_fixed_length": 320,
    "video_position_fps": 24.0,
    "sampler": "euler",
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "guidance_scale_action": None,
    "sampler_shift": 6.93,
    "inference_seed": 42,
}


def bimanual_overrides(features_in: dict, features_out: dict, stats: dict, canvas_hw=GRID_CANVAS) -> dict[str, Any]:
    action_dim = int(features_out["action"].shape[0])
    if action_dim != 2 * ARM_DIM:
        raise ValueError(f"expected a 12-D action feature, got {action_dim}")
    missing = [k for k in CAMERA_KEYS if k not in features_in]
    if missing:
        raise ValueError(f"dataset lacks cameras {missing}; record with these keys or rename them")
    return {
        "input_features": features_in,
        "output_features": features_out,
        "camera_layout": "grid",
        "camera_keys": list(CAMERA_KEYS),
        "canvas_hw": tuple(canvas_hw),
        "delta_absolute_dims": list(ABSOLUTE_DIMS),
        "action_channel_weights": list(CHANNEL_WEIGHTS),
        "normalization_stats": stats,
        "use_peft": False,
        "pretrained_path": None,
        "pretrained_revision": None,
    }


def build_bimanual_policy(
    source: str,
    features_in: dict,
    features_out: dict,
    stats: dict,
    *,
    source_kind: str = "so101",
    device: str = "cpu",
    canvas_hw=GRID_CANVAS,
    config_overrides: dict[str, Any] | None = None,
):
    """Return ``(policy, preprocessor, postprocessor)`` of a fresh 12-D base (not saved)."""
    import dataclasses

    import lerobot.policies.flux3  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.flux3 import Flux3Config, Flux3Policy, make_flux3_pre_post_processors

    overrides = bimanual_overrides(features_in, features_out, stats, canvas_hw)
    overrides["device"] = device
    overrides.update(config_overrides or {})
    if source_kind == "so101":
        base_cfg = PreTrainedConfig.from_pretrained(source)
        cfg = dataclasses.replace(base_cfg, **overrides)
        policy = Flux3Policy.from_pretrained(source, config=cfg, strict=False)
    elif source_kind == "base":
        cfg = Flux3Config(**{**SO101_HISTORY, **overrides, "trunk_weights": source})
        policy = Flux3Policy(cfg)
    else:
        raise ValueError("source_kind must be so101 or base")
    policy.config.pretrained_path = None
    pre, post = make_flux3_pre_post_processors(policy.config)
    return policy, pre, post


def save_base(policy, pre, post, output: str | Path, extra: dict | None = None) -> Path:
    """Save like export_base.py:128-134: the processor state is the only statistics source."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    stats = policy.config.normalization_stats
    policy.config.normalization_stats = None
    policy.save_pretrained(output)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    (output / "normalization_source.json").write_text(json.dumps({"stats": stats, **(extra or {})}, indent=2) + "\n")
    return output


def lora_train_config(dataset_repo: str, base_dir: str | Path, steps: int = 10000, batch_size: int = 2) -> dict:
    """``examples/flux3/lora.json`` (lerobot@146d847) pointed at our base and dataset, plus a head-first schedule.

    Fresh heads start at zero output, so the trunk adapters wait for them: BFL's new-head recipe freezes the trunk
    and warms it up later, scaled for 3000 updates to 200 / 600 / 200 (flux-action@830f1ce:docs/embodiments.md:63-64).
    LeRobot counts these phases in microsteps (lerobot flux3.mdx:115-119), hence x4 for accumulation 4.
    """
    return {
        "dataset": {"repo_id": dataset_repo, "eval_split": 0.1},
        "policy": {
            "path": str(Path(base_dir).resolve()),
            "device": "cuda",
            "push_to_hub": False,
            "caption_dropout": 0.1,
            "augment": False,
            "gradient_checkpointing": True,
            "optimizer_lr": 0.0001,
            "optimizer_lr_heads_multiplier": 5,
            "optimizer_betas": [0.9, 0.999],
            "optimizer_weight_decay": 0,
            "optimizer_eps": 1e-8,
            "optimizer_grad_clip_norm": 1,
            "scheduler_freeze_backbone_steps": 800,
            "scheduler_warmup_steps": 2400,
            "scheduler_warmup_steps_heads": 800,
            "scheduler_decay_steps": 0,
            "scheduler_cooldown_steps": 0,
        },
        "peft": {"method_type": "LORA", "r": 32, "lora_alpha": 32},
        "ema": {"enable": True, "decay": 0.999},
        "accelerator": {"mixed_precision": "bf16", "gradient_accumulation": {"steps": 4}},
        "batch_size": batch_size,
        "steps": steps,
        "save_freq": 500,
        "seed": 42,
        "num_workers": 4,
        "prefetch_factor": 2,
        "persistent_workers": True,
        "dataloader_multiprocessing_context": "spawn",
        "log_freq": 20,
        "env_eval_freq": 0,
        "eval_steps": 500,
        "output_dir": "outputs/xlerobot_bimanual_lora",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, help="LeRobot v3 repo id of 12-D XLeRobot demos at 30 Hz")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-kind", choices=("so101", "base"), default="so101")
    parser.add_argument("--source", default=None, help=f"default: {SO101_REPO} or {BASE_TRUNK}")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--canvas", type=int, nargs=2, default=list(GRID_CANVAS), metavar=("H", "W"))
    args = parser.parse_args(argv)

    from lerobot.configs.types import FeatureType
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.utils.feature_utils import dataset_to_policy_features

    from .package_tools import compute_quantiles, dataset_arrays, describe_stats

    meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    if meta.fps != 30:
        raise ValueError(f"dataset fps {meta.fps}; the SO-101 history contract runs at 30 Hz")
    features = dataset_to_policy_features(meta.features)
    features_in = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    features_out = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    arrays = dataset_arrays(args.dataset, args.dataset_root)
    stats = compute_quantiles(
        arrays["action"], arrays["observation.state"], arrays["episode_index"], absolute_dims=tuple(ABSOLUTE_DIMS)
    )
    print(describe_stats(stats))
    source = args.source or (SO101_REPO if args.source_kind == "so101" else BASE_TRUNK)
    policy, pre, post = build_bimanual_policy(
        source,
        features_in,
        features_out,
        stats,
        source_kind=args.source_kind,
        device=args.device,
        canvas_hw=args.canvas,
    )
    out = save_base(policy, pre, post, args.output, {"source": source, "dataset": args.dataset})
    train = lora_train_config(args.dataset, out)
    (out / "lora_train.json").write_text(json.dumps(train, indent=2) + "\n")
    print(f"saved {out}; train with: lerobot-train --config_path={out / 'lora_train.json'}")


if __name__ == "__main__":
    main()
