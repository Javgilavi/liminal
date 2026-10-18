"""Dependency-free contracts for the bounded fixed-window fitting experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def freeze_windows(manifest, history=8, chunk=42, count=8):
    if count < 2 or history < 1 or chunk < 1:
        raise ValueError("FIT_WINDOWS: require count >= 2 and positive history/chunk")
    rows = []
    offset = 0
    selected = set()
    for episode in manifest["episodes"]:
        split = episode["split"]
        if split not in {"train", "validation"}:
            raise ValueError(f"FIT_SPLIT: unknown split {split!r}")
        if episode["skill"] == "pick_gear" and split not in selected:
            if split == "train" and episode["episode_index"] != 0:
                raise ValueError("FIT_EPISODE: training must use atomic episode 0")
            low, high = history, episode["frames"] - chunk - 1
            if high - low + 1 < count:
                raise ValueError("FIT_WINDOW_BOUNDS: insufficient unpadded anchors")
            anchors = [low + i * (high - low) // (count - 1) for i in range(count)]
            for anchor in anchors:
                rows.append({**episode, "local_anchor": anchor, "dataset_index": offset + anchor,
                             "source_anchor": episode["start"] + anchor})
            selected.add(split)
        offset += episode["frames"]
    if selected != {"train", "validation"}:
        raise ValueError("FIT_SPLIT: require separate training and validation pickups")
    if len({row["source"] for row in rows}) != 2 or len({row["seed"] for row in rows}) != 2:
        raise ValueError("FIT_LEAKAGE: training and validation must have disjoint sources and seeds")
    return {"history": history, "chunk": chunk, "fps": manifest["fps"], "windows": rows}


def fit_indices(frozen):
    rows = [r for r in frozen["windows"] if r["split"] == "train"]
    if not rows or any(r["episode_index"] != 0 or r["skill"] != "pick_gear" for r in rows):
        raise ValueError("FIT_LEAKAGE: only episode-0 pickup may be fitted")
    indices = [r["dataset_index"] for r in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("FIT_WINDOWS: duplicate training indices")
    return indices


def training_config(template, checkpoint, output, steps=1000):
    if not 1 <= steps <= 1000 or steps % 4:
        raise ValueError("FIT_BUDGET: microbatches must be a multiple of 4, at most 1000")
    cfg = json.loads(json.dumps(template))
    cfg["dataset"].update(episodes=[0], eval_split=0.0)
    cfg["policy"].update(path=str(checkpoint), caption_dropout=0.0, augment=False)
    cfg.update(steps=steps, save_freq=min(250, steps), output_dir=str(output),
               batch_size=2, num_workers=0, persistent_workers=False)
    cfg["accelerator"]["gradient_accumulation"]["steps"] = 4
    return cfg


def decide(before, after, rollout_reports):
    """Return an explicit scientific gate, independently of process completion."""
    train_before = next(r for r in before["groups"] if r["split"] == "train")
    train_after = next(r for r in after["groups"] if r["split"] == "train")
    a, b = train_before["joint_rmse"]["32"], train_after["joint_rmse"]["32"]
    grip = train_after["gripper_mae"]["32"]
    if not all(math.isfinite(v) and v >= 0 for v in (a, b, grip)):
        raise ValueError("FIT_METRICS: invalid prediction metrics")
    if len(rollout_reports) != 2 or {r["seed"] for r in rollout_reports} != {100, 10000}:
        raise ValueError("FIT_ROLLOUTS: require exactly the frozen train/validation pickup pair")
    fitted = b <= 2.0 and b <= 0.5 * a and grip <= 5.0
    pickup = {r["seed"]: r["task_report"]["goal_verified"] is True for r in rollout_reports}
    if not fitted:
        action = "inspect_optimization_and_serialization"
    elif not pickup[100]:
        action = "inspect_startup_history_and_closed_loop_drift"
    elif not pickup[10000]:
        action = "collect_targeted_recovery_development_data"
    else:
        action = "evaluate_new_scenes_before_expanding_skills"
    return {"window_fit_passed": fitted, "training_pickup": pickup[100],
            "validation_pickup": pickup[10000], "promoted": False, "next_action": action}
