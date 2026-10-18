"""Normalization statistics and LeRobot ``flux3`` package surgery.

A history package's normalization lives in its saved processor state, not in the model weights: q01/q99 of the
measured state and of the consecutive command deltas (absolute channels such as the gripper stay absolute)
(lerobot@146d847:src/lerobot/policies/flux3/processor_flux3.py:223-290). LeRobot's docs say the SO-101
package's quantiles "assume its joint order, units and calibration" and that different statistics need "a
separate base with matching processors" (lerobot@146d847:docs/source/flux3.mdx:180-184). BFL's own SO-101 full
fine-tune normalized every community dataset with its own q01/q99 (flux-action@830f1ce:
configs/so101/community_corpus.json, ``builder.normalization``), so re-normalizing per arm with our own
quantiles matches how the trunk saw SO-101 data. ``renormalize_package`` writes such a package: same weights
(symlinked), new processor state.

``compute_quantiles`` reproduces lerobot@146d847:examples/flux3/export_base.py:38-78.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .contract import ARM_DIM, arm_slice

HISTORY_STEPS = (
    "flux3_observation_history_normalizer",
    "flux3_action_target_normalizer",
    "flux3_action_history_unnormalizer",
)


def compute_quantiles(
    commands: np.ndarray,
    states: np.ndarray,
    episode_index: np.ndarray,
    absolute_dims: tuple[int, ...] = (-1,),
    representation: str = "delta",
) -> dict[str, dict[str, list[float]]]:
    """Per-channel q01/q99 of command deltas (first frame of an episode: zero delta) and of states."""
    commands = np.asarray(commands, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    episode_index = np.asarray(episode_index)
    if commands.shape != states.shape or commands.ndim != 2 or len(episode_index) != len(commands):
        raise ValueError("commands and states must be (N, D) and episode_index (N,)")
    values = commands.copy()
    if representation == "delta":
        values = np.zeros_like(commands)
        same = np.zeros(len(commands), dtype=bool)
        same[1:] = episode_index[1:] == episode_index[:-1]
        values[1:][same[1:]] = commands[1:][same[1:]] - commands[:-1][same[1:]]
        dims = list(absolute_dims)
        values[:, dims] = commands[:, dims]
    elif representation != "absolute":
        raise ValueError("representation must be delta or absolute")
    out = {}
    for key, array in (("action", values), ("state", states)):
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite {key} values")
        out[key] = {
            "q01": np.percentile(array, 1, axis=0).tolist(),
            "q99": np.percentile(array, 99, axis=0).tolist(),
        }
    return out


def arm_columns(x: np.ndarray, arm: str | None) -> np.ndarray:
    """All 12 columns (``arm=None``) or one arm's 6 columns of a ``[left | right]`` array."""
    x = np.asarray(x)
    return x if arm is None else x[:, arm_slice(arm)]


def dataset_arrays(repo_id: str, root: str | None = None, episodes: list[int] | None = None) -> dict[str, np.ndarray]:
    """``action``, ``observation.state`` and ``episode_index`` columns of a LeRobot dataset (no videos)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root, episodes=episodes, download_videos=False)
    table = dataset.hf_dataset.select_columns(["episode_index", "action", "observation.state"]).with_format("numpy")
    return {
        "action": np.stack(table["action"]),
        "observation.state": np.stack(table["observation.state"]),
        "episode_index": np.asarray(table["episode_index"]),
    }


def stats_to_state_dicts(stats: dict[str, dict[str, list[float]]]):
    """Tensors in the layout ``_HistoryQuantiles.load_state_dict`` expects (processor_flux3.py:259-270)."""
    import torch

    values = {
        f"{s}.{q}": torch.tensor(stats[s][q], dtype=torch.float32) for s in ("state", "action") for q in ("q01", "q99")
    }
    action_only = {k: v for k, v in values.items() if k.startswith("action.")}
    return values, action_only


def set_history_quantiles(preprocessor: Any, postprocessor: Any, stats: dict[str, dict[str, list[float]]]) -> None:
    from lerobot.policies.flux3.processor_flux3 import (
        ActionHistoryUnnormalizerProcessorStep,
        ActionTargetNormalizerProcessorStep,
        ObservationHistoryNormalizerProcessorStep,
    )

    values, action_only = stats_to_state_dicts(stats)
    found = 0
    for step in list(preprocessor.steps) + list(postprocessor.steps):
        if isinstance(step, ObservationHistoryNormalizerProcessorStep):
            step.load_state_dict(values)
            found += 1
        elif isinstance(step, ActionTargetNormalizerProcessorStep | ActionHistoryUnnormalizerProcessorStep):
            step.load_state_dict(action_only)
            found += 1
    if found != 3:
        raise ValueError(f"expected the three flux3 history steps, found {found}")


def read_history_quantiles(preprocessor: Any) -> dict[str, dict[str, list[float]]]:
    from lerobot.policies.flux3.processor_flux3 import ObservationHistoryNormalizerProcessorStep

    step = next(s for s in preprocessor.steps if isinstance(s, ObservationHistoryNormalizerProcessorStep))
    state = step.state_dict()
    return {s: {q: state[f"{s}.{q}"].tolist() for q in ("q01", "q99")} for s in ("state", "action")}


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copyfile(src, dst)


def renormalize_package(
    src: str | Path, dst: str | Path, stats: dict[str, dict[str, list[float]]], device: str | None = None
) -> Path:
    """Copy a flux3 package to ``dst`` with new history quantiles; weights are symlinked, not copied.

    ``src`` is a local package directory or a Hub repo id (resolved through the HF cache, where
    ws4/cloud/bootstrap.sh prefetches the SO-101 package). ``device`` is written into the saved device step
    (LeRobot refuses to load "cuda" steps without CUDA).
    """
    import lerobot.policies.flux3  # noqa: F401  (registers the flux3 config type)
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.flux3.processor_flux3 import make_flux3_pre_post_processors_from_pretrained

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(src).is_dir():
        from huggingface_hub import snapshot_download

        src = snapshot_download(str(src))
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"refusing to overwrite {dst}")
    config = PreTrainedConfig.from_pretrained(src)
    dim = config.action_dim
    for stream in ("state", "action"):
        for q in ("q01", "q99"):
            if len(stats[stream][q]) != dim:
                raise ValueError(f"{stream}.{q} needs {dim} values")
    pre, post = make_flux3_pre_post_processors_from_pretrained(
        config,
        str(src),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    set_history_quantiles(pre, post, stats)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name.startswith("policy_") or item.name == "dataset_statistics.json":
            continue  # rewritten below
        if item.is_file():
            # copyfile, not copy2: S3 FUSE volumes (RunPod geesefs) refuse the metadata copy
            (_link_or_copy if item.suffix == ".safetensors" else shutil.copyfile)(item, dst / item.name)
    pre.save_pretrained(dst)
    post.save_pretrained(dst)
    (dst / "normalization_source.json").write_text(
        json.dumps({"renormalized_from": str(src), "stats": stats}, indent=2)
    )
    return dst


def describe_stats(stats: dict[str, dict[str, list[float]]], names: list[str] | None = None) -> str:
    names = names or [f"d{i}" for i in range(len(stats["state"]["q01"]))]
    rows = ["channel            state q01..q99            action q01..q99"]
    for i, name in enumerate(names):
        s, a = stats["state"], stats["action"]
        rows.append(f"{name:<18} {s['q01'][i]:9.3f} .. {s['q99'][i]:9.3f}   {a['q01'][i]:9.3f} .. {a['q99'][i]:9.3f}")
    return "\n".join(rows)


__all__ = [
    "ARM_DIM",
    "arm_columns",
    "compute_quantiles",
    "dataset_arrays",
    "describe_stats",
    "read_history_quantiles",
    "renormalize_package",
    "set_history_quantiles",
]
