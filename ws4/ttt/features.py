"""Per-re-plan features and targets for the progress memory.

Two feature sources, one sequence format:

* ``frame``: :class:`FrameFeaturizer`, cheap and CPU-only. Coarse colour grids of the head and wrist images, the
  change of the head image since the oldest FLUX snapshot (motion), and the 12-D joint state. No weights, no
  download, works on the laptop at the robot's rate.
* ``flux``: pooled hidden states of FLUX 3 Action itself, one vector per re-plan, written to
  ``features_flux.npz`` in each episode directory by ``extract_flux_features.py`` on a GPU (``FluxFeatureTap``).

:func:`episode_sequence` returns, for one episode, the feature matrix ``(R, D)`` and the per-re-plan targets
from ``labels.replan_targets``: skill id, occurrence (how many times this skill already ran: the aliased part),
stage index, progress inside the stage, overall progress and the "stage done" flag.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .labels import EpisodeLabels, replan_targets

CAMERAS = ("head", "left_wrist", "right_wrist")
MAX_OCCURRENCE = 6  # occurrence classes 0..5, the last one means "5 or more"


def grid_mean(img: np.ndarray, gh: int, gw: int) -> np.ndarray:
    """Mean colour of a ``gh x gw`` grid of cells, in [0, 1]: ``(gh, gw, 3)``."""
    h, w = img.shape[:2]
    if h % gh == 0 and w % gw == 0:
        return img.reshape(gh, h // gh, gw, w // gw, 3).mean(axis=(1, 3)) / 255.0
    from PIL import Image

    return np.asarray(Image.fromarray(img).resize((gw, gh), Image.BOX), dtype=np.float64) / 255.0


@dataclass(frozen=True)
class FrameFeaturizer:
    """Head and wrist colour grids + head motion since the oldest snapshot + joint state."""

    head_grid: tuple[int, int] = (12, 16)
    wrist_grid: tuple[int, int] = (6, 8)
    state_scale: float = 100.0  # LeRobot degrees and gripper 0..100 -> about unit range
    motion: bool = True

    @property
    def dim(self) -> int:
        h = 3 * self.head_grid[0] * self.head_grid[1]
        w = 3 * self.wrist_grid[0] * self.wrist_grid[1]
        return h * (2 if self.motion else 1) + 2 * w + 12

    def __call__(
        self, frames: dict[str, np.ndarray], state12: np.ndarray, prev_head: np.ndarray | None = None
    ) -> np.ndarray:
        head = grid_mean(frames["head"], *self.head_grid)
        parts = [head.ravel()]
        if self.motion:
            prev = head if prev_head is None else grid_mean(prev_head, *self.head_grid)
            parts.append((head - prev).ravel() * 4.0)
        for cam in ("left_wrist", "right_wrist"):
            img = frames.get(cam)
            parts.append(
                np.zeros(3 * self.wrist_grid[0] * self.wrist_grid[1])
                if img is None
                else grid_mean(img, *self.wrist_grid).ravel()
            )
        parts.append(np.asarray(state12, dtype=np.float64).reshape(12) / self.state_scale)
        return np.concatenate(parts).astype(np.float32)

    def config(self) -> dict:
        return asdict(self)


def frame_features(ep_dir: Path, featurizer: FrameFeaturizer, cache: bool = True) -> np.ndarray:
    """``(R, D)`` frame features at the episode's re-plan ticks (cached as ``features_frame.npz``)."""
    from .sim_episodes import decode_jpeg, load_frames

    path = ep_dir / "features_frame.npz"
    if cache and path.exists():
        z = np.load(path)
        if z["config"].item() == repr(featurizer.config()):
            return z["feat"]
    labels = EpisodeLabels.load(ep_dir / "episode.json")
    state = np.load(ep_dir / "ticks.npz")["state"]
    frame_ticks, frames = load_frames(ep_dir)
    where = {int(t): i for i, t in enumerate(frame_ticks)}
    feats = []
    for rp in labels.extra["replans"]:
        t = rp["tick"]
        oldest = rp.get("snapshot_ticks", [t, t])[0]
        now = {cam: decode_jpeg(frames[cam][where[t]]) for cam in CAMERAS if cam in frames}
        prev = decode_jpeg(frames["head"][where[oldest]]) if oldest in where else None
        feats.append(featurizer(now, state[min(t, len(state) - 1)], prev))
    feat = np.stack(feats)
    if cache:
        np.savez(path, feat=feat, config=np.array(repr(featurizer.config())))
    return feat


def flux_features(ep_dir: Path, parts: list[str] | None = None) -> np.ndarray:
    """``(R, D)`` FLUX tap features written by ``extract_flux_features.py`` (one row per re-plan).

    ``parts`` keeps only some taps, as ``"<layer>:<group>"`` (e.g. ``["27:imagined"]``); taps are stored in the
    order ``for layer in layers for group in groups``, 3072 columns each.
    """
    z = np.load(ep_dir / "features_flux.npz")
    labels = EpisodeLabels.load(ep_dir / "episode.json")
    if list(z["replan_ticks"]) != list(labels.replan_ticks):
        raise ValueError(f"{ep_dir}: FLUX features do not match the episode's re-plan ticks")
    feat = z["feat"].astype(np.float32)
    if parts:
        names = [f"{int(layer)}:{g}" for layer in z["layers"] for g in z["groups"]]
        width = feat.shape[1] // len(names)
        feat = np.concatenate([feat[:, names.index(p) * width : (names.index(p) + 1) * width] for p in parts], 1)
    return feat


@dataclass
class Sequence:
    episode: str
    task: str
    feat: np.ndarray  # (R, D)
    skill: np.ndarray  # (R,) skill id, -1 between steps
    occurrence: np.ndarray  # (R,)
    stage: np.ndarray  # (R,) index into steps, -1 between steps
    stage_progress: np.ndarray  # (R,)
    progress: np.ndarray  # (R,)
    done: np.ndarray  # (R,)
    plan_skills: list[str]  # skill of each plan step


def episode_sequence(
    ep_dir: Path, skills: list[str], source: str = "frame", featurizer=None, flux_parts: list[str] | None = None
) -> Sequence:
    labels = EpisodeLabels.load(ep_dir / "episode.json")
    if source == "frame":
        feat = frame_features(ep_dir, featurizer or FrameFeaturizer())
    elif source == "flux":
        feat = flux_features(ep_dir, flux_parts)
    else:  # "flux+frame"
        feat = np.concatenate(
            [flux_features(ep_dir, flux_parts), frame_features(ep_dir, featurizer or FrameFeaturizer())], 1
        )
    tg = replan_targets(labels)
    plan_skills = [s["skill"] for s in labels.steps]
    skill = np.array([skills.index(plan_skills[k]) if k >= 0 else -1 for k in tg.stage], dtype=np.int64)
    occ = np.minimum(tg.occurrence, MAX_OCCURRENCE - 1)
    return Sequence(
        ep_dir.name, labels.task, feat, skill, occ, tg.stage, tg.stage_progress, tg.progress, tg.done, plan_skills
    )


def plan_occurrences(plan_skills: list[str]) -> list[int]:
    """Occurrence index of every plan step (0 for the first time a skill appears, 1 for the second, ...)."""
    seen: dict[str, int] = {}
    out = []
    for s in plan_skills:
        out.append(min(seen.get(s, 0), MAX_OCCURRENCE - 1))
        seen[s] = seen.get(s, 0) + 1
    return out


def stage_posterior(skill_probs: np.ndarray, occ_probs: np.ndarray, plan_skills: list[str], skills: list[str]):
    """``P(stage k) ∝ P(skill of k) * P(occurrence of k)`` for a known plan: ``(..., n_steps)``."""
    occ = plan_occurrences(plan_skills)
    idx = [skills.index(s) if s in skills else None for s in plan_skills]
    cols = [
        (skill_probs[..., i] if i is not None else np.zeros(skill_probs.shape[:-1])) * occ_probs[..., o]
        for i, o in zip(idx, occ, strict=True)
    ]
    p = np.stack(cols, -1)
    return p / np.clip(p.sum(-1, keepdims=True), 1e-12, None)
