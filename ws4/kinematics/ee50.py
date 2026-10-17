"""EE50 codec: two-arm joint trajectories <-> FLUX 3 Action's 50-D end-effector action chunks.

What is verified (https://bfl.ai/models/flux-3-action, read 2026-10-17):
  * 50 dims = two 25-D blocks, first = left/primary, second = right/secondary end effector.
  * per block: 3 translation, then 6 rotation "as in Zhou et al. 2019 but using the first two rows instead
    of the first two columns of the rotation matrix", then 16 dims of hand angles or gripper state
    (0 closed, 1 open). Unused dims are zero padded.
  * poses are relative to the anchor state = the action state closest to the last conditioning frame;
    hand and gripper values stay absolute.
  * every dim is z-scored per embodiment, "indexed by the signed position offset from anchor state".

What is NOT in the released code (black-forest-labs/flux-action@830f1ce has no EE50 encoder at all, and
the base checkpoint that carries the EE50 heads is gated), so these are parameters here, all UNVERIFIED:
  * ``rel_frame``: "anchor" expresses each pose in the anchor end-effector frame (T_a^-1 T_t);
    "base" keeps the robot base axes (p_t - p_a, R_t R_a^T). Default "anchor".
  * translation units: metres.
  * gripper slot: index 9 of each block (the first of the 16 auxiliary dims).
  * the 6D layout is [R00 R01 R02 R10 R11 R12] (row-major first two rows).

For our 5-DoF arms the convention only has to be self-consistent: per-embodiment z-scores are refit on our
own data either way, and decoding gives back absolute targets for the IK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .se3 import inv_pose, make_pose

DIM = 50
BLOCK = 25
TRANS = slice(0, 3)
ROT = slice(3, 9)
AUX = slice(9, 25)
ARMS = ("left", "right")


@dataclass(frozen=True)
class EE50Layout:
    rel_frame: str = "anchor"  # UNVERIFIED, see module docstring
    gripper_index: int = 9  # UNVERIFIED, within a 25-D block

    def __post_init__(self):
        if self.rel_frame not in ("anchor", "base"):
            raise ValueError("rel_frame must be 'anchor' or 'base'")
        if not AUX.start <= self.gripper_index < AUX.stop:
            raise ValueError("gripper_index must be one of the 16 auxiliary dims (9..24)")

    def block(self, arm: str) -> int:
        return ARMS.index(arm) * BLOCK

    def active_mask(self, arms=ARMS) -> np.ndarray:
        """Dims that carry data for an embodiment with these arms and a 1-DoF gripper each."""
        m = np.zeros(DIM, dtype=bool)
        for a in arms:
            b = self.block(a)
            m[b : b + 9] = True
            m[b + self.gripper_index] = True
        return m


DEFAULT_LAYOUT = EE50Layout()


def rot_to_6d(R: np.ndarray) -> np.ndarray:
    """First two ROWS of R, row-major: (..., 3, 3) -> (..., 6)."""
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def rot_from_6d(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`rot_to_6d` via Gram-Schmidt on the rows (Zhou et al., transposed)."""
    x = np.asarray(x, dtype=float)
    a1, a2 = x[..., :3], x[..., 3:6]
    r1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - np.sum(r1 * a2, axis=-1, keepdims=True) * r1
    r2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    r3 = np.cross(r1, r2)
    return np.stack([r1, r2, r3], axis=-2)


def relative_pose(T_anchor: np.ndarray, T: np.ndarray, rel_frame: str) -> tuple[np.ndarray, np.ndarray]:
    """(translation, rotation) of ``T`` relative to ``T_anchor``. Broadcasts over leading axes."""
    if rel_frame == "anchor":
        Tr = inv_pose(T_anchor) @ T
        return Tr[..., :3, 3], Tr[..., :3, :3]
    Ra = T_anchor[..., :3, :3]
    return T[..., :3, 3] - T_anchor[..., :3, 3], T[..., :3, :3] @ np.swapaxes(Ra, -1, -2)


def absolute_pose(T_anchor: np.ndarray, p_rel: np.ndarray, R_rel: np.ndarray, rel_frame: str) -> np.ndarray:
    if rel_frame == "anchor":
        return T_anchor @ make_pose(R_rel, p_rel)
    return make_pose(R_rel @ T_anchor[..., :3, :3], p_rel + T_anchor[..., :3, 3])


def encode(
    poses: dict[str, np.ndarray],
    grippers: dict[str, np.ndarray],
    anchor: dict[str, np.ndarray],
    layout: EE50Layout | None = None,
) -> np.ndarray:
    """Raw (unnormalized) EE50 vectors.

    ``poses[arm]``: (T, 4, 4) absolute EE poses in a fixed robot frame (base or station), ``grippers[arm]``:
    (T,) open fraction, ``anchor[arm]``: (4, 4) anchor pose. Arms missing from ``poses`` stay zero.
    """
    layout = layout or DEFAULT_LAYOUT
    T = len(next(iter(poses.values())))
    out = np.zeros((T, DIM))
    for arm, Ts in poses.items():
        b = layout.block(arm)
        p, R = relative_pose(anchor[arm], Ts, layout.rel_frame)
        out[:, b + TRANS.start : b + TRANS.stop] = p
        out[:, b + ROT.start : b + ROT.stop] = rot_to_6d(R)
        out[:, b + layout.gripper_index] = grippers[arm]
    return out


def decode(
    x: np.ndarray, anchor: dict[str, np.ndarray], layout: EE50Layout | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Raw EE50 vectors (T, 50) -> absolute poses (T, 4, 4) and grippers (T,) per arm in ``anchor``."""
    layout = layout or DEFAULT_LAYOUT
    poses, grippers = {}, {}
    for arm, Ta in anchor.items():
        b = layout.block(arm)
        p = x[:, b + TRANS.start : b + TRANS.stop]
        R = rot_from_6d(x[:, b + ROT.start : b + ROT.stop])
        poses[arm] = absolute_pose(Ta, p, R, layout.rel_frame)
        grippers[arm] = np.clip(x[:, b + layout.gripper_index], 0.0, 1.0)
    return poses, grippers


def windows(
    poses: dict[str, np.ndarray],
    grippers: dict[str, np.ndarray],
    offsets: np.ndarray,
    layout: EE50Layout | None = None,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Every anchor ``t`` of an episode -> EE50 at ``t + offsets``: (N, len(offsets), 50), anchors (N,).

    ``offsets`` are the signed frame offsets from the anchor (negative = history, positive = future
    actions); the anchor itself is offset 0 and encodes to the identity pose.
    """
    layout = layout or DEFAULT_LAYOUT
    offsets = np.asarray(offsets, dtype=int)
    T = len(next(iter(poses.values())))
    lo, hi = -offsets.min(), T - offsets.max()
    anchors = np.arange(max(lo, 0), hi, stride)
    out = np.zeros((len(anchors), len(offsets), DIM))
    for i, t in enumerate(anchors):
        idx = t + offsets
        out[i] = encode(
            {a: P[idx] for a, P in poses.items()},
            {a: g[idx] for a, g in grippers.items()},
            {a: P[t] for a, P in poses.items()},
            layout,
        )
    return out, anchors


@dataclass
class EE50Stats:
    """Per-embodiment z-score statistics indexed by signed offset from the anchor: mean/std (K, 50).

    Dims whose std is below ``min_std`` (zero padding, the identity anchor row) keep std 1 so they pass
    through unchanged instead of exploding. How BFL floors the std is UNVERIFIED.
    """

    offsets: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    min_std: float = 1e-6
    meta: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, chunks: np.ndarray, offsets: np.ndarray, min_std: float = 1e-6, **meta) -> EE50Stats:
        mean = chunks.mean(axis=0)
        std = chunks.std(axis=0)
        std = np.where(std < min_std, 1.0, std)
        return cls(np.asarray(offsets, dtype=int), mean, std, min_std, meta)

    def normalize(self, chunks: np.ndarray) -> np.ndarray:
        return (chunks - self.mean) / self.std

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean

    def save(self, path: str | Path) -> None:
        np.savez(path, offsets=self.offsets, mean=self.mean, std=self.std, min_std=self.min_std)

    @classmethod
    def load(cls, path: str | Path) -> EE50Stats:
        d = np.load(path)
        return cls(d["offsets"], d["mean"], d["std"], float(d["min_std"]))


def joints_to_poses(station, q: dict[str, np.ndarray], frame: str = "station") -> dict[str, np.ndarray]:
    """Joint trajectories (T, 5) per arm -> TCP poses (T, 4, 4) in the station (or each arm's base) frame."""
    if frame == "station":
        return {a: station.fk(a, qa) for a, qa in q.items()}
    return {a: station.arm.fk(qa) for a, qa in q.items()}
