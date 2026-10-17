"""Observation / action contract between the XLeRobot station and FLUX 3 Action policies.

Robot side (shared with ws4/sim): ``observation.state`` is 12-D ``[left 6 | right 6]``, each arm in SO-101 joint
order (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper), plus three RGB cameras
``observation.images.head``, ``observation.images.left_wrist``, ``observation.images.right_wrist`` (HWC uint8).

Model side: a ``ModelContract`` describes what one policy package consumes. The values of ``SO101_CONTRACT`` come
from the released SO-101 package schema (flux-action@830f1ce:tests/fixtures/so101_history/config.json:4-84) and
the model card (https://huggingface.co/black-forest-labs/flux-3-action-so101): two cameras ``scene`` / ``wrist``
side by side, each 256x256, 6-D state and actions (degrees, gripper in percentage points), eight observation
ticks, two visual snapshots, past-command conditioning, 42 predicted / 32 executed actions at 30 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_DIM = len(ARM_JOINTS)
ARMS: tuple[str, ...] = ("left", "right")
GRIPPER_INDEX = ARM_DIM - 1

OBS_STATE = "observation.state"
IMG_HEAD = "observation.images.head"
IMG_LEFT_WRIST = "observation.images.left_wrist"
IMG_RIGHT_WRIST = "observation.images.right_wrist"
ROBOT_IMAGE_KEYS: tuple[str, ...] = (IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST)
WRIST_KEY = {"left": IMG_LEFT_WRIST, "right": IMG_RIGHT_WRIST}

# Keys of the released SO-101 package (flux-action@830f1ce:tests/fixtures/so101_history/config.json:76-79).
SO101_SCENE = "observation.images.scene"
SO101_WRIST = "observation.images.wrist"


def arm_slice(arm: str) -> slice:
    """Columns of ``arm`` inside the 12-D ``[left | right]`` vector."""
    i = ARMS.index(arm)
    return slice(i * ARM_DIM, (i + 1) * ARM_DIM)


def split_bimanual(x: np.ndarray) -> dict[str, np.ndarray]:
    """``(..., 12)`` -> ``{"left": (..., 6), "right": (..., 6)}`` (copies)."""
    x = np.asarray(x)
    if x.shape[-1] != ARM_DIM * len(ARMS):
        raise ValueError(f"expected a trailing dimension of {ARM_DIM * len(ARMS)}, got {x.shape}")
    return {arm: x[..., arm_slice(arm)].copy() for arm in ARMS}


def merge_bimanual(parts: dict[str, np.ndarray]) -> np.ndarray:
    """Inverse of :func:`split_bimanual`; both arms are required."""
    missing = [arm for arm in ARMS if arm not in parts]
    if missing:
        raise ValueError(f"missing arms {missing}")
    return np.concatenate([np.asarray(parts[arm], dtype=np.float64) for arm in ARMS], axis=-1)


def snapshot_indices(n_obs_steps: int, history_snapshots: int) -> tuple[int, ...]:
    """History ticks the model actually encodes as images.

    Mirrors lerobot@146d847:src/lerobot/policies/flux3/f3/packing.py:457-461 (``pack_history_video``):
    8 ticks and 2 snapshots encode ticks 0 (oldest) and 7 (current). Other ticks only feed state/commands.
    """
    if not 1 <= history_snapshots <= n_obs_steps:
        raise ValueError("need 1 <= history_snapshots <= n_obs_steps")
    if history_snapshots == 1:
        return (n_obs_steps - 1,)
    return tuple(round(j * (n_obs_steps - 1) / (history_snapshots - 1)) for j in range(history_snapshots))


@dataclass(frozen=True)
class ModelContract:
    """What one policy package consumes and produces (model units, not robot units)."""

    name: str
    action_dim: int
    camera_keys: tuple[str, ...]  # model camera keys, in layout order
    n_obs_steps: int = 8
    history_snapshots: int = 2
    chunk_size: int = 42
    n_action_steps: int = 32
    fps: float = 30.0
    # Every camera is resized to this (H, W) on the robot side before transport. The model's own canvas
    # compositor resizes again (packing.py:162-177), so this only has to preserve what the model will see.
    image_hw: tuple[int, int] = (256, 256)
    absolute_dims: tuple[int, ...] = (ARM_DIM - 1,)
    output_space: str = "joints"  # "joints" or "ee50"
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_space not in ("joints", "ee50"):
            raise ValueError("output_space must be joints or ee50")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("need 1 <= n_action_steps <= chunk_size")
        snapshot_indices(self.n_obs_steps, self.history_snapshots)

    @property
    def snapshots(self) -> tuple[int, ...]:
        return snapshot_indices(self.n_obs_steps, self.history_snapshots)


SO101_CONTRACT = ModelContract(
    name="flux-3-action-so101",
    action_dim=ARM_DIM,
    camera_keys=(SO101_SCENE, SO101_WRIST),
)

# Option C (new 12-D joint heads, see ws4/research/flux_dual_arm.md): three cameras on a 2x2 grid canvas of
# 384x512, so every cell is 192x256 = the 4:3 aspect of the cameras (lerobot utils.py:52-72 compose_grid).
BIMANUAL_GRID_CONTRACT = ModelContract(
    name="xlerobot-bimanual-grid",
    action_dim=2 * ARM_DIM,
    camera_keys=(IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST),
    image_hw=(192, 256),
    absolute_dims=(ARM_DIM - 1, 2 * ARM_DIM - 1),
)
