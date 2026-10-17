"""Small real joint trajectories from the Hugging Face Hub for the EE50/IK gate.

Only ``meta/info.json`` and one data parquet per dataset are downloaded (no videos), into
``ws4/.cache/kin/hub/<owner>__<name>/``. Parquet is parsed with pyarrow, never executed.

Units: LeRobot SO-101 joints are either ``DEGREES`` (mid-range zero, current default) or the older
``RANGE_M100_100``. The parquet does not say which, so :func:`guess_units` looks at the value range: a
wrist roll or pan beyond +-101 can only be degrees. Everything else is treated as M100 and converted with
an approximate per-joint scale (see ``so101_kinematics.M100_DEG_PER_UNIT``). For the round-trip gate the
exact calibration does not matter: we need realistic joint-space motion, not the true table poses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .so101_kinematics import lerobot_deg_to_rad, lerobot_gripper_to_open, lerobot_m100_to_rad

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "kin" / "hub"

# id -> (data file, kind, column, left slice, right slice). Chosen from a Hub search on 2026-10-17
# (sizes checked with HfApi.get_paths_info) to cover single SO-101, bimanual SO-101 and XLeRobot.
DATASETS: dict[str, dict] = {
    "lerobot/svla_so101_pickplace": {"file": "data/chunk-000/file-000.parquet", "kind": "so101", "arms": {"right": 0}},
    "Zekai-Chen/bi_so101_fold_towel_4cam": {
        "file": "data/chunk-000/file-001.parquet",
        "kind": "bi_so101",
        "arms": {"left": 0, "right": 6},
    },
    "xlerobot-team/xlerobot-left-pick-cup-sep2-20260902-1819": {
        "file": "data/chunk-000/file-000.parquet",
        "kind": "xlerobot",
        "arms": {"left": 0, "right": 6},
    },
    "saivishwak/xlerobot-vr-teleop-medicine-bowl": {
        "file": "data/chunk-000/file-000.parquet",
        "kind": "xlerobot",
        "arms": {"left": 0, "right": 6},
    },
    "Odog16/xlerobot_generalist_v2": {
        "file": "data/chunk-000/file-000.parquet",
        "kind": "xlerobot",
        "arms": {"left": 0, "right": 6},
    },
}
UMI_DATASET = {"id": "lerobot/umi_cup_in_the_wild", "file": "data/chunk-000/file-000.parquet"}


def local_dir(repo_id: str) -> Path:
    return CACHE / repo_id.replace("/", "__")


def fetch(repo_ids: list[str] | None = None) -> list[Path]:
    """Download info.json + the configured parquet of each dataset. Needs network; not used by tests."""
    from huggingface_hub import hf_hub_download

    specs = dict(DATASETS)
    specs[UMI_DATASET["id"]] = {"file": UMI_DATASET["file"]}
    out = []
    for rid in repo_ids or list(specs):
        spec = specs[rid]
        for f in ("meta/info.json", spec["file"]):
            out.append(Path(hf_hub_download(rid, f, repo_type="dataset", local_dir=local_dir(rid))))
    return out


@dataclass
class Episode:
    dataset: str
    episode: int
    fps: float
    q: dict[str, np.ndarray]  # arm -> (T, 5) radians
    gripper: dict[str, np.ndarray]  # arm -> (T,) open fraction
    units: str


def guess_units(joints: np.ndarray) -> str:
    """``deg`` if any joint leaves +-100.5 (impossible in RANGE_M100_100, which LeRobot clamps), else ``m100``.

    Degree recordings that never leave +-100 are misread as M100 and scaled by 1.0 to 1.8; harmless here.
    """
    return "deg" if np.abs(joints).max() > 100.5 else "m100"


def load_episodes(repo_id: str, column: str = "action", min_len: int = 60) -> list[Episode]:
    import pyarrow.parquet as pq

    spec = DATASETS[repo_id]
    d = local_dir(repo_id)
    info = json.loads((d / "meta" / "info.json").read_text())
    table = pq.read_table(d / spec["file"], columns=[column, "episode_index"])
    data = np.stack(table.column(column).to_numpy(zero_copy_only=False)).astype(float)
    ep = table.column("episode_index").to_numpy()
    # One unit guess per arm over the whole file: a degree recording only shows itself when some joint
    # leaves +-100, which a single short episode may never do.
    units = {arm: guess_units(data[:, start : start + 5]) for arm, start in spec["arms"].items()}
    episodes = []
    for e in np.unique(ep):
        rows = data[ep == e]
        if len(rows) < min_len:
            continue
        q, g = {}, {}
        for arm, start in spec["arms"].items():
            j = rows[:, start : start + 5]
            q[arm] = lerobot_deg_to_rad(j) if units[arm] == "deg" else lerobot_m100_to_rad(j)
            g[arm] = lerobot_gripper_to_open(rows[:, start + 5])
        episodes.append(Episode(repo_id, int(e), float(info.get("fps", 30)), q, g, "/".join(units.values())))
    return episodes


def load_umi(max_episodes: int | None = None, min_len: int = 30) -> list[dict]:
    """UMI handheld-gripper episodes: ``pos`` (T, 3) metres and ``rotvec`` (T, 3), ``gripper_width`` (T,).

    ``observation.state`` in ``lerobot/umi_cup_in_the_wild`` is 7-D ``motor_0..6`` without semantics in
    info.json; the UMI paper's cup task stores eef_pos (3) + eef_rot_axis_angle (3) + gripper width (1),
    which is what the values look like (metre-scale positions, rotation vectors with norm < pi). The
    layout is therefore UNVERIFIED beyond that sanity check, done in :func:`umi_sanity`.
    """
    import pyarrow.parquet as pq

    d = local_dir(UMI_DATASET["id"])
    table = pq.read_table(d / UMI_DATASET["file"], columns=["observation.state", "episode_index"])
    data = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(float)
    ep = table.column("episode_index").to_numpy()
    out = []
    for e in np.unique(ep):
        rows = data[ep == e]
        if len(rows) < min_len:
            continue
        out.append({"episode": int(e), "pos": rows[:, :3], "rotvec": rows[:, 3:6], "gripper_width": rows[:, 6]})
        if max_episodes and len(out) >= max_episodes:
            break
    return out


def umi_sanity(episodes: list[dict]) -> dict:
    pos = np.concatenate([e["pos"] for e in episodes])
    rv = np.concatenate([e["rotvec"] for e in episodes])
    step = np.concatenate([np.linalg.norm(np.diff(e["pos"], axis=0), axis=1) for e in episodes])
    return {
        "pos_range_m": [float(pos.min()), float(pos.max())],
        "rotvec_norm_max": float(np.linalg.norm(rv, axis=1).max()),
        "median_step_m": float(np.median(step)),
        "gripper_range": [float(np.min([e["gripper_width"].min() for e in episodes])),
                          float(np.max([e["gripper_width"].max() for e in episodes]))],
    }


if __name__ == "__main__":
    for p in fetch():
        print(p)
