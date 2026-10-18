"""Build ws4/data/candidates.yaml from cached Hub metadata plus the hand-checked notes below.

Usage:
    python -I ws4/data/build_candidates.py            # reads ws4/.cache/{hub_meta_full,hub_sizes.json,datasets}

Inputs come from survey_hub.py / inspect_dataset.py --hub / hub_sizes.py / a data+meta download. Every hand
note says how it was checked: `frames` (contact sheet from sample_frames.py), `data` (trajectory parquet
read locally), `meta` (info.json/stats.json only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contracts import FLUX_SO101, XLE_12D  # noqa: E402
from inspect_dataset import describe, diff_against, load_meta  # noqa: E402

CACHE = HERE.parent / ".cache"

# Uses: 1 per-arm task LoRA on flux-3-action-so101; 2 EE50 fine-tune (via FK); 3 native 12-D joint fine-tune;
# 4 video-only / world-model co-training. Rank 1 is best; None means not useful for that use.
NOTES: dict[str, dict] = {
    "zonglin11/xlerobot_clean_table": dict(
        viewpoint="XLeRobot head camera behind and above both arms, tabletop (frames). Closest match to our mast.",
        gripper="white slatted fin-ray style fingers (frames), closest to our TPU fin-ray",
        flags=["head_motor_2.pos is a constant 1500 (junk channel, dropped)", "base unused (data)"],
        verified=["frames", "data"], ranks={1: 1, 2: 2, 3: 1, 4: 2}),
    "bigse0u1/xlerobot_scrub_7tool": dict(
        viewpoint="XLeRobot head camera behind the arms looking at a tool rack (frames). Good match.",
        gripper="stock-style white XLeRobot jaws (frames)",
        flags=["gripper ACTION is inverted vs state (state = 100 - action, data): flip before use",
               "wrist cams 240x320", "episodes are short (12 s) single-tool picks"],
        verified=["frames", "data"], ranks={1: 3, 2: 4, 3: 3, 4: 3}),
    "rk000000/xlerobot-handkerchief-quarters-mix-20260918-480": dict(
        viewpoint="XLeRobot head camera looking down at a table, both grippers at the image edges (frames). Good.",
        gripper="orange stock-style jaws (frames)",
        flags=["head tilt moved during recording (head_tilt.pos up to 61, data)", "cloth folding, not pick-place"],
        verified=["frames", "data"], ranks={1: 4, 2: 5, 3: 4, 4: 4}),
    "saivishwak/xlerobot-desk-marker-phase1": dict(
        viewpoint="XLeRobot head camera behind the arms, office desk (frames). Good match.",
        gripper="white slatted fin-ray style fingers (frames)",
        flags=["range_m100_100 (all arm values inside +-100, two channels stop at exactly -100): degrees need "
               "that robot's calibration; XLeRobot calibrates wrist_roll by a manual sweep, so its span is unknown",
               "robot_type xlerobot-bimanual-so101: already 12-D, no base/head channels"],
        verified=["frames", "data"], ranks={1: 6, 2: 6, 3: 2, 4: 5}),
    "saivishwak/xlerobot-vr-teleop-medicine-bowl": dict(
        viewpoint="same rig as xlerobot-desk-marker-phase1 (same author and camera keys, frames not checked)",
        flags=["VR teleop through IK; units not checked on trajectories"],
        verified=["meta"], ranks={1: 9, 2: 9, 3: 6, 4: 8}),
    "MrC4t/bi_so_bag": dict(
        viewpoint="'head' is a close top-down camera over a cutting mat, not a mast view (frames)",
        gripper="green printed jaws (frames)",
        flags=["degrees, bi_so_follower, clean contract (data)"],
        verified=["frames", "data"], ranks={1: 5, 2: 3, 3: 5, 4: 6}),
    "cortx-labs/bimanual-fruit-sorting": dict(
        viewpoint="high camera behind/above both arms (left_top), arms at the bottom of the image (frames). Good.",
        gripper="teal and orange printed jaws (frames)",
        flags=["long episodes (153 s mean): many sorting sub-steps per episode, good for long-horizon"],
        verified=["frames", "data"], ranks={1: 2, 2: 1, 3: 7, 4: 1}),
    "cortx-labs/bimanual-trash-and-plates-sorting": dict(
        viewpoint="same rig as cortx-labs/bimanual-fruit-sorting (same keys, frames not checked)",
        verified=["meta"], ranks={1: 7, 2: 7, 3: 8, 4: 7}),
    "Elvinky/bi-so101-insert-screw-562ep": dict(
        viewpoint="high front camera (right_front) over a white table, arms in view (frames). Medium match.",
        gripper="dark grey slatted fingers (frames)",
        flags=["wrist_flex action/state slope 0.5 (data): leader/follower calibration mismatch, check before use",
               "single task, 19 h, 21.5 GB of video"],
        verified=["frames", "data"], ranks={1: 8, 2: 8, 3: 9, 4: 9}),
    "nitred/bimanual-so-stack-cube_100": dict(
        viewpoint="front camera facing the robots (right_front), opposite of our head view (frames). Poor match.",
        verified=["frames", "data"], ranks={1: None, 2: 10, 3: None, 4: 12}),
    "Odog16/xlerobot_generalist_v2": dict(
        viewpoint="XLeRobot head camera, picking from the floor with a mobile base (frames)",
        flags=["base moves in 23% of frames (data): drop those windows", "gantry.height_mm channel"],
        verified=["frames", "data"], ranks={1: 10, 2: 11, 3: 10, 4: 10}),
    "siyulw2025/xlerobot-20251026-alldata-v3.0": dict(
        viewpoint="mobile XLeRobot in varied scenes; 'top' is the head camera; 'left'/'right' are wrists (frames)",
        flags=["range_m100_100 (data): degrees conversion needs that robot's calibration (approximate)",
               "base moves in 10% of frames", "22 tasks with explicit left/right arm language"],
        verified=["frames", "data"], ranks={1: 11, 2: 12, 3: 11, 4: 11}),
    "armnet/armnetbench_v01_lerobot_bimanual_so101": dict(
        flags=["20 fps (needs resampling to 30)", "43 GB of video", "4 tasks, 1219 episodes"],
        verified=["meta"], ranks={1: None, 2: 13, 3: None, 4: 13}),
    "Zekai-Chen/bi_so101_fold_towel_merged_r1": dict(
        flags=["towel folding only", "24.6 GB of video"], verified=["meta"], ranks={1: 12, 2: 14, 3: 12, 4: 14}),
}


# End-effector (handheld / UMI-style) sources for use 2, checked with ee_convert.py where `tested` is set.
# Full survey with sources: ws4/data/notes/human_handheld_datasets.md.
EE_CANDIDATES: list[dict] = [
    dict(id="axiboai/umi_stack_sohaib_no_roll", revision="2ae818e", kind="bimanual handheld UMI", fps=30,
         episodes=100, hours=0.38, size_gb=0.24, licence="apache-2.0", preset="axibo", tested="ee_convert (100 eps)",
         pose="per arm xyz + rot6d (row/col convention undocumented) + gripper", cameras="2 wrist 480x640",
         rank_use2=3),
    dict(id="rocochallenge2025/roco_iros2026_umi_dataset", revision="c00d869", kind="bimanual handheld (ISuit-V3)",
         fps=30, episodes=4928, hours=69.1, size_gb=248.6, trajectories_gb=0.96, licence=None, preset="roco",
         tested="ee_convert (40 eps)", pose="per arm xyz + quat xyzw, gripper command 0.24..1.05",
         cameras="fpv_left/right head stereo + 2 per hand, 384x480", tasks="industrial assembly (5 strings, Chinese)",
         flags=["no licence stated"], rank_use2=4),
    dict(id="Pantheon-inc/DiverseUMI-sample25", kind="bimanual handheld UMI", fps=30, episodes=4587, hours=25.0,
         size_gb=69, licence="Pantheon Commercial Use Licence v1.0", gate="manual",
         pose="per wrist xyz + rotvec + opening 0..1", cameras="top exo + 2 wrist 640x480",
         flags=["request access now", "not LeRobot: per-episode mp4 + npz"], rank_use2=5),
    dict(id="simple-world-lab/HiFi-UMI-2K", kind="bimanual handheld + head stereo", fps=25, hours=2000,
         licence="CC BY 4.0", pose="xyz + rot6d first two rows (EE50 style), gravity-aligned shared frame",
         flags=["smallest slice with video about 7.3 GB"], rank_use2=6),
    dict(id="IPEC-COMMUNITY/FastUMI_100k_lerobot", subset="dual_arm/*", kind="bimanual handheld UMI", fps=20,
         licence="apache-2.0", pose="xyz + roll/pitch/yaw + gripper 0..1, undefined relative frame",
         flags=["20 Hz", "likely already in FLUX midtraining"], rank_use2=7),
]


def build() -> list[dict]:
    sizes = json.loads((CACHE / "hub_sizes.json").read_text())
    out = []
    for d in sorted((CACHE / "hub_meta_full").iterdir()):
        rid = d.name.replace("__", "/", 1)
        try:
            desc = describe(load_meta(d))
        except (FileNotFoundError, KeyError):
            continue
        s = sizes.get(rid, {})
        note = NOTES.get(rid, {})
        local = CACHE / "datasets" / d.name
        cams = {k.split("observation.images.")[-1]: f"{v['hw'][0]}x{v['hw'][1]} {v['role']}"
                for k, v in desc["cameras"].items()}
        vec = desc["vectors"].get("action", [])
        row = {
            "id": rid,
            "url": f"https://huggingface.co/datasets/{rid}",
            "revision": s.get("sha"),
            "robot_type": desc["robot_type"],
            "codebase_version": desc["codebase_version"],
            "fps": desc["fps"],
            "episodes": desc["episodes"],
            "frames": desc["frames"],
            "hours": desc["hours"],
            "cameras": cams,
            "action_names": [c["name"] for c in vec],
            "arm_units": desc.get("arm_units"),
            "gripper_units": sorted({c.get("units", "?") for c in vec if c["joint"] == "gripper"}),
            "tasks": desc["n_tasks"],
            "tasks_sample": desc["tasks_sample"][:3],
            "license": s.get("license"),
            "size_gb": round(sum(s["bytes"].values()) / 1e9, 2) if "bytes" in s else None,
            "trajectories_mb": round(s["bytes"]["data"] / 1e6, 1) if "bytes" in s else None,
            "local_trajectories": str(local.relative_to(HERE.parent.parent)) if local.exists() else None,
            "contract_issues": {c.name: diff_against(desc, c)["issues"] for c in (FLUX_SO101, XLE_12D)},
            **{k: v for k, v in note.items() if k != "ranks"},
            "rank": {f"use{k}": v for k, v in note.get("ranks", {}).items()},
        }
        out.append(row)
    out.sort(key=lambda r: (r["rank"].get("use1") or 99, r["rank"].get("use3") or 99, -(r["hours"] or 0)))
    return out


HEADER = """\
# Candidate open datasets for adapting FLUX 3 Action to the XLeRobot station (two SO-101 arms, head camera,
# two wrist cameras, fin-ray grippers). Generated by ws4/data/build_candidates.py; edit NOTES there, not here.
# Uses: use1 per-arm task LoRA on flux-3-action-so101 | use2 EE50 fine-tune via FK | use3 native 12-D joint
# fine-tune | use4 video-only co-training. rank 1 = best, null = not useful. See ws4/research/datasets.md.
# arm_units per arm (action and state together): degrees (proved by a value beyond +-100), range_m100_100
# (saturates at +-100), ambiguous (cannot tell from metadata).
# ee_candidates: handheld/UMI-style sources for use 2; more in ws4/data/notes/human_handheld_datasets.md.
"""


def main() -> int:
    import yaml

    rows = build()
    text = HEADER + yaml.safe_dump({"candidates": rows, "ee_candidates": EE_CANDIDATES}, sort_keys=False,
                                   width=120, allow_unicode=True)
    (HERE / "candidates.yaml").write_text(text)
    print(f"{len(rows)} candidates -> {HERE / 'candidates.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
