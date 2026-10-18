"""The Haller skills of the final demo chain and the two task arms, in one table for the eval tooling.

Robot sides follow ws4 naming; the haller (operator) naming is mirrored: robot LEFT = haller right =
/dev/haller_arm_right = 12-D dataset columns ``right`` = wrist camera USB 7.2 (``right_wrist``); robot RIGHT = haller
left = /dev/haller_arm_left = columns ``left`` = wrist camera USB 7.3 (``left_wrist``). Both wrist cameras are mounted
upside down and recorded rotated 180 (haller_ws config.flux-real.yaml), so the runner rotates both by 180.

Per-skill artifacts live in ``skills/<name>/`` (``build_skill.py``): ``skill.json`` (dataset snapshot, verbatim task
prompt, split, keep-out validation), ``workspace.json`` (station keep-out of the task arm + the envelope of the train
takes) and ``layouts/`` (held-out start frames: placement and home-check references).

    python ws4/single_arm/skills.py env box      # shell assignments for eval_skill.sh
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL_DIR = Path(os.environ.get("WS4_SKILL_DIR", HERE / "skills"))  # env: tests only
CACHE = Path.home() / ".cache/huggingface/lerobot"
WRIST_72 = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0"
WRIST_73 = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.3:1.0-video-index0"


@dataclass(frozen=True)
class Arm:
    side: str  # robot side: left / right
    port: str
    robot_id: str  # LeRobot calibration id (haller naming)
    other_port: str  # the parked arm, read only for the keep-out's other_arm zone
    other_id: str
    wrist_cam: str  # by-path: both wrist cams report serial SN0001
    columns: str  # this arm's columns in a 12-D dataset (haller naming)
    wrist_key: str  # this arm's wrist camera in the dataset


ARMS = {
    "left": Arm("left", "/dev/haller_arm_right", "haller_right", "/dev/haller_arm_left", "haller_left", WRIST_72,
                "right", "observation.images.right_wrist"),
    "right": Arm("right", "/dev/haller_arm_left", "haller_left", "/dev/haller_arm_right", "haller_right", WRIST_73,
                 "left", "observation.images.left_wrist"),
}  # fmt: skip


@dataclass(frozen=True)
class Skill:
    name: str
    arm: str  # robot side of the task arm
    dataset: str  # LeRobot repo id in the local cache (new skills: UNVERIFIED until recorded)
    keywords: tuple[str, ...]  # to find the dataset in the cache if its name differs
    trial_task: str
    steps: tuple[str, str, str]  # graded steps, in order
    parked: tuple[float, ...] | None  # the other arm's pose in the demos when known before the build (tucked arm)
    layouts: str = "H H H H H"  # default attempt sequence over the held-out start frames


# The other arm during each skill (orchestrator, 18 Oct). Box skill: the pen arm parked where it sat in takes 7-74
# (bumped in take 7 and kept there; the early tuck of takes 0-6 is excluded from the box skill). Pen-arm skills:
# the box arm HOLDS the lid open (it closes if the arm leaves), at a pose that varies a little per run, so the green
# and cutter range can only come from their demos (build_skill.py records min/max per joint). The red-pen dataset
# was recorded with the box arm tucked (within 0.3 deg over all 96 good takes).
PEN_ARM_PARKED_FOR_BOX = (3.0, -83.6, 81.9, 82.8, -10.2, 0.9)
BOX_ARM_TUCKED = (3.1, -102.7, 96.7, 78.4, 13.0, 1.5)

SKILLS = {
    "box": Skill("box", "right", "ws4/haller_open_the_box", ("open_the_box", "box", "handle"), "open_box",
                 ("reach the handle", "grasp the handle", "open the box"), PEN_ARM_PARKED_FOR_BOX),
    "green": Skill("green", "left", "ws4/haller_put_the_green_pen_in_the_green_slot", ("green",),
                   "put_green_pen_in_slot",
                   ("reach the green pen", "grasp the green pen", "insert it in the green slot"), None),
    "cutter": Skill("cutter", "left", "ws4/haller_put_the_yellow_cutter_in_the_yellow_slot", ("cutter", "yellow"),
                    "put_cutter_in_slot", ("reach the cutter", "grasp the cutter", "insert it in the yellow slot"),
                    None),
    # The chain's red pen (box at its final position, lid held by the box arm): recorded on the morning of 18 Oct.
    # Separate from "pen" (the graded B84 / median / LoRA rows on the old setup) so the two never mix in a log.
    "pen_lid": Skill("pen_lid", "left", "ws4/haller_put_the_red_pen_in_its_slot_lid_held", ("lid_held",),
                     "put_pen_in_slot_lid_held", ("reach the pen", "grasp the pen", "insert it in its slot"), None),
    "pen": Skill("pen", "left", "ws4/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box", ("red_pen",),
                 "put_pen_in_slot", ("reach the pen", "grasp the pen", "insert it in its slot"), BOX_ARM_TUCKED,
                 "A B C D A"),
}  # fmt: skip


def skill_info(name: str) -> dict:
    """The built ``skills/<name>/skill.json``, with relative paths resolved against the repo root."""
    path = SKILL_DIR / name / "skill.json"
    if not path.exists():
        raise SystemExit(f"skill {name!r} is not built yet: python ws4/single_arm/build_skill.py {name}")
    return json.loads(path.read_text())


def find_dataset(skill: Skill) -> list[Path]:
    """Cache dirs that are the skill's dataset, or match its keywords (newest first)."""
    exact = CACHE / skill.dataset
    if exact.exists():
        return [exact]
    hits = [p for p in (CACHE / "ws4").glob("*") if p.is_dir() and any(k in p.name for k in skill.keywords)]
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)


def parked_range(skill: Skill, info: dict) -> dict[str, str]:
    """SKILL_PARKED_LO / _HI: per-joint min / max of the other arm over the skill's good takes (skill.json), or the
    tucked pose known before the build. The runner widens them by --other-arm-tol."""
    gate = info.get("other_arm_gate")
    if gate:  # pen-arm chain skills: loose sanity range (already widened) + the geometric gate (gate.py)
        return {"SKILL_PARKED_LO": " ".join(f"{v:g}" for v in gate["joint_lo"]),
                "SKILL_PARKED_HI": " ".join(f"{v:g}" for v in gate["joint_hi"]), "SKILL_PARKED_TOL": "0",
                "SKILL_GATE_ARGS": f"--other-arm-demo-poses {gate['demo_poses']} "
                                   f"--other-arm-gate-max-below {gate['max_below']:g}"}  # fmt: skip
    parked = info.get("parked_other_arm") or {}
    if "min" in parked and "max" in parked:
        lo, hi = parked["min"], parked["max"]
    elif skill.parked is not None:
        lo = hi = skill.parked
    else:
        raise SystemExit(f"skill {skill.name!r}: no parked-arm range (rebuild with build_skill.py {skill.name})")
    return {"SKILL_PARKED_LO": " ".join(f"{v:g}" for v in lo), "SKILL_PARKED_HI": " ".join(f"{v:g}" for v in hi),
            "SKILL_PARKED_TOL": "5", "SKILL_GATE_ARGS": ""}


def shell_env(name: str) -> str:
    """Assignments for eval_skill.sh: arm I/O, prompt, steps, keep-out, references."""
    skill, info = SKILLS[name], skill_info(name)
    arm = ARMS[skill.arm]
    values = {
        "SKILL_TASK": info["task"], "SKILL_TRIAL_TASK": skill.trial_task, "SKILL_ARM": arm.side,
        "SKILL_PORT": arm.port, "SKILL_ROBOT_ID": arm.robot_id, "SKILL_OTHER_PORT": arm.other_port,
        "SKILL_OTHER_ID": arm.other_id, "SKILL_WRIST_CAM": arm.wrist_cam, "SKILL_WORKSPACE": info["workspace"],
        "SKILL_LAYOUT_REF": info["layouts"], "SKILL_HOME_REF": info.get("home_ref", info["layouts"]),
        "SKILL_LAYOUTS": info.get("eval_layouts", skill.layouts),
        "SKILL_VALID": int(bool(info.get("validation", {}).get("ok"))),
        "SKILL_REFS": int(info.get("refs", True)),
        "SKILL_RUNNER_ARGS": " ".join(info.get("runner_args") or []),  # per-skill runner flags (cutter: slow zone)
        # Chain skills with the lid-holding box arm: the runner takes the box arm over from the HMI and holds it.
        "SKILL_HOLD_OTHER": int(bool(info.get("other_arm_gate"))),
        "SKILL_PARKED_MEDIAN": " ".join(f"{v:g}" for v in (info.get("parked_other_arm") or {}).get("median", [])),
        **parked_range(skill, info),
    }  # fmt: skip
    lines = [f"{k}={shlex.quote(str(v))}" for k, v in values.items()]
    lines.append("SKILL_STEPS=(" + " ".join(shlex.quote(s) for s in skill.steps) + ")")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "env" and sys.argv[2] in SKILLS:
        print(shell_env(sys.argv[2]))
    else:
        raise SystemExit(f"usage: skills.py env {{{','.join(SKILLS)}}}")
