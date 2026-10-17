"""Turn one haller VR-teleop recording into FLUX training datasets with robot-frame sides.

The haller HMI recorder writes one bimanual dataset per task (12-D `[left 6 | right 6]`, cameras `head`,
`left_wrist`, `right_wrist`, joints in LeRobot degrees). Its `left` / `right` are the OPERATOR's sides, standing in
front of the bench (haller_ws config.desktop-real.yaml "WHOSE LEFT?", confirmed by Oscar 2026-10-17), so haller
`left` is the robot's right arm. ws4/sim, the public XLeRobot datasets and the head camera image all use the
robot's own sides. This wrapper maps them and calls ws4/data/convert_bimanual.py `remap` (no video re-encode):

    per_arm   <name>_left, <name>_right: one robot arm each, 6-D, cameras scene (= head) + wrist (that arm's
              gripper cam), robot_type so_follower, task suffix " with the <robot side> arm" (the same
              convention ws4/data/fetch_public.sh used for the public per-arm sets)
    xle12     <name>_xle12: 12-D [robot left 6 | robot right 6], cameras head / left_wrist / right_wrist,
              robot_type bi_so_follower

    python -I ws4/teleop/to_flux.py SRC OUT_DIR [--name NAME] [--targets per_arm xle12]
        [--check-against PUBLIC_PER_ARM_ROOT ...] [--sides-as-recorded]

`--sides-as-recorded` skips the swap, for recordings already in robot-frame sides (e.g. lerobot-record with
bi_so_follower). Joint units must be degrees: the haller recorder says so in info.json
(`haller_joint_calibration.state_unit`), anything else needs --src-units.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data"))
from contracts import ARM_JOINTS  # noqa: E402
from convert_bimanual import remap  # noqa: E402
from merge_datasets import compatibility  # noqa: E402

ROBOT_SIDES = ("left", "right")
# robot side -> haller (operator-frame) side
HALLER_TO_ROBOT_SWAP = {"left": "right", "right": "left"}
IDENTITY = {"left": "left", "right": "right"}


def side_map(sides_as_recorded: bool) -> dict[str, str]:
    """robot side -> side name used in the recording."""
    return dict(IDENTITY if sides_as_recorded else HALLER_TO_ROBOT_SWAP)


def detect_units(info: dict, override: str | None) -> str:
    if override:
        return override
    block = info.get("haller_joint_calibration") or {}
    if block.get("state_unit") == "deg":
        return "degrees"
    raise SystemExit("cannot tell the joint units: no haller_joint_calibration.state_unit; pass --src-units")


def _annotate(dst: Path, block: dict, names: list[str] | None = None) -> dict:
    p = dst / "meta" / "info.json"
    info = json.loads(p.read_text())
    if names is not None:
        for v in ("observation.state", "action"):
            info["features"][v]["names"] = list(names)
    info["ws4_teleop"] = block
    p.write_text(json.dumps(info, indent=4))
    return info


def convert(src: Path, out_dir: Path, name: str, targets: tuple[str, ...], sides_as_recorded: bool = False,
            src_units: str | None = None, task_suffix: str | None = None) -> dict[str, Path]:
    info = json.loads((src / "meta" / "info.json").read_text())
    units = detect_units(info, src_units)
    smap = side_map(sides_as_recorded)
    frame = "recorded" if sides_as_recorded else "haller_operator_frame"
    written: dict[str, Path] = {}
    if "per_arm" in targets:
        for side in ROBOT_SIDES:
            rec = smap[side]
            dst = out_dir / f"{name}_{side}"
            suffix = f" with the {side} arm" if task_suffix is None else task_suffix
            remap(src, dst, (rec,), {"observation.images.scene": "observation.images.head",
                                     "observation.images.wrist": f"observation.images.{rec}_wrist"},
                  units, "so_follower", task_suffix=suffix)
            _annotate(dst, {"source": str(src), "robot_side": side, "recorded_side": rec, "recorded_frame": frame,
                            "task_suffix": suffix},
                      names=[f"{j}.pos" for j in ARM_JOINTS])
            written[side] = dst
    if "xle12" in targets:
        dst = out_dir / f"{name}_xle12"
        recorded_order = (smap["left"], smap["right"])
        remap(src, dst, recorded_order,
              {"observation.images.head": "observation.images.head",
               "observation.images.left_wrist": f"observation.images.{smap['left']}_wrist",
               "observation.images.right_wrist": f"observation.images.{smap['right']}_wrist"},
              units, "bi_so_follower", task_suffix="" if task_suffix is None else task_suffix)
        # remap names the channels after the recorded sides it read; relabel them as robot sides
        _annotate(dst, {"source": str(src), "column_order": "robot left 6 | robot right 6",
                        "recorded_sides_in_order": list(recorded_order), "recorded_frame": frame},
                  names=[f"{s}_{j}.pos" for s in ROBOT_SIDES for j in ARM_JOINTS])
        written["xle12"] = dst
    return written


def precheck(refs: list[Path], outs: list[Path]) -> list[tuple[Path, list[str]]]:
    infos = [json.loads((p / "meta" / "info.json").read_text()) for p in [*refs, *outs]]
    return list(zip([*refs, *outs], compatibility(infos), strict=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--name", default=None, help="output prefix, default: source directory name")
    ap.add_argument("--targets", nargs="+", choices=("per_arm", "xle12"), default=["per_arm", "xle12"])
    ap.add_argument("--sides-as-recorded", action="store_true", help="recording already uses robot-frame sides")
    ap.add_argument("--src-units", choices=("degrees", "m100"), default=None)
    ap.add_argument("--task-suffix", default=None, help="override the per-arm ' with the <side> arm' suffix")
    ap.add_argument("--check-against", type=Path, nargs="*", default=[],
                    help="per-arm reference datasets (e.g. converted public sets) for LeRobot's merge rule")
    args = ap.parse_args(argv)

    name = args.name or args.src.resolve().name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = convert(args.src, args.out_dir, name, tuple(args.targets), args.sides_as_recorded, args.src_units,
                      args.task_suffix)
    for k, p in written.items():
        print(f"wrote {k:5s} {p}")
    rc = 0
    per_arm = [written[s] for s in ROBOT_SIDES if s in written]
    if args.check_against and per_arm:
        for p, why in precheck(args.check_against, per_arm):
            print(f"{'ok ' if not why else 'NO '} merge {p}" + ("" if not why else "\n      " + "\n      ".join(why)))
            rc |= bool(why)
    return rc


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)  # same reason as convert_bimanual.py: pyarrow teardown deadlocks on this laptop
