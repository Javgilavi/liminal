"""Check a haller VR-teleop recording against the FLUX data contract before it goes anywhere.

    python -I ws4/teleop/validate.py DATASET_ROOT [--prompts ws4/teleop/prompts.yaml] [--calib-dir DIR] [--json]

Hard failures (exit 1) are what would break or silently corrupt FLUX training: fps, codec and frame size (merge
with the public core set), camera keys, 12-D layout and joint order, units not degrees, gripper outside 0..100,
video frames != parquet rows in any episode, action identical to state (no teleop session was running), more
than one task per episode, a task string not in the frozen prompt list, and a calibration in info.json that
differs from the files deployment will load. Warnings are worth reading but do not block: sampling gaps,
measured rate drift, short episodes, sim-rig data.

Contract sources: ws4/data/contracts.py (FLUX SO-101, XLE 12-D), haller_ws hmi/backend/haller_hmi/recorder.py
(schema, `haller_*` info.json blocks), LeRobot datasets/feature_utils.py:143-171 (merge rule).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
HALLER_SIDES = ("left", "right")
STATE_NAMES = [f"{s}_{j}" for s in HALLER_SIDES for j in ARM_JOINTS]
CAMERAS = ("head", "left_wrist", "right_wrist")
FPS = 30
FRAME_HW = (480, 640)
CODEC = "av1"
CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
# haller arm id -> calibration id in ws4/teleop/configs/station-real.yaml
HALLER_CALIB_IDS = {"left": "haller_left", "right": "haller_right"}
CALIB_FIELDS = (("range_min_ticks", "range_min"), ("range_max_ticks", "range_max"),
                ("homing_offset", "homing_offset"), ("drive_mode", "drive_mode"))


@dataclass
class Report:
    root: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _names(feature: dict) -> list[str]:
    n = feature.get("names")
    if isinstance(n, dict):
        n = next(iter(n.values()))
    return [str(x).removesuffix(".pos") for x in (n or [])]


def load_prompts(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    import yaml

    raw = yaml.safe_load(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("prompts", raw)
    out = []
    for v in (raw.values() if isinstance(raw, dict) else raw):
        out.extend(v if isinstance(v, list) else [v])
    return [str(x) for x in out]


def check_info(info: dict, rep: Report) -> None:
    if info.get("codebase_version") != "v3.0":
        rep.err(f"codebase_version {info.get('codebase_version')} != v3.0")
    if info.get("fps") != FPS:
        rep.err(f"fps {info.get('fps')} != {FPS} (start the backend with ws4/teleop/backend.py --session-hz 30)")
    feats = info["features"]
    for v in ("observation.state", "action"):
        if v not in feats:
            rep.err(f"missing {v}")
            continue
        if feats[v]["shape"] != [12]:
            rep.err(f"{v} shape {feats[v]['shape']} != [12]")
        if _names(feats[v]) != STATE_NAMES:
            rep.err(f"{v} names {_names(feats[v])} != haller order {STATE_NAMES}")
    for cam in CAMERAS:
        key = f"observation.images.{cam}"
        f = feats.get(key)
        if f is None:
            rep.err(f"missing camera {key} (dataset_key in the config)")
            continue
        vi = f.get("info") or {}
        hw = (vi.get("video.height"), vi.get("video.width"))
        if hw != FRAME_HW:
            rep.err(f"{key}: {hw[0]}x{hw[1]} != 480x640 (public core set and FLUX recipes)")
        if vi.get("video.codec") != CODEC:
            rep.err(f"{key}: codec {vi.get('video.codec')} != av1, not mergeable with the public core set "
                    "(record with backend.py --vcodec libsvtav1 or run ws4/teleop/transcode.py)")
    extra = sorted(k for k in feats if k.startswith("observation.images.") and k.split(".")[-1] not in CAMERAS)
    if extra:
        rep.warn(f"extra cameras recorded, dropped by to_flux.py: {extra}")
    cal = info.get("haller_joint_calibration") or {}
    if cal.get("state_unit") != "deg":
        rep.err(f"joint units {cal.get('state_unit')!r} != 'deg' (haller_joint_calibration.state_unit)")
    sources = {j.get("source") for j in (cal.get("joints") or {}).values()}
    rep.facts["rig"] = "real" if sources == {"feetech_calibration"} else "sim" if "declared_joint_range" in sources \
        else f"unknown {sources}"
    if rep.facts["rig"] == "sim":
        rep.warn("sim rig: haller's sim reports the gripper as the MJCF jaw angle and its zero is the old SO-ARM100 "
                 "pose, so sim takes check the pipeline only. Do not train FLUX on them.")
    rate = info.get("haller_rate") or {}
    if rate.get("measured_hz"):
        rep.facts["measured_hz"] = round(float(rate["measured_hz"]), 3)
        drift = abs(float(rate["measured_hz"]) - FPS) / FPS
        if drift > 0.005:
            rep.err(f"measured rate {rate['measured_hz']:.3f} Hz is {drift:.2%} off {FPS}")
        elif drift > 0.003:
            rep.warn(f"measured rate {rate['measured_hz']:.3f} Hz is {drift:.2%} off {FPS} (recorder refuses >0.5%)")


def check_calibration(info: dict, calib_dir: Path, rep: Report) -> None:
    """The ticks the recorder saw must equal the files deployment loads (official brief: keep calibration fixed)."""
    joints = (info.get("haller_joint_calibration") or {}).get("joints") or {}
    for side, cid in HALLER_CALIB_IDS.items():
        path = calib_dir / f"{cid}.json"
        if not path.exists():
            rep.err(f"calibration file {path} missing (ws4/teleop/calib.py harvest)")
            continue
        file = json.loads(path.read_text())
        diffs = []
        for j in ARM_JOINTS:
            rec = joints.get(f"{side}_{j}") or {}
            for rk, fk in CALIB_FIELDS:
                if rec.get(rk) is not None and int(rec[rk]) != int(file[j][fk]):
                    diffs.append(f"{j}.{fk} {rec[rk]} != {file[j][fk]}")
        if diffs:
            rep.err(f"{side} arm: recorded calibration differs from {path.name}: " + "; ".join(diffs[:6]))
        rep.facts.setdefault("calibration_files", {})[side] = str(path)


def video_frames(path: Path) -> int:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return s.frames or sum(1 for _ in c.demux(s) if _.size)


def check_episodes(root: Path, info: dict, prompts: list[str] | None, rep: Report) -> None:
    import numpy as np
    import pandas as pd

    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    rep.facts["episodes"] = int(len(eps))
    rep.facts["frames"] = int(len(data))
    tasks = sorted({t for ts in eps["tasks"] for t in ts})
    rep.facts["tasks"] = tasks
    if prompts is not None:
        bad = [t for t in tasks if t not in prompts]
        if bad:
            rep.err(f"task strings not in the frozen prompt list: {bad}")
    st = np.stack(data["observation.state"].to_numpy())
    ac = np.stack(data["action"].to_numpy())
    grip = [i for i, n in enumerate(STATE_NAMES) if n.endswith("gripper")]
    if rep.facts.get("rig") == "real":
        for arr, nm in ((st, "observation.state"), (ac, "action")):
            lo, hi = float(arr[:, grip].min()), float(arr[:, grip].max())
            if lo < -0.5 or hi > 100.5:
                rep.err(f"{nm} gripper range [{lo:.1f}, {hi:.1f}] outside 0..100")
    arm = [i for i in range(12) if i not in grip]
    if np.abs(st[:, arm]).max() > 180.5:
        rep.err("an arm joint exceeds +-180 deg: not LeRobot degrees with a mid-range zero")
    for _, e in eps.iterrows():
        idx = int(e["episode_index"])
        n = int(e["length"])
        rows = data[data["episode_index"] == idx]
        if len(e["tasks"]) != 1:
            rep.err(f"episode {idx}: {len(e['tasks'])} tasks, want exactly 1")
        if len(rows) != n:
            rep.err(f"episode {idx}: {len(rows)} parquet rows != length {n}")
        if n < FPS:
            rep.warn(f"episode {idx}: only {n} frames")
        d = np.abs(np.stack(rows["action"].to_numpy()) - np.stack(rows["observation.state"].to_numpy()))
        if float(d.max()) < 1e-3:
            rep.err(f"episode {idx}: action == observation.state, no teleop session drove this take")
        if "observation.wall_clock" in rows:
            wc = np.stack(rows["observation.wall_clock"].to_numpy()).ravel()
            gaps = np.diff(wc)
            if len(gaps) and float(gaps.max()) > 2.5 / FPS:
                rep.warn(f"episode {idx}: sampling gap {gaps.max() * 1000:.0f} ms at frame {int(gaps.argmax()) + 1}")
        for cam in CAMERAS:
            key = f"observation.images.{cam}"
            if key not in info["features"]:
                continue
            ci = int(e[f"videos/{key}/chunk_index"])
            fi = int(e[f"videos/{key}/file_index"])
            path = root / info["video_path"].format(video_key=key, chunk_index=ci, file_index=fi)
            if not path.exists():
                rep.err(f"episode {idx}: missing {path.relative_to(root)}")
                continue
            t0 = float(e[f"videos/{key}/from_timestamp"])
            t1 = float(e[f"videos/{key}/to_timestamp"])
            have = video_frames(path)
            want = round((t1 - t0) * FPS)
            if have < want or want != n:
                rep.err(f"episode {idx} {cam}: video {have} frames for [{t0:.3f}, {t1:.3f}] s, parquet {n} rows "
                        "(encoder dropped frames?)")


def validate(root: Path, prompts: list[str] | None = None, calib_dir: Path | None = CALIB_DIR) -> Report:
    rep = Report(str(root))
    info = json.loads((root / "meta" / "info.json").read_text())
    check_info(info, rep)
    if calib_dir is not None and rep.facts.get("rig") == "real":
        check_calibration(info, calib_dir, rep)
    check_episodes(root, info, prompts, rep)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", type=Path, nargs="+")
    ap.add_argument("--prompts", type=Path, default=None, help="YAML list (or {prompts: [...]}) of allowed tasks")
    ap.add_argument("--calib-dir", type=Path, default=CALIB_DIR)
    ap.add_argument("--no-calib", action="store_true", help="skip the calibration-file comparison")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    prompts = load_prompts(args.prompts)
    rc = 0
    for root in args.roots:
        rep = validate(root, prompts, None if args.no_calib else args.calib_dir)
        rc |= bool(rep.errors)
        if args.json:
            print(json.dumps(rep.__dict__, indent=1))
            continue
        print(f"{'PASS' if not rep.errors else 'FAIL'} {root}")
        print("     " + ", ".join(f"{k}={v}" for k, v in rep.facts.items()))
        for m in rep.errors:
            print(f"  ERROR {m}")
        for m in rep.warnings:
            print(f"  warn  {m}")
    return rc


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)
