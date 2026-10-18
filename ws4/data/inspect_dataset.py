"""Print a LeRobot dataset's data contract and diff it against the FLUX SO-101 and XLeRobot 12-D contracts.

Usage:
    python -I ws4/data/inspect_dataset.py /path/to/local/dataset_root
    python -I ws4/data/inspect_dataset.py --hub MrC4t/bi_so_bag          # downloads meta/ files only
    python -I ws4/data/inspect_dataset.py --hub lerobot/community_dataset_v3::356c/so100_nut_sort_1
    python -I ws4/data/inspect_dataset.py --hub MrC4t/bi_so_bag --json   # machine-readable report

Only metadata is read (info.json, stats.json, tasks). Nothing from the dataset is executed. Run with `python -I`
so a file planted next to downloaded data cannot shadow a module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import (  # noqa: E402
    ARM_JOINTS,
    CONTRACTS,
    Contract,
    arm_units,
    camera_role,
    canonical_joint,
    guess_units,
)

META_FILES_V3 = ("meta/info.json", "meta/stats.json", "meta/tasks.parquet")
META_FILES_V2 = ("meta/info.json", "meta/stats.json", "meta/tasks.jsonl")
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".cache" / "hub_meta_full"


# ----------------------------------------------------------------------------------------------- loading


def download_meta(repo_id: str, cache: Path = DEFAULT_CACHE) -> Path:
    """Download only the meta/ files of a Hub dataset. `repo::sub/dir` addresses a dataset inside a collection."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    repo, _, sub = repo_id.partition("::")
    root = cache / repo_id.replace("/", "__").replace("::", "___")
    prefix = f"{sub}/" if sub else ""
    local = root / sub if sub else root
    info_path = Path(hf_hub_download(repo, prefix + "meta/info.json", repo_type="dataset", local_dir=root))
    v3 = json.loads(info_path.read_text()).get("codebase_version", "").startswith("v3")
    for fname in (META_FILES_V3 if v3 else META_FILES_V2)[1:]:
        if (local / fname).exists():
            continue
        try:
            hf_hub_download(repo, prefix + fname, repo_type="dataset", local_dir=root)
        except EntryNotFoundError:
            pass
    return local


def load_meta(root: Path) -> dict:
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    stats = {}
    if (root / "meta/stats.json").exists():
        stats = json.loads((root / "meta/stats.json").read_text())
    tasks: list[str] = []
    if (root / "meta/tasks.parquet").exists():
        import pandas as pd

        df = pd.read_parquet(root / "meta/tasks.parquet")
        tasks = [str(t) for t in (df["task"] if "task" in df.columns else df.index)]
    elif (root / "meta/tasks.jsonl").exists():
        for line in (root / "meta/tasks.jsonl").read_text().splitlines():
            if line.strip():
                tasks.append(json.loads(line).get("task", ""))
    return {"info": info, "stats": stats, "tasks": tasks}


# ----------------------------------------------------------------------------------------------- analysis


def _names(feature: dict) -> list[str]:
    names = feature.get("names")
    if isinstance(names, dict):  # v2.0 style {"motors": [...]}
        names = next(iter(names.values()), None)
    if not names:
        names = [f"dim_{i}" for i in range(int((feature.get("shape") or [0])[0]))]
    return list(names)


def describe(meta: dict) -> dict:
    """Summarize vectors (with per-joint unit guesses), cameras and size of one dataset."""
    info, stats = meta["info"], meta["stats"]
    feats = info.get("features", {})
    out: dict = {
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "hours": round(info["total_frames"] / info["fps"] / 3600, 3) if info.get("fps") else None,
        "n_tasks": len(meta["tasks"]) or info.get("total_tasks"),
        "tasks_sample": meta["tasks"][:8],
        "cameras": {},
        "vectors": {},
    }
    for key, f in feats.items():
        if f.get("dtype") in ("video", "image"):
            shape = f.get("shape") or [None, None]
            vi = f.get("info") or {}
            hw = (shape[0], shape[1]) if f.get("names", ["height"])[0] in ("height", None) else (shape[1], shape[2])
            out["cameras"][key] = {
                "hw": list(hw),
                "fps": vi.get("video.fps"),
                "codec": vi.get("video.codec"),
                "depth": vi.get("video.is_depth_map", False),
                "role": camera_role(key),
            }
        elif key in ("action", "observation.state") or (
            key.startswith("observation.") and str(f.get("dtype", "")).startswith("float")
        ):
            names = _names(f)
            s = stats.get(key, {})
            chans = []
            for i, n in enumerate(names):
                side, joint = canonical_joint(n) if key in ("action", "observation.state") else (None, None)
                lo = s.get("min", [None] * len(names))[i] if s else None
                hi = s.get("max", [None] * len(names))[i] if s else None
                mean = s.get("mean", [None] * len(names))[i] if s else None
                ch = {"name": n, "side": side, "joint": joint}
                if lo is not None and hi is not None:
                    lo, hi = float(_scalar(lo)), float(_scalar(hi))
                    ch.update(min=round(lo, 2), max=round(hi, 2), mean=round(float(_scalar(mean)), 2))
                    if joint:
                        ch["units"] = guess_units(joint, lo, hi, float(_scalar(mean)))
                chans.append(ch)
            out["vectors"][key] = chans
    # One decision per arm from action and state together: LeRobot records leader (action) and follower
    # (state) with the same norm mode, so a value beyond +-100 in either proves degrees for both.
    out["arm_units"] = {}
    chans = out["vectors"].get("action", []) + out["vectors"].get("observation.state", [])
    for side in sorted({str(c["side"]) for c in chans if c["joint"]}):
        arm = [(c["joint"], c["min"], c["max"]) for c in chans if c["joint"] and str(c["side"]) == side and "min" in c]
        if arm:
            out["arm_units"][side] = arm_units(arm)
    return out


def _scalar(v):
    while isinstance(v, list):
        v = v[0]
    return v


def diff_against(desc: dict, contract: Contract) -> dict:
    """List what must change to turn this dataset into `contract`, plus index maps converters can use."""
    issues: list[str] = []
    plan: dict = {"contract": contract.name, "ok": True}

    ver = desc["codebase_version"]
    if ver != contract.codebase_version:
        issues.append(
            f"codebase {ver} -> {contract.codebase_version} (lerobot convert_dataset_v21_to_v30; v2.0 needs v2.1 first)"
        )

    fps = desc["fps"]
    if fps != contract.fps:
        if fps and fps % contract.fps == 0:
            issues.append(f"fps {fps} -> {contract.fps}: keep every {fps // contract.fps}th frame (re-encode videos)")
        else:
            issues.append(f"fps {fps} -> {contract.fps}: resample (interpolate joints, re-time video)")
        plan["fps_ratio"] = (fps or 0) / contract.fps

    sides = ("left", "right") if contract.sides == ("left", "right") else None
    for vec in ("action", "observation.state"):
        chans = desc["vectors"].get(vec)
        if not chans:
            issues.append(f"missing {vec}")
            plan["ok"] = False
            continue
        by_key = {}
        for i, c in enumerate(chans):
            if c["joint"]:
                by_key[(c["side"], c["joint"])] = i
        found_sides = sorted({s for (s, _) in by_key}, key=str)
        if sides is None:  # per-arm contract: any side with a full arm qualifies
            per_side = {}
            for s in found_sides:
                idx = [by_key.get((s, j)) for j in ARM_JOINTS]
                if None not in idx:
                    per_side[s or "single"] = idx
            plan[f"{vec}_index"] = per_side
            if not per_side:
                issues.append(f"{vec}: no complete 6-joint arm among {[c['name'] for c in chans]}")
                plan["ok"] = False
        else:
            idx = [by_key.get((s, j)) for s in sides for j in ARM_JOINTS]
            plan[f"{vec}_index"] = idx
            if None in idx:
                issues.append(f"{vec}: cannot find both arms ({[c['name'] for c in chans]})")
                plan["ok"] = False
        extra = [c["name"] for c in chans if not c["joint"]]
        if extra:
            issues.append(f"{vec}: drop {len(extra)} non-arm channels {extra}")
        units = sorted(set(desc.get("arm_units", {}).values())) or ["unknown"]
        if units != [contract.arm_units]:
            issues.append(f"{vec}: arm units {units} -> {contract.arm_units}")
        gunits = sorted({c.get("units", "?") for c in chans if c["joint"] == "gripper"})
        if gunits and gunits != [contract.gripper_units]:
            issues.append(f"{vec}: gripper units {gunits} -> {contract.gripper_units}")

    roles = {k: v["role"] for k, v in desc["cameras"].items() if not v.get("depth")}
    scene = [k for k, r in roles.items() if r == "scene"]
    cam_map: dict[str, str | None] = {}
    for target, role in contract.cameras.items():
        if role == "scene":
            cam_map[target] = scene[0] if scene else None
        elif role == "wrist:self":
            wr = [k for k, r in roles.items() if r.startswith("wrist") or r.startswith("unknown:")]
            cam_map[target] = "per-side: " + ", ".join(wr) if wr else None
        else:
            side = role.split(":")[1]
            wr = [k for k, r in roles.items() if r == f"wrist:{side}"] or [
                k for k, r in roles.items() if r == f"unknown:{side}"
            ]
            cam_map[target] = wr[0] if wr else None
    plan["camera_map"] = cam_map
    missing = [t for t, s in cam_map.items() if s is None]
    if missing:
        issues.append(f"cameras: no source for {missing} (have {roles})")
        plan["ok"] = False
    if len(scene) > 1:
        issues.append(f"cameras: several scene candidates {scene}, picked {scene[0]}; check frames")
    plan["issues"] = issues
    return plan


# ----------------------------------------------------------------------------------------------- CLI


def format_report(name: str, desc: dict, diffs: list[dict]) -> str:
    lines = [f"== {name}"]
    lines.append(
        f"   {desc['codebase_version']}  robot_type={desc['robot_type']}  fps={desc['fps']}"
        f"  episodes={desc['episodes']}"
        f"  frames={desc['frames']}  hours={desc['hours']}  tasks={desc['n_tasks']}"
    )
    lines.append(f"   arm units: {desc.get('arm_units')}")
    for t in desc["tasks_sample"][:5]:
        lines.append(f"   task: {t[:110]}")
    for k, c in desc["cameras"].items():
        depth = " DEPTH" if c["depth"] else ""
        lines.append(f"   cam {k}: {c['hw'][0]}x{c['hw'][1]} {c['codec']} role={c['role']}{depth}")
    for vec, chans in desc["vectors"].items():
        lines.append(f"   {vec} [{len(chans)}]")
        for c in chans:
            rng = f"[{c['min']:8.2f} {c['max']:8.2f}] mean {c['mean']:8.2f}" if "min" in c else ""
            lines.append(f"      {c['name']:<34} {str(c['side']):<6} {str(c['joint']):<14} {rng} {c.get('units', '')}")
    for d in diffs:
        lines.append(f"   vs {d['contract']}: {'convertible' if d['ok'] else 'NOT convertible'}")
        for i in d["issues"]:
            lines.append(f"      - {i}")
        lines.append(f"      camera map: {d['camera_map']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="*", type=Path, help="local dataset roots")
    ap.add_argument("--hub", nargs="*", default=[], help="Hub dataset ids (meta/ only is downloaded)")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--contracts", nargs="*", default=list(CONTRACTS))
    ap.add_argument("--json", action="store_true", help="print one JSON object per dataset")
    args = ap.parse_args(argv)

    targets = [(str(r), Path(r)) for r in args.roots]
    for rid in args.hub:
        targets.append((rid, download_meta(rid, args.cache)))
    for name, root in targets:
        try:
            desc = describe(load_meta(root))
        except FileNotFoundError as e:
            print(f"== {name}: {e}", file=sys.stderr)
            continue
        diffs = [diff_against(desc, CONTRACTS[c]) for c in args.contracts]
        if args.json:
            print(json.dumps({"id": name, **desc, "diffs": diffs}))
        else:
            print(format_report(name, desc, diffs))
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)  # skip interpreter teardown: pyarrow deadlocked there in 2 of ~12 laptop runs
