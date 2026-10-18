"""Build one skill's real-arm eval artifacts from its finalised VR dataset.

1. Snapshot: copy the dataset from the local LeRobot cache and check that the frame counts agree (a dataset that is
   still being recorded has open parquet writers and fails: ask ws4-vr2 to finalise first).
2. Split: ``ws4_split.json`` at the dataset root when it exists (layouts, held-out, drop); otherwise review.json
   rejects are dropped and the last 3 good takes are held out (provisional, marked in skill.json).
3. Station keep-out of the task arm (``KeepOut.for_haller(side)``): every good take, states and commanded actions,
   replayed against the measured state with the parked arm's recorded pose per frame. Every demonstrated motion must
   pass; refusals are reported per zone and take and block eval_skill.sh until someone looks.
4. Demo envelope from the train takes' commanded actions (``KeepOut.build_envelope``), validated on the held-out
   takes (must be 0 refusals) and saved with the station zones as ``skills/<skill>/workspace.json``.
5. Placement and home-check references: frame 0 of each held-out take, head + the task arm's wrist (``layouts/``).

    python ws4/single_arm/build_skill.py box                      # newest matching dataset in the cache
    python ws4/single_arm/build_skill.py green --dataset ws4/<repo name>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS4 = HERE.parent
sys.path.insert(0, str(WS4))
sys.path.insert(0, str(HERE))

from gate import envelope_gap, point_segment  # noqa: E402, F401 (re-exported for tests)
from keepout import ZONES, KeepOut  # noqa: E402
from skills import ARMS, CACHE, SKILL_DIR, SKILLS, find_dataset  # noqa: E402

SNAPSHOTS = WS4 / ".cache" / "single_arm_diag"


def snapshot(src: Path, dst: Path) -> dict:
    """Copy a finalised dataset and check that info.json, the episode table and the data rows agree."""
    import pyarrow.parquet as pq

    shutil.copytree(src, dst, symlinks=True)
    info = json.loads((dst / "meta" / "info.json").read_text())
    eps = pq.read_table(sorted((dst / "meta" / "episodes").glob("chunk-*/file-*.parquet"))).to_pandas()
    rows = sum(pq.read_metadata(f).num_rows for f in sorted((dst / "data").glob("chunk-*/file-*.parquet")))
    if not rows == info["total_frames"] == int(eps["length"].sum()):
        shutil.rmtree(dst)
        raise SystemExit(f"{src} is not finalised (info {info['total_frames']} frames, episode table "
                         f"{int(eps['length'].sum())}, data {rows}): ask ws4-vr2 to finalise, then rerun")  # fmt: skip
    return info


def task_prompt(root: Path) -> str:
    import pyarrow.parquet as pq

    tasks = pq.read_table(root / "meta" / "tasks.parquet").to_pandas()
    names = [str(t) for t in tasks.index] if "task" not in tasks.columns else [str(t) for t in tasks["task"]]
    if len(names) != 1:
        raise SystemExit(f"expected one task string, got {names}")
    return names[0]


def split_of(root: Path, episodes: list[int]) -> dict:
    path = root / "ws4_split.json"
    if path.exists():
        s = json.loads(path.read_text())
        heldout = s["heldout"] if isinstance(s["heldout"], dict) else {"H": list(s["heldout"])}
        return {"source": f"ws4_split.json {s.get('updated', '')}".strip(), "heldout": heldout,
                "drop": list(s.get("drop", [])), "train": list(s["train"]), "provisional": False}  # fmt: skip
    review = root / "review.json"
    drop = []
    if review.exists():
        drop = sorted(int(e) for e, v in json.loads(review.read_text()).get("episodes", {}).items()
                      if v.get("status") == "reject")  # fmt: skip
    good = [e for e in episodes if e not in drop]
    return {"source": "provisional: no ws4_split.json yet, review.json rejects dropped, last 3 good takes held out",
            "heldout": {"H": good[-3:]}, "drop": drop, "train": good[:-3], "provisional": True}  # fmt: skip


def replay(ko: KeepOut, state: np.ndarray, action: np.ndarray, other: np.ndarray | None, ep: np.ndarray,
           takes: list[int]) -> dict:  # fmt: skip
    """Refusals of every recorded step (state and commanded action vs the measured state) of ``takes``."""
    zones: Counter = Counter()
    per_take: Counter = Counter()
    steps = 0
    for i in np.flatnonzero(np.isin(ep[:-1], takes) & (ep[1:] == ep[:-1])):
        if other is not None:
            ko.other_q = tuple(float(v) for v in other[i])
        for traj in (state, action):
            steps += 1
            refuse, zone = ko.refuse(traj[i + 1], state[i])
            if refuse:
                zones[zone] += 1
                per_take[int(ep[i])] += 1
    return {"steps": steps, "refusals": sum(zones.values()), "zones": dict(zones), "takes": dict(per_take)}


def write_refs_only(skill, out: Path) -> None:
    """References for a skill built with --no-refs: decode frame 0 of its held-out takes from the recorded snapshot."""
    from make_layout_refs import write_refs

    info = json.loads((out / "skill.json").read_text())
    arm = ARMS[info["arm"]]
    write_refs(info["snapshot"], info["split"]["heldout"], out / "layouts", repo_id="local/skill",
               wrist_key=arm.wrist_key, columns=arm.columns, meta={"dataset": Path(info["dataset"]).name,
                                                                    "task": info["task"]})  # fmt: skip
    info["refs"] = True
    (out / "skill.json").write_text(json.dumps(info, indent=1))
    print(f"{skill.name}: references written to {out / 'layouts'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("skill", choices=sorted(set(SKILLS) - {"pen"}))
    p.add_argument("--dataset", default=None, help="repo id in the local cache (default: the skill's, or a match)")
    p.add_argument("--root", default=None, help="dataset directory instead of the cache")
    p.add_argument("--out", default=None, help="artifact dir (default skills/<skill>)")
    p.add_argument("--arm", choices=sorted(ARMS), default=None, help="override the skill's task arm (tests)")
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="takes left out of everything (with --reason)")
    p.add_argument("--midline", type=float, default=None,
                   help="override the task arm's midline backstop: |y| in m toward the other arm (station: 0.11)")
    p.add_argument("--other-clearance", type=float, default=None,
                   help="override the other-arm capsule clearance (m, station: 0.04); orchestrator approval only")
    p.add_argument("--torso-margin", type=float, default=None,
                   help="override the torso box margin (m, station: 0.02); orchestrator approval only")
    p.add_argument("--slow-zone", type=float, default=None,
                   help="m: the runner's slow zone near the other arm for this skill (0.5 deg/tick, load freeze at 50 "
                   "%% for 0.2 s); written into skill.json runner_args")
    p.add_argument("--env-gap", type=float, default=None,
                   help="drop envelope voxels with any point closer than this (m) to the other arm's capsules, at the "
                   "min / median / max of its pose range in the demos")
    p.add_argument("--reason", default="", help="why the overrides; recorded in skill.json")
    p.add_argument("--no-refs", action="store_true", help="skip the start frames (video decoding): keep-out and "
                   "envelope only, e.g. while the laptop records (run the full build afterwards)")
    p.add_argument("--refs-only", action="store_true",
                   help="only write the held-out start frames into an already built skill (its snapshot and split)")
    args = p.parse_args()
    skill = SKILLS[args.skill]
    if args.refs_only:
        write_refs_only(skill, Path(args.out) if args.out else SKILL_DIR / skill.name)
        return
    arm = ARMS[args.arm or skill.arm]
    other_arm = ARMS["left" if arm.side == "right" else "right"]
    if args.root:
        src = Path(args.root)
    elif args.dataset:
        src = CACHE / args.dataset
    else:
        hits = find_dataset(skill)
        if not hits:
            have = sorted(p.name for p in (CACHE / "ws4").glob("*")) if (CACHE / "ws4").exists() else []
            raise SystemExit(f"no dataset for {skill.name!r} ({skill.dataset} or names with {skill.keywords}) in "
                             f"{CACHE / 'ws4'}; have {have}")  # fmt: skip
        src = hits[0]
    out = Path(args.out) if args.out else SKILL_DIR / skill.name
    snap = SNAPSHOTS / f"skill_{skill.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    info = snapshot(src, snap)
    task = task_prompt(snap)
    print(f"{skill.name}: {src} -> {snap} ({info['total_episodes']} takes, {info['total_frames']} frames), "
          f"task {task!r}, task arm robot {arm.side} (columns {arm.columns})", flush=True)  # fmt: skip

    from flux.package_tools import arm_columns, dataset_arrays

    arrays = dataset_arrays("local/skill", root=str(snap))
    ep = arrays["episode_index"]
    state, action, other = arrays["observation.state"].astype(float), arrays["action"].astype(float), None
    if state.shape[1] == 12:
        other = arm_columns(state, other_arm.columns)
        state, action = arm_columns(state, arm.columns), arm_columns(action, arm.columns)
    split = split_of(snap, sorted(set(ep.tolist())))
    if args.exclude:  # an explicit, recorded override (e.g. takes recorded with the other arm parked elsewhere)
        split["train"] = [e for e in split["train"] if e not in args.exclude]
        split["heldout"] = {k: [e for e in v if e not in args.exclude] for k, v in split["heldout"].items()}
        split["excluded"] = sorted(args.exclude)
    split["heldout"] = {k: v for k, v in split["heldout"].items() if v}  # a layout with no held-out take has no refs
    heldout = [e for takes in split["heldout"].values() for e in takes]
    good = sorted(set(split["train"]) | set(heldout))

    station = KeepOut.for_haller(arm.side)
    ws = KeepOut.for_haller(arm.side)
    if args.midline is not None:  # midline_y is stored as -|limit| for both sides (see KeepOut.depths)
        station.midline_y = ws.midline_y = -abs(args.midline)
    if args.other_clearance is not None:
        station.other_clearance = ws.other_clearance = args.other_clearance
    if args.torso_margin is not None:
        station.torso_margin = ws.torso_margin = args.torso_margin
    station_replay = replay(station, state, action, other, ep, good)
    train_mask = np.isin(ep, split["train"])
    ws.build_envelope(action[train_mask], source=f"{skill.name}: {src.name} train takes {split['train']}")
    trimmed = 0
    if args.env_gap is not None and other is not None:
        good_other = other[np.isin(ep, good)]
        poses = [good_other.min(axis=0), np.median(good_other, axis=0), good_other.max(axis=0)]
        # The gap is sampled on a 6^3 grid per voxel; a point between samples can be up to half a sub-cell diagonal
        # closer, so the threshold includes that bound and the trimmed envelope keeps the full --env-gap.
        keep = envelope_gap(ws, poses) >= args.env_gap + ws.env_voxel * np.sqrt(3) / (2 * 6)
        trimmed = int((~keep).sum())
        ws.env_cells = tuple(c for c, k in zip(ws.env_cells, keep, strict=True) if k)
        object.__setattr__(ws, "_grid_cache", None)
        print(f"envelope trimmed by {trimmed} voxels closer than {args.env_gap} m to the other arm", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    ws.save(out / "workspace.json")
    heldout_replay = replay(KeepOut.load(out / "workspace.json"), state, action, other, ep, heldout)
    print(f"station keep-out over {len(good)} good takes: {station_replay['refusals']} of {station_replay['steps']} "
          f"steps refused {station_replay['zones']} {station_replay['takes']}", flush=True)
    print(f"envelope from {int(train_mask.sum())} frames of {len(split['train'])} train takes: joints "
          f"{np.round(ws.env_joint_lo, 1)} .. {np.round(ws.env_joint_hi, 1)}, {len(ws.env_cells)} TCP voxels; "
          f"held-out {heldout_replay['refusals']} of {heldout_replay['steps']} steps refused "
          f"{heldout_replay['zones']}", flush=True)  # fmt: skip

    parked = None
    if other is not None:  # the parked arm's pose in the demos: the runner refuses to start a skill off it
        good_mask = np.isin(ep, good)
        median = np.median(other[good_mask], axis=0)
        parked = {"median": [round(float(v), 1) for v in median],
                  "min": [round(float(v), 1) for v in other[good_mask].min(axis=0)],
                  "max": [round(float(v), 1) for v in other[good_mask].max(axis=0)],
                  "max_dev_deg": round(float(np.abs(other[good_mask] - median)[:, :5].max()), 1),
                  "source": f"{other_arm.columns}_* columns over {len(good)} good takes"}  # fmt: skip
        off = np.abs(median[:5] - np.array(skill.parked[:5])) if skill.parked is not None else np.zeros(5)
        drift = "" if off.max() <= 5 else f"; differs from skills.py by {off.max():.1f} deg"
        print(f"parked arm (robot {other_arm.side}) in the demos: {parked['min']} .. {parked['max']} (median "
              f"{parked['median']}){drift}", flush=True)  # fmt: skip

    runner_args, slow = [], None
    if args.slow_zone is not None:
        runner_args = ["--slow-zone", f"{args.slow_zone:g}", "--slow-step", "0.5", "--slow-load-sustain", "0.5",
                       "--slow-load-ticks", "6"]  # fmt: skip
        if other is not None:  # how much of the demonstrated motion runs in the slow zone (measured poses)
            good_idx = np.flatnonzero(np.isin(ep, good))
            ko = KeepOut.load(out / "workspace.json")
            near = np.zeros(len(good_idx), dtype=bool)
            for k, i in enumerate(good_idx):
                ko.other_q = tuple(float(v) for v in other[i])
                near[k] = float(ko.other_arm_gap(state[i])) < args.slow_zone
            takes = sorted({int(ep[i]) for i in good_idx[near]})
            slow = {"zone_m": args.slow_zone, "tick_fraction": round(float(near.mean()), 4), "takes": len(takes),
                    "of_takes": len(good)}  # fmt: skip
            print(f"slow zone {args.slow_zone} m: {near.mean() * 100:.1f} % of the demo ticks, in {len(takes)} of "
                  f"{len(good)} takes", flush=True)  # fmt: skip
    rel = out.resolve().relative_to(WS4.parent) if out.resolve().is_relative_to(WS4.parent) else out.resolve()
    if args.no_refs:
        print("references skipped (--no-refs): rerun without it before an arm session", flush=True)
    else:
        from make_layout_refs import write_refs

        write_refs(str(snap), split["heldout"], out / "layouts", repo_id="local/skill", wrist_key=arm.wrist_key,
                   columns=arm.columns, meta={"dataset": src.name, "task": task})  # fmt: skip
    layouts = list(split["heldout"])
    record = {
        "skill": skill.name, "arm": arm.side, "dataset": str(src), "snapshot": str(snap), "task": task,
        "workspace": f"{rel}/workspace.json", "layouts": f"{rel}/layouts", "home_ref": f"{rel}/layouts",
        "eval_layouts": " ".join((layouts * 5)[:5]), "split": split, "parked_other_arm": parked,
        "validation": {"station": station_replay, "heldout_with_envelope": heldout_replay,
                       "zones": list(ZONES), "ok": station_replay["refusals"] == 0 and heldout_replay["refusals"] == 0},
        "overrides": {"exclude": sorted(args.exclude), "midline": args.midline, "env_gap": args.env_gap,
                      "env_voxels_trimmed": trimmed, "other_clearance": args.other_clearance,
                      "torso_margin": args.torso_margin,
                      "slow_zone": slow, "reason": args.reason}
        if args.exclude or args.midline is not None or args.env_gap is not None or args.other_clearance is not None
        or args.slow_zone is not None or args.torso_margin is not None else None,
        "runner_args": runner_args,
        "refs": not args.no_refs, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }  # fmt: skip
    (out / "skill.json").write_text(json.dumps(record, indent=1))
    verdict = "OK" if record["validation"]["ok"] else "REFUSALS: review the zones before any arm run"
    print(f"wrote {out}/skill.json: keep-out validation {verdict}"
          f"{' (provisional split)' if split['provisional'] else ''}")  # fmt: skip
    if arm.side == "left" and skill.name != "pen" and other is not None:  # chain skills with the lid-holding box arm
        if (SKILL_DIR / "box" / "skill.json").exists():
            from gate import write as write_gate

            write_gate(skill.name, out)
        else:
            print("no box skill yet: the box-arm gate is not written (gate.py write <skill> later)", flush=True)


if __name__ == "__main__":
    main()
