"""Start gate for the pen-arm skills of the chain: is the box arm, wherever the box skill left it holding the lid,
compatible with this skill's demonstrated motion?

The box skill's end pose varies run to run (box demos: lift -25..25, roll 45..85 deg, p5..p95), so a joint range
around one recorded hold would refuse nearly every real run (orchestrator, 18 Oct). The gate instead looks at
geometry, with the live box-arm pose:
- refuse if any voxel of the skill's TCP envelope lies inside the box arm's capsules;
- refuse if more than ``max_below`` of the skill's saved demo poses (pen arm) would be closer than the skill's
  other-arm clearance (green 4 cm, cutter 1 cm).
A loose joint sanity range stays in front of it (box-skill end poses and the skill's own recorded hold, each widened
by 5 deg). The per-tick capsule check with the live pose, and the cutter's slow zone, stay the safety guard.

    python ws4/single_arm/gate.py write green          # demo_poses.npz + "other_arm_gate" in skills/green/skill.json
    python ws4/single_arm/gate.py report 10.6 0.7 11.9 -0.3 62.7 0.9   # which pen skills a box end pose allows
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS4 = HERE.parent
sys.path.insert(0, str(WS4))
sys.path.insert(0, str(HERE))

REASON = "the box-skill end pose varies run to run; a joint range would refuse nearly every real run"
MAX_BELOW = 0.01
MAX_POSES = 3000
_POSES: dict[str, np.ndarray] = {}


def point_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances from points (N, 3) to the segment a-b."""
    ab = b - a
    t = np.clip((points - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def envelope_gap(ko, other_poses: list[np.ndarray], samples: int = 6) -> np.ndarray:
    """Per envelope voxel: the smallest distance from any point inside it (``samples``^3 grid) to the other arm's
    capsule surfaces, over ``other_poses`` (negative: inside the arm)."""
    from kinematics.collision import LINKS

    names = ["base", "j0", "j1", "j2", "j3", "j4", "tcp"]
    v = ko.env_voxel
    corners = np.asarray(ko.env_origin) + np.asarray(ko.env_cells, float) * v
    g = (np.arange(samples) + 0.5) / samples
    sub = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * v
    pts = (corners[:, None, :] + sub[None]).reshape(-1, 3)
    gap = np.full(len(pts), np.inf)
    for pose in other_poses:
        caps = ko._capsule_points(np.asarray(pose, float), ko.other_base_pos)  # noqa: SLF001
        for x, y, r in LINKS:
            gap = np.minimum(gap, point_segment(pts, caps[names.index(x)], caps[names.index(y)]) - r)
    return gap.reshape(len(corners), -1).min(axis=1)


def load_poses(path: str | Path) -> np.ndarray:
    key = str(path)
    if key not in _POSES:
        _POSES[key] = np.load(path)["q"].astype(float)
    return _POSES[key]


def check(ko, other_q, demo_poses: np.ndarray, max_below: float = MAX_BELOW, samples: int = 4) -> tuple[bool, dict]:
    """Gate verdict for the box arm at ``other_q`` (sets ``ko.other_q``); clearance = the skill's other-arm one."""
    ko.other_q = tuple(float(v) for v in other_q)
    gaps = np.asarray(ko.other_arm_gap(demo_poses), float)
    below = float((gaps < ko.other_clearance).mean())
    inside = int((envelope_gap(ko, [np.asarray(other_q, float)], samples) < 0).sum()) if ko.env_cells else 0
    detail = {"demo_min_gap": round(float(gaps.min()), 4), "demo_below": round(below, 4),
              "envelope_voxels_inside": inside, "clearance": ko.other_clearance}  # fmt: skip
    return inside == 0 and below <= max_below, detail


def box_end_poses() -> np.ndarray:
    """Box-arm pose over the last second (median of 30 frames) of each good box take."""
    from flux.package_tools import arm_columns, dataset_arrays
    from skills import ARMS, SKILLS, skill_info

    info = skill_info("box")
    a = dataset_arrays("local/box", root=info["snapshot"])
    ep = a["episode_index"]
    state = arm_columns(a["observation.state"], ARMS[SKILLS["box"].arm].columns).astype(float)
    split = info["split"]
    good = sorted(set(split["train"]) | {e for v in split["heldout"].values() for e in v})
    return np.array([np.median(state[np.flatnonzero(ep == e)[-30:]], axis=0) for e in good])


def write(name: str, skill_dir: Path | None = None) -> dict:
    """Save the skill's demo poses and record the gate in its skill.json (pen-arm skills only)."""
    from flux.package_tools import arm_columns, dataset_arrays
    from skills import ARMS, SKILL_DIR, SKILLS

    skill = SKILLS[name]
    out = Path(skill_dir) if skill_dir else SKILL_DIR / name
    info = json.loads((out / "skill.json").read_text())
    a = dataset_arrays(f"local/{name}", root=info["snapshot"])
    ep = a["episode_index"]
    split = info["split"]
    good = sorted(set(split["train"]) | {e for v in split["heldout"].values() for e in v})
    q = arm_columns(a["observation.state"], ARMS[skill.arm].columns).astype(float)[np.isin(ep, good)]
    q = q[:: max(1, len(q) // MAX_POSES)]
    np.savez_compressed(out / "demo_poses.npz", q=q)
    ends = box_end_poses()
    parked = info["parked_other_arm"]
    lo = np.minimum(ends.min(axis=0), parked["min"]) - 5.0
    hi = np.maximum(ends.max(axis=0), parked["max"]) + 5.0
    lo[5], hi[5] = 0.0, 100.0  # the gripper is not checked
    rel = (out / "demo_poses.npz").resolve()
    rel = rel.relative_to(WS4.parent) if rel.is_relative_to(WS4.parent) else rel
    info["other_arm_gate"] = {
        "joint_lo": [round(float(v), 1) for v in lo], "joint_hi": [round(float(v), 1) for v in hi],
        "demo_poses": str(rel), "n_demo_poses": len(q), "max_below": MAX_BELOW, "box_end_takes": len(ends),
        "reason": REASON,
    }  # fmt: skip
    (out / "skill.json").write_text(json.dumps(info, indent=1))
    print(f"{name}: gate written, {len(q)} demo poses, joint range {np.round(lo[:5], 1)} .. {np.round(hi[:5], 1)}")
    return info["other_arm_gate"]


def report(pose: np.ndarray) -> list[str]:
    """One line per pen-arm skill: does the gate allow the box arm at ``pose``?"""
    from keepout import KeepOut
    from skills import SKILL_DIR, SKILLS

    lines = []
    for name, skill in SKILLS.items():
        if skill.arm != "left" or name == "pen":
            continue
        path = SKILL_DIR / name / "skill.json"
        info = json.loads(path.read_text()) if path.exists() else {}
        gate = info.get("other_arm_gate")
        if not gate:
            lines.append(f"{name}: no gate yet (not built)")
            continue
        ko = KeepOut.load(WS4.parent / info["workspace"])
        in_range = bool(np.all(pose[:5] >= np.array(gate["joint_lo"][:5])) and
                        np.all(pose[:5] <= np.array(gate["joint_hi"][:5])))  # fmt: skip
        ok, d = check(ko, pose, load_poses(WS4.parent / gate["demo_poses"]), gate["max_below"])
        verdict = "ALLOWED" if ok and in_range else "REFUSED"
        lines.append(f"{name}: {verdict} (joint range {'ok' if in_range else 'OUT'}, demo min gap "
                     f"{d['demo_min_gap']:.3f} m, {d['demo_below'] * 100:.1f} % of demo poses under "
                     f"{d['clearance']} m, envelope voxels inside the box arm "
                     f"{d['envelope_voxels_inside']})")  # fmt: skip
    return lines


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "write":
        write(sys.argv[2])
    elif len(sys.argv) >= 7 and sys.argv[1] == "report":
        pose = np.array([float(v) for v in sys.argv[2:8]] + [0.0] * (8 - len(sys.argv)))
        print(f"box end pose {np.round(pose[:5], 1)}:")
        for line in report(pose):
            print(f"  {line}")
    else:
        raise SystemExit("usage: gate.py write <skill> | gate.py report q1 q2 q3 q4 q5 [q6]")
