"""Build the meshes in ws4/assets/{so101,xlerobot_040} from the upstream CAD (run once, offline).

Needs gmsh, trimesh and fast-simplification, which the sim itself does not need. Run it in a throwaway venv:

    uv venv ws4/.cache/sim/cadenv --python 3.12
    uv pip install --python ws4/.cache/sim/cadenv/bin/python gmsh trimesh fast-simplification numpy scipy shapely
    ws4/.cache/sim/cadenv/bin/python -I ws4/sim/tools/prepare_assets.py \
        --xlerobot ws4/.cache/sim/xlerobot --so-arm100 ws4/.cache/sim/so_arm100

Upstream (both Apache-2.0):
- XLeRobot @ b017b5e6354bd9f61f4247a920c72622ca0aade0 (hardware/step/...)
- SO-ARM100 @ a758567c3978dfeefe282ede0500085a48fe8f78 (Simulation/SO101/assets)

The XLeRobot STEP assemblies come in their own CAD frames. This script moves them into the station
world frame of station_geometry.yaml (origin on the table top below the centre of the arm-base box,
+x forward, +y left, +z up), so the MJCF can place them at the identity pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent.parent / "assets"


def step_solids(path: Path, size_mm: float) -> list[np.ndarray]:
    """Mesh every solid of a STEP file; returns one (n, 3, 3) triangle array per solid, in metres."""
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setString("Geometry.OCCTargetUnit", "MM")
    gmsh.model.occ.importShapes(str(path))
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", size_mm * 0.3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm * 4)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    pos = np.zeros((int(tags.max()) + 1, 3))
    pos[tags.astype(int)] = coords.reshape(-1, 3) / 1000.0
    solids = []
    for _, vt in gmsh.model.getEntities(3):
        tris = []
        for _, st in gmsh.model.getBoundary([(3, vt)], oriented=False):
            types, _, nodes = gmsh.model.mesh.getElements(2, abs(st))
            tris += [n.reshape(-1, 3).astype(int) for t, n in zip(types, nodes, strict=True) if t == 2]
        solids.append(pos[np.concatenate(tris)] if tris else np.zeros((0, 3, 3)))
    gmsh.finalize()
    return solids


def to_trimesh(tris: np.ndarray):
    import trimesh

    m = trimesh.Trimesh(vertices=tris.reshape(-1, 3), faces=np.arange(len(tris) * 3).reshape(-1, 3), process=True)
    m.merge_vertices()
    m.fix_normals()
    return m


def save(mesh, path: Path, max_faces: int) -> None:
    if len(mesh.faces) > max_faces:
        mesh = mesh.simplify_quadric_decimation(face_count=max_faces)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    print(f"  {path.relative_to(ASSETS)}: {len(mesh.faces)} faces, bounds {np.round(mesh.bounds, 4).tolist()}")


def bbox(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = tris.reshape(-1, 3)
    return v.min(0), v.max(0)


def build_xlerobot(xle: Path) -> dict:
    out = ASSETS / "xlerobot_040"
    info: dict = {}

    # Arm-base assembly (0.4.0). Solids: 10/12 arm-base boxes, 11/13 turntables, 14 centre shell, 6 refined neck.
    ab = step_solids(xle / "hardware/step/XLeRobot_040/XLeRobot040_armbase.step", 4.0)
    lo10, hi10 = bbox(ab[9])
    lo12, hi12 = bbox(ab[11])
    assert abs((hi10 - lo10)[2] - 0.0824) < 0.002 and abs((hi12 - lo12)[2] - 0.0824) < 0.002, "unexpected solid order"
    box_lo, box_hi = np.minimum(lo10, lo12), np.maximum(hi10, hi12)
    origin = np.array([(box_lo[0] + box_hi[0]) / 2, (box_lo[1] + box_hi[1]) / 2, box_lo[2]])
    info["armbase_origin_in_cad"] = origin.round(5).tolist()
    for idx, name in [(9, "armbase_left"), (11, "armbase_right"), (10, "turntable_left"), (12, "turntable_right"),
                      (13, "centre_shell")]:
        tris = ab[idx] - origin
        lo, hi = bbox(tris)
        info[name] = {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()}
        save(to_trimesh(tris), out / f"{name}.stl", 6000)
    neck_new_lo, neck_new_hi = bbox(ab[5] - origin)
    info["neck_refined"] = {"min": neck_new_lo.round(4).tolist(), "max": neck_new_hi.round(4).tolist()}

    # RGBD gimbal assembly: octagon-hole neck_old (what Oscar printed) + pan/tilt head with mock servos.
    gb = step_solids(xle / "hardware/step/RGBD_Gimbal/Assembly_RGBD_Gimbal.step", 3.0)
    nlo, nhi = bbox(gb[0])
    # Put neck_old on the centre shell, on the same footprint centre as the refined 0.4.0 neck.
    target = np.array([(neck_new_lo[0] + neck_new_hi[0]) / 2, (neck_new_lo[1] + neck_new_hi[1]) / 2, neck_new_lo[2]])
    shift = target - np.array([(nlo[0] + nhi[0]) / 2, (nlo[1] + nhi[1]) / 2, nlo[2]])
    info["gimbal_shift"] = shift.round(5).tolist()
    # Tilt servo horn (solid 8) centre = tilt pivot.
    hlo, hhi = bbox(gb[7] + shift)
    pivot = (hlo + hhi) / 2
    info["tilt_pivot"] = pivot.round(4).tolist()
    static = {"neck_old": [0], "head_pan_servo": [1, 2, 3, 4, 5], "head_base": [6], "head_yaw": [12, 13],
              "head_tilt_servo": [7, 8, 9, 10, 11]}
    for name, ids in static.items():
        tris = np.concatenate([gb[i] for i in ids]) + shift
        lo, hi = bbox(tris)
        info[name] = {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()}
        save(to_trimesh(tris), out / f"{name}.stl", 6000)
    # Pitch holder, expressed relative to the tilt pivot (the MJCF rotates it by the fixed tilt angle).
    tris = np.concatenate([gb[i] for i in (14, 15, 16, 17, 18)]) + shift - pivot
    lo, hi = bbox(tris)
    info["head_pitch_holder_rel_pivot"] = {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()}
    save(to_trimesh(tris), out / "head_pitch_holder.stl", 6000)

    return info


# 24 proper rotations that map coordinate axes onto coordinate axes (ICP starting orientations).
def _axis_rotations() -> list[np.ndarray]:
    import itertools

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs, strict=True)):
                R[i, p] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def _register(src, dst) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigid fit of mesh `src` onto mesh `dst` (shared CAD features): ICP from 24 axis-aligned starts, then the
    rotation is snapped to the nearest axis-aligned one and the translation re-fitted on the closest 70 %.
    Returns (R, t, median residual in metres)."""
    import trimesh
    from scipy.spatial import cKDTree
    from trimesh import registration

    sp = trimesh.sample.sample_surface(src, 3000, seed=0)[0]
    dp = trimesh.sample.sample_surface(dst, 40000, seed=0)[0]
    tree = cKDTree(dp)
    best = None
    for R in _axis_rotations():
        T0 = np.eye(4)
        T0[:3, :3] = R
        T0[:3, 3] = dp.mean(0) - (sp @ R.T).mean(0)
        T, _, _ = registration.icp(sp, dp, initial=T0, threshold=1e-9, max_iterations=80, reflection=False, scale=False)
        d, _ = tree.query(trimesh.transform_points(sp, T))
        if best is None or np.median(d) < best[0]:
            best = (np.median(d), T)
    R = np.round(best[1][:3, :3])
    t = best[1][:3, 3].copy()
    for _ in range(50):
        p = sp @ R.T + t
        d, idx = tree.query(p)
        keep = d < np.percentile(d, 70)
        t += (dp[idx[keep]] - p[keep]).mean(0)
    d, _ = tree.query(sp @ R.T + t)
    return R, t, float(np.median(d))


def build_soft_gripper(xle: Path, so: Path) -> dict:
    """XLeRobot soft gripper (hardware/SO101_soft_fin.stl, a print plate with 3 parts): modified fixed jaw
    (gripper body), modified moving jaw, and the TPU fin used on both jaws.

    The two jaws are registered onto the stock SO-101 meshes they replace, so they drop into the same MJCF
    geom poses as wrist_roll_follower_so101_v1 / moving_jaw_so101_v1. The fin is written in its own frame:
    straight (contact) edge on x = 0 with the slatted body toward -x, width centred on y = 0, base end at
    z = 0 and tip toward -z.
    """
    import trimesh

    plate = trimesh.load(xle / "hardware/SO101_soft_fin.stl")  # millimetres
    plate.apply_scale(0.001)
    parts = sorted(plate.split(only_watertight=False), key=lambda m: -m.area)
    body, fin, moving = parts[0], parts[1], parts[2]
    stock = so / "Simulation/SO101/assets"
    info: dict = {}
    for name, part, target in [("soft_fixed_jaw", body, "wrist_roll_follower_so101_v1"),
                               ("soft_moving_jaw", moving, "moving_jaw_so101_v1")]:
        R, t, res = _register(part, trimesh.load(stock / f"{target}.stl"))
        m = part.copy()
        m.vertices = m.vertices @ R.T + t
        info[name] = {"registered_onto": target, "R": R.tolist(), "t": t.round(5).tolist(),
                      "median_residual_mm": round(res * 1000, 3)}
        save(m, ASSETS / "so101" / f"{name}.stl", 6000)

    # Fin: plate frame has length along x (base at max x), slat profile in xy with the straight edge at max y
    # (ignoring the small tip hook), width along z (flat side on z = 0, 24 mm at the base, 12 mm at the tip).
    v = fin.vertices
    lo, hi = v.min(0), v.max(0)
    base = v[v[:, 0] > hi[0] - 0.006]
    y_edge = base[:, 1].max()
    local = np.column_stack([v[:, 1] - y_edge, v[:, 2] - (lo[2] + hi[2]) / 2, v[:, 0] - hi[0]])
    f = fin.copy()
    f.vertices = local
    save(f, ASSETS / "so101" / "soft_fin.stl", 6000)
    info["soft_fin"] = {"min": f.bounds[0].round(4).tolist(), "max": f.bounds[1].round(4).tolist()}
    return info


def build_so101(so: Path) -> None:
    import trimesh

    src = so / "Simulation/SO101/assets"
    names = ["base_motor_holder_so101_v1", "base_so101_v2", "sts3215_03a_v1", "waveshare_mounting_plate_so101_v2",
             "motor_holder_so101_base_v1", "rotation_pitch_so101_v1", "upper_arm_so101_v1", "under_arm_so101_v1",
             "motor_holder_so101_wrist_v1", "sts3215_03a_no_horn_v1", "wrist_roll_pitch_so101_v2"]
    for n in names:
        save(trimesh.load(src / f"{n}.stl"), ASSETS / "so101" / f"{n}.stl", 5000)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlerobot", type=Path, required=True)
    ap.add_argument("--so-arm100", type=Path, required=True)
    a = ap.parse_args()
    build_so101(a.so_arm100)
    info = build_xlerobot(a.xlerobot)
    info.update(build_soft_gripper(a.xlerobot, a.so_arm100))
    (ASSETS / "xlerobot_040" / "cad_frames.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
