"""The EE50 -> XLeRobot gate (Simone's plan, step 2): can EE50 end-effector actions be executed on 5-DoF arms?

Run from the repo root (CPU only, about 5 minutes on 16 cores; ``--quick`` for a 1-minute smoke run):

    python -I ws4/kinematics/validate.py [--quick] [--workers 16]

Needs the small Hub samples from ``python -I -m kinematics.hub_data`` (run inside ws4/) in ws4/.cache/kin/hub.
Writes ``ws4/kinematics/results.json`` and plots to ``ws4/kinematics/plots/``.

Sections (see ws4/research/ee50_ik.md for the write-up):
  a. real robot joint trajectories: q -> FK -> EE50 -> decode -> IK -> q', errors, failures, jumps
  b. human handheld 6-DoF motion (UMI) and synthetic 6-DoF motion re-anchored on robot poses: how much of it
     the 5-DoF arm can execute, plus table-workspace coverage
  c. policy-like noise on FK-derived targets: how much off-manifold error is tolerable
  d. capsule collisions (self, inter-arm, torso, mast, table) on real bimanual joint data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "kinematics"

from kinematics import collision, ee50, hub_data  # noqa: E402
from kinematics.ik import DLSIK, AnalyticIK  # noqa: E402
from kinematics.se3 import make_pose, matrix_to_rotvec, rotvec_to_matrix  # noqa: E402
from kinematics.so101_kinematics import Station  # noqa: E402

HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
FPS = 30.0
CHUNK = 32  # EE50 chunk length used throughout (FLUX SO-101 executes 32 of 42 at 30 Hz)
JUMP = 0.3  # rad per 30 Hz frame (9 rad/s, ~3x an STS3215 at full speed): a branch flip, not motion

THRESHOLDS = {
    "codec_round_trip_m": 1e-6,
    "onmanifold_median_pos_mm": 0.1,
    "onmanifold_p99_pos_mm": 1.0,
    "onmanifold_success_rate": 0.995,  # pos < 5 mm and rot < 1 deg, both after joint limits
    "onmanifold_jump_rate": 0.001,  # frames with |dq| > JUMP that the source does not have
    "policy_noise_exec_vs_clean_ratio": 1.0,  # executed error vs the clean pose must not exceed the noise
    "policy_noise_jump_rate": 0.005,
    "realtime_ms_per_arm": 5.0,  # 30 Hz, two arms, leaves > 20 ms of the 33 ms tick
}


# ---------------------------------------------------------------------------------------------------------
# helpers


def _station() -> Station:
    return Station.load()


def track(solver, T_seq: np.ndarray, q0: np.ndarray | None) -> dict[str, np.ndarray]:
    keys = ("q", "pos", "rot", "app", "proj", "clamped", "reachable", "degenerate", "t", "cpu")
    q, out = q0, {k: [] for k in keys}
    for T in T_seq:
        t0, c0 = time.perf_counter(), time.thread_time()
        r = solver.solve(T, q)
        out["t"].append(time.perf_counter() - t0)
        out["cpu"].append(time.thread_time() - c0)  # CPU time: robust to other jobs sharing the machine
        q = r.q
        for k, v in (("q", r.q), ("pos", r.pos_err), ("rot", r.rot_err), ("app", r.approach_err),
                     ("proj", r.proj_rot_err), ("clamped", r.clamped), ("reachable", r.reachable),
                     ("degenerate", r.degenerate)):
            out[k].append(v)
    return {k: np.array(v) for k, v in out.items()}


def smooth_noise(rng, n: int, dim: int, sigma: float, width: float = 5.0) -> np.ndarray:
    """Temporally correlated Gaussian noise with per-dim std ``sigma`` (a chunked policy's errors are smooth)."""
    if sigma == 0:
        return np.zeros((n, dim))
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / width) ** 2)
    k /= np.sqrt(np.sum(k**2))
    w = rng.normal(size=(n + len(k) - 1, dim))
    out = np.stack([np.convolve(w[:, d], k, mode="valid") for d in range(dim)], axis=1)
    return out * sigma


def interp_poses(T: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a pose sequence by ``factor`` (linear position, geodesic rotation)."""
    out = []
    for a, b in zip(T[:-1], T[1:], strict=True):
        w = matrix_to_rotvec(a[:3, :3].T @ b[:3, :3])
        for s in np.arange(factor) / factor:
            out.append(make_pose(a[:3, :3] @ rotvec_to_matrix(w * s), a[:3, 3] * (1 - s) + b[:3, 3] * s))
    out.append(T[-1])
    return np.array(out)


def stats(x, scale=1.0) -> dict:
    x = np.asarray(x, dtype=float) * scale
    if x.size == 0:
        return {}
    return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
            "max": float(np.max(x)), "mean": float(np.mean(x))}


# ---------------------------------------------------------------------------------------------------------
# (a) real joint trajectories


def _job_a(args):
    repo, ep_index, side, q_src, grip = args
    st = _station()
    arm = st.arm
    lim = arm.limits
    out_of_limits = float(np.mean(np.any((q_src < lim[:, 0]) | (q_src > lim[:, 1]), axis=1)))
    q = arm.clip(q_src)
    T = arm.fk(q)
    # EE50 codec on the way: encode chunks relative to an anchor, decode back to absolute targets
    layout = ee50.EE50Layout()
    codec_err = 0.0
    T_dec = np.empty_like(T)
    for s in range(0, len(T), CHUNK):
        idx = slice(s, min(s + CHUNK, len(T)))
        x = ee50.encode({side: T[idx]}, {side: grip[idx]}, {side: T[s]}, layout)
        x32 = x.astype(np.float32).astype(np.float64)  # the model side works in float32 at best
        P, _ = ee50.decode(x32, {side: T[s]}, layout)
        T_dec[idx] = P[side]
        codec_err = max(codec_err, float(np.max(np.linalg.norm(P[side][:, :3, 3] - T[idx, :3, 3], axis=1))))
    res = {}
    for name, solver in (("analytic", AnalyticIK(arm)), ("dls", DLSIK(arm))):
        tr = track(solver, T_dec, None)
        dq_ik = np.abs(np.diff(tr["q"], axis=0)).max(axis=1)
        dq_src = np.abs(np.diff(q, axis=0)).max(axis=1)
        res[name] = {
            "pos": tr["pos"], "rot": tr["rot"], "t": tr["t"],
            "fail": (tr["pos"] > 5e-3) | (tr["rot"] > np.deg2rad(1.0)),
            "branch": np.abs(tr["q"] - q).max(axis=1) > 1e-2,
            "jump": (dq_ik > JUMP) & (dq_src < JUMP / 3),
            "clamped": tr["clamped"],
        }
    return {"repo": repo, "episode": ep_index, "side": side, "n": len(T), "out_of_limits": out_of_limits,
            "codec_err": codec_err, "res": res}


def section_a(pool, max_frames: int) -> dict:
    jobs = []
    for repo in hub_data.DATASETS:
        try:
            eps = hub_data.load_episodes(repo)
        except FileNotFoundError:
            continue
        budget = max_frames
        for ep in eps:
            for side, q in ep.q.items():
                if np.ptp(q, axis=0).max() < 1e-3 or budget <= 0:
                    continue  # arm never moves (e.g. single-arm recordings on a two-arm robot)
                n = min(len(q), budget)
                jobs.append((repo, ep.episode, side, q[:n], ep.gripper[side][:n]))
                budget -= n
    results = list(pool.map(_job_a, jobs))
    summary = {"datasets": {}, "solvers": {}}
    for repo in hub_data.DATASETS:
        rs = [r for r in results if r["repo"] == repo]
        if rs:
            n = [r["n"] for r in rs]
            summary["datasets"][repo] = {
                "episodes": len({r["episode"] for r in rs}),
                "frames": int(sum(n)),
                "frac_frames_outside_urdf_limits": float(np.average([r["out_of_limits"] for r in rs], weights=n)),
            }
    summary["codec_max_err_m"] = float(max(r["codec_err"] for r in results))
    for name in ("analytic", "dls"):
        keys = ("pos", "rot", "t", "fail", "branch", "jump", "clamped")
        cat = {k: np.concatenate([r["res"][name][k] for r in results]) for k in keys}
        summary["solvers"][name] = {
            "frames": int(len(cat["pos"])),
            "pos_mm": stats(cat["pos"], 1e3),
            "rot_deg": stats(np.degrees(cat["rot"])),
            "success_rate": float(1 - cat["fail"].mean()),
            "other_branch_rate": float(cat["branch"].mean()),
            "jump_rate": float(cat["jump"].mean()),
            "clamped_rate": float(cat["clamped"].mean()),
            "ms_per_solve": stats(cat["t"], 1e3),
        }
        per_ds = {}
        for repo in summary["datasets"]:
            rs = [r for r in results if r["repo"] == repo]
            pos = np.concatenate([r["res"][name]["pos"] for r in rs])
            fail = np.concatenate([r["res"][name]["fail"] for r in rs])
            per_ds[repo] = {"pos_mm_p99": float(np.percentile(pos * 1e3, 99)), "success_rate": float(1 - fail.mean())}
        summary["solvers"][name]["per_dataset"] = per_ds
    summary["_raw_pos"] = np.concatenate([r["res"]["analytic"]["pos"] for r in results])
    summary["_raw_t"] = {n: np.concatenate([r["res"][n]["t"] for r in results]) for n in ("analytic", "dls")}
    return summary


# ---------------------------------------------------------------------------------------------------------
# (b) human / synthetic 6-DoF motion


def _robot_anchor_configs(n: int, seed: int) -> np.ndarray:
    """Realistic arm configurations: frames of the real datasets with the TCP above the table workspace."""
    st = _station()
    qs = []
    for repo in hub_data.DATASETS:
        try:
            for ep in hub_data.load_episodes(repo):
                for q in ep.q.values():
                    if np.ptp(q, axis=0).max() > 1e-3:
                        qs.append(st.arm.clip(q[::15]))
        except FileNotFoundError:
            pass
    q = np.concatenate(qs)
    T = st.fk("left", q)
    z, x = T[:, 2, 3], T[:, 0, 3]
    keep = (z > st.table_z + 0.02) & (z < st.table_z + 0.30) & (x > 0.05)
    q = q[keep]
    return q[np.random.default_rng(seed).choice(len(q), size=min(n, len(q)), replace=False)]


def _umi_chunks(max_chunks: int, seed: int) -> list[np.ndarray]:
    """UMI episodes (10 Hz) -> relative pose chunks of CHUNK frames at 30 Hz, in the anchor (gripper) frame."""
    eps = hub_data.load_umi(max_episodes=400)
    chunks = []
    for e in eps:
        p, rv = e["pos"], e["rotvec"]
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        if np.abs(p).max() > 2.0 or step.max() > 0.08:  # SLAM jumps: >0.8 m/s between 10 Hz frames
            continue
        T = make_pose(rotvec_to_matrix(rv), p)
        for s in range(0, len(T) - 12, 11):
            Tc = interp_poses(T[s : s + 12], 3)[:CHUNK]
            chunks.append(np.linalg.inv(Tc[0]) @ Tc)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chunks), size=min(max_chunks, len(chunks)), replace=False)
    return [chunks[i] for i in idx]


def _synthetic_chunks(n: int, seed: int, trans_sigma: float = 0.03, rot_sigma_deg: float = 15.0) -> list[np.ndarray]:
    """Smooth isotropic 6-DoF chunks: what an unconstrained handheld tool can do in ~1 s."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, CHUNK)[:, None]
    out = []
    for _ in range(n):
        v, w = rng.normal(0, trans_sigma, 3), rng.normal(0, np.deg2rad(rot_sigma_deg), 3)
        a, b = rng.normal(0, trans_sigma / 2, 3), rng.normal(0, np.deg2rad(rot_sigma_deg) / 2, 3)
        p = t * v + t**2 * a
        r = t * w + t**2 * b
        out.append(make_pose(rotvec_to_matrix(r), p))
    return out


def _job_b(args):
    kind, q_anchor, rel = args
    st = _station()
    arm = st.arm
    T_anchor = arm.fk_single(q_anchor)
    targets = T_anchor @ rel
    tr = track(AnalyticIK(arm), targets, q_anchor)
    dq = np.abs(np.diff(np.vstack([q_anchor, tr["q"]]), axis=0)).max(axis=1)
    return {"kind": kind, "pos": tr["pos"], "rot": tr["rot"], "app": tr["app"], "proj": tr["proj"],
            "clamped": tr["clamped"], "jump": dq > JUMP,
            "rel_rot_deg": float(np.degrees(np.linalg.norm(matrix_to_rotvec(rel[-1][:3, :3])))),
            "rel_trans_m": float(np.linalg.norm(rel[-1][:3, 3]))}


def section_b(pool, n_chunks: int) -> dict:
    anchors = _robot_anchor_configs(n_chunks, 0)
    jobs = []
    try:
        umi = _umi_chunks(n_chunks, 1)
        jobs += [("umi", anchors[i % len(anchors)], c) for i, c in enumerate(umi)]
    except FileNotFoundError:
        umi = []
    syn = _synthetic_chunks(n_chunks, 2)
    jobs += [("synthetic", anchors[i % len(anchors)], c) for i, c in enumerate(syn)]
    results = list(pool.map(_job_b, jobs))
    out = {}
    for kind in ("umi", "synthetic"):
        rs = [r for r in results if r["kind"] == kind]
        if not rs:
            continue
        pos = np.concatenate([r["pos"] for r in rs])
        rot = np.concatenate([r["rot"] for r in rs])
        proj = np.concatenate([r["proj"] for r in rs])
        clamped = np.concatenate([r["clamped"] for r in rs])
        chunk_pos_ok = np.array([r["pos"].max() < 5e-3 for r in rs])
        chunk_ok = np.array([(r["pos"].max() < 5e-3) and (np.degrees(r["rot"].max()) < 10) for r in rs])
        out[kind] = {
            "chunks": len(rs),
            "chunk_motion": {"trans_m": stats([r["rel_trans_m"] for r in rs]),
                             "rot_deg": stats([r["rel_rot_deg"] for r in rs])},
            "pos_mm": stats(pos, 1e3),
            "rot_deg_vs_requested": stats(np.degrees(rot)),
            "irreducible_rot_deg": stats(np.degrees(proj)),
            "frac_chunks_position_feasible": float(chunk_pos_ok.mean()),
            "frac_chunks_feasible_pos5mm_rot10deg": float(chunk_ok.mean()),
            "frac_frames_joint_limit": float(clamped.mean()),
            "jump_rate": float(np.concatenate([r["jump"] for r in rs]).mean()),
        }
        out[kind]["_raw_proj"] = proj
    return out


def _reach_point(args):
    """Can each arm put its TCP at ``p`` with this approach (some roll about the approach axis allowed)?"""
    p, approach = args
    st = _station()
    solver = AnalyticIK(st.arm)
    res = {}
    for side in ("left", "right"):
        R = _approach_rotation(approach, p, st.T_station_base[side])
        ok = False
        for roll in np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]):
            c, s_ = np.cos(roll), np.sin(roll)
            Rr = R @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            r = solver.solve(st.to_base(side, make_pose(Rr, p)), None)
            if approach == "any":
                ok = r.pos_err < 2e-3 and not r.clamped
                break
            if r.pos_err < 2e-3 and r.rot_err < np.deg2rad(2.0) and not r.clamped:
                ok = True
                break
        res[side] = bool(ok)
    return res


def _approach_rotation(approach: str, p: np.ndarray, Tb: np.ndarray) -> np.ndarray:
    """Tool rotation (z = approach) for a grasp at ``p``: straight down, or 45 deg down along the radial line."""
    if approach in ("top_down", "any"):
        return np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    radial = p[:2] - Tb[:2, 3]
    radial = radial / max(np.linalg.norm(radial), 1e-9)
    z = np.array([radial[0], radial[1], -1.0]) / np.sqrt(2)
    x = np.array([-radial[1], radial[0], 0.0])
    return np.stack([x, np.cross(z, x), z], axis=1)


def workspace(pool, res: float) -> dict:
    st = _station()
    xs = np.arange(-0.05, 0.50 + 1e-9, res)
    ys = np.arange(-0.50, 0.50 + 1e-9, res)
    out = {"x": xs.tolist(), "y": ys.tolist(), "maps": {}}
    for approach, z in (("top_down", 0.03), ("angled45", 0.05), ("any", 0.03)):
        pts = [(np.array([x, y, st.table_z + z]), approach) for x in xs for y in ys]
        rs = list(pool.map(_reach_point, pts, chunksize=16))
        grid = {s: np.array([r[s] for r in rs]).reshape(len(xs), len(ys)) for s in ("left", "right")}
        # table points under the torso box are inside the robot: never count them
        c, h = np.asarray(st.torso["center"]), np.asarray(st.torso["half_extents"])
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        body = (np.abs(X - c[0]) <= h[0]) & (np.abs(Y - c[1]) <= h[1])
        grid = {s: g & ~body for s, g in grid.items()}
        out["torso_mask"] = body.tolist()
        out["maps"][f"{approach}_z{int(z * 100)}cm"] = {s: g.tolist() for s, g in grid.items()}
        both = grid["left"] & grid["right"]
        out[f"area_m2_{approach}"] = {s: float(g.sum() * res * res) for s, g in grid.items()}
        out[f"area_m2_{approach}"]["both"] = float(both.sum() * res * res)
    return out


# ---------------------------------------------------------------------------------------------------------
# (c) policy-like noise on FK-derived targets

NOISE_LEVELS = ((0.0, 0.0), (0.002, 2.0), (0.005, 5.0), (0.010, 10.0), (0.005, 20.0), (0.020, 20.0))


def _job_c(args):
    q_src, sigma_p, sigma_r_deg, seed = args
    st = _station()
    arm = st.arm
    q = arm.clip(q_src)
    T_clean = arm.fk(q)
    rng = np.random.default_rng(seed)
    n_p = smooth_noise(rng, len(q), 3, sigma_p)
    n_r = smooth_noise(rng, len(q), 3, np.deg2rad(sigma_r_deg))
    T_noisy = T_clean.copy()
    T_noisy[:, :3, 3] += n_p
    T_noisy[:, :3, :3] = rotvec_to_matrix(n_r) @ T_clean[:, :3, :3]
    tr = track(AnalyticIK(arm), T_noisy, q[0])
    T_exec = arm.fk(tr["q"])
    def rot_err(A, B):
        return np.degrees(np.linalg.norm(matrix_to_rotvec(np.swapaxes(A[:, :3, :3], 1, 2) @ B[:, :3, :3]), axis=1))
    dq = np.abs(np.diff(tr["q"], axis=0)).max(axis=1)
    dq_src = np.abs(np.diff(q, axis=0)).max(axis=1)
    lim = arm.limits
    parked = np.any((q <= lim[:, 0] + 1e-6) | (q >= lim[:, 1] - 1e-6), axis=1)  # resting on a joint stop
    acc = np.linalg.norm(np.diff(tr["q"], 2, axis=0), axis=1)
    acc_src = np.linalg.norm(np.diff(q, 2, axis=0), axis=1)
    return {
        "noise_pos_mm": np.linalg.norm(n_p, axis=1) * 1e3, "noise_rot_deg": np.degrees(np.linalg.norm(n_r, axis=1)),
        "exec_vs_target_pos_mm": tr["pos"] * 1e3, "exec_vs_target_rot_deg": np.degrees(tr["rot"]),
        "exec_vs_clean_pos_mm": np.linalg.norm(T_exec[:, :3, 3] - T_clean[:, :3, 3], axis=1) * 1e3,
        "exec_vs_clean_rot_deg": rot_err(T_exec, T_clean),
        "jump": (dq > JUMP) & (dq_src < JUMP / 3),
        "jump_active": (dq > JUMP) & (dq_src < JUMP / 3) & ~parked[1:],
        "pos_fail": tr["pos"] > 5e-3,
        "pos_fail_active": (tr["pos"] > 5e-3) & ~parked,
        "parked": parked,
        "joint_acc_ratio": float(np.sqrt(np.mean(acc**2)) / max(np.sqrt(np.mean(acc_src**2)), 1e-9)),
    }


def section_c(pool, max_frames: int) -> dict:
    trajs = []
    repos = ("lerobot/svla_so101_pickplace", "Zekai-Chen/bi_so101_fold_towel_4cam", "Odog16/xlerobot_generalist_v2")
    for repo in repos:
        try:
            budget = max_frames
            for ep in hub_data.load_episodes(repo):
                for _side, q in ep.q.items():
                    if np.ptp(q, axis=0).max() > 1e-3 and budget > 0:
                        trajs.append(q[: min(len(q), budget)])
                        budget -= len(trajs[-1])
        except FileNotFoundError:
            pass
    out = {}
    for sp, sr in NOISE_LEVELS:
        rs = list(pool.map(_job_c, [(q, sp, sr, i) for i, q in enumerate(trajs)]))
        cat = {k: np.concatenate([r[k] for r in rs]) for k in rs[0] if k != "joint_acc_ratio"}
        key = f"pos{sp * 1e3:.0f}mm_rot{sr:.0f}deg"
        out[key] = {
            "noise_pos_mm": stats(cat["noise_pos_mm"]), "noise_rot_deg": stats(cat["noise_rot_deg"]),
            "exec_vs_target_pos_mm": stats(cat["exec_vs_target_pos_mm"]),
            "exec_vs_target_rot_deg": stats(cat["exec_vs_target_rot_deg"]),
            "exec_vs_clean_pos_mm": stats(cat["exec_vs_clean_pos_mm"]),
            "exec_vs_clean_rot_deg": stats(cat["exec_vs_clean_rot_deg"]),
            "jump_rate": float(cat["jump"].mean()),
            "jump_rate_excl_parked": float(cat["jump_active"].mean()),
            "pos_fail_rate": float(cat["pos_fail"].mean()),
            "pos_fail_rate_excl_parked": float(cat["pos_fail_active"].mean()),
            "frac_frames_parked": float(cat["parked"].mean()),
            "joint_acc_ratio_median": float(np.median([r["joint_acc_ratio"] for r in rs])),
            "frames": int(len(cat["jump"])),
        }
    return out


# ---------------------------------------------------------------------------------------------------------
# (d) collisions on real bimanual joint data placed on our station


def section_d(max_frames: int) -> dict:
    """Mesh collisions on the ws4-sim MuJoCo model (convex hulls of the SO-101 collision meshes, the fin-ray
    fingers, the torso and the table); falls back to capsules when MuJoCo or the model is missing."""
    st = _station()
    try:
        from kinematics.mj_collision import MjCollision

        mj = MjCollision()
        backend = "mujoco ws4/sim/xlerobot_station.xml (penetration > 2 mm)"
    except Exception as exc:  # no mujoco or no sim model: capsules
        mj = None
        backend = f"capsules ({exc})"
    out = {"backend": backend}
    for repo in ("Zekai-Chen/bi_so101_fold_towel_4cam", "saivishwak/xlerobot-vr-teleop-medicine-bowl",
                 "Odog16/xlerobot_generalist_v2"):
        try:
            eps = hub_data.load_episodes(repo)
        except FileNotFoundError:
            continue
        acc = {}
        n = 0
        for ep in eps:
            if n >= max_frames:
                break
            q = {s: st.arm.clip(v[: max_frames - n]) for s, v in ep.q.items()}
            g = {s: v[: max_frames - n] for s, v in ep.gripper.items()}
            n += len(next(iter(q.values())))
            if mj is not None:
                res = {k: -v for k, v in mj.check(q, g).items()}  # penetration depth -> signed clearance
            else:
                res = collision.check(st, q)
            for k, v in res.items():
                acc.setdefault(k, []).append(v)
        res = {"frames": n}
        for k, v in acc.items():
            v = np.concatenate(v)
            res[k] = {"frac_contact": float(np.mean(v < 0)), "worst_mm": float(v.min() * 1e3)}
        out[repo] = res
    return out


def bench(n: int = 1000) -> dict:
    """Single-process solve times on a real trajectory (what one 30 Hz control loop would see)."""
    st = _station()
    eps = hub_data.load_episodes("lerobot/svla_so101_pickplace")
    q = st.arm.clip(np.concatenate([e.q["right"] for e in eps])[:n])
    T = st.arm.fk(q)
    out = {}
    for name, solver in (("analytic", AnalyticIK(st.arm)), ("dls", DLSIK(st.arm))):
        tr = track(solver, T, None)
        t, c = tr["t"][1:] * 1e3, tr["cpu"][1:] * 1e3
        out[name] = {"cpu_median_ms": float(np.median(c)), "cpu_p99_ms": float(np.percentile(c, 99)),
                     "cpu_max_ms": float(c.max()), "both_arms_cpu_p99_ms": float(2 * np.percentile(c, 99)),
                     "wall_median_ms": float(np.median(t)), "wall_p99_ms": float(np.percentile(t, 99)),
                     "load_avg_1min": float(os.getloadavg()[0]), "_raw_ms": c}
    return out


# ---------------------------------------------------------------------------------------------------------
# plots


def plots(res: dict, raw: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(exist_ok=True)
    files = []

    # (a) on-manifold error CDF and solve time
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    pos = np.sort(raw["a_pos"] * 1e3)
    ax[0].plot(np.maximum(pos, 1e-10), np.linspace(0, 1, len(pos)))
    ax[0].set_xscale("log")
    ax[0].axvline(THRESHOLDS["onmanifold_p99_pos_mm"], ls="--", c="gray")
    ax[0].set_xlabel("position error after FK -> EE50 -> IK (mm)")
    ax[0].set_ylabel("fraction of frames")
    ax[0].set_title("(a) real joint trajectories, round trip")
    for name, t in raw["bench_ms"].items():
        ax[1].hist(t, bins=np.logspace(-1, 2, 60), histtype="step", label=name)
    ax[1].set_xscale("log")
    ax[1].axvline(THRESHOLDS["realtime_ms_per_arm"], ls="--", c="gray")
    ax[1].set_xlabel("CPU time per solve, one arm (ms, real trajectory)")
    ax[1].legend()
    fig.tight_layout()
    f = PLOTS / "a_roundtrip.png"
    fig.savefig(f, dpi=120)
    files.append(str(f))
    plt.close(fig)

    # (b) irreducible orientation error for human vs synthetic chunks
    if raw.get("b_proj"):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for kind, proj in raw["b_proj"].items():
            ax.hist(np.degrees(proj), bins=np.linspace(0, 60, 61), histtype="step", density=True, label=kind)
        ax.set_xlabel("orientation the 5-DoF arm cannot execute (deg)")
        ax.set_ylabel("density")
        ax.set_title("(b) 6-DoF chunks re-anchored on robot poses")
        ax.legend()
        fig.tight_layout()
        f = PLOTS / "b_irreducible_rotation.png"
        fig.savefig(f, dpi=120)
        files.append(str(f))
        plt.close(fig)

    # workspace maps
    ws = res.get("workspace")
    if ws:
        st = _station()
        keys = list(ws["maps"])
        fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 5.4))
        xs, ys = np.array(ws["x"]), np.array(ws["y"])
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch

        cmap = ListedColormap(["#bbbbbb", "#f4f4f4", "#4c78a8", "#f58518", "#54a24b"])
        names = ["torso", "unreachable", "left only", "right only", "both arms"]
        body = np.array(ws.get("torso_mask", np.zeros((len(xs), len(ys)), bool)))
        for ax, key in zip(np.atleast_1d(axes), keys, strict=True):
            L, R = np.array(ws["maps"][key]["left"]), np.array(ws["maps"][key]["right"])
            img = 1 + L.astype(int) + 2 * R.astype(int)
            img[body] = 0
            d = (xs[1] - xs[0]) / 2
            ax.imshow(img.T, origin="lower", extent=[xs[0] - d, xs[-1] + d, ys[0] - d, ys[-1] + d], cmap=cmap,
                      vmin=0, vmax=4, interpolation="nearest")
            for s in ("left", "right"):
                b = st.T_station_base[s][:3, 3]
                ax.plot(b[0], b[1], "k^", ms=7)
            ax.set_title(key.replace("_", " "))
            ax.set_xlabel("x forward (m)")
            ax.set_ylabel("y left (m)")
            ax.set_aspect("equal")
        fig.legend(handles=[Patch(color=cmap(i), label=n) for i, n in enumerate(names)], loc="lower center", ncol=5)
        fig.suptitle("TCP reach on the table (station frame, fin-ray TCP, triangles = arm bases)")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        f = PLOTS / "b_workspace.png"
        fig.savefig(f, dpi=120)
        files.append(str(f))
        plt.close(fig)

    # (c) noise tolerance
    c = res.get("c_policy_noise")
    if c:
        keys = list(c)
        x = np.arange(len(keys))
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for metric, lab in (("noise_pos_mm", "injected noise"), ("exec_vs_target_pos_mm", "executed vs noisy target"),
                            ("exec_vs_clean_pos_mm", "executed vs clean")):
            ax[0].plot(x, [c[k][metric]["median"] for k in keys], "o-", label=lab)
        ax[0].set_ylabel("position error, median (mm)")
        for metric, lab in (("noise_rot_deg", "injected noise"), ("exec_vs_target_rot_deg", "executed vs noisy target"),
                            ("exec_vs_clean_rot_deg", "executed vs clean")):
            ax[1].plot(x, [c[k][metric]["median"] for k in keys], "o-", label=lab)
        ax[1].set_ylabel("orientation error, median (deg)")
        for a in ax:
            a.set_xticks(x, [k.replace("_", "\n") for k in keys], fontsize=8)
            a.legend(fontsize=8)
        fig.suptitle("(c) policy-like smooth noise on FK targets, after 5-DoF IK")
        fig.tight_layout()
        f = PLOTS / "c_noise_tolerance.png"
        fig.savefig(f, dpi=120)
        files.append(str(f))
        plt.close(fig)
    return files


# ---------------------------------------------------------------------------------------------------------


def _json_ready(x):
    if isinstance(x, dict):
        return {k: _json_ready(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_json_ready(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def verdict(res: dict) -> dict:
    a = res["a_real_trajectories"]["solvers"]["analytic"]
    checks = {
        "codec_round_trip": res["a_real_trajectories"]["codec_max_err_m"] < THRESHOLDS["codec_round_trip_m"],
        "onmanifold_median_pos": a["pos_mm"]["median"] < THRESHOLDS["onmanifold_median_pos_mm"],
        "onmanifold_p99_pos": a["pos_mm"]["p99"] < THRESHOLDS["onmanifold_p99_pos_mm"],
        "onmanifold_success": a["success_rate"] > THRESHOLDS["onmanifold_success_rate"],
        "onmanifold_jumps": a["jump_rate"] < THRESHOLDS["onmanifold_jump_rate"],
        "realtime": res["bench"]["analytic"]["cpu_p99_ms"] < THRESHOLDS["realtime_ms_per_arm"],
    }
    c = res.get("c_policy_noise", {})
    for k, v in c.items():
        if k.startswith("pos0mm"):
            continue
        ratio = THRESHOLDS["policy_noise_exec_vs_clean_ratio"]
        ok_pos = v["exec_vs_clean_pos_mm"]["median"] <= ratio * v["noise_pos_mm"]["median"] + 0.1
        ok_rot = v["exec_vs_clean_rot_deg"]["median"] <= ratio * v["noise_rot_deg"]["median"] + 0.1
        ok_jump = v["jump_rate_excl_parked"] < THRESHOLDS["policy_noise_jump_rate"]
        checks[f"noise_{k}"] = bool(ok_pos and ok_rot and ok_jump and v["pos_fail_rate_excl_parked"] < 0.01)
    return checks


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    args = ap.parse_args(argv)
    q = args.quick
    st = _station()
    res = {"station_source": st.source, "tcp_in_flange": st.arm.T_flange_tcp[:3, 3].tolist(), "thresholds": THRESHOLDS}
    raw = {}
    t0 = time.time()
    res["bench"] = bench(300 if q else 2000)
    raw["bench_ms"] = {k: v.pop("_raw_ms") for k, v in res["bench"].items()}
    print(f"[bench] {res['bench']}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        a = section_a(pool, 600 if q else 6000)
        raw["a_pos"], raw["a_t"] = a.pop("_raw_pos"), a.pop("_raw_t")
        res["a_real_trajectories"] = a
        print(f"[a] done {time.time() - t0:.0f}s", flush=True)
        b = section_b(pool, 100 if q else 1500)
        raw["b_proj"] = {k: v.pop("_raw_proj") for k, v in b.items()}
        res["b_6dof_motion"] = b
        res["workspace"] = workspace(pool, 0.05 if q else 0.02)
        print(f"[b] done {time.time() - t0:.0f}s", flush=True)
        res["c_policy_noise"] = section_c(pool, 300 if q else 3000)
        print(f"[c] done {time.time() - t0:.0f}s", flush=True)
    res["d_collisions"] = section_d(2000 if q else 20000)
    print(f"[d] done {time.time() - t0:.0f}s", flush=True)
    res["checks"] = verdict(res)
    res["plots"] = plots(res, raw)
    res["runtime_s"] = time.time() - t0
    args.out.write_text(json.dumps(_json_ready(res), indent=1))
    print(json.dumps(_json_ready({k: res[k] for k in ("checks", "runtime_s")}), indent=1))
    return res


if __name__ == "__main__":
    main()
