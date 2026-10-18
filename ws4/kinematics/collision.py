"""Capsule collision checks for two SO-101 arms on the XLeRobot torso (numpy, no MuJoCo needed).

Each arm is a chain of capsules between joint origins, sized from the STS3215 servo (45 x 24 x 32 mm) and the
fin-ray fingers (123 x 26 x 40 mm, ws4/sim/station_geometry.yaml). Radii are deliberately generous: this is
a screening tool for the gate, not a planner. The torso is the arm-base box from station_geometry.yaml and the
mast a vertical capsule. Distances are signed: negative means the capsules overlap.
"""

from __future__ import annotations

import numpy as np

from .so101_kinematics import Station

# (from point, to point, radius). Points: "base" (arm base origin), "j0".."j4" (joint frame origins after
# rotation), "tcp". Adjacent capsules share a joint and are never tested against each other.
LINKS = (
    ("base", "j0", 0.035),  # base motor + turntable
    ("j0", "j1", 0.030),  # shoulder bracket
    ("j1", "j2", 0.025),  # upper arm
    ("j2", "j3", 0.025),  # forearm
    ("j3", "j4", 0.025),  # wrist
    ("j4", "tcp", 0.030),  # gripper body + fingers
)
# Self-collision pairs (link indices) that are not adjacent and can physically meet.
SELF_PAIRS = ((0, 3), (0, 4), (0, 5), (1, 4), (1, 5), (2, 5))


def link_points(station: Station, side: str, q: np.ndarray) -> dict[str, np.ndarray]:
    """Capsule end points in the station frame for joints ``q`` (T, 5): name -> (T, 3)."""
    q = np.atleast_2d(q)
    Tb = station.T_station_base[side]
    frames = Tb @ station.arm.joint_frames(q)  # (T, 5, 4, 4)
    tcp = Tb @ station.arm.fk(q)
    pts = {"base": np.broadcast_to(Tb[:3, 3], (len(q), 3))}
    for i in range(5):
        pts[f"j{i}"] = frames[:, i, :3, 3]
    pts["tcp"] = tcp[:, :3, 3]
    return pts


def capsules(station: Station, side: str, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, float]]:
    pts = link_points(station, side, q)
    return [(pts[a], pts[b], r) for a, b, r in LINKS]


def segment_distance(p1, q1, p2, q2) -> np.ndarray:
    """Closest distance between segments [p1, q1] and [p2, q2], batched over the leading axis."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("...i,...i->...", d1, d1)
    e = np.einsum("...i,...i->...", d2, d2)
    f = np.einsum("...i,...i->...", d2, r)
    c = np.einsum("...i,...i->...", d1, r)
    b = np.einsum("...i,...i->...", d1, d2)
    denom = a * e - b * b
    eps = 1e-12
    s = np.where(denom > eps, np.clip((b * f - c * e) / np.maximum(denom, eps), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, eps)
    t_c = np.clip(t, 0, 1)
    s_lo = np.clip(-c / np.maximum(a, eps), 0, 1)
    s_hi = np.clip((b - c) / np.maximum(a, eps), 0, 1)
    s = np.where(t < 0, s_lo, np.where(t > 1, s_hi, s))
    t = t_c
    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


def segment_box_distance(p, q, center, half, n: int = 9) -> np.ndarray:
    """Approximate distance from segment [p, q] to an axis-aligned box (sampled along the segment)."""
    s = np.linspace(0, 1, n)
    pts = p[:, None, :] + (q - p)[:, None, :] * s[None, :, None]
    d = np.abs(pts - np.asarray(center)) - np.asarray(half)
    outside = np.linalg.norm(np.maximum(d, 0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0)
    return np.min(outside + inside, axis=-1)


def check(station: Station, q: dict[str, np.ndarray], table_margin: float = 0.01) -> dict[str, np.ndarray]:
    """Signed clearances per frame. ``q[side]`` (T, 5). Returns arrays (T,) in metres; < 0 means contact.

    ``table``: lowest capsule surface above the table minus ``table_margin`` (grasping touches the table, so
    only penetration deeper than the margin counts). ``torso`` and ``mast``: clearance to the torso box and the
    mast capsule, excluding each arm's own base capsule (it sits on the box). ``self``: non-adjacent links.
    ``inter_arm``: closest pair across the two arms (only if both are given).
    """
    caps = {s: capsules(station, s, qs) for s, qs in q.items()}
    out = {}
    torso = station.torso
    mast = station.mast
    for side, cs in caps.items():
        T = len(cs[0][0])
        table = np.full(T, np.inf)
        body = np.full(T, np.inf)
        mast_d = np.full(T, np.inf)
        for i, (p, e, r) in enumerate(cs):
            table = np.minimum(table, np.minimum(p[:, 2], e[:, 2]) - r - station.table_z + table_margin)
            if i == 0:
                continue
            body = np.minimum(body, segment_box_distance(p, e, torso["center"], torso["half_extents"]) - r)
            m0 = np.broadcast_to(np.asarray(mast["p0"], float), p.shape)
            m1 = np.broadcast_to(np.asarray(mast["p1"], float), p.shape)
            mast_d = np.minimum(mast_d, segment_distance(p, e, m0, m1) - r - mast["radius"])
        selfd = np.full(T, np.inf)
        for i, j in SELF_PAIRS:
            selfd = np.minimum(selfd, segment_distance(cs[i][0], cs[i][1], cs[j][0], cs[j][1]) - cs[i][2] - cs[j][2])
        out[f"{side}_table"] = table
        out[f"{side}_torso"] = body
        out[f"{side}_mast"] = mast_d
        out[f"{side}_self"] = selfd
    if len(caps) == 2:
        (_, ca), (_, cb) = caps.items()
        inter = np.full(len(ca[0][0]), np.inf)
        for pa, ea, ra in ca[1:]:
            for pb, eb, rb in cb[1:]:
                inter = np.minimum(inter, segment_distance(pa, ea, pb, eb) - ra - rb)
        out["inter_arm"] = inter
    return out
