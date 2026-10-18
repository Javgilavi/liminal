"""Read-only MuJoCo joint mirror. It never opens devices or advances physics."""

from __future__ import annotations

import math
import time
from pathlib import Path

from liminal.core.types import SO101_JOINTS


def telemetry_source_age(message: dict) -> float:
    """Haller uses Unix seconds in `t`; older bridges may send milliseconds."""
    stamp = message.get("t")
    if stamp is None and "t_ms" in message:
        stamp = float(message["t_ms"]) / 1000
    if stamp is None:
        return 0.0  # legacy senders without a timestamp still use receive-age checks
    age = time.time() - float(stamp)
    if not math.isfinite(age) or abs(age) > 0.5:
        raise ValueError("Controller timestamp is stale or clocks differ by >500 ms")
    return max(0.0, age)


class JointMirror:
    def __init__(self, scene: str | Path):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.apply({side: [0, -25, 35, 35, 0, 35] for side in ("left", "right")})

    def apply(self, arms: dict[str, list[float]]) -> None:
        for side, values in arms.items():
            if side not in ("left", "right") or len(values) != 6 or not all(math.isfinite(x) for x in values):
                raise ValueError("expected left/right arms with six finite native degree/percent values")
            for name, value in zip(SO101_JOINTS, values, strict=True):
                joint = self.model.joint(f"{side}_{name}")
                lo, hi = joint.range
                # Measured joints must not pass through the policy's normalization or offsets.
                self.data.qpos[joint.qposadr[0]] = (
                    lo + value / 100 * (hi - lo) if name == "gripper" else math.radians(value)
                )
        self.mujoco.mj_forward(self.model, self.data)

    def poses(self) -> list:
        return [list(map(float, (*p, *q))) for p, q in zip(self.data.xpos, self.data.xquat, strict=True)]

    def geometry(self) -> dict:
        import base64

        m = self.model
        geometries, meshes = [], {}
        for i in range(m.ngeom):
            if m.geom_group[i] == 3 or m.geom(i).name in ("room_floor", "table_pedestal"):
                continue
            mesh_id = int(m.geom_dataid[i]) if int(m.geom_type[i]) == 7 else None
            if mesh_id is not None and mesh_id not in meshes:
                start, count = m.mesh_vertadr[mesh_id], m.mesh_vertnum[mesh_id]
                fstart, fcount = m.mesh_faceadr[mesh_id], m.mesh_facenum[mesh_id]
                meshes[mesh_id] = {
                    "vertices_b64": base64.b64encode(
                        m.mesh_vert[start : start + count].astype("<f4", copy=False).tobytes()
                    ).decode("ascii"),
                    "faces_b64": base64.b64encode(
                        m.mesh_face[fstart : fstart + fcount].astype("<u4", copy=False).tobytes()
                    ).decode("ascii"),
                }
            material = m.geom_matid[i]
            color = m.mat_rgba[material] if material >= 0 else m.geom_rgba[i]
            geometries.append(
                {
                    "body": int(m.geom_bodyid[i]),
                    "type": int(m.geom_type[i]),
                    "mesh": mesh_id,
                    "size": m.geom_size[i].tolist(),
                    "position": m.geom_pos[i].tolist(),
                    "quaternion": m.geom_quat[i].tolist(),
                    "rgba": color.tolist(),
                    "name": m.geom(i).name,
                }
            )
        return {
            "geometries": geometries,
            "meshes": meshes,
            "poses": self.poses(),
            "body_names": [m.body(i).name for i in range(m.nbody)],
            "calibration": "Estimated scene geometry; joint coordinates are native degrees/gripper percent",
        }


def parse_telemetry(message: dict) -> dict:
    """Decode Haller's existing public telemetry without acquiring motor ownership."""
    arms = {}
    for name in ("left", "right"):
        arm = message.get("arms", {}).get(name)
        if not arm:
            continue
        joints = arm.get("joints", {})
        values = [float(joints[j]["pos"]) for j in SO101_JOINTS]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite measured joint position")
        arms[name] = {"positions": values, "torque": bool(arm.get("torque", False)), "mode": arm.get("mode", "unknown")}
    if not arms:
        raise ValueError("no complete SO-101 arm in telemetry")
    return arms
