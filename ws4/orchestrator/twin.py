"""MuJoCo digital twin of the Haller for the HMI and the live lab: the measured joints of both arms, applied 1:1 to
the SO-101 CAD of the scene main's live lab uses (examples/dual_so101_sim/research_replay/scene.xml), then
``mj_forward`` only (no physics, no clipping, no policy normalization), as ``src/liminal/ui/twin.py`` does for Haller
telemetry. The browser renders the compiled geometry with site/viewer.js.

The scene names its arms in Haller's operator frame: its ``right_`` arm stands on the mat side (+x), which is the
robot's LEFT arm (the pen arm, /dev/haller_arm_right), and its ``left_`` arm on the box side is the robot's RIGHT arm
(the box arm). ``SCENE_SIDE`` maps the orchestrator's robot-frame names onto it.

Only the arms are mirrored: the table, box, lid and objects keep the scene's estimated poses."""

from __future__ import annotations

import base64
import math
import threading
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "examples" / "dual_so101_sim" / "research_replay" / "scene.xml"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SCENE_SIDE = {"left": "right", "right": "left"}  # robot side -> the scene's (Haller operator-frame) arm prefix


class Twin:
    def __init__(self, scene: str | Path = SCENE):
        import mujoco  # noqa: PLC0415

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self._lock = threading.Lock()
        self._geometry: dict | None = None
        self.applied: dict[str, list[float]] = {}

    def apply(self, arms: dict[str, Any]) -> None:
        """``arms``: robot side -> 6 measured values (LeRobot degrees, gripper 0..100 %). Unknown or incomplete arms
        are skipped, so a missing arm keeps its last pose."""
        with self._lock:
            for side, values in arms.items():
                if side not in SCENE_SIDE or values is None:
                    continue
                values = [float(v) for v in values][:6]
                if len(values) != 6 or not all(math.isfinite(v) for v in values):
                    continue
                for name, value in zip(JOINTS, values, strict=True):
                    joint = self.model.joint(f"{SCENE_SIDE[side]}_{name}")
                    lo, hi = joint.range
                    self.data.qpos[joint.qposadr[0]] = lo + value / 100 * (hi - lo) if name == "gripper" else \
                        math.radians(value)
                self.applied[side] = values
            self.mujoco.mj_forward(self.model, self.data)

    def poses(self) -> list[list[float]]:
        with self._lock:
            return [list(map(float, (*p, *q))) for p, q in zip(self.data.xpos, self.data.xquat, strict=True)]

    def geometry(self) -> dict:
        """Compiled geometry for site/viewer.js (``TwinView``), as src/liminal/ui/twin.py exports it."""
        if self._geometry is not None:
            return {**self._geometry, "poses": self.poses()}
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
                        m.mesh_vert[start: start + count].astype("<f4", copy=False).tobytes()).decode("ascii"),
                    "faces_b64": base64.b64encode(
                        m.mesh_face[fstart: fstart + fcount].astype("<u4", copy=False).tobytes()).decode("ascii"),
                }
            material = m.geom_matid[i]
            color = m.mat_rgba[material] if material >= 0 else m.geom_rgba[i]
            geometries.append({"body": int(m.geom_bodyid[i]), "type": int(m.geom_type[i]), "mesh": mesh_id,
                               "size": m.geom_size[i].tolist(), "position": m.geom_pos[i].tolist(),
                               "quaternion": m.geom_quat[i].tolist(), "rgba": color.tolist(),
                               "name": m.geom(i).name})
        self._geometry = {"geometries": geometries, "meshes": meshes,
                          "body_names": [m.body(i).name for i in range(m.nbody)],
                          "calibration": "measured joints of both arms (LeRobot degrees, gripper %); table and "
                                         "objects are the scene's estimate"}
        return {**self._geometry, "poses": self.poses()}
