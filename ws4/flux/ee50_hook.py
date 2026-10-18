"""Hook for EE50 policies (option B): end-effector chunks -> joint commands through ws4/kinematics.

This module owns no kinematics. The EE50 codec (layout, 6D rotation rows, relative frames, z-scores) and the
SO-101 forward/inverse kinematics belong to ws4/kinematics; here we only adapt them to the runner's
``ActionSpaceAdapter`` protocol (``dual_arm_runner.py``).

Contract with an EE50 backend: the backend exchanges *raw, absolute* EE50 vectors with the runner, i.e.
``kinematics.ee50.encode`` with an identity anchor. Re-anchoring to the conditioning frame and per-embodiment
z-scoring stay inside the backend, next to the model, exactly like delta integration stays inside the joint
backends. No released code implements EE50 for FLUX (flux-action@830f1ce has no EE50 path at all; see
ws4/research/flux_dual_arm.md, option B), so every convention below is UNVERIFIED.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .contract import ARM_DIM, GRIPPER_INDEX

FK = Callable[[str, np.ndarray], np.ndarray]  # (arm, SO-101 joints in model units (6,)) -> (4, 4) TCP pose
IK = Callable[[str, np.ndarray, np.ndarray], np.ndarray]  # (arm, (4, 4) target, seed joints (6,)) -> joints (6,)


@dataclass
class EE50Adapter:
    """``ActionSpaceAdapter`` for an EE50 target driving ``arms``.

    ``encode`` / ``decode`` are ws4/kinematics' raw codec functions; ``fk`` / ``ik`` work in SO-101 model units
    (degrees, gripper %). ``open_fraction`` / ``gripper_percent`` convert the gripper (EE50 stores 0 closed,
    1 open; which end of LeRobot's 0..100 is open on our fin-ray gripper is UNVERIFIED).
    """

    arms: tuple[str, ...]
    fk: FK
    ik: IK
    encode: Callable
    decode: Callable
    open_fraction: Callable[[np.ndarray], np.ndarray] = lambda g: np.clip(np.asarray(g) / 100.0, 0.0, 1.0)
    gripper_percent: Callable[[np.ndarray], np.ndarray] = lambda f: np.clip(np.asarray(f), 0.0, 1.0) * 100.0
    ik_failures: int = 0

    def _identity_anchor(self) -> dict[str, np.ndarray]:
        return {arm: np.eye(4) for arm in self.arms}

    def joints_to_model(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float64).reshape(len(self.arms), ARM_DIM)
        poses = {arm: self.fk(arm, q)[None] for arm, q in zip(self.arms, joints, strict=True)}
        grippers = {
            arm: self.open_fraction(q[GRIPPER_INDEX : GRIPPER_INDEX + 1])
            for arm, q in zip(self.arms, joints, strict=True)
        }
        return np.asarray(self.encode(poses, grippers, self._identity_anchor()))[0]

    def model_to_joints(self, chunk: np.ndarray, joints_now: np.ndarray) -> np.ndarray:
        poses, grippers = self.decode(np.asarray(chunk, dtype=np.float64), self._identity_anchor())
        seeds = np.asarray(joints_now, dtype=np.float64).reshape(len(self.arms), ARM_DIM).copy()
        out = np.zeros((len(chunk), ARM_DIM * len(self.arms)))
        for t in range(len(chunk)):
            for i, arm in enumerate(self.arms):
                q = np.asarray(self.ik(arm, poses[arm][t], seeds[i]), dtype=np.float64)
                if q.shape != (ARM_DIM,) or not np.isfinite(q).all():
                    self.ik_failures += 1
                    q = seeds[i].copy()  # hold the last reachable joints rather than jump
                q[GRIPPER_INDEX] = float(self.gripper_percent(grippers[arm][t]))
                out[t, i * ARM_DIM : (i + 1) * ARM_DIM] = q
                seeds[i] = q
        return out


def from_ws4_kinematics(arms: tuple[str, ...] = ("left", "right"), ik: IK | None = None) -> EE50Adapter:
    """Wire ws4/kinematics (``ee50.encode/decode``, ``Station.fk``) into an adapter.

    ws4/kinematics is being written in parallel; pass ``ik`` explicitly until its IK entry point is settled.
    """
    from kinematics import ee50 as codec  # ws4/ on sys.path
    from kinematics.so101_kinematics import Station, lerobot_deg_to_rad

    station = Station.load()

    def fk(arm: str, q: np.ndarray) -> np.ndarray:
        return station.fk(arm, lerobot_deg_to_rad(np.asarray(q)[:5]))

    if ik is None:
        raise NotImplementedError("pass ik=... (ws4/kinematics IK in SO-101 degrees) until its API is final")
    return EE50Adapter(arms=arms, fk=fk, ik=ik, encode=codec.encode, decode=codec.decode)
