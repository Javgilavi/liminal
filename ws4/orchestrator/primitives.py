"""Scripted primitives (``home``, ``open_gripper``, ``close_gripper``) behind the FLUX backend interface. Never FLUX.

``hold`` needs no backend: an arm with no active target keeps its last command (``DualArmRunner.stop`` and
``step``, ws4/flux/dual_arm_runner.py:247-254, 322-324), which is exactly "hold the lid where it is".
"""

from __future__ import annotations

import numpy as np
from flux.backends import PolicyRequest, PolicyResponse
from flux.contract import SO101_CONTRACT

# Ready pose in LeRobot degrees, the same as ws4/sim/env.py HOME_DEG. UNVERIFIED on the real arms.
HOME_DEG = np.array([0.0, -10.0, 10.0, 75.0, 0.0, 30.0])
GRIPPER_OPEN = 80.0
GRIPPER_CLOSED = 0.0


class PrimitiveBackend:
    contract = SO101_CONTRACT

    def __init__(self, home: np.ndarray | None = None, joint_step: float = 2.0, gripper_step: float = 5.0,
                 release_first: bool = False):
        self.home = np.asarray(HOME_DEG if home is None else home, dtype=float)
        self.joint_step = joint_step
        self.gripper_step = gripper_step
        # ``home`` opens the gripper in place before the arm moves: a FLUX step can end (its check passed) with the
        # pen still gripped in its slot, and moving while the gripper opens would drag the pen out (Haller)
        self.release_first = release_first

    def goal(self, task: str, anchor: np.ndarray) -> np.ndarray:
        goal = np.asarray(anchor, dtype=float).copy()
        if task == "home":
            goal = self.home.copy()
        elif task == "open_gripper":
            goal[5] = GRIPPER_OPEN
        elif task == "close_gripper":
            goal[5] = GRIPPER_CLOSED
        else:
            raise ValueError(f"unknown primitive {task!r}")
        return goal

    def reached(self, task: str, command6: np.ndarray, state6: np.ndarray, tol_deg: float = 3.0) -> bool:
        """Done when the command has arrived (the gripper may stop on an object, so its state is not checked)."""
        goal = self.goal(task, command6)
        return bool(np.abs(np.asarray(command6)[:5] - goal[:5]).max() < 1e-6
                    and abs(command6[5] - goal[5]) < 1e-6
                    and np.abs(np.asarray(state6)[:5] - goal[:5]).max() < tol_deg)

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        anchor = np.asarray(request.commands[-1], dtype=float)
        goal = self.goal(request.task, anchor)
        limit = np.r_[np.full(5, self.joint_step), self.gripper_step]
        frames, cur = [], anchor
        for _ in range(self.contract.chunk_size):
            step = np.clip(goal - cur, -limit, limit)
            if self.release_first and request.task == "home" and goal[5] > cur[5] + 1e-6:
                step[:5] = 0.0  # gripper first, arm still
            cur = cur + step
            frames.append(cur)
        return PolicyResponse(np.asarray(frames), 0.0, {"backend": "primitive", "task": request.task})
