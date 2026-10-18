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
                 release_first: bool = False, poses: dict | None = None, arrive_tol: float | np.ndarray = 3.0):
        self.home = np.asarray(HOME_DEG if home is None else home, dtype=float)
        # named poses of this arm (station.poses): a task with that name ramps there; "tuck" goes to "tucked"
        self.poses = {k: np.asarray(v, dtype=float) for k, v in (poses or {}).items()}
        self.joint_step = joint_step
        self.gripper_step = gripper_step
        # a ramp that opens the gripper (``home``) opens it in place before the arm moves: a FLUX step can end (its
        # check passed) with the pen still gripped in its slot, and moving while it opens would drag the pen out
        self.release_first = release_first
        # deg per arm joint at which the measured pose counts as arrived; the Haller uses run_single_arm.ARRIVE_TOL
        # (lift 8: the 7.4 V lift settles 4 to 6 deg short against gravity, the other joints 6)
        self.arrive_tol = np.broadcast_to(np.asarray(arrive_tol, dtype=float), (5,)).copy()
        # (joint order, goal) after a keep-out refusal on the ramp to that goal: the arm joints move one at a time in
        # that order, then the gripper (run_single_arm.move_to's fallback); None: straight ramps
        self.fallback: tuple[list[int], np.ndarray] | None = None

    def set_fallback(self, order: list[int] | None, goal: np.ndarray | None = None) -> None:
        self.fallback = None if order is None else (list(order), np.asarray(goal, dtype=float).copy())

    def goal(self, task: str, anchor: np.ndarray) -> np.ndarray:
        goal = np.asarray(anchor, dtype=float).copy()
        if task == "home":
            goal = self.home.copy()
        elif task in self.poses or (task == "tuck" and "tucked" in self.poses):
            goal = self.poses[task if task in self.poses else "tucked"].copy()
        elif task == "open_gripper":
            goal[5] = GRIPPER_OPEN
        elif task == "close_gripper":
            goal[5] = GRIPPER_CLOSED
        else:
            raise ValueError(f"unknown primitive {task!r}")
        return goal

    def reached(self, task: str, command6: np.ndarray, state6: np.ndarray, tol_deg: float | None = None) -> bool:
        """Done when the command has arrived (the gripper may stop on an object, so its state is not checked)."""
        goal = self.goal(task, command6)
        tol = self.arrive_tol if tol_deg is None else tol_deg
        return bool(np.abs(np.asarray(command6)[:5] - goal[:5]).max() < 1e-6
                    and abs(command6[5] - goal[5]) < 1e-6
                    and (np.abs(np.asarray(state6)[:5] - goal[:5]) < tol).all())

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        anchor = np.asarray(request.commands[-1], dtype=float)
        goal = self.goal(request.task, anchor)
        limit = np.r_[np.full(5, self.joint_step), self.gripper_step]
        order = self.fallback[0] if self.fallback is not None and np.allclose(self.fallback[1], goal) else None
        frames, cur = [], anchor
        for _ in range(self.contract.chunk_size):
            step = np.clip(goal - cur, -limit, limit)
            if self.release_first and goal[5] > cur[5] + 1e-6 and request.task not in ("open_gripper",):
                step[:5] = 0.0  # gripper first, arm still
            elif order is not None:
                moving = next((j for j in order if abs(goal[j] - cur[j]) > 1e-9), None)
                if moving is not None:  # one arm joint at a time; the gripper after the arm joints
                    keep = step[moving]
                    step[:] = 0.0
                    step[moving] = keep
            cur = cur + step
            frames.append(cur)
        return PolicyResponse(np.asarray(frames), 0.0, {"backend": "primitive", "task": request.task})
