"""Inter-arm collision gate on the ws4/kinematics MuJoCo mesh model (``kinematics/mj_collision.py``).

Chunk level (``GatedDualArmRunner``): when a chunk joins an arm's queue, every frame that will execute is checked
against the other arm at the same tick: its queued commands, then its last command once the queue runs out. At the
first frame that penetrates, the rest of the chunk is replaced by the last safe frame a few ticks earlier
("truncate"); if no frame is safe the arm keeps its current command ("refuse"). The arm asks for a new chunk once
these frames have run, so a blocked arm waits until the other one has moved away.

Tick level (``TickGuardRobot``): the merged 12-D command is checked once more just before it is sent. A hit here means
the chunk check was bypassed (for example the safety guard slowed one arm), so the e-stop latches and nothing is sent.

The gate hooks ``DualArmRunner._accept`` and reads the runner's per-target queues (``_targets[...].queue``,
ws4/flux/dual_arm_runner.py:173-183, 287-300): both are private, so this module pins that coupling in one place.
Joint units: LeRobot degrees and gripper 0..100 in, MJCF radians out through ``sim/units.py`` (``to_radians``).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flux.backends import PolicyResponse
from flux.contract import ARM_DIM, ARMS, arm_slice
from flux.dual_arm_runner import DualArmRunner


@dataclass
class GateStats:
    chunks: int = 0
    truncated: int = 0
    refused: int = 0
    tick_blocks: int = 0
    frames: int = 0
    seconds: float = 0.0
    last: dict = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        ms = 1000 * self.seconds / self.frames if self.frames else None
        return {"chunks": self.chunks, "truncated": self.truncated, "refused": self.refused,
                "tick_blocks": self.tick_blocks, "ms_per_frame": None if ms is None else round(ms, 3),
                "last": self.last}


class CollisionGate:
    """``allowed(depth, base)``: a pose may not penetrate deeper than the arms already do (``base``, from the current
    command or the measured state) plus ``tolerance``, and never deeper than ``hard_limit``. Moving apart is always
    allowed, so arms that already touch (for example after an e-stop mid-motion) can separate."""

    def __init__(self, depth: float = 0.002, margin_frames: int = 3, classes: tuple[str, ...] = ("inter_arm",),
                 tolerance: float = 0.0005, hard_limit: float = 0.010):
        from kinematics.mj_collision import MjCollision
        from sim.units import StationUnits

        self.mj = MjCollision()
        self.units = StationUnits("degrees")
        self.depth = depth
        self.margin_frames = margin_frames
        self.classes = classes
        self.tolerance = tolerance
        self.hard_limit = hard_limit
        self.stats = GateStats()
        self._lock = threading.Lock()  # one MjData

    def _q(self, arm: str, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        conv = self.units.arms[arm]
        rad = np.array([conv.to_radians(f) for f in np.atleast_2d(frames)])
        grip = np.clip(np.atleast_2d(frames)[:, 5] / 100.0, 0.0, 1.0)
        return rad[:, :5], grip

    def penetration(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Worst penetration (m) of the gated classes per frame; ``frames[arm]`` is (T, 6) in robot units."""
        q, g = {}, {}
        for arm, f in frames.items():
            q[arm], g[arm] = self._q(arm, f)
        start = time.perf_counter()
        with self._lock:
            res = self.mj.check(q, g, self.depth)
        T = len(next(iter(q.values())))
        self.stats.frames += T
        self.stats.seconds += time.perf_counter() - start
        worst = np.zeros(T)
        for key in self.classes:
            if key in res:
                worst = np.maximum(worst, res[key])
        return worst

    def allowed(self, depth: np.ndarray | float, base: float = 0.0) -> np.ndarray:
        depth = np.asarray(depth, dtype=float)
        # Above the hard limit only non-increasing penetration passes: the arms may separate, never creep deeper.
        return (depth <= 0) | ((depth <= base + self.tolerance) & ((depth <= self.hard_limit) | (depth <= base)))

    def pose_penetration(self, *poses12: np.ndarray | None) -> float:
        """Deepest current penetration over the given 12-D poses (commands, measured state)."""
        out = 0.0
        for pose in poses12:
            if pose is not None:
                out = max(out, self.command_penetration(pose))
        return out

    def first_hit(self, arm: str, chunk: np.ndarray, other: str, other_frames: np.ndarray,
                  base: float = 0.0) -> tuple[int | None, float]:
        """First frame that is not ``allowed`` (see the class docstring), and its penetration depth."""
        depth = self.penetration({arm: chunk, other: other_frames})
        hits = np.flatnonzero(~self.allowed(depth, base))
        return (int(hits[0]), float(depth[hits[0]])) if len(hits) else (None, 0.0)

    def command_penetration(self, command12: np.ndarray) -> float:
        c = np.asarray(command12, dtype=float)
        return float(self.penetration({a: c[None, arm_slice(a)] for a in ARMS})[0])


class GatedDualArmRunner(DualArmRunner):
    """``DualArmRunner`` whose accepted chunks pass the collision gate first."""

    def __init__(self, *args, gate: CollisionGate | None = None,
                 on_event: Callable[[str, str, dict], None] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate
        self.on_event = on_event or (lambda level, msg, data: None)
        self.on_blocked: Callable[[str, str], None] | None = None  # (blocked arm, arm in the way)

    def other_future(self, arm: str, n: int, state12: np.ndarray) -> np.ndarray:
        """(n, 6) commands the arm will receive over the next n ticks, robot units."""
        now = (self._command if self._command is not None else state12)[arm_slice(arm)]
        frames: list[np.ndarray] = []
        for ts in self._targets.values():
            if ts.active and arm in ts.config.arms:
                i = ts.config.arms.index(arm)
                calib = ts.config.calibration[arm]
                for _, joints in list(ts.queue):
                    frames.append(calib.to_robot(joints[i * ARM_DIM:(i + 1) * ARM_DIM]))
                    if len(frames) >= n:
                        break
        frames = frames[:n]
        tail = frames[-1] if frames else now
        frames += [tail] * (n - len(frames))
        return np.asarray(frames, dtype=float)

    def _accept(self, ts, response: PolicyResponse, state12: np.ndarray) -> None:
        if self.gate is None or len(ts.config.arms) != 1:
            return super()._accept(ts, response, state12)
        arm = ts.config.arms[0]
        other = next(a for a in ARMS if a != arm)
        calib = ts.config.calibration[arm]
        n_exec = ts.contract.n_action_steps
        chunk = np.asarray(response.commands, dtype=float)
        robot_chunk = np.array([calib.to_robot(c) for c in chunk[:n_exec]])
        base = self.gate.pose_penetration(self._command, state12)
        hit, depth = self.gate.first_hit(arm, robot_chunk, other, self.other_future(other, n_exec, state12), base)
        self.gate.stats.chunks += 1
        if hit is None:
            return super()._accept(ts, response, state12)
        safe = hit - self.gate.margin_frames
        if safe >= 0:
            chunk = chunk.copy()
            chunk[safe + 1:] = chunk[safe]
            self.gate.stats.truncated += 1
            action = f"truncated at frame {safe + 1}/{n_exec}"
        else:
            now = (self._command if self._command is not None else state12)[arm_slice(arm)]
            chunk = np.repeat(calib.to_model(now)[None], len(chunk), 0)
            self.gate.stats.refused += 1
            action = "refused (holding)"
        info = {"target": ts.config.name, "arm": arm, "hit_frame": hit, "depth_mm": round(depth * 1000, 1),
                "action": action}
        self.gate.stats.last = info
        self.on_event("warn", f"collision gate: {arm} chunk {action}; it would reach the {other} arm at frame "
                      f"{hit} ({depth * 1000:.1f} mm)", info)
        if self.on_blocked is not None:
            self.on_blocked(arm, other)
        super()._accept(ts, PolicyResponse(chunk, response.latency_s, {**response.info, "gate": info}), state12)


class TickGuardRobot:
    """Last line before the motors: latch the e-stop instead of sending a command that makes the arms touch."""

    def __init__(self, robot: Any, gate: CollisionGate, estop: Callable[[str], None]):
        self.robot = robot
        self.gate = gate
        self.estop = estop
        self._measured: np.ndarray | None = None
        self._sent: np.ndarray | None = None

    def get_observation(self) -> dict:
        obs = self.robot.get_observation()
        self._measured = np.asarray(obs["observation.state"], dtype=float).copy()
        return obs

    def send_action(self, action: np.ndarray) -> None:
        depth = self.gate.command_penetration(action)
        base = self.gate.pose_penetration(self._sent, self._measured) if depth > 0 else 0.0
        if not self.gate.allowed(depth, base):
            self.gate.stats.tick_blocks += 1
            self.estop(f"tick guard: commanded poses would make the arms touch ({depth * 1000:.1f} mm); "
                       "nothing sent, e-stop latched")
            return
        self.robot.send_action(action)
        self._sent = np.asarray(action, dtype=float).copy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.robot, name)
