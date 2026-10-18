"""Per-joint stall evidence from sustained load and lack of measured motion."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class _Waiting:
    since: float
    position: float


class StallMonitor:
    def __init__(self, *, window_s: float = 1.0, min_load: float = 0.4,
                 min_error: float = 0.75, min_progress: float = 0.1):
        if not all(math.isfinite(v) and v > 0 for v in (window_s, min_load, min_error, min_progress)):
            raise ValueError("stall thresholds must be finite and positive")
        if min_load > 1:
            raise ValueError("stall min_load is a fraction in (0, 1]")
        self.window_s, self.min_load = window_s, min_load
        self.min_error, self.min_progress = min_error, min_progress
        self._waiting: dict[str, _Waiting] = {}

    def reset(self) -> None:
        self._waiting.clear()

    def observe(self, now: float, positions: dict[str, float], targets: dict[str, float],
                loads: dict[str, float]) -> None:
        if not math.isfinite(now):
            raise ValueError("invalid stall observation timestamp")
        for joint in list(self._waiting):
            if joint not in targets:
                del self._waiting[joint]
        for joint, target in targets.items():
            position, load = positions[joint], abs(loads[joint])
            if not all(math.isfinite(v) for v in (position, target, load)) or load > 1:
                raise RuntimeError(f"invalid stall telemetry for {joint}")
            error = abs(target - position)
            if load < self.min_load or error < self.min_error:
                self._waiting.pop(joint, None)
                continue
            waiting = self._waiting.get(joint)
            if waiting is None or abs(position - waiting.position) >= self.min_progress:
                self._waiting[joint] = _Waiting(now, position)
                continue
            if now - waiting.since >= self.window_s:
                raise RuntimeError(
                    f"possible stall on {joint}: load {load:.0%}, error {error:.2f}, "
                    f"movement <{self.min_progress:g} for {now - waiting.since:.2f}s"
                )
