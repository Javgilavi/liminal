"""Per-tick safety monitors for the real arm. A returned reason latches the e-stop (``Executor.monitors``).

Both rebuild, tick by tick, the stop checks that ``ws4/single_arm/run_single_arm.py`` runs inside its own attempt loop
(read-only reuse, same defaults as its CLI):
- ``StallMonitor``: an arm joint that lags its command by more than ``stall_deg`` for ``stall_ticks`` ticks is blocked
  (table, box, its own base); the gripper is excluded because it legitimately squeezes the pen
  (run_single_arm.py:572-582; ``--stall-deg 8 --stall-ticks 15``).
- ``KeepOutMonitor``: the FK keep-out guard (``KeepOutGuard``) refused the target ``ticks`` times in a row, so the
  policy keeps pushing into a forbidden zone (run_single_arm.py:564-571; ``--keepout-ticks 45``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from flux.contract import ARM_DIM, ARM_JOINTS, OBS_STATE, arm_slice


class StallMonitor:
    def __init__(self, arm: str = "left", stall_deg: float = 8.0, stall_ticks: int = 15):
        self.arm, self.stall_deg, self.stall_ticks = arm, stall_deg, stall_ticks
        self.count = np.zeros(ARM_DIM - 1, dtype=int)
        self._last_tick: int | None = None

    def __call__(self, report: Any, obs: dict) -> str | None:
        if self._last_tick is not None and report.tick != self._last_tick + 1:
            self.count[:] = 0  # resumed after a pause: start counting afresh
        self._last_tick = report.tick
        s = arm_slice(self.arm)
        lag = np.abs(np.asarray(report.command, dtype=float)[s] - np.asarray(obs[OBS_STATE], dtype=float)[s])
        lag = lag[: ARM_DIM - 1]
        self.count = np.where(lag > self.stall_deg, self.count + 1, 0)
        if (self.count >= self.stall_ticks).any():
            j = int(np.argmax(self.count))
            n = int(self.count[j])
            self.count[:] = 0
            return f"stall: {ARM_JOINTS[j]} lagged its command by {lag[j]:.0f} deg for {n} ticks"
        return None


class KeepOutMonitor:
    def __init__(self, guard: Any, ticks: int = 45):
        self.guard, self.ticks = guard, ticks
        self.run = 0

    def __call__(self, report: Any, obs: dict) -> str | None:
        zone = getattr(self.guard, "last_zone", "")
        self.run = self.run + 1 if zone else 0
        if self.run >= self.ticks:
            n, self.run = self.run, 0
            return f"keep-out: the {zone} zone refused the target for {n} ticks in a row"
        return None
