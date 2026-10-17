"""Per-tick safety monitors for the real arm. A returned reason latches the e-stop (``Executor.monitors``).

Both rebuild, tick by tick, the stop checks that ``ws4/single_arm/run_single_arm.py`` runs inside its own attempt loop
(read-only reuse, same defaults as its CLI):
- ``StallMonitor``: an arm joint that lags its command by more than ``stall_deg`` for ``stall_ticks`` ticks is blocked
  (table, box, its own base); the gripper is excluded because it legitimately squeezes the pen
  (run_single_arm.py:572-582; ``--stall-deg 8 --stall-ticks 15``).
- ``KeepOutMonitor``: the FK keep-out guard (``KeepOutGuard``) refused the target ``ticks`` times in a row, so the
  policy keeps pushing into a forbidden zone (run_single_arm.py:564-571; ``--keepout-ticks 45``).
- ``LoadMonitor``: the load freeze of ``run_attempt`` (policy only, like there): a command clamped at a limit can sit at
  low lag while the servo pushes into an end stop, which the stall check never sees. Trips at >= 70% of the torque cap
  for 9 ticks or >= 90% on one tick (``--load-sustain 0.7 --load-ticks 9 --load-peak 0.9``; it trips on the red arm's
  two damaging attempts and on none of the Haller runs, max 29%, per so101-zeroshot 2026-10-17).
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
    """While ``ramping()`` (the scripted move to the start pose), the first refusal already stops, as
    run_single_arm.move_to does (it holds and returns False on a keep-out refusal)."""

    def __init__(self, guard: Any, ticks: int = 45, ramping: Any = None):
        self.guard, self.ticks, self.ramping = guard, ticks, ramping
        self.run = 0

    def __call__(self, report: Any, obs: dict) -> str | None:
        zone = getattr(self.guard, "last_zone", "")
        self.run = self.run + 1 if zone else 0
        if zone and self.ramping is not None and self.ramping():
            self.run = 0
            return f"keep-out: the {zone} zone refused the ramp to the start pose"
        if self.run >= self.ticks:
            n, self.run = self.run, 0
            return f"keep-out: the {zone} zone refused the target for {n} ticks in a row"
        return None


class LoadMonitor:
    def __init__(self, read_load: Any, cap: float, sustain: float = 0.7, peak: float = 0.9, ticks: int = 9,
                 ramping: Any = None):
        self.read_load, self.cap, self.sustain, self.peak, self.ticks = read_load, cap, sustain, peak, ticks
        self.ramping = ramping
        self.count = np.zeros(ARM_DIM - 1, dtype=int)
        self._last_tick: int | None = None

    def __call__(self, report: Any, obs: dict) -> str | None:
        if self._last_tick is not None and report.tick != self._last_tick + 1:
            self.count[:] = 0
        self._last_tick = report.tick
        if self.ramping is not None and self.ramping():  # run_single_arm's scripted ramps have no load freeze
            self.count[:] = 0
            return None
        load = self.read_load()
        if load is None:
            return None
        frac = np.abs(np.asarray(load, dtype=float))[: ARM_DIM - 1] / self.cap
        self.count = np.where(frac >= self.sustain, self.count + 1, 0)  # same rule as run_single_arm.load_trip
        if (frac >= self.peak).any() or (self.count >= self.ticks).any():
            j = int(np.argmax(frac))
            self.count[:] = 0
            return f"load: {ARM_JOINTS[j]} at {100 * frac[j]:.0f}% of the torque cap"
        return None
