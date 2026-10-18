"""Per-tick safety monitors for the real arm. A returned reason latches the e-stop (``Executor.monitors``).

Both rebuild, tick by tick, the stop checks that ``ws4/single_arm/run_single_arm.py`` runs inside its own attempt loop
(read-only reuse, same defaults as its CLI):
- ``StallMonitor``: an arm joint that lags its command by more than ``stall_deg`` for ``stall_ticks`` ticks is blocked
  (table, box, its own base); the gripper is excluded because it legitimately squeezes the pen
  (run_single_arm.py:572-582; ``--stall-deg 8 --stall-ticks 15``).
- ``KeepOutMonitor``: the FK keep-out guard refused the arm's target on ``ticks`` of its last ``window`` policy ticks,
  so the policy keeps pushing into a forbidden zone; ticks that only hold the pose between chunks ("waiting") neither
  count nor clear (run_single_arm.KeepoutStop, ``--keepout-ticks 45 --keepout-window 60``: a consecutive count was
  reset by those holds and never reached 45 with FLUX's chunks).
- ``LoadMonitor``: the load freeze of ``run_attempt``, here on scripted ramps too (the tuck <-> home transitions of the
  final chain, parent 2026-10-18): a command clamped at a limit can sit at low lag while the servo pushes into an end
  stop, which the stall check never sees. Trips at >= 70% of the torque cap
  for 9 ticks or >= 90% on one tick (``--load-sustain 0.7 --load-ticks 9 --load-peak 0.9``; it trips on the red arm's
  two damaging attempts and on none of the Haller runs, max 29%, per so101-zeroshot 2026-10-17).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from flux.contract import ARM_DIM, ARM_JOINTS, OBS_STATE, arm_slice


class StallMonitor:
    """``active(arm)``: only an arm that some target drives can stall. An arm that only holds its last command is
    exempt: the box arm holding the lid open lags its command under the lid's weight by design."""

    def __init__(self, arm: str = "left", stall_deg: float = 8.0, stall_ticks: int = 15, active: Any = None):
        self.arm, self.stall_deg, self.stall_ticks, self.active = arm, stall_deg, stall_ticks, active
        self.count = np.zeros(ARM_DIM - 1, dtype=int)
        self._last_tick: int | None = None

    def __call__(self, report: Any, obs: dict) -> str | None:
        if self._last_tick is not None and report.tick != self._last_tick + 1:
            self.count[:] = 0  # resumed after a pause: start counting afresh
        self._last_tick = report.tick
        if self.active is not None and not self.active(self.arm):
            self.count[:] = 0
            return None
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
    """While ``ramping(slot)`` (a scripted ramp of that arm), the first refusal already stops, as
    run_single_arm.move_to does (it holds and returns False on a keep-out refusal). ``owner(slot)`` names the runner
    target driving the arm, whose status tells the "waiting" holds apart."""

    def __init__(self, guard: Any, ticks: int = 45, ramping: Any = None, slot: str | None = None, window: int = 60,
                 owner: Any = None):
        self.guard, self.ticks, self.ramping, self.slot, self.owner = guard, ticks, ramping, slot, owner
        self.window: deque[bool] = deque(maxlen=max(window, ticks))
        self.zone = ""

    def __call__(self, report: Any, obs: dict) -> str | None:
        zones = getattr(self.guard, "zones", None)
        zone = zones.get(self.slot, "") if self.slot and zones is not None else getattr(self.guard, "last_zone", "")
        who = f"the {self.slot} arm's" if self.slot else "the"
        if zone and self.ramping is not None and self.ramping(self.slot):
            self.window.clear()
            return f"keep-out: the {zone} zone refused {who} ramp to its start pose"
        status = ""
        if self.owner is not None and report is not None:
            status = getattr(report, "status", {}).get(self.owner(self.slot), "")
        if zone or status != "waiting":
            self.window.append(bool(zone))
            self.zone = zone or self.zone
        if sum(self.window) >= self.ticks:
            n, m = sum(self.window), len(self.window)
            self.window.clear()
            return f"keep-out: the {self.zone} zone refused {who} target on {n} of the last {m} policy ticks"
        return None


class LoadMonitor:
    def __init__(self, read_load: Any, cap: float, sustain: float = 0.7, peak: float = 0.9, ticks: int = 9,
                 ramping: Any = None, slot: str | None = None):
        self.read_load, self.cap, self.sustain, self.peak, self.ticks = read_load, cap, sustain, peak, ticks
        self.ramping, self.slot = ramping, slot
        self.count = np.zeros(ARM_DIM - 1, dtype=int)
        self._last_tick: int | None = None

    def __call__(self, report: Any, obs: dict) -> str | None:
        if self._last_tick is not None and report.tick != self._last_tick + 1:
            self.count[:] = 0
        self._last_tick = report.tick
        load = self.read_load()
        if load is None:
            return None
        frac = np.abs(np.asarray(load, dtype=float))[: ARM_DIM - 1] / self.cap
        self.count = np.where(frac >= self.sustain, self.count + 1, 0)  # same rule as run_single_arm.load_trip
        if (frac >= self.peak).any() or (self.count >= self.ticks).any():
            j = int(np.argmax(frac))
            self.count[:] = 0
            arm = f"{self.slot} arm " if self.slot else ""
            return f"load: {arm}{ARM_JOINTS[j]} at {100 * frac[j]:.0f}% of the torque cap"
        return None
