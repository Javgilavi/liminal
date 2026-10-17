"""Collaboration & safety supervisor (owned by WS5).

Separate from the planner on purpose: two commands can be logically compatible but
physically unsafe. Responsibilities:

* arm ownership   - exactly one node may command an arm at a time
* workspace zones - only one *moving* arm inside a shared zone at a time
                    (a static, holding arm does not take the zone lock)
* action limits   - clip to joint limits and max per-step delta, reject NaN
* e-stop          - freeze all arms, make every running node fail fast
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass

from liminal.core.interfaces import Arm
from liminal.core.types import JointPositions


class ArmConflict(RuntimeError):
    """Raised when a node tries to command an arm another node owns (a planning bug)."""


class EmergencyStop(RuntimeError):
    pass


@dataclass
class JointLimits:
    lower: Sequence[float]
    upper: Sequence[float]
    max_step: float = 10.0  # max change per action, same units as the arm (deg or normalized)


class SafetySupervisor:
    def __init__(self, arms: dict[str, Arm], limits: dict[str, JointLimits] | None = None):
        self.arms = arms
        self.limits = limits or {}
        self._owners: dict[str, str] = {}
        self._zones: dict[str, asyncio.Lock] = {}
        self._estopped = False

    # -- ownership ---------------------------------------------------------------------

    @asynccontextmanager
    async def claim(self, arm: str, owner: str) -> AsyncIterator[None]:
        if arm not in self.arms:
            raise KeyError(f"unknown arm {arm!r}; known: {sorted(self.arms)}")
        current = self._owners.get(arm)
        if current is not None:
            raise ArmConflict(f"arm {arm!r} is owned by {current!r}, {owner!r} cannot command it")
        self._owners[arm] = owner
        try:
            yield
        finally:
            self._owners.pop(arm, None)

    def owner_of(self, arm: str) -> str | None:
        return self._owners.get(arm)

    @asynccontextmanager
    async def zones(self, names: Sequence[str]) -> AsyncIterator[None]:
        # Sorted acquisition order prevents deadlock between two multi-zone skills.
        locks = [self._zones.setdefault(n, asyncio.Lock()) for n in sorted(set(names))]
        acquired: list[asyncio.Lock] = []
        try:
            for lock in locks:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

    # -- actions -----------------------------------------------------------------------

    def validate(self, arm: str, target: JointPositions, current: JointPositions | None = None) -> JointPositions:
        self.check()
        if any(math.isnan(x) or math.isinf(x) for x in target):
            raise ValueError(f"non-finite action for {arm}: {target}")
        lim = self.limits.get(arm)
        if lim is None:
            return list(target)
        out = [min(max(x, lo), hi) for x, lo, hi in zip(target, lim.lower, lim.upper, strict=True)]
        if current is not None:
            out = [c + max(-lim.max_step, min(lim.max_step, x - c)) for x, c in zip(out, current, strict=True)]
        return out

    def send(self, arm: str, target: JointPositions) -> JointPositions:
        """Validate and send one action. Use this instead of arm.send_action directly."""
        a = self.arms[arm]
        safe = self.validate(arm, target, a.read_state())
        a.send_action(safe)
        return safe

    # -- e-stop ------------------------------------------------------------------------

    def estop(self) -> None:
        self._estopped = True
        for a in self.arms.values():
            try:
                a.stop()
            except Exception:  # noqa: BLE001, S110 - keep stopping the others
                pass

    def reset_estop(self) -> None:
        self._estopped = False

    @property
    def estopped(self) -> bool:
        return self._estopped

    def check(self) -> None:
        if self._estopped:
            raise EmergencyStop("e-stop active")
