"""Shared symbolic world state with an append-only event log.

Perception writes predicates, the orchestrator reads/awaits them. Conditions are
predicate names; prefix with "!" for negation (e.g. "!lid_closed").
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Listener = Callable[[str, Any, Any], None]


class WorldState:
    def __init__(self, initial: dict[str, Any] | None = None, log_path: str | Path | None = None):
        self._facts: dict[str, Any] = dict(initial or {})
        self._changed = asyncio.Event()
        self._listeners: list[Listener] = []
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []

    # -- reads -------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._facts.get(key, default)

    def holds(self, condition: str) -> bool:
        if condition.startswith("!"):
            return not bool(self._facts.get(condition[1:], False))
        return bool(self._facts.get(condition, False))

    def snapshot(self) -> dict[str, Any]:
        return dict(self._facts)

    # -- writes ------------------------------------------------------------------------

    def set(self, key: str, value: Any, source: str = "unknown") -> None:
        old = self._facts.get(key)
        if old == value:
            return
        self._facts[key] = value
        self.log("fact", key=key, value=value, old=old, source=source)
        for listener in list(self._listeners):
            listener(key, old, value)
        # Wake every waiter, then arm a fresh event for the next change.
        self._changed.set()
        self._changed = asyncio.Event()

    def update(self, facts: dict[str, Any], source: str = "unknown") -> None:
        for key, value in facts.items():
            self.set(key, value, source=source)

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def log(self, kind: str, **data: Any) -> None:
        event = {"t": time.time(), "kind": kind, **data}
        self.events.append(event)
        if self._log_path:
            with self._log_path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")

    # -- waits -------------------------------------------------------------------------

    async def wait_for(self, condition: str, timeout: float | None = None) -> bool:
        """Wait until `condition` holds. Returns False on timeout."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while not self.holds(condition):
            event = self._changed
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return False
            try:
                await asyncio.wait_for(event.wait(), remaining)
            except asyncio.TimeoutError:
                return self.holds(condition)
        return True

    async def wait_until_violated(self, condition: str) -> None:
        """Return as soon as `condition` stops holding (used by WHILE monitors)."""
        while self.holds(condition):
            await self._changed.wait()
