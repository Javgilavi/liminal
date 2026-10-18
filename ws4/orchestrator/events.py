"""Thread-safe event log: shown in the HMI, appended to ``runs/<run>/events.jsonl``."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: str | Path | None = None, maxlen: int = 600, echo: bool = False):
        self.path = Path(path) if path else None
        self._events: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._seq = 0
        self.echo = echo

    def set_path(self, path: str | Path | None) -> None:
        with self._lock:
            self.path = Path(path) if path else None

    def add(self, level: str, msg: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "time": round(time.time(), 3), "level": level, "msg": msg}
            if data:
                event["data"] = data
            self._events.append(event)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
        if self.echo:
            print(f"[{level}] {msg}", flush=True)
        return event

    def recent(self, n: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)[-n:]

    def find(self, text: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._events if text in e["msg"]]
