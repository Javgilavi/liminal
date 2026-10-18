"""HMI panel: what FLUX imagined next to what the cameras saw, per chunk (opt-in: ``--imagine lazy``).

FLUX 3 Action denoises a short video together with each action chunk; ws4/imagine/serving.py makes the policy server
keep it (``imagine="lazy"``: no extra latency, decoded on fetch, 0.12 s per chunk on the RTX PRO 6000) and returns its
id in the chunk's info. For every new chunk of a running FLUX target this panel fetches the imagination in the
background, keeps the frame imagined for the next replan (control offset +32, 1.07 s ahead; the frame whose offset is
closest), and pairs it with the real head and wrist frames recorded closest to that moment. Imagined frames are the
model's canvas (scene | wrist, 256x256 each); the actual pair is built the same way. Nothing here touches the arm, and a
failed fetch only shows as an error in the panel.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from flux.contract import IMG_HEAD, WRIST_KEY

NEXT_REPLAN_OFFSET = 32  # control steps: the imagined view when the next chunk is requested (research/imagination.md)
TILE = 256


def _jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _tile(img: np.ndarray) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB").resize((TILE, TILE)))


def decode_frame(payload: dict[str, Any], offset: int = NEXT_REPLAN_OFFSET) -> tuple[np.ndarray, int]:
    """The imagined frame closest to ``offset`` from a server payload ``{"meta", "arrays"}`` (JPEGs packed back to back,
    ``meta["lengths"]["frames"]``), and its offset."""
    from PIL import Image

    meta, arrays = payload["meta"], payload["arrays"]
    lengths = list(meta["lengths"]["frames"])
    offsets = list(meta.get("frame_offsets") or range(len(lengths)))
    i = int(np.argmin([abs(o - offset) for o in offsets]))
    raw = np.asarray(arrays["frames"], dtype=np.uint8).tobytes()
    start = sum(lengths[:i])
    frame = np.asarray(Image.open(io.BytesIO(raw[start: start + lengths[i]])).convert("RGB"))
    return frame, int(offsets[i])


class ImaginationPanel:
    def __init__(self, runner: Any, backends: dict[str, Any], fps: float = 30.0, history: int = 48):
        self.runner, self.backends, self.fps = runner, backends, fps  # backends: target name -> RemoteBackend
        self._seen: dict[str, str] = {}
        self._history: deque[tuple[float, dict[str, np.ndarray]]] = deque(maxlen=history)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imagination")
        self._lock = threading.Lock()
        self.latest: dict[str, Any] | None = None
        self.error = ""
        self.fetched = 0

    def on_frame(self, obs: dict) -> None:
        """Control thread, once per HMI frame: remember the camera views, start fetching any new chunk's imagination."""
        now = time.monotonic()
        views = {k: _tile(obs[k]) for k in (IMG_HEAD, *WRIST_KEY.values()) if obs.get(k) is not None}
        with self._lock:
            self._history.append((now, views))
        for name, backend in self.backends.items():
            ts = self.runner._targets.get(name)  # noqa: SLF001 - the runner keeps each target's last response
            resp = getattr(ts, "last_response", None)
            iid = (resp.info or {}).get("imagination_id") if resp is not None else None
            if iid and self._seen.get(name) != iid and name in self.runner.active_targets():
                self._seen[name] = iid
                self._pool.submit(self._fetch, name, backend, iid, now)

    def _fetch(self, name: str, backend: Any, iid: str, seen_at: float) -> None:
        try:
            frame, offset = decode_frame(backend.fetch_imagination(iid, wait_s=10.0))
        except Exception as exc:  # the panel is a view: never let it disturb the run
            self.error = f"{type(exc).__name__}: {str(exc)[:160]}"
            return
        with self._lock:
            self.latest = {"target": name, "arm": name.split(":")[0], "id": iid, "offset": offset,
                           "t_target": seen_at + offset / self.fps, "imagined": _jpeg(frame),
                           "t_fetched": time.monotonic()}
            self.fetched += 1
            self.error = ""

    def actual(self) -> bytes | None:
        """The real head | wrist pair recorded closest to the imagined moment (the newest one until it has passed)."""
        with self._lock:
            latest, history = self.latest, list(self._history)
        if latest is None or not history:
            return None
        _, views = min(history, key=lambda h: abs(h[0] - latest["t_target"]))
        head, wrist = views.get(IMG_HEAD), views.get(WRIST_KEY[latest["arm"]])
        if head is None:
            return None
        return _jpeg(np.hstack([head, wrist if wrist is not None else np.zeros_like(head)]))

    def frame(self, name: str) -> bytes | None:
        if name == "imagined":
            with self._lock:
                return None if self.latest is None else self.latest["imagined"]
        if name == "actual":
            return self.actual()
        return None

    def state(self) -> dict[str, Any]:
        with self._lock:
            latest = self.latest
        out: dict[str, Any] = {"enabled": True, "ready": latest is not None, "fetched": self.fetched,
                               "error": self.error}
        if latest is not None:
            out.update(target=latest["target"], arm=latest["arm"], offset=latest["offset"],
                       ahead_s=round(latest["offset"] / self.fps, 2),
                       age_s=round(time.monotonic() - latest["t_fetched"], 1))
        return out

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
