"""Imagination next to actions in the policy server (``flux.remote`` / ``flux.backends``).

A request may ask for the chunk's imagined frames with ``imagine``:

* ``"sync"``: decode before answering; the frames travel in the same response (adds the decode time to the
  round trip).
* ``"async"``: answer with the actions at once; a server thread decodes as soon as the model is idle; the
  client fetches the frames later by id.
* ``"lazy"``: answer at once and keep only the latents; decode when the client fetches (e.g. after an attempt,
  while the arm is parked). Zero extra GPU work while the robot moves.

Every mode returns exactly the actions of a plain request: the chunk comes from the same
``predict_action_chunk`` call (``Imaginer.run``), and capturing the video stream only copies 540 KB to the host.
Decoding takes the backend's model lock, so it never runs concurrently with a prediction on the same GPU.

Frames travel as JPEG (quality 90, about 30 KB per 256x512 canvas). The client turns a payload back into an
``Imagination`` with ``imagination_from_payload`` without torch.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from typing import Any

import numpy as np

from .predict_with_video import Candidate, Imagination, Imaginer, LatentDecoder, VideoSpec, video_vae_of

MODES = ("sync", "async", "lazy")


def jpeg_encode(frames: np.ndarray, quality: int = 90) -> tuple[np.ndarray, list[int]]:
    from PIL import Image

    chunks = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
        chunks.append(buf.getvalue())
    return np.frombuffer(b"".join(chunks), dtype=np.uint8), [len(c) for c in chunks]


def jpeg_decode(blob: np.ndarray, lengths: list[int]) -> np.ndarray:
    from PIL import Image

    raw, frames, start = np.asarray(blob, dtype=np.uint8).tobytes(), [], 0
    for n in lengths:
        frames.append(np.asarray(Image.open(io.BytesIO(raw[start : start + n])).convert("RGB"), dtype=np.uint8))
        start += n
    return np.stack(frames)


def spec_from_json(data: dict[str, Any]) -> VideoSpec:
    d = dict(data)
    for key in ("latent_hw", "canvas_hw", "camera_keys"):
        d[key] = tuple(d[key])
    return VideoSpec(**d)


class _Entry:
    def __init__(self, record: Imagination, meta: dict[str, Any], mode: str):
        self.record, self.meta, self.mode = record, meta, mode
        self.arrays: dict[str, np.ndarray] | None = None
        self.error: str | None = None
        self.done = threading.Event()


class ImagineService:
    """Per loaded model: capture each requested chunk's imagination, decode it now, soon or on demand.

    ``lock`` is the backend's model lock. ``predict`` must be called with it held (the backend does); decoding in
    the worker or in ``fetch`` acquires it.
    """

    def __init__(self, policy, lock: threading.Lock, *, decode_device=None, quality: int = 90,
                 max_items: int = 256):
        self.imaginer = Imaginer(policy, decode="none")
        self.decoder = LatentDecoder(video_vae_of(policy), device=decode_device)
        self.lock = lock
        self.quality = quality
        self.max_items = max_items
        self.items: OrderedDict[str, _Entry] = OrderedDict()
        self._items_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="imagine-serve-decoder", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ called by the backend, model lock held
    def predict(self, prepared, mode: str):
        """``(normalized_actions, meta, arrays)``; ``arrays`` holds JPEG frames in ``sync`` mode, else nothing."""
        if mode not in MODES:
            raise ValueError(f"imagine must be one of {MODES}, got {mode!r}")
        t0 = time.perf_counter()
        actions, records = self.imaginer.run(prepared)
        sample_s = time.perf_counter() - t0
        if len(records) != 1:
            raise ValueError(f"imagination serves one window per request, got {len(records)}")
        rec = records[0]
        meta = {
            "id": uuid.uuid4().hex[:16],
            "mode": mode,
            "spec": asdict(rec.spec),
            "frame_offsets": list(rec.frame_offsets),
            "task": rec.task,
            "seed": rec.seed,
            "predict_s": sample_s,
            "t_request": rec.t_request,
        }
        entry = _Entry(rec, meta, mode)
        arrays: dict[str, np.ndarray] = {}
        if mode == "sync":
            self._decode(entry)  # entry.meta is meta: it gains decode_s and the JPEG lengths
            if entry.error is not None:
                raise RuntimeError(entry.error)
            arrays = entry.arrays or {}
        else:
            self._store(entry)
            if mode == "async":
                self._queue.put(meta["id"])
        return actions, meta, arrays

    # ------------------------------------------------------------------ fetching
    def fetch(self, imagination_id: str, wait_s: float = 30.0) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        with self._items_lock:
            entry = self.items.get(imagination_id)
        if entry is None:
            raise KeyError(f"unknown or expired imagination {imagination_id!r}")
        if not entry.done.is_set():
            if entry.mode == "lazy":
                with self.lock:
                    if not entry.done.is_set():
                        self._decode(entry)
            elif not entry.done.wait(wait_s):
                raise TimeoutError(f"imagination {imagination_id} not decoded within {wait_s} s")
        if entry.error is not None:
            raise RuntimeError(entry.error)
        return entry.meta, entry.arrays or {}

    # ------------------------------------------------------------------ internals
    def _store(self, entry: _Entry) -> None:
        with self._items_lock:
            self.items[entry.meta["id"]] = entry
            while len(self.items) > self.max_items:
                self.items.popitem(last=False)

    def _decode(self, entry: _Entry) -> None:
        """Decode and JPEG-encode one entry. Caller holds the model lock."""
        try:
            t0 = time.perf_counter()
            self.decoder.decode_record(entry.record)
            decode_s = time.perf_counter() - t0
            frames, lengths = jpeg_encode(entry.record.frames, self.quality)
            arrays = {"frames": frames}
            meta = {"decode_s": decode_s, "lengths": {"frames": lengths}}
            if entry.record.cond_frames is not None:
                arrays["cond"], meta["lengths"]["cond"] = jpeg_encode(entry.record.cond_frames, self.quality)
            entry.arrays = arrays
            entry.meta.update(meta)
            entry.record.latents = None  # frames are kept as JPEG; free the latents
        except Exception as exc:  # reported to the client on fetch
            entry.error = f"{type(exc).__name__}: {exc}"
        finally:
            entry.done.set()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            with self._items_lock:
                entry = self.items.get(item)
            if entry is None or entry.done.is_set():
                continue
            with self.lock:
                if not entry.done.is_set():
                    self._decode(entry)

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5)


def imagination_from_payload(meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> Imagination:
    """Client side: rebuild an ``Imagination`` (frames, offsets, spec) from a server payload. No torch needed."""
    rec = Imagination(
        spec=spec_from_json(meta["spec"]),
        task=meta.get("task", ""),
        seed=int(meta.get("seed", 0)),
        latents=None,
        t_request=float(meta.get("t_request", 0.0)),
        candidates=[Candidate(seed=int(meta.get("seed", 0)), sample_s=float(meta.get("predict_s", 0.0)))],
    )
    lengths = meta.get("lengths", {})
    if "frames" in arrays:
        rec.frames = jpeg_decode(arrays["frames"], lengths["frames"])
    if "cond" in arrays:
        rec.cond_frames = jpeg_decode(arrays["cond"], lengths["cond"])
    rec.timings = {k: float(meta[k]) for k in ("predict_s", "decode_s") if k in meta}
    return rec
