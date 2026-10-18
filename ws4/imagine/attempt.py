"""Per-attempt imagination log for the real-arm runner, and the ACTUAL | IMAGINED video made from it.

The runner (ws4/single_arm/run_single_arm.py, ``--imagine lazy|async``) calls three things:

    irec = AttemptImagination(out)                      # before the control loop
    irec.tick(i, robot.last_raw, replanned, response)   # once per tick: queues the frames, no work in the loop
    irec.finish(backend)                                # after the attempt, arm parked: fetch, save, optionally render

Alignment: the runner requests a chunk at tick k, holds the arm still while waiting and starts executing it at
the tick reported as ``replanned`` (``DualArmRunner`` with ``async_inference=True``). The first action leaves at
that tick, so imagined offset ``o`` shows tick ``replanned + o``; ``Imagination.step`` is set to that tick.

Saved under ``<attempt>/imagination/``:
- ``actual.npz``: per tick, the scene | wrist canvas as the model sees it (each camera resized into its half of
  the 256x512 canvas), JPEG-encoded, plus tick indices and times.
- ``chunk_XXX.npz``: one ``Imagination`` per chunk (frames, offsets, step), readable with ``load_imagination``.
- ``imagination.json``: chunk ticks, ids, round trips, server predict and decode times, failures.

Render later (or with ``finish(render=True)``):

    python -m imagine.attempt <attempt_dir> [--gif]
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .compose import ActualFrame, make_canvas, render_run
from .predict_with_video import load_imagination
from .serving import imagination_from_payload, jpeg_decode, jpeg_encode


class AttemptImagination:
    """Never raises from ``tick`` and never blocks the control loop: frames go through a bounded queue (dropped,
    and counted, when the encoder falls behind) and the first error disables recording with one printed line."""

    def __init__(self, out: str | Path, *, canvas_hw: tuple[int, int] = (256, 512), quality: int = 85,
                 fetch_wait_s: float = 30.0, max_queue: int = 90):
        self.dir = Path(out) / "imagination"
        self.canvas_hw = canvas_hw
        self.quality = quality
        self.fetch_wait_s = fetch_wait_s
        self.chunks: list[dict[str, Any]] = []
        self.dropped = 0
        self.disabled: str | None = None
        self._frames: list[tuple[int, float, bytes]] = []
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._t0 = time.monotonic()
        self._worker = threading.Thread(target=self._encode_loop, name="imagine-attempt-frames", daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------ in the control loop (cheap)
    def tick(self, step: int, raw: dict[str, Any], replanned: bool = False, response: Any = None) -> None:
        if self.disabled:
            return
        try:
            wrist = raw.get("wrist")
            if wrist is not None:
                scene = raw.get("scene")
                item = (step, time.monotonic() - self._t0, None if scene is None else np.array(scene), np.array(wrist))
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    self.dropped += 1
            if replanned and response is not None and getattr(response, "imagination", None) is not None:
                self.chunks.append({
                    "step": int(step), "t": time.monotonic() - self._t0, "imagination": response.imagination,
                    "round_trip_s": float(getattr(response, "latency_s", float("nan"))),
                    "server_latency_s": (response.info or {}).get("server_latency_s"),
                })  # fmt: skip
        except Exception as exc:  # never let the recorder touch the control loop
            self.disabled = f"{type(exc).__name__}: {exc}"
            print(f"imagination recording disabled for this attempt: {self.disabled}", flush=True)

    def _encode_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                step, t, scene, wrist = item
                if scene is None:
                    scene = np.zeros_like(wrist)
                canvas = make_canvas(scene, wrist, self.canvas_hw)
                blob, _ = jpeg_encode(canvas[None], self.quality)
                self._frames.append((step, t, blob.tobytes()))
            except Exception:  # a bad frame is dropped, never the thread
                self.dropped += 1

    # ------------------------------------------------------------------ after the attempt
    def finish(self, backend: Any = None, *, render: bool = False, task: str = "",
               budget_s: float | None = None) -> dict[str, Any]:
        """Save the actual frames, fetch every chunk's imagined frames (``lazy`` ones are decoded now) and write
        ``imagination.json``. Stops fetching when ``budget_s`` runs out or after the first server error; the
        remaining chunks are listed as ``pending`` and can be fetched later with ``python -m imagine.attempt fetch``
        while the server still holds them (its last 256 chunks)."""
        deadline = None if budget_s is None else time.monotonic() + budget_s
        try:
            self._queue.put(None, timeout=2.0)
        except queue.Full:
            pass
        self._worker.join(timeout=5.0)
        self.dir.mkdir(parents=True, exist_ok=True)
        frames = list(self._frames)
        blobs = [f[2] for f in frames]
        np.savez(self.dir / "actual.npz", step=np.array([f[0] for f in frames], dtype=np.int64),
                 t=np.array([f[1] for f in frames]), lengths=np.array([len(b) for b in blobs], dtype=np.int64),
                 jpeg=np.frombuffer(b"".join(blobs), dtype=np.uint8))  # fmt: skip
        summary: dict[str, Any] = {"chunks": [], "ticks": len(blobs), "task": task, "dropped_frames": self.dropped,
                                   "disabled": self.disabled}  # fmt: skip
        t0, stop = time.perf_counter(), None
        for k, chunk in enumerate(self.chunks):
            payload = chunk["imagination"]
            row = {"index": k, "step": chunk["step"], "id": payload["meta"].get("id"),
                   "round_trip_s": chunk["round_trip_s"], "server_latency_s": chunk["server_latency_s"]}  # fmt: skip
            if stop is None and deadline is not None and time.monotonic() > deadline:
                stop = "budget"
            if stop is not None and "frames" not in payload["arrays"]:
                row["pending"] = stop
                row["meta"] = payload["meta"]
                summary["chunks"].append(row)
                continue
            try:
                if "frames" not in payload["arrays"]:
                    wait = self.fetch_wait_s if deadline is None else max(1.0, min(self.fetch_wait_s,
                                                                                   deadline - time.monotonic()))
                    payload = backend.fetch_imagination(payload["meta"]["id"], wait)
                self._save_chunk(payload, chunk["step"], k, row)
            except Exception as exc:  # a lost imagination never fails the attempt log
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["meta"] = payload["meta"]
                stop = "server error"
            summary["chunks"].append(row)
        summary["fetch_s"] = time.perf_counter() - t0
        summary["pending"] = sum(1 for c in summary["chunks"] if "pending" in c or "error" in c)
        (self.dir / "imagination.json").write_text(json.dumps(summary, indent=2, default=float))
        if render and summary["chunks"] and not summary["pending"]:
            summary["video"] = str(render_attempt(self.dir.parent, task=task))
        return summary

    def _save_chunk(self, payload: dict, step: int, k: int, row: dict) -> None:
        save_chunk(self.dir, payload, step, k, row)


def save_chunk(out_dir: Path, payload: dict, step: int, k: int, row: dict) -> None:
    """One fetched imagination -> ``<out_dir>/chunk_XXX.npz``; timings go into ``row``."""
    rec = imagination_from_payload(payload["meta"], payload["arrays"])
    rec.step, rec.index = step, k
    rec.save(Path(out_dir) / f"chunk_{k:03d}.npz")
    row.update({"predict_s": rec.timings.get("predict_s"), "decode_s": rec.timings.get("decode_s")})


def finish_with_timeout(irec: AttemptImagination, backend: Any, *, timeout_s: float = 30.0, render: bool = False,
                        task: str = "") -> dict[str, Any] | None:
    """``irec.finish`` on a daemon thread with a hard timeout, for the runner: whatever happens it returns within
    ``timeout_s`` (plus a few seconds), prints one line and never raises."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["summary"] = irec.finish(backend, render=render, task=task, budget_s=max(1.0, timeout_s - 5.0))
        except Exception as exc:
            box["error"] = f"{type(exc).__name__}: {exc}"

    t0 = time.monotonic()
    thread = threading.Thread(target=run, name="imagine-attempt-finish", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        print(f"imagination: still fetching after {timeout_s:.0f} s; continuing without waiting", flush=True)
        return None
    if "error" in box:
        print(f"imagination: not saved ({box['error']})", flush=True)
        return None
    s = box["summary"]
    done = len(s["chunks"]) - s["pending"]
    print(f"imagination: {done}/{len(s['chunks'])} chunks saved in {irec.dir} ({time.monotonic() - t0:.1f} s)"
          + (f", {s['pending']} pending: python -m imagine.attempt fetch {irec.dir.parent}" if s["pending"] else ""),
          flush=True)  # fmt: skip
    return s


def fetch_pending(attempt_dir: str | Path, backend: Any) -> dict[str, Any]:
    """Fetch the chunks an attempt left pending (the server keeps its last 256), then rewrite ``imagination.json``."""
    d = Path(attempt_dir) / "imagination"
    summary = json.loads((d / "imagination.json").read_text())
    for row in summary["chunks"]:
        if "pending" not in row and "error" not in row:
            continue
        try:
            payload = backend.fetch_imagination(row["id"], 60.0)
            save_chunk(d, payload, row["step"], row["index"], row)
            row.pop("pending", None)
            row.pop("error", None)
            row.pop("meta", None)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    summary["pending"] = sum(1 for c in summary["chunks"] if "pending" in c or "error" in c)
    (d / "imagination.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def load_attempt(attempt_dir: str | Path) -> tuple[list, list[ActualFrame]]:
    """``(imaginations, actual_frames)`` saved by ``AttemptImagination``."""
    d = Path(attempt_dir) / "imagination"
    data = np.load(d / "actual.npz")
    frames = jpeg_decode(data["jpeg"], data["lengths"].tolist()) if len(data["lengths"]) else []
    actual = [ActualFrame(img, step=int(s), t=float(t)) for img, s, t in zip(frames, data["step"], data["t"],
                                                                             strict=True)]  # fmt: skip
    records = [load_imagination(p) for p in sorted(d.glob("chunk_*.npz"))]
    return records, actual


def offset_errors(attempt_dir: str | Path, max_offset: int = 42) -> dict[str, dict[int, float]]:
    """Mean imagined-vs-actual distance (``score.frame_distance``, 0 = identical) per control offset, for the scene
    half, the wrist half and the whole canvas. Lower is a more accurate imagination; compare packages on the same
    replayed takes."""
    from .score import frame_distance

    records, actual = load_attempt(attempt_dir)
    by_step = {a.step: a.image for a in actual}
    sums: dict[str, dict[int, list[float]]] = {"scene": {}, "wrist": {}, "canvas": {}}
    for rec in records:
        for off in rec.frame_offsets:
            if off > max_offset or rec.step + off not in by_step:
                continue
            real, imag = by_step[rec.step + off], rec.frame_at(off)
            half = imag.shape[1] // 2
            parts = {"scene": (real[:, :half], imag[:, :half]), "wrist": (real[:, half:], imag[:, half:]),
                     "canvas": (real, imag)}  # fmt: skip
            for name, (a, b) in parts.items():
                sums[name].setdefault(off, []).append(frame_distance(a, b, size=(64, 64 if name != "canvas" else 128)))
    return {name: {o: float(np.mean(v)) for o, v in sorted(d.items())} for name, d in sums.items()}


def render_attempt(attempt_dir: str | Path, *, task: str = "", gif: bool = False, out: str | Path | None = None):
    records, actual = load_attempt(attempt_dir)
    if not records:
        raise RuntimeError(f"no imagined chunks in {attempt_dir}")
    if not task:
        task = records[0].task
    name = "imagined_vs_actual" + (".gif" if gif else ".mp4")
    path = Path(out) if out else Path(attempt_dir) / "imagination" / name
    surprise = render_run(records, actual, path, task=task)
    np.save(path.with_suffix(".surprise.npy"), surprise)
    return path


def main(argv: list[str] | None = None) -> None:
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "fetch":
        p = argparse.ArgumentParser(prog="python -m imagine.attempt fetch",
                                    description="Fetch the imagined chunks an attempt left pending.")
        p.add_argument("attempt_dir")
        p.add_argument("--server", default="http://127.0.0.1:8765")
        p.add_argument("--package", default="so101")
        p.add_argument("--token-file", default=None, help="else $FLUX_TOKEN")
        args = p.parse_args(argv[1:])
        import os

        from flux.remote import RemoteBackend

        token = Path(args.token_file).read_text().strip() if args.token_file else os.environ.get("FLUX_TOKEN")
        summary = fetch_pending(args.attempt_dir, RemoteBackend(args.server, args.package, token=token))
        print(f"{len(summary['chunks']) - summary['pending']}/{len(summary['chunks'])} chunks saved")
        return
    p = argparse.ArgumentParser(description="Render ACTUAL | IMAGINED for a runner attempt saved with --imagine "
                                            "(or: python -m imagine.attempt fetch <attempt_dir> ...).")
    p.add_argument("attempt_dir")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--task", default="")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    print(render_attempt(args.attempt_dir, task=args.task, gif=args.gif, out=args.out))


if __name__ == "__main__":
    main()
