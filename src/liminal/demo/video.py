"""Demonstration folders: frames (or a video), an optional transcript, optional manual keyframes.

A demo folder holds

* ``frames/*.png|jpg`` and/or ``video.mp4`` (frames win when both exist),
* ``transcript.txt`` (optional narration, as in Video-to-BT's transcribed audio),
* ``keyframes.json`` (optional; a list of frame stems like ``"f002"`` or timestamps in seconds),
* ``demo.yaml`` (``name``, ``scenario`` = what it was recorded against, ``notes``, ``fps``),
* ``annotation.yaml`` (optional ground truth of what the human did; only the scripted model reads it).

Frames get stable ids ``demo:<name>:f012`` (the number is the source frame index). Which selection
strategy produced them is recorded on the ``Demo``: the paper used *manual* keyframes, so a run with
``strategy="sampled"`` must never be reported as if it were.

Frames are historical. Nothing here feeds ``WorldState``; they are only prompts and provenance.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liminal.orchestration.procedure import ProcedureError

STRATEGIES = ("sampled", "manual")
IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
MAX_VIDEO_FRAMES = 240
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_FRAME_FPS = 1.0


class DemoError(ProcedureError):
    """A demo folder or a demo-derived model reply is unusable."""


@dataclass(frozen=True)
class DemoFrame:
    frame_id: str  # demo:<name>:f012
    index: int  # position in the source (video frame number at ``fps``, or sorted file position)
    t_s: float
    media_type: str
    data: bytes = field(repr=False)
    path: str = ""  # source file for frame folders; empty for video-decoded frames

    def jpeg(self) -> bytes:
        """JPEG bytes for the model call (``ClaudeModel.complete`` labels every image image/jpeg)."""
        if self.media_type == "image/jpeg":
            return self.data
        if shutil.which("ffmpeg") is None:
            raise DemoError("ffmpeg is needed to convert non-JPEG demo frames for the model")
        run = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", "3", "pipe:1"],
            input=self.data, capture_output=True, check=False)
        if run.returncode != 0 or not run.stdout:
            raise DemoError(f"ffmpeg could not convert {self.frame_id}: {run.stderr.decode(errors='replace')[:200]}")
        return run.stdout


@dataclass(frozen=True)
class Demo:
    name: str
    path: str
    frames: tuple[DemoFrame, ...]
    strategy: str  # "sampled" | "manual": how ``frames`` were chosen
    source: str  # "frames" | "video"
    transcript: str = ""
    scenario: str = ""  # what the demo was recorded against (informational; never trusted as current scene)
    notes: str = ""
    fps: float = DEFAULT_FRAME_FPS
    annotation: dict[str, Any] | None = None

    def frame_ids(self) -> tuple[str, ...]:
        return tuple(f.frame_id for f in self.frames)

    def resolve_frame(self, ref: Any) -> str:
        """Accept ``demo:<name>:f002``, ``f002`` or an int index; return the full id of a LOADED frame."""
        ids = self.frame_ids()
        if isinstance(ref, int) and not isinstance(ref, bool):
            ref = f"f{ref:03d}"
        if not isinstance(ref, str):
            raise DemoError(f"frame reference {ref!r} is not a string")
        full = ref if ref.startswith("demo:") else f"demo:{self.name}:{ref}"
        if full not in ids:
            raise DemoError(f"frame {ref!r} is not one of the loaded frames {list(ids)}")
        return full

    def nearest_frame(self, ref: Any) -> str:
        """Like ``resolve_frame`` but falls back to the loaded frame with the closest index (scripted model)."""
        try:
            return self.resolve_frame(ref)
        except DemoError:
            m = re.search(r"f(\d+)$", str(ref))
            if not m:
                raise
            want = int(m.group(1))
            return min(self.frames, key=lambda f: (abs(f.index - want), f.index)).frame_id

    def model_images(self) -> list[bytes]:
        return [f.jpeg() for f in self.frames]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path, "strategy": self.strategy, "source": self.source,
                "scenario": self.scenario, "notes": self.notes, "has_transcript": bool(self.transcript),
                "frames": [{"id": f.frame_id, "index": f.index, "t_s": round(f.t_s, 3),
                            "media_type": f.media_type} for f in self.frames]}


def _stem_index(stem: str) -> int | None:
    m = re.fullmatch(r"f(\d+)", stem)
    return int(m.group(1)) if m else None


def _sample(items: list[Any], k: int) -> list[Any]:
    """Evenly spaced, deterministic, always including the first and last item."""
    n = len(items)
    if k < 1:
        raise DemoError("max_frames must be at least 1")
    if n <= k:
        return list(items)
    if k == 1:
        return [items[0]]
    picks = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in picks]


def _read_frame_folder(folder: Path, fps: float) -> list[tuple[int, float, str, bytes, str, str]]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [(i, i / fps, IMAGE_SUFFIXES[p.suffix.lower()], p.read_bytes(), str(p), p.stem)
            for i, p in enumerate(files)]


def _decode_video(video: Path, fps: float) -> list[tuple[int, float, str, bytes, str, str]]:
    if shutil.which("ffmpeg") is None:
        raise DemoError("video demos need the ffmpeg CLI (not found on PATH)")
    with tempfile.TemporaryDirectory(prefix="liminal-demo-") as tmp:
        run = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps}", "-frames:v", str(MAX_VIDEO_FRAMES),
             "-q:v", "3", str(Path(tmp) / "%04d.jpg")], capture_output=True, check=False)
        if run.returncode != 0:
            raise DemoError(f"ffmpeg could not decode {video.name}: {run.stderr.decode(errors='replace')[:300]}")
        files = sorted(Path(tmp).glob("*.jpg"))
        return [(i, i / fps, "image/jpeg", p.read_bytes(), "", f"f{i:03d}") for i, p in enumerate(files)]


def _manual_picks(raw: Any, all_frames: list[tuple[int, float, str, bytes, str, str]], fps: float,
                  max_frames: int) -> list[tuple[int, float, str, bytes, str, str]]:
    entries = raw.get("frames") if isinstance(raw, dict) else raw
    if not isinstance(entries, list) or not entries:
        raise DemoError("keyframes.json must be a non-empty list (or {\"frames\": [...]}) of frame stems/seconds")
    if len(entries) > max_frames:
        raise DemoError(f"keyframes.json lists {len(entries)} frames but max_frames={max_frames}")
    by_stem = {f[5]: f for f in all_frames}
    picked = {}
    for entry in entries:
        if isinstance(entry, str):
            if entry not in by_stem:
                raise DemoError(f"keyframes.json: no frame {entry!r} (have {sorted(by_stem)[:6]}...)")
            frame = by_stem[entry]
        elif isinstance(entry, (int, float)) and not isinstance(entry, bool) and entry >= 0:
            idx = round(entry * fps)
            if idx >= len(all_frames):
                raise DemoError(f"keyframes.json: time {entry}s is past the end of the demo")
            frame = all_frames[idx]
        else:
            raise DemoError(f"keyframes.json: unsupported entry {entry!r}")
        picked[frame[0]] = frame
    return [picked[i] for i in sorted(picked)]


def load_demo(path: str | Path, *, max_frames: int = 8, strategy: str = "sampled") -> Demo:
    """Load a demo folder. ``strategy="manual"`` requires ``keyframes.json``; ``"sampled"`` ignores it."""
    import yaml

    if strategy not in STRATEGIES:
        raise DemoError(f"strategy must be one of {STRATEGIES}")
    folder = Path(path)
    if not folder.is_dir():
        raise DemoError(f"demo folder not found: {folder}")
    meta: dict[str, Any] = {}
    if (folder / "demo.yaml").exists():
        meta = yaml.safe_load((folder / "demo.yaml").read_text()) or {}
        if not isinstance(meta, dict):
            raise DemoError("demo.yaml must be a mapping")
    name = str(meta.get("name") or folder.name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", name):
        raise DemoError(f"demo name {name!r} must be 1-40 characters of [A-Za-z0-9_.-]")

    frames_dir, video = folder / "frames", folder / "video.mp4"
    if frames_dir.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in frames_dir.iterdir()):
        source, fps = "frames", float(meta.get("fps", DEFAULT_FRAME_FPS))
    elif video.exists():
        source, fps = "video", float(meta.get("fps", DEFAULT_VIDEO_FPS))
    else:
        raise DemoError(f"{folder} has neither frames/*.png|jpg nor video.mp4")
    if fps <= 0:
        raise DemoError("fps must be positive")
    all_frames = _read_frame_folder(frames_dir, fps) if source == "frames" else _decode_video(video, fps)
    if not all_frames:
        raise DemoError(f"no frames could be read from {folder}")

    if strategy == "manual":
        keyframes = folder / "keyframes.json"
        if not keyframes.exists():
            raise DemoError("strategy='manual' needs keyframes.json in the demo folder")
        chosen = _manual_picks(json.loads(keyframes.read_text()), all_frames, fps, max_frames)
    else:
        chosen = _sample(all_frames, max_frames)

    frames = tuple(DemoFrame(f"demo:{name}:f{idx:03d}", idx, t, media, data, src)
                   for idx, t, media, data, src, _ in chosen)
    transcript = ""
    if (folder / "transcript.txt").exists():
        transcript = (folder / "transcript.txt").read_text().strip()[:4000]
    annotation = None
    if (folder / "annotation.yaml").exists():
        annotation = yaml.safe_load((folder / "annotation.yaml").read_text()) or {}
    return Demo(name=name, path=str(folder), frames=frames, strategy=strategy, source=source, transcript=transcript,
                scenario=str(meta.get("scenario", "")), notes=str(meta.get("notes", "")), fps=fps,
                annotation=annotation)
