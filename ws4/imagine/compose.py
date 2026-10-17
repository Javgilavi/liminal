"""Imagined | actual side-by-side frames, GIF/MP4 writers and contact sheets for the UI and the demo video.

Alignment: an ``Imagination`` made at control step ``s`` with frame offsets ``o`` predicts the camera view at
step ``s + o``. ``align`` pairs every logged actual frame with the imagined frame for the same step
(``mode="same_time"``, divergence = surprise) or for a fixed number of steps ahead (``mode="lookahead"``,
"what the robot is about to do"). Records without a step index fall back to wall-clock alignment from
``t_return`` (the robot starts executing when the chunk is handed back; inference is synchronous).
PIL and numpy only; MP4 needs imageio-ffmpeg or OpenCV, imported lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .score import frame_distance

YELLOW = (250, 204, 21)
CYAN = (34, 211, 238)
WHITE = (240, 240, 240)
GREY = (120, 120, 120)
BG = (18, 18, 22)


def make_canvas(scene: np.ndarray, wrist: np.ndarray, canvas_hw: tuple[int, int] = (256, 512)) -> np.ndarray:
    """Actual cameras in the model's ``side_by_side`` layout: each resized to (H, W/2), scene left
    (lerobot f3/packing.py:162-171), so actual and imagined panels are directly comparable."""
    from PIL import Image

    h, w = canvas_hw
    halves = [np.asarray(Image.fromarray(np.asarray(c, dtype=np.uint8)).resize((w // 2, h), Image.BILINEAR))
              for c in (scene, wrist)]
    return np.concatenate(halves, axis=1)


@dataclass
class ActualFrame:
    image: np.ndarray  # (H, W, 3) uint8, ideally a make_canvas() of the real cameras
    step: int | None = None  # control step (select_action call count) when the frame was captured
    t: float | None = None  # wall clock


@dataclass
class Row:
    actual: np.ndarray
    imagined: np.ndarray | None
    step: int
    chunk: int | None
    offset: int | None
    chunk_start: bool
    surprise: float | None


def _imagined_for(rec, offset: int) -> np.ndarray | None:
    if rec.frames is None:
        return None
    offsets = rec.frame_offsets
    if offset <= 0:
        return rec.current_frame()
    if offset > max(offsets):
        return None  # beyond the imagined horizon
    return rec.frame_at(offset)


def align(records: Sequence, actual: Sequence[ActualFrame], *, mode: str = "same_time", lookahead: int = 0,
          fps: float | None = None, surprise_roi=None) -> list[Row]:
    """Pair each actual frame with the matching imagined frame (see module docstring)."""
    if mode not in ("same_time", "lookahead"):
        raise ValueError("mode must be same_time or lookahead")
    recs = sorted(records, key=lambda r: (r.step if r.step is not None else -1, r.t_return))
    rows: list[Row] = []
    last_chunk = None
    for i, fr in enumerate(actual):
        step = fr.step if fr.step is not None else i
        rec = None
        if fr.step is not None and all(r.step is not None for r in recs):
            started = [r for r in recs if r.step <= fr.step]
            rec = started[-1] if started else None
            offset = None if rec is None else fr.step - rec.step
        else:  # wall-clock alignment
            rate = fps or (recs[0].spec.fps if recs else 30.0)
            started = [r for r in recs if fr.t is not None and r.t_return <= fr.t]
            rec = started[-1] if started else None
            offset = None if rec is None else int(round((fr.t - rec.t_return) * rate))
        imagined = None
        if rec is not None:
            imagined = _imagined_for(rec, offset + (lookahead if mode == "lookahead" else 0))
        chunk = None if rec is None else rec.index
        surprise = None
        if imagined is not None and mode == "same_time" and offset and offset > 0:
            a = fr.image
            if a.shape != imagined.shape:
                from PIL import Image

                a = np.asarray(Image.fromarray(a).resize(imagined.shape[1::-1], Image.BILINEAR))
            surprise = frame_distance(a, imagined, roi=surprise_roi)
        rows.append(Row(fr.image, imagined, step, chunk, offset, chunk is not None and chunk != last_chunk, surprise))
        last_chunk = chunk if chunk is not None else last_chunk
    return rows


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------
def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _fit(img: np.ndarray, height: int) -> np.ndarray:
    from PIL import Image

    h, w = img.shape[:2]
    if h == height:
        return img
    return np.asarray(Image.fromarray(img).resize((round(w * height / h), height), Image.BILINEAR))


@dataclass
class Renderer:
    """Draws one output frame: header, ACTUAL | IMAGINED panels, chunk progress bar and surprise trace."""

    panel_height: int = 256
    fps: float = 30.0
    chunk_size: int = 42
    n_action_steps: int = 32
    title: str = "FLUX 3 Action: imagine, then act"
    task: str = ""
    surprise_window: int = 150

    def render(self, row: Row, history: Sequence[float | None] = ()) -> np.ndarray:
        from PIL import Image, ImageDraw

        left = _fit(row.actual, self.panel_height)
        right = _fit(row.imagined, self.panel_height) if row.imagined is not None else np.zeros_like(left)
        if right.shape[1] != left.shape[1]:
            right = np.asarray(Image.fromarray(right).resize((left.shape[1], self.panel_height)))
        gap, header, footer = 8, 30, 46
        width = left.shape[1] * 2 + gap * 3
        canvas = Image.new("RGB", (width, header + self.panel_height + footer + gap), BG)
        canvas.paste(Image.fromarray(left), (gap, header))
        x_right = gap * 2 + left.shape[1]
        canvas.paste(Image.fromarray(right), (x_right, header))
        draw = ImageDraw.Draw(canvas)
        f_big, f_small = _font(15), _font(12)
        title = self.title + (f"   |   task: {self.task}" if self.task else "")
        draw.text((gap, 7), title, fill=WHITE, font=f_big)
        t = row.step / self.fps
        draw.text((gap + 6, header + 6), f"ACTUAL  t={t:5.2f} s", fill=WHITE, font=f_small, stroke_width=2,
                  stroke_fill=BG)
        if row.imagined is not None and row.offset is not None:
            label = f"IMAGINED  chunk {row.chunk}  +{row.offset} steps ({row.offset / self.fps:+.2f} s)"
        elif row.chunk is not None:
            label = "IMAGINED  (beyond the predicted horizon)"
        else:
            label = "IMAGINED  (waiting for the first chunk)"
        draw.text((x_right + 6, header + 6), label, fill=CYAN, font=f_small, stroke_width=2, stroke_fill=BG)
        if row.chunk_start:  # mark the replanning moment
            draw.rectangle([x_right - 3, header - 3, x_right + left.shape[1] + 2, header + self.panel_height + 2],
                           outline=YELLOW, width=3)
        # chunk progress: executed part (n_action_steps) vs imagined-only tail (chunk_size)
        y0 = header + self.panel_height + 8
        bar_w = width - 2 * gap
        draw.rectangle([gap, y0, gap + bar_w, y0 + 8], outline=GREY)
        exec_w = bar_w * self.n_action_steps / self.chunk_size
        draw.rectangle([gap, y0, gap + exec_w, y0 + 8], fill=(40, 70, 90))
        if row.offset is not None and row.offset >= 0:
            x = gap + bar_w * min(row.offset, self.chunk_size) / self.chunk_size
            draw.rectangle([x - 2, y0 - 2, x + 2, y0 + 10], fill=YELLOW)
        # surprise trace
        y1 = y0 + 14
        vals = [np.nan if v is None else v for v in list(history)[-self.surprise_window:]]
        if vals:
            draw.text((gap, y1), "surprise", fill=GREY, font=f_small)
            x0, w_trace, h_trace = gap + 64, bar_w - 64, 18
            pts = [(x0 + w_trace * i / max(len(vals) - 1, 1), y1 + h_trace - min(v / 0.25, 1.0) * h_trace)
                   for i, v in enumerate(vals) if np.isfinite(v)]
            if len(pts) > 1:
                draw.line(pts, fill=YELLOW, width=2)
        return np.asarray(canvas)

    def render_rows(self, rows: Sequence[Row]) -> list[np.ndarray]:
        out, hist = [], []
        for row in rows:
            hist.append(row.surprise)
            out.append(self.render(row, hist))
        return out


def write_gif(frames: Sequence[np.ndarray], path: str | Path, fps: float = 30.0) -> Path:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=max(int(1000 / fps), 20), loop=0)
    return path


def write_mp4(frames: Sequence[np.ndarray], path: str | Path, fps: float = 30.0, crf: int = 18) -> Path:
    """H.264 (yuv420p, +faststart) so browsers, the app and slide tools play it. Pipes RGB frames to the ``ffmpeg``
    binary (libx264); falls back to imageio's ffmpeg, then to OpenCV ``mp4v`` with a warning (MPEG-4 Part 2, which
    browsers do not play)."""
    import shutil
    import subprocess

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    ph, pw = h + h % 2, w + w % 2  # yuv420p needs even sides

    def even(f: np.ndarray) -> np.ndarray:
        if (ph, pw) == f.shape[:2]:
            return np.ascontiguousarray(f, dtype=np.uint8)
        out = np.zeros((ph, pw, 3), np.uint8)
        out[: f.shape[0], : f.shape[1]] = f
        return out

    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{pw}x{ph}",
               "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
               "-preset", "medium", "-movflags", "+faststart", str(path)]  # fmt: skip
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for f in frames:
                proc.stdin.write(even(f).tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        if proc.wait() == 0:
            return path
        print(f"ffmpeg failed: {proc.stderr.read().decode(errors='replace')[-300:]}")
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p", quality=8,
                                macro_block_size=2, output_params=["-movflags", "+faststart"]) as wr:  # fmt: skip
            for f in frames:
                wr.append_data(even(f))
        return path
    except Exception:
        import cv2

        print(f"WARNING: no H.264 encoder; {path} is MPEG-4 Part 2 (mp4v), which browsers do not play")
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph))
        for f in frames:
            vw.write(cv2.cvtColor(even(f), cv2.COLOR_RGB2BGR))
        vw.release()
        return path


def contact_sheet(rec, actual_by_offset: dict[int, np.ndarray] | None = None,
                  offsets: Sequence[int] = (4, 12, 22, 32, 42), height: int = 128) -> np.ndarray:
    """One chunk as a grid for slides: imagined frames at ``offsets`` on top, actual frames below."""
    from PIL import Image, ImageDraw

    offsets = [o for o in offsets if o <= max(rec.frame_offsets)]
    tiles = [_fit(rec.frame_at(o), height) for o in offsets]
    rows = [tiles]
    if actual_by_offset:
        rows.append([_fit(actual_by_offset[o], height) if o in actual_by_offset else np.zeros_like(tiles[0])
                     for o in offsets])
    w = tiles[0].shape[1]
    label_h, gap = 18, 4
    sheet = Image.new("RGB", (len(offsets) * (w + gap) + gap, len(rows) * (height + label_h + gap) + gap), BG)
    draw = ImageDraw.Draw(sheet)
    for r, (name, tiles_r) in enumerate(zip(("imagined", "actual"), rows, strict=False)):
        y = gap + r * (height + label_h + gap)
        for c, (o, tile) in enumerate(zip(offsets, tiles_r, strict=True)):
            x = gap + c * (w + gap)
            draw.text((x, y), f"{name} +{o / rec.spec.fps:.2f} s", fill=CYAN if r == 0 else WHITE, font=_font(12))
            sheet.paste(Image.fromarray(tile), (x, y + label_h))
    return np.asarray(sheet)


def render_run(records: Sequence, actual: Sequence[ActualFrame], path: str | Path, *, task: str = "",
               mode: str = "same_time", lookahead: int = 0, fps: float | None = None) -> np.ndarray:
    """Align, render and write a GIF (``.gif``) or MP4 (anything else). Returns the per-frame surprise."""
    spec = records[0].spec
    rate = fps or spec.fps
    rows = align(records, actual, mode=mode, lookahead=lookahead, fps=rate)
    renderer = Renderer(fps=rate, chunk_size=spec.chunk_size, n_action_steps=spec.n_action_steps, task=task)
    frames = renderer.render_rows(rows)
    (write_gif if str(path).endswith(".gif") else write_mp4)(frames, path, fps=rate)
    return np.array([np.nan if r.surprise is None else r.surprise for r in rows], dtype=np.float32)
