"""Score imagined futures: cheap pixel and latent checks first, a Claude VLM judge for decision points.

Every scorer follows one call convention so ``Imaginer(scorer=...)`` can rank best-of-N candidates:

    scorer(clips, *, current=None, offsets=None, task="", spec=None) -> list[float]   # higher is better

``clips`` are imagined clips ``(T, H, W, 3)`` uint8 (one per candidate), ``current`` the VAE view of the
observation, ``offsets`` the control step of each clip frame (``VideoSpec.frame_offsets``). A scorer with
``space = "latent"`` receives latents ``(96, n, h, w)`` instead and needs no decoding.

The Claude judge reads ``ANTHROPIC_API_KEY`` from the environment, is never called in tests (inject a fake
``client``) and costs network latency, so use it at subtask boundaries or off the control thread.
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Roi = tuple[float, float, float, float]  # (x0, y0, x1, y1) as fractions of width and height


# ---------------------------------------------------------------------------------------------
# Pixel helpers
# ---------------------------------------------------------------------------------------------
def crop(img: np.ndarray, roi: Roi | None) -> np.ndarray:
    if roi is None:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    return img[int(y0 * h) : max(int(y1 * h), int(y0 * h) + 1), int(x0 * w) : max(int(x1 * w), int(x0 * w) + 1)]


def small(img: np.ndarray, size: tuple[int, int] = (64, 128), roi: Roi | None = None) -> np.ndarray:
    """Crop, then area-resample to ``size`` (h, w); float32 in [0, 1]. Downsampling forgives the VAE's blur."""
    from PIL import Image

    arr = crop(np.asarray(img), roi)
    pil = Image.fromarray(arr.astype(np.uint8)).resize((size[1], size[0]), Image.BOX)
    return np.asarray(pil, dtype=np.float32) / 255.0


def frame_distance(a: np.ndarray, b: np.ndarray, *, size=(64, 128), roi: Roi | None = None) -> float:
    """Mean absolute RGB difference in [0, 1] after crop + downsample. 0 = identical."""
    return float(np.abs(small(a, size, roi) - small(b, size, roi)).mean())


def goal_similarity(frame: np.ndarray, goal: np.ndarray, **kw) -> float:
    return 1.0 - frame_distance(frame, goal, **kw)


def surprise_curve(actual: Sequence[np.ndarray], imagined: Sequence[np.ndarray | None], **kw) -> np.ndarray:
    """Per-step imagined-vs-actual distance (NaN where nothing was imagined). Spikes flag the unexpected."""
    out = np.full(len(actual), np.nan, dtype=np.float32)
    for i, (a, m) in enumerate(zip(actual, imagined, strict=True)):
        if m is not None:
            out[i] = frame_distance(a, m, **kw)
    return out


def select_frame(clip: np.ndarray, offsets: Sequence[int] | None, offset: int | None) -> np.ndarray:
    """Frame of ``clip`` at ``offset`` control steps (nearest); the last frame when ``offset`` is None."""
    if offset is None or offsets is None:
        return clip[-1]
    return clip[int(np.argmin(np.abs(np.asarray(offsets) - offset)))]


class SurpriseMonitor:
    """Failure anticipation during execution: actual frame vs the frame imagined for the same control step.

    ``update`` returns ``(surprise, fired)``. It fires when the mean surprise over the last ``window`` steps
    exceeds ``threshold`` (calibrate on replays of successful runs, e.g. their 95th percentile per offset).
    Offsets below ``min_offset`` are skipped: right after a replan the imagined and actual frames agree trivially.
    On ``fired``, call ``imaginer.replan()`` and/or tell the orchestrator.
    """

    def __init__(self, threshold: float = 0.12, window: int = 5, min_offset: int = 4, roi: Roi | None = None):
        self.threshold, self.window, self.min_offset, self.roi = threshold, window, min_offset, roi
        self.values: list[float] = []

    def update(self, record, actual: np.ndarray, step: int) -> tuple[float | None, bool]:
        if record is None or record.frames is None or record.step is None:
            return None, False
        offset = step - record.step
        if offset < self.min_offset or offset > max(record.frame_offsets):
            return None, False
        imagined = record.frame_at(offset)
        if actual.shape != imagined.shape:
            from PIL import Image

            actual = np.asarray(Image.fromarray(actual).resize(imagined.shape[1::-1], Image.BILINEAR))
        value = frame_distance(actual, imagined, roi=self.roi)
        self.values.append(value)
        recent = self.values[-self.window :]
        return value, len(recent) == self.window and float(np.mean(recent)) > self.threshold

    def reset(self) -> None:
        self.values.clear()


# ---------------------------------------------------------------------------------------------
# Cheap scorers
# ---------------------------------------------------------------------------------------------
@dataclass
class GoalImageScorer:
    """Similarity of the imagined frame at ``offset`` (default: last) to a goal image of the same canvas.

    ``relative=True`` scores progress: similarity gain over the current frame, so candidates are compared by
    how much closer to the goal they imagine getting.
    """

    goal: np.ndarray
    offset: int | None = None
    roi: Roi | None = None
    relative: bool = True
    size: tuple[int, int] = (64, 128)
    space: str = "pixels"

    def __call__(self, clips, *, current=None, offsets=None, task="", spec=None) -> list[float]:
        scores = []
        for clip in clips:
            s = goal_similarity(select_frame(clip, offsets, self.offset), self.goal, size=self.size, roi=self.roi)
            if self.relative and current is not None:
                s -= goal_similarity(current, self.goal, size=self.size, roi=self.roi)
            scores.append(s)
        return scores


@dataclass
class ColorFractionScorer:
    """Fraction of ROI pixels inside an HSV box at the imagined ``offset``: "is the red cube in the bowl?".

    HSV follows PIL (all channels 0..255; red hue wraps, so pass ``hue=(240, 15)`` for red).
    Use ``predicate`` on an actual or imagined frame as a cheap success check for the orchestrator.
    """

    hue: tuple[int, int]
    sat: tuple[int, int] = (80, 255)
    val: tuple[int, int] = (60, 255)
    roi: Roi | None = None
    offset: int | None = None
    threshold: float = 0.02
    space: str = "pixels"

    def fraction(self, frame: np.ndarray) -> float:
        from PIL import Image

        hsv = np.asarray(Image.fromarray(crop(frame, self.roi).astype(np.uint8)).convert("HSV")).astype(np.int32)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = self.hue
        in_hue = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
        mask = in_hue & (s >= self.sat[0]) & (s <= self.sat[1]) & (v >= self.val[0]) & (v <= self.val[1])
        return float(mask.mean())

    def predicate(self, frame: np.ndarray) -> bool:
        return self.fraction(frame) >= self.threshold

    def __call__(self, clips, *, current=None, offsets=None, task="", spec=None) -> list[float]:
        return [self.fraction(select_frame(c, offsets, self.offset)) for c in clips]


@dataclass
class LatentGoalScorer:
    """Cosine similarity between the last predicted latent frame and a goal image's latent. No decoding.

    Build ``goal_latent`` once with ``encode_goal``. UNVERIFIED as a ranking signal on real weights: the last
    future latent mixes 4 frames, the goal latent is a single-frame encoding.
    """

    goal_latent: Any  # (96, h, w) tensor
    space: str = "latent"

    def __call__(self, latents, *, spec=None, task="", cond_latents=None) -> list[float]:
        import torch

        g = torch.as_tensor(self.goal_latent).float().flatten()
        return [float(torch.nn.functional.cosine_similarity(torch.as_tensor(z).float()[:, -1].flatten(), g, dim=0))
                for z in latents]

    @staticmethod
    def encode_goal(vae, canvas: np.ndarray, latent_hw: tuple[int, int], device="cpu"):
        """Goal canvas ``(H, W, 3)`` uint8 (same layout and size as the model canvas) -> ``(96, h, w)``.

        Matches how history snapshots are encoded: one frame, [-1, 1], bf16 (packing.py:462-464)."""
        import torch

        x = torch.as_tensor(canvas).permute(2, 0, 1).float().div(127.5).sub(1)[None, :, None]
        encode = getattr(vae, "encode_task", vae.encode)
        with torch.inference_mode():
            lat = encode(x.to(device=device, dtype=torch.bfloat16))
        return lat[0, :, 0, : latent_hw[0], : latent_hw[1]].float().cpu()


@dataclass
class EmbeddingScorer:
    """Cosine similarity of an image embedding (SigLIP, CLIP, DINO) of the imagined frame to a goal embedding."""

    embed: Callable[[list[np.ndarray]], np.ndarray]  # images -> (N, d)
    goal_embedding: np.ndarray
    offset: int | None = None
    space: str = "pixels"

    def __call__(self, clips, *, current=None, offsets=None, task="", spec=None) -> list[float]:
        e = np.asarray(self.embed([select_frame(c, offsets, self.offset) for c in clips]), dtype=np.float32)
        g = np.asarray(self.goal_embedding, dtype=np.float32).reshape(-1)
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        return list(e @ (g / np.linalg.norm(g)))


def siglip_embedder(model_id: str = "google/siglip2-base-patch16-224", device: str = "cuda"):
    """Image and text embedders from a SigLIP checkpoint (transformers). UNVERIFIED on the cloud box."""
    import torch
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_id).to(device).eval()
    proc = AutoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def images(frames: list[np.ndarray]) -> np.ndarray:
        inputs = proc(images=list(frames), return_tensors="pt").to(device)
        return model.get_image_features(**inputs).float().cpu().numpy()

    @torch.inference_mode()
    def texts(prompts: list[str]) -> np.ndarray:
        inputs = proc(text=prompts, padding="max_length", return_tensors="pt").to(device)
        return model.get_text_features(**inputs).float().cpu().numpy()

    return images, texts


# ---------------------------------------------------------------------------------------------
# Claude VLM judge
# ---------------------------------------------------------------------------------------------
DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = (
    "You evaluate short video predictions made by a robot's world action model. Each prediction is what the "
    "model imagines the cameras will show while it executes its next action chunk. The canvas shows two "
    "cameras side by side: the fixed scene camera on the left and the gripper wrist camera on the right. "
    "Frames are low resolution and slightly blurry; judge the arrangement of objects and the gripper, not "
    "image quality. Be strict about physically implausible predictions: objects appearing, vanishing, "
    "teleporting, or a gripper changing shape."
)

RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "progress": {"type": "integer"},
                    "risk": {"type": "integer"},
                    "plausible": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["label", "progress", "risk", "plausible", "reason"],
                "additionalProperties": False,
            },
        },
        "best_label": {"type": "string"},
    },
    "required": ["candidates", "best_label"],
    "additionalProperties": False,
}

ANTICIPATE_SCHEMA = {
    "type": "object",
    "properties": {
        "success_probability": {"type": "number"},
        "imagined_outcome": {"type": "string"},
        "failure_mode": {"type": "string"},
        "plausible": {"type": "boolean"},
    },
    "required": ["success_probability", "imagined_outcome", "failure_mode", "plausible"],
    "additionalProperties": False,
}

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "matches_imagination": {"type": "boolean"},
        "task_done": {"type": "boolean"},
        "difference": {"type": "string"},
        "recommendation": {"type": "string", "enum": ["continue", "retry", "recover", "ask_human"]},
    },
    "required": ["matches_imagination", "task_done", "difference", "recommendation"],
    "additionalProperties": False,
}


def png_block(img: np.ndarray, max_side: int = 512) -> dict:
    """Image content block (base64 PNG). Vision input costs roughly one token per 28x28 patch, so a 512x256
    canvas is about 170 input tokens."""
    from PIL import Image

    pil = Image.fromarray(np.asarray(img).astype(np.uint8))
    scale = max_side / max(pil.size)
    if scale < 1:
        pil = pil.resize((round(pil.size[0] * scale), round(pil.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


class JudgeRefused(RuntimeError):
    pass


@dataclass
class JudgeResult:
    scores: list[float]
    best: int
    raw: dict = field(default_factory=dict)


def _pick(clip: np.ndarray, offsets: Sequence[int] | None, n: int) -> list[tuple[int | None, np.ndarray]]:
    idx = np.unique(np.linspace(0, len(clip) - 1, n).round().astype(int))
    return [(None if offsets is None else int(offsets[i]), clip[i]) for i in idx]


class ClaudeJudge:
    """Claude as a judge of imagined futures: rank candidates, anticipate success, compare with reality.

    ``model`` defaults to Claude Opus 5.5 with low effort (thinking is always on for that model; effort is the
    latency knob). Server-side refusal fallbacks are enabled by default (``fallbacks="default"``).
    """

    space = "pixels"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        frames_per_candidate: int = 3,
        max_side: int = 512,
        max_tokens: int = 16000,
        timeout: float = 60.0,
        use_fallbacks: bool = True,
        client: Any = None,
    ):
        self.model = model
        self.effort = effort
        self.frames_per_candidate = frames_per_candidate
        self.max_side = max_side
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.use_fallbacks = use_fallbacks
        self._client = client
        self.last_result: dict | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------ transport
    @property
    def client(self):
        if self._client is None:
            import anthropic

            key = os.environ.get("ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout) if key else anthropic.Anthropic(
                timeout=self.timeout
            )
        return self._client

    def request(self, content: list[dict], schema: dict) -> dict:
        """Keyword arguments for ``messages.create``: one user turn, JSON constrained by ``schema``."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        }

    def send(self, kwargs: dict) -> dict:
        if self.use_fallbacks:
            response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.create(**kwargs)
        return self.parse(response)

    @staticmethod
    def parse(response) -> dict:
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise JudgeRefused(str(getattr(details, "category", None) or "refusal"))
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise RuntimeError(f"no text block in response (stop_reason={getattr(response, 'stop_reason', None)})")
        return json.loads(text)

    # ------------------------------------------------------------------ prompts
    def _clip_blocks(self, clip, offsets, fps: float, title: str) -> list[dict]:
        blocks: list[dict] = []
        for off, frame in _pick(clip, offsets, self.frames_per_candidate):
            when = "" if off is None else f" at +{off / fps:.2f} s"
            blocks += [{"type": "text", "text": f"{title}, imagined{when}:"}, png_block(frame, self.max_side)]
        return blocks

    def rank_request(self, clips, *, current, offsets, task: str, fps: float = 30.0) -> dict:
        labels = [chr(ord("A") + i) for i in range(len(clips))]
        content: list[dict] = [{"type": "text", "text": f"Task given to the robot: {task!r}."}]
        if current is not None:
            content += [{"type": "text", "text": "Current view:"}, png_block(current, self.max_side)]
        for label, clip in zip(labels, clips, strict=True):
            content += self._clip_blocks(clip, offsets, fps, f"Candidate {label}")
        content.append({"type": "text", "text": (
            f"Candidates {', '.join(labels)} are alternative imagined futures for the same moment; the robot will "
            "execute exactly one. For each, rate progress toward completing the task (integer 0-10), risk of a "
            "collision, drop or knocked-over object (integer 0-10) and whether the prediction is physically "
            "plausible. Then name the candidate the robot should execute."
        )})
        return self.request(content, RANK_SCHEMA)

    def rank(self, clips, *, current=None, offsets=None, task: str = "", fps: float = 30.0) -> JudgeResult:
        raw = self.send(self.rank_request(clips, current=current, offsets=offsets, task=task, fps=fps))
        self.last_result = raw
        by_label = {c["label"].strip().upper()[:1]: c for c in raw.get("candidates", [])}
        scores = []
        for i in range(len(clips)):
            c = by_label.get(chr(ord("A") + i))
            if c is None:
                scores.append(-10.0)
                continue
            progress = float(np.clip(c["progress"], 0, 10))
            risk = float(np.clip(c["risk"], 0, 10))
            scores.append(progress - 0.5 * risk - (0.0 if c["plausible"] else 3.0))
        return JudgeResult(scores=scores, best=int(np.argmax(scores)), raw=raw)

    def __call__(self, clips, *, current=None, offsets=None, task: str = "", spec=None) -> list[float]:
        fps = getattr(spec, "fps", 30.0)
        try:
            self.last_error = None
            return self.rank(clips, current=current, offsets=offsets, task=task, fps=fps).scores
        except Exception as exc:  # the judge must never stop the robot: fall back to the policy's own seed
            self.last_error = f"{type(exc).__name__}: {exc}"
            return [1.0] + [0.0] * (len(clips) - 1)

    def anticipate(self, clip, *, current=None, offsets=None, task: str = "", fps: float = 30.0) -> dict:
        """Before or during execution: does the imagined future complete the (sub)task?"""
        content: list[dict] = [{"type": "text", "text": f"Task given to the robot: {task!r}."}]
        if current is not None:
            content += [{"type": "text", "text": "Current view:"}, png_block(current, self.max_side)]
        content += self._clip_blocks(clip, offsets, fps, "Prediction")
        content.append({"type": "text", "text": (
            "If the robot's future looks like this prediction, will the task be completed at the end of it? "
            "Give a probability between 0 and 1, describe the imagined outcome in one sentence and name the most "
            "likely failure mode (or 'none')."
        )})
        out = self.send(self.request(content, ANTICIPATE_SCHEMA))
        self.last_result = out
        return out

    def compare(self, imagined: np.ndarray, actual: np.ndarray, *, task: str = "") -> dict:
        """After execution: the imagined end frame vs the real one. Feeds retry or recovery decisions."""
        content = [
            {"type": "text", "text": f"Task given to the robot: {task!r}."},
            {"type": "text", "text": "What the world model imagined at the end of the chunk:"},
            png_block(imagined, self.max_side),
            {"type": "text", "text": "What the cameras actually show now:"},
            png_block(actual, self.max_side),
            {"type": "text", "text": (
                "Does reality match the imagination in the ways that matter for the task (object positions, "
                "grasp state)? Is the task done? Recommend continue, retry, recover or ask_human."
            )},
        ]
        out = self.send(self.request(content, COMPARE_SCHEMA))
        self.last_result = out
        return out
