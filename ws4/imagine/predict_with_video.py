"""Return FLUX 3 Action's imagined future frames together with each action chunk.

Why this is needed (pinned sources, details in ws4/research/imagination.md):

* LeRobot ``Flux3Policy._sample`` (lerobot@146d847 src/lerobot/policies/flux3/modeling_flux3.py:513-578)
  denoises the dict ``{"x_video", "x_<modality>"}`` jointly with ``sampling.euler`` or
  ``sampling.cosmos_unipc_order2`` and returns only the action stream (line 578). flux-action has the same
  structure (flux-action@830f1ce src/flux_action/policy.py:786-879). The BFL inference docs state that the
  API "returns actions only" and offers no video flag.
* Both solvers return the final dict of every stream (f3/sampling.py:69-83 and 134-220), so the clean video
  tokens exist at the end of every call. They are ``(1, n_latent * h * w, 96)`` in ``(t h w)`` order
  (f3/positional.py:38-51).

This module captures that dict without editing either library:

1. a transparent, thread-local hook on the policy's ``sampling`` module records each solver output;
2. an instance-level wrapper around ``policy._sample(cond, caption, seed)`` (the same hook point in both
   libraries) runs one or more seeds, keeps the video tokens and the conditioning tokens, optionally scores
   the candidates, and returns the chosen action chunk to the unchanged ``predict_action_chunk`` /
   ``select_action`` code;
3. the frozen video VAE decodes the latents, synchronously or on a background thread on another GPU.

Nothing here imports torch at module import time. Everything runs on CPU with the fakes in ``fakes.py``.
"""

from __future__ import annotations

import functools
import queue
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Video VAE token format (lerobot f3/packing.py:52-54, flux-action processing/packing.py).
LATENT_CHANNELS = 96
SPATIAL_DOWNSAMPLE = 32
TEMPORAL_DOWNSAMPLE = 4


# ---------------------------------------------------------------------------------------------
# Frame timing: which control step each decoded frame shows
# ---------------------------------------------------------------------------------------------
def latent_frames(n_frames: int) -> int:
    """Latent frames for a clip of ``n_frames`` (packing.latent_frames)."""
    return 1 + (n_frames - 1) // TEMPORAL_DOWNSAMPLE


def encoder_slots(n_frames: int) -> list[tuple[int, ...]]:
    """Clip frame index in each of the 4 temporal slots of every latent frame.

    Mirrors the encoder's two ``TemporalMerging`` layers (lerobot f3/video_vae.py:194-224, applied at
    encoder stages 2 and 3, lines 556-601): an odd-length sequence duplicates its first element at the front,
    then consecutive pairs merge. For a 4k+1 clip this gives the usual causal layout (latent 0 = frame 0).
    For the SO-101 future clip of 42 frames (even) latent 0 holds frames (0, 1) and latent j holds
    4j-2 .. 4j+1.
    """
    if n_frames < 1:
        raise ValueError("need at least one frame")
    groups: list[tuple[int, ...]] = [(i,) for i in range(n_frames)]
    for _ in range(2):
        if len(groups) % 2:
            groups = [groups[0], *groups]
        groups = [groups[i] + groups[i + 1] for i in range(0, len(groups), 2)]
    return groups


def decoded_slots(n_latents: int) -> list[tuple[int, int]]:
    """``(latent, slot)`` shown by each decoded frame.

    Mirrors the decoder's two ``TemporalExpansion`` layers (video_vae.py:249-270): every latent becomes two
    frames and the first output frame is dropped, so ``n`` latents decode to ``4n - 3`` frames, with frame 0
    taken from slot 3 of latent 0 and frames ``4j-3 .. 4j`` from slots 0..3 of latent ``j``.
    """
    items: list[tuple[int, ...]] = [(j,) for j in range(n_latents)]
    for _ in range(2):
        items = [it + (k,) for it in items for k in (0, 1)][1:]
    return [(it[0], 2 * it[1] + it[2]) for it in items]


def decoded_frame_sources(n_clip_frames: int) -> list[tuple[int, int]]:
    """``(latent, clip_frame)`` for each decoded frame of a clip of ``n_clip_frames`` frames."""
    slots = encoder_slots(n_clip_frames)
    return [(j, slots[j][s]) for j, s in decoded_slots(len(slots))]


@dataclass(frozen=True)
class VideoSpec:
    """The parts of a FLUX 3 Action config that fix the imagined video's shape and timing."""

    conditioning: str  # "history" (SO-101, task LoRA) or "frame" (DROID reference layout)
    chunk_size: int
    n_action_steps: int
    fps: float
    latent_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    camera_layout: str
    camera_keys: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, cfg: Any) -> VideoSpec:
        """Read a LeRobot ``Flux3Config`` or a flux-action ``PolicyConfig``."""
        conditioning = getattr(cfg, "conditioning", None)
        if conditioning is None:  # flux-action names it inference_profile ("history" / "default")
            conditioning = "history" if getattr(cfg, "inference_profile", "default") == "history" else "frame"
        latent_hw = tuple(int(x) for x in cfg.latent_hw)
        keys = getattr(cfg, "camera_order", None) or getattr(cfg, "camera_keys", None) or ()
        return cls(
            conditioning=str(conditioning),
            chunk_size=int(cfg.chunk_size),
            n_action_steps=int(cfg.n_action_steps),
            fps=float(cfg.fps),
            latent_hw=(latent_hw[0], latent_hw[1]),
            canvas_hw=(int(cfg.canvas_hw[0]), int(cfg.canvas_hw[1])),
            camera_layout=str(cfg.camera_layout),
            camera_keys=tuple(keys),
        )

    @property
    def n_pred_latents(self) -> int:
        """Predicted latent frames: ``latent_frames(chunk)`` for history, ``latent_frames(chunk + 1) - 1`` for
        frame conditioning (lerobot f3/packing.py:556-557 and 607-608)."""
        if self.conditioning == "history":
            return latent_frames(self.chunk_size)
        return latent_frames(self.chunk_size + 1) - 1

    @property
    def decode_with_condition(self) -> bool:
        """Frame conditioning encodes the current frame and the future as ONE clip (packing.py:571-576), so its
        predicted latents continue latent 0 and must be decoded after it. History conditioning encodes the
        future clip on its own (packing.py:469-473) and decodes alone."""
        return self.conditioning != "history"

    @property
    def frame_offsets(self) -> tuple[int, ...]:
        """Control steps after the observation that each decoded imagined frame shows.

        History: the future clip holds the frames at offsets +1 .. +chunk (image delta indices
        ``range(1 - n_obs, chunk + 1)``, configuration_flux3.py:382-387), so clip frame c is offset c + 1.
        SO-101 (chunk 42): 41 frames at offsets +2 .. +42.
        Frame: the window is offsets 0 .. chunk; the decoded frames of latent 0 (the current observation) are
        dropped. DROID (chunk 32): 32 frames at offsets +1 .. +32.
        """
        if self.conditioning == "history":
            return tuple(c + 1 for _, c in decoded_frame_sources(self.chunk_size))
        return tuple(c for j, c in decoded_frame_sources(self.chunk_size + 1) if j > 0)

    @property
    def decoded_hw(self) -> tuple[int, int]:
        return self.latent_hw[0] * SPATIAL_DOWNSAMPLE, self.latent_hw[1] * SPATIAL_DOWNSAMPLE

    def seconds(self, offset: int) -> float:
        return offset / self.fps


# ---------------------------------------------------------------------------------------------
# Tokens <-> latents
# ---------------------------------------------------------------------------------------------
def tokens_to_latents(tokens, n_frames: int, latent_hw: tuple[int, int]):
    """``(B, n*h*w, 96)`` video tokens in ``(t h w)`` order -> latents ``(B, 96, n, h, w)``."""
    b, n_tok, c = tokens.shape
    h, w = latent_hw
    if n_tok != n_frames * h * w:
        raise ValueError(f"{n_tok} video tokens do not form {n_frames} x {h} x {w} latent frames")
    return tokens.reshape(b, n_frames, h, w, c).permute(0, 4, 1, 2, 3).contiguous()


def condition_latents(cond_tokens, latent_hw: tuple[int, int]):
    """Conditioning tokens ``(1, k*h*w, 96)`` -> one single-frame latent clip per snapshot, ``(k, 96, 1, h, w)``.

    Every history snapshot is encoded independently as a single frame (packing.py:462-467) and the frame
    packer's condition is latent 0 of the window, so each decodes as a 1-latent clip.
    """
    h, w = latent_hw
    k = cond_tokens.shape[1] // (h * w)
    lat = tokens_to_latents(cond_tokens, k, latent_hw)  # (1, 96, k, h, w)
    return lat[0].permute(1, 0, 2, 3)[:, :, None].contiguous()


# ---------------------------------------------------------------------------------------------
# Capturing the solver output
# ---------------------------------------------------------------------------------------------
_SOLVERS = ("euler", "cosmos_unipc_order2")
_capture_state = threading.local()


def _ensure_solver_hook(sampling_module) -> None:
    """Wrap the module's solvers once. The wrapper records outputs only while this thread is capturing, so
    concurrent policies in other threads (one model per arm) are unaffected."""
    for name in _SOLVERS:
        fn = getattr(sampling_module, name, None)
        if fn is None or getattr(fn, "_imagine_hooked", False):
            continue

        def make(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                out = original(*args, **kwargs)
                stack = getattr(_capture_state, "stack", None)
                if stack:
                    stack[-1].append(out)
                return out

            wrapped._imagine_hooked = True  # type: ignore[attr-defined]
            return wrapped

        setattr(sampling_module, name, make(fn))


class _Capture:
    def __init__(self, sampling_module):
        _ensure_solver_hook(sampling_module)
        self.outputs: list[dict] = []

    def __enter__(self) -> list[dict]:
        if not hasattr(_capture_state, "stack"):
            _capture_state.stack = []
        _capture_state.stack.append(self.outputs)
        return self.outputs

    def __exit__(self, *exc) -> None:
        _capture_state.stack.pop()


def capture_solver_outputs(sampling_module) -> _Capture:
    """``with capture_solver_outputs(mod) as outs:`` collects every solver result dict in this thread."""
    return _Capture(sampling_module)


def find_sampling_module(policy) -> Any:
    """The ``sampling`` module the policy's ``_sample`` calls (looked up at call time, so the hook applies)."""
    for klass in type(policy).__mro__:
        module = sys.modules.get(klass.__module__)
        sampling = getattr(module, "sampling", None)
        if sampling is not None and all(hasattr(sampling, n) for n in ("euler",)):
            return sampling
    raise RuntimeError(f"cannot find the sampling module used by {type(policy).__name__}; pass sampling_module=")


# ---------------------------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------------------------
def video_vae_of(policy) -> Any:
    """The policy's frozen video VAE: LeRobot keeps it in ``policy.frozen.video_vae`` (modeling_flux3.py:147),
    flux-action exposes ``policy.video_vae`` (policy.py:293)."""
    frozen = getattr(policy, "frozen", None)
    if frozen is not None and getattr(frozen, "video_vae", None) is not None:
        return frozen.video_vae
    vae = getattr(policy, "video_vae", None)
    if vae is None:
        raise RuntimeError("policy exposes no video VAE")
    return vae


def to_uint8_frames(video) -> np.ndarray:
    """VAE output ``(B, 3, T, H, W)`` in [-1, 1] -> ``(B, T, H, W, 3)`` uint8."""
    x = video.float().clamp(-1, 1).add(1).mul(127.5).round().to("cpu")
    return x.permute(0, 2, 3, 4, 1).numpy().astype(np.uint8)


class LatentDecoder:
    """Decode predicted latents with the frozen video VAE.

    ``vae.decode(latents)`` takes normalized latents ``(B, 96, n, h, w)`` and returns ``(B, 3, 4n-3, 32h, 32w)``
    in [-1, 1] (VideoVAE.decode -> ViTNormInference.decode, video_vae.py:881-895 and 983-985). The VAE runs in
    bf16 while the solver state is fp32 (modeling_flux3.py:528-530), so latents are cast here.
    """

    def __init__(self, vae, *, device=None, dtype=None):
        self.vae = vae
        self.device = device
        self.dtype = dtype

    def vae_device(self):
        """Where the VAE's weights live. Captured latents arrive on CPU, so with ``device=None`` they must follow
        the VAE (LeRobot's decode also queries ``torch.cuda.mem_get_info(z.device)``, video_vae.py:876-879)."""
        import torch

        module = getattr(self.vae, "module", self.vae)
        try:
            return next(module.parameters()).device
        except (AttributeError, StopIteration):  # parameter-free fakes
            return torch.device("cpu")

    def _prepare(self, latents):
        import torch

        dtype = self.dtype or torch.bfloat16
        device = self.device if self.device is not None else self.vae_device()
        return latents.to(device=device, dtype=dtype)

    def decode_latents(self, latents) -> np.ndarray:
        """``(B, 96, n, h, w)`` -> uint8 ``(B, 4n-3, H, W, 3)``."""
        import torch

        with torch.inference_mode():
            video = self.vae.decode(self._prepare(latents))
        return to_uint8_frames(video)

    def decode_record(self, record: Imagination) -> Imagination:
        """Fill ``record.frames`` (imagined future) and ``record.cond_frames`` (VAE view of the observation)."""
        import torch

        t0 = time.perf_counter()
        spec = record.spec
        latents = record.latents[None]
        cond = record.cond_latents
        if spec.decode_with_condition:
            if cond is None:
                raise ValueError("frame conditioning needs the condition latent to decode its future")
            full = self.decode_latents(torch.cat([cond[-1:], latents], dim=2))[0]
            n_drop = sum(1 for j, _ in decoded_slots(1 + latents.shape[2]) if j == 0)
            record.cond_frames = full[:n_drop]
            record.frames = full[n_drop:]
        else:
            record.frames = self.decode_latents(latents)[0]
            if cond is not None:
                record.cond_frames = self.decode_latents(cond)[:, 0]
        if record.frames.shape[0] != len(spec.frame_offsets):
            raise RuntimeError(f"decoded {record.frames.shape[0]} frames, expected {len(spec.frame_offsets)}")
        record.timings["decode_s"] = time.perf_counter() - t0
        return record


class AsyncDecoder:
    """Decode on a background thread (optionally on another GPU) so decoding never delays control.

    Latents are copied to CPU in the control thread right after sampling (a few hundred KB, and the action
    chunk is synchronized at the same point anyway), then moved to ``device`` by the worker, which avoids any
    cross-stream ordering question between GPUs.
    """

    def __init__(self, decoder: LatentDecoder, *, max_pending: int = 4):
        self.decoder = decoder
        self._queue: queue.Queue = queue.Queue(maxsize=max_pending)
        self._worker = threading.Thread(target=self._run, name="imagine-decoder", daemon=True)
        self._worker.start()

    def submit(self, record: Imagination, callback: Callable[[Imagination], None] | None = None) -> Future:
        future: Future = Future()
        record.future = future
        try:
            self._queue.put_nowait((record, callback, future))
        except queue.Full:  # never block control: skip decoding this chunk
            record.timings["decode_skipped"] = 1.0
            future.set_result(record)
        return future

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            record, callback, future = item
            try:
                self.decoder.decode_record(record)
                if callback is not None:
                    callback(record)
                future.set_result(record)
            except BaseException as exc:  # surfaced through the future, never into control
                future.set_exception(exc)

    def close(self) -> None:
        self._queue.put(None)
        self._worker.join(timeout=5)


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------
@dataclass
class Candidate:
    seed: int
    sample_s: float
    score: float | None = None


@dataclass
class Imagination:
    """One action chunk and what the model imagined while producing it."""

    spec: VideoSpec
    task: str
    seed: int
    latents: Any  # (96, n, h, w) float32 CPU tensor, the chosen candidate
    cond_latents: Any = None  # (k, 96, 1, h, w): the conditioning snapshots
    action: Any = None  # (chunk, D) chunk returned to the policy, in model/normalized units
    step: int | None = None  # control step at which the chunk was requested (select_action count)
    index: int = 0  # chunk counter
    t_request: float = 0.0  # wall clock when sampling started
    t_return: float = 0.0  # wall clock when the chunk was handed back to the controller
    candidates: list[Candidate] = field(default_factory=list)
    chosen: int = 0
    score: float | None = None
    frames: np.ndarray | None = None  # (T, H, W, 3) uint8 imagined future
    cond_frames: np.ndarray | None = None  # (k, H, W, 3) uint8 VAE view of the conditioning frames
    candidate_frames: list[np.ndarray] | None = None  # every candidate's clip, when best-of-N decoded them
    timings: dict[str, float] = field(default_factory=dict)
    future: Future | None = None

    def wait(self, timeout: float | None = None) -> Imagination:
        if self.future is not None:
            self.future.result(timeout)
        return self

    @property
    def frame_offsets(self) -> tuple[int, ...]:
        return self.spec.frame_offsets

    def frame_at(self, offset: int) -> np.ndarray:
        """Imagined frame closest to ``offset`` control steps after the observation."""
        if self.frames is None:
            raise RuntimeError("frames not decoded yet (call wait() or decode)")
        offsets = np.asarray(self.frame_offsets)
        return self.frames[int(np.argmin(np.abs(offsets - offset)))]

    def final_frame(self) -> np.ndarray:
        return self.frame_at(max(self.frame_offsets))

    def executed_end_frame(self) -> np.ndarray:
        """Imagined view right after the executed part of the chunk (offset ``n_action_steps``)."""
        return self.frame_at(self.spec.n_action_steps)

    def current_frame(self) -> np.ndarray | None:
        return None if self.cond_frames is None else self.cond_frames[-1]

    def views(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Split a canvas into camera views: ``side_by_side`` is scene left, wrist right (packing.py:162-171,
        flux-action docs/so101-lora.md: "Left = scene; right = wrist")."""
        if self.spec.camera_layout == "side_by_side":
            half = frame.shape[1] // 2
            names = self.spec.camera_keys or ("scene", "wrist")
            short = [k.rsplit(".", 1)[-1] for k in names]
            return {short[0]: frame[:, :half], short[1]: frame[:, half:]}
        return {"canvas": frame}

    def save(self, path: str | Path) -> Path:
        """Write frames, offsets and metadata to one ``.npz`` (read back with ``load_imagination``)."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "spec": self.spec.__dict__,
            "task": self.task,
            "seed": self.seed,
            "step": self.step,
            "index": self.index,
            "t_request": self.t_request,
            "t_return": self.t_return,
            "chosen": self.chosen,
            "score": self.score,
            "candidates": [c.__dict__ for c in self.candidates],
            "timings": self.timings,
        }
        arrays = {"meta": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)}
        if self.frames is not None:
            arrays["frames"] = self.frames
        if self.cond_frames is not None:
            arrays["cond_frames"] = self.cond_frames
        if self.action is not None:
            arrays["action"] = np.asarray(self.action.float().cpu() if hasattr(self.action, "cpu") else self.action)
        arrays["latents"] = np.asarray(self.latents.float().cpu() if hasattr(self.latents, "cpu") else self.latents)
        np.savez_compressed(path, **arrays)
        return path


def load_imagination(path: str | Path) -> Imagination:
    import json

    data = np.load(path)
    meta = json.loads(bytes(data["meta"]).decode())
    spec_d = dict(meta["spec"])
    spec_d["latent_hw"] = tuple(spec_d["latent_hw"])
    spec_d["canvas_hw"] = tuple(spec_d["canvas_hw"])
    spec_d["camera_keys"] = tuple(spec_d["camera_keys"])
    rec = Imagination(
        spec=VideoSpec(**spec_d),
        task=meta["task"],
        seed=meta["seed"],
        latents=data["latents"],
        step=meta["step"],
        index=meta["index"],
        t_request=meta["t_request"],
        t_return=meta["t_return"],
        chosen=meta["chosen"],
        score=meta["score"],
        candidates=[Candidate(**c) for c in meta["candidates"]],
        timings=meta["timings"],
    )
    rec.frames = data["frames"] if "frames" in data else None
    rec.cond_frames = data["cond_frames"] if "cond_frames" in data else None
    rec.action = data["action"] if "action" in data else None
    return rec


# ---------------------------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------------------------
class Imaginer:
    """Wrap a FLUX 3 Action policy so every chunk also yields its imagined future.

    Two ways to use it:

    * ``records = imaginer.imagine(batch)``: one call on a preprocessed observation batch; returns one
      ``Imagination`` per batch item with ``.action`` (the chunk ``predict_action_chunk`` returns),
      ``.frames`` and ``.score``.
    * ``imaginer.install()``: the policy's own ``select_action`` / ``predict_action_chunk`` now imagine on
      every chunk; records go to ``on_imagination`` and ``imaginer.history``. Any control loop works unchanged.

    ``n_candidates > 1`` samples that many seeds (``seed, seed + stride, ...``), scores them and executes the
    best; candidate 0 is the policy's own seed, so ``n_candidates=1`` reproduces stock actions exactly.
    ``decode``: ``"sync"`` (frames ready on return), ``"async"`` (background thread, see ``AsyncDecoder``) or
    ``"none"`` (latents only). Best-of-N with a pixel scorer always decodes the candidates synchronously.
    """

    def __init__(
        self,
        policy,
        *,
        decoder: LatentDecoder | None = None,
        decode: str = "sync",
        decode_device=None,
        scorer: Callable[..., Sequence[float]] | None = None,
        n_candidates: int = 1,
        seed_stride: int = 1,
        on_imagination: Callable[[Imagination], None] | None = None,
        on_frames: Callable[[Imagination], None] | None = None,
        sampling_module=None,
        spec: VideoSpec | None = None,
        history_size: int = 64,
    ):
        if decode not in ("sync", "async", "none"):
            raise ValueError("decode must be sync, async or none")
        if n_candidates < 1:
            raise ValueError("n_candidates must be >= 1")
        if n_candidates > 1 and scorer is None:
            raise ValueError("best-of-N needs a scorer")
        self.policy = policy
        self.spec = spec or VideoSpec.from_config(policy.config)
        self.sampling_module = sampling_module or find_sampling_module(policy)
        self.decode_mode = decode
        self.decoder = decoder
        if self.decoder is None and (decode != "none" or n_candidates > 1):
            self.decoder = LatentDecoder(video_vae_of(policy), device=decode_device)
        self._async = AsyncDecoder(self.decoder) if decode == "async" else None
        self.scorer = scorer
        self.n_candidates = n_candidates
        self.seed_stride = seed_stride
        self.on_imagination = on_imagination
        self.on_frames = on_frames
        self.history: list[Imagination] = []
        self.history_size = history_size
        self._pending: list[Imagination] = []
        self._installed = False
        self._originals: dict[str, Any] = {}
        self._step = 0
        self._current_step: int | None = None
        self._count = 0
        self._lock = threading.Lock()

    @property
    def last_step(self) -> int | None:
        """Control step of the latest ``select_action`` call: tag the camera frame of that tick with it."""
        return self._current_step

    # ---------------------------------------------------------------- hooks
    def install(self) -> Imaginer:
        if self._installed:
            return self
        p = self.policy
        self._originals = {
            "_sample": p._sample,
            "predict_action_chunk": p.predict_action_chunk,
            "select_action": getattr(p, "select_action", None),
            "reset": getattr(p, "reset", None),
        }
        p._sample = self._sample_hook
        p.predict_action_chunk = self._predict_hook
        if self._originals["select_action"] is not None:
            p.select_action = self._select_hook
        if self._originals["reset"] is not None:
            p.reset = self._reset_hook
        self._installed = True
        return self

    def uninstall(self) -> None:
        if not self._installed:
            return
        for name in self._originals:
            # Instance attributes shadow the class methods; deleting them restores the originals.
            self.policy.__dict__.pop(name, None)
        self._installed = False

    def close(self) -> None:
        self.uninstall()
        if self._async is not None:
            self._async.close()

    def _reset_hook(self, *args, **kwargs):
        # Steps stay monotonic across resets so frames logged before and after a forced replan stay aligned.
        return self._originals["reset"](*args, **kwargs)

    def new_episode(self) -> None:
        """Restart step counting and history (call with the processors' reset at an episode boundary)."""
        self._step = 0
        self._current_step = None
        self.history.clear()

    def replan(self) -> None:
        """Drop the queued actions so the next ``select_action`` predicts a fresh chunk from the current
        observation (e.g. when ``SurpriseMonitor`` fires). Leaves the processors' observation history intact;
        LeRobot's ``Flux3Policy.reset`` only clears the action queue (modeling_flux3.py:240-241)."""
        queue_ = getattr(self.policy, "_action_queue", None)
        if queue_ is not None:
            queue_.clear()
        else:
            self.policy.reset()

    def _select_hook(self, batch, *args, **kwargs):
        self._current_step = self._step
        self._step += 1
        return self._originals["select_action"](batch, *args, **kwargs)

    def _predict_hook(self, batch, *args, **kwargs):
        with self._lock:  # single-flight per wrapped model
            self._pending = []
            actions = self._originals["predict_action_chunk"](batch, *args, **kwargs)
            records, self._pending = self._pending, []
        now = time.time()
        for i, rec in enumerate(records):
            rec.action = actions[i].detach().to("cpu") if i < len(actions) else None
            rec.t_return = now
            self._finish(rec)
        self._last_records = records
        return actions

    def _sample_hook(self, cond, caption, seed):
        import torch

        t_request = time.time()
        seeds = [int(seed) + k * self.seed_stride for k in range(self.n_candidates)]
        actions, tokens, cands = [], [], []
        for s in seeds:
            t0 = time.perf_counter()
            with capture_solver_outputs(self.sampling_module) as outs:
                act = self._originals["_sample"](cond, caption, s)
            if not outs or "x_video" not in outs[-1]:
                raise RuntimeError("solver output did not contain x_video; was _sample changed upstream?")
            video = outs[-1]["x_video"].detach().float().to("cpu")  # also synchronizes the device
            cands.append(Candidate(seed=s, sample_s=time.perf_counter() - t0))
            actions.append(act)
            tokens.append(video)
        cond_lat = None
        if "x_video_cond" in cond:
            cond_lat = condition_latents(cond["x_video_cond"].detach().float().to("cpu"), self.spec.latent_hw)
        n = self.spec.n_pred_latents
        latents = [tokens_to_latents(t, n, self.spec.latent_hw)[0] for t in tokens]
        rec = Imagination(
            spec=self.spec,
            task=str(caption),
            seed=seeds[0],
            latents=latents[0],
            cond_latents=cond_lat,
            step=self._current_step,
            index=self._count,
            t_request=t_request,
            candidates=cands,
        )
        self._count += 1
        rec.timings["sample_s"] = sum(c.sample_s for c in cands)
        if self.n_candidates > 1:
            rec.chosen = self._choose(rec, latents)
            rec.seed = seeds[rec.chosen]
            rec.latents = latents[rec.chosen]
            if rec.candidate_frames is not None:
                rec.frames = rec.candidate_frames[rec.chosen]
        self._pending.append(rec)
        chosen = actions[rec.chosen]
        return chosen if isinstance(chosen, torch.Tensor) else torch.as_tensor(chosen)

    # ---------------------------------------------------------------- selection and decoding
    def _choose(self, rec: Imagination, latents: list) -> int:
        t0 = time.perf_counter()
        space = getattr(self.scorer, "space", "pixels")
        if space == "latent":
            scores = list(self.scorer(latents, spec=self.spec, task=rec.task, cond_latents=rec.cond_latents))
        else:
            clips = []
            for lat in latents:
                trial = Imagination(spec=self.spec, task=rec.task, seed=rec.seed, latents=lat,
                                    cond_latents=rec.cond_latents)
                self.decoder.decode_record(trial)
                clips.append(trial.frames)
                rec.cond_frames = trial.cond_frames
            rec.candidate_frames = clips
            scores = list(self.scorer(clips, current=rec.current_frame(), offsets=self.spec.frame_offsets,
                                      task=rec.task, spec=self.spec))
        if len(scores) != len(latents):
            raise RuntimeError(f"scorer returned {len(scores)} scores for {len(latents)} candidates")
        for cand, s in zip(rec.candidates, scores, strict=True):
            cand.score = float(s)
        best = int(np.argmax(scores))
        rec.score = float(scores[best])
        rec.timings["score_s"] = time.perf_counter() - t0
        return best

    def _finish(self, rec: Imagination) -> None:
        if rec.frames is None and self.decode_mode == "sync":
            self.decoder.decode_record(rec)
            if self.on_frames is not None:
                self.on_frames(rec)
        elif rec.frames is None and self.decode_mode == "async":
            self._async.submit(rec, self.on_frames)
        elif rec.frames is not None and self.on_frames is not None:
            self.on_frames(rec)
        self.history.append(rec)
        del self.history[: -self.history_size]
        if self.on_imagination is not None:
            self.on_imagination(rec)

    # ---------------------------------------------------------------- one-shot API
    def imagine(self, batch, *, n_candidates: int | None = None) -> list[Imagination]:
        """Run ``predict_action_chunk`` on a preprocessed batch and return one record per item."""
        was_installed = self._installed
        saved_n = self.n_candidates
        if n_candidates is not None:
            if n_candidates > 1 and self.scorer is None:
                raise ValueError("best-of-N needs a scorer")
            self.n_candidates = n_candidates
        self.install()
        try:
            self.policy.predict_action_chunk(batch)
            return list(self._last_records)
        finally:
            self.n_candidates = saved_n
            if not was_installed:
                self.uninstall()


def alternatives_batch(batch: dict, instructions: Sequence[str]) -> dict:
    """Repeat one preprocessed observation once per candidate instruction (subtask-level search).

    ``imaginer.imagine(alternatives_batch(batch, ["push the box left", "lift the box"]))`` returns one
    ``Imagination`` per instruction from the same observation; rank them (e.g. ``ClaudeJudge.rank``) and send
    the winning instruction to the policy. Tensors must carry batch size 1. The text context of each new
    instruction is encoded once and cached by the policy (modeling_flux3.py:399-409).
    """
    out: dict = {}
    n = len(instructions)
    for key, value in batch.items():
        if key == "task":
            continue
        if hasattr(value, "shape") and getattr(value, "ndim", 0) >= 1:
            if value.shape[0] != 1:
                raise ValueError(f"{key}: expected batch size 1, got {value.shape[0]}")
            out[key] = value.expand(n, *value.shape[1:]) if hasattr(value, "expand") else np.repeat(value, n, 0)
        else:
            out[key] = value
    out["task"] = list(instructions)
    return out


def to_robot_actions(postprocessor, chunk):
    """Normalized chunk ``(chunk, D)`` or ``(B, chunk, D)`` -> absolute commands through the saved postprocessor.

    For history checkpoints the unnormalizer integrates deltas from the preprocessor's command anchor and only
    updates ``last_command`` for 2-D (per-tick) inputs (processor_flux3.py:455-475), so passing a 3-D chunk
    has no side effect on the control loop. UNVERIFIED for postprocessor steps other than the history
    unnormalizer and device move.
    """
    x = chunk if chunk.ndim == 3 else chunk[None]
    return postprocessor(x)
