"""CPU stand-ins for FLUX 3 Action, shaped like the real code paths, for tests and ``run_imagine.py --fake``.

``FakeFluxPolicy`` mirrors the structure of LeRobot ``Flux3Policy`` that ``predict_with_video`` relies on
(modeling_flux3.py:513-623): ``predict_action_chunk`` packs a condition per batch item and calls
``self._sample(cond, caption, seed)``, which builds ``{"x_video", "x_action"}`` from seeded noise and runs
``sampling.euler`` from the module-level ``sampling`` name. ``FakeVideoVAE`` keeps the real latent geometry
(96 channels, /32 spatial, /4 temporal with the same slot layout) so decoded clips have the real frame count.
"""

from __future__ import annotations

import types
from collections import deque
from dataclasses import dataclass, field

import torch

from .predict_with_video import (
    LATENT_CHANNELS,
    SPATIAL_DOWNSAMPLE,
    decoded_slots,
    encoder_slots,
    latent_frames,
)


def _euler(samples, predict_velocity, *, n_steps, alpha):
    """Rectified-flow Euler on a shifted grid, same signature and return value as f3/sampling.euler."""
    ts = torch.linspace(1.0, 0.0, n_steps + 1)
    ts = alpha * ts / (1 + (alpha - 1) * ts)
    for t_curr, t_prev in zip(ts[:-1].tolist(), ts[1:].tolist(), strict=True):
        pred = predict_velocity(samples, t_curr)
        samples = {k: samples[k] + (t_prev - t_curr) * pred[k] for k in samples}
    return samples


def _unipc(samples, predict_velocity, *, n_steps, shift):
    return _euler(samples, predict_velocity, n_steps=n_steps, alpha=shift)


# The fake policy looks this name up at call time, exactly like ``sampling.euler(...)`` in modeling_flux3.py.
sampling = types.ModuleType("ws4_imagine_fake_sampling")
sampling.euler = _euler
sampling.cosmos_unipc_order2 = _unipc


class FakeVideoVAE:
    """Encode: 32x32 average pooling into channels 0..2, slots per ``encoder_slots``. Decode: the inverse layout."""

    def __init__(self):
        self.module = torch.nn.Identity()
        self.decode_calls = 0

    def _encode(self, video: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video.shape
        pooled = torch.nn.functional.avg_pool2d(
            video.float().transpose(1, 2).reshape(b * t, c, h, w), SPATIAL_DOWNSAMPLE
        ).reshape(b, t, c, h // SPATIAL_DOWNSAMPLE, w // SPATIAL_DOWNSAMPLE)
        slots = encoder_slots(t)
        lat = torch.zeros(b, LATENT_CHANNELS, len(slots), h // SPATIAL_DOWNSAMPLE, w // SPATIAL_DOWNSAMPLE)
        for j, frames in enumerate(slots):
            for s, f in enumerate(frames):  # slot s of latent j stores frame f in channels 3s .. 3s+2
                lat[:, 3 * s : 3 * s + 3, j] = pooled[:, f]
        return lat.to(torch.bfloat16)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        return self._encode(video)

    def encode_task(self, video: torch.Tensor) -> torch.Tensor:
        return self._encode(video)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_calls += 1
        lat = latents.float()
        b, _, n, h, w = lat.shape
        frames = [lat[:, 3 * s : 3 * s + 3, j] for j, s in decoded_slots(n)]
        video = torch.stack(frames, dim=2)  # (B, 3, 4n-3, h, w)
        video = video.repeat_interleave(SPATIAL_DOWNSAMPLE, -1).repeat_interleave(SPATIAL_DOWNSAMPLE, -2)
        return video.clamp(-1, 1)


@dataclass
class FakeConfig:
    conditioning: str = "history"
    chunk_size: int = 42
    n_action_steps: int = 32
    n_obs_steps: int = 8
    fps: float = 30.0
    canvas_hw: tuple[int, int] = (64, 128)
    camera_layout: str = "side_by_side"
    camera_keys: list[str] = field(default_factory=lambda: ["observation.images.scene", "observation.images.wrist"])
    action_dim: int = 6
    action_scale: float = 2.0
    inference_seed: int = 42
    num_inference_steps: int = 4
    sampler: str = "euler"

    @property
    def latent_hw(self) -> tuple[int, int]:
        return (-(-self.canvas_hw[0] // SPATIAL_DOWNSAMPLE), -(-self.canvas_hw[1] // SPATIAL_DOWNSAMPLE))

    @property
    def camera_order(self) -> list[str]:
        return list(self.camera_keys)


class FakeFluxPolicy:
    """Imagines the current canvas drifting right by a seed-dependent amount; actions follow the drift."""

    def __init__(self, config: FakeConfig | None = None):
        self.config = config or FakeConfig()
        self.frozen = types.SimpleNamespace(video_vae=FakeVideoVAE())
        self.sample_calls: list[int] = []
        self.reset()

    def reset(self) -> None:
        self._action_queue: deque = deque([], maxlen=self.config.n_action_steps)

    def _canvas(self, batch, i) -> torch.Tensor:
        cfg = self.config
        cams = []
        for key in cfg.camera_order:
            img = batch[key][i].float()
            if img.ndim == 4:  # (T, C, H, W) history: the latest frame
                img = img[-1]
            cams.append(
                torch.nn.functional.interpolate(img[None], size=(cfg.canvas_hw[0], cfg.canvas_hw[1] // 2))[0]
            )
        canvas = torch.cat(cams, dim=-1) if cfg.camera_layout == "side_by_side" else cams[0]
        return canvas.mul(2).sub(1)  # (3, H, W) in [-1, 1]

    def _sample(self, cond, caption, seed):
        cfg = self.config
        self.sample_calls.append(int(seed))
        h, w = cfg.latent_hw
        history = cfg.conditioning == "history"
        n_pred = latent_frames(cfg.chunk_size) if history else latent_frames(cfg.chunk_size + 1) - 1
        rng = torch.Generator().manual_seed(int(seed))
        x_video = torch.randn(1, n_pred * h * w, LATENT_CHANNELS, generator=rng)
        x_action = torch.randn(1, cfg.chunk_size, cfg.action_dim, generator=rng)
        base = cond["x_video_cond"][:, -h * w :].float()  # the latest snapshot tokens
        shift = 1 + int(seed) % 3
        target_frames = []
        for k in range(n_pred):
            grid = base.reshape(1, h, w, LATENT_CHANNELS)
            target_frames.append(torch.roll(grid, shifts=(shift * (k + 1)) // n_pred + 1, dims=2).reshape(1, h * w, -1))
        video_target = torch.cat(target_frames, dim=1)
        action_target = torch.full_like(x_action, float(shift) * cfg.action_scale)

        def predict(samples, t):  # velocity of the straight path noise -> target
            t = max(float(t), 1e-3)
            return {
                "x_video": (samples["x_video"] - video_target) / t,
                "x_action": (samples["x_action"] - action_target) / t,
            }

        out = sampling.euler(
            {"x_video": x_video, "x_action": x_action}, predict, n_steps=cfg.num_inference_steps, alpha=1.0
        )
        return out["x_action"][0].float() / cfg.action_scale

    def predict_action_chunk(self, batch):
        cfg = self.config
        vae = self.frozen.video_vae
        b = batch["observation.state"].shape[0]
        task = batch.get("task", [""] * b)
        task = [task] * b if isinstance(task, str) else list(task)
        chunks = []
        h, w = cfg.latent_hw
        for i in range(b):
            canvas = self._canvas(batch, i)
            snapshots = 2 if cfg.conditioning == "history" else 1
            lat = vae.encode_task(canvas[None, :, None])  # (1, 96, 1, h, w)
            tokens = lat.float().permute(0, 2, 3, 4, 1).reshape(1, h * w, LATENT_CHANNELS)
            cond = {"x_video_cond": tokens.repeat(1, snapshots, 1)}
            chunks.append(self._sample(cond, task[i], cfg.inference_seed))
        return torch.stack(chunks)

    def select_action(self, batch):
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()


def fake_batch(config: FakeConfig | None = None, batch_size: int = 1, seed: int = 0, task: str = "pick the cube"):
    """A preprocessed-looking observation batch with a bright square in the scene camera."""
    cfg = config or FakeConfig()
    g = torch.Generator().manual_seed(seed)
    batch = {"observation.state": torch.zeros(batch_size, cfg.n_obs_steps, cfg.action_dim), "task": task}
    for k, key in enumerate(cfg.camera_order):
        img = torch.rand(batch_size, cfg.n_obs_steps, 3, 96, 96, generator=g) * 0.2
        img[:, :, k % 3, 24:56, 16 + 16 * k : 48 + 16 * k] = 1.0
        batch[key] = img
    return batch
