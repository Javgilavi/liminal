"""Lazy, persistent Qwen3-VL adapter for scene inspection."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from liminal.perception.vlm import ModelResponse


@dataclass(frozen=True)
class QwenVisionConfig:
    checkpoint: str = "Qwen/Qwen3-VL-4B-Instruct"
    device: str = "cuda:2"
    dtype: str = "bfloat16"
    max_new_tokens: int = 150
    revision: str | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, str) or not self.checkpoint.strip():
            raise ValueError("Qwen checkpoint must be non-empty")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Qwen device must be non-empty")
        if not isinstance(self.dtype, str) or self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("Qwen dtype must be auto, bfloat16, float16, or float32")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("Qwen max_new_tokens must be positive")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("Qwen revision must be a non-empty string")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("Qwen local_files_only must be a boolean")


@dataclass(frozen=True)
class QwenLoadMetrics:
    duration_s: float
    gpu_memory_allocated_mb: float | None


@dataclass(frozen=True)
class QwenInferenceMetrics:
    preprocess_s: float
    time_to_first_token_s: float
    generation_s: float
    total_s: float
    gpu_memory_allocated_mb: float | None
    gpu_peak_memory_allocated_mb: float | None


@dataclass(frozen=True)
class _EngineResult:
    text: str
    metrics: QwenInferenceMetrics


class _Engine(Protocol):
    gpu_memory_allocated_mb: float | None

    def inspect(self, prompt: str, images: list[Any]) -> _EngineResult: ...


EngineFactory = Callable[[QwenVisionConfig], _Engine]


class QwenVisionModel:
    """One Qwen model kept resident and serialized behind the SceneInspector.

    Imports and checkpoint loading happen only on the first call to ``load`` or
    ``inspect``. This keeps mock tests and non-VLM commands dependency-free.
    """

    def __init__(self, config: QwenVisionConfig, *, engine_factory: EngineFactory | None = None):
        self.config = config
        self._engine_factory = engine_factory or _TransformersQwenEngine
        self._engine: _Engine | None = None
        self._load_lock = threading.Lock()
        self.load_metrics: QwenLoadMetrics | None = None
        self.inference_metrics: list[QwenInferenceMetrics] = []

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def load(self) -> None:
        if self._engine is not None:
            return
        with self._load_lock:
            if self._engine is not None:
                return
            started = time.perf_counter()
            engine = self._engine_factory(self.config)
            self._engine = engine
            self.load_metrics = QwenLoadMetrics(
                duration_s=time.perf_counter() - started,
                gpu_memory_allocated_mb=engine.gpu_memory_allocated_mb,
            )

    async def inspect(self, prompt: str, images: list[Any]) -> ModelResponse:
        if not prompt.strip():
            raise ValueError("Qwen inspection prompt must be non-empty")
        if not images:
            raise ValueError("Qwen inspection needs at least one image")
        return await asyncio.to_thread(self._inspect_sync, prompt, images)

    def _inspect_sync(self, prompt: str, images: list[Any]) -> str:
        self.load()
        assert self._engine is not None
        result = self._engine.inspect(prompt, images)
        self.inference_metrics.append(result.metrics)
        return result.text


class _TransformersQwenEngine:
    def __init__(self, config: QwenVisionConfig):
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL dependencies are missing; install the project with the 'vlm-qwen' extra"
            ) from exc

        self.config = config
        self.torch = torch
        self.device = torch.device(config.device)
        self._validate_device()
        dtype = self._dtype()
        before_mb = self._memory_allocated_mb()
        processor_kwargs = {"local_files_only": config.local_files_only}
        model_kwargs: dict[str, Any] = {
            "device_map": {"": config.device},
            "dtype": dtype,
            "local_files_only": config.local_files_only,
        }
        if config.revision is not None:
            processor_kwargs["revision"] = config.revision
            model_kwargs["revision"] = config.revision
        self.processor = AutoProcessor.from_pretrained(config.checkpoint, **processor_kwargs)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(config.checkpoint, **model_kwargs)
        self.model.eval()
        self._synchronize()
        after_mb = self._memory_allocated_mb()
        self.gpu_memory_allocated_mb = None if after_mb is None else max(0.0, after_mb - (before_mb or 0.0))

    def inspect(self, prompt: str, images: list[Any]) -> _EngineResult:
        started = time.perf_counter()
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

        pil_images = [_as_rgb_image(image) for image in images]
        messages = [{
            "role": "user",
            "content": [
                *({"type": "image"} for _ in pil_images),
                {"type": "text", "text": prompt},
            ],
        }]
        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=rendered,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)
        self._synchronize()
        preprocess_s = time.perf_counter() - started

        generation_started = time.perf_counter()
        first_token = _FirstTokenTimer(self._synchronize)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                logits_processor=[first_token],
            )
        self._synchronize()
        generation_finished = time.perf_counter()
        prompt_tokens = inputs["input_ids"].shape[-1]
        text = self.processor.batch_decode(
            generated[:, prompt_tokens:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        total_s = time.perf_counter() - started
        generation_s = generation_finished - generation_started
        metrics = QwenInferenceMetrics(
            preprocess_s=preprocess_s,
            time_to_first_token_s=(first_token.at or generation_finished) - generation_started,
            generation_s=generation_s,
            total_s=total_s,
            gpu_memory_allocated_mb=self._memory_allocated_mb(),
            gpu_peak_memory_allocated_mb=self._peak_memory_allocated_mb(),
        )
        return _EngineResult(text.strip(), metrics)

    def _validate_device(self) -> None:
        if self.device.type != "cuda":
            return
        if not self.torch.cuda.is_available():
            raise RuntimeError(f"Qwen requested {self.config.device}, but CUDA is unavailable")
        index = self.device.index if self.device.index is not None else self.torch.cuda.current_device()
        if index >= self.torch.cuda.device_count():
            raise RuntimeError(
                f"Qwen requested {self.config.device}, but only {self.torch.cuda.device_count()} CUDA devices exist"
            )

    def _dtype(self) -> Any:
        return {
            "auto": "auto",
            "bfloat16": self.torch.bfloat16,
            "float16": self.torch.float16,
            "float32": self.torch.float32,
        }[self.config.dtype]

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _memory_allocated_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return self.torch.cuda.memory_allocated(self.device) / (1024 * 1024)

    def _peak_memory_allocated_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return self.torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)


class _FirstTokenTimer:
    def __init__(self, synchronize: Callable[[], None]):
        self.synchronize = synchronize
        self.at: float | None = None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.at is None:
            self.synchronize()
            self.at = time.perf_counter()
        return scores


def _as_rgb_image(image: Any) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Qwen3-VL image conversion requires numpy and Pillow") from exc

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str | Path):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"Qwen images must be HxWx3 or HxWx4, got shape {array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("Qwen images must contain finite numeric pixels")
        if array.min() < 0 or array.max() > 255:
            raise ValueError("Qwen image pixels must be in the range 0..255")
        array = array.astype(np.uint8)
    mode = "RGBA" if array.shape[2] == 4 else "RGB"
    return Image.fromarray(array, mode=mode).convert("RGB")
