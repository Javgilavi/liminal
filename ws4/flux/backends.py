"""Policy backends: something that turns one target's history window into a chunk of absolute commands.

``LeRobotFlux3Backend`` wraps a saved LeRobot ``flux3`` package (``black-forest-labs/flux-3-action-so101`` or a
LoRA / bimanual package made from it). It is stateless per request: every call passes the full window through
the package's own preprocessor with ``observation.command_history`` (offline path), predicts one chunk and
integrates it with the package's own postprocessor. One loaded model can therefore serve both arms; calls are
single-flight behind one lock because the history processor stores the delta anchor between pre- and
post-processing (lerobot@146d847:src/lerobot/policies/flux3/processor_flux3.py:394, 464-470).

Batching both arms into one call does not save time: ``predict_action_chunk`` samples batch elements one after
another (lerobot@146d847:src/lerobot/policies/flux3/modeling_flux3.py:594-608).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .contract import OBS_STATE, ModelContract, snapshot_indices

COMMAND_HISTORY = "observation.command_history"


@dataclass
class PolicyRequest:
    task: str
    states: np.ndarray  # (n_obs_steps, D) model units
    commands: np.ndarray  # (n_obs_steps, D) command issued before each observation
    images: dict[str, np.ndarray]  # model camera key -> (n_snapshots, H, W, 3) uint8
    snapshot_ticks: tuple[int, ...]
    target: str = ""
    imagine: str = ""  # "" (off), "sync", "async" or "lazy": also return imagined frames (imagine/serving.py)


@dataclass
class PolicyResponse:
    commands: np.ndarray  # (chunk_size, D) absolute commands, model units
    latency_s: float
    info: dict[str, Any] = field(default_factory=dict)
    imagination: dict[str, Any] | None = None  # {"meta": ..., "arrays": ...} when the request asked for it


class FluxBackend(Protocol):
    contract: ModelContract

    def predict(self, request: PolicyRequest) -> PolicyResponse: ...


def validate_request(request: PolicyRequest, contract: ModelContract) -> None:
    n, d = contract.n_obs_steps, contract.action_dim
    if request.states.shape != (n, d) or request.commands.shape != (n, d):
        raise ValueError(f"states/commands must be ({n}, {d}), got {request.states.shape} / {request.commands.shape}")
    if tuple(request.snapshot_ticks) != contract.snapshots:
        raise ValueError(f"snapshot ticks {request.snapshot_ticks} differ from the model's {contract.snapshots}")
    for key in contract.camera_keys:
        img = request.images.get(key)
        if img is None:
            raise KeyError(f"request lacks camera {key!r}; has {sorted(request.images)}")
        if img.ndim != 4 or img.shape[0] != len(contract.snapshots) or img.shape[-1] != 3 or img.dtype != np.uint8:
            raise ValueError(f"{key}: expected ({len(contract.snapshots)}, H, W, 3) uint8, got {img.shape} {img.dtype}")


def expand_snapshots(snapshots: np.ndarray, ticks: tuple[int, ...], n_obs_steps: int) -> np.ndarray:
    """``(S, H, W, 3)`` snapshots at ``ticks`` -> ``(n_obs_steps, H, W, 3)``.

    Only the snapshot ticks are encoded by the model (packing.py:457-466); the other ticks are filled with the
    nearest snapshot so shapes match ``Flux3Policy._cameras`` (modeling_flux3.py:590-591).
    """
    ticks_arr = np.asarray(ticks)
    nearest = np.abs(np.arange(n_obs_steps)[:, None] - ticks_arr[None]).argmin(axis=1)
    return np.ascontiguousarray(snapshots[nearest])


class FakeBackend:
    """Deterministic stand-in with the same interface: integrates a constant per-tick velocity from the anchor.

    ``velocity`` is in model units per tick; channels in ``contract.absolute_dims`` are set to ``absolute``
    instead (a gripper target). ``delay_s`` simulates inference latency; every request is kept in ``requests``.
    """

    def __init__(
        self,
        contract: ModelContract,
        velocity: np.ndarray | float = 0.0,
        absolute: np.ndarray | float | None = None,
        delay_s: float = 0.0,
        output: np.ndarray | None = None,
    ):
        self.contract = contract
        self.velocity = np.broadcast_to(np.asarray(velocity, dtype=np.float64), (contract.action_dim,)).copy()
        self.absolute = None if absolute is None else np.broadcast_to(absolute, (contract.action_dim,)).copy()
        self.delay_s = delay_s
        self.output = output
        self.requests: list[PolicyRequest] = []
        self._lock = threading.Lock()

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        validate_request(request, self.contract)
        with self._lock:
            start = time.perf_counter()
            self.requests.append(request)
            if self.delay_s:
                time.sleep(self.delay_s)
            if self.output is not None:
                chunk = np.array(self.output, dtype=np.float64)
            else:
                anchor = request.commands[-1]
                steps = np.arange(1, self.contract.chunk_size + 1)[:, None]
                chunk = anchor[None] + steps * self.velocity[None]
                if self.absolute is not None:
                    dims = list(self.contract.absolute_dims)
                    chunk[:, dims] = self.absolute[dims]
            return PolicyResponse(chunk, time.perf_counter() - start, {"backend": "fake"})


def contract_from_flux3_config(config: Any, name: str) -> ModelContract:
    """Read the contract of a LeRobot ``Flux3Config`` (history-conditioned packages only)."""
    if config.conditioning != "history":
        raise ValueError("the runner keeps history itself; only history-conditioned flux3 packages are supported")
    if config.action_representation == "delta" and not config.condition_on_past_actions and config.n_obs_steps > 1:
        pass  # still valid: commands only anchor the deltas
    cams = list(config.camera_order)
    shapes = [tuple(config.input_features[k].shape[-2:]) for k in cams]
    hw = (max(s[0] for s in shapes), max(s[1] for s in shapes))
    absolute = tuple(sorted(d % config.action_dim for d in config.delta_absolute_dims))
    return ModelContract(
        name=name,
        action_dim=int(config.action_dim),
        camera_keys=tuple(cams),
        n_obs_steps=int(config.n_obs_steps),
        history_snapshots=int(config.history_snapshots),
        chunk_size=int(config.chunk_size),
        n_action_steps=int(config.n_action_steps),
        fps=float(config.fps),
        image_hw=hw,
        absolute_dims=absolute,
    )


class LeRobotFlux3Backend:
    """A loaded LeRobot ``Flux3Policy`` plus its saved processors, called statelessly per request.

    Load from a package with :meth:`from_pretrained` (GPU box), or pass an already built policy and processors
    (tests build a tiny random one on CPU).
    """

    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any, name: str = "flux3"):
        self.policy = policy.eval()
        self.pre = preprocessor
        self.post = postprocessor
        self.contract = contract_from_flux3_config(policy.config, name)
        if snapshot_indices(self.contract.n_obs_steps, self.contract.history_snapshots) != self.contract.snapshots:
            raise AssertionError("snapshot rule drifted from the model's packer")
        self._lock = threading.Lock()
        self._imagine = None  # imagine.serving.ImagineService, built on the first request that asks for frames

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: str = "cuda",
        revision: str | None = None,
        overrides: dict[str, Any] | None = None,
        text_encoder_on_cpu: bool = False,
    ) -> LeRobotFlux3Backend:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.flux3 import Flux3Policy

        kwargs = {"revision": revision} if revision else {}
        policy = Flux3Policy.from_pretrained(checkpoint, **kwargs)
        for key, value in (overrides or {}).items():
            if not hasattr(policy.config, key):
                raise AttributeError(f"Flux3Config has no field {key!r}")
            setattr(policy.config, key, value)
        policy.config.device = device
        policy.to(device).eval()
        if text_encoder_on_cpu:
            # About 8.3 GiB of Qwen weights move to host RAM; captions are cached (lerobot docs flux3.mdx:234-242).
            policy.frozen.text_encoder.cpu()
        pre, post = make_pre_post_processors(
            policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
            **kwargs,
        )
        return cls(policy, pre, post, name=checkpoint)

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        import torch

        contract = self.contract
        validate_request(request, contract)
        batch: dict[str, Any] = {
            OBS_STATE: torch.as_tensor(request.states, dtype=torch.float32)[None],
            COMMAND_HISTORY: torch.as_tensor(request.commands, dtype=torch.float32)[None],
            "task": [request.task],
        }
        for key in contract.camera_keys:
            frames = expand_snapshots(request.images[key], tuple(request.snapshot_ticks), contract.n_obs_steps)
            batch[key] = torch.from_numpy(frames).permute(0, 3, 1, 2)[None]  # (1, n, 3, H, W) uint8
        imagination = None
        with self._lock:
            start = time.perf_counter()
            prepared = self.pre(batch)
            if request.imagine:
                # Same predict_action_chunk call and output tensor, plus the video stream it already denoises.
                normalized, meta, arrays = self.imagine_service().predict(prepared, request.imagine)
                imagination = {"meta": meta, "arrays": arrays}
            else:
                normalized = self.policy.predict_action_chunk(prepared)  # (1, chunk, D), normalized deltas
            commands = self.post(normalized)  # (1, chunk, D), absolute, integrated onto the last command
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
        info: dict[str, Any] = {"backend": "lerobot_flux3"}
        if torch.cuda.is_available():
            info["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        out = commands[0].detach().to("cpu", torch.float64).numpy()
        return PolicyResponse(out, latency, info, imagination)

    def imagine_service(self):
        """The imagination service of this model (ws4/imagine/serving.py; ws4/ is on sys.path)."""
        if self._imagine is None:
            from imagine.serving import ImagineService

            self._imagine = ImagineService(self.policy, self._lock)
        return self._imagine

    def fetch_imagination(self, imagination_id: str, wait_s: float = 30.0):
        """``(meta, arrays)`` of an ``async`` / ``lazy`` imagination, decoding it now if needed."""
        return self.imagine_service().fetch(imagination_id, wait_s)
