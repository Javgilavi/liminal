"""LeRobot ACT checkpoint behind the runner's policy interface (``flux.backends.FluxBackend``).

ACT is the baseline and fallback for the FLUX 3 Action LoRA on the Haller pen task. It conditions on the current
state and the current frame of each camera only (``n_obs_steps`` 1) and predicts ``chunk_size`` absolute joint
targets in the dataset's units (LeRobot degrees, gripper 0..100), so the contract below asks the runner for a
one-tick window with one snapshot and full-resolution frames. The returned chunk is absolute model-unit commands,
exactly what ``DualArmRunner`` expects from a FLUX backend: every command still goes through its guard and the
single-arm runner's keep-out, envelope, ramps, stall/load freeze and torque cap. Nothing here talks to a motor.

The call is stateless: ``predict_action_chunk`` on one observation, without the policy's own action queue or
temporal ensembling, so one loaded model can serve any number of requests (one at a time, behind a lock).

    backend = ActBackend.from_pretrained("runs/act_pen84/checkpoints/080000/pretrained_model", device="cuda")
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from flux.backends import PolicyRequest, PolicyResponse, validate_request
from flux.contract import OBS_STATE, ModelContract

ACTION = "action"
IMAGE_PREFIX = "observation.images."


@contextlib.contextmanager
def compatible_checkpoint(checkpoint: str):
    """A temporary view of ``checkpoint`` (files symlinked) whose config.json loads offline on this LeRobot.

    - ``pretrained_backbone_weights`` is set to None: building ``ACTPolicy`` would otherwise download the ImageNet
      ResNet18 from download.pytorch.org before ``model.safetensors`` overwrites every backbone tensor (the frozen
      BatchNorm statistics included), so the robot computer needs no network.
    - Fields this LeRobot's ``ACTConfig`` lacks are dropped: the pod trains with LeRobot 146d847, whose
      ``PreTrainedConfig`` added ``dtype`` and ``pretrained_revision``, and the laptop's 0.5.1 refuses unknown
      fields. Both are None in our checkpoints (parameters stay float32). An unknown field with a value raises.
    """
    from lerobot.policies.act.configuration_act import ACTConfig

    src = Path(checkpoint)
    raw = json.loads((src / "config.json").read_text())
    known = {f.name for f in dataclasses.fields(ACTConfig)} | {"type"}
    extra = {k: v for k, v in raw.items() if k not in known}
    if any(v is not None for v in extra.values()):
        raise ValueError(f"{checkpoint}: config fields unknown to this LeRobot carry values: {extra}")
    config = {k: v for k, v in raw.items() if k in known}
    config["pretrained_backbone_weights"] = None
    with tempfile.TemporaryDirectory(prefix="act_ckpt_") as tmp:
        for item in src.iterdir():
            if item.name != "config.json":
                (Path(tmp) / item.name).symlink_to(item.resolve())
        (Path(tmp) / "config.json").write_text(json.dumps(config, indent=2))
        yield tmp


def contract_from_act_config(config: Any, name: str, n_action_steps: int | None = None,
                             fps: float = 30.0) -> ModelContract:  # fmt: skip
    """What an ``ACTConfig`` consumes: the current tick only, its cameras at their trained resolution."""
    cams = tuple(k for k in config.input_features if k.startswith(IMAGE_PREFIX))
    if not cams:
        raise ValueError("ACT checkpoint has no camera inputs")
    shapes = {tuple(config.input_features[k].shape[-2:]) for k in cams}  # (C, H, W) features
    if len(shapes) != 1:
        raise ValueError(f"cameras of different sizes are not supported: {shapes}")
    if int(config.n_obs_steps) != 1:
        raise ValueError("ACT conditions on one observation tick")
    action_dim = int(config.output_features[ACTION].shape[0])
    steps = int(n_action_steps or config.n_action_steps)
    return ModelContract(
        name=name,
        action_dim=action_dim,
        camera_keys=cams,
        n_obs_steps=1,
        history_snapshots=1,
        chunk_size=int(config.chunk_size),
        n_action_steps=steps,
        fps=fps,
        image_hw=shapes.pop(),
        absolute_dims=tuple(range(action_dim)),  # ACT predicts absolute targets on every channel
        extra={"policy": "act"},
    )


class ActBackend:
    """A loaded LeRobot ``ACTPolicy`` plus its saved processors, called statelessly per request.

    ``n_action_steps`` is how many of the predicted commands the runner executes before it re-plans (ACT's own
    default is the whole 100-step chunk). Build from a checkpoint with :meth:`from_pretrained`, or pass a policy
    and processors (tests build a tiny random one on the CPU).
    """

    def __init__(self, policy: Any, preprocessor: Any, postprocessor: Any, name: str = "act",
                 n_action_steps: int | None = None):  # fmt: skip
        self.policy = policy.eval()
        self.pre = preprocessor
        self.post = postprocessor
        self.contract = contract_from_act_config(policy.config, name, n_action_steps)
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, device: str = "auto", n_action_steps: int | None = None,
                        name: str | None = None) -> ActBackend:  # fmt: skip
        """``checkpoint``: a ``pretrained_model`` folder (config.json, model.safetensors, processor JSONs)."""
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        with compatible_checkpoint(checkpoint) as path:
            policy = ACTPolicy.from_pretrained(path)
            policy.config.device = device
            policy.to(device).eval()
            device_override = {"device_processor": {"device": device}}
            pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                                 preprocessor_overrides=device_override)  # fmt: skip
        return cls(policy, pre, post, name=name or checkpoint, n_action_steps=n_action_steps)

    def batch(self, request: PolicyRequest) -> dict[str, Any]:
        """The newest tick of the window as LeRobot's ``predict_action`` builds it: float images in [0, 1], CHW."""
        import torch

        batch: dict[str, Any] = {OBS_STATE: torch.as_tensor(request.states[-1], dtype=torch.float32)[None]}
        for key in self.contract.camera_keys:
            frame = np.ascontiguousarray(request.images[key][-1])  # (H, W, 3) uint8, the current frame
            batch[key] = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)[None]
        batch["task"] = [request.task]  # unused by ACT; kept so the processors see LeRobot's usual keys
        return batch

    def predict(self, request: PolicyRequest) -> PolicyResponse:
        import torch

        validate_request(request, self.contract)
        with self._lock, torch.inference_mode():
            start = time.perf_counter()
            prepared = self.pre(self.batch(request))
            normalized = self.policy.predict_action_chunk(prepared)  # (1, chunk, D)
            commands = self.post(normalized)  # (1, chunk, D) absolute, dataset units, on the CPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency = time.perf_counter() - start
        out = commands[0].detach().to("cpu", torch.float64).numpy()
        if out.shape != (self.contract.chunk_size, self.contract.action_dim) or not np.isfinite(out).all():
            raise RuntimeError(f"ACT returned {out.shape} (finite: {np.isfinite(out).all()})")
        return PolicyResponse(out, latency, {"backend": "lerobot_act", "device": str(self.policy.config.device)})
