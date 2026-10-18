"""FLUX inference on the remote policy server instead of in-process CUDA (``engine: remote`` in a policy config).

The laptop GPU (4 GB) cannot hold FLUX. The server (ws4/cloud/lora_adapter.py ``serve``, protocol ws4/flux/remote.py)
serves one LoRA package per skill, each with its own normalization, and is reached through an SSH tunnel. It is
stateless: every call carries the whole window (states, previous commands, snapshot frames), so this engine keeps
LeRobotEngine's interface and FluxPolicyBackend keeps its history, chunking, safety and checkpoint switching. The
request is built with ws4's client (flux.remote.RemoteBackend), the one the Haller runs use.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


def policy_server_token(token_file: str = "", token_cmd_env: str = "WS4_TOKEN_CMD") -> str | None:
    """The shared token: a file, or the output of the command in ``$token_cmd_env`` (never on argv or in a log)."""
    if token_file:
        return Path(token_file).expanduser().read_text().strip()
    command = os.environ.get(token_cmd_env, "").strip() if token_cmd_env else ""
    if not command:
        return None
    out = subprocess.run(shlex.split(os.path.expanduser(command)), capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=30)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"${token_cmd_env} failed (exit {out.returncode}): {out.stderr.strip()[:200]}")
    return out.stdout.strip()


class RemoteFluxEngine:
    def __init__(self, cfg: Any, package: str):
        from liminal.ws4_bridge import ws4_on_path

        ws4_on_path()
        from flux.remote import RemoteBackend  # noqa: PLC0415

        if not cfg.server:
            raise ValueError("engine: remote needs a server URL")
        self.cfg, self.package = cfg, package
        self.backend = RemoteBackend(cfg.server, package, token=policy_server_token(cfg.token_file, cfg.token_cmd_env),
                                     timeout_s=cfg.inference_timeout_s)
        c = self.backend.contract
        if (c.n_obs_steps, c.chunk_size, c.fps, c.action_dim) != (
            cfg.obs_history, cfg.chunk_size, cfg.control_hz, cfg.action_dim
        ):
            raise ValueError(f"{package}: served contract {c} does not match the policy config")
        if list(c.camera_keys) != [f"observation.images.{role}" for role in cfg.camera_roles]:
            raise ValueError(f"{package}: served camera keys {c.camera_keys} differ from roles {cfg.camera_roles}")
        if c.output_space != "joints":
            raise ValueError(f"{package}: served output space {c.output_space!r}, expected joints")
        self.contract = c

    def identity(self) -> dict[str, str]:
        return {"engine": "remote", "server": self.cfg.server, "package": self.package, "contract": self.contract.name}

    def reset(self) -> None:
        """Nothing server-side: the history lives in FluxPolicyBackend and travels with every request."""

    def predict(self, history: list, commands: list[list[float]], instruction: str) -> list[list[float]]:
        import numpy as np
        from flux.backends import PolicyRequest  # noqa: PLC0415
        from flux.cameras import CameraView, prepare_image  # noqa: PLC0415

        cfg, c = self.cfg, self.contract
        if len(history) != c.n_obs_steps or len(commands) != c.n_obs_steps or not instruction.strip():
            raise ValueError("complete synchronized history and a prompt are required")
        arrays = {}
        for key, values in (("state", [o.state for o in history]), ("command", commands)):
            arrays[key] = np.asarray(values, dtype=np.float64)
            if arrays[key].shape != (c.n_obs_steps, cfg.action_dim) or not np.isfinite(arrays[key]).all():
                raise ValueError(f"invalid {key} history")
        images = {}
        for role, key in zip(cfg.camera_roles, c.camera_keys, strict=True):
            # as ws4's runner: only the snapshot ticks are encoded, each resized to the served image size
            images[key] = np.stack([prepare_image(np.asarray(history[t].images[role]), CameraView(key), c.image_hw)
                                    for t in c.snapshots])
        request = PolicyRequest(instruction, arrays["state"], arrays["command"], images, c.snapshots,
                                target=history[-1].arm)
        actions = np.asarray(self.backend.predict(request).commands, dtype=np.float64)
        if actions.shape != (cfg.chunk_size, cfg.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"{self.package}: invalid action chunk {actions.shape}")
        return actions.tolist()
