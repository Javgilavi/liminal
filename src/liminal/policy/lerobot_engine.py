"""Motor-free official FLUX inference, with explicit measured and commanded history.

Uses the same API exercised by scripts/flux_lerobot_worker.py. Heavy imports stay
inside methods so orchestration and its tests do not require a GPU installation.
"""
from __future__ import annotations

from typing import Any


def processor_contract(pre: Any, post: Any, checkpoint: str) -> dict:
    saved = []
    for step in [*pre.steps, *post.steps]:
        values = step.state_dict()
        if "action.q01" in values:
            saved.append({k: v.float().cpu().tolist() for k, v in values.items()})
    observations = [v for v in saved if "state.q01" in v]
    if len(saved) != 3 or len(observations) != 1:
        raise ValueError("expected all three saved FLUX history quantile processors")
    values = observations[0]
    if any(item[k] != values[k] for item in saved for k in ("action.q01", "action.q99")):
        raise ValueError("FLUX processors disagree on action normalization")
    import json
    from pathlib import Path

    statistics = {s: {q: values[f"{s}.{q}"] for q in ("q01", "q99")} for s in ("state", "action")}
    source_path = Path(checkpoint) / "normalization_source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    profile = source.get("profile", "released")
    if profile == "hf_space_native" and source.get("normalization") != statistics:
        raise ValueError("normalization provenance differs from saved processor tensors")
    return {"normalization_profile": profile, "state_statistics": statistics["state"],
            "action_statistics": statistics["action"], "normalization_source": source}


class LeRobotEngine:
    def __init__(self, checkpoint: str, cfg: Any, normalization_profile: str = "released"):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.flux3.modeling_flux3 import Flux3Policy

        self.cfg = cfg
        self.policy = Flux3Policy.from_pretrained(checkpoint).to(cfg.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=checkpoint)
        self.statistics = processor_contract(self.pre, self.post, checkpoint)
        if self.statistics["normalization_profile"] != normalization_profile:
            raise ValueError("normalization profile does not match saved processors")
        model_cfg = self.policy.config
        if (model_cfg.n_obs_steps, model_cfg.chunk_size, model_cfg.fps) != (
            cfg.obs_history, cfg.chunk_size, cfg.control_hz
        ):
            raise ValueError("checkpoint history/horizon/rate is incompatible")
        if list(model_cfg.camera_keys) != [f"observation.images.{role}" for role in cfg.camera_roles]:
            raise ValueError("checkpoint camera roles/order are incompatible")
        # VERIFY on each deployment's LeRobot revision; mismatch fails before motion.
        if model_cfg.output_features["action"].shape[-1] != cfg.action_dim:
            raise ValueError("checkpoint action dimension is incompatible")
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def predict(self, history: list, commands: list[list[float]], instruction: str) -> list[list[float]]:
        import numpy as np
        import torch

        cfg = self.cfg
        if len(history) != cfg.obs_history or len(commands) != cfg.obs_history or not instruction.strip():
            raise ValueError("complete synchronized history and a prompt are required")
        batch = {"task": [instruction]}
        for key, values in (("state", [o.state for o in history]), ("command_history", commands)):
            array = np.asarray(values, dtype=np.float32)
            if array.shape != (cfg.obs_history, cfg.action_dim) or not np.isfinite(array).all():
                raise ValueError(f"invalid {key} history")
            batch[f"observation.{key}"] = torch.from_numpy(array)[None].to(cfg.device)
        for role in cfg.camera_roles:
            frames = np.stack([o.images[role] for o in history])
            if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
                raise ValueError(f"{role}: expected RGB uint8 frames")
            batch[f"observation.images.{role}"] = (
                torch.from_numpy(frames).permute(0, 3, 1, 2)[None].to(cfg.device).float() / 255
            )
        with torch.inference_mode():
            actions = self.post(self.policy.predict_action_chunk(self.pre(batch))).float().cpu().numpy()
        if actions.shape != (1, cfg.chunk_size, cfg.action_dim) or not np.isfinite(actions).all():
            raise ValueError("invalid official processor output")
        return actions[0].tolist()
