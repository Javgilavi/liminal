"""Official LeRobot FLUX policy/processors behind an SSH NDJSON transport.

This process has no robot access. Inputs contain explicit eight-frame histories;
outputs are absolute commands produced by the checkpoint's saved postprocessor.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path


def processor_contract(pre, post, checkpoint: str) -> dict:
    """Report the quantiles actually loaded, including both sides of action scaling."""
    saved = {}
    for step in [*pre.steps, *post.steps]:
        values = step.state_dict()
        if "action.q01" in values:
            saved[type(step).__name__] = {k: v.float().cpu().tolist() for k, v in values.items()}
    observation = [v for v in saved.values() if "state.q01" in v]
    if len(saved) != 3 or len(observation) != 1:
        raise ValueError("expected all three saved FLUX history quantile processors")
    values = observation[0]
    for item in saved.values():
        if any(item[k] != values[k] for k in ("action.q01", "action.q99")):
            raise ValueError("FLUX processors disagree on action normalization")
    statistics = {s: {q: values[f"{s}.{q}"] for q in ("q01", "q99")} for s in ("state", "action")}
    path = Path(checkpoint) / "normalization_source.json"
    source = json.loads(path.read_text()) if path.exists() else {}
    profile = source.get("profile", "released")
    if profile == "hf_space_native" and source.get("normalization") != statistics:
        raise ValueError("HF Space normalization provenance differs from loaded processor tensors")
    return {"normalization_profile": profile, "state_statistics": statistics["state"],
            "action_statistics": statistics["action"], "normalization_source": source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/liminal/models/so101")
    parser.add_argument("--normalization-profile", choices=("released", "hf_space_native"), default="released")
    args = parser.parse_args()
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.flux3.modeling_flux3 import Flux3Policy
        from lerobot.utils.utils import init_logging
        from PIL import Image

        init_logging(console_level="INFO")
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(0.40)
        policy = Flux3Policy.from_pretrained(args.checkpoint).to("cuda").eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
        normalization = processor_contract(pre, post, args.checkpoint)
        if normalization["normalization_profile"] != args.normalization_profile:
            raise ValueError("loaded processor normalization does not match the requested profile")
        policy.reset()
        pre.reset()
        post.reset()
        cfg = policy.config
        if (cfg.n_obs_steps, cfg.chunk_size, cfg.fps) != (8, 42, 30):
            raise ValueError("unsupported checkpoint history/horizon/rate")
        if list(cfg.camera_keys) != ["observation.images.scene", "observation.images.wrist"]:
            raise ValueError("unsupported camera contract")
        revision = subprocess.check_output(
            ["git", "-C", "/liminal/src/lerobot-lora", "rev-parse", "HEAD"], text=True).strip()
    print(json.dumps({"type": "ready", "integration": "lerobot.Flux3Policy + saved pre/post processors",
                      "checkpoint": args.checkpoint, "source_revision": revision, **normalization,
                      "inference": {key: getattr(cfg, key) for key in (
                          "chunk_size", "n_action_steps", "fps", "num_inference_steps", "guidance_scale",
                          "guidance_scale_action", "sampler", "sampler_shift", "inference_seed")}}),
          file=wire, flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            history = request["history"]
            if len(history) != 8 or not str(request["prompt"]).strip():
                raise ValueError("eight observations and a prompt are required")
            batch = {"task": [request["prompt"]]}
            for source, destination in (("state", "observation.state"),
                                        ("command_history", "observation.command_history")):
                values = np.asarray([f[source] for f in history], dtype=np.float32)
                if values.shape != (8, 6) or not np.isfinite(values).all():
                    raise ValueError(f"invalid {source}")
                batch[destination] = torch.from_numpy(values)[None].to("cuda")
            for role in ("scene", "wrist"):
                frames = []
                for frame in history:
                    with Image.open(io.BytesIO(base64.b64decode(frame[role], validate=True))) as image:
                        frames.append(np.asarray(image.convert("RGB")).copy())
                batch[f"observation.images.{role}"] = (
                    torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to("cuda").float() / 255)
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                start = time.perf_counter()
                processed = pre(batch)
                normalized = policy.predict_action_chunk(processed)
                actions = post(normalized).float().cpu().numpy()
            if actions.shape != (1, 42, 6) or not np.isfinite(actions).all():
                raise ValueError("invalid official processor output")
            reply = {"type": "actions", "actions": actions[0].tolist(),
                     "inference_seconds": time.perf_counter() - start}
        except Exception as exc:
            reply = {"type": "error", "detail": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply, allow_nan=False), file=wire, flush=True)


if __name__ == "__main__":
    main()
