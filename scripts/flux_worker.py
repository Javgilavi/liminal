"""GPU-side NDJSON worker. No robot access; one model, one request at a time."""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import io
import json
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/liminal/models/so101")
    parser.add_argument("--n-action-steps", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--guidance-scale-action", type=float)
    parser.add_argument("--sampler-shift", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    # Reserve stdout for the wire, including during third-party imports/loading.
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from flux_action.inference.so101 import load_policy
        from PIL import Image

        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(0.40)
        policy = load_policy(args.checkpoint).to("cuda")
        cfg = policy.config
        if (cfg.action_dim, cfg.n_obs_steps, cfg.fps, cfg.chunk_size) != (6, 8, 30, 42):
            raise ValueError("unsupported checkpoint control contract")
        if tuple(cfg.camera_order) != ("images.scene", "images.wrist"):
            raise ValueError(f"unsupported camera order: {cfg.camera_order}")
        overrides = {name: value for name, value in (
            ("n_action_steps", args.n_action_steps), ("num_inference_steps", args.num_inference_steps),
            ("guidance_scale", args.guidance_scale), ("guidance_scale_action", args.guidance_scale_action),
            ("sampler_shift", args.sampler_shift), ("inference_seed", args.seed),
        ) if value is not None}
        cfg = dataclasses.replace(cfg, **overrides)
        cfg.validate_inference()
        policy.config = cfg
        # Rebuild history/queue after overrides. Explicit request windows supply
        # all subsequent history; keep the text cache warm between predictions.
        policy.reset()
        settings = {"chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps,
                    "control_hz": cfg.fps, "num_inference_steps": cfg.num_inference_steps,
                    "sampler": cfg.sampler, "guidance_scale": cfg.guidance_scale,
                    "guidance_scale_action": cfg.guidance_scale_action, "sampler_shift": cfg.sampler_shift,
                    "seed": cfg.inference_seed, "n_obs_steps": cfg.n_obs_steps,
                    "history_snapshots": cfg.history_snapshots}
    stats = json.loads((Path(args.checkpoint) / "dataset_statistics.json").read_text())
    print(json.dumps({"type": "ready", "state_statistics": stats["state"], "checkpoint": args.checkpoint,
                      "inference": settings}),
          file=wire, flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            history = request["history"]
            if len(history) != 8 or not str(request["prompt"]).strip():
                raise ValueError("eight real observations and a task instruction are required")
            batch = {"task": [request["prompt"]]}
            for key in ("state", "command_history"):
                data = np.asarray([frame[key] for frame in history], dtype=np.float32)
                if data.shape != (8, 6) or not np.isfinite(data).all():
                    raise ValueError(f"invalid {key}")
                batch[key] = torch.from_numpy(data)[None].to("cuda")
            for role in ("scene", "wrist"):
                frames = []
                for frame in history:
                    raw = base64.b64decode(frame[role], validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        frames.append(np.asarray(image.convert("RGB").resize((256, 256))).copy())
                batch[f"images.{role}"] = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to("cuda")
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                started = time.perf_counter()
                chunk = policy.predict_action_chunk(batch).float().cpu().numpy()
            if chunk.shape != (1, 42, 6) or not np.isfinite(chunk).all():
                raise ValueError("invalid model output")
            reply = {"type": "actions", "actions": chunk[0].tolist(),
                     "inference_seconds": time.perf_counter() - started}
        except Exception as exc:
            reply = {"type": "error", "detail": str(exc)}
        print(json.dumps(reply, allow_nan=False), file=wire, flush=True)


if __name__ == "__main__":
    main()
