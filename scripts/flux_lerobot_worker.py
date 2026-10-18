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
from types import SimpleNamespace

# The legacy SSH transport and BT runtime share one motor-free inference implementation.
from liminal.policy.lerobot_engine import processor_contract  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/liminal/models/so101")
    parser.add_argument("--normalization-profile", choices=("released", "hf_space_native", "dataset_native"),
                        default="released")
    args = parser.parse_args()
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from lerobot.utils.utils import init_logging
        from PIL import Image

        from liminal.core.types import Observation
        from liminal.policy.lerobot_engine import LeRobotEngine

        init_logging(console_level="INFO")
        torch.set_num_threads(8)
        torch.cuda.set_per_process_memory_fraction(0.40)
        contract = SimpleNamespace(device="cuda", obs_history=8, chunk_size=42, control_hz=30,
                                   camera_roles=("scene", "wrist"), action_dim=6)
        engine = LeRobotEngine(args.checkpoint, contract, args.normalization_profile)
        cfg = engine.policy.config
        normalization = engine.statistics
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
            observations = []
            for frame in history:
                images = {}
                for role in contract.camera_roles:
                    with Image.open(io.BytesIO(base64.b64decode(frame[role], validate=True))) as image:
                        images[role] = np.asarray(image.convert("RGB")).copy()
                observations.append(Observation("right", frame["state"], images))
            with contextlib.redirect_stdout(sys.stderr):
                start = time.perf_counter()
                actions = engine.predict(observations, [f["command_history"] for f in history], request["prompt"])
            reply = {"type": "actions", "actions": actions,
                     "inference_seconds": time.perf_counter() - start}
        except Exception as exc:
            reply = {"type": "error", "detail": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply, allow_nan=False), file=wire, flush=True)


if __name__ == "__main__":
    main()
