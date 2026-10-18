"""The GPU worker of liminal.policy.official_live, on the ws4 policy server instead of the GCP VM.

Same NDJSON protocol as scripts/flux_lerobot_worker.py (a ready line, then one request line in, one actions line
out), run locally: each request goes to the server (ws4/cloud/lora_adapter.py serve, 8790 through an SSH tunnel)
through liminal.policy.remote_engine, so the Flux3Policy + saved processors of one served LoRA package answer it.
No robot access.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8790")
    parser.add_argument("--package", required=True, help="e.g. skill_red_pen")
    parser.add_argument("--n-action-steps", type=int, default=16, help="commands executed per 42-step chunk")
    parser.add_argument("--token-cmd-env", default="WS4_TOKEN_CMD")
    parser.add_argument("--token-file", default="")
    args = parser.parse_args()
    wire = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        from PIL import Image

        from liminal.core.types import Observation
        from liminal.policy.flux import FluxConfig
        from liminal.policy.remote_engine import RemoteFluxEngine

        cfg = FluxConfig(engine="remote", server=args.server, package=args.package, token_file=args.token_file,
                         token_cmd_env=args.token_cmd_env, chunk_size=42, obs_history=8, control_hz=30,
                         execute_steps=args.n_action_steps)
        engine = RemoteFluxEngine(cfg, args.package)
    print(json.dumps({"type": "ready",
                      "integration": f"lerobot.Flux3Policy + saved processors on the ws4 policy server "
                                     f"({args.server}, package {args.package})",
                      "checkpoint": args.package, "contract": engine.contract.name,
                      "inference": {"chunk_size": 42, "n_action_steps": args.n_action_steps, "fps": 30}}),
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
                for role in cfg.camera_roles:
                    with Image.open(io.BytesIO(base64.b64decode(frame[role], validate=True))) as image:
                        images[role] = np.asarray(image.convert("RGB")).copy()
                observations.append(Observation("right", frame["state"], images))
            with contextlib.redirect_stdout(sys.stderr):
                start = time.perf_counter()
                actions = engine.predict(observations, [f["command_history"] for f in history], request["prompt"])
            reply = {"type": "actions", "actions": actions, "inference_seconds": time.perf_counter() - start}
        except Exception as exc:
            reply = {"type": "error", "detail": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(reply, allow_nan=False), file=wire, flush=True)


if __name__ == "__main__":
    main()
