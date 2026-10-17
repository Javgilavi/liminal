"""Serve ACT checkpoints with ``flux.remote``'s protocol, so ``RemoteBackend`` (and ``run_single_arm.py --backend
remote``) can call them exactly like a FLUX package. Same transport, token and 127.0.0.1-only binding as
``python -m flux.remote``; from another machine, reach it through ``ssh -L``.

    python ws4/act/serve.py --package act_100k=/workspace/act/runs/act_pen84/checkpoints/100000/pretrained_model \\
        --port 8791 [--token-file /workspace/flux_token]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WS4 = Path(__file__).resolve().parents[1]
if str(WS4) not in sys.path:
    sys.path.insert(0, str(WS4))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--package", action="append", required=True, metavar="NAME=PRETRAINED_MODEL_DIR[@DEVICE]")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--token-file", default=None, help="file holding the shared token (else $FLUX_TOKEN)")
    p.add_argument("--act-steps", type=int, default=None, help="commands executed per chunk (default: checkpoint's)")
    args = p.parse_args(argv)
    from flux.remote import make_server

    from act.backend import ActBackend

    backends = {}
    for item in args.package:
        name, spec = item.split("=", 1)
        path, _, device = spec.partition("@")
        backends[name] = ActBackend.from_pretrained(path, device=device or "auto", n_action_steps=args.act_steps,
                                                    name=f"{name} ({path})")  # fmt: skip
        print(f"{name}: {path} -> {backends[name].contract}", flush=True)
    token = os.environ.get("FLUX_TOKEN")
    if args.token_file:
        with open(args.token_file) as f:
            token = f.read().strip()
    server = make_server(backends, args.host, args.port, token)
    print(f"serving {sorted(backends)} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
