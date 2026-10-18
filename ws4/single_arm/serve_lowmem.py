"""``flux.remote`` policy server for a 24 GB GPU (RTX 4090): same HTTP protocol and token handling
(``flux.remote.make_server``), but the model is loaded with ``offline_diag.load_backend``, which never puts Qwen on the
GPU. ``LeRobotFlux3Backend.from_pretrained`` moves the whole policy to the GPU before moving Qwen back (23.6 GiB peak),
which does not fit 23.5 GiB.

    python ws4/single_arm/serve_lowmem.py --package so101=black-forest-labs/flux-3-action-so101 --guidance 1.0 \\
        --token-file /workspace/flux_token --port 8765
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--package", action="append", required=True, metavar="NAME=CHECKPOINT")
    p.add_argument("--guidance", type=float, default=1.0, help="video and action guidance")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token-file", required=True)
    args = p.parse_args()
    from flux.remote import make_server
    from offline_diag import load_backend

    overrides = {"guidance_scale": args.guidance, "guidance_scale_action": args.guidance, "compile_model": args.compile}
    loaded, backends = {}, {}
    for item in args.package:
        name, checkpoint = item.split("=", 1)
        if checkpoint not in loaded:
            loaded[checkpoint] = load_backend(checkpoint, overrides, text_encoder_on_cpu=True)
        backends[name] = loaded[checkpoint]
        print(f"{name}: {checkpoint} -> {backends[name].contract}", flush=True)
    token = Path(args.token_file).read_text().strip()
    server = make_server(backends, args.host, args.port, token)
    print(f"serving {sorted(backends)} on http://{args.host}:{args.port} (guidance {args.guidance})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
