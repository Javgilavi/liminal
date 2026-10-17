"""Pre-encode the demo's prompts on a running policy server, so no prompt is encoded live during the demo.

With the text encoder on the CPU (needed on the 32 GB RunPod card), the first request with a new prompt pays a
one-time encode of about 16 s (measured on the pod: the 17.3 s p90 outlier); later requests with the same prompt
use the cached embedding (median 1.46 s per chunk). This sends one synthetic request per (package, prompt)
right after the server starts.

    python ws4/cloud/warm_prompts.py --url http://127.0.0.1:8765 --token-file /workspace/flux_token \
        --prompts-file /workspace/prompts.txt

Prompts must match the strings the robot runner sends, character for character. Stateless: the server keeps no
history, so synthetic requests leave nothing behind but the cached text embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ws4/, for `flux`

# Skill prompts from ws4/research/xlerobot_recording_protocol.md (S1 to S6). Edit /workspace/prompts.txt on the
# pod to change them without touching code.
DEFAULT_PROMPTS = [
    "Push the bottle away from the box",
    "Open the box lid with the left arm and hold it open",
    "Put the apple in the box",
    "Put the yellow block in the box",
    "Close the box lid",
    "Hand the cup to the right arm",
]


def wait_for_server(url: str, token: str | None, timeout_s: float) -> list[str]:
    from flux.remote import TOKEN_HEADER

    deadline = time.monotonic() + timeout_s
    while True:
        req = urllib.request.Request(url.rstrip("/") + "/packages")
        if token:
            req.add_header(TOKEN_HEADER, token)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except OSError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"policy server at {url} not up after {timeout_s:.0f}s") from None
            time.sleep(5)


def read_prompts(path: Path | None) -> list[str]:
    if path is not None and path.exists():
        prompts = [line.strip() for line in path.read_text().splitlines()]
        return [p for p in prompts if p and not p.startswith("#")]
    return list(DEFAULT_PROMPTS)


def warm(url: str, token: str | None, packages: list[str], prompts: list[str]) -> list[dict]:
    from flux.measure_latency import synthetic_request
    from flux.remote import RemoteBackend

    rows = []
    for pkg in packages:
        client = RemoteBackend(url, pkg, token=token, timeout_s=300)
        for prompt in prompts:
            t0 = time.perf_counter()
            client.predict(synthetic_request(client.contract, prompt))
            rows.append({"package": pkg, "prompt": prompt, "seconds": round(time.perf_counter() - t0, 2)})
            print(f"warmed {pkg!r} {prompt!r} in {rows[-1]['seconds']} s", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--token-file", type=Path, default=Path("/workspace/flux_token"))
    ap.add_argument("--prompts-file", type=Path, default=Path("/workspace/prompts.txt"))
    ap.add_argument("--packages", nargs="*", default=None, help="default: every package the server lists")
    ap.add_argument("--wait-s", type=float, default=1200, help="how long to wait for the server to load")
    args = ap.parse_args(argv)

    token = args.token_file.read_text().strip() if args.token_file.exists() else None
    packages = args.packages or wait_for_server(args.url, token, args.wait_s)
    if args.packages:
        wait_for_server(args.url, token, args.wait_s)
    prompts = read_prompts(args.prompts_file)
    rows = warm(args.url, token, packages, prompts)
    print(json.dumps({"warmed": len(rows), "slowest_s": max((r["seconds"] for r in rows), default=0)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
