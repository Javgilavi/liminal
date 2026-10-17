"""Read what a gated FLUX checkpoint contains without downloading its weights.

* ``--keys``: the safetensors header (names, shapes, dtypes) through one HTTP range request, e.g. to find the
  EE50 head names and widths in ``flux-3-action-base.safetensors`` (its README says "ee50/gaming action heads";
  their key names decide which ``action_modality`` reuses them, lerobot f3/wiring.py:131-156).
* ``--quantiles``: the history quantiles saved in a package's processor state files (tiny downloads), to compare
  the SO-101 package's units against our arms.

Needs ``hf auth login`` with access to the gated repos (or ``HF_TOKEN``).

    python -m flux.inspect_checkpoint --repo black-forest-labs/flux-3-action-base \
        --file flux-3-action-base.safetensors --keys --grep 'emb_in|final_layer'
    python -m flux.inspect_checkpoint --repo black-forest-labs/flux-3-action-so101 --quantiles
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import urllib.request


def _token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        import os

        return os.environ.get("HF_TOKEN")


def _get(url: str, headers: dict[str, str]) -> bytes:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def safetensors_header(repo: str, filename: str, revision: str = "main") -> dict:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    headers = {"Range": "bytes=0-7"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    (length,) = struct.unpack("<Q", _get(url, headers)[:8])
    headers["Range"] = f"bytes=8-{8 + length - 1}"
    return json.loads(_get(url, headers))


def history_quantiles(repo: str, revision: str = "main") -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    names = [
        "policy_preprocessor_step_3_flux3_observation_history_normalizer.safetensors",
        "policy_preprocessor_step_4_flux3_action_target_normalizer.safetensors",
        "policy_postprocessor_step_0_flux3_action_history_unnormalizer.safetensors",
    ]
    out = {}
    for name in names:
        tensors = load_file(hf_hub_download(repo, name, revision=revision))
        out[name] = {k: [round(float(x), 4) for x in v] for k, v in tensors.items()}
    for extra in ("dataset_statistics.json", "config.json"):
        try:
            out[extra] = json.loads(open(hf_hub_download(repo, extra, revision=revision)).read())
        except Exception as exc:  # optional files
            out[extra] = f"unavailable: {exc}"
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--file", default="model.safetensors")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--keys", action="store_true")
    parser.add_argument("--grep", default=None, help="regex filter on tensor names")
    parser.add_argument("--quantiles", action="store_true")
    args = parser.parse_args(argv)
    if args.keys:
        header = safetensors_header(args.repo, args.file, args.revision)
        meta = header.pop("__metadata__", None)
        pattern = re.compile(args.grep) if args.grep else None
        n_params = 0
        for name, info in sorted(header.items()):
            numel = 1
            for s in info["shape"]:
                numel *= s
            n_params += numel
            if pattern is None or pattern.search(name):
                print(f"{name}  {info['dtype']}  {info['shape']}")
        print(f"{len(header)} tensors, {n_params / 1e9:.3f} B parameters, metadata={meta}")
    if args.quantiles:
        print(json.dumps(history_quantiles(args.repo, args.revision), indent=1))


if __name__ == "__main__":
    main()
