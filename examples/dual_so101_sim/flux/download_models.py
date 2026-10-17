"""Populate the pinned FLUX cache without writing authentication credentials."""

from __future__ import annotations

import os


def main() -> None:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    entries = (
        ("black-forest-labs/flux-3-action-so101", "c9e13b2aca6a0a472b3ea03fd90cff32d3e85849", None),
        ("black-forest-labs/flux-3-action-base", "62878e2925e59b7a89ec14463ce89932624c490d",
         ["video_vae.safetensors", "text_encoder/*"]),
    )
    for repo, revision, patterns in entries:
        print(f"Caching {repo}@{revision}", flush=True)
        path = snapshot_download(repo, revision=revision, allow_patterns=patterns, token=token, max_workers=4)
        print(f"Cached {repo} in {path}", flush=True)
    print("FLUX_MODELS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
