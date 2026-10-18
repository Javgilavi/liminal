"""Merge a pinned SO101 task adapter with its saved processors; no robot access."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/liminal/models/so101"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.flux3.modeling_flux3 import Flux3Policy
    from peft import PeftModel
    from safetensors.numpy import load_file

    torch.set_num_threads(8)
    # Check all saved processors before allocating the model or writing output.
    tensors = []
    files = ["policy_preprocessor.json", "policy_postprocessor.json"]
    for filename in files.copy():
        for step in json.loads((args.adapter / filename).read_text())["steps"]:
            if "state_file" in step:
                name = step["state_file"]
                if Path(name).name != name:
                    raise ValueError("processor state must be a local filename")
                files.append(name)
                tensors.append(load_file(str(args.adapter / name)))
    observations = [v for v in tensors if "state.q01" in v]
    if len(tensors) != 3 or len(observations) != 1:
        raise ValueError("expected three saved history normalization states")
    values = observations[0]
    for item in tensors:
        for key in ("action.q01", "action.q99"):
            if not np.array_equal(item[key], values[key]):
                raise ValueError("inconsistent action normalization")
    statistics = {s: {q: values[f"{s}.{q}"].tolist() for q in ("q01", "q99")}
                  for s in ("state", "action")}
    source = json.loads((args.adapter / "normalization_source.json").read_text())
    for stream, bounds in statistics.items():
        for quantile, value in bounds.items():
            if not np.allclose(value, source["stats"][stream][quantile], atol=1e-5, rtol=1e-6):
                raise ValueError("adapter normalization provenance differs from saved tensors")

    args.output.mkdir(parents=True, exist_ok=False)
    config = PreTrainedConfig.from_pretrained(str(args.adapter))
    config.use_peft = False
    config.device = "cpu"  # Merge on CPU without competing with other GPU services.
    config.guidance_scale = 1.0
    config.guidance_scale_action = 1.0
    config.gradient_checkpointing = False
    config.pretrained_path = str(args.base)
    print("Loading base and merging adapter on CPU", flush=True)
    policy = Flux3Policy.from_pretrained(str(args.base), config=config)
    policy = PeftModel.from_pretrained(policy, str(args.adapter), autocast_adapter_dtype=False)
    policy = policy.merge_and_unload(safe_merge=True)
    policy.config.device = "cuda"
    policy.config.pretrained_path = str(args.output)
    policy.config.use_peft = False
    policy.save_pretrained(args.output)
    for filename in files + ["LICENSE.md", "NOTICE", "README.md"]:
        shutil.copy2(args.adapter / filename, args.output / filename)
    provenance = {
        "profile": "dataset_native",
        "adapter_repo": "Oskrt/flux-3-action-so101-haller-pen-lora",
        "adapter_revision": args.revision,
        "adapter_sha256": hashlib.file_digest(
            (args.adapter / "adapter_model.safetensors").open("rb"), "sha256").hexdigest(),
        "base_checkpoint": str(args.base),
        "normalization": statistics,
        "training_normalization_source": source,
        "coordinate_mapping": "identity; native degrees and gripper percent",
    }
    (args.output / "normalization_source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(args.output), **provenance}), flush=True)


if __name__ == "__main__":
    main()
