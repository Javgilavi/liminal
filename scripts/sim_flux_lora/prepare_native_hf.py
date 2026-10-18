"""Prepare a separate frozen base with HF statistics and the recordings' true rate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main(root: Path, source: Path) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    root.mkdir(parents=True, exist_ok=True)
    stats = json.loads((source / "hf-sim-normalization.json").read_text())
    base = Path("/liminal/models/so101")
    prepared = root / "base-native-hf"
    prepared.mkdir(exist_ok=False)
    for file in base.iterdir():
        target = prepared / file.name
        if file.name == "config.json":
            config = json.loads(file.read_text())
            config["fps"] = 20.0
            config["normalization_stats"] = stats
            target.write_text(json.dumps(config, indent=2) + "\n")
        elif file.name.startswith("policy_") and "processor_step_" in file.name and file.suffix == ".safetensors":
            values = load_file(str(file))
            for key in values:
                stream, quantile = key.split(".")
                values[key] = torch.tensor(stats[stream][quantile], dtype=values[key].dtype)
            save_file(values, str(target))
        elif file.suffix == ".json":
            shutil.copy2(file, target)
        else:
            target.symlink_to(file.resolve(), target_is_directory=file.is_dir())
    assert len(list(prepared.glob("policy_*processor_step_*.safetensors"))) == 3
    for path in prepared.glob("policy_*processor_step_*.safetensors"):
        for key, value in load_file(str(path)).items():
            stream, quantile = key.split(".")
            assert torch.equal(value, torch.tensor(stats[stream][quantile], dtype=value.dtype))
    config = json.loads((source / "train-config.json").read_text())
    config["dataset"]["root"] = str(root / "dataset-native-v3")
    config["dataset"]["repo_id"] = "local/liminal_so101_native_hf_pick_place"
    config["output_dir"] = str(root / "training-1000")
    (root / "train-config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ["collection-inference-audit.json", "old-lora-deleted.json", "hf-sim-normalization.json"]:
        shutil.copy2(source / name, root / name)
    provenance = {
        "source_recordings": str(source / "raw"),
        "base_weights": str(base),
        "prepared_base": str(prepared),
        "coordinate_frame": "native joint degrees; gripper percent",
        "normalization": stats,
        "fps": 20,
        "timing": "10 physics steps of 5 ms per command; corrected dataset, policy and video rate",
        "old_adapter_reused": False,
        "training_budget": "1000 microbatches, 250 optimizer updates, matching the earlier initial experiment",
        "normalization_source": json.loads((source / "hf-sim-normalization-source.json").read_text()),
    }
    original_provenance = json.loads((source / "provenance.json").read_text())
    for key in ["official_docs", "bfl_recipe", "lerobot_revision", "version_note"]:
        provenance[key] = original_provenance[key]
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.source)
