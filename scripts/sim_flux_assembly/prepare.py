"""Prepare a fresh shared four-skill LoRA with fixed native/HF/20 Hz settings."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main(root: Path, steps: int):
    source = Path("/liminal/experiments/sim-task-native-hf-20261017")
    base = root / "base-native-hf"
    base.mkdir(exist_ok=False)
    for file in (source / "base-native-hf").iterdir():
        target = base / file.name
        if file.suffix == ".json" or file.name.startswith("policy_"):
            shutil.copy2(file, target)
        else:
            target.symlink_to(file.resolve(), target_is_directory=file.is_dir())
    config = json.loads((source / "train-config.json").read_text())
    config["dataset"]["repo_id"] = "local/liminal_nist_atomic_assembly"
    config["dataset"]["root"] = str(root / "dataset-v3")
    config["steps"] = steps
    config["save_freq"] = 250
    config["output_dir"] = str(root / "training")
    (root / "train-config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(source / "hf-sim-normalization.json", root / "hf-sim-normalization.json")
    provenance = json.loads((source / "provenance.json").read_text())
    provenance.update(
        {
            "experiment": "NIST gear and collar assembly, four prompt-conditioned atomic skills",
            "base": str(base),
            "steps": steps,
            "optimizer_updates": steps // 4,
            "old_pick_place_adapter_reused": False,
            "scene_assets": "assets/provenance.json",
            "source_recordings": str(root / "raw"),
            "prepared_base": str(base),
            "timing": "25 physics steps of 2 ms per command; dataset, policy and video at 20 Hz",
            "training_budget": f"{steps} microbatches, {steps // 4} optimizer updates",
        }
    )
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"steps": steps, "effective_batch": 8, "base": str(base)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    main(args.root, args.steps)
