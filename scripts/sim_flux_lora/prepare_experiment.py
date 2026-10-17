"""Prepare a fresh native/HF/20 Hz simulation experiment; no training or robot access."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_hf_space_checkpoint import NORMALIZATION, SPACE_REVISION, prepare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    prepared = root / "base-native-hf"
    provenance = prepare(args.base, prepared)
    # prepare() links config.json. Replace the link before changing the sim rate.
    config_path = prepared / "config.json"
    policy_config = json.loads(config_path.read_text())
    policy_config.update(fps=20.0, normalization_stats=NORMALIZATION)
    config_path.unlink()
    config_path.write_text(json.dumps(policy_config, indent=2) + "\n")
    repo = Path(__file__).resolve().parents[2]
    train = json.loads((repo / "experiments/single-arm-v1/train.json").read_text())
    train["dataset"]["root"] = str(root / "dataset-native-v3")
    train["policy"]["path"] = str(prepared)
    train["output_dir"] = str(root / "training-1000")
    (root / "train-config.json").write_text(json.dumps(train, indent=2) + "\n")
    (root / "hf-sim-normalization.json").write_text(json.dumps(NORMALIZATION, indent=2) + "\n")
    provenance.update({
        "fps": 20,
        "coordinate_frame": "native",
        "space_revision": SPACE_REVISION,
        "lerobot_revision": "e624f3f7f8411ec3a02635d06e79373341e5ef35",
        "purpose": "fresh reproduction of the first single-arm validation recipe",
    })
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
