"""Create a separate inference package using the pinned HF Space statistics.

Model weights and config remain linked to the released package. Only processor
tensor files and provenance are written here; this script has no robot access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SPACE_REVISION = "d7da38e032da0e136d6de21c218dc616982a8cc9"
NORMALIZATION = {
    "state": {
        "q01": [-37.969, -99.316, -45.78, 22.987, -68.712, 0.45],
        "q99": [35.288, 43.076, 90.308, 95.704, 17.753, 40.324],
    },
    "action": {
        "q01": [-1.652, -3.235, -3.147, -2.198, -1.708, 0.0],
        "q99": [1.701, 3.487, 3.307, 2.072, 1.699, 40.733],
    },
}


def prepare(source: Path, output: Path) -> dict:
    import numpy as np
    from safetensors.numpy import load_file, save_file

    source = source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    if (config["conditioning"], config["action_representation"], config["delta_absolute_dims"]) != (
        "history", "delta", [-1]
    ):
        raise ValueError("expected the released SO101 history/delta processor contract")
    tensors = {f"{s}.{q}": np.asarray(v, dtype=np.float32)
               for s, bounds in NORMALIZATION.items() for q, v in bounds.items()}
    pre = json.loads((source / "policy_preprocessor.json").read_text())
    post = json.loads((source / "policy_postprocessor.json").read_text())
    required = {
        "flux3_observation_history_normalizer": tensors,
        "flux3_action_target_normalizer": {k: v for k, v in tensors.items() if k.startswith("action.")},
        "flux3_action_history_unnormalizer": {k: v for k, v in tensors.items() if k.startswith("action.")},
    }
    replacement = {}
    for step in pre["steps"] + post["steps"]:
        name = step["registry_name"]
        if name in required:
            filename = step["state_file"]
            if Path(filename).name != filename or name in replacement:
                raise ValueError("ambiguous processor tensor file")
            saved = load_file(str(source / filename))
            if set(saved) != set(required[name]) or any(v.shape != (6,) for v in saved.values()):
                raise ValueError("unexpected saved quantile keys or dimensions")
            replacement[name] = (filename, required[name])
    if set(replacement) != set(required):
        raise ValueError("missing a required FLUX history processor")
    output.mkdir(parents=True, exist_ok=False)
    changed = {filename for filename, _ in replacement.values()}
    for path in source.iterdir():
        if path.name not in changed | {"normalization_source.json", "dataset_statistics.json"}:
            (output / path.name).symlink_to(path, target_is_directory=path.is_dir())
    for filename, values in replacement.values():
        save_file(values, str(output / filename))
        saved = load_file(str(output / filename))
        assert all(np.array_equal(saved[k], v) for k, v in values.items())
    (output / "dataset_statistics.json").write_text(json.dumps(NORMALIZATION, indent=2) + "\n")
    provenance = {
        "profile": "hf_space_native", "source_checkpoint": str(source),
        "source": "https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim/blob/"
                  + SPACE_REVISION + "/app.py",
        "space_revision": SPACE_REVISION,
        "coordinate_mapping": "identity; native degrees and gripper percent",
        "normalization": {s: {q: tensors[f"{s}.{q}"].tolist() for q in bounds}
                          for s, bounds in NORMALIZATION.items()},
        "source_processor_sha256": {
            filename: hashlib.sha256((source / filename).read_bytes()).hexdigest()
            for filename, _ in replacement.values()
        },
        "physical_robot_compatibility": "experimental; not established by simulator results",
    }
    (output / "normalization_source.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/liminal/models/so101", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
