"""Check native labels, encoded observations, timestamps and prepared processors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(root: Path) -> None:
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from safetensors.torch import load_file

    torch.set_num_threads(4)
    dataset_root = root / "dataset-native-v3"
    manifest = json.loads((dataset_root / "simulation-provenance.json").read_text())
    assert manifest["fps"] == 20 and len(manifest["episodes"]) == 80
    assert manifest["coordinate_mapping"] == "native joint degrees; gripper 0-100 percent"
    dataset = LeRobotDataset("local/liminal_so101_native_hf_pick_place", root=dataset_root, video_backend="pyav")
    offset = 0
    video_checks = []
    for entry in manifest["episodes"]:
        arrays = np.load(entry["source"])
        for tick in [0, 200, entry["frames"] - 1]:
            # Read labels directly to avoid decoding every video during this audit.
            row = dataset.hf_dataset[offset + tick]
            for key, source_key in [("observation.state", "states"), ("action", "actions")]:
                expected = arrays[source_key][tick].copy()
                expected[1] = 90 - expected[1]
                expected[2] -= 90
                np.testing.assert_allclose(np.asarray(row[key]), expected, atol=1e-5, rtol=0)
            assert abs(float(row["timestamp"]) - tick / 20) < 1e-4
            if entry["episode_index"] in [0, 64, 79] and tick in [0, 200]:
                decoded = dataset[offset + tick]
                for role in ["scene", "wrist"]:
                    pixels = decoded[f"observation.images.{role}"].numpy().transpose(1, 2, 0) * 255
                    mse = float(np.mean((pixels - arrays[f"images.{role}"][tick].astype(float)) ** 2))
                    psnr = 10 * np.log10(255**2 / max(mse, 1e-10))
                    assert psnr > 28
                    video_checks.append(
                        {"episode": entry["episode_index"], "frame": tick, "camera": role, "psnr": psnr}
                    )
        arrays.close()
        offset += entry["frames"]
    assert offset == manifest["total_frames"] == len(dataset)
    stats = json.loads((root / "hf-sim-normalization.json").read_text())
    for file in (root / "base-native-hf").glob("policy_*processor_step_*.safetensors"):
        for key, value in load_file(str(file)).items():
            stream, quantile = key.split(".")
            assert torch.equal(value, torch.tensor(stats[stream][quantile], dtype=value.dtype))
    report = {
        "status": "passed",
        "episodes": 80,
        "frames": offset,
        "fps": 20,
        "coordinate_frame": "native",
        "checks": "Converted state/action endpoints in every episode; timestamps; three prepared HF processors",
        "video_checks": video_checks,
        "source_physics_audit": "collection-inference-audit.json",
    }
    (root / "dataset-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
