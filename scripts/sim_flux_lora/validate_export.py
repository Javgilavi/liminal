"""Check video encoding, camera roles and absolute-command preservation in the v3 export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(args):
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.set_num_threads(4)
    root = Path(args.root)
    dataset_root = root / "dataset-v3-final"
    manifest = json.loads((dataset_root / "simulation-provenance.json").read_text())
    checks = []
    for index in [0, 79]:
        provenance = manifest["episodes"][index]
        dataset = LeRobotDataset(
            "local/liminal_so101_sim_pick_place",
            root=dataset_root,
            episodes=[index],
            video_backend="pyav",
            return_uint8=True,
        )
        with np.load(provenance["source"]) as arrays:
            cameras = {role: arrays["images." + role] for role in ["scene", "wrist"]}
            state, action = arrays["states"], arrays["actions"]
            for tick in [0, 200]:
                row = dataset[tick]
                np.testing.assert_allclose(row["observation.state"].numpy(), state[tick], rtol=0, atol=1e-6)
                np.testing.assert_allclose(row["action"].numpy(), action[tick], rtol=0, atol=1e-6)
                assert int(row["episode_index"]) == index
                assert row["task"] == manifest["task"]
                scores = {}
                for role in ["scene", "wrist"]:
                    decoded = row["observation.images." + role].numpy().transpose(1, 2, 0).astype(float)
                    expected = cameras[role][tick].astype(float)
                    mse = float(np.mean((decoded - expected) ** 2))
                    psnr = float(10 * np.log10(255**2 / max(mse, 1e-10)))
                    other = cameras["wrist" if role == "scene" else "scene"][tick].astype(float)
                    assert mse < float(np.mean((decoded - other) ** 2)), "Camera roles swapped"
                    assert psnr > 28, f"Unexpected video encoding degradation: {psnr}"
                    scores[role] = psnr
                checks.append({"episode": index, "frame": tick, "psnr_db": scores})
    report = {
        "status": "passed",
        "checks": checks,
        "meaning": "Decoded videos retain camera roles; raw states and absolute issued commands survive export.",
    }
    (root / "export-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    main(parser.parse_args())
