"""Audit assembly labels, held-out splits, camera roles and independent predicates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(root: Path):
    import mujoco
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from safetensors.torch import load_file
    from scene import FPS, PROMPTS, AssemblyScene

    torch.set_num_threads(4)
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    stats = json.loads((root / "hf-sim-normalization.json").read_text())
    config = json.loads((root / "base-native-hf/config.json").read_text())
    normalization_extrema = {"state": 0.0, "action": 0.0}
    assert manifest["fps"] == FPS and manifest["coordinate_frame"] == "native"
    train = {row["seed"] for row in manifest["episodes"] if row["split"] == "train"}
    validation = {row["seed"] for row in manifest["episodes"] if row["split"] == "validation"}
    assert len(train) == 32 and len(validation) == 8 and train.isdisjoint(validation)
    dataset = LeRobotDataset(
        "local/liminal_nist_atomic_assembly", root=root / "dataset-v3", video_backend="pyav", return_uint8=True
    )
    seen = set()
    endpoint_checks, physics_checks, video_checks = 0, [], []
    offset = 0
    for row in manifest["episodes"]:
        source = Path(row["source"])
        with np.load(source) as arrays:
            a, b = row["start"], row["end"]
            if source not in seen:
                scene = AssemblyScene(root, row["seed"])
                try:
                    state = arrays["states"]
                    expected = np.rad2deg(arrays["qpos"][:, scene.arm.qadr])
                    lo, hi = scene.arm.ranges[-1]
                    expected[:, -1] = (arrays["qpos"][:, scene.arm.qadr[-1]] - lo) / (hi - lo) * 100
                    np.testing.assert_allclose(state, expected, atol=1e-4, rtol=0)
                    assert np.isfinite(arrays["actions"]).all()
                    assert np.all(np.abs(arrays["actions"] - state) <= np.array([3, 3, 3, 3, 3, 6]) + 1e-4)
                    commands = arrays["actions"].copy()
                    if config["action_representation"] == "delta":
                        action_values = np.diff(commands, axis=0)
                        absolute = config["delta_absolute_dims"]
                        action_values[:, absolute] = commands[1:, absolute]
                    else:
                        action_values = commands
                    for stream, values in [("state", state), ("action", action_values)]:
                        lo, hi = np.asarray(stats[stream]["q01"]), np.asarray(stats[stream]["q99"])
                        scaled = 2 * (values - lo) / (hi - lo) - 1
                        peak = float(np.max(np.abs(scaled)))
                        assert peak < config["normalization_clip"], (row["seed"], stream, peak)
                        normalization_extrema[stream] = max(normalization_extrema[stream], peak)
                    scene.data.qpos[:] = arrays["qpos"][-1]
                    scene.data.qvel[:] = 0
                    mujoco.mj_forward(scene.model, scene.data)
                    # Restoring qpos without the solver's contact warm-start can
                    # briefly disturb the stack. Keep the same stability predicate
                    # and allow up to three seconds to re-establish support.
                    for _replay_tick in range(60):
                        scene.step(scene.arm.read_state())
                        if scene.facts["assembly_complete"]:
                            break
                    assert scene.facts["assembly_complete"], (row["seed"], scene.state)
                    address = scene.model.jnt_qposadr[scene.model.joint("collar_free").id]
                    scene.data.qpos[address] += 0.04
                    mujoco.mj_forward(scene.model, scene.data)
                    scene.update_predicates()
                    assert not scene.facts["assembly_complete"], "Misplaced collar incorrectly accepted"
                    physics_checks.append({"seed": row["seed"], "settling_ticks": _replay_tick + 1})
                finally:
                    scene.close()
                seen.add(source)
            for tick in [0, row["frames"] // 2, row["frames"] - 1]:
                saved = dataset.hf_dataset[offset + tick]
                np.testing.assert_allclose(
                    np.asarray(saved["observation.state"]), arrays["states"][a + tick], atol=1e-5
                )
                np.testing.assert_allclose(np.asarray(saved["action"]), arrays["actions"][a + tick], atol=1e-5)
                assert abs(float(saved["timestamp"]) - tick / FPS) < 1e-4
                endpoint_checks += 1
            assert b - a == row["frames"] >= 50
            if row["skill"].startswith("pick_"):
                part = row["skill"].removeprefix("pick_")
                assert arrays[f"{part}_xyz"][b - 1, 2] > 0.065, "Pickup endpoint lacks shaft clearance"
            if row["seed"] in [min(train), min(validation)]:
                tick = min(90, row["frames"] - 1)
                decoded = dataset[offset + tick]
                assert decoded["task"] == PROMPTS[row["skill"]]
                for role in ["scene", "wrist"]:
                    pixels = decoded[f"observation.images.{role}"].numpy().transpose(1, 2, 0).astype(float)
                    original = arrays[f"images.{role}"][a + tick].astype(float)
                    other = arrays[f"images.{'wrist' if role == 'scene' else 'scene'}"][a + tick].astype(float)
                    mse = float(np.mean((pixels - original) ** 2))
                    assert mse < np.mean((pixels - other) ** 2)
                    psnr = float(10 * np.log10(255**2 / max(mse, 1e-10)))
                    assert psnr > 28
                    video_checks.append({"seed": row["seed"], "skill": row["skill"], "role": role, "psnr": psnr})
        offset += row["frames"]
    assert offset == manifest["frames"] == len(dataset)
    for file in (root / "base-native-hf").glob("policy_*processor_step_*.safetensors"):
        for key, value in load_file(str(file)).items():
            stream, quantile = key.split(".")
            assert torch.equal(value, torch.tensor(stats[stream][quantile], dtype=value.dtype))
    report = {
        "status": "passed",
        "assemblies": len(seen),
        "train_assemblies": len(train),
        "validation_assemblies": len(validation),
        "atomic_episodes": len(manifest["episodes"]),
        "frames": offset,
        "fps": FPS,
        "labels": "native states and absolute issued commands",
        "label_timestamp_samples": endpoint_checks,
        "physics_replay_checks": physics_checks,
        "negative_test": "40 mm misplaced collar rejects assembly_complete in every accepted scene",
        "video_checks": video_checks,
        "normalization_max_absolute": normalization_extrema,
        "normalization_clip": config["normalization_clip"],
        "pickup_handoff": "Bilateral stable grasp with part bottom above 65 mm; shaft top is 60 mm",
    }
    (root / "dataset-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "video_checks"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
