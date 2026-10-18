"""Verify the same native/HF/20 Hz contract in data, checkpoints and rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(root: Path) -> None:
    import numpy as np
    import torch
    from safetensors.torch import load_file

    stats = json.loads((root / "hf-sim-normalization.json").read_text())
    dataset = json.loads((root / "dataset-validation.json").read_text())
    assert dataset["status"] == "passed" and dataset["coordinate_frame"] == "native" and dataset["fps"] == 20
    rows = []
    scenes = None
    for kind in ["zeroshot", "raw", "ema"]:
        folder = root / f"evaluation-{kind}"
        config = json.loads((folder / "run-config.json").read_text())
        assert config["frame"] == "native" and config["normalization_json"] is None
        checkpoint = Path(config["checkpoint"])
        assert (checkpoint / "adapter_model.safetensors").exists() == (kind != "zeroshot")
        policy_config = json.loads((checkpoint / "config.json").read_text())
        assert policy_config["fps"] == 20
        processors = list(checkpoint.glob("policy_*processor_step_*.safetensors"))
        assert len(processors) == 3
        for file in processors:
            for key, value in load_file(str(file)).items():
                stream, quantile = key.split(".")
                assert torch.equal(value, torch.tensor(stats[stream][quantile], dtype=value.dtype))
        episodes = json.loads((folder / "summary.json").read_text())
        assert [episode["seed"] for episode in episodes] == list(range(20000, 20005))
        for episode in episodes:
            assert episode["status"] == "completed" and episode["timing_consistent"]
            assert episode["joint_frame"] == "native" and episode["fps"] == episode["physics_control_hz"] == 20
            trajectory = np.load(folder / f"seed-{episode['seed']}" / "trajectory.npz")
            np.testing.assert_allclose(
                trajectory["state"][0, :5], episode["randomization"]["initial_native_degrees"][:5], atol=1e-4
            )
        current_scenes = [episode["randomization"] for episode in episodes]
        assert scenes is None or scenes == current_scenes
        scenes = current_scenes
        rows.append(
            {
                "policy": kind,
                "episodes": 5,
                "contact": sum(episode["touched_block"] for episode in episodes),
                "bilateral_contact": sum(episode["bilateral_contact"] for episode in episodes),
                "lift_over_3_5cm": sum(episode["max_lift_m"] > 0.035 for episode in episodes),
                "placement_success": sum(episode["manipulation_success"] for episode in episodes),
                "max_lift_cm": 100 * max(episode["max_lift_m"] for episode in episodes),
                "checkpoint": str(checkpoint),
                "saved_hf_processors_verified": [file.name for file in processors],
                "per_episode": episodes,
            }
        )
    report = {
        "contract": "native joint degrees, gripper percent, HF simulator statistics, 20 Hz throughout",
        "dataset": dataset,
        "provenance": json.loads((root / "provenance.json").read_text()),
        "training_metrics": json.loads((root / "training-metrics.json").read_text()),
        "comparison": rows,
        "limitations": [
            "64 training and 16 validation demos from a narrow randomization distribution; five new test scenes.",
            "1000 microbatches / 250 optimizer updates, shorter than the official 10000-microbatch example.",
            "The same staged teacher demos retain pauses; this experiment repairs coordinates, statistics and timing.",
            "Simulation only. Success is independently checked contact, lift, release and stable placement.",
        ],
    }
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    text = "# Consistent native-coordinate HF-normalization LoRA\n\n"
    text += "64 train + 16 validation demos; 32,220 frames, exported at their actual 20 Hz. "
    text += "Both absolute states and commands were converted to native coordinates. "
    text += "A fresh rank-32 LoRA trained for 1,000 microbatches / 250 optimizer updates. "
    text += "HF normalization was saved before training and verified unchanged in all evaluation checkpoints. "
    text += "No shoulder/elbow mapping or normalization override is applied during inference. "
    text += "Policy, dataset, video and physics all use 20 Hz.\n\n"
    text += "| Model | Contact | Bilateral contact | Lift > 3.5 cm | Placement | Maximum lift |\n"
    text += "|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        text += (
            f"| {row['policy']} | {row['contact']}/5 | {row['bilateral_contact']}/5 | "
            f"{row['lift_over_3_5cm']}/5 | {row['placement_success']}/5 | {row['max_lift_cm']:.2f} cm |\n"
        )
    text += "\nAll models use the same scene seeds 20000–20004 and a 600-command / 30-second limit.\n\n"
    for row in rows:
        best = max(row["per_episode"], key=lambda episode: (episode["manipulation_success"], episode["max_lift_m"]))
        text += f"[{row['policy']} video](evaluation-{row['policy']}/seed-{best['seed']}/rollout.mp4)\n\n"
    text += "## Limits\n\n" + "\n".join(f"- {item}" for item in report["limitations"]) + "\n"
    text += "\n[Full results and verification](results.json). Old adapters were deleted; see old-lora-deleted.json.\n"
    (root / "RESULTS.md").write_text(text)
    print(json.dumps([{key: value for key, value in row.items() if key != "per_episode"} for row in rows]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
