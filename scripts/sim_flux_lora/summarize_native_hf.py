"""Audit matched zero-shot/LoRA rollouts with native coordinates and HF statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main(root: Path) -> None:
    import numpy as np

    norms = json.loads((root / "hf-sim-normalization.json").read_text())
    rows = []
    reference_randomization = None
    for kind in ("zeroshot", "lora"):
        folder = root / f"evaluation-native-hf-{kind}"
        config = json.loads((folder / "run-config.json").read_text())
        assert config["frame"] == "native" and config["episodes"] == 5
        override = json.loads((folder / "normalization-override.json").read_text())
        assert override["statistics"] == norms and len(override["updated_processors"]) == 3
        episodes = json.loads((folder / "summary.json").read_text())
        assert [episode["seed"] for episode in episodes] == list(range(20000, 20005))
        for episode in episodes:
            assert episode["status"] == "completed"
            assert episode["joint_frame"] == "native"
            assert episode["normalization"] == "HF simulation override"
            assert episode["checkpoint"] == config["checkpoint"]
            trajectory = np.load(folder / f"seed-{episode['seed']}" / "trajectory.npz")
            # Check recorded policy observations, independently of the CLI flag.
            np.testing.assert_allclose(
                trajectory["state"][0, :5], episode["randomization"]["initial_native_degrees"][:5], atol=1e-4
            )
        randomization = [episode["randomization"] for episode in episodes]
        if reference_randomization is None:
            reference_randomization = randomization
        else:
            assert randomization == reference_randomization
        rows.append(
            {
                "policy": kind,
                "checkpoint": config["checkpoint"],
                "episodes": len(episodes),
                "contact": sum(episode["touched_block"] for episode in episodes),
                "bilateral_contact": sum(episode["bilateral_contact"] for episode in episodes),
                "lift_over_3_5cm": sum(episode["max_lift_m"] > 0.035 for episode in episodes),
                "placement_success": sum(episode["manipulation_success"] for episode in episodes),
                "max_lift_cm": 100 * max(episode["max_lift_m"] for episode in episodes),
                "per_episode": episodes,
            }
        )
    report = {
        "task": rows[0]["per_episode"][0]["task"],
        "joint_frame": "native: no shoulder/elbow mapping",
        "normalization": norms,
        "normalization_sha256": hashlib.sha256((root / "hf-sim-normalization.json").read_bytes()).hexdigest(),
        "additional_training": False,
        "policies": rows,
        "limitations": [
            "LoRA was trained with dataset90 coordinates and original checkpoint statistics; "
            "this requested inference configuration changes both.",
            "Five matched simulated scenes; this does not establish real-robot performance.",
            "600 commands advance 30 simulated seconds but encode 20 seconds of 30 fps video. "
            "The dataset and collection share this timing mismatch.",
        ],
    }
    (root / "native-hf-results.json").write_text(json.dumps(report, indent=2) + "\n")
    text = "# Native coordinates + HF normalization: zero-shot versus LoRA\n\n"
    text += "Both use native angles with **no shoulder/elbow mapping** and identical HF normalization. "
    text += "Five matched scene seeds (20000–20004), prompt, cameras, action horizon and supervisor. "
    text += "No additional training was performed. LoRA is the raw step-1000 adapter.\n\n"
    text += "| Model | Contact | Bilateral contact | Lift > 3.5 cm | Successful placement | Maximum lift |\n"
    text += "|---|---:|---:|---:|---:|---:|\n"
    for row in rows:
        text += (
            f"| {row['policy']} | {row['contact']}/5 | {row['bilateral_contact']}/5 | "
            f"{row['lift_over_3_5cm']}/5 | {row['placement_success']}/5 | {row['max_lift_cm']:.2f} cm |\n"
        )
    text += "\nSuccess requires bilateral contact, lift, release and stable placement for 15 commands.\n\n"
    for row in rows:
        best = max(row["per_episode"], key=lambda item: (item["manipulation_success"], item["max_lift_m"]))
        text += f"[{row['policy']} example](evaluation-native-hf-{row['policy']}/seed-{best['seed']}/rollout.mp4)\n\n"
    text += "## Limitations\n\n" + "\n".join(f"- {item}" for item in report["limitations"]) + "\n"
    text += "\nFull per-scene evidence: [native-hf-results.json](native-hf-results.json).\n"
    (root / "NATIVE-HF-RESULTS.md").write_text(text)
    print(json.dumps([{key: value for key, value in row.items() if key != "per_episode"} for row in rows]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
