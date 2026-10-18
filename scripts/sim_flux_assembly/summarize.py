"""Report learned assembly outcomes, separate from teacher and isolated-skill results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(root: Path):
    import torch
    from harness import SKILLS
    from safetensors.torch import load_file

    stats = json.loads((root / "hf-sim-normalization.json").read_text())
    dataset = json.loads((root / "dataset-validation.json").read_text())
    assert dataset["status"] == "passed"
    configs = json.loads((root / "train-config.json").read_text())
    collections = {}
    for split in ["train", "validation"]:
        rows = json.loads((root / "raw" / split / "summary.json").read_text())
        collections[split] = {"attempted": len(rows), "accepted": sum(row["success"] for row in rows)}
    results = []
    order = [row[0] for row in SKILLS]
    for kind in ["zeroshot", "raw", "ema"]:
        rows = json.loads((root / f"evaluation-{kind}" / "summary.json").read_text())
        assert len(rows) == 5 and [row["seed"] for row in rows] == list(range(20000, 20005))
        checkpoints = {row["checkpoint"] for row in rows}
        assert len(checkpoints) == 1
        checkpoint = Path(next(iter(checkpoints)))
        assert json.loads((checkpoint / "config.json").read_text())["fps"] == 20
        checked = []
        for file in checkpoint.glob("policy_*processor_step_*.safetensors"):
            for key, value in load_file(str(file)).items():
                stream, quantile = key.split(".")
                assert torch.equal(value, torch.tensor(stats[stream][quantile], dtype=value.dtype))
            checked.append(file.name)
        assert len(checked) == 3
        successes = dict.fromkeys(order, 0)
        reached = dict.fromkeys(order, 0)
        for row in rows:
            assert row["atomic_skill"] is None and not row["teacher_start_state_only"]
            assert row["learned_actions_only"] and row["contract"]["coordinates"] == "native"
            for event in row["skill_results"]:
                reached[event["skill"]] += 1
                successes[event["skill"]] += event["status"] == "success"
        isolated = {}
        atomic = root / f"atomic-{kind}" / "summary.json"
        if atomic.exists():
            for name in order:
                subset = [row for row in json.loads(atomic.read_text()) if row["atomic_skill"] == name]
                assert all(row["teacher_start_state_only"] for row in subset)
                isolated[name] = {
                    "episodes": len(subset),
                    "successes": sum(row["task_report"]["goal_verified"] for row in subset),
                }
        results.append(
            {
                "policy": kind,
                "episodes": len(rows),
                "full_assembly_success": sum(row["assembly_success"] for row in rows),
                "skill_successes_in_sequence": successes,
                "skills_reached_in_sequence": reached,
                "isolated_heldout_start_diagnostics": isolated,
                "verified_saved_HF_processors": checked,
                "checkpoint": str(checkpoint),
                "per_episode": rows,
            }
        )
    report = {
        "dataset": dataset,
        "collection": collections,
        "geometry": json.loads((root / "assets/provenance.json").read_text()),
        "provenance": json.loads((root / "provenance.json").read_text()),
        "training": {
            "microbatches": configs["steps"],
            "optimizer_updates": configs["steps"] // 4,
            "metrics": json.loads((root / "training-metrics.json").read_text()),
        },
        "evaluation": results,
        "limitations": [
            "Five simulated test scenes from narrow object/fixture/lighting/initial-joint randomization.",
            "CAD-derived, scaled clearance-fit parts; not a measured physical reconstruction or force-fit task.",
            "Ground-truth object poses and contacts provide simulator predicates; real perception is not validated.",
            "The IK teacher uses privileged poses; learned action inference receives only images, joints and prompt.",
            "20 Hz and the initial 3000-microbatch budget depart from the official 30 Hz / 10000-microbatch example.",
            "Isolated skills start from held-out teacher states and do not count as full assembly completion.",
        ],
    }
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    text = "# NIST gear and spacer collar assembly\n\n"
    text += f"{dataset['train_assemblies']} train + {dataset['validation_assemblies']} validation assemblies, "
    text += f"{dataset['atomic_episodes']} atomic episodes, {dataset['frames']:,} frames. "
    text += "Four prompts share one fresh rank-32 LoRA trained for 3,000 microbatches / 750 optimizer updates. "
    text += "Native joint angles, HF simulator statistics and 20 Hz are consistent throughout.\n\n"
    text += "The harness runs pick gear → seat gear → pick collar → seat collar. "
    text += "Each step and its prerequisites are checked through contact, pose and settling predicates. "
    text += "Inference has no scripted fallback. Later steps stop when an earlier step fails.\n\n"
    text += "| Model | Pick gear | Seat gear | Pick collar | Seat collar | Full assembly |\n"
    text += "|---|---:|---:|---:|---:|---:|\n"
    for row in results:
        text += (
            f"| {row['policy']} | "
            + " | ".join(
                f"{row['skill_successes_in_sequence'][name]}/{row['skills_reached_in_sequence'][name]}"
                for name in order
            )
            + f" | {row['full_assembly_success']}/5 |\n"
        )
    text += "\nSkill fractions are successes / attempts reached in the sequence; 0/0 means not reached.\n\n"
    for row in results:
        best = max(
            row["per_episode"],
            key=lambda item: (
                item["assembly_success"],
                sum(event["status"] == "success" for event in item["skill_results"]),
            ),
        )
        text += f"[{row['policy']} rollout](evaluation-{row['policy']}/seed-{best['seed']}/rollout.mp4)\n\n"
    text += "## Isolated skill diagnostics\n\n"
    text += "These start at held-out teacher states; they are not end-to-end assembly success.\n\n"
    text += "| Model | Pick gear | Seat gear | Pick collar | Seat collar |\n|---|---:|---:|---:|---:|\n"
    for row in results:
        if row["isolated_heldout_start_diagnostics"]:
            text += (
                f"| {row['policy']} | "
                + " | ".join(
                    f"{row['isolated_heldout_start_diagnostics'][name]['successes']}/"
                    f"{row['isolated_heldout_start_diagnostics'][name]['episodes']}"
                    for name in order
                )
                + " |\n"
            )
    text += "\n## Preview and evidence\n\n"
    text += "[Scripted teacher video](teacher-preview.mp4) · [CAD close-up](assembly-closeup.png) · "
    text += "[Portable 3D assembly](gear-collar-assembly.glb) · [Full results](results.json)\n\n"
    text += "## Limits\n\n" + "\n".join(f"- {item}" for item in report["limitations"]) + "\n"
    (root / "RESULTS.md").write_text(text)
    print(
        json.dumps([{key: value for key, value in row.items() if key != "per_episode"} for row in results]), flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
