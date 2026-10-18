"""Build a reviewable experiment report without treating process completion as success."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def main(args):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from safetensors.torch import load_file

    root = Path(args.root)
    config = json.loads((root / "train-config.json").read_text())
    rows = []
    for kind in ["A", "base", "C", "raw", "ema"]:
        summary = root / f"evaluation-{kind}/summary.json"
        if not summary.exists():
            raise FileNotFoundError(summary)
        episodes = json.loads(summary.read_text())
        assert len(episodes) == 8
        assert {r["seed"] for r in episodes} == set(range(20000, 20008))
        row = {
            "policy": kind,
            "episodes": len(episodes),
            "successes": sum(r["manipulation_success"] for r in episodes),
            "episodes_touching_block": sum(r["touched_block"] for r in episodes),
            "episodes_with_bilateral_contact": sum(r["bilateral_contact"] for r in episodes),
            "episodes_lifting_3_5cm": sum(r["max_lift_m"] > 0.035 for r in episodes),
            "mean_max_lift_m": float(np.mean([r["max_lift_m"] for r in episodes])),
            "mean_safety_limited_ticks": float(np.mean([r["safety_limited_ticks"] for r in episodes])),
            "per_episode": episodes,
        }
        offline = root / f"offline-{kind}.json"
        if offline.exists():
            row["offline_validation"] = json.loads(offline.read_text())
        rows.append(row)
    # Check saved processor numbers exactly, independent of serialization hashes.
    unchanged = {}
    source = Path("/liminal/models/so101")
    checkpoints = root / "training-1000/checkpoints/001000"
    for kind, dirname in [("raw", "pretrained_model"), ("ema", "pretrained_model_ema")]:
        checked = []
        for path in source.glob("policy_*processor_step_*.safetensors"):
            other = checkpoints / dirname / path.name
            original_values, saved_values = load_file(str(path)), load_file(str(other))
            assert original_values.keys() == saved_values.keys()
            assert all(torch.equal(value, saved_values[key]) for key, value in original_values.items())
            checked.append(path.name)
        assert len(checked) == 3
        unchanged[kind] = checked
    log = (root / "training.log").read_text(errors="replace")
    losses = re.findall(r"step:(\d+).*?loss:([0-9.]+).*?action_mse:([0-9.]+)", log)
    report = {
        "task": "Pick up the red block and place it in the blue container.",
        "dataset": json.loads((root / "dataset-validation.json").read_text()),
        "provenance": json.loads((root / "provenance.json").read_text()),
        "training": {
            "microbatches": config["steps"],
            "optimizer_updates": config["steps"] // config["accelerator"]["gradient_accumulation"]["steps"],
            "batch_size": config["batch_size"],
            "peft": config["peft"],
            "ema": config["ema"],
            "metrics": json.loads((root / "training-metrics.json").read_text()),
            "loss_trace": [
                {"step": int(step), "loss": float(loss), "action_mse": float(action)} for step, loss, action in losses
            ],
        },
        "normalization_unchanged": unchanged,
        "evaluation": rows,
        "limitations": [
            "Initial 1000-microbatch budget versus the official example 10000.",
            "Eight held-out scene seeds from the same narrow randomization distribution.",
            "The staged waypoint teacher includes pauses that the adapter may imitate.",
            "Dataset/video timestamps are 30 Hz, but 10 physics steps of 5 ms advance 50 ms per command "
            "(20 Hz physics); collection and every evaluation share this mismatch.",
            "Simulation only; does not establish real robot or broad task performance.",
            "The specific BFL fork was unavailable; exact official LeRobot revision recorded.",
        ],
    }
    (root / "experiment-report.json").write_text(json.dumps(report, indent=2) + "\n")
    table = [
        f"| {r['policy']} | {r['successes']}/8 | {r['episodes_touching_block']}/8 | "
        f"{r['episodes_lifting_3_5cm']}/8 | {100 * r['mean_max_lift_m']:.2f} |"
        for r in rows
    ]
    readme = "# Synthetic simulation task LoRA experiment\n\n"
    readme += "64 training demonstrations, 16 held-out validation demonstrations, 30 Hz. "
    readme += f"The dataset contains {report['dataset']['frames']:,} verified frames. "
    readme += "The official LeRobot trainer ran 1,000 microbatches / 250 optimizer updates with rank-32 LoRA. "
    readme += "Saved checkpoint normalization was verified unchanged.\n\n"
    readme += "| Policy | Placement success | Block contact | Lift > 3.5 cm | Mean max lift (cm) |\n"
    readme += "|---|---:|---:|---:|---:|\n" + "\n".join(table) + "\n\n"
    readme += "A = native coordinates + checkpoint statistics; base (B) = dataset90 coordinates + checkpoint "
    readme += "statistics; C = native coordinates + HF simulator statistics. Raw and EMA LoRA use dataset90 "
    readme += "coordinates + checkpoint statistics, matching their training.\n\n"
    readme += "All policies used the same eight new scene seeds, 600-command limit, cameras, "
    readme += "32-step execution horizon, supervisor, and physics-based task checks. "
    readme += "The 30 Hz video/dataset timestamps span 20 seconds, while physics advances 30 seconds: "
    readme += "each command advances ten 5 ms steps. Collection shares this timing mismatch. "
    readme += "A process completing successfully does not count as a successful manipulation.\n\n"
    readme += "The initial budget is shorter than the official 10,000-microbatch example. "
    readme += "The inaccessible BFL fork revision and the actual public LeRobot revision are recorded in provenance. "
    readme += (
        "Full results, normalization checks and caveats are in [experiment-report.json](experiment-report.json).\n\n"
    )
    readme += "[Teacher demonstration](teacher-demo.mp4) · "
    readme += "[Collection preview](teacher-contact-sheet.png) · "
    readme += "[Evaluation final frames](evaluation-final-frames.png)\n"
    (root / "RESULTS.md").write_text(readme)
    panels = []
    for kind in ["A", "base", "C", "raw", "ema"]:
        video = root / f"evaluation-{kind}/seed-20000/rollout.mp4"
        final = root / f"evaluation-{kind}/seed-20000/final.png"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-sseof", "-0.04", "-i", str(video), "-frames:v", "1", str(final)],
            check=True,
        )
        panel = Image.new("RGB", (512, 280), "#202020")
        panel.paste(Image.open(final).resize((512, 256)), (0, 24))
        ImageDraw.Draw(panel).text((8, 5), kind + " / held-out seed 20000 / final frame", fill="white")
        panels.append(panel)
    sheet = Image.new("RGB", (512, 280 * len(panels)))
    for i, panel in enumerate(panels):
        sheet.paste(panel, (0, i * 280))
    sheet.save(root / "evaluation-final-frames.png")
    print(
        json.dumps(
            [{k: v for k, v in row.items() if k not in ["per_episode", "offline_validation"]} for row in rows], indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    main(parser.parse_args())
