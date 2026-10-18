"""Combine the requested five extra HF-normalization trials with the first eight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(args):
    import numpy as np

    root = Path(args.root)
    groups = []
    rows = []
    for directory, expected in [("evaluation-C", 8), ("evaluation-C-extra5", 5)]:
        reports = json.loads((root / directory / "summary.json").read_text())
        assert len(reports) == expected
        assert all(r["status"] == "completed" for r in reports)
        for report in reports:
            seed = report["seed"]
            with np.load(root / directory / f"seed-{seed}/trajectory.npz") as trajectory:
                positions = trajectory["block_xyz"]
                bowl = np.asarray(report["randomization"]["bowl_xy"])
                distance = np.linalg.norm(positions[:, :2] - bowl, axis=1)
                report["initial_block_to_container_xy_m"] = float(distance[0])
                report["closest_block_to_container_xy_m"] = float(distance.min())
                report["final_block_to_container_xy_m"] = float(distance[-1])
            report["source_directory"] = directory
        rows.extend(reports)
        groups.append({"group": directory, **counts(reports)})
    assert len({r["seed"] for r in rows}) == 13
    report = {
        "configuration": "C: native simulated joint degrees and HF simulator normalization; unchanged weights",
        "task": rows[0]["task"],
        "fps": 30,
        "per_episode_limit_seconds": 20,
        "sampler_seed": 42,
        "scene_seeds": [r["seed"] for r in rows],
        "groups": groups,
        "combined": counts(rows),
        "episodes": rows,
        "caveat": "Thirteen distinct scenes from the same narrow randomization distribution; "
        "not thirteen repetitions of an identical deterministic scene.",
    }
    (root / "C-13-results.json").write_text(json.dumps(report, indent=2) + "\n")
    text = "# HF-normalization inference: five additional rollouts\n\n"
    text += "**Prompt:** “" + report["task"] + "”\n\n"
    text += "Unchanged released SO-101 weights, native simulator joint coordinates, and the HF simulator quantiles. "
    text += "Five new scene seeds 20008–20012 were added to the original eight seeds 20000–20007.\n\n"
    text += "| Group | Episodes | Contact | Both jaws contact | Lift >3.5 cm | Placement success |\n"
    text += "|---|---:|---:|---:|---:|---:|\n"
    for title, values in [(g["group"], g) for g in groups] + [("Combined", report["combined"])]:
        text += (
            f"| {title} | {values['episodes']} | {values['contact']} | {values['bilateral_contact']} | "
            f"{values['lift_over_3_5cm']} | {values['placement_success']} |\n"
        )
    text += "\nVideos from the five new runs:\n\n"
    for row in rows[-5:]:
        seed = row["seed"]
        text += f"- [Seed {seed}](evaluation-C-extra5/seed-{seed}/rollout.mp4)\n"
    text += "\nFull per-trial measurements: [C-13-results.json](C-13-results.json).\n"
    (root / "C-13-results.md").write_text(text)
    print(json.dumps({"groups": groups, "combined": report["combined"]}, indent=2), flush=True)


def counts(rows):
    return {
        "episodes": len(rows),
        "contact": sum(r["touched_block"] for r in rows),
        "bilateral_contact": sum(r["bilateral_contact"] for r in rows),
        "lift_over_3_5cm": sum(r["max_lift_m"] > 0.035 for r in rows),
        "placement_success": sum(r["manipulation_success"] for r in rows),
        "maximum_lift_m": max(r["max_lift_m"] for r in rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    main(parser.parse_args())
