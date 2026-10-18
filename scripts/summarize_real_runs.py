"""Summarize recorded official-policy trials without inferring manipulation success.

Reads existing logs only; never opens a robot connection or imports model code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(root: Path) -> dict:
    trials = []
    for report_file in sorted(root.glob("*/report.json")):
        report = json.loads(report_file.read_text())
        worker = report.get("worker", {})
        integration = report.get("integration", worker.get("integration", ""))
        if integration != "lerobot.Flux3Policy + saved pre/post processors":
            continue
        folder = report_file.parent
        events_file = folder / "events.jsonl"
        events = [json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
        plans = [event for event in events if event.get("type") == "plan"]
        committed = [event for event in events if event.get("type") in ("committed", "command")]
        # Older direct-driver trials save plans separately instead of plan events.
        plan_count = len(plans) or len(list(folder.glob("plan-*.json")))
        source = report.get("normalization_source") or worker.get("normalization_source") or {}
        profile = report.get("normalization_profile", worker.get("normalization_profile"))
        checkpoint = report.get("checkpoint", worker.get("checkpoint"))
        trials.append({
            "run_id": folder.name,
            "execute_requested": report.get("execute", False),
            "prompt": report.get("prompt"),
            "integration": integration,
            "checkpoint": checkpoint,
            "normalization_profile": profile or "released (not explicitly recorded by this runner version)",
            "normalization_revision": source.get("space_revision"),
            "coordinate_mapping": report.get("coordinate_mapping"),
            "source_revision": report.get("source_revision", worker.get("source_revision")),
            "status": report["status"],
            "stop_reason": report.get("error", report.get("execution_blocker")),
            "plans": plan_count,
            "commands": len(committed),
            "gpu_prediction_seconds": [event["inference_s"] for event in plans if "inference_s" in event],
            "manipulation_success": None,
            "evidence_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [report_file, events_file, folder / "plans.jsonl", folder / "submitted-actions.jsonl"]
                if path.is_file()
            },
        })
    return {
        "schema_version": 1,
        "scope": "All recorded trials reporting the official LeRobot FLUX policy/processors in this source folder",
        "interpretation": "Execution requests, command delivery and process completion are not manipulation scores. "
                          "No independent real-robot task-success labels were recorded.",
        "trials": trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(result['trials'])} official-policy trial summaries to {args.output}")


if __name__ == "__main__":
    main()
