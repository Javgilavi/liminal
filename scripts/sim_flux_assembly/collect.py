"""Collect complete verified assemblies before exporting atomic skill episodes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scene import AssemblyScene
from teacher import Teacher


async def main(args):
    output = args.root / "raw" / args.split
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in range(args.start_seed, args.start_seed + args.episodes * 4):
        folder = output / f"episode-{seed:05d}"
        if (folder / "report.json").exists():
            report = json.loads((folder / "report.json").read_text())
        else:
            scene = AssemblyScene(args.root, seed, render=False)
            try:
                teacher = Teacher(scene)
                success = await teacher.run()
                teacher.save(folder, success)
                report = json.loads((folder / "report.json").read_text())
            finally:
                scene.close()
        reports.append({"seed": seed, "success": report["success"], "frames": report["frames"]})
        (output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "split": args.split,
                    "attempted": len(reports),
                    "accepted": sum(row["success"] for row in reports),
                    "seed": seed,
                }
            ),
            flush=True,
        )
        if sum(row["success"] for row in reports) >= args.episodes:
            return
    raise RuntimeError("Too few complete contact-verified assembly demonstrations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--start-seed", type=int, default=0)
    asyncio.run(main(parser.parse_args()))
