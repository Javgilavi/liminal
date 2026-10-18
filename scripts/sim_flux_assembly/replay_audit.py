"""Replay issued teacher commands through the same safety/physics path and audit drift."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from scene import AssemblyScene


async def main(args):
    import numpy as np

    manifest = json.loads((args.root / "dataset-manifest.json").read_text())
    results = []
    for split in ["train", "validation"]:
        sources = sorted({r["source"] for r in manifest["episodes"] if r["split"] == split})
        for source in sources[: args.assemblies_per_split]:
            segments = [r for r in manifest["episodes"] if r["source"] == source]
            seed = segments[0]["seed"]
            with np.load(source) as data:
                actions, states = data["actions"], data["states"]
                object_positions = {part: data[f"{part}_xyz"] for part in ["gear", "collar"]}
            scene = AssemblyScene(args.root, seed, render=False)
            try:
                state_errors, object_errors, issued_errors = [], [], []
                async with scene.safety.claim("sim", "teacher-command-replay-audit"):
                    for tick, command in enumerate(actions):
                        state_errors.append(np.asarray(scene.arm.read_state()) - states[tick])
                        issued = scene.step(command)
                        issued_errors.append(np.asarray(issued) - command)
                        object_errors.append(
                            [
                                np.linalg.norm(scene.data.body(part).xpos - object_positions[part][tick])
                                for part in ["gear", "collar"]
                            ]
                        )
                state_errors = np.asarray(state_errors)
                report = {
                    "split": split,
                    "seed": seed,
                    "frames": len(actions),
                    "assembly_complete": scene.facts["assembly_complete"],
                    "state_rmse_per_joint": np.sqrt(np.mean(state_errors**2, axis=0)).tolist(),
                    "max_state_error_per_joint": np.max(np.abs(state_errors), axis=0).tolist(),
                    "max_object_position_error_m": np.max(object_errors, axis=0).tolist(),
                    "max_issued_command_difference": float(np.max(np.abs(issued_errors))),
                    "final_state": scene.state,
                }
                results.append(report)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps(report), flush=True)
            finally:
                scene.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assemblies-per-split", type=int, default=2)
    asyncio.run(main(parser.parse_args()))
