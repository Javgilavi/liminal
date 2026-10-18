"""Render accepted physics recordings without changing poses, cameras or labels."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def render_episode(root, folder):
    import mujoco
    import numpy as np
    from scene import AssemblyScene

    report = json.loads((folder / "report.json").read_text())
    if not report["success"]:
        return {"source": str(folder), "skipped": "failed physics verification"}
    path = folder / "episode.npz"
    with np.load(path) as arrays:
        records = {key: arrays[key] for key in arrays.files if not key.startswith("images.")}
        if "images.scene" in arrays.files and len(arrays["images.scene"]) == len(arrays["states"]):
            return {"source": str(folder), "frames": len(records["states"]), "resumed": True}
    scene = AssemblyScene(root, report["metadata"]["seed"], render=True)
    pixels = {role: [] for role in ["scene", "wrist"]}
    try:
        assert scene.metadata == report["metadata"]
        for qpos in records["qpos"]:
            scene.data.qpos[:] = qpos
            mujoco.mj_forward(scene.model, scene.data)
            for role, image in scene.frames().items():
                pixels[role].append(image)
        temporary = folder / "rendered.npz"
        np.savez_compressed(
            temporary, **records, **{f"images.{role}": np.asarray(values) for role, values in pixels.items()}
        )
        temporary.replace(path)
        return {"source": str(folder), "frames": len(records["states"])}
    finally:
        scene.close()


def main(args):
    folders = [
        p.parent
        for p in sorted((args.root / "raw").glob("*/episode-*/report.json"))
        if json.loads(p.read_text())["success"]
    ]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render_episode, args.root, folder) for folder in folders]
        for future in as_completed(futures):
            reports.append(future.result())
            print(json.dumps({"completed": len(reports), "total": len(folders), **reports[-1]}), flush=True)
    (args.root / "render-report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    main(parser.parse_args())
