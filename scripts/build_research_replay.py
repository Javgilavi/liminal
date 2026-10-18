"""Package one compact, explicitly simulated replay for the research page."""

import argparse
import json
from pathlib import Path


def main():
    import numpy as np

    from liminal.ui.twin import JointMirror

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    mirror = JointMirror(root / "examples/dual_so101_sim/research_replay/scene.xml")
    trajectory = np.load(args.source)
    frames = []
    for i in range(0, len(trajectory["qobs"]), 2):
        mirror.data.qpos[:] = trajectory["qobs"][i]
        mirror.mujoco.mj_forward(mirror.model, mirror.data)
        frames.append(
            {"time": round(i / 30, 4), "poses": [list(map(lambda x: round(x, 6), p)) for p in mirror.poses()]}
        )
    output = root / "src/liminal/ui/static/research/replay.json"
    output.write_text(
        json.dumps(
            {
                "label": "Recorded scripted simulation",
                "duration": len(trajectory["qobs"]) / 30,
                "fps": 15,
                "source_fps": 30,
                "frames": frames,
                "note": "Left arm parked; lid fixture; no real robot data",
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
