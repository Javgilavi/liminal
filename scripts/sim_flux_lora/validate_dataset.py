"""Check demonstration provenance, pose/action alignment and observed physics outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(args):
    import mujoco
    import numpy as np
    from common import OFFSETS, SIGNS, SIM_ROOT, FrameArm

    model = mujoco.MjModel.from_xml_path(str(SIM_ROOT / "scene.xml"))
    arm = FrameArm(model, mujoco.MjData(model))
    seen = set()
    checked = []
    for split in ["train", "validation"]:
        for file in sorted((Path(args.raw) / split).glob("episode-*/episode.npz")):
            report = json.loads((file.parent / "report.json").read_text())
            seed = report["seed"]
            assert seed not in seen
            seen.add(seed)
            with np.load(file, allow_pickle=False) as data:
                state, action = data["states"], data["actions"]
                expected = np.rad2deg(data["qpos"][:, arm.qadr])
                expected[:, :5] = expected[:, :5] * SIGNS + OFFSETS
                lo, hi = arm.ranges[-1]
                expected[:, -1] = (data["qpos"][:, arm.qadr[-1]] - lo) / (hi - lo) * 100
                np.testing.assert_allclose(state, expected, atol=1e-4, rtol=0)
                assert np.isfinite(state).all() and np.isfinite(action).all()
                assert np.all(np.abs(action - state) <= np.array([3, 3, 3, 3, 3, 6]) + 1e-4)
                block = data["block_xyz"]
                bowl = np.array(report["bowl_xy"])
                assert len(state) >= 50 and len(block) == len(state)
                assert block[:, 2].max() - block[0, 2] > 0.04
                assert data["bilateral"].any()
                assert np.all(np.abs(block[-15:, :2] - bowl) < 0.033)
                assert np.all((block[-15:, 2] > 0) & (block[-15:, 2] < 0.06))
                assert np.all(np.linalg.norm(np.diff(block[-16:], axis=0) / (10 * model.opt.timestep), axis=1) < 0.05)
                checked.append({"seed": seed, "split": split, "frames": len(state)})
    assert sum(r["split"] == "train" for r in checked) == 64
    assert sum(r["split"] == "validation" for r in checked) == 16
    report = {
        "status": "passed",
        "episodes": len(checked),
        "frames": sum(r["frames"] for r in checked),
        "checks": [
            "disjoint seeds",
            "states match native joint poses after mapping",
            "recorded actions match supervisor step limits",
            "finite values",
            "complete 50-frame windows",
            "bilateral contact and observed lift",
            "observed final placement and settling",
        ],
        "provenance": checked,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
