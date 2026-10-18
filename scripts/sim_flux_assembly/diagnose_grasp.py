"""Compare jaw approach geometry in saved teacher and learned gear-pickup trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(args):
    import mujoco
    import numpy as np
    from scene import FrameArm

    model = mujoco.MjModel.from_xml_path(str(args.root / "scene.xml"))
    data = mujoco.MjData(model)
    arm = FrameArm(model, data, frame="native")
    tips = [model.geom(name).id for name in ["fixed_jaw_sph_tip1", "moving_jaw_sph_tip1"]]
    gripper = model.body("gripper").id
    rows = []
    for filename in args.trajectories:
        with np.load(filename) as arrays:
            teacher = "states" in arrays
            states = arrays["states" if teacher else "state"]
            commands = arrays["actions" if teacher else "issued"]
            selected = np.flatnonzero(arrays["skill"] == "pick_gear")
            gear = arrays["gear_xyz"]
            samples = []
            for tick in selected:
                # Kinematics only: no motion command or physics state replay.
                data.qpos[arm.qadr] = arm.to_sim(states[tick])
                mujoco.mj_forward(model, data)
                midpoint = data.geom_xpos[tips].mean(0)
                target = gear[tick] + [0, 0, 0.024]
                delta = midpoint - target
                samples.append(
                    {
                        "tick": int(tick),
                        "distance_mm": float(np.linalg.norm(delta) * 1000),
                        "xy_error_mm": float(np.linalg.norm(delta[:2]) * 1000),
                        "z_error_mm": float(delta[2] * 1000),
                        "gripper_axis_tilt_deg": float(
                            np.rad2deg(np.arccos(np.clip(data.xmat[gripper].reshape(3, 3)[2, 2], -1, 1)))
                        ),
                        "measured_gripper_pp": float(states[tick, -1]),
                        "commanded_gripper_pp": float(commands[tick, -1]),
                    }
                )
            closest = min(samples, key=lambda sample: sample["distance_mm"])
            rows.append(
                {
                    "trajectory": str(filename),
                    "source": "teacher" if teacher else "learned",
                    "closest_to_teacher_grasp_target": closest,
                    "ticks_within_5mm": sum(s["distance_mm"] < 5 for s in samples),
                    "note": "Jaw-tip midpoint versus gear body origin +24mm; kinematic diagnostic, not grasp success",
                }
            )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("trajectories", type=Path, nargs="+")
    main(parser.parse_args())
