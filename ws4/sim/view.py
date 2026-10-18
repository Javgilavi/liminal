"""Interactive MuJoCo viewer for the station.

    python ws4/sim/view.py                                   # robot + table only
    python ws4/sim/view.py --scene ws4/sim/scenes/tabletop_props.xml --randomize --seed 3

Use the viewer's Control panel (right side) to drive the 12 position actuators in radians, and the
Rendering panel to show group 3 (collision geoms) or group 4 (TCP sites). Ctrl+right-drag applies forces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    import mujoco.viewer

    from ws4.sim.env import HOME_DEG, XLeRobotStationEnv
    from ws4.sim.geometry import SIM_DIR

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=str(SIM_DIR / "xlerobot_station.xml"))
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose", choices=["home", "zero"], default="home",
                    help="home = ready pose, zero = LeRobot calibration middle (all 0 deg)")
    a = ap.parse_args()
    env = XLeRobotStationEnv(scene=a.scene, randomize=a.randomize, render_images=False)
    home = HOME_DEG if a.pose == "home" else np.zeros(6)
    env.reset(seed=a.seed, options={"home_deg": home})
    mujoco.viewer.launch(env.model, env.data)


if __name__ == "__main__":
    main()
