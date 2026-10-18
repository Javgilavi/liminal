# Haller digital twin (MuJoCo)

A MuJoCo copy of the physical Haller robot and its tabletop. The goal is a **live mirror**: the twin
reads the real arms' joint positions and moves with them, so you can see in simulation what the real robot
does. The same twin is also used to generate demonstrations and test FLUX LoRA adapters.

| Layer | Files | What it contains | Status |
|---|---|---|---|
| Rig | `scene.xml`, `so101/` | 2× SO-101 on the XLeRobot tower, D455 `top` camera, wrist cameras, bench | Already versioned; audited in [README.md](README.md) |
| Tabletop twin | `twin/` | The rig plus the round table, black mat, white box with hinged lid, green/red markers, yellow cutter | Built from a phone video; sizes are estimates |
| Live mirror (real → twin) | not written yet | Read both real arms at 30 Hz and pose the twin | **Planned**; design below |

## 1. Setup

Everything runs on a laptop CPU. A GPU is only needed for FLUX evaluation (section 5).

```bash
cd liminal/examples/dual_so101_sim
uv venv .venv
uv pip install --python .venv/bin/python mujoco==3.16.0 pillow numpy
MUJOCO_GL=egl .venv/bin/python check.py        # smoke test of the bare rig -> renders/*.png
```

- `MUJOCO_GL=egl` renders headless (servers, SSH). On a desktop, leave it out to use the normal OpenGL window.
- Run every `twin` command **from `examples/dual_so101_sim/`**. `twin` is a package that finds the arm model
  at `../so101/so101.xml` relative to itself.
- For the real robot, the same environment also needs LeRobot with Feetech support
  (`uv pip install --python .venv/bin/python 'lerobot[feetech]'`) and access to the arms' serial ports.

## 2. Look at the twin

Bare rig in the interactive viewer (needs a display):

```bash
.venv/bin/python -m mujoco.viewer --mjcf=scene.xml
```

Tabletop twin with a random object layout (change the seed for another layout):

```bash
.venv/bin/python - <<'EOF'
import mujoco.viewer
import numpy as np
from twin.generate import random_layout
from twin.scene import build

model, data = build(random_layout("place_red_marker", np.random.default_rng(0)))
mujoco.viewer.launch(model, data)
EOF
```

Headless render of every camera instead of the viewer:

```bash
MUJOCO_GL=egl .venv/bin/python - <<'EOF'
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from twin.generate import random_layout
from twin.scene import build

model, data = build(random_layout("place_red_marker", np.random.default_rng(0)))
Path("renders").mkdir(exist_ok=True)
r = mujoco.Renderer(model, 480, 640)
for cam in ("front", "top", "left_wrist", "right_wrist"):
    r.update_scene(data, camera=cam)
    Image.fromarray(r.render()).save(f"renders/twin_{cam}.png")
EOF
```

`build(layout)` returns a normal `MjModel`/`MjData` pair. The layout (`twin.scene.Layout`) sets the box pose and
lid angle, each object's `(x, y, yaw)`, the light level and a small D455 pose jitter. Default dimensions and
colours are in `twin/config.py`.

## 3. Names, frames and units (real ↔ twin)

This is what the live mirror needs, and what any code moving data between the robot and the twin must respect.

- **Joints:** `left_` / `right_` + `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`,
  the same names and order as LeRobot's `so101_follower` (and `liminal.core.types.SO101_JOINTS`).
- **Frame:** bench top at z = 0, arms at x = ±0.15 m on a 0.10 m riser, reaching toward −y. The left arm is the
  operator's left when facing the arms.
- **Arm joints:** twin `qpos` is in radians with zero at mid-range, the same zero as LeRobot's calibration.
  `liminal.hardware.so101.SO101Arm` always reads in LeRobot **degree mode**, which maps directly:
  `qpos = deg2rad(value)`. LeRobot's `RANGE_M100_100` mode does **not** map; don't use it for the mirror.
  VERIFY on the rig by posing one arm and comparing (see section 6).
- **Gripper:** LeRobot always reports 0–100. Twin joint range is −0.1745 … 1.7453 rad, so
  `qpos = -0.1745 + pct / 100 * 1.9199`. This is the conversion `twin.expert.Arm.state()`/`command()` use.
- **Cameras:** `top` = D455 RGB 640×480; `left_wrist` / `right_wrist` = OV2710 wrist cameras; `front` is sim-only.
  Real camera devices are in `configs/robot/xlerobot.yaml`.
- **Not the FLUX convention.** The FLUX 3 Action SO-101 checkpoint was trained with `shoulder_lift + 180°` and
  `elbow_flex + 90°` relative to these joints. That offset belongs only at the policy boundary
  (`twin/export.py: to_checkpoint`). Never apply it to the mirror.

Real robot settings already in the repo: `configs/robot/xlerobot.yaml` identifies each arm by its USB board serial
(left `...5B79018460`, right `...5B14030445`). That file marks the physical left/right sides as VERIFY; confirm them
before mirroring, or the twin will show the arms swapped.

## 4. Live mirror: how it will work

Not implemented yet. This is the design so whoever builds it keeps it safe and simple.

1. **Read-only.** Build both arms from `configs/robot/xlerobot.yaml` as `liminal.hardware.so101.SO101Arm` and only
   ever call `read_state()`. `connect()` attaches read-only with torque off, and the mirror never calls
   `set_torque()` or `send_action()`, so it can't move the robot (AGENTS.md rule 1 still holds for anything that
   later sends twin → real).
2. **Kinematic, not physical.** Each tick (30 Hz): convert both arms' readings with section 3, write them into
   `data.qpos`, call `mujoco.mj_forward`. Don't step physics, so the twin can never drift from the robot.
3. **Show it.** Use `mujoco.viewer.launch_passive(model, data)` and `viewer.sync()` each tick, or render `front`
   to the operator UI.
4. **Objects stay where the layout puts them.** Without perception the twin only knows the arms. Placing the
   layout to match the real table (or updating it from the D455) is a later step.
5. **Torque.** Because `SO101Arm` attaches with torque off, you can move the arms by hand and the twin follows.
   Support them while doing it: with torque off they drop under gravity. (LeRobot's own follower `connect()` turns
   torque **on**; don't use it directly for the mirror.)

Suggested home: `src/liminal/sim/mirror.py` (WS4), importing `mujoco` and `lerobot` inside functions so tests stay
hardware-free. Command line, for example: `liminal mirror --robot configs/robot/xlerobot.yaml`.

## 5. Demonstrations and FLUX LoRA (already working)

The `twin/` package also contains scripted experts and the full dataset → LoRA → evaluation loop. Details in
[twin/README.md](twin/README.md).

```bash
python -m twin.generate --out RAW --tasks place_green_marker place_red_marker --episodes 100 --workers 48
python -m twin.export   --raw RAW --out DATASET --repo-id local/haller-twin-v1 --media MEDIA   # needs lerobot + av
python -m twin.evaluate --label NAME [--adapter CHECKPOINT/pretrained_model]                  # needs GPU + peft
```

`evaluate.py` has the Pod's model paths (`/opt/flux3-action/...`) hard-coded; change `BASE` to run elsewhere.

First results (2026-10-18): 243 demos (133 green, 110 red marker), 2,400-step LoRA. 0/20 full pick-and-places in
the twin, but the gripper now reaches the red marker (median 5–7 cm, against 14 cm for base FLUX). Best checkpoint
so far: step 1,600. The cutter and box-opening experts are parked; see `twin/README.md`.

## 6. Making the twin match the real robot

In priority order. Each one needs a measurement or a photo from the real rig.

1. **D455 pose.** Capture one real `top` frame and match the render to it (`top` camera in `twin/scene.py`). Today
   the rough tower plate model covers about half of the simulated view.
2. **Joint zero and direction.** Put each real arm in a known pose (e.g. all joints at 0°), mirror it, and compare
   with a photo. Fix any sign or offset in the conversion, not in the model.
3. **Joint ranges.** Copy `~/.cache/huggingface/lerobot/calibration/robots/*/left_follower.json` and
   `right_follower.json` from the rig; `range_min`/`range_max` (4096 ticks = 360°) give the true limits.
4. **Arm spacing.** Measure bolt centre to bolt centre across the plate; set the mounts to ±half
   (`twin/scene.py` and `scene.xml`).
5. **Tabletop sizes.** Table, mat, box, markers and cutter in `twin/config.py` are estimates (`EST`) from the video.
6. **Wrist image roll** and **soft fin-ray fingers**: see the list in [README.md](README.md).
