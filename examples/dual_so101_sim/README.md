# dual_so101_sim

This is the versioned source snapshot of the working `../dual_so101_sim/`
directory. Existing local launch commands still use that original directory.
Virtual environments, downloaded LeRobot source, model weights, rendered outputs
and private SSH configuration are deliberately excluded.

To use this copy, run the setup below from `examples/dual_so101_sim/`.
The browser app (`flux/`, with its remote GPU workflow `flux/RUNPOD_FLUX.md`) and the sim-to-real plan
(`PLAN_SIM2REAL_LORA.md`) are not part of this snapshot: they stay in the original
`~/erh_hack/dual_so101_sim/` directory, together with its Pod host pin (`flux/.runpod/known_hosts`).
No host trust metadata or credentials are committed.

MuJoCo model of the **Haller** rig ([oscardvs/haller_ws](https://github.com/oscardvs/haller_ws)) as it physically
exists today (Phase A): two **SO-ARM101** followers on a static **XLeRobot tower** over a bench, a RealSense D455 on
the tower, an OV2710 camera on each wrist. The mobile base (Phase B) isn't built yet, so it isn't modelled.

```bash
uv venv .venv && uv pip install --python .venv/bin/python mujoco pillow
MUJOCO_GL=egl .venv/bin/python check.py                 # headless smoke test -> renders/*.png
.venv/bin/python -m mujoco.viewer --mjcf=scene.xml      # interactive viewer (needs a display)

# Through liminal (configs/robot/mujoco_dual.yaml points here; run from liminal/)
uv pip install --python ../dual_so101_sim/.venv/bin/python -e .
MUJOCO_GL=egl ../dual_so101_sim/.venv/bin/liminal run configs/tasks/lid_hold_insert.yaml \
    --robot configs/robot/mujoco_dual.yaml --backend mock
```

## Names and units

- Joints/actuators: `left_` / `right_` + `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
  (LeRobot names and order).
- Units: radians, zero = middle of each joint's range = LeRobot SO-101 "new calibration". A real arm read in LeRobot
  degree mode maps to `deg2rad(value)`. LeRobot's default `RANGE_M100_100` mode and its 0–100 gripper don't map
  directly. Convert them first.
- Cameras: `top` (D455 RGB, 640×480), `left_wrist`, `right_wrist` (1280×720) are the real cameras; `front` is a sim-only view.
- Frame: bench top z = 0, arms reach toward −y.

## How close to the real robot is it?

Audited against haller_ws @ `b46b827` (2026-09-11): `docs/hardware_inventory.md`, the website BOM,
`docs/setup/{so101-arm,gripper-cameras}.md`, `hmi/backend/config.desktop-real.yaml` (the current two-arm rig),
`hmi/backend/haller_hmi/arm.py`, `scripts/99-haller-devices.rules`.

| Item | Real robot (haller_ws) | Sim | Status |
|------|------------------------|-----|--------|
| Arms | 2× SO-ARM101, mechanically identical, both followers | official SO-101 MJCF (TheRobotStudio `a758567`), CAD masses/inertias | ✅ 1:1 |
| Joint names / zero | LeRobot `so101_follower`, zero = mid-range | same names, zero = mid-range | ✅ 1:1 |
| Joint ranges | measured sweeps, e.g. shoulder_pan 238°, shoulder_lift 211° (calibration files not in the repo) | CAD limits: pan 220°, lift 200° | ⚠️ ~5–10% short |
| Servos | 12× STS3215 **7.4 V, 19 kg·cm, C001 1/345** | forcerange ±1.86 N·m | ✅ spec |
| Servo stiffness | `P_Coefficient=32` on every motor (`arm.py`) | kp 1996 (upstream 998 assumed P=16, so ×2) | ✅ derived, damping/friction upstream estimates |
| Fingers | soft fin-ray TPU 95A (XLeRobot `hardware/SO101_soft_fin.stl`) | stock SO-101 rigid jaws | ❌ not modelled (see below) |
| Wrist camera mount | TheRobotStudio Hex-Nut 32×32 wrist mount | same part, from CAD | ✅ 1:1 |
| Wrist cameras | OV2710 (Sonix 0c45:6230), **87.0° × 56.2° @ 1280×720**, 30 fps, mounted upside down + `rotation: 180` | fovy 56.2 at 1280×720, at the lens face | ✅ FOV measured, image roll VERIFY |
| Base camera | **RealSense D455 RGB** (`/dev/haller_cam_mast`), **640×480**, dataset key `top`, on the tower looking over the arms toward the operator | `top`, fovy 65 (Intel RGB 90×65; haller's "87×58" is the depth FOV), 640×480 | ✅ model/res, **pose VERIFY** |
| Mount height | arm plate 0.10 m above bench ("the measured riser") | 0.10 m | ✅ measured |
| Arm spacing | **never measured**. haller uses ±0.20 m marked "PLACEHOLDER, measure" | ±0.15 m (XLeRobot tower CAD) | ❌ VERIFY |
| Tower plate / mast | XLeRobot tower + arm base plate (cart, mecanum base, head gimbal not used) | rough boxes | ⚠️ visual approx |
| Left/right | operator's left facing the arms = −x, reach −y | same | ✅ |
| Mobile base | Phase B, not built | not modelled | n/a |
| Bench, cubes, place zone | haller's sim preset (not real objects) | same | sim only |

### Still not 1:1, and what's needed to fix it

1. **Arm spacing.** Measure bolt centre to bolt centre across the plate, then set `left_mount`/`right_mount` x to ±half.
2. **D455 pose.** Measure the lens position and tilt relative to the plate, set `top` in `scene.xml`, and compare a
   render to a real 640×480 frame.
3. **Fin-ray fingers.** `SO101_soft_fin.stl` is one fused part (gripper body + TPU finger) in raw CAD coordinates.
   Fitting it means splitting it in CAD and registering it to the SO-101 jaw. TPU compliance can't be reproduced
   with a rigid mesh anyway, so friction/softness would have to be tuned on real grasps.
4. **Joint ranges.** Copy `haller_left.json` / `haller_right.json` from `~/.cache/huggingface/lerobot/calibration/`
   on the rig. `range_min`/`range_max` (4096 ticks = 360°) give the real limits.
5. **Wrist image roll.** Compare a sim `left_wrist` render with a real frame at the same pose. If it's rotated,
   change the `xyaxes` of the `wrist` camera in `so101/so101.xml`.

## Files

| File | What |
|------|------|
| `scene.xml` | the rig. Arms come in via `<attach prefix="left_"/"right_">`. |
| `so101/so101.xml` | upstream `so101_new_calib_camera.xml` (Apache-2.0, see `so101/LICENSE`). Local edits: forcerange → ±1.86, kp ×2, added `wrist` camera at the lens face |
| `check.py` | puts both arms in a pose, steps 2 s, prints tracking error, renders all cameras to `renders/` |
| `twin/` | tabletop digital twin (table, mat, box, markers, cutter), scripted demos and FLUX LoRA tooling. See [`DIGITAL_TWIN.md`](DIGITAL_TWIN.md) |
| `flux/` (not in this snapshot) | browser UI that prompts one arm with FLUX 3 Action running on a GPU box; in the original `~/erh_hack/dual_so101_sim/flux/` |
