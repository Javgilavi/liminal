# Haller twin: scripted demonstrations → FLUX LoRA

Simulated copy of the real Haller tabletop (dual SO-101 on the XLeRobot tower, D455, round table,
black mat, white box with hinged lid, green/red markers, yellow cutter), used to generate expert
demonstrations and fine-tune FLUX 3 Action. Plan: `../PLAN_SIM2REAL_LORA.md`.

| File | Purpose |
|---|---|
| `config.py` | All dimensions/colours. Values marked EST come from the phone video; replace with measurements. |
| `scene.py` | `build(Layout)` → MuJoCo model for one episode layout (object poses, box pose/lid angle, light, D455 jitter). |
| `ik.py` | Tool-point IK (jaw-centred point, gripper orientation), multi-start for robustness. |
| `expert.py` | Scripted experts (`place_green_marker`, `place_red_marker`, `place_cutter`, `open_box`), 30 Hz servo-ramped execution, success checks. |
| `generate.py` | Randomized layouts → experts in parallel → keep successes (`.npz` rows + qpos, `.json` layout/result). |
| `export.py` | Replay + render D455/wrist at 256×256 → LeRobot v3 dataset in checkpoint coordinates. |
| `evaluate.py` | Closed-loop FLUX (base or base+LoRA) on held-out layouts, same inputs as training. |

```bash
# On the Pod (code in /opt/flux3-action/haller/twin_src):
python -m twin.generate --out RAW --tasks place_green_marker place_red_marker --episodes 100 --workers 48
python -m twin.export --raw RAW --out DATASET --repo-id local/haller-twin-v1 --media MEDIA   # train-venv
python -m twin.evaluate --label NAME [--adapter CHECKPOINT/pretrained_model]                 # train-venv
```

## Expert details that matter

- **Tool point:** the `gripperframe` site is at the fixed fingertip (x along fingers, z toward the moving
  jaw); its convex hull reaches 6.3 mm past the site. Grasps keep the fingertips 4 mm above the mat and
  centre loose objects in the open jaws (22 mm from the fixed face at 30 % opening).
- **Contacts** (from Codex's marker work): objects `condim=6`, friction `0.8 0.003 0.0001`, priority 1,
  so a held cylinder cannot spin out of the rigid jaws.
- **Gentle grip** (Codex): gripper torque capped at 0.5 N·m approaching/closing, 0.2 N·m carrying.
- **Closed-loop expert:** hover above the object, then servo the measured tool point onto the target
  horizontally (≤15 mm) before grasping and releasing; large moves interpolate in joint space.
- **Diversity:** box ±2 cm/±8°, lid 100–115°, all three objects anywhere on the mat at any yaw,
  lighting ×0.75–1.25, D455 ±1 cm/±2°, speed ×0.85–1.2, 15 % deliberate first-grasp misses + recovery.

## Status (2026-10-18)

- Markers: reliable experts; first dataset 133 green + 110 red successful episodes (46 % of random layouts).
- Cutter: parked. Fingertips touch the mat beside it and the gripper tilts; needs an angled grasp.
- Open box: parked. Reachable to ~85°, but the flap grasp slips; needs the real box's flap geometry.
- Not yet verified against the real robot: camera poses/intrinsics, wrist image roll, object sizes,
  servo dynamics, soft fin-ray fingers (rigid stock jaws modelled).
