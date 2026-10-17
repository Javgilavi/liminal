# ACT baseline for the Haller pen task

LeRobot ACT trained on the same 84 VR-teleop train takes as the FLUX 3 Action LoRA (`ws4_split.json` "final"),
evaluated offline with the same protocol, and runnable through the single-arm runner's safety stack. It is the
baseline and fallback; the headline is FLUX LoRA vs ACT on the same demos and held-out takes
(`ws4/research/pen_lora.md`, "ACT baseline").

| File | What |
|---|---|
| `backend.py` | `ActBackend`: an ACT checkpoint behind `flux.backends.FluxBackend` (contract + `predict`), absolute 6-D joint chunks |
| `serve.py` | the same backend behind `flux.remote`'s HTTP protocol (`RemoteBackend` and `run_single_arm.py --backend remote` work unchanged) |
| `offline_eval.py` | held-out action error with `single_arm/offline_diag.py`'s frames, metrics and table |
| `dry_run.py` | `run_single_arm.py` on a recorded take (fake arm showing the recorded frames), all guards on |
| `train_view.py` | dataset view with train-only normalization statistics |
| `train_act.sh`, `eval_act.sh`, `curve_watch.sh` | pod scripts (outputs under `/workspace/act/` only) |
| `tests/` | CPU tests on a tiny random ACT (no GPU, network or serial port) |

## Contract

ACT conditions on the current state and the current frame of each camera, so the backend's `ModelContract` has
`n_obs_steps` 1, one snapshot, `image_hw` (480, 640) (the trained resolution: no resize of the live 640x480
frames), `chunk_size` 100, `absolute_dims` all six. The runner executes `n_action_steps` commands per chunk (ACT's
default 100 = 3.3 s; `--act-steps` / `serve.py --act-steps` lower it) and re-plans with async inference, so the arm
holds for one inference (about 70 ms on the laptop's T550, 270 ms on its CPU). Units are the dataset's: LeRobot
degrees, gripper 0..100, no joint transform (`--scale` / `--offset` unset), same cameras and wrist rotation as the
FLUX B84 runs. Every command goes through `DualArmRunner` and the runner's guard: keep-out and demo envelope,
`--max-step` per tick, soft limits, stall and load freeze, torque cap. Nothing in `ws4/act` sends motor commands.

Checkpoints are written by the pod's LeRobot (146d847); the laptop runs LeRobot 0.5.1. `backend.py` loads them
through a temporary config view that drops the two newer `PreTrainedConfig` fields (`dtype`, `pretrained_revision`,
both None) and sets `pretrained_backbone_weights` to None, so loading never downloads the ImageNet ResNet18 (every
backbone tensor comes from `model.safetensors`).

## Run it on the arm (Oscar's call; not done)

Without any runner change, serve the checkpoint on the laptop and point the runner at it:

```bash
python ws4/act/serve.py --package act_100k=ws4/.cache/act/act_pen84_100k --port 8791
```

then `run_single_arm.py` with eval_pen.sh's Haller flags and `--backend remote --server http://127.0.0.1:8791
--package act_100k --variant ACT-100k --checkpoint "<path> (LeRobot ACT)" --condition post_training`.
With the `--backend act` runner patch applied (`act_run_single_arm.patch` in the shared notes), the same runs
in-process: `--backend act --act-checkpoint <path>`.

Dry run on a held-out take first (no arm, no camera):

```bash
python ws4/act/dry_run.py --checkpoint ws4/.cache/act/act_pen84_100k --root ws4/.cache/real_lora/pen_in_right_all --episode 20
```
