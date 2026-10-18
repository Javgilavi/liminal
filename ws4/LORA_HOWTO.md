# Train your own FLUX 3 Action LoRA (Haller / SO-101): the short version

What worked for us on 17 to 18 October 2026, in the order you need it. Every number here was measured on our runs.

## 1. Record the data

- One task, one arm, one LeRobot v3 dataset at 30 fps, with a scene camera and that arm's wrist camera.
  VR teleop via haller_ws works well.
- **Use one exact task string** (it is the prompt at run time). Start every take from the same pose, and end in a
  consistent state.
- **Keep the scene as it will be at run time**, including where the other arm is parked and anything it holds. The
  head camera sees it.
- 60 good takes was enough per skill. Mark bad takes in `review.json`, then write `ws4_split.json` with `"status":
  "final"`: hold out the last 3 good takes per layout, drop the rejects.
- Don't run heavy jobs on the recording laptop: CPU spikes break the 30 Hz loop.

Our datasets are public, in exactly this format:
- [open the box](https://huggingface.co/datasets/Oskrt/haller_open_the_box)
- [green pen](https://huggingface.co/datasets/Oskrt/haller_put_the_green_pen_in_the_green_slot)
- [yellow cutter](https://huggingface.co/datasets/Oskrt/haller_put_the_yellow_cutter_in_the_yellow_slot)
- [red pen](https://huggingface.co/datasets/Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box)

## 2. The one thing that matters most: normalization

The released `black-forest-labs/flux-3-action-so101` package normalizes joints with statistics from one old SO-100
dataset (legacy degrees). Today's LeRobot calibration does not match it, and zero-shot FLUX drives the arm into its
limits (0 of 16 attempts for us).

The fix, before any training: **recompute q01/q99 from your own train takes** ("variant B", `ws4/single_arm/a0r_stats.py`
plus `flux.package_tools.renormalize_package`). The pipeline below does it for you. Train the LoRA from that base, and
always serve the adapter with the same statistics.

## 3. Get a GPU pod

- **GPU:** an RTX PRO 6000 96 GB works best. An H100 80 GB trains faster, but serves at most 3 models plus margin.
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, about 200 GB of disk, TCP 22 exposed.
- **Env vars:** `HF_HOME=/workspace/hf`, `LIMINAL=/workspace/liminal`, `WS4_ROOT=/workspace/ws4`,
  `UV_CACHE_DIR=/workspace/uv-cache`, `UV_PYTHON_INSTALL_DIR=/workspace/uv-python`, plus your SSH public key as
  `PUBLIC_KEY`.
- Your HF account must have clicked "Agree and access" on `flux-3-action-base` and `flux-3-action-so101`.

Add the pod to `~/.ssh/config` (`Host mypod`, then `HostName IP`, `Port PORT`, `User root`, `IdentityFile ~/.ssh/KEY`),
then:

```bash
bash ws4/cloud/push_ws4.sh mypod /workspace/liminal    # copy ws4/ to the pod (or git clone the ws4-handoff branch there)
ssh mypod 'export HF_TOKEN=hf_...; bash /workspace/liminal/ws4/cloud/bootstrap.sh'   # about 11 to 15 min
```

**Arm a watchdog** (`ws4/ttt/pod_job_watchdog.sh 30`) so an idle pod stops itself. Idle GPUs are where the money goes.

## 4. Train

```bash
export KEY=~/.ssh/YOUR_KEY QUOTA_GB=200    # your SSH key; QUOTA_GB = the pod's disk or volume size (free-space check)
# laptop: checks the split, rsyncs the raw dataset to the pod (SIDE: left = robot RIGHT arm, right = robot LEFT arm)
UPLOAD_HOST=root@HOST UPLOAD_PORT=PORT bash ws4/cloud/skill_lora.sh prep  MYSKILL ~/.cache/huggingface/lerobot/ws4/MYDATASET right
# pod: per-arm conversion, B statistics, base package, 5,000-step LoRA
bash ws4/cloud/skill_lora.sh train MYSKILL HOST PORT
# pod: held-out action error for RAW and EMA, then pick the winner
bash ws4/cloud/skill_lora.sh eval  MYSKILL HOST PORT
```

- **Settings:** the LeRobot preset `examples/flux3/lora.json`: rank and alpha 32, LR 1e-4 (heads 5e-4), EMA 0.999,
  batch 2 x accumulation 4. That is about 1.0 s per step on an RTX PRO 6000, so 5,000 steps take about 1 h 30 min.
- **Pick on held-out action error, not on eval loss.** Measure arm error in degrees and direction agreement. The loss
  barely moves while the actions keep improving.
- **Evaluate RAW and EMA:** early in training RAW usually wins (pen at 5k: 3.24 vs 3.62 deg).
- **Guidance:** keep 1/1. Guidance 3/3 and 4/1 gained nothing at twice the compute.

## 5. Serve and run on the robot

```bash
python ws4/cloud/lora_adapter.py serve --port 8790 --token-file TOKEN --text-encoder-on-cpu \
  --package myskill=/workspace/packages/so101_haller_MYSKILL_B+/workspace/skills/MYSKILL/lora/checkpoints/005000/pretrained_model \
  --override guidance_scale=1.0 --override guidance_scale_action=1.0 --override compile_model=true
```

- The adapter is folded into the weights in memory (no 14 GB merge). That's about 16.5 GB of GPU memory per model,
  and about 0.35 s per chunk on the server.
- **Never send policy output straight to the motors.** Run it through `ws4/single_arm/run_single_arm.py`, with a skill
  workspace built from your demos (`build_skill.py`). That gives you the keep-out, the demo envelope, 1 deg/tick, the
  stall and load freeze, and a torque cap. Zero-shot FLUX broke a servo on our first arm before that existed.

## 6. What to expect (red pen, 12 held-out takes)

| Model | Arm error (deg) | Direction | Gripper error |
|---|---|---|---|
| Hold still | 5.6 | n/a | 8.5 % |
| Zero-shot, our statistics (B84) | 5.5 | 66.5 % | 11.3 % |
| LoRA RAW 5k | 3.2 | 83 % | 4.7 % |
| **LoRA RAW 10k** | **3.15** | **89 %** | **4.5 %** |
| ACT baseline (100k) | 4.3 | 85 % | 4.7 % |

That adapter is public: [Oskrt/flux-3-action-so101-haller-pen-lora](https://huggingface.co/Oskrt/flux-3-action-so101-haller-pen-lora).

## Pitfalls we hit

- **Arm naming:** the recorder uses the operator frame, so `right_*` and `right_wrist` are the robot's LEFT arm.
- **Cameras:** `scene` = head camera (not rotated); `wrist` = the task arm's wrist camera, rotated 180 at capture.
  Check live frames against `ws4/single_arm/frames/reference_*` before a run.
- **Keep-out:** a skill whose slot is across the robot's midline needs a per-skill midline override, recorded in
  `skill.json`.
- **Storage:** a network volume full mid-checkpoint kills the run. Check `du` and keep 40 GB free.

More detail:
- [pen_lora.md](research/pen_lora.md) and [skills.md](research/skills.md): our numbers.
- BFL's `docs/so101-lora.md` (flux-action) and LeRobot's `docs/source/flux3.mdx`: the upstream recipe.
