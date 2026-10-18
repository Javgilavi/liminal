# Single-arm MuJoCo validation

The first experiment collects scripted SO101 pick-and-place demonstrations,
exports a LeRobot dataset, trains a fresh rank-32 task LoRA, and compares zero-shot,
raw and EMA policies on held-out scenes. It has no serial or real-robot access.
[Recorded results](validation/single-arm-v1.md) show partial grasp improvement and
zero completed placements across the five test scenes for each model.

## Contract

| Property | Current reproduction |
|---|---|
| Task | Red block into blue container |
| Coordinates | Five native joint angles in degrees; gripper 0–100% open |
| Normalization | Saved HF Space quantiles in all three processors before training |
| Rate | Dataset, policy and videos at 20 Hz; ten 5 ms physics steps per command |
| Cameras | Scene + wrist, separate RGB images at 256 × 256 |
| Model history | Eight observed ticks, two visual snapshots, 42 predictions, 32 executed |
| Dataset | 64 accepted training episodes + 16 validation episodes |
| Adaptation | Fresh rank/alpha 32; 1,000 microbatches / 250 optimizer updates |
| Evaluation | Matched seeds 20000–20004; at most 600 commands / 30 simulated seconds |

The scene setup approximates BFL's published camera roles. It is not a measured
copy of BFL's extrinsics or this real bench. Simulation and real trials use
separate profiles: the original real-robot baseline in this experiment used 30 Hz without a task LoRA.
The later [real skill runtime](showcase-runtime.md) uses task-LoRA bundles; do not
apply this historical baseline description to the current policy architecture.

## 1. Environment and assets

Use a GPU environment with official LeRobot revision
`e624f3f7f8411ec3a02635d06e79373341e5ef35`, its FLUX/training/PEFT dependencies,
and compatible Torch/NATTEN versions. The recorded GPU environment used Python
3.12 and Torch 2.10/CUDA 12.8. Install this repo into that environment and the
simulation/export dependencies:

```bash
pip install -e '.[sim]' numpy pillow imageio imageio-ffmpeg
```

LeRobot supplies the dataset/training stack; a core Liminal installation alone is
insufficient. FFmpeg/video decoding and EGL rendering must work. The recorded
training peak allocated memory was about 28.8 GiB (31.4 GiB reserved), not including
other processes. Checkpoints and optimizer states need additional disk space.

Obtain a complete released SO101 package and its encoders using the
[real-inference preparation guide](official-inference.md#2-prepare-the-gpu-worker).
Keep it separate from outputs. The scripts create a linked simulation package
with HF processors and `fps=20`; they never rewrite the released base.

The robot meshes come from MuJoCo Menagerie at a recorded revision:

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/google-deepmind/mujoco_menagerie outputs/mujoco_menagerie
git -C outputs/mujoco_menagerie sparse-checkout set robotstudio_so101
git -C outputs/mujoco_menagerie checkout 62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35
```

The modified scene/model and camera setup are versioned in
[`sim/single_arm`](../sim/single_arm/README.md), including the upstream license.
Meshes stay in the external checkout.

## 2. Run the pipeline

From the repository root, with the GPU environment active:

```bash
# Explicitly starts simulation collection, GPU training and evaluation.
SIM_PYTHON=python MUJOCO_GL=egl bash scripts/sim_flux_lora/run.sh \
  outputs/single-arm-v1 /absolute/path/to/released-so101 outputs/mujoco_menagerie
```

Use a new output directory. This can take hours and uses substantial GPU/disk
resources. It performs no Hub uploads. Progress is printed by stage; detailed
logs are under the output root. Collection requires the requested number of
independently accepted demos and fails if it cannot find enough within its seed
budget. Failed teacher attempts stay in logs and are not exported.

The stages are:

1. Prepare the base processors and 20 Hz configuration, and materialize the scene.
2. Collect rendered teacher demos, using separate train and validation seed ranges.
3. Validate measured-state/command alignment, contact, lift and settled placement.
4. Convert legacy teacher labels to native coordinates and export videos at 20 Hz.
5. Validate the exported labels, timestamps, videos and saved processor statistics.
6. Train with the official LeRobot trainer and [`train.json`](../experiments/single-arm-v1/train.json).
7. Evaluate zero-shot/raw/EMA on the same five new scenes and produce `RESULTS.md`.

This is a new reproduction command assembled from the recorded stages. It was
checked without rerunning the complete GPU training experiment. Recorded numbers
come from the saved [simulation results](../experiments/single-arm-v1/simulation-results.json),
not from an unreported rerun.

## 3. Inspect the results

Each `evaluation-*/seed-*/` contains a video, `report.json` and `trajectory.npz`.
The report records seed, scene randomization, actual physics rate, coordinates,
contact, maximum lift, stable-placement ticks and safety interventions.
`dataset-validation.json` records label/video/processor checks;
`training-metrics.json` records time and memory.

Placement requires bilateral contact, observed lift greater than 3.5 cm, release,
and slow settled placement inside the container for 15 ticks. Process completion,
contact, or a brief lift alone cannot satisfy it. The shared simulation supervisor
clips joint travel and per-step motion for every comparison model and records
interventions; this differs from the real runner's whole-chunk rejection.

For one stage at a time, run a script's `--help` under `scripts/sim_flux_lora/`.
`LIMINAL_SIM_ROOT` can point to a prepared scene directory. Inference uses official
`select_action` and both processors every control tick, preserves the model's
queued horizon, records issued commands as the next history anchor, and resets
policy/processors between episodes.

## Historical comparisons and next task

The initial collection/evaluation labeled ten 5 ms substeps as 30 Hz. The corrected
experiment reuses the same images/states at their actual 20 Hz, with native
coordinates and HF statistics consistently saved before training. Earlier adapters
trained under the old contract were deleted; their results are historical controls,
not the current validation baseline. The original scripts and their assumptions
are indexed in [`scripts/sim_flux_lora/README.md`](../scripts/sim_flux_lora/README.md).

`sim_flux_assembly/` contains a subsequent scripted gear/collar assembly prototype.
It is separate from the pick-and-place LoRA validation. No learned assembly
success or real-robot transfer is established by these scripts.
