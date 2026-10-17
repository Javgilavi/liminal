# Simulation experiment scripts

Start with the [simulation guide](../../docs/simulation-validation.md) and
[first validation results](../../docs/validation/single-arm-v1.md).

| Scripts | Purpose |
|---|---|
| `run.sh`, `prepare_experiment.py`, `prepare_scene.py` | Current native/HF/20 Hz reproduction |
| `common.py`, `generate.py` | Shared physics/coordinate contract and scripted teacher |
| `export_dataset.py`, `validate_dataset.py`, `validate_native_export.py` | Collection and dataset checks |
| `train.py`, `evaluate.py`, `summarize_consistent.py` | Official trainer and matched evaluation |
| `prepare_native_hf.py` | Historical conversion using the original experiment directory |
| `validate_export.py`, `offline_loss.py`, other `summarize*.py` | Earlier comparison/audit utilities; original directory layout required |

The remaining notes record the experiment history. The corrected native/HF/20 Hz
experiment is the current baseline; earlier 30 Hz labels and coordinate overrides
must not be mixed with it.

## Experiment history

This experiment generates contact-verified scripted pick-and-place demonstrations,
exports a local LeRobot v3 dataset, trains a task adapter with the official LeRobot
trainer, and evaluates base/raw/EMA policies on new simulation seeds.

The VM experiment directory is `/liminal/experiments/sim-task-lora-20261017`.
Its `run-pipeline.sh`, configs, provenance, logs and checkpoints make the run reproducible.
All simulation rollouts use `common.py`: no hardware or serial interface is constructed.

## Contract

- Dataset/video timestamps are 30 Hz; measured joint states and the absolute commands actually issued.
- Five angles in degrees, gripper last in simulated 0–100% open units.
- Policy shoulder = 90 − sim shoulder; policy elbow = sim elbow + 90.
- Separate scene and wrist images, 256 × 256 each, in the checkpoint's order.
- Eight observed ticks, two visual snapshots, 42 predicted actions, 32 executed.
- Original checkpoint normalization, weights, and processors retained for task adaptation.
- Rank/alpha 32, BF16, activation checkpointing, effective batch 8, EMA 0.999.

## Data and evaluation

The 64 training and 16 validation episodes vary object/container positions,
initial joints, object yaw, lighting, table/object appearance, and camera position/FOV.
Only episodes with bilateral contact, observed lift, release and settled placement
are accepted. `validate_dataset.py` independently checks logged states, commands,
pose mapping, success geometry and disjoint seeds. Validation episodes are appended
last so the official per-task 20% split holds them out.

The current teacher uses separate waypoint stages and a 20-tick minimum duration
per stage. These demonstrations contain visible pauses, which the adapter may
imitate. Smooth transit trajectories are a separate data-quality improvement;
this experiment preserves the staged dataset throughout training and testing.

The result audit found a timing mismatch: each command advances ten 5 ms physics
steps (20 Hz physical control), while dataset timestamps and videos use 30 Hz.
The 600-command evaluation limit therefore spans 30 simulated seconds and 20 video
seconds. Collection and all comparison policies share this mismatch, but it is
a departure from a true 30 Hz control setup and a limitation of these results.

Inference uses seed IDs 20000–20007, which are absent from collection. The same
scenes, supervisor and success checks are used for the base, raw and EMA models.
The official policy and saved pre/postprocessors run on every control tick.
Issued commands are acknowledged as the next delta anchor after safety limiting;
the model retains the checkpoint's 32-tick action queue. All histories reset
between episodes. A completed inference process does not imply task success.

## Version and budget

References:

- https://github.com/black-forest-labs/flux-action/blob/main/docs/so101-lora.md
- https://huggingface.co/docs/lerobot/main/en/flux3

The fork revision `c027ee36` named by BFL was not publicly fetchable. This experiment
uses the clean official `huggingface/lerobot` checkout at
`e624f3f7f8411ec3a02635d06e79373341e5ef35`, its own `examples/flux3/lora.json`,
and its official saved-processor/PEFT loading APIs. Torch 2.10/cu128 and matching
NATTEN are used as documented by LeRobot. Frozen base weights and encoders remain local.

The initial task experiment uses 1,000 microbatches (250 optimizer updates),
compared with the example's 10,000 microbatches. Automatic full-dataset validation
is disabled for this first budget; task outcomes use held-out scene rollouts.
The training and validation episodes remain disjoint. The 8-microbatch benchmark
only verifies the training path and is not the final adapter.

Dataset and checkpoints are local; this workflow performs no Hub uploads.

## Additional normalization comparison

After training completed, the user requested an immediate test of the HF simulator
normalization. Its source was fetched at revision
`d7da38e032da0e136d6de21c218dc616982a8cc9` from
`multimodalart/flux-3-action-so101-sim`. The `NORMALIZATION` constant was extracted
without executing the downloaded source. Source hashes and exact quantiles are saved.

Configuration A uses native sim degrees and checkpoint normalization. B uses the
legacy shoulder/elbow adapter and checkpoint normalization. C uses native degrees
and the HF simulation normalization. For C, the observation history normalizer,
action target normalizer, and action history unnormalizer are explicitly updated
in memory; the base package on disk and the trained adapters stay unchanged.
C was prioritized before resuming A/B/raw/EMA evaluation. Every configuration uses
the same eight held-out scenes and supervisor. The LoRA adapters use B, consistent
with the training data; C is a base-policy inference experiment without training.

## Requested native-angle comparison

`run-native-hf.sh` in the experiment directory runs five matched scenes for the
original checkpoint and raw step-1000 LoRA. Both receive `--frame native` and the
same `--normalization-json hf-sim-normalization.json`: neither applies the shoulder
or elbow mapping. Results are isolated in `evaluation-native-hf-zeroshot` and
`evaluation-native-hf-lora`; earlier runs and checkpoints are retained.

`summarize_native_hf.py` checks the recorded first observation against native
joint angles, the identical scene parameters, and all three processor overrides.
The adapter is not retrained. Its training used dataset90 coordinates and original
normalization, so this changes its inference contract. The report records this
limitation along with the control/video timing discrepancy.

## Consistent replacement experiment

The user requested deletion of the old adapters after the audit. Both old
checkpoint trees (including the benchmark and raw/EMA variants) were deleted;
their demos, logs and evaluation evidence remain. See `old-lora-deleted.json`.

The replacement root is `/liminal/experiments/sim-task-native-hf-20261017`.
`prepare_native_hf.py` creates a separate frozen base with HF statistics in all
three saved processors, and `fps=20` to match the recordings' actual physics.
Large unchanged base assets are linked; original base files are never modified.

`export_dataset.py --frame native --fps 20` converts both absolute states and
commands back to native coordinates and re-encodes the original images at their
true rate. `validate_native_export.py` checks label endpoints and timestamps in
all 80 episodes, decoded camera images, and saved processor statistics.
The source audit independently checks all 32,220 native states against recorded
MuJoCo joint positions. No images or successful demonstrations need regeneration.

The fresh rank-32 LoRA uses the same 1,000-microbatch / 250-update initial budget.
Zero-shot, raw and EMA inference all use native coordinates and their saved HF
processors with no runtime override. Dataset, policy, physics and videos are
consistently 20 Hz. `summarize_consistent.py` verifies this contract and compares
five new matched scene seeds. Staged teacher pauses remain in the reused demos.
