# First single-arm validation

**17 October 2026 · WS1 · SO101 · FLUX 3 Action**

This validation establishes a working model/processor integration, a reproducible
MuJoCo adaptation experiment, and real-camera/real-motor inference trials.
Simulation shows partial grasp improvement after adaptation. Completed placement
and physical task success have not been established. Bimanual orchestration is
outside this validation.

## What was tested

| Experiment | Policy | Coordinates / statistics | Rate | Outcome |
|---|---|---|---|---|
| Corrected MuJoCo comparison | Official LeRobot; zero-shot vs fresh raw/EMA LoRA | Native / saved HF Space | 20 Hz physics, data, policy and video | Raw adapter lifts in 2/5 scenes; all placements 0/5 |
| Current physical profile | Official LeRobot; released weights, no adapter | Native / saved HF Space | 30 Hz playback, synchronous inference pauses | Predictions and motor commands recorded; task success unverified |
| Physical released baseline | Official LeRobot; original processors | Released coordinates and optional historical mapping | 30 Hz playback | Bounds refusals and tracking faults recorded |
| Early physical trials | BFL standalone policy API, custom rebasing | Original statistics | Limited/clipped commands | Historical debugging evidence, superseded execution path |

The official component is LeRobot's `Flux3Policy` plus its saved pre/postprocessors.
SSH transport, observation collection and execution are Liminal code. Haller
owns the motors/cameras for the default physical runner. HF Space normalization
is an explicit choice that differs from the original released processor statistics;
using official model code does not make that choice a validated physical calibration.

## MuJoCo result

Task: pick up a red block and place it in a blue container. The corrected experiment
uses 64 accepted training and 16 validation episodes (32,220 frames), native
coordinates, HF normalization saved before training, and consistent 20 Hz timing.
A fresh rank/alpha-32 adapter trains for 1,000 microbatches / 250 optimizer updates.

All policies see the same five new scene seeds (20000–20004), with a maximum of
600 commands / 30 simulated seconds per episode.

| Model | Contact | Bilateral contact | Lift > 3.5 cm | Completed placement | Maximum lift |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 5/5 | 0/5 | 0/5 | 0/5 | 1.92 cm |
| Raw LoRA | 5/5 | 3/5 | 2/5 | 0/5 | 15.69 cm |
| EMA LoRA | 2/5 | 0/5 | 0/5 | 0/5 | 1.28 cm |

Placement is checked independently: bilateral contact, lift, release, and slow
settled placement inside the container for 15 ticks. A successful teacher episode
or completed evaluation process is not counted as learned-policy success.

Training took approximately 18 minutes; peak GPU allocation was 28.8 GiB, with
31.4 GiB reserved. These measurements cover training, not the complete pipeline.
The small test set and narrow randomization do not support broad success claims.
Teacher trajectories contain staged pauses that the adapter may imitate.

Earlier experiments mislabeled 20 Hz physical control as 30 Hz and used a legacy
shoulder/elbow mapping. The corrected run consistently fixes rate, coordinates
and statistics before training/evaluation. Earlier incompatible adapters were
deleted; their comparison scripts remain historical audit tools.

[Simulation guide](../simulation-validation.md) ·
[Full recorded results](../../experiments/single-arm-v1/simulation-results.json)

## Physical inference result

The committed trial index contains the official-policy reports available when
this validation was assembled. It includes observation-only checks, execution
requests, both normalization profiles and explicit stop reasons. These are
engineering trials with different prompts/starting poses, not a controlled task
success-rate benchmark.

The HF Space profile produced finite predictions and executed motor commands.
Two red-marker trials reached their configured duration; both were explicitly
labeled `task_success_unverified`. Other trials stopped on disabled torque,
calibrated wrist travel violations, or sustained non-tracking under load. Released
baseline trials also recorded wrist-travel refusals and an elbow stall.
No independent real-task success labels were recorded.

The early 16:29 standalone run executed 96 commands. Its second shoulder chunk
requested about 69.3° of net motion but the then-active limiter committed only
21.3°. Wrist targets exceeded calibrated travel, and elbow load reached 84.8%
with a pre-release target error of about 17.8°. The larger final error occurred
after torque release and sagging. These observations motivated the subsequent
unchanged-target LeRobot path; they do not describe the current limiter settings.

Current official execution bypasses rebasing and policy slew limiting, rejects
out-of-travel horizons, and checks committed targets. Telemetry, load/stall,
stale-input and stop guards remain local. These changes address integration
behavior, but do not establish the cause of physical resistance or solve task
performance. Do not interpret a bounds pass as proof of calibration compatibility.

[Real-robot guide](../official-inference.md) ·
[Official trial index](../../experiments/single-arm-v1/real-robot-results.json)

## Reproduction and evidence

- [Experiment index](../../experiments/single-arm-v1/README.md): configs, versions,
  artifact hashes and raw-data locations.
- [Simulation scripts](../../scripts/sim_flux_lora/README.md): collection, validation,
  training and matched evaluation.
- [Haller integration](../../integrations/haller/README.md): versioned controller patch.
- [Legacy inference](../live-inference.md): interpretation of early logs.

Model/data revisions are pinned in the manifest. Complete datasets, checkpoints,
video and camera histories remain on the experiment machines; no public artifact
bundle is claimed. Early physical runs lack complete image histories. The new
portable reproduction pipeline is packaged from the recorded stages; the full
GPU training/robot trials were not repeated during repository cleanup.

## Repository checks

Packaging checks passed with 155 repository tests and one separate Haller
integration test using fake servos/in-memory streams. The default suite skips
that optional external dependency. Ruff, shell syntax, all five task-file
validations, the mock holding demo, local documentation links and artifact hashes
also passed. The controller patch applies to its recorded base. A synthetic
checkpoint-preparation check verified all three saved quantile tensors, the
20 Hz simulation config and unchanged source-package hashes.

These checks exercise software contracts. GPU training and live robot execution
were not repeated during packaging; the experimental results above come from
retained runs.

## Next validation

Resolve the observed physical tracking resistance, verify camera and joint
conventions on the real rig, improve the simulation teacher's transit/release
behavior, and evaluate new adapters on a larger matched held-out set. Real
adaptation requires demonstrations and independent task-success labeling on this
bench. Integration tests and mock task tests cannot substitute for those outcomes.
