# PLAN.md

The strategy and the key technical decisions. [TASK.md](TASK.md) is the hour-by-hour
breakdown per workstream; [docs/architecture.md](docs/architecture.md) is the design.

## Goal

Two SO-101 arms on the XLeRobot rig that **depend on each other**. One arm
changes or maintains a condition in the scene so the other can finish a task it
couldn't do alone. Long-horizon tasks are broken into verifiable subtasks, coordinated
by a behavior tree, and executed with FLUX 3 Action as the learned manipulation backend.

**Demos:**
- **A, cooperative holding:** the left arm opens and holds a lid, the right arm inserts a block,
  and the left releases only after the insert is verified (`lid_hold_insert`).
- **B, long-horizon dependency:** the target is blocked, so one arm clears the obstacle and
  the other picks the target (`clear_obstacle_pick`, then `tidy_table_long_horizon`).

## Current decision: task LoRA and two parallel per-arm policies

The initial plan was to fine-tune a unified bimanual XLeRobot embodiment (12-D actions). That plan
was superseded by task LoRA fine-tuning of the existing 6-D SO-101 policy, with two independent FLUX
instances, one per arm, coordinated by the real skill-runtime harness (behavior tree, checkpoint
registry, perception and safety).

- `configs/policy/flux3_so101.yaml` and `showcase_registry.yaml` select
  `instances: per_arm` (formerly called A2). Each arm has its own model, history,
  checkpoint state and locks. `devices` can assign separate GPUs; one sufficiently
  large GPU can host both, subject to measured memory and latency.
- Skills retain the existing 6-D action contract. Task LoRA adapters are merged
  into complete bundles with saved processors and validated provenance.
- The harness chooses skills, sequences dependencies, permits parallel branches,
  enforces ownership and checks independent perception. Coordination is supplied
  by the harness rather than learned by a joint 12-D policy.
- The showcase right arm opens and locally holds the lid; left-arm placements
  run sequentially. Parallel policy capacity does not override these dependencies.
- `instances: shared` (formerly A1) remains an explicit serialized fallback for
  memory-constrained deployments. It is not the selected two-policy architecture.

See [the runtime guide](docs/showcase-runtime.md) for checkpoint availability and
validation limits. Independent policy construction and orchestration are tested
without hardware; full learned dual-arm performance is not yet established.

### Original embodiment plan (superseded)

We initially planned a single XLeRobot bimanual policy with concatenated
`[left 6 | right 6]` actions, bimanual demonstrations and an adapted action head.
That was called Option B. It is no longer an active fine-tuning workstream.
`flux3_dual_bimanual.yaml`, `so101_dual.yaml` in the embodiment folder, and
`lid_hold_insert_bimanual.yaml` retain this unvalidated design for compatibility
and mock tests. They are not deployment instructions or evidence of trained weights.
The physical XLeRobot profile is still used with the selected per-arm policies.

## Assembly policy: bounded fitting and execution investigation

This recorded single-arm simulation investigation informs task LoRA training.
Its historical measurements do not establish two-arm policy performance. Implementation entry points are
`scripts/sim_flux_assembly/fitting.py`, `fit_train.py`, `fit_audit.py`, and
`run_fitting.py`. Run evidence belongs outside Git under the experiment directory.

### Evidence and hypothesis

Frozen baseline: native SO-101 coordinates, no shoulder/elbow adapter, saved HF
simulator normalization, 20 Hz, Euler 4 / CFG 3, 42 predicted / 32 executed actions.
The 3,000-microbatch four-skill adapter achieved 0/5 full assemblies and 0/12
isolated skills. Four teacher-command replays reproduced complete assemblies
without drift. A separate 500-microbatch pickup continuation improved recorded
training-history error but still failed both training and held-out pickups.
The model has not demonstrated reproduction of its training pickup.

Hypothesis: fixed-window fitting separates optimization/serialization faults from
closed-loop startup and distribution shift. Command-error improvement alone does
not establish manipulation success. No claim about physical calibration or
sim-to-real accuracy follows from these experiments.

### Frozen inputs and protocol

- Repository: this checkout; remote workspace:
  `/liminal/experiments/sim-assembly-20261017` on the existing inference VM.
- Warm start: original raw `training/checkpoints/003000/pretrained_model`;
  preserve it and the previous one-demo adapter.
- Fit only eight complete windows from atomic episode 0, `pick_gear`, seed 100.
  Freeze dataset indices and source ticks before measuring the baseline.
  Use the official episode sampler's valid-window boundaries. Never fit holdout.
- Report those exact training windows and eight matching phase-spanning windows
  from held-out seed 10000 separately. The latter are diagnostic development
  validation, not an untouched final test set.
- Keep saved normalization, camera order, command deltas, gripper units, sampling
  and 20 Hz physics unchanged. The official BFL recipe uses 30 Hz and a different
  pinned checkout; record and audit those departures. A 30 Hz comparison must
  regenerate or properly resample all time-aligned streams in a separate run.
- Disable caption dropout and image augmentation for the memorization diagnostic.
  Keep the official stochastic diffusion objective; do not claim deterministic
  training merely because the sampler and evaluation seeds are frozen.

### Implementation and acceptance

1. **Contract audit before training.** Hash source, config, manifest, checkpoint
   tensors and processor artifacts. Check actual decoded dataset windows have no
   padding and match source states/absolute commands at the declared indices.
   Compare official training preprocessing with offline inference preprocessing:
   measured-state history, past commands, camera history and round-trip command
   denormalization. Validate loaded normalization against the frozen statistics.
   Fail with an actionable error on any mismatch; retain the failure report.
2. **Bounded fixed-window fit.** Wrap the pinned official trainer only to restrict
   its sampler and collect evidence; retain its optimizer, accumulation, PEFT,
   processors and checkpoint writer. Batch 2, accumulation 4, rank/alpha 32,
   BF16, gradient checkpointing and the existing 42% allocator cap. At most
   1,000 microbatches / 250 optimizer updates, save at 250-microbatch intervals.
   Approximately 17 minutes training at the measured rate; allow 30 minutes for
   training startup and checkpoint I/O. Validate finite loss, nonzero finite
   gradients, optimizer coverage of all trainable modules, and adapter/head
   changes. Load the saved checkpoint for evaluation in a fresh process.
3. **Matched prediction and pickup evaluation.** Evaluate the same frozen windows
   before and after, with fixed inference seeds. Report per-joint and gripper
   error at horizons 1/8/32/42, cumulative command drift, and separate splits.
   Run pickup from the recorded training and held-out initial states, using only
   learned actions and independent physics predicates. Save video, raw/issued
   commands and task failures. No teacher fallback in learned rollouts.
4. **Decision gate.** A provisional fitting pass requires at least 50% reduction
   in training horizon-32 joint RMSE and at most 2 degrees mean joint RMSE there;
   also report gripper error (target at most 5 percentage points). These are
   engineering thresholds, not calibrated success predictors. Pickup success is
   a separate mandatory requirement before promotion. Failure to fit triggers
   optimization/gradient/serialization review. Good window fitting with failed
   training pickup triggers startup/history and off-trajectory investigation.
   Successful training pickup with failed new scenes justifies targeted recovery
   collection. Do not automatically launch a second fit or a broad data campaign.
5. **Conditional expansion.** Once nominal pickup works, collect targeted approach,
   closure, missed-grasp and recovery examples with disjoint development seeds.
   Compare against the frozen baseline under the same budget. Use a fresh set of
   20 scene seeds for a provisional 18/20 pickup gate; reserve a separate final
   test set after choices are frozen. Extend this procedure to the other skills,
   then measure complete assemblies independently. Atomic rates do not establish
   end-to-end success. Adopt bounded retries only after their recovery skills work.

### Budget, isolation, failure handling and validation

One fit only; one model process at a time; no GPU provisioning, physical robot
motion, production-service changes or harness-branch merge. Check current GPU
headroom and disk before allocation. Use a unique output directory, an exclusive
experiment lock and a resumable evidence manifest. A process timeout or OOM is a
failed experiment, never a successful task. Propagate subprocess exit status,
retain logs and completed checkpoints, and stop dependent work on failed gates.
Report source/config/input hashes, actual optimizer updates, wall time, GPU
memory, trainable-parameter coverage and limitations. Preserve shared dirty files.

Tests use mocks and standard-library fixtures: frozen-window boundaries, holdout
exclusion, sampler repetition, budget bounds, decision gates and downstream
failure propagation. Run the complete mock pytest suite and Ruff before handing
off. Scientific acceptance additionally requires the real contract audit, fit and
saved-checkpoint rollout evidence. Record outcomes in the run report; do not turn
this strategy document into a session log.

Run inside the existing remote training environment with `LIMINAL_SIM_ROOT` and
the Liminal source import path configured:

```bash
python scripts/sim_flux_assembly/run_fitting.py \
  --root "$ASSEMBLY_ROOT" --output "$ASSEMBLY_ROOT/unique-fixed-window-run" \
  --lerobot "$LEROBOT_CHECKOUT" --evaluator "$ASSEMBLY_EVALUATOR" --steps 1000
```

`--audit-only` runs the boundary checks without loading policy weights or fitting.
Existing output directories are refused. `status.json` records each exact command,
timeouts and input hashes; `optimization.json` records applied updates and gradient
probes; `decision.json` separates scientific acceptance from process completion.
Training and inference logs, prediction tensors, checkpoints and rollout videos
remain in that unique directory. Inspect a failed run before choosing a new output
directory; there is no automatic restart or hidden continuation.
The evaluator CLI is checked before fitting. `--evaluator PATH` selects a compatible
prepared evaluator. If rollouts fail after training and prediction evaluation have
completed, `--resume-evaluation --evaluator PATH` verifies the original input and
checkpoint hashes, then runs only the matched rollout pair. It retains the original
failed `status.json` and writes separate recovery evidence; it never repeats training.

References checked 2026-10-18:
[BFL SO-101 LoRA guide](https://github.com/black-forest-labs/flux-action/blob/main/docs/so101-lora.md),
[LeRobot FLUX integration](https://huggingface.co/docs/lerobot/main/en/flux3).

## Phases

| Phase | Hours | Outcome |
|-------|-------|---------|
| 0. Contracts | 0–2 | interfaces, predicates, task grammar agreed; `pytest` green everywhere |
| 1. Bring-up | 2–6 | arms + cameras calibrated; FLUX inference measured; sim scene loads |
| 2. First skills | 6–12 | one FLUX skill on hardware; dual-arm task end-to-end in mock + sim |
| 3. Collaboration | 12–18 | physical WHILE (hold + insert); demo LoRA trained; per-arm checkpoint validation |
| 4. Long horizon | 18–24 | obstacle-clearing + long-horizon task with recovery on hardware |
| 5. Freeze + demo | 24–30 | ≥5 trials per demo, success rates, video, slides |

## Top risks

| Risk | Mitigation |
|------|-----------|
| FLUX not running by hour 6 | scripted waypoint backend; team never blocks on GPU |
| Not enough VRAM | Explicit shared-model fallback (serial inference); measure capacity before loading two instances |
| Out-of-the-box FLUX poor on our table | WS1 LoRA on 50–100 demos with a fixed eval protocol |
| Lid slips (position-only hold) | light lid, short holds, WHILE monitor cancels + retries |
| Original bimanual adaptation scope | Superseded: retain the existing 6-D action head and train task LoRAs |
| Perception false positives | cheap colour zones first, VLM verifier only for hard predicates, human check on trials |
