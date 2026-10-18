# Real SO101 inference

The current workflow runs LeRobot's official `Flux3Policy` and saved
pre/postprocessors on a GPU VM. Liminal sends explicit camera/joint histories over
SSH; Haller owns the local motors, camera streams and UI stop control.

The default profile uses native joint coordinates and HF Space normalization
statistics. Those statistics differ from the original released package. This is
an experimental inference configuration with released weights and no task adapter.
See the [first validation results](validation/single-arm-v1.md) before interpreting
an inference pass as a successful grasp.

## 1. Prepare the laptop and controller

From the repository root:

```bash
./scripts/setup_robot.sh
```

This creates `.venv-robot`, including the local `live` dependencies. It does not
connect motors. Haller is an external dependency: apply the versioned
[controller patch](../integrations/haller/README.md) and configure its scene and
wrist camera streams. Run Haller on `127.0.0.1:8000` with policy transport on port
8781. ROS is not required.

`configs/policy/official_haller_right.yaml` is the commissioned bench profile.
On that bench, software `right` is the physical arm on the left of the scene view,
with wrist camera A. Port IDs and camera paths are installation-specific. For
another rig, copy the profile to an ignored `*.local.yaml`, set the actual device
and VM details, and verify the physical camera/arm binding before marking
`camera_verified` and `units_verified` true. Reuse a measured calibration; do not
infer joint offsets from training quantiles.

## 2. Prepare the GPU worker

The recorded environment used official `huggingface/lerobot` revision
`e624f3f7f8411ec3a02635d06e79373341e5ef35`, Python 3.12, Torch 2.10/CUDA 12.8
and a matching NATTEN build. Install FLUX dependencies using the
[LeRobot instructions](https://huggingface.co/docs/lerobot/main/en/flux3).
The laptop's CPU hardware environment and GPU model environment are separate.

The worker expects a complete local SO101 package and its text/video encoders in
the offline cache. The validation base package revision is
`0713a7af6768562e618e81d534372c6795d41b2e`. Obtain model access and download the
package and its referenced encoders using the
[BFL setup guide](https://github.com/black-forest-labs/flux-action/blob/main/docs/setup.md).
Keep the released package intact.

Install this repository in the GPU environment (`pip install -e /path/to/liminal`) so the worker can import the shared `LeRobotEngine`. Deploy `scripts/flux_lerobot_worker.py` to the profile's `worker` path. The
commissioned path is `/liminal/scripts/flux_lerobot_worker.py`; both normalization
profiles now use this same source file. Example deployment for that bench:

```bash
gcloud compute scp --zone=europe-west1-b --project=motion-mvp \
  --tunnel-through-iap scripts/flux_lerobot_worker.py \
  prd-inf-vm-01:/liminal/scripts/flux_lerobot_worker.py
```

Prepare a separate HF-normalized package **on the VM**, once, with the GPU
environment activated and the preparation script copied there:

```bash
python scripts/prepare_hf_space_checkpoint.py \
  --source /liminal/models/so101 \
  --output /liminal/models/so101-hf-space-native-20261017
```

The script refuses an existing output directory. It links unchanged assets,
replaces all three quantile processor tensors together, and records provenance
from HF Space revision `d7da38e032da0e136d6de21c218dc616982a8cc9`. Retain the source
package because the prepared directory contains links to it. No downloaded Python
source is executed to obtain the statistics.

The current transport sources `/liminal/env.sh` and `/liminal/offline_env.sh`
before starting the configured Python executable. These are local environment
files for cache/dependency paths and credentials; they are not committed. Adapt
the transport setup for a different VM layout. Existing authenticated `gcloud`
and IAP SSH access are required; no public inference endpoint is opened.

## 3. Validate real observations

Keep Haller running and use:

```bash
./scripts/infer_robot.sh --verbose
# Another installation:
./scripts/infer_robot.sh --config configs/policy/right.local.yaml --verbose
```

This loads the model, checks the profile and processor provenance, and predicts
from real observations without commanding motors. Inspect the saved scene/wrist
views and any reported execution blocker.

## 4. Execute a supervised trial

Support the selected arm and select **MANUAL** for software **right** in Haller.
This enables torque and clears a previous STOP. Torque **Engage** alone does not
clear STOP. Then:

```bash
./scripts/infer_robot.sh --execute --engage --verbose \
  --prompt "Grasp the white cup."
```

`--engage` acknowledges that UI action; this runner never enables torque or
clears STOP automatically. The default trial deadline is 120 seconds. At the
deadline, `hold_on_duration: true` ends inference and leaves torque enabled at
the last committed target. The terminal stays open to monitor the arm and keep
Haller's policy connection alive; closing that connection releases torque.
The GPU worker exits during the hold. Ctrl+C, Haller STOP/E-stop, or a fault
still stops and releases torque, so support the arm before stopping it.
Set `hold_on_duration: false` to release torque at the deadline instead.
The prior elbow/wrist resistance remains unresolved.

The default official path executes the first 16 of 42 predictions at 30 Hz, then holds
while synchronous inference runs. It applies no delta rebasing, interpolation or
policy speed limiting. The default profile sets `gripper_clamping: true` to
clamp finite gripper predictions to 0–100%, matching LeRobot's gripper driver.
Every arm-joint violation still rejects the entire executed chunk, and nonfinite
predictions always stop execution.
Raw predictions, `execution_actions`, and `gripper_adjustments` are saved in
`plans.jsonl`; each correction also prints `[debug:gripper_clamp]` in verbose mode.
Committed targets, including the corrected gripper value, feed subsequent command history.
Set `gripper_clamping: false` for strict gripper validation. If Haller changes a valid target,
execution stops. Load, stall, stale-input and failed-acknowledgement checks remain.
These are local execution checks around the official policy API.

## Profiles and alternatives

| Entry point/profile | Model path | Coordinates/statistics |
|---|---|---|
| `infer_robot.sh` / `official_haller_right.yaml` | Official LeRobot, Haller owner | Native / saved HF Space statistics; default |
| `infer_robot_hf_space.sh` | Alias of the default launcher | Same default profile |
| `released_haller_right.yaml` | Official LeRobot, Haller owner | Original released processors |
| `infer_robot_official.sh` / `official_right.yaml` | Official LeRobot, direct SO101 driver | Original released processors; alternative owner |
| `infer_robot_haller.sh` / `live_right.yaml` | BFL standalone API | Historical comparison; see [legacy guide](live-inference.md) |

The optional `--legacy-shoulder-elbow` mapping is a historical comparison:
`model shoulder = 90 - robot shoulder`, `model elbow = robot elbow + 90`.
It is accepted only with an identity released-statistics profile and is refused
with HF normalization. It changes no motor calibration. A bounds pass does not
validate its physical correspondence or task performance.

The direct-driver alternative requires Haller stopped, serial/cameras released
and torque already off. It reuses `haller_right` calibration; observation-only is
the default. `--execute` engages after validation. It has no rollout deadline and
the Haller UI stop control is unavailable. It uses our SSH runner, not the stock
single-process `lerobot-rollout` CLI. Use one motor owner at a time.

## Logs and interpretation

Runs are saved under `~/.local/share/liminal/inference/<run-id>/`:

| Artifact | Contents |
|---|---|
| `report.json` | Integration, checkpoint, profile, settings, status and stop reason |
| `input-*.json` | Exact eight-frame numeric and image conditioning |
| `plans.jsonl` | Model predictions, robot-coordinate targets and numeric history |
| `submitted-actions.jsonl` | Requested/submitted commands |
| `events.jsonl` | Acknowledgements, telemetry, timing and faults |
| `worker.log` | LeRobot logs and GPU-process stderr |

Direct-driver runs instead save outputs as `plan-*.json`. Early historical runs
lack exact image windows. Protect/share raw camera logs deliberately; compact
results belong in the repo, while weights, datasets and full logs stay outside Git.

Rebuild the compact trial index without opening the robot:

```bash
python scripts/summarize_real_runs.py \
  --source ~/.local/share/liminal/inference \
  --output outputs/real-robot-results.json
```

A completed duration, finite prediction or acknowledged command does not establish
manipulation success. This first validation has no independent real-task success
labels. See [results and limitations](validation/single-arm-v1.md).

## Red-pen LoRA deployment

`configs/policy/pen_lora_right.yaml` selects the merged red-pen adapter at Hub
revision `492e06e50f94fdb26286bf88b90e8faf657c4bb2` (EMA step 4,000). It uses
that adapter's saved B84 state/action quantiles, native joint coordinates,
guidance 1.0 for both modalities and 32 executed actions per 42-step chunk.
The commissioned software `right` arm is the physical left/task arm, with
wrist A rotated 180 degrees. The prompt is exactly
`put the red pen in its slot in the cardboard box`.

The GPU package is `/liminal/models/so101-haller-pen-lora-492e06e5`, separate
from the released and HF Space packages. `scripts/prepare_lora_checkpoint.py`
merges the downloaded adapter on CPU, copies its processors and records the
adapter revision, checksum and actual tensor quantiles. The dedicated remote
worker is `/liminal/scripts/flux_lerobot_worker_pen.py`. Both ends validate
`dataset_native` normalization and require identity coordinates.

With Haller running, first predict without motion:

```bash
./scripts/infer_robot.sh --config configs/policy/pen_lora_right.yaml --verbose
```

For a trial, arrange the training scene and start from the demonstrated home
pose `[0, 0, 0, 0, 0, 100]` (degrees; gripper percent, 100 open). Select MANUAL
for software right in Haller, then run:

```bash
./scripts/infer_robot.sh --config configs/policy/pen_lora_right.yaml \
  --execute --engage --verbose
```

The same 120-second deadline, hold behavior, Haller ownership and execution
checks described above apply. Model loading or synthetic prediction checks do
not establish success on the red-pen task.

### Try the Hugging Face normalization with the pen LoRA

`pen_lora_hf_space_right.yaml` selects the same merged LoRA weights with the
state/action quantiles from Space commit
`09edebb006d793cb215ce8c0fccfc9eacd71dfc6`, replacing B84 in all three saved
processors. This is an experimental change from the normalization used during
fine-tuning. Other inference and execution settings match `pen_lora_right.yaml`.
The separate GPU package is
`/liminal/models/so101-haller-pen-lora-492e06e5-hf-fix-09edebb`.

```bash
./scripts/infer_robot.sh --config configs/policy/pen_lora_hf_space_right.yaml \
  --execute --engage --verbose
```

Omit `--execute --engage` for observation-only inference. The original
`pen_lora_right.yaml` still selects B84. A concurrent GPU training job may leave
insufficient memory to load inference; finish or explicitly stop that job before
running a trial.

## Optional action-level Claude hybrid controller

The previous task-tree LLM recovery prototype has been removed. The live runners
now support the FLUX report's action-level decision cycle, disabled by default.
It uses Claude instead of the report's Astra and the SO-101 instead of the Panda;
it is not a reproduction of their measured success rate.

Sources: [FLUX 3 Action, Hybrid policies](https://bfl.ai/models/flux-3-action#hybrid-policies),
[reference RoboLab action contract](https://github.com/anonymous-report-421/GPT-as-Policy/blob/main/hybrid_rollout/robolab/skill/context/eef_control.md),
[reference outcome/intent gate](https://github.com/anonymous-report-421/GPT-as-Policy/blob/main/hybrid_rollout/robolab/skill/gate_prompt.md).

At every FLUX chunk boundary, Claude receives the task, current scene and wrist
JPEGs, measured joints, full robot-only FK proposal, prior decision/execution history,
and previous pre-execution RGB for outcome comparison. It returns one structured
`student`, `edit`, `eef`, or `stop` decision:

- `student`: execute an unchanged prefix of 1–15 predicted absolute joint targets.
  The unused suffix is discarded. The normal profile's smaller horizon still wins.
- `edit`: apply root-frame translation and rotation-vector deltas to the first
  1–5 FK poses, scaling the offsets by `(i+1)/steps`. Rotation left-multiplies the
  proposal orientation; the gripper override applies to every selected pose.
- `eef`: track one fixed root-frame pose for 1–5 controls. Position is meters and
  quaternion order is wxyz. Both correction modes recompute a local DLS joint
  update from measured joints after every actual command acknowledgment.
- `stop`: end the rollout through normal cleanup. The report explicitly says task
  success is unverified. Claude's progress assessment never writes success facts.

The correction gate requires observed execution failure or misaligned next intent,
not uncertainty alone. Between all segments, including corrections, FLUX predicts
again. Its original task instruction and real committed-command history are retained.
No named recovery skills, scripted task restart, arbitrary Python, or LLM joint
arrays are accepted. Every executed target still goes through `SafetySupervisor.send()`.

The local IK uses the checked-in `sim/single_arm/so101.xml` body/joint chain and
`gripperframe` site without loading meshes or a simulator. It uses damped least
squares with damping 0.05, translation/rotation error caps of 0.02 m / 0.1 rad,
and default joint increments of 0.05 rad. Corrections are limited to 0.05 m / 0.35 rad;
calibrated travel and gripper limits remain enforced. The SO-101's five arm joints
cannot realize arbitrary six-dimensional poses. Local IK is not collision planning,
and an acknowledged command is not evidence of arrival or a successful grasp.
The existing hardware rate remains 30 Hz; the reference Panda experiment used 15 Hz.

Install `pip install -e '.[hybrid]'`, set `ANTHROPIC_API_KEY` in the environment,
and copy `configs/hybrid/claude_so101.yaml` to a local configuration. Set
`enabled: true` to test the multimodal gate without executing motion:

```bash
scripts/infer_robot.sh --config configs/policy/pen_lora_right.yaml \
  --hybrid-config /path/to/hybrid.yaml
```

This observation-only run uses the actual camera/GPU/API paths but sends no motor
commands. The default file has both `enabled: false` and `kinematics_verified: false`.
Before physical execution, verify the root frame, TCP site, joint signs/offsets,
and gripper endpoints against the calibrated robot, then set
`kinematics_verified: true`. The formula is
`q_model_deg = (q_robot_deg - joint_offsets_deg) / joint_signs`.
The supplied native identity mapping is a VERIFY assumption, not hardware calibration.
The frame description is sent to Claude; describe verified camera/root relationships
there when available. No camera extrinsics are invented by the implementation.

For physical execution, use the existing explicit engagement workflow and append
`--hybrid-config /path/to/hybrid.yaml` to the live command with `--execute --engage`.
The direct-driver `scripts/infer_robot_official.sh` also accepts `--hybrid-config`;
its existing Haller-off and `--execute` requirements are unchanged. The Haller path
requires the official absolute-joint worker, as used by the shipped pen profiles.

API calls are asynchronous with no SDK retries and a finite per-decision timeout.
Telemetry and stop monitoring continue while reasoning. A timeout, malformed tool
call, stale request ID, non-finite value, ungrounded takeover, exceeded bound, or
robot drift during reasoning stops execution rather than silently delegating back
to a bad VLA. The unchanged student prefix is checked against calibrated travel
before any command; gripper clamping from the ordinary runner does not rewrite the
hybrid student proposal. The LLM may choose a valid shorter prefix or a correction.

Haller logs complete `hybrid_request`, `hybrid_decision`, `hybrid_executed` and
`hybrid_error` records in `hybrid.jsonl`; the direct runner uses `events.jsonl`.
Requests include RGB and numerical context, decisions include provider token usage,
and execution records contain actual committed commands and final measured joints.
These logs can be large and include camera images. The task-wide rollout deadline
still applies; it can expire during reasoning. No real API or motor tests are run
by pytest; lifecycle tests use fake providers, workers, and transport with real FK/IK.

Independent mathematical validation: the FK implementation matched MuJoCo 3.16.0
on 100 seeded poses using the same MJCF with visual assets removed (maximum
position error 2.22e-16 m; rotation-matrix element error 1.22e-15). The optional
`test_fk_matches_mujoco_robot_only_oracle_when_available` reproduces that check.
This validates model math, not the model-to-physical-robot calibration.
