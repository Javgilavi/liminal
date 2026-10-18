# Oskrt red-pen checkpoint

The runtime uses [Oskrt/flux-3-action-so101-haller-pen-lora](https://huggingface.co/Oskrt/flux-3-action-so101-haller-pen-lora)
at immutable revision `e85fc5831ae9d28a27ef175aa1c0196b6bc5ec16`, raw step 10,000.
The adapter weight SHA256 is
`997284e4e0335baccf3849cf6824611a2f8de1feaf70373917d1c49ceb98d76e`.
This is one combined pick-and-place policy, not separate grasp/place checkpoints.
Its exact task prompt is `put the red pen in its slot in the cardboard box`.

`configs/policy/pen_place_registry.yaml` maps `put_red_pen_in_slot` to this bundle.
`configs/tasks/pen_place.yaml` requires the independently verified
`red_pen_in_slot` predicate; the model cannot report task success itself.
`configs/perception/pen_place.yaml` defines visible seating and gripper release.
No automatic retries are configured because the adapter's recovery behavior has
not been established.

## Preserve the training contract

- The base is SO101 revision `0713a7af6768562e618e81d534372c6795d41b2e`.
- Use the adapter's three saved B84 normalization states, never the stock or HF
  Space quantiles. The new `dataset_native` profile verifies their provenance.
- Model outputs represent joint deltas with absolute gripper position. The saved
  postprocessor integrates them into absolute commands before the supervisor.
- Use eight history frames, 42 predictions and 30 Hz. Execute only the first
  16 actions per prediction (about 0.53 seconds), then inspect and replan. This
  runtime override leaves the checkpoint's saved 32-step execution setting unchanged.
  Staging explicitly sets both guidance scales to 1, as documented by the model
  card; the uploaded config otherwise defaults to video guidance 3.
- Native joint degrees and 0–100 gripper percent; 100 means open. Training starts
  were generally near home with an open gripper. Camera roles are head/scene and
  task-arm wrist; the training wrist image was rotated 180 degrees.
- The model card calls the physical task arm LEFT while its dataset columns are
  `right_*`. The registry now uses physical `left`, with `top` and `left_wrist` cameras.
  Verify the physical mapping in the robot configuration before any engagement.
  This document and the staging scripts do not engage motors.

## Prepare a compatible bundle

In the existing LeRobot/PEFT GPU environment, with the licensed base already
available locally, run from this branch's repository root:

```bash
python scripts/prepare_pen_checkpoint.py \
  --base /liminal/models/so101 \
  --adapter /liminal/models/pen-adapter-e85fc583 --download \
  --output /liminal/models/so101-haller-pen-raw10000-e85fc583-left \
  --registry /liminal/models/pen_place_resolved.yaml
python scripts/check_checkpoint_registry.py /liminal/models/pen_place_resolved.yaml
```

The script pins the Hub revision, verifies adapter and base weight hashes, checks
all saved normalization states, merges on CPU, and copies the adapter's processors.
It records adapter/base provenance and writes a resolved registry with the complete
bundle digest. Existing output directories are refused. The checked-in profile
pins the bundle staged on `prd-inf-vm-01`; regenerate a resolved registry when
staging to another environment, since serialized paths can change the digest.

## Motor-free validation

```bash
python scripts/flux_smoke_test.py \
  --policy /liminal/models/pen_place_resolved.yaml \
  --skill put_red_pen_in_slot \
  --prompt 'put the red pen in its slot in the cardboard box' \
  --state 0 0 0 0 0 100 --iters 1 --gpu-memory-fraction 0.4
```

This checks real model loading and finite output shape using blank camera frames.
It does not prove grasping, placement, visual verification or MuJoCo transfer. The
current branch's separate MuJoCo assembly harness is not wired to this pen task.
The registry loads on the orchestration host; it does not automatically route
predictions through the older live runner's SSH transport.

After evaluating real observations and confirming the robot mapping, the task can
be selected with `liminal run configs/tasks/pen_place.yaml --backend flux
--policy /liminal/models/pen_place_resolved.yaml --perception
configs/perception/pen_place.yaml --robot /path/to/calibrated_robot.yaml`.
Hardware engagement remains explicit via the normal operator workflow.

## Recorded validation

The pinned raw-10,000 adapter was downloaded and merged on `prd-inf-vm-01`.
Both source weight hashes passed, the merged bundle passed registry and saved
processor validation, and the real GPU engine returned a finite 42-by-6 action
chunk in approximately 601 ms for its first prediction. Inputs were blank images
and the open-gripper home vector; no motor commands were sent. This is loading and
inference evidence, not a physical pen-placement trial or a real checkpoint-swap
trial. See [the recorded identities and result](validation/pen-checkpoint-smoke.json).

## Left-arm deployment

The staged `...e85fc583-left` bundle links the same validated weights and processors,
with a separate provenance file allowing `left`. Its digest is
`43e4c9c772d1a475ecf2b964d8f8d5504e763f86e46ba9538fddfe0e2b55f858`.
The original bundle and recorded GPU smoke evidence retain their historical `right`
deployment label. This metadata remapping is not a new physical grasp validation.
