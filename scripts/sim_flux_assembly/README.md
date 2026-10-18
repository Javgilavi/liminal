# Gear and collar assembly skills

Scope: single-arm training and isolated-skill evaluation only. This experiment
is not the dual-arm runtime harness. The current architecture runs independent
per-arm FLUX policies under the [real orchestrator](../../docs/showcase-runtime.md);
the former simulation assembly sequence driver has been removed.

This follow-on experiment teaches four prompt-conditioned skills on one SO-101:
pick a blue gear, seat it on a shaft, pick a brass spacer collar, and seat the
collar over the gear. One shared task LoRA learns the four prompts. A small
isolated-skill evaluator checks each physical result from a recorded start state.
Teacher demonstration success and learned inference success are reported separately.

This prototype requires the prepared single-arm scene, MuJoCo, NumPy and the
external NIST CAD archive expected at `<root>/assets/source/nist-stl.zip`.
Set `LIMINAL_SIM_ROOT` to the prepared scene. Run each script with `--help` for
its required arguments. The active remote experiment directory is
`/liminal/experiments/sim-assembly-20261017`; `run-pipeline.sh` records its commands.

## Geometry and data

`assets.py` converts NIST's ICRA 2022 task-board STL parts from millimetres and
their full-board poses to centred metre-scale meshes. The gear and collar are
scaled 1.6 times for the gripper; the shaft is resized to a 50 mm height and
12 mm maximum diameter. These are CAD-derived assets, not a scan of real parts.
Source filenames, hashes, dimensions and simplifications are in
`assets/provenance.json`. The functional through-holes use compound annular
collision pieces; gear teeth remain detailed visual geometry. The fixed base
and part supports are custom. The task is a clearance-fit stack, without
threading, snap fits or a force-fit model.

The collector targets 32 complete successful training assemblies and eight
validation assemblies with disjoint seeds. Each contributes four atomic episodes,
for 160 episodes. Failed teacher attempts stay outside the training dataset.
The teacher uses IK and privileged pose feedback only during generation, never
welds, attachments, teleports or model inference. Physics is recorded first;
`render.py` then renders the identical saved poses with the identical cameras.

Native joint degrees and gripper percent are used throughout. There is no
shoulder/elbow adapter. HF simulator normalization is saved in the prepared base
before training and retained by the adapters. Commands, dataset timestamps,
policy rate and video are all 20 Hz: 25 physics steps of 2 ms per command.

`validate.py` checks all recorded states against MuJoCo poses, bounded issued
commands, exported label samples, timing, camera roles and saved HF statistics.
It replays every accepted final assembly and rejects an intentionally misplaced
collar. Only the two camera streams, measured joints, absolute issued commands
and skill prompt are exported to the official LeRobot v3 dataset.

## Training and execution

The initial training budget is 3,000 microbatches / 750 optimizer updates with
rank/alpha 32, BF16, activation checkpointing, effective batch eight and EMA
0.999. It uses the official LeRobot trainer and adapter loader from the same
recorded checkout as the prior experiment. This budget is shorter than BFL's
10,000-microbatch example; 20 Hz is an explicit departure from its 30 Hz recipe.
The old pick-and-place adapter is not reused.
Checkpoints are saved every 250 microbatches. After the user stopped the first
attempt at step 520 before its first scheduled save, the run restarted from the
frozen base with the same validated dataset; the interrupted log is retained.

The simulation assembly sequence driver has been removed. Collection, training,
replay audits and isolated-skill diagnostics remain available. `evaluate.py` runs
each selected atomic skill from a recorded teacher start, independently checks its
success predicate, and does not chain the skills into an assembly task. Historical
full-assembly reports can still be read with `summarize.py`.

```bash
python scripts/sim_flux_assembly/evaluate.py \
  --root "$ASSEMBLY_ROOT" --checkpoint "$ASSEMBLY_CHECKPOINT" \
  --output "$ASSEMBLY_ROOT/atomic-new" --episodes 5 --skills pick_gear
```

For the real skill-runtime, checkpoint registry and live perception, see
[showcase runtime](../../docs/showcase-runtime.md).

## Preview artifacts

`teacher-preview.mp4` shows one verified teacher assembly. It is not learned
inference. `assembly-closeup.png` and `gear-collar-assembly.glb` expose the assembled
CAD geometry. `export_glb.py` creates the portable model without ML dependencies.

NIST source: https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/robotic-grasping-0

Training references:

- https://github.com/black-forest-labs/flux-action/blob/main/docs/so101-lora.md
- https://huggingface.co/docs/lerobot/main/en/flux3

See [the first validation](../../docs/validation/single-arm-v1.md) and
[simulation guide](../../docs/simulation-validation.md) for the supported experiment.
