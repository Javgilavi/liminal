# WS4: Simulation and bimanual XLeRobot embodiment adaptation

**Historical research brief — embodiment fine-tuning superseded.** The initial plan was to fine-tune
a unified bimanual XLeRobot embodiment (12-D actions). That plan was superseded by task LoRA fine-
tuning of the existing 6-D SO-101 policy, with two independent FLUX instances, one per arm,
coordinated by the real skill-runtime harness (behavior tree, checkpoint registry, perception and
safety).

The original alternatives and proposed milestones below are retained as history,
not an active training plan. Current simulation work evaluates single-arm LoRAs;
there is no simulation assembly orchestrator in the runtime. See [PLAN.md](../../PLAN.md).

## Two separate goals

1. **Sim for orchestration** (needed by H12): same `Arm` API, physics-backed task tests,
   failure injection. Image realism doesn't matter.
2. **Embodiment adaptation** (research, H14+): get a FLUX-based policy driving XLeRobot's
   two arms. Here data and action-space choices matter.

## Sim setup

- `sim/mujoco_arm.py`: one `MujocoWorld` and one `MujocoArm` per actuator prefix
  (`left_`, `right_`). `MujocoCamera` renders named cameras.
- Candidate scenes (check licence and fidelity before adopting): a community dual
  SO-101 MuJoCo environment and a MuJoCo LeRobot SO-101 adapter (both from the planning
  notes, unverified), XLeRobot's own sim assets, and SO-101 MJCF from LeRobot / MuJoCo Menagerie.
- Units: MJCF is radians; LeRobot calibrated units differ. Do the conversion in config,
  not in the orchestrator.
- Sim predicates: write a `PredicateDetector` that reads object poses from `MjData`
  (ground truth), e.g. `block_in_bowl` = block centre inside the bowl volume.

## Embodiment: action-space options (`sim/embodiment.py`, decision in [PLAN.md](../../PLAN.md))

| Mode | How | Data needed | Pros | Cons |
|------|-----|-------------|------|------|
| `per_arm` | two sessions of the single-arm checkpoint, each its own 6-D slice + wrist cam | single-arm data per arm | works with the released checkpoint; orchestrator coordinates | no learned inter-arm coordination |
| `bimanual` | fine-tune on concatenated 12-D (+ head/base) actions; skills target the `both` group | bimanual demos | learned coordination (handovers, bimanual carry) | new action head/normalisation; much more data |

Original recommendation (superseded): ship `per_arm` (option A, matches the hackathon architecture), research
`bimanual` (option B). Code path for B: `configs/policy/flux3_dual_bimanual.yaml` +
`configs/embodiment/so101_dual.yaml` + `configs/tasks/lid_hold_insert_bimanual.yaml`.

## Datasets to survey (record: action space, cameras, fps, licence, size)

- Open XLeRobot / bimanual SO-100/101 datasets on the Hugging Face hub (LeRobot format).
- UMI (Universal Manipulation Interface, Chi et al., 2024) style egocentric gripper data.
  Useful for wrist-camera visual priors; the action mapping to SO-101 joints is non-trivial.
- Large cross-embodiment sets (Open X-Embodiment, DROID) for reference on mixing embodiments.

## First embodiment fine-tune (original plan, not pursued)

1. Convert chosen datasets to the `xlerobot.yaml` layout with `Embodiment.split/merge`.
2. Smallest run that produces a curve (few k steps), eval in sim on one bimanual task.
3. Go/no-go note: data volume needed, compute, expected gain vs `per_arm`.

## References (verify before citing)

- XLeRobot (open-source low-cost dual-arm mobile robot built on SO-101-style arms).
- LeRobot (Hugging Face): dataset format, SO-101 support.
- MuJoCo / MuJoCo Menagerie.
- UMI (Chi et al., RSS 2024); Open X-Embodiment (2023); DROID (2024).
