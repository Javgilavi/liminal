# AGENT.md

Instructions for anyone, human or AI coding agent, changing this repo during the
hackathon. Read this before your first commit.

## What this project is

A dual-arm orchestration layer on top of a learned VLA (FLUX 3 Action). **FLUX is the
manipulation backend, not the coordinator.** It executes bounded, text-prompted skills
on one arm. The behavior tree decides what runs, when, on which arm, and what must stay
true while it runs. Perception decides whether it worked.

## Commands

```bash
pip install -e ".[dev]"          # core + pytest + ruff
pytest                           # must pass before every push
ruff check src tests scripts     # must pass before every push
liminal validate configs/tasks/*.yaml
liminal run configs/tasks/lid_hold_insert.yaml        # mock arms, no hardware
```

Tests run on mock arms only, with no GPU, no network and no serial ports. Keep it that way.

## Hard rules

1. **Safety first.** Every motor command goes through `SafetySupervisor.send()`
   (limits, max step, e-stop). Never call `arm.send_action()` directly from policy or
   orchestration code. Primitives and the hold controller live in `hardware/` and
   `orchestration/skills.py`, and those are the only exceptions.
2. **No hardware in tests or imports.** Heavy or optional deps (`lerobot`, `torch`,
   `mujoco`, `cv2`, `anthropic`, `google-genai`, `fastapi`) are imported *inside*
   functions, never at module top level. `pip install -e .` plus `pytest` must work on a laptop.
3. **Never trust a skill's own success claim.** A skill counts as done only when its
   `success` predicate holds in `WorldState`. `SkillNode` and `run_task` re-check it. Don't
   bypass that.
4. **LLMs propose, the tree enforces.** Planners output task-file YAML that goes
   through `parse_task`. No LLM output ever reaches motors or marks a goal complete.
5. **One arm, one owner.** Arm ownership is claimed by nodes (`safety.claim`). An
   `ArmConflict` is a planning bug. Fix the tree, don't catch and ignore it.
6. **Per-arm policy state.** FLUX observation history is kept per arm. Never mix the
   two arms' queues. Inference is single-flight behind one lock.
7. **Interfaces are contracts.** `core/types.py`, `core/interfaces.py` and the task-file
   grammar in `orchestration/task_spec.py` are shared by all workstreams. Change them
   only with a heads-up to the team, and update every implementation and test in the
   same commit.
8. **Mark unverified facts.** Numbers or APIs taken from docs you haven't run against
   get a `VERIFY` comment (see `configs/policy/flux3_so101.yaml`). Remove the marker
   once confirmed on real hardware or code.

## Ownership (see TASK.md)

| Path | Workstream |
|------|-----------|
| `policy/`, `data/`, `scripts/record_demos.sh`, `scripts/finetune_lora.sh`, `scripts/flux_smoke_test.py` | WS1 single-arm + LoRA |
| `bimanual/`, `scripts/plan_obstacle.py` | WS2 bimanual obstacle |
| `orchestration/`, `perception/`, `configs/tasks/`, `configs/skills.yaml` | WS3 long-horizon |
| `sim/`, `configs/embodiment/`, `configs/robot/mujoco_*.yaml` | WS4 sim + XLeRobot |
| `hardware/`, `safety/`, `ui/`, `configs/robot/so101_*.yaml`, `scripts/record_waypoints.py` | WS5 hardware + UI |
| `core/`, `app.py`, `AGENT.md`, `TASK.md`, `README.md` | shared, so coordinate first |

Touching another workstream's path is fine for small fixes. Tell the owner.

## How to extend

- **New skill:** add it to `configs/skills.yaml` (so planners can use it), give it a
  verifiable `success` predicate, add waypoints in `data/waypoints/<arm>.yaml` for the
  scripted fallback, and use it in a task file. Hardware-only skills go into
  `PrimitiveBackend` instead.
- **New predicate:** add a detector rule in `perception/detectors.py` (cheap colour or
  zone check first, VLM only if needed) and list it in `configs/skills.yaml`.
- **New task:** write `configs/tasks/<name>.yaml`. It is automatically covered by
  `tests/test_tasks.py::test_shipped_tasks_succeed_on_mock`.
- **New BT node:** subclass `bt.Node`, implement `_run`, add a `_<kind>` method in
  `task_spec._Compiler`, document it in the grammar docstring, and add a test in `tests/test_bt.py`.
- **New arm/sim backend:** implement the `Arm` protocol and register the type in `app.build_arms`.
- **New planner:** return a `TaskSpec` built with `parse_task`. Feed compile errors back
  to the model (see `LLMPlanner`).

## Style

- Python ≥3.10, type hints, dataclasses, `async` for anything that waits on the world.
- Match the surrounding code. Comments explain *why*, not *what*.
- Line length 120 (ruff). Keep modules small and dependency-light.
- Commits: `ws<N>: <imperative summary>` (e.g. `ws3: add Fallback recovery to lid task`).

## Hackathon decision points

- **Hour 6:** if FLUX inference is not running, everyone else integrates against
  `--backend scripted` / `--backend mock`. Do not block on the GPU.
- **Hour 18:** if simultaneous WHILE-holding is not reliable on hardware, freeze
  features and demo sequential AFTER dependencies reliably.
- **Hour 24:** feature freeze. Only reliability fixes, trials and demo prep after this.
