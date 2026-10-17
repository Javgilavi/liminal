# liminal

**Long-horizon, dual-arm collaboration on SO-101 / XLeRobot, using FLUX 3 Action for manipulation.**

One arm changes or holds something in the scene so the other arm can do what it
otherwise couldn't. FLUX 3 Action handles the *learned manipulation*: bounded,
text-prompted skills on one arm. Everything above that is this repo: deciding
*why* a skill is needed, *when* the other arm is ready, *which conditions must stay
true*, and *whether the goal really happened*.

```
            goal ("put the block in the box")
                         │
       ┌─────────────────▼──────────────────┐
       │ Planner   template | LLM | Gemini ER│  proposes a task tree (YAML)
       └─────────────────┬──────────────────┘
       ┌─────────────────▼──────────────────┐   ┌──────────────────────┐
       │ Behavior tree executor             │◄──┤ World state + memory │◄── perception
       │ AFTER · WHILE · UNTIL · retry      │   │ predicates, event log│    (cameras, VLM)
       └───────┬─────────────────┬──────────┘   └──────────────────────┘
       ┌───────▼─────────────────▼──────────┐
       │ Safety supervisor                  │  arm ownership, zones, limits, e-stop
       └───────┬─────────────────┬──────────┘
        ┌──────▼──────┐   ┌──────▼──────┐
        │ left arm    │   │ right arm   │   skills → FLUX (single-flight, shared model)
        │ hold / FLUX │   │ hold / FLUX │          → scripted fallback / mock
        └─────────────┘   └─────────────┘          → primitives (hold, release, home)
         SO-101 (LeRobot) · MuJoCo · mock
```

Full design: [docs/architecture.md](docs/architecture.md). Who does what, and by when:
[TASK.md](TASK.md). Rules for humans and coding agents: [AGENT.md](AGENT.md).

## Workstreams

| # | Workstream | Code | Brief |
|---|-----------|------|-------|
| WS1 | Single-arm FLUX baseline + 50–100 demos + task LoRA | `policy/`, `data/`, `scripts/record_demos.sh` | [01](docs/research/01_single_arm_baseline.md) |
| WS2 | Bimanual: move obstacles away to reach a target | `bimanual/` | [02](docs/research/02_bimanual_obstacle.md) |
| WS3 | Long-horizon decomposition + orchestration (BT / FSM / LLM / Gemini ER / VLM) | `orchestration/`, `perception/` | [03](docs/research/03_long_horizon_orchestration.md) |
| WS4 | Simulation + bimanual XLeRobot embodiment adaptation | `sim/`, `configs/embodiment/` | [04](docs/research/04_xlerobot_sim.md) |
| WS5 | Hardware setup, safety, operator UI / triggering | `hardware/`, `safety/`, `ui/` | [05](docs/research/05_hardware_ui.md) |

## Quick start (no hardware needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # 33 tests, all on mock arms

# Run the cooperative-holding demo on mock arms
liminal run configs/tasks/lid_hold_insert.yaml
# Check every task file compiles
liminal validate configs/tasks/*.yaml
# Goal -> task tree (template planner; --planner claude|gemini needs `.[llm]` + API key)
liminal plan "hold the lid and insert the block"
# Bimanual obstacle planner (WS2)
python scripts/plan_obstacle.py --strategy hold_away
```

## On real hardware

```bash
pip install -e ".[hardware,perception,ui]"
lerobot-find-port                          # then edit configs/robot/so101_*.yaml
lerobot-calibrate ...                      # once per arm, same id as in the config

# Without FLUX (scripted waypoints, the hour-6 fallback)
liminal run configs/tasks/single_pick_place.yaml --robot configs/robot/so101_single.yaml --backend scripted

# With FLUX 3 Action (after WS1 wires FluxPolicyBackend.load/predict)
python scripts/flux_smoke_test.py
liminal run configs/tasks/lid_hold_insert.yaml --robot configs/robot/so101_dual.yaml \
    --backend flux --policy configs/policy/flux3_so101.yaml

# Operator API: POST /run, POST /estop, GET /state
liminal serve --robot configs/robot/so101_dual.yaml --backend flux --policy configs/policy/flux3_so101.yaml
```

Every run writes a JSONL event log under `runs/` (facts, node start/end, retries,
WHILE violations). Use it to count success rates over trials.

## Task files

Planners (hand-written, LLM, the bimanual obstacle planner) all emit the same
grammar, and one compiler validates it:

```yaml
tree:
  sequence:                                  # AFTER
    - skill: {arm: left, name: open_lid, prompt: "Lift the lid open", success: lid_open}
    - while:                                 # WHILE: left keeps holding...
        arm: left
        maintain: lid_open                   # ...this must stay true, or the body is cancelled
        until: object_inserted               # UNTIL
        do:
          skill: {arm: right, name: pick_place, prompt: "Place the block inside the open box",
                  success: object_inserted}
    - parallel:
        - {skill: {arm: left, name: release}}
        - {skill: {arm: right, name: home}}
    - check: object_inserted
```

Grammar reference: `src/liminal/orchestration/task_spec.py`. Examples: `configs/tasks/`.

## Repo layout

```
src/liminal/
  core/           types + interface contracts (Arm, Camera, SkillBackend, PredicateDetector)
  orchestration/  behavior tree, task-file compiler, executor, planners (template / LLM)
  perception/     world state + event log, colour-zone detector, VLM verifier
  safety/         arm ownership, zones, joint limits, e-stop
  policy/         FLUX 3 Action backend, scripted fallback, mock
  hardware/       SO-101 via LeRobot (with position hold), cameras, mock arm
  sim/            MuJoCo arms/cameras, embodiment action-space mapping (XLeRobot)
  bimanual/       obstacle-clearing role assignment and plan generation
  data/           demo-collection variation protocol
  ui/             CLI and HTTP operator API
configs/          robot, policy, embodiment, skill catalog, task trees
scripts/          demo recording, LoRA, FLUX smoke test, waypoint recording, obstacle planner
docs/             architecture and per-workstream research briefs
```

## Status and caveats

- The orchestration stack (BT, WHILE/UNTIL, safety, world state, planners, mock and
  scripted backends) runs and is tested.
- **FLUX 3 Action loading is a stub.** `FluxPolicyBackend.load/predict` raise
  `NotImplementedError` until WS1 wires them against the official BFL docs. All FLUX
  numbers in `configs/policy/flux3_so101.yaml` (chunk 42 / execute 32 / 30 Hz /
  8-frame history / ~32 GB BF16) come from planning notes and are marked `VERIFY`.
- LeRobot imports and CLI flags change between releases. They are isolated in
  `hardware/so101.py` and `scripts/record_demos.sh`.
- SO-101 holding is **position** holding, not torque control. Keep loads light.
