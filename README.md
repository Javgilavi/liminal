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

Strategy and key decisions: [PLAN.md](PLAN.md). Full design: [docs/architecture.md](docs/architecture.md).
Who does what, and by when: [TASK.md](TASK.md). Rules for humans and coding agents:
[AGENT.md](AGENT.md) (`CLAUDE.md` is an identical copy for Claude Code).

## Two ways to drive two arms with FLUX

| Option | What | Skills target | Policy config |
|--------|------|---------------|---------------|
| **A, per-arm** (default) | one embodied FLUX policy per arm (A1: shared model; A2: one instance per arm) | `left`, `right` | `configs/policy/flux3_so101.yaml` |
| **B, bimanual** | one new model fine-tuned on both arms together (12-D actions) | `both` | `configs/policy/flux3_dual_bimanual.yaml` |

Same executor, safety and task grammar for both. Comparison and decision: [PLAN.md](PLAN.md).

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
uv sync --extra dev
uv run pytest                            # all tests run on mock arms

# Run the cooperative-holding demo on mock arms
uv run liminal run configs/tasks/lid_hold_insert.yaml
# Check every task file compiles
uv run liminal validate configs/tasks/*.yaml
# Goal -> task tree (template planner; --planner claude|gemini needs `.[llm]` + API key)
uv run liminal plan "hold the lid and insert the block"
# Bimanual obstacle planner (WS2)
uv run python scripts/plan_obstacle.py --strategy hold_away
```

`pyproject.toml` declares direct dependencies and `uv.lock` pins the resolved
environment. The committed `.python-version` selects Python 3.12. Use `uv lock
--check` in CI or before deployment to detect dependency changes that were not
locked.

## Qwen scene inspection

The real VLM adapter is optional and loads its checkpoint lazily, once per process.
Check the device in `configs/perception/qwen3_vl_4b.yaml`, then inspect one synchronized
set of recorded views before enabling any robot motion:

```bash
uv sync --extra dev --extra vlm-qwen
uv run liminal inspect --perception configs/perception/qwen3_vl_4b.yaml \
  --image top=path/to/top.jpg \
  --image left_wrist=path/to/left_wrist.jpg \
  --image right_wrist=path/to/right_wrist.jpg
```

The shipped configuration uses bitsandbytes NF4 quantization with bfloat16
computation. Set `quantization: none` to measure the unquantized BF16 baseline.

The command prints schema-validated predicates plus load, preprocessing,
time-to-first-token, total-latency, and GPU-memory measurements. Pass the same
configuration file to `liminal run` using its `--perception` option.

Before robot deployment, label synchronized camera views using the manifest format
in `data/perception_eval/labels.example.yaml`, then run repeated offline evaluation:

```bash
uv run liminal evaluate-perception data/perception_eval/labels.yaml \
  --perception configs/perception/qwen3_vl_4b.yaml \
  --repeats 3
```

The JSON report retains raw model responses and parsed evidence, and summarizes
accuracy, unknowns, false-positive transitions, repeat agreement, latency, and
Qwen preprocessing/token-generation/GPU metrics per call. Treat any false-positive
transition as a deployment blocker even if aggregate accuracy looks acceptable.

## On real hardware

The connected bimanual XLeRobot has an isolated setup and a local operator UI:

```bash
./scripts/setup_robot.sh
./scripts/control_robot.sh           # http://127.0.0.1:8080; starts with torque OFF
```

Stop other robot controllers first. Support the arm before clicking **Engage**.
Setup, device mapping, calibration, limits, E-stop behavior and the API are in
[Robot control](docs/robot-control.md). This controller does not require Haller,
ROS, Node.js, a GPU, or FLUX weights. Scripted and FLUX task backends still use the
same safety supervisor; see the guide before running tasks on hardware.

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
  numbers in `configs/policy/flux3_so101.yaml` (chunk 42 / execute 12 / 30 Hz /
  8-frame history / ~32 GB BF16) come from planning notes and are marked `VERIFY`.
- LeRobot imports and CLI flags change between releases. They are isolated in
  `hardware/so101.py` and `scripts/record_demos.sh`.
- SO-101 holding is **position** holding, not torque control. Keep loads light.
