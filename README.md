# liminal

**[Project page](https://javgilavi.github.io/liminal/) · [Live lab](https://javgilavi.github.io/liminal/live.html)**

**Long-horizon, dual-arm collaboration on SO-101 / XLeRobot, using FLUX 3 Action for manipulation.**

One arm changes or holds something in the scene so the other arm can do what it
otherwise couldn't. FLUX 3 Action handles the *learned manipulation*: bounded,
text-prompted skills on one arm. Everything above that is this repo: deciding
*why* a skill is needed, *when* the other arm is ready, *which conditions must stay
true*, and *whether the goal really happened*.

[![Liminal architecture: a deterministic behavior-tree harness coordinates independent left and right FLUX 3 Action policies adapted with task LoRA. A skill registry selects checkpoint bundles; safety checks gate robot commands. Scene and wrist cameras feed Qwen3-VL-4B-Instruct perception, which updates verified world-state predicates for the harness.](docs/assets/liminal-architecture.png)](docs/assets/liminal-architecture.png)

*Configured runtime: FLUX 3 Action + task LoRA on each arm; Qwen3-VL-4B-Instruct for
perception. The harness, registry and safety checks are deterministic software.
Full learned dual-arm hardware performance remains unverified.*

<details>
<summary>Figure details and design references</summary>

The figure shows the main learned-skill path. Local hold/release primitives bypass
VLA inference; optional task-planning models and the disabled standalone hybrid
controller are omitted. Each policy receives its arm's joint state and camera
observations; joint states come from the arm drivers. The embodiment panel uses
the supplied photograph of our green XLeRobot rig, with its room background removed
and the composition adapted for the figure. It illustrates the hardware, not a
recorded successful rollout.
See [the architecture](docs/architecture.md) for the complete contracts.

Figure generated and refined with the built-in image-generation tool using our
robot photograph; [generation and refinement prompts](docs/assets/architecture-rounded-prompt.txt).
Visual references: [OpenVLA](https://openvla.github.io/) and its
[GitHub repo](https://github.com/openvla/openvla),
[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) and its
[GitHub repo](https://github.com/real-stanford/diffusion_policy),
[π₀](https://www.pi.website/blog/pi0), and
[Nerfies](https://nerfies.github.io/).
These informed the layout and visual hierarchy, not the Liminal architecture.

</details>

Strategy and key decisions: [PLAN.md](PLAN.md). Full design: [docs/architecture.md](docs/architecture.md).
Who does what, and by when: [TASK.md](TASK.md). Rules for humans and coding agents:
[AGENTS.md](AGENTS.md) (`CLAUDE.md` is an identical copy for Claude Code).

## First single-arm validation

The first SO101 validation includes **official LeRobot FLUX inference on the real
robot** and **MuJoCo task-LoRA training/evaluation**. In five matched simulation
scenes, the raw adapter achieved bilateral contact in 3/5 and lifts over 3.5 cm
in 2/5; all models completed 0/5 placements. Physical inference and motor execution
are recorded, with task success still unverified.

- [Results, evidence and limitations](docs/validation/single-arm-v1.md)
- [Run real-robot inference](docs/official-inference.md)
- [Reproduce the MuJoCo experiment](docs/simulation-validation.md)
- [Documentation index](docs/README.md)

## Current policy architecture

The initial plan was to fine-tune a unified bimanual XLeRobot embodiment (12-D actions). That plan
was superseded by task LoRA fine-tuning of the existing 6-D SO-101 policy, with two independent FLUX
instances, one per arm, coordinated by the real skill-runtime harness (behavior tree, checkpoint
registry, perception and safety).

The dual-arm profiles select `instances: per_arm`: separate model state and locks allow
concurrent inference when the task permits. LoRA adapters are merged into validated
skill bundles for the checkpoint registry. Holding uses a local controller; two
policies do not imply both arms always move. See [the runtime guide](docs/showcase-runtime.md).

The old 12-D configs and group tests remain compatibility scaffolds, with no validated
bimanual checkpoint. XLeRobot remains the physical rig. `instances: shared` is an
explicit memory-saving fallback, not the selected dual-arm configuration.

## Workstreams

| # | Workstream | Code | Brief |
|---|-----------|------|-------|
| WS1 | Single-arm FLUX baseline + 50–100 demos + task LoRA | `policy/`, `data/`, `scripts/record_demos.sh` | [01](docs/research/01_single_arm_baseline.md) |
| WS2 | Bimanual: move obstacles away to reach a target | `bimanual/` | [02](docs/research/02_bimanual_obstacle.md) |
| WS3 | Long-horizon decomposition + orchestration (BT / FSM / LLM / Gemini ER / VLM) | `orchestration/`, `perception/` | [03](docs/research/03_long_horizon_orchestration.md) |
| WS4 | Simulation + per-arm LoRA evaluation (original embodiment plan superseded) | `sim/`, `configs/embodiment/` | [04](docs/research/04_xlerobot_sim.md) |
| WS5 | Hardware setup, safety, operator UI / triggering | `hardware/`, `safety/`, `ui/` | [05](docs/research/05_hardware_ui.md) |

## On the Haller (ws4 bridge)

On the green Haller the tree runs through the ws4 stack: each skill becomes one ws4 step (kit keep-outs, start gate,
held box arm, stall and load freezes, e-stop that freezes with torque on), FLUX runs on the remote policy server with
one LoRA package per skill, and `ws4_haiku` perception reuses ws4's measured Haiku checks. Command, arm naming and
what is verified: [ws4/HARNESS_BRIDGE.md](ws4/HARNESS_BRIDGE.md).

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

Once the labeled evaluation passes, verify live camera capture and blackboard
updates without connecting to either arm bus or issuing arm commands:

```bash
uv run liminal inspect-live \
  --robot configs/robot/so101_dual.yaml \
  --perception configs/perception/qwen3_vl_4b.yaml \
  --samples 3
```

This command cannot determine whether another controller has left motor torque
enabled, so verify that separately. It connects only the perception cameras and
writes the captured frames, raw responses, parsed predicates, observation versions,
and final blackboard state to `runs/`.

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

For Haller UI ownership and remote GPU inference, follow the
[official LeRobot inference guide](docs/official-inference.md). It defaults to
real-camera inference without motor commands; execution requires a verified rig
profile and explicit UI engagement. Liminal supplies SSH transport and execution;
LeRobot supplies `Flux3Policy` and its saved pre/postprocessors.

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
scripts/          real/GPU inference, simulation collection/training/evaluation, hardware utilities
experiments/      compact validation results, provenance and training recipes
sim/single_arm/   versioned MuJoCo scene/model; external meshes
integrations/     external Haller controller patch and installation guide
docs/             guides, validation reports, architecture and research briefs
```

## Status and caveats

- The orchestration stack (BT, WHILE/UNTIL, safety, world state, planners, mock and
  scripted backends) runs and is tested.
- **Single-arm inference is implemented** through the dedicated official LeRobot
  runner. The HF Space normalization profile is experimental on the physical rig.
  The real skill runtime loads validated bundles through `LeRobotEngine` and routes
  skills to independent per-arm backends. The showcase registry still has unstaged
  opening/green-pen/boxcutter entries; a full learned dual-arm hardware run is unverified.
  Planning configs retain `VERIFY` markers and are not commissioned inference profiles.
- The simulation adapter improves grasp-related measures on a small test set;
  completed placement, real-task success and bimanual transfer remain unverified.
- LeRobot imports and CLI flags change between releases. They are isolated in
  `hardware/so101.py` and `scripts/record_demos.sh`.
- SO-101 holding is **position** holding, not torque control. Keep loads light.


## Public project page and live digital twin lab

Explore the longer interactive project page at **https://javgilavi.github.io/liminal/**.
The separate [full-screen lab](https://javgilavi.github.io/liminal/live.html) includes a
rotatable recorded MuJoCo scene, scene/wrist views and explicit pairing to a local
Haller companion. The project page distinguishes real red-pen insertion highlights,
scripted teachers, simulation comparisons and older unlabeled hardware trials.

```bash
uv sync --extra research --extra dev
./scripts/research_ui.sh \
  --profile configs/policy/official_haller_right.yaml \
  --site-url https://javgilavi.github.io/liminal/
# Open http://127.0.0.1:18003/pair on the robot-side machine.
```

**Liminal** is the project; **Haller** is the hardware/controller. Live cameras and
measured joints travel directly from the local companion to the browser. Trials
use the existing guarded official runner and save locally under `runs/research/`.
The viewer mirrors arm joints; object poses, contact and physical registration are
not automatically tracked. [Setup, safety gates and implementation](docs/research-ui.md).
The static site is deployed independently from the dedicated `gh-pages` branch.
