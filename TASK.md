# TASK.md: workstreams, milestones, acceptance

30-hour plan for five people on two SO-101 arms (→ bimanual XLeRobot). Strategy and the
FLUX option A (per-arm) vs option B (bimanual fine-tune) decision: [PLAN.md](PLAN.md). One owner
per workstream; everyone codes against the shared interfaces from **hour 2**.
Fill in owners:

| WS | Workstream | Owner | Primary code |
|----|-----------|-------|--------------|
| WS1 | Single-arm FLUX baseline + demos + LoRA | _TBD_ | `policy/`, `data/` |
| WS2 | Bimanual obstacle removal to reach a target | _TBD_ | `bimanual/` |
| WS3 | Long-horizon decomposition + orchestration | _TBD_ | `orchestration/`, `perception/` |
| WS4 | Sim + XLeRobot embodiment adaptation | _TBD_ | `sim/`, `configs/embodiment/` |
| WS5 | Hardware setup + safety + UI / triggering | _TBD_ | `hardware/`, `safety/`, `ui/` |

---

## Shared milestones

| Hour | Milestone | Gate |
|------|-----------|------|
| H0–2 | Agree on demo task, physical layout, predicate names, interfaces (`core/`, task grammar) | `pytest` green on everyone's machine |
| H6 | **Decision point 1:** FLUX inference running on one arm? If not, the team proceeds on `--backend scripted` | WS1 smoke test numbers posted |
| H12 | Single-arm FLUX skill on hardware; dual-arm task runs end-to-end in mock/sim; holding primitive works on hardware | `liminal run lid_hold_insert` on mock + sim |
| H18 | **Decision point 2:** physical WHILE (hold + insert) works? If not, freeze and demo sequential AFTER (`clear_obstacle_pick`) | ≥1 real success |
| H24 | Feature freeze. Long-horizon task + recovery, ≥5 trials logged | success-rate table |
| H30 | Demo, video, slides | acceptance checklist below |

## Final demo acceptance checklist

- [ ] Both SO-101 arms operate safely from the same executor (e-stop tested)
- [ ] At least one real manipulation skill runs on FLUX 3 Action
- [ ] A skill waits on / depends on the other arm's physical condition (WHILE or AFTER)
- [ ] Perception verifies success, and a skill claiming success without the effect is caught
- [ ] A changed scene or failed action produces a safe recovery (retry / fallback)
- [ ] ≥5 physical trials with varied start layouts, success rate reported from `runs/*.jsonl`

---

## WS1: Single-arm FLUX 3 Action baseline → demos → task LoRA

**Goal:** know how well the out-of-the-box SO-101 checkpoint does on *our* table, then
improve one skill with a task-specific LoRA from 50–100 demos.
Brief: [docs/research/01_single_arm_baseline.md](docs/research/01_single_arm_baseline.md)

- [ ] **H0–3** Load checkpoint + processors; implement `FluxPolicyBackend.load/predict`
      (`src/liminal/policy/flux.py`); run `scripts/flux_smoke_test.py`; record VRAM + latency.
      Replace every `VERIFY` in `configs/policy/flux3_so101.yaml` with checked values.
- [ ] **H3–6** Zero-shot eval on one arm: `single_pick_place`, 10 trials × 3 layouts.
      Log success rate + failure modes (grasp miss, wrong object, drift, timeout).
- [ ] **H4–10** Collect demos with `scripts/record_demos.sh` following a variation plan
      from `scripts/make_demo_plan.py` (zones × yaw × distractors × lighting). Target
      60 episodes first, 100 if time allows. Same camera poses as eval.
- [ ] **H10–18** LoRA fine-tune (`scripts/finetune_lora.sh`, fill in the BFL recipe), hold
      out ~10% of episodes. Re-run the same eval and compare against zero-shot.
- [ ] **H10–14** Option A1 vs A2: if a second GPU is available, try `instances: per_arm` and
      per-arm LoRAs (`arm_adapters`), especially if one arm's viewpoint hurts zero-shot success.
- [ ] **Stretch** Prefetch the next chunk during execution to remove inter-chunk pauses.
      Second skill (`open_lid` or `push_aside`) for WS2/WS3.

**Provides:** a `SkillBackend` that works for `pick_place` (+ ideally `open_lid`,
`push_aside`). **Needs:** WS5 arm + cameras, a GPU (≈32 GB BF16, VERIFY).
**Done when:** a measured zero-shot vs LoRA success rate on the same eval protocol.

## WS2: Bimanual obstacle removal to reach a target

**Goal:** a reliable recipe for "the target is blocked, so one arm clears or holds the
obstacle and the other arm gets the target", expressed as task trees the executor runs.
Brief: [docs/research/02_bimanual_obstacle.md](docs/research/02_bimanual_obstacle.md)

- [ ] **H0–4** Survey and pick 2–3 obstacle classes for the demo table (slide-able cup,
      pick-able block, hinged lid/flap that needs holding). Define predicates for each.
- [ ] **H4–8** Calibrate a shared table frame (arm bases in `configs/robot/so101_dual.yaml`
      → `bases`). Iterate on `bimanual/obstacle.py`: role assignment, strategy choice,
      reachability. `scripts/plan_obstacle.py` prints generated trees.
- [ ] **H8–14** Run generated trees in mock and sim (WS4), then on hardware with
      scripted/FLUX skills. Measure: clear success, collisions, time.
- [ ] **H14–20** Get obstacle positions from perception (WS3 detectors) instead of
      hand-entered coordinates. Add a "re-check blockers after clearing" loop.
- [ ] **Stretch** Learned/VLM strategy selection (ask the VLM *how* to clear an object).

**Provides:** `plan_clear_and_act()` → task spec; obstacle predicates.
**Needs:** WS1 skills (`push_aside`, `pick_place`, `lift_aside`), WS3 detectors, WS5 safety zones.
**Done when:** the `clear_obstacle_pick` demo succeeds on hardware with generated
(not hand-written) trees.

## WS3: Long-horizon decomposition + orchestration

**Goal:** break long tasks into verifiable subtasks and run them with AFTER / WHILE /
UNTIL dependencies, memory and recovery. Compare BT vs FSM vs LLM vs Gemini ER vs VLM planning.
Brief: [docs/research/03_long_horizon_orchestration.md](docs/research/03_long_horizon_orchestration.md)

- [x] Behavior tree with `Sequence / Parallel / Fallback / While / Retry / Timeout /
      WaitFor / Check / Set` (`orchestration/bt.py`), task grammar + compiler, executor
      with goal re-verification, mock-tested.
- [ ] **H0–6** Agree on the predicate list with WS1/WS2/WS5. Implement `ColorZoneDetector`
      rules for `lid_open`, `object_inserted`, `block_in_bowl`, `obstacle_cleared` on the
      real top camera. Run `PerceptionLoop` during tasks.
- [ ] **H6–12** Planner bake-off on 5 goals: `TemplatePlanner` (baseline) vs `LLMPlanner`
      with Claude vs Gemini Robotics-ER (with scene image). Metric: valid-tree rate,
      tree correctness (manual), latency. Results go in the brief.
- [ ] **H12–20** Long-horizon task `tidy_table_long_horizon` on hardware: subgoal
      markers, resume after failure (skip done subgoals via `fallback: [check, ...]`),
      VLM verifier for the hardest predicate.
- [ ] **H20–24** Replanning: on subtree failure, ask the planner for a new tree given
      the current world state + event log.
- [ ] **Stretch** FSM baseline for comparison; LLM querying the event-log memory.

**Provides:** executor, task grammar, world state, detectors, planners.
**Needs:** skills from WS1, cameras from WS5.
**Done when:** a ≥4-subtask task runs on hardware with at least one recovered failure.

## WS4: Simulation + bimanual XLeRobot embodiment adaptation

**Goal:** a sim with the same `Arm` API for orchestration testing, then a first
embodiment fine-tune toward bimanual XLeRobot.
Brief: [docs/research/04_xlerobot_sim.md](docs/research/04_xlerobot_sim.md)

- [ ] **H0–6** Pick a sim scene: dual SO-101 MJCF or XLeRobot assets. Actuators named
      `left_*` / `right_*`. Wire `configs/robot/mujoco_dual.yaml`; `MujocoArm` passes the
      same task trees as mock (`--robot configs/robot/mujoco_dual.yaml`).
- [ ] **H6–12** Sim predicates from ground-truth object poses (a `PredicateDetector`
      reading MuJoCo state). Gives WS2/WS3 a physics-backed test bed.
- [ ] **H6–14** Dataset survey: open bimanual SO-101/XLeRobot LeRobot datasets, UMI-style
      egocentric human data. Record format, action space, camera setup, licence.
- [ ] **H8–14** Option B prerequisites: confirm with BFL docs how to fine-tune a 12-D action
      head; record a first batch of bimanual demos (two leader arms) for `open_lid_and_insert`.
- [ ] **H14–18** Option B first fine-tune into `configs/policy/flux3_dual_bimanual.yaml`
      (layout `configs/embodiment/so101_dual.yaml`). Evaluate `lid_hold_insert_bimanual` vs
      `lid_hold_insert` in sim. **Go/no-go for hardware at H18** (PLAN.md).
- [ ] **H18–24** Extend the recipe to XLeRobot (`configs/embodiment/xlerobot.yaml`, `Embodiment.split/merge`).
- [ ] **Stretch** Sim-collected demos to augment WS1 data. Domain randomisation.

**Provides:** `MujocoArm/MujocoCamera`, sim predicates, embodiment configs, dataset notes.
**Needs:** task trees from WS3, FLUX training access from WS1.
**Done when:** shipped task files pass in sim, and there's a written go/no-go on the
XLeRobot fine-tuning path with a first training curve.

## WS5: Hardware setup + safety + UI / triggering

**Goal:** two arms, cameras and a safe, boring, reliable hardware layer, plus a simple
way to trigger tasks and see state.
Brief: [docs/research/05_hardware_ui.md](docs/research/05_hardware_ui.md)

- [ ] **H0–3** Mount both arms + top camera + 2 wrist cameras rigidly. `lerobot-find-port`,
      calibrate both arms, fill `configs/robot/so101_dual.yaml` (ports, limits, home).
- [ ] **H3–6** Verify `SO101Arm`: read/send, `hold()` keeps pose under a light load,
      `stop()`. Record scripted waypoints (`scripts/record_waypoints.py`) for the
      fallback backend.
- [ ] **H6–10** Safety: measured joint limits, max step, workspace zones for the shared
      box area, physical e-stop key bound to `POST /estop` (and Ctrl-C in CLI).
- [ ] **H10–16** Operator UI on top of `liminal serve`: task buttons, live world facts,
      arm owners, e-stop. Optional voice → `POST /run {"goal": ...}`.
- [ ] **H16–30** Integration owner for hardware runs: reset procedure, trial logging,
      demo video.

**Provides:** `Arm`/`Camera` implementations, safety limits, trigger UI.
**Needs:** nothing, which is why it starts first.
**Done when:** e-stop works from UI and keyboard, both arms hold and release on command,
and any teammate can start a demo from the UI.

---

## Integration contracts (agree at H2, change only with a heads-up)

- **Joint order:** `core/types.py::SO101_JOINTS`; units = LeRobot calibrated units.
- **Arm names:** `left`, `right`. **Camera names:** `top`, `left_wrist`, `right_wrist`.
- **Predicates:** `configs/skills.yaml` is the single list. Names are snake_case booleans.
- **Skills:** `configs/skills.yaml`. Primitives (`hold`, `release`, `home`, grippers)
  never use the VLA.
- **Task files:** grammar in `orchestration/task_spec.py`; every planner emits it.
- **Logs:** `runs/<timestamp>_<task>.jsonl`, one event per line (`kind`, `t`, ...).
