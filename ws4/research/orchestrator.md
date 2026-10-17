# Claude-driven orchestrator for the two XLeRobot arms

Owner: ws4-orchestrator session, 2026-10-17. Code: [`ws4/orchestrator/`](../orchestrator/) (run commands in its
README). Reused read-only: `ws4/flux`, `ws4/sim`, `ws4/kinematics`, `ws4/eval/trial_log.py`, `ws4/cloud/warm_prompts.py`.
Line citations below are for the checkout at `e39c443` (branch `ws4-handoff`).

## Report

**Done (laptop, CPU):**
- A localhost HMI where the operator types a goal, sees a validated plan as per-arm lanes with dependencies and the
  concurrent pairs marked, approves it, and watches live step status, camera thumbnails, events and Claude spend. E-stop
  on a button and on the spacebar. One command: `cd ws4 && python -m orchestrator hmi --open`.
- Planner: Claude Opus 5.5 through structured output, validated (schema, vocabulary, arms, DAG, per-arm order, hold
  deadlocks), validator errors fed back at most twice. An offline stub planner for tests and for demos without a key.
- DAG executor on `flux.dual_arm_runner`: independent steps on the two arms run at the same time, one step per arm,
  verification every N chunks with Claude Haiku 5.5 or Sonnet 5.5 (chosen per skill), timeouts, retries, replanning
  with human approval, e-stop with release and resume, operator overrides, scripted primitives (`hold`, `home`,
  `open_gripper`, `close_gripper`) that never use FLUX.
- Inter-arm collision gate on the ws4/kinematics MuJoCo model: every chunk is checked against the other arm's queued
  commands and truncated or refused; a tick-level check latches the e-stop as a last line. An idle arm that blocks the
  other is parked ("make way").
- Laptop backend (a): a pack-box sim scene and scripted IK stand-ins served through the FLUX backend interface, so the
  same runner, gate, guard and executor code runs. "put the apple and the yellow block in the box, then close the lid"
  plans in 0.0 s (stub) as apple (left) || block (right), then lid (right), and runs to completion in sim.
- Every run is logged as a graded attempt with `ws4/eval/trial_log.py` (sim runs in a separate log).
- 49 CPU tests (about 30 s) plus one live Claude test that skips without `ANTHROPIC_API_KEY`; ruff clean.

**Verified:** everything in the test suite on this laptop; the end-to-end sim run on 4 random layouts (seeds 0 to 3, all
3/3 steps); e-stop freezes the sim (no physics step, no command, active time stops) and the run completes after release
and resume with the intervention logged; the gate's cost (0.05 to 0.06 ms per frame); the concurrency model below with a
fake model at the pod's measured latency.

**Also verified:** the request shapes against the real Python SDK (`anthropic` 1.15.0, installed in the laptop's base
env) through a mock transport: the planner call goes to the beta endpoint with `server-side-fallback-2026-07-01`,
`fallbacks: "default"`, adaptive thinking, effort `medium`, a JSON-schema output format and a cached system prompt; the
Haiku verifier call goes to the plain endpoint with effort `low`.

**Not verified:** no live Claude call has been made (no key in this environment), so planner and verifier latency and
cost are estimates; no FLUX inference through the orchestrator (pod not used, needs Oscar's OK); nothing on the real
robot. The skill vocabulary is a placeholder (the recording protocol's S1 to S5) until the demo task is final.

**Next 3 steps:**
1. Oscar: export `ANTHROPIC_API_KEY` and run `python -m pytest orchestrator/tests/test_claude_live.py -s`, then
   backend (b) with `--verifier shadow --image-size 480x640` to measure Claude's plan and check latency, cost and
   agreement with sim ground truth.
2. With the pod up: backend (c) on the sim and `measure_concurrency --backend remote` to replace the modelled numbers.
3. Replace `skills.yaml` with the final demo prompts (exact strings as trained and as in `/workspace/prompts.txt` on the
   pod), then run (d) zero-shot (A0, A0r) and with the LoRAs, logging every attempt.

## Architecture

```
operator (HMI, localhost) --goal--> planner (Claude Opus 5.5 | stub) --plan JSON--> validator --errors--> planner (<= 2x)
                                                                         |
                                       human approves (every plan, every replan)
                                                                         v
executor (30 Hz control thread) --start/stop--> GatedDualArmRunner (flux.dual_arm_runner + collision gate)
   |   ^                                             | per-arm targets: "<arm>:<package>" (FLUX), "<arm>:sim", "<arm>:prim"
   |   | verdicts                                    v
   |   +---- verifier (Claude Haiku/Sonnet on the head frame | sim ground truth | operator)
   |                                                 SafetyGuard (step + joint limits, e-stop latch) -> TickGuardRobot -> robot
   +--> trial_log.py (one line per attempt), runs/<run>/ (plans, events, Claude calls, result)
```

What is reused, and where:
- `DualArmRunner.start/stop/step/release_estop` (`flux/dual_arm_runner.py:235`, `:247`, `:311`, `:303`): per-target
  history and queues, arm ownership, one worker thread per backend for async inference (`:228`), "an arm with no active
  target keeps its last command" (that is `hold`). `per_arm_target` (`:140`) maps head = scene, wrist = that arm's
  gripper camera, as decided for the SO-101 checkpoint.
- `SafetyGuard.filter` and `estop` (`flux/safety.py:95`, `:89`): unchanged and always on.
- `RemoteBackend` (`flux/remote.py:178`); packages with the same checkpoint share one loaded model and one lock on the
  server (`flux/remote.py:121`, `:244`). `FakeBackend` (`flux/backends.py:76`) for tests and the concurrency model.
- Real robot: `build_robot` and `LeRobotBimanualRobot` (`flux/run_xlerobot.py:47`, `:28`), same flags.
- Sim: `XLeRobotStationEnv` (`sim/env.py:63`) with a new scene, `ArmIK` and `top_down_rotation` (`sim/ik.py:23`, `:15`),
  `grasp_offset_x` and `gripper_opening_pct` (`sim/scripted_pick.py:31`, `:41`), `StationUnits` for LeRobot degrees to
  MJCF radians.
- Collision model: `MjCollision.check` (`kinematics/mj_collision.py:81`).
- Trial log: `Trial`, `append`, `summarize` (`eval/trial_log.py:32`, `:84`, `:111`), imported by path, not edited.
- The FLUX prompts in `skills.yaml` are checked by a test against `cloud/warm_prompts.py` `DEFAULT_PROMPTS`, because the
  pod pre-encodes exactly those strings (a new prompt costs about 16 s of CPU text encoding on first use).

Two couplings to private runner state are pinned in one place each: the gate overrides `_accept` and reads
`_targets[...].queue` (`orchestrator/gate.py`), and the executor reads `_owner` for per-arm accounting.

## Plan JSON format (for WS3 or any other planner)

`python -m orchestrator schema` prints the JSON schema (closed objects, every key required, `skill` restricted to the
vocabulary). Example, as the stub produces it for the demo goal:

```json
{
  "rationale": "Both arms pick in parallel, then the right arm, which is next to the lid, closes the box.",
  "steps": [
    {"id": "put_apple_in_box", "arm": "left", "skill": "put_apple_in_box", "depends_on": [], "hold_until": [],
     "success_check": "The red apple is inside the box.", "max_attempts": 2},
    {"id": "put_block_in_box", "arm": "right", "skill": "put_block_in_box", "depends_on": [], "hold_until": [],
     "success_check": "The yellow block is inside the box.", "max_attempts": 2},
    {"id": "close_lid", "arm": "right", "skill": "close_lid", "depends_on": ["put_apple_in_box", "put_block_in_box"],
     "hold_until": [], "success_check": "The lid lies flat on top of the box and covers its opening.", "max_attempts": 2}
  ]
}
```

Rules (`orchestrator/plan.py`, `validate_plan`):
- `id`: 1 to 40 of `[A-Za-z0-9_.-]`, unique. `skill`: a vocabulary name. `arm`: `left` or `right`, and allowed for that
  skill. `max_attempts`: 1 to 3. FLUX steps need a `success_check` (empty means the vocabulary default).
- `depends_on`: existing step ids (or, when replanning, ids of finished steps); no cycles.
- Two steps on the same arm must be ordered through `depends_on` (directly or transitively). This makes each arm's order
  independent of timing, which is what rules out deadlocks below.
- `hold` (only): `hold_until` lists steps on the other arm; the hold starts when its dependencies are done and ends
  when those steps are done. Rejected: a hold whose `hold_until` steps finish before it starts, and any hold whose
  release depends on work queued behind it on the same arm (checked as a cycle in a start/end event graph).
- Execution semantics: a step starts as soon as its dependencies are done and its arm is free. Steps on different arms
  with no ordering between them run concurrently (the HMI marks them with `||`).

Feed a plan from elsewhere with `python -m orchestrator validate plan.json`, `run --plan-file plan.json`, or
`POST /api/plan_json` in the HMI; it is validated and must be approved like any Claude plan.

## Executor

- A FLUX step ends only on a verified check, never on the policy's say-so: checks run every `verify_every_chunks`
  chunks (default 2, about 2.1 s of motion) while the arm keeps moving; Claude calls run in a worker thread. A step
  fails its attempt on `timeout_s` of active time; attempts are retried up to `max_attempts`.
- A step that fails for good pauses everything (running steps are stopped into hold and marked interrupted) and asks the
  planner to replan from the current state: finished steps, the failed step with its reason and the verifier's last
  observation, and a fresh head frame. The operator approves the new plan (or `--auto-approve-replans` in headless
  runs); one replan by default. Nothing startable and nothing running means the run fails as stalled.
- Time is active control time: it stops during e-stop, pause and replanning, so timeouts only count time the arm was
  allowed to move (`clock="ticks"` in sim, wall clock on the robot).
- E-stop latches the guard first (the next runner tick sends nothing; both arms keep their last command), then stops all
  targets. Release, then Resume restarts interrupted steps as fresh attempts (they do not count against
  `max_attempts`); every e-stop, override and abort counts as a human intervention in the trial log.
- Make way: when the gate blocks an arm because of the other arm, and that arm runs no step and holds nothing, the
  executor parks it with the scripted `home` primitive. Found in sim: after "apple in box" is verified the left arm
  stops above the box and, without this rule, the right arm waits until its timeout.

## Safety

- `SafetyGuard` limits stay on (per-tick step and joint limits, UNVERIFIED defaults from `flux/safety.py:22-25`;
  `--calibration-limits` reads the LeRobot calibration on the robot).
- Collision gate: at `_accept`, the first `n_action_steps` frames of a new chunk are checked with `MjCollision` (meshes of
  both arms, fin-ray grippers, torso, table) against the other arm's queued commands, then its last command. At the first
  penetrating frame the chunk is truncated 3 frames earlier (held at the last safe frame); if no frame is safe the arm
  keeps its command. The arm asks for a new chunk once those frames ran, so it waits until the way is clear. A pose may
  never penetrate deeper than the arms already do (plus 0.5 mm), and above 10 mm only non-increasing penetration
  passes, so arms that touch after an e-stop mid-motion can separate. The same rule is checked on every merged command
  before it is sent; a violation latches the e-stop. Measured: 0.05 to 0.06 ms per frame (2 ms per chunk).
- Claude output never reaches the motors: the planner picks vocabulary skills (enum in the schema), the executor maps
  them to fixed prompts or scripted primitives, and a human approves every plan.
- The HMI binds to 127.0.0.1 only, refuses non-local Host headers (DNS rebinding) and requires `X-HMI: 1` with a JSON
  body on every POST, so another page in the operator's browser cannot approve a plan.

## Claude: models, cost, latency

| Use | Model | Settings | Estimated cost per call (UNVERIFIED) |
|---|---|---|---|
| Plan | `claude-opus-5-5` | adaptive thinking, effort `medium`, structured output, system prompt + vocabulary cached, `fallbacks: "default"` | ~2.3k input + 1k to 3k output tokens: about USD 0.03 to 0.07 |
| Plain checks ("X inside the box") | `claude-haiku-5-5` | effort `low`, structured output, head frame at 640 px | ~0.6k input + 0.1k to 0.3k output: about USD 0.0002 |
| Subtle checks (lid closed) | `claude-sonnet-5-5` | effort `low`, `fallbacks: "default"` | about USD 0.002 to 0.004 |

Prices from the claude-api skill's model table (cached 2026-10-06): Opus 5.5 USD 4 / 20 per million input / output
tokens, Sonnet 5.5 USD 2 / 10, Haiku 5.5 USD 0.10 / 0.50 (prompts up to 100K tokens). Images cost about
width x height / 750 tokens. A three-step run with one plan and 10 to 20 checks is roughly USD 0.05 to 0.10, so the USD
200 of credits cover well over a thousand runs. The session budget guard (`--budget-usd`, default 5) refuses calls
once spent; a check that cannot run leaves the step running and tells the operator to mark it in the HMI.
Every call's tokens, cost and latency are logged; replace these estimates with the live test's numbers.

## Concurrency on one shared model (backend c)

Both arms run FLUX through the runner exactly as the executor does. Modelled with a fake model at the pod's measured
1.46 s per chunk (`ws4/cloud/RUNPOD.md`), 32 of 42 actions executed per chunk (1.07 s of motion), 30 Hz, 30 s per run
(`python -m orchestrator.measure_concurrency --backend fake ...`):

| Configuration | Each arm moving | Both arms moving at once | Chunks per minute per arm | Longest control-loop stall |
|---|---|---|---|---|
| async, one shared model (the 32 GB RTX PRO 4500 pod) | 35% | **0%** | 20 | 0.04 s |
| async, two model instances | 41% | 40% | 24 | 0.04 s |
| sync, one shared model | 24% | 24% | 15.6 | 2.96 s |
| (one arm alone, for reference: 1.07 / (1.07 + 1.46)) | 42% | | 23.6 | |

What this means:
- On one shared model the arms take turns: one arm's 1.46 s inference fills the other arm's 1.07 s of motion and
  more, so they settle into strict alternation and never move at the same instant. Total arm motion is 0.70 arm-equivalents
  vs 0.82 with two instances. "Concurrent" plan steps still overlap in time and finish sooner than in sequence, and
  "one arm holds while the other acts" loses nothing, but true simultaneous motion needs two model instances.
- Synchronous inference is the worst case: each arm's inference freezes the whole loop, including the other arm, for up
  to 2.96 s. The orchestrator uses async inference for remote backends by default (as `flux/run_xlerobot.py` does).
- True parallelism needs two loaded models, each on its own lock: BF16 peak was 23.6 GiB per instance with Qwen on
  the CPU (RUNPOD.md), so two instances need about 47 GiB plus decoding headroom: a 96 GB RTX PRO 6000 pod in EU-RO-1
  (about USD 2.49/h, same network volume), or two GPUs. Serve it as two packages on different devices or two server
  processes, then `--left-package` / `--right-package` select them. Not changed on the pod (that is the orchestrator
  session's call).
- If the guidance-1.0 and compile levers bring latency under the 1.07 s of motion (UNVERIFIED), a shared model starts to
  overlap the arms' motion: with 0.73 s per chunk each arm's cycle would be max(1.07 + 0.73, 2 x 0.73) = 1.8 s, about
  59% motion each (model, not measured).
- Not yet measured on the pod: replace the table's first row with `measure_concurrency --backend remote` (the pod's
  real latency distribution, network and JPEG transport included).

## Laptop backend (a): the sim stand-in

- Scene `orchestrator/scenes/pack_box.xml` includes the ws4/sim robot (read-only) and adds an open 15 cm box at
  (0.25, 0) m, where both arms reach it top-down with zero IK error, a red "apple" (4.4 cm flattened ellipsoid with rolling
  friction; a perfect sphere rolled away from the descending fin), a 4 cm yellow block, and a loose 17 cm lid with a knob
  that the scripted skills pick and place. Layout jitter of 2 cm and yaw per seed; ws4/sim's colour, light and camera
  randomisation stay on.
- Each scripted skill is planned once (IK waypoints with the sim's grasp geometry, cosine-profiled at 2.5 deg per tick)
  and served chunk by chunk; progress re-anchors to the last executed command every chunk, so gate truncation only
  delays it. It reads privileged object poses, so it is a stand-in for orchestration tests, not a policy.
- Measured on this laptop (16 cores, EGL, three 320x240 cameras): the demo goal takes 19.3 s of sim time (578 ticks) and
  about 13 s of wall time in `--fast` mode; 4 of 4 seeds succeed. In that run both arms moved together for 275 ticks
  (9.2 s), the gate truncated or refused 3 right-arm chunks while the left arm was over the box, and make-way parked the
  idle left arm once.

## Findings for other workstreams

- ws4/sim: `lidbox_hinge` in `sim/scenes/tabletop_props.xml:77` (and `lid_box.xml`) has `axis="0 -1 0"`, so positive
  angles swing the lid down through the box and table (the handle is at z = -0.04 m at 1.0 rad). It should probably be
  `0 1 0`. The lidbox at x = 0.42 m is also out of top-down reach for both arms (IK error 6 to 13 cm).
- WS3: the plan JSON above is the contract; `validate` and `run --plan-file` accept plans from any planner.
- Hardware: `HOME_DEG` (`sim/env.py:28`) is used as the park pose and the `home` primitive on the real arms. UNVERIFIED
  on hardware; check it before the first robot run.
