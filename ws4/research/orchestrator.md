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
- 58 CPU tests (about 30 s) plus one opt-in live Claude test; ruff clean. Offline tests cannot reach the API (an
  autouse fixture hides any key).

**Verified:** everything in the test suite on this laptop; the end-to-end sim run on 4 random layouts (seeds 0 to 3, all
3/3 steps); e-stop freezes the sim (no physics step, no command, active time stops) and the run completes after release
and resume with the intervention logged; the gate's cost (0.05 to 0.06 ms per frame); two-arm concurrency on the real pod (one shared FLUX
model: the arms alternate, 30% / 28% motion each, never simultaneous, 1.76 s round trip per chunk).

**Verified live against the Claude API (2026-10-17 15:06, `python -m orchestrator.live_check`, USD 0.04 in total):**
the Opus 5.5 planner returns a valid plan from a 640x480 sim head frame in 12.1 s cold and 5.9 s with the cached system
prompt; Haiku 5.5 checks take 1.2 to 2.1 s and answered 4 of 4 correctly (apple on the table / in the box); a Sonnet 5.5
lid check takes 1.7 s and was correct. Request shapes were also checked against the real SDK (`anthropic` 1.15.0)
through a mock transport. Details in "Claude: measured latency and cost" and "Claude API: docs and SDK".

**Not verified:** FLUX driving a whole orchestrated task (the pod measurement exercised the runner and policy server
with both arms, not a plan); nothing on the real robot; Claude's checks on real camera frames (only sim renders so far). The skill vocabulary is a placeholder (the
recording protocol's S1 to S5) until the demo task is final.

**Next 3 steps:**
1. Run backend (b) in the HMI with `--planner claude --verifier shadow --image-size 480x640` on several seeds to measure
   Claude's check accuracy against sim ground truth over whole runs (the shadow verifier logs every disagreement).
2. Backend (c) end to end: FLUX zero-shot driving the sim arms through the HMI (`--policy remote`), to see whether the
   released checkpoint does anything sensible on the sim images, and to log it as a sim attempt.
3. Replace `skills.yaml` with the final demo prompts (exact strings as trained and as in `/workspace/prompts.txt` on the
   pod), then run (d) zero-shot (A0, A0r) and with the LoRAs, logging every attempt.

## Sunday demo: the Haller pen task (wired 2026-10-17 evening)

- **Task** (final, from the orchestrator session): the robot's LEFT arm (`/dev/haller_arm_right`, calibration
  `haller_right`) runs two FLUX skills with the datasets' exact prompts, "put the red pen in its slot in the cardboard
  box" and "take the red pen out of its slot and put it on the mat", plus the `home` primitive at
  `[0, 0, 0, 0, 0, 100]` and `hold`. The robot-RIGHT arm stays parked and is not connected. `skills.yaml` now holds
  this vocabulary with a `station` block (available arms, home pose, notes for the planner); the sim pack-box vocabulary
  moved to `skills_sim_pack_box.yaml`.
- **Arm side**: `orchestrator/haller.py` imports `ws4/single_arm/run_single_arm.py` read-only (`connect_so101`,
  `SingleArmRobot`, `build_guard`, `KeepOutGuard`, `hold_here`, `warm_up`, `disconnect_so101`, `FakeSO101`) and the
  `keepout` module; `orchestrator/monitors.py` rebuilds its per-tick stall stop (8 deg for 15 ticks, arm joints only)
  and keep-out stop (45 consecutive refusals). The inter-arm collision gate is off on this station (one moving arm, the
  other not connected); the keep-out's midline zone guards that side. Every FLUX step first ramps to the home pose at
  1 deg per tick through the same guard, because every VR demo starts there. A stop freezes the arm at its MEASURED
  pose (`hold_here`), as the single-arm runner does.
- **Logging**: one `trial_log.py` line per FLUX attempt (task = the skill, 1/1 or 0/1 with the failure reason and the
  interventions during that attempt) and one per chain (task `pen_chain`). The fake arm and observe mode log to
  `runs/trials_mock.jsonl`, never the graded log.
- **Verified (no hardware, no pod)**: on run_single_arm's `FakeSO101` with a fake policy, "put the pen in the box,
  take it out, put it back" plans as three chained left-arm steps; the arm is within 3 deg of home each time a FLUX
  policy starts; a failed attempt (timeout) is logged and retried; a monitor stop latches the e-stop and calls the hold
  (`tests/test_haller.py`, 7 tests). Live Claude plan for "put the pen in the box, take it out, put it back, and
  repeat the whole cycle twice": 14.2 s, USD 0.030, 8 steps, left arm only, pen states consistent (it inserted a
  take-out between cycles) and a final `home`.
- **Not verified**: anything on the real arm (VR recording holds it tonight), the policy server on the RTX PRO 6000 pod
  for this run, and Haiku's checks on real head frames: the pen in its slot is a small detail in a 640x480 frame. In
  the observe dry run, look at a few checks; if Haiku misjudges, add `--verifier-wrist` (the wrist frame goes in too)
  or switch these two skills to `claude-sonnet-5-5` in `skills.yaml`.
- **Run command**: `orchestrator/README.md`, "Sunday demo".

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

## Claude: measured latency and cost

Measured with `python -m orchestrator.live_check` on 2026-10-17 at 15:06 CEST (`runs/live_claude_20261017_150603.json`,
SDK 1.15.0, frames from the pack-box sim rendered at 640x480 and sent as JPEG, 414 image tokens each). One call each:

| Call | Model, settings | Latency | Input (uncached / cache write / cache read) | Output (thinking) | Cost USD | Result |
|---|---|---|---|---|---|---|
| Plan, cold | `claude-opus-5-5`, effort `medium`, adaptive thinking, structured output, fallbacks | 12.1 s | 465 / 2,207 / 0 | 565 (93) | 0.0242 | valid, 5 steps |
| Plan, same request again | same | 5.9 s | 465 / 0 / 2,207 | 522 (90) | 0.0127 | valid, 5 steps |
| Check "apple in box", apple on the table | `claude-haiku-5-5`, effort `low`, adaptive | 1.89 s | 826 | 58 (0) | 0.00011 | no (correct) |
| same, apple in the box | same | 1.54 s | 826 | 57 (0) | 0.00011 | yes (correct) |
| same two checks, thinking disabled | `claude-haiku-5-5`, effort `low`, `disabled` | 2.06 s, 1.15 s | 825 | 52, 49 (0) | 0.00011 | no, yes (correct) |
| Check "lid on the box", lid on the table | `claude-sonnet-5-5`, effort `low`, adaptive, fallbacks | 1.67 s | 835 | 68 (0) | 0.0024 | no (correct) |

What the numbers say:
- Prompt caching works on the planner: the system prompt plus vocabulary (2,207 tokens, above the 512 minimum) is
  written on the first call and read on the second, which halves latency (12.1 to 5.9 s) and cost (0.024 to 0.013).
  It stays warm for 5 minutes between plans; replans within a run hit it.
- Haiku 5.5 at effort `low` did not think at all (0 thinking tokens) on these checks, so disabling thinking changed
  nothing measurable; the default stays adaptive. A check costs USD 0.0001 and about 1.5 s, so checking every 2 chunks
  (2.1 s) keeps at most one check in flight per step.
- Claude's plan (both calls): apple (left) || block (right), then `home` on the left arm, then the lid with the right
  arm, then `home` on the right arm. It parks the left arm before the lid on its own, which is the blocking case the
  executor's make-way rule exists for.
- A three-step run with one plan and 10 to 20 checks costs about USD 0.03 to 0.05 (Opus plan plus Haiku checks, one
  or two Sonnet lid checks). The USD 200 of credits are not a constraint; the session guard (`--budget-usd`, default 5)
  refuses calls once spent, and a check that cannot run leaves the step running and asks the operator to mark it.
- One sample per call: latency will vary (the planner's p90 is unknown). Every call in a session is logged to
  `runs/<run>/claude_calls.jsonl` with tokens, cost, latency and the request id.

## Claude API: docs and SDK

SDK: `anthropic==1.15.0` (pinned in `orchestrator/requirements.txt`, Python >= 3.10, httpx2 transport), installed in the
laptop's base env. Key: `ANTHROPIC_API_KEY`, else `ANTHROPIC_API_KEY_FILE` (default `~/.config/ws4/anthropic.env`); never
logged. Official pages read on 2026-10-17 for every feature used (also cited in `orchestrator/claude.py`):

| Feature | How it is used | Docs |
|---|---|---|
| Messages API | `client.messages.create` (Haiku), `client.beta.messages.create` (Opus, Sonnet: fallback beta) | https://platform.claude.com/docs/en/api/messages/create |
| Structured outputs | `output_config.format = {type: json_schema}`; closed objects, `enum` for skills and arms, no numeric bounds (checked by the validator); enum values normalised to lower case because the docs warn casing may differ | https://platform.claude.com/docs/en/build-with-claude/structured-outputs |
| Why not tool use | we want the reply itself as JSON, not a function call, and forced `tool_choice` is a 400 on Opus 5.5 and Sonnet 5.5 | same page, "Strict tool use vs JSON outputs"; https://platform.claude.com/docs/en/api/errors |
| Adaptive thinking, effort | `thinking: adaptive`; effort `medium` (planner) and `low` (checks); thinking tokens billed as output, reported in `usage.output_tokens_details.thinking_tokens` | https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost |
| Vision | base64 JPEG, head frame downscaled to 640 px long edge (414 tokens = ceil(w/28) x ceil(h/28)); images placed before the text; limits 8000x8000 px and 10 MB per image, 32 MB per request | https://platform.claude.com/docs/en/build-with-claude/vision |
| Prompt caching | `cache_control: ephemeral` on the planner's system block (5-minute TTL, write 1.25x, read 0.05x on Opus 5.5); minimum 512 tokens on all three 5.5 models; changing effort or the output format invalidates it | https://platform.claude.com/docs/en/build-with-claude/prompt-caching |
| Refusals, fallback | `stop_reason == "refusal"` checked before reading content; `fallbacks: "default"` + beta `server-side-fallback-2026-07-01` on Opus 5.5 and Sonnet 5.5; Haiku 5.5 has no server-side fallback, so a refused check returns "not verified" and the step keeps running | https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback |
| Errors, retries | SDK typed exceptions; it retries 408, 409, 429, 5xx and connection errors twice with backoff and honours `retry-after`; a spend-cap 429 has no `retry-after` | https://platform.claude.com/docs/en/api/errors, https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python |
| Rate limits | Start tier: 1,000 RPM and 2M input tokens per minute per 5.5 model; cache reads do not count; far above our rate (one plan, one check every 2 s) | https://platform.claude.com/docs/en/api/rate-limits |
| Pricing | Opus 5.5 4 / 20, Sonnet 5.5 2 / 10, Haiku 5.5 0.10 / 0.50 USD per million input / output tokens (Haiku up to 100k-token prompts); cache reads 0.20 / 0.10 / 0.01 | https://platform.claude.com/docs/en/about-claude/pricing |
| Streaming | not used: replies are short (under 1k tokens), so non-streaming calls with a 180 s timeout are well inside the SDK's 10-minute limit | https://platform.claude.com/docs/en/build-with-claude/streaming |

Mismatches found:
- Pricing: the claude-api skill (cached 2026-10-06) lists Sonnet 5.5 cache reads at USD 0.20 per million tokens; the
  pricing page says USD 0.10 (0.05x of input). The code uses the pricing page.
- Image tokens: an earlier comment here (and `imagine/score.py`'s docstring) used width x height / 750; the vision page
  defines ceil(w/28) x ceil(h/28) patches (414 tokens for 640x480, about the same; diverges for small images).
- The structured-outputs page does not say whether it combines with adaptive thinking, images and server-side
  fallbacks. The live planner call uses all four together on the beta endpoint and works.
- No doc vs SDK mismatch for the parameters used: SDK 1.15.0 accepts `fallbacks`, `betas`, `output_config` and
  `thinking` as typed arguments and serialises them as the docs describe (checked through a mock transport).

## Concurrency on one shared model (backend c)

Both arms run FLUX through `DualArmRunner` exactly as the executor does (`python -m orchestrator.measure_concurrency`),
each with its own prompt ("Put the apple in the box" / "Put the yellow block in the box"), static 640x480 frames, 30 Hz.
**Measured on the pod** on 2026-10-17 at 15:25 CEST: the RunPod RTX PRO 4500 32 GB, one `flux.remote` server with
package `so101` (one loaded model, default guidance, text encoder on the CPU), reached through an SSH tunnel from the
laptop, 20 s warm-up, then 60 s timed per mode (`runs/concurrency_remote_pod_{async,sync}.json`). Policy requests
only; no change to the pod.

| Configuration | Each arm moving | Both arms moving at once | Chunks per minute per arm | Round trip per chunk (median) | Longest control-loop stall |
|---|---|---|---|---|---|
| **Pod, async, one shared model (measured)** | **30% / 28%** | **0%** | **17 / 16** | **1.76 s** | **0.11 s** (29.6 Hz held) |
| **Pod, sync, one shared model (measured)** | **20% / 20%** | 20% | **12 / 12** | 1.73 s / 1.75 s | **3.84 s** (6 ticks/s on average) |
| Model: async, one shared model, 1.46 s per chunk | 35% | 0% | 20 | 1.46 s | 0.04 s |
| Model: async, two model instances, 1.46 s each | 41% | 40% | 24 | 1.46 s | 0.04 s |
| Model: sync, one shared model, 1.46 s | 24% | 24% | 15.6 | 1.46 s | 2.96 s |

The model rows use a fake backend with the latency measured earlier on the same pod (RUNPOD.md) and match the
measurements' shape; the measured numbers are lower because a round trip through the tunnel is 1.76 s, not 1.46 s
(about 0.3 s of network, JPEG encode and decode, and server overhead per chunk).

What this means:
- On one shared model the arms take turns. One arm's inference (1.76 s round trip) is longer than the other arm's
  1.07 s of motion, so they settle into strict alternation and never move at the same instant (measured 0%). Each arm
  moves about 29% of the time, so the station delivers about 0.58 arm-equivalents of motion. "Concurrent" plan steps
  still overlap in time and finish sooner than in sequence, and "one arm holds while the other acts" loses nothing,
  but simultaneous motion needs two model instances or a faster model.
- Synchronous inference is the worst case: each arm's inference freezes the whole control loop, the other arm
  included, for up to 3.84 s, and each arm moves 20% of the time. The orchestrator uses async inference for remote
  backends by default (as `flux/run_xlerobot.py` does).
- True parallelism needs two loaded models, each with its own lock: BF16 peak was 23.6 GiB per instance with Qwen on
  the CPU (RUNPOD.md), so two instances need about 47 GiB plus decoding headroom. That means a 96 GB RTX PRO 6000 pod in
  EU-RO-1 (about USD 2.49/h, same network volume) or two GPUs, served as two packages on different devices or two
  server processes; `--left-package` / `--right-package` then select them. Modelled at 1.46 s: 41% motion per arm and
  40% simultaneous. Not changed on the pod (that is the orchestrator session's call).
- Faster inference also helps a shared model. With guidance 1.0 and compile (0.63 s per chunk measured on the pod by
  the orchestrator session; effect on action quality still being tested), the round trip would be about 0.93 s, below
  the 1.07 s of motion, so each arm's cycle becomes max(1.07 + 0.93, 2 x 0.93) = 2.0 s: about 53% motion per arm with
  partial overlap (model, not measured).

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
