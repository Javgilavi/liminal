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
with both arms, not a plan); nothing on the real robot. Claude's checks on real camera frames: measured offline on
recorded Haller frames ("Success checks on real frames"). The skill vocabulary is a placeholder (the
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
  `[0, 0, 0, 0, 0, 100]` and `hold`. The robot-RIGHT arm stays parked and is never commanded; its joints are only read
(read-only) for the keep-out. `skills.yaml` now holds
  this vocabulary with a `station` block (available arms, home pose, notes for the planner); the sim pack-box vocabulary
  moved to `skills_sim_pack_box.yaml`.
- **Arm side**: `orchestrator/haller.py` imports `ws4/single_arm/run_single_arm.py` read-only (`connect_so101`,
  `SingleArmRobot`, `build_guard`, `KeepOutGuard`, `hold_here`, `warm_up`, `disconnect_so101`, `FakeSO101`) and the
  `keepout` module; `orchestrator/monitors.py` rebuilds its per-tick stall stop (8 deg for 15 ticks, arm joints only)
  and keep-out stop (45 consecutive refusals; a single refusal while ramping to the start pose, as `move_to` does). The
  inter-arm collision gate is off on this station (one moving arm, the other parked); the keep-out's midline and
  `other_arm` zones guard that side. Every FLUX step first ramps to the home pose at 1 deg per tick through the same
  guard, because every VR demo starts there. Ramps (also the `home` step) use `build_guard(ramp=True)` limits (1 deg
  from the stops, widened to the rest pose; the policy's soft limits, 5 deg inside and not widened since the policy
  starts at home, put lift at +-99.9 and elbow and flex at +-93.5 deg, which excludes the folded rest) and the keep-out
  without the envelope (`KeepOut.without_envelope`): the demonstrated joint and TCP envelope (`joint_env`, `tcp_env`) bounds what the policy
  may command, and the unfolding ramp from the folded rest `[0, -96, 97, 1, 0, 2]` leaves the demonstrated TCP workspace
  for a moment (refused at step 20 with the full keep-out, 0 refusals without the envelope); the policy runs under all
  9 zones. A load freeze (`monitors.LoadMonitor`, run_attempt's rule: 70% of the torque cap for 9 ticks or 90% on one
  tick, policy only) catches a servo pushing into an end stop at low lag. Observe mode skips the ramp, since nothing is
  sent. Before the ramp, `keepout.read_arm_joints` reads the
  parked arm's pose for the `other_arm` zone. That check fails closed (agreed with so101-zeroshot): a port held by
  another process, or a failed read with no pose younger than 180 s from this session, latches the e-stop before the
  ramp or FLUX starts; Resume retries. Each attempt's trial_log notes say how the pose was known (`other_arm=read`, `cached N s`,
  `DISABLED (--no-other-arm)`, `fake arm`). The port check scans `/proc` and sees only the same user's processes. A stop freezes the arm at its MEASURED
  pose (`hold_here`), as the single-arm runner does.
- **Logging**: one `trial_log.py` line per FLUX attempt (task = the skill, 1/1 or 0/1 with the failure reason and the
  interventions during that attempt) and one per chain (task `pen_chain`). The fake arm and observe mode log to
  `runs/trials_mock.jsonl`, never the graded log.
- **Verified (no hardware, no pod)**: on run_single_arm's `FakeSO101` with a fake policy, "put the pen in the box,
  take it out, put it back" plans as three chained left-arm steps; the arm is within 3 deg of home each time a FLUX
  policy starts; a failed attempt (timeout) is logged and retried; a monitor stop latches the e-stop and calls the hold;
  the parked arm's pose is read once per FLUX attempt, before the ramp; an unknown pose (port held, stale cache) blocks
  the attempt until Resume; a keep-out refusal on the ramp stops before FLUX starts; the ramp from the folded rest
  passes with the real `haller_right_workspace.json` (station zones) while the policy keeps the envelope; observe mode
  starts FLUX without a ramp; the policy and ramp limits built from the real `haller_right.json` match run_single_arm's;
  the load freeze stops the policy but not the ramp (`tests/test_haller.py`, 19 tests, `read_arm_joints` faked). Live Claude plan for "put the pen in the box, take it out, put it back, and
  repeat the whole cycle twice": 14.2 s, USD 0.030, 8 steps, left arm only, pen states consistent (it inserted a
  take-out between cycles) and a final `home`.
- **Not verified**: anything on the real arm (VR recording holds it tonight) and the policy server on the RTX PRO 6000
  pod for this run. Haiku's checks were measured OFFLINE on 282 recorded head frames ("Success checks on real
  frames" below: 97.1% / 98.2%, no false pass); live frames are uncompressed and the LoRA's motions may differ from the
  VR demos, so look at a few checks in the observe dry run.
- **Run command**: `orchestrator/README.md`, "Sunday demo".

## Final chain: 4 skills, both Haller arms (2026-10-18 night)

**What changed for Oscar's final demo** (right arm opens the box, then the left arm puts the green pen, the yellow
cutter and the red pen in their slots; `orchestrator/skills.yaml` version 3, the pen-only vocabulary moved to
`skills_haller_pen.yaml`):
- **Both arms in one process** (`haller.py`): one `SingleArmRobot` per arm from run_single_arm.py, merged by `PairRobot`
  into the runner's 12-D interface (head camera from the left arm's connection, each arm's own wrist camera: USB 7.2 /
  7.3, both rotated 180, from haller_ws' recorder config). `ArmsKeepOutGuard` applies run_single_arm's keep-out rule to
  each arm; with both arms connected, each arm's `other_arm` zone follows the other arm's measured pose every tick, so
  the bus read of a parked arm (and its fail-closed block) is only used by the pen-only vocabulary. Limits, ramp
  limits and the stall / keep-out / load monitors are per arm; the e-stop holds both.
- **Per-skill packages** (`--skill-package SKILL=PACKAGE`, one LoRA per skill on 8790) and per-skill keep-out
  envelopes (`keepout:` in the vocabulary; the red pen keeps its 84-take envelope, the new skills run under station
  zones until theirs exist). Warm-up encodes each prompt once per (arm, package).
- **Transitions from the demos' start states** (`station.poses`, measured in the datasets; parent's corrected chain,
  2026-10-18): `open_box` tucks the pen arm, ramps the box arm home and runs until the lid check passes; then the box
  arm FREEZES holding the lid open (no ramp: the runner holds its last command; the lid closes if it lets go). Each pen
  skill (re-recorded with the box arm holding the lid) first runs the lid check again (`precheck: open_box`): a closed
  lid latches the e-stop before the pen arm moves (all arms hold; Resume re-checks); then the pen arm ramps from its
  parked pose `[3, -83.6, 81.9, 82.8, -10.2, 0.9]` (box takes 7 to 74) to home with the holding box arm as a live
  collision capsule and runs its skill. No scripted ramp ever leaves the box (parent: it could hit it): green and
  yellow (`finish: home`, their takes end at home) let the POLICY bring the arm back within 10 deg of home after the
  check passed, then check again (a pen dragged out fails and retries; 20 s at most, else the step counts and the arm
  holds); the red pen step ends on its check holding at the slot. Any ramp from where a policy left an arm (a retry, a
  policy that did not come home) farther than 15 deg from its target waits for the operator (e-stop; Release, Resume)
  on this station (`confirm_unknown_ramps`; off for the pen-only scene, where run_single_arm also ramps back). One arm
  moves at a time, ramps under the ramp limits, the station keep-out (other arm's capsules included) and the load
  freeze. Offline check of the parked <-> home ramps at 1 deg/tick against all 65 recorded box-arm holding poses: no
  keep-out zone touched, inside the calibrated ramp limits.
- **Holds keep the grip**: an e-stop freezes each arm's joints at the measured pose (run_single_arm's hold_here) but
  keeps each gripper's last command, so the box arm keeps squeezing the lid's tab (and a carried pen is not dropped);
  the stall stop only watches arms that a target drives (the holding box arm lags its command under the lid's weight by
  design); the load freeze watches every arm.
- **Keep-out stop** now so101-zeroshot's rule: 45 refusals among the last 60 policy ticks, "waiting" holds neither count
  nor clear (the consecutive count never fired with FLUX's chunks).
- **HMI**: a numbered chain strip with per-step status above the two arm lanes, the goal pre-filled from
  `station.demo_goal`, and with `--imagine lazy` an imagined-vs-actual panel (`imagination_panel.py`: for each new chunk
  the server's lazily decoded imagination is fetched in the background and the frame imagined for the next replan,
  +32 steps = 1.07 s ahead, is shown next to the real head | wrist pair recorded closest to that moment).
- **Verified offline**: `tests/test_chain.py` (9 tests): the whole chain end to end on both fake arms against
  flux.remote's real HTTP server (one fake package per skill, token): the exact sequence of 12 ramps and skills, never
  two targets moving at once, the box arm frozen from the end of open_box on, the lid prechecked before each pen
  skill, each pen skill at home on its own package and prompt and checked twice (before and after parking), one trial
  line per attempt; a closed lid holding everything before the pen arm moves, then Resume; the final tuck; the lid-falls-back retry; the live
  other-arm keep-out; the imagination panel against a fake imagining server; the CLI. The Claude planner plans the chain
  with the right arms (live: 6.4 s, USD 0.020).

**Success checks for the new skills** (`verify_eval.py --dataset <name> --skill <skill>`, the same method as the pen:
start frames = not done, end frames of kept takes = done, rejects by note, by-eye mid frames = not done, measured
exactly as `ClaudeVerifier` sends it):

| Skill | Takes so far | Frames | Live prompt | Generic prompt | Status |
|-------|--------------|--------|-------------|----------------|--------|
| open_box: "The lid of the box is open." | 25 (01:30) + 21 held out (takes 25 to 48) | 75 + 63 | 75/75 and 63/63 held out, no false pass, 1.3 s median, USD 0.00015 | 58/75 and 45/63 (71 %) | done |
| put_green_pen_in_slot | 6 (tuned) + 37 held out (takes 6 to 45) | 18 + 119 | held out: 114/119 (96 %), 2 false passes in 82 not-done frames | 39/56 (passes no end) | shipped; see below |
| put_yellow_cutter_in_slot | 31 (prompt written before any frame was seen: all held out) | 91 | 91/91, no false pass, 1.2 s median, USD 0.00014 | 15/21 (first 7) | done |
| put_pen_in_slot | 22 lid-held takes (new scene, all held out) | 66 | 65/66, 1 false pass (take 18: pen against the green pen, 1.5 cm off its strip) | 48/66 | done (prompt unchanged) |

open_box in detail (`research/verify_eval_open_box_2026-10-18.json`): frames = start (closed), 35 % (the arm at the
lid, still closed; checked by eye on all 25), end (kept takes open; rejects 3, 7, 16, 19, 21 end closed). The scene
prompt written on takes 0 to 4 scored 73/75 on all 25 (held-out odd takes 35/36), no false pass; both misses said the
lid was "not visible" (it swings back out of the frame), as did all 17 misses of the generic prompt. The hint now
says the lid may be out of the picture: 75/75 on those, and **63/63 on takes 25 to 48, which it never saw** (21 takes:
start, 35 % and end; fragments 28, 46, 47 left out; take 41 ends closed and was not passed), no false pass, median
1.36 s; the generic prompt got 45/63 there. Not covered: half-open lids (no labelled frames). The review notes call the box takes' intended end state
"the box arm holding the lid open", and the lid closes if the arm lets go (take 3 shows it):
the corrected chain (parent, 2026-10-18) keeps the box arm holding it, and checks the lid again before each pen skill.

put_green_pen_in_slot (first 6 takes, box arm holding the lid): frames = start (pen on the mat), 35 % (pen being picked
up from the mat, slot empty; checked by eye), end (pen standing in the green-strip holder). The generic prompt passed
none of the 6 ends ("lies flat", "no green slot visible"). The scene prompt says the pen in its slot covers the green
strip and that seating cannot be judged from this camera: 4/6 ends passed, the other 2 said done at confidence 0.6
(threshold 0.7 kept: a miss only costs another check two chunks later, a false pass stops FLUX early), every not-done
frame was answered not done at 0.7 to 0.9. Held out on takes 6 to 24 (take 14 a hand in the box: edge case; take 18
ends with the pen lying diagonally across the slot: not done, and not passed): 53/56 (15/17 done, 38/39 not done), one
false pass: take 17 at 35 %, the pen still in the gripper over the mat, the EMPTY slot's vertical green strip taken
for a standing pen. Two fixes were tried and dropped: (1) a hint warning about that strip: no false pass on takes 6 to
24 but on the clean takes 25 to 45 only 54/63, and its misses described the pen in its slot and still answered
done=false (too skeptical); (2) a location answer (`verify_states`: the model names where the pen is, done = "standing
in the green slot"): it read the empty slot's strip as the pen on most start frames (17 to 22 false passes per set).
Shipped: the plain prompt, clean on takes 25 to 45: 61/63, one false pass (take 44's end: the pen stands BESIDE the
slot; a yes / no check from this camera cannot tell), one miss at confidence 0.6. Held out over takes 6 to 45:
114/119, 2 false passes in 82 not-done frames, median 1.3 s, USD 0.00018 per check. Sonnet 5.5 on takes 6 to 24: the
same accuracy, median 2.0 s, USD 0.0032 per check (18x Haiku): Haiku stays. A false pass with the pen still in the
gripper costs one attempt (the arm lets go, parks, the re-check fails, the step retries); a pen left beside the slot
is not caught.

put_yellow_cutter_in_slot: the "yellow slot" is the LOWER WHITE rectangle in the box, left unpainted on purpose (Oscar);
the check is written around "the yellow cutter lies in the lower white slot of the box" and says the green pen standing
in its painted slot says nothing about the cutter. Frames: start (cutter on the mat), 35 % (the gripper taking it from
the mat; checked by eye), end (cutter flat in the lower white slot). The FLUX task string stays "put the yellow cutter
in the yellow slot". All four task strings are now confirmed by the recordings. Takes 0 to 30: 91/91 correct, no false
pass (takes 29 and 30 end with the cutter still on the mat and were not passed; rejects 10 and 27, the pen arm saved
over the box, scored as edge cases).

put_pen_in_slot on the new scene (`ws4/haller_put_the_red_pen_in_its_slot_lid_held`, 22 takes, box at P3, lid held by
the box arm, green pen and cutter already in their slots): the prompt tuned on the old pen scene, unchanged, scored
65/66 held out (20 ends in the slot passed; 46 not-done frames incl. reject 12, saved mid-carry with the pen in the
gripper over the mat: one false pass, reject 18, the pen pressed against the green pen 1.5 cm off its strip, the same
"beside the slot" case as old reject 100 and green take 44). Not retuned on these held-out takes for one subtle case.
All four chain checks are now measured on the final scene.

The new scene differs from the pen recordings: the pens and the cutter lie on the black mat, the box (closed at the
start, a tape handle on the lid's front edge) stands to its right and the box arm swings the lid up and back. The
open_box prompt describes that scene; the red pen's prompt still describes the old one.

## Success checks on real frames (2026-10-17 night, offline, Haiku 5.5)

**Result.** The check as first configured (generic verifier prompt plus the skill's sentence) almost never passes a
real "pen in its slot" frame: **2 of 107** true cases passed (accuracy 62%). With the tuned station prompt now in
`skills.yaml` it is **97.1%** for the put check and **98.2%** for the take check, with **no false pass** on 171 + 107
negatives, including 52 frames where the pen is still carried in the gripper. Held-out episodes: 97.8% and 99.1%.
Median latency 1.3 s and 1.2 s, about USD 0.00018 per check.

**Data.** Read-only snapshots of `ws4/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box` (104 episodes) and
`ws4/haller_take_the_red_pen_out_of_its_slot` (11), head camera, as recorded (H.264 640x480, same mast camera and
orientation as the live run). Per episode: the first and the last frame, plus the 50% frame of 52 put episodes (checked
by eye: pen in the gripper in the air, slot empty). Truth: put start = pen on the mat, put end = pen in its slot, take
start = in the slot, take end = on the mat; the review.json rejects by their note, every frame checked by eye. Two
labels disagree with the data and were set from the frames: rejects 15 and 16 ("ends with the pen on the mat, head
frame at save") end, in the video, with the pen standing in its slot (scored as edge cases), and take episode 10
already starts with the pen on the mat. Edge cases scored apart: a hand in the box (43), the pen in its slot but never
released (61), 15 and 16. Prompt tuning used only even episodes (dev); odd episodes are the held-out test.

Confusion counts at the shipped threshold (confidence >= 0.7), all frames (held-out odd episodes in brackets):

| Check | Variant | TP | FP | TN | FN | Accuracy | Median / p90 latency | USD per check |
|-------|---------|----|----|----|----|----------|----------------------|---------------|
| put: "standing in its slot" | generic prompt (before) | 2 (1) | 0 (0) | 171 (83) | 105 (53) | 62.2% (61.3%) | 1.20 / 1.74 s | 0.00012 |
| put | **station prompt (shipped)** | 99 (51) | 0 (0) | 171 (83) | 8 (3) | **97.1% (97.8%)** | 1.29 / 2.46 s | 0.00018 |
| take: "lying on the mat, outside the box" | generic prompt (before) | 96 (45) | 0 (0) | 107 (54) | 22 (12) | 90.2% (89.2%) | 1.07 / 1.45 s | 0.00012 |
| take | **station prompt (shipped)** | 114 (56) | 0 (0) | 107 (54) | 4 (1) | **98.2% (99.1%)** | 1.15 / 1.73 s | 0.00016 |

Why the generic check failed, from Haiku's own observations: the head camera looks down at an angle, so a pen standing
in its slot and a pen lying on the mat pointing at the camera both look like a vertical stripe ("appears to be lying
horizontally rather than standing", "standing upright on the mat, not lying flat"), and at the end of a put the gripper
still touches the pen ("held by the gripper rather than clearly seated"). The station prompt (`station.verify_scene`)
describes the scene (mat, box, knife, green marker, the white slot immediately LEFT of the green marker), tells the
model to judge by WHERE the pen is, not by how vertical it looks, and each skill's `verify_hint` lists what counts as
done and not done (the gripper may still touch the pen; a pen right of the green marker or held above the box is not
in its slot). Edge frames: 15, 16, 43 pass the put check (pen in its slot), 61 (still gripped) stays below threshold,
and 100 (pen standing next to the slot) never passes the put check, the hard negative it is meant to be.

Variants measured (same frames, `verify_eval.py`; full table in `research/verify_eval_pen_2026-10-17.json`):
- a table crop upscaled to 896 px: no better overall (put 96.0%), slower (median 1.7 s), and it passed one frame of
  the pen hovering just above the slot at 0.7. Not shipped (`station.verify_crop` stays unset).
- adding the wrist frame: put 96.8%, take 98.7%, one false pass on a hovering frame. Not shipped.
- effort medium instead of low: no gain, 0.5 s slower.
- threshold 0.6: recovers most misses (Haiku says 0.6 when the gripper touches the pen) but lets the two hovering
  frames pass (put_021, put_028 at 50%). Kept at 0.7: a missed pass only means one more check 2 chunks later; a false
  pass stops FLUX before the pen is in.
Not measured: Sonnet 5.5 (cost), live (uncompressed) frames, frames from the LoRA's own motions. Cost of the whole
study: USD 0.65 for 3,896 checks.

**Two consequences for the chain run, both handled in the code now:**
- A check can pass while the gripper still holds the pen in its slot. The ramp to the start pose of the NEXT step now
  opens the gripper in place first and only then moves the arm (`PrimitiveBackend(release_first=True)` on the Haller),
  so the arm never drags the pen out of its slot.
- The take-out dataset is not robot take-outs: in episodes 0 to 8 the arm retracts and a person's hand takes the pen
  out; only episode 9 is a robot take-out, episode 10 has no motion. A LoRA trained on it has one robot demo of that
  skill. Hence the second chain: N x put-in with an `operator_reset` step (the arm waits at home, Oscar resets the pen
  and presses Done; planned, not an intervention).

Re-run: `cd ws4 && python -m orchestrator.verify_eval --snapshot <dir> --variants live --split test` (copy the datasets'
`meta/`, `review.json` and `videos/` to `<dir>` first; about USD 0.05 per variant on the test split).

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
