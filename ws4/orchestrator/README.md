# ws4/orchestrator: Claude-driven orchestration of the two XLeRobot arms

The operator types a long task. Claude (or an offline stub) turns it into a plan of per-arm FLUX skills from a fixed
vocabulary; a human approves it in a localhost HMI; a DAG executor runs it on both arms, concurrently where the plan
allows, verifies every step with a vision check, retries or replans, and logs the attempt for the graded evaluation.
Design, plan format, measurements and open items: [`../research/orchestrator.md`](../research/orchestrator.md).

## Digital twin and the public live lab (branch ws4-orchestrator-twin)

- **Twin in the HMI.** The side column shows a "Digital twin" panel: main's MuJoCo scene
  (`examples/dual_so101_sim/research_replay/scene.xml`) posed every 50 ms at both arms' measured joints
  (`GET /api/twin/state`; `orchestrator/twin.py`, `mj_forward` only). It is drawn with the live lab's own viewer
  (`site/viewer.js`, served under `/twin/`). The scene names its arms in Haller's operator frame, so the robot's LEFT
  (pen) arm drives the scene's `right_` arm. Only the arms are mirrored; the table, box and objects are the scene's
  estimate. The header line shows each arm's mode (policy, ramp, holding, idle).
- **"Run trial" on the live lab runs the orchestrator.** Start the HMI as usual (the unguarded launch command), then
  the companion in bridge mode, from the repository root:

  ```bash
  PYTHONPATH=src python -m liminal research --orchestrator http://127.0.0.1:8080 --site-url https://javgilavi.github.io/liminal/
  ```

  Open http://127.0.0.1:18003/pair and follow the link to the paired full-screen lab. Its twin follows the
  orchestrator's measured joints (about 30 Hz), its cameras are the orchestrator's head and wrist frames, and **Run
  trial** sends the instruction to the orchestrator: Claude plans it and, with "Execute on the real robot" and the
  acknowledgement checked, the companion approves the plan, so the chain runs exactly as from the HMI. Without
  execution the plan only appears in the HMI. **Stop this trial** latches the orchestrator's e-stop (both arms hold).

## FINAL demo: the 4-skill chain on the Haller, both arms (2026-10-18, run this)

Oscar's plan: the robot RIGHT arm opens the closed box by pulling its handle, then the robot LEFT arm puts the green
pen in the green slot, the yellow cutter in the yellow slot and the red pen in its slot. Vocabulary: `skills.yaml`
(version 3, the default). One process holds both arms: robot left = `/dev/haller_arm_right` (calibration
haller_right, wrist camera USB 7.2), robot right = `/dev/haller_arm_left` (haller_left, wrist camera USB 7.3), head =
`/dev/haller_cam_mast`. One LoRA per skill on the 8790 demo server under fixed names (`package:` in skills.yaml):
`skill_open_box`, `skill_green_pen`, `skill_yellow_cutter`, `skill_red_pen` (ws4-datasets maps each to a checkpoint in
`/workspace/serve/demo_packages.txt` on the volume); the session refuses to start if one is not served. `--server`
defaults to `http://127.0.0.1:8790` on the haller.

```bash
# 0. One process per arm: nothing else may hold either arm (this must print nothing).
fuser /dev/haller_arm_right /dev/haller_arm_left
# 1. Tunnel to the 8790 demo server (pod host and SSH port from ws4-datasets / the RunPod console).
ssh -N -L 8790:127.0.0.1:8790 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519_osrdvs root@<pod-ip> -p <ssh-port>
# 2. Observe-mode dry run on the real Haller: torque is never enabled and nothing is sent (connect writes no
#    register, no ramp runs, FLUX commands are computed and logged only). Checks the cameras, the plan, the lid and
#    pen checks, the start gate's verdict with the box arm's live pose before each pen step, the imagination panel.
#    Attempts go to runs/trials_mock.jsonl, never to the graded log.
cd ws4 && python -m orchestrator hmi --open --env haller --observe --planner claude --imagine lazy \
    --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token"
# 3a. Rehearse ONE skill through the HMI. open_box alone: both arms folded at rest, type "open the box"; in the same
#     HMI session each pen skill can then be planned and run on its own while the box arm keeps holding the lid (an
#     arm a policy left stays "unknown" across runs: its next ramp asks the operator first). A pen skill alone, with
#     the box arm already holding the lid (handover from the haller HMI, below): add --hold-arm right and type e.g.
#     "put the yellow cutter in the yellow slot".
cd ws4 && python -m orchestrator hmi --open --env haller --hold-arm right --planner stub --operator oscar \
    --condition post_training --variant <variant> --checkpoint "<what skill_yellow_cutter serves>" \
    --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token"
# 3b. The full chain: both arms folded at rest with torque off, box closed, objects on the mat.
cd ws4 && python -m orchestrator hmi --open --env haller --planner claude --imagine lazy --operator oscar \
    --condition post_training --variant lora_chain --checkpoint "<the four skill_* checkpoints>" --layout L1 \
    --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token"
```

The goal box is pre-filled with the chain (`station.demo_goal`): press **Plan** (Claude: right arm first, then the three
left-arm steps; checked live, 6.4 s, USD 0.02), check the lanes and the numbered chain strip, **Approve and run**.

**What each step runs with (what ws4/single_arm learned on the real arm, 18 Oct).** Every chain skill's `keepout:`
names its kit, `ws4/single_arm/skills/<box|green|cutter|pen_lid>/workspace.json` (build_skill.py: the station zones
with the skill's approved overrides plus its demonstrated envelope); the `skill.json` next to it is read at load (its
verbatim task must equal the FLUX prompt, its `runner_args` and `other_arm_gate` are applied):
- box: right-arm midline +0.17 m (takes 7-74); green: pen-arm midline -0.23 m, 2 cm envelope trim near the box arm;
  cutter: midline -0.21 m, 1 cm box-arm clearance and a 4 cm **slow zone** (0.5 deg/tick, load freeze at 50 % for
  0.2 s, from its `runner_args`; policy commands only, as in run_single_arm); red pen: the lid-held takes
  (`pen_lid`), midline -0.15 m, torso margin 1.4 cm. Ramps use the same zones without the envelope.
- **Start gate** before every pen-arm attempt, before anything moves: gate.py with the box arm's LIVE pose (the loose
  joint range of box-skill end poses and the skill's recorded hold, then the geometric check: no envelope voxel inside
  the box arm, at most 1 % of the demo poses under the skill's clearance). A refusal ends the chain, failed, with the
  reason in red next to the run state and in the events ("REFUSED before it moved"); no retry, no replan, both arms
  keep holding their last command (the box arm keeps the lid). `python ws4/single_arm/gate.py report q1 .. q6` gives
  the verdict per skill for a box-arm pose. `open_box` has no gate: the executor tucks the pen arm itself, and its
  pose at the start of open_box is logged in the events and the trial notes against the box takes' range (not a
  check: the lift's sag could refuse a good tuck).
- **Box arm holds**: `open_box` has `finish: hold`: once its check passes, nothing drives the box arm again in that
  run (a planned step on it is refused before it moves) and its torque stays on when the session closes, even at its
  launch pose (run_single_arm `--hold-other-arm` semantics). `--hold-arm right` does the same from launch for a box
  arm handed over holding the lid: connect sets its goal to its present pose before torque goes on, nothing ever
  commands it elsewhere.
- **Runner settings behind the first real inserts** (FLUX red pen RAW 5k, 3/5): 1 deg/tick policy guard (`--max-step`,
  as eval_skill.sh), action horizon 16 (`--action-steps` / `WS4_ACTION_STEPS`), shoulder-lift stall 25 deg
  (`--lift-stall-deg` / `WS4_LIFT_STALL_DEG`; other joints 8), a ramp
  arrives within run_single_arm's `ARRIVE_TOL` (lift 8 deg, others 6), 120 s per attempt, and a ramp the keep-out
  refuses holds and then goes one joint at a time in an order the keep-out accepts at every step (once per ramp; no
  such order: e-stop, both arms hold). Chunk prefetch is wired (`--prefetch-ticks` / `WS4_PREFETCH_TICKS`) but OFF by
  default. The settings in force are in the events at start and in every attempt's trial notes.

**Box fallback: `--skill-backend open_box=replay`** (the FLUX box LoRA went 0/4 on the arm, 18 Oct): the box step
replays recorded take 72 (`replay:` in skills.yaml; `--replay-episode open_box=73` for another take) instead of FLUX,
through the same stack (1 deg/tick time warp with the gripper on its recorded timing, the box kit's keep-out with its
envelope, the freezes, `finish: hold`); its lid check runs once the take has ended. Place the box on
`ws4/single_arm/skills/box/layouts/P3_ep072_head.jpg` (the take's first head frame; the events name it). The step card
says `REPLAY scripted replay of recorded demo 72, not FLUX`, the events say `SCRIPTED REPLAY OF RECORDED DEMO 72, NOT
FLUX`, its attempts are logged with condition `scripted` and variant `scripted-replay-ep72` (never with the FLUX rows), and the chain's trial line notes and checkpoint name it; prefetch is
forced off (it would skip recorded steps).

**Cutter fallback: `--skill-backend put_yellow_cutter_in_slot=replay`** (last resort: the augmented FLUX cutter grasped
4/5 but inserted 0/5): recorded take 67 on the pen arm the same way, under the cutter kit's keep-out, start gate and
slow zone with the box arm held, warped at the slow step within the slow zone of the box arm's live pose. Place the
cutter as in `ws4/single_arm/skills/cutter/layouts/P3_ep067_head.jpg` (the take's first head frame; the session names
it in the events at start; demo-rehearsal's align refs in `~/haller_demo/align_refs/cutter` include it). The take
ends at home, so `finish: home` checks again at once. Both flags can be combined. `skill_open_box` need not be served. Add the flag to 3a or 3b; the pen skills stay FLUX.

**Handover from the haller HMI (pen-skill rehearsal with `--hold-arm right`)**, as single_arm README "Handover from
the HMI": Oscar supports the lid and both arms; `bash ~/haller_ws/scripts/quest-teleop/down.sh`; the ports must be free
(`fuser $(readlink -f /dev/haller_arm_right /dev/haller_arm_left) || echo free`); fold the pen arm onto its rest by
hand; start 3a. The event "right arm handed over holding (--hold-arm): held at [...]" means the box arm holds the lid
and Oscar can let go. A start gate REFUSED means the lid is held where the skill's demos never had the box arm:
support the lid, re-place the arm (`gate.py report` until ALLOWED), restart. At the end the box arm keeps holding the
lid with torque on, and the pen arm holds where its skill left it: support both before switching the power off.

What runs, so each skill starts from the state its demos had (`station.poses`, measured in the datasets): the pen arm
ramps to its tucked pose, the box arm to its home `[0, 0.8, 2.0, 0.4, 0, 99.7]`, and `open_box` runs until the lid
check passes; then the box arm FREEZES and keeps holding the lid open (the lid closes if it lets go; no ramp, the
runner holds its last command). Before each pen skill the lid check runs again (`precheck: open_box`); if the lid is
not open, the e-stop latches and every arm holds (Resume checks again, Abort ends). Then the pen arm ramps tucked ->
home `[0, 0, 0, 0, 0, 100]` with the holding box arm as a live collision capsule, its skill's package runs FLUX with
the exact task string, and a Haiku 5.5 check every 2 chunks decides. No scripted ramp ever leaves the box: the green
pen and yellow cutter takes end at home, so after their check passes the POLICY keeps running until the arm is back
within 10 deg of home, then the check runs again (20 s at most; otherwise the step counts and the arm holds); the red
pen step, the last one, ends on its check with the arm holding at the slot, as its takes end. A ramp from where a
policy left an arm (a retry, a policy that did not come home) farther than 15 deg from its target waits for the
operator: e-stop, check the path or move the arm by hand, Release, Resume. One arm moves at a time; ramps run under the
ramp limits, the keep-out without the demonstrated envelope and the load freeze. The parked <-> home ramps of the pen
arm were checked offline against all 65 recorded box-arm holding poses: no keep-out zone touched.
Safety per arm: soft limits from each calibration, the FK keep-out with each arm's `other_arm` zone following
the other arm's live pose every tick, stall (only for a driven arm: the box arm holding the lid lags its command by
design), keep-out (45 of 60 policy ticks) and load freezes; an e-stop freezes both arms' joints at their measured pose
but keeps each gripper's last command, so the lid stays held. The HMI shows the 4-step chain with per-step status, both arm lanes, and with `--imagine lazy` FLUX's
imagined head | wrist view next to the real one (ws4/imagine serving: no added latency, 0.12 s decode per chunk).

Checks (research/orchestrator.md, "Final chain"; re-run with `python -m orchestrator.verify_eval --snapshot <dir>
--dataset <name> --skill <skill>`): `open_box` 63/63 on 21 held-out takes (75/75 on the 25 it was written on), no
false pass; green pen 114/119 held out (2 false passes in 82); yellow cutter 91/91 on 31 takes, no false
pass; red pen 65/66 on the 22 lid-held takes (1 false pass: the pen pressed against the green pen, off its strip).

Tested without hardware (`tests/test_chain.py`, 20 tests): both fake arms, flux.remote's real HTTP server with one fake
package per skill under its fixed name and a token, the whole chain end to end (the fake box policy ends holding the
lid at the green takes' recorded hold; each policy under its kit's workspace and envelope, each ramp without it; the
start gate allowing every pen step with the box arm's live pose; the box arm held and kept torqued at close; horizon
16, lift stall 25 and the settings in every attempt's trial notes; one arm moving at a time, one trial line per
attempt), a gate refusal (box arm at the first red-pen session's hold) ending the chain before the pen arm moves, the
cutter's slow zone (0.5 deg/tick and the 50 % load freeze only under the cutter policy), the joint-by-joint ramp
fallback and its stop, the `--hold-arm` handover (never commanded, refused steps), prefetch through the gated runner,
kits that do not match failing the vocabulary, a closed lid holding everything before the pen arm moves, holds that
keep the grip, the live other-arm keep-out, the imagination panel against a fake imagining backend, and the CLI. Not
verified: anything on the arms or the pod; the right arm's start pose and table height (VERIFY with ws4-vr2 /
so101-zeroshot); the slow zone and the ramp fallback with real capsule geometry in the chain (unit-level on the real
keep-outs in ws4/single_arm, faked distances here).

Rehearsal without hardware: `python -m orchestrator hmi --open --env haller --robot fake --policy fake --planner stub
--verifier manual`. The fake policy never moves the box arm, so the start gate refuses the first pen step (the
refusal path); to rehearse a pen skill past the gate, start the fake box arm holding where the green takes had it:
`--hold-arm right --right-fake-start 12.13 -13.8 12.26 37.54 29.93 0.87`.

## Pen-only runs: the green Haller, one arm, the red pen (2026-10-17)

Task (final, `skills.yaml`): the robot's LEFT arm puts the red pen in its slot in the cardboard box and takes it out
again, on a chain Claude plans. The robot-RIGHT arm stays parked and holding; this process never commands it and only
reads its joints (read-only, torque untouched) before each FLUX attempt, for the keep-out's `other_arm` zone.
FLUX: the per-arm SO-101 package (zero-shot) or tonight's LoRA, served from the RTX PRO 6000 pod (`so101-flux-serve`).
The arm side is `ws4/single_arm/run_single_arm.py`, imported read-only: torque-safe connect, camera parity with the
VR recording (head unrotated, wrist rotated 180, default fourcc), joint and step limits, LeRobot's goal cap, the FK
keep-out, a 60% torque limit; the orchestrator adds its stall, keep-out and load-freeze stops every tick (load: 70% of
the torque cap for 9 ticks or 90% once, policy only), and every FLUX step first ramps to the demos' start pose
`[0, 0, 0, 0, 0, 100]` under the runner's ramp guard: ramp joint limits (1 deg from the stops, rest included) and the
station zones plus the parked arm only (`KeepOut.without_envelope`). The policy then runs under the soft limits
(calibrated range minus 5 deg, rest excluded) and all 9 keep-out zones, including the demonstrated envelope. One keep-out refusal on that ramp stops and holds, as `move_to` does; observe mode skips
the ramp (nothing is sent). Before the ramp it reads the parked arm's pose on `/dev/haller_arm_left`
(`keepout.read_arm_joints`) so the `other_arm` zone knows where it is. An unknown pose blocks the attempt (fail closed: the midline alone misses some
reaches toward that arm): if another process holds that port (someone may be driving the arm), or the read fails and
no pose younger than 180 s was read before, the e-stop latches with the reason, the arm holds at home and FLUX does not
start (nor the ramp); fix it, then **Release** and **Resume**. `--no-other-arm` skips the read for deliberate tests only and is
written into every attempt's notes (`other_arm=...`). Each step ends on a Claude Haiku 5.5 check of the head frame,
with the station prompt tuned on 282 recorded Haller frames (put 97.1%, take 98.2%, no false pass; research doc,
"Success checks on real frames"; re-run with `python -m orchestrator.verify_eval`).

```bash
# 0. One process per arm: stop the haller HMI and the VR recorder first (this must print nothing).
fuser /dev/ttyACM* /dev/haller_arm_right /dev/haller_arm_left
# 1. Tunnel to the policy server. Pod host and SSH port: ask the orchestrator session or read them in the RunPod console.
ssh -N -L 8765:127.0.0.1:8765 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519_osrdvs root@<pod-ip> -p <ssh-port>
# 2. Dry run on the real arm: observe mode, torque never enabled, nothing sent (check cameras, plan, checks).
cd ws4 && python -m orchestrator hmi --open --env haller --vocabulary orchestrator/skills_haller_pen.yaml \
    --task pen_chain --observe --planner claude \
    --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token"
# 3. The demo, zero-shot (graded attempts go to ws4/eval/trials.jsonl).
cd ws4 && python -m orchestrator hmi --open --env haller --vocabulary orchestrator/skills_haller_pen.yaml \
    --task pen_chain --planner claude --package so101 \
    --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token" \
    --condition zero_shot --variant A0 --checkpoint black-forest-labs/flux-3-action-so101@0713a7a --layout L1 \
    --operator oscar
```

### Chain run with the LoRA

The pod serves the LoRA under its own package name next to `so101` (the name comes from whoever starts the server
on the pod; the server's `GET /packages` lists them, with the token header as for every request). Only
`--package` and the logging fields change; the token, tunnel and arm are the same:

```bash
cd ws4 && python -m orchestrator hmi --open --env haller --vocabulary orchestrator/skills_haller_pen.yaml \
    --task pen_chain --planner claude --package <lora-package> \
    --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token" \
    --condition post_training --variant lora_A --checkpoint <hf-repo>@<rev> --layout L1 --operator oscar
```

Goals to type (the planner turns them into a chain; the stub planner, `--planner stub`, does the same offline):

| Goal | Plan |
|------|------|
| `put the pen in, take it out, put it back` | put_pen_in_slot, take_pen_out, put_pen_in_slot |
| `put the pen in its slot 5 times, I reset it each time` | put_pen_in_slot, then 4 x (operator_reset, put_pen_in_slot) |

`operator_reset`: the arm ramps home (gripper opened first), the step card shows **Done: Take the red pen out of its
slot and lay it on the mat**; Oscar resets the pen and presses it (headless `run`: Enter in the terminal). It is planned
work, not an intervention, so a chain with only resets still counts as a success without interventions. Prefer the
reset chain unless the LoRA saw robot take-outs: the recorded take-out dataset has ONE robot take-out in 11 episodes
(episodes 0 to 8 are the arm retracting while a hand takes the pen out; research/orchestrator.md).

Tested end to end without hardware (`tests/test_haller.py::test_lora_*`): flux.remote's real HTTP server on localhost
with a fake policy under a LoRA package name and a token, the fake arm, both chains, per-attempt trial lines with
`post_training` / `lora_A` / the checkpoint.

Then in the HMI, type the goal, for example `put the pen in the box, take it out, put it back` (or `... 3 times`),
press **Plan**, check the lane, **Approve and run**. **E-STOP** (button or spacebar) latches the runner and freezes the
arm at its measured pose; **Release**, then **Resume** or **Abort**. Keep a hand near the power cable as with every
single-arm run. At the end the arm keeps holding with torque on unless it is back at its launch pose and torque was off
at launch (same rule as `run_single_arm.py`).

Logging: one `trial_log.py` line per FLUX attempt (task = the skill: `put_pen_in_slot`, `take_pen_out`; steps 1/1 when
the check passed, the failure reason otherwise, interventions counted) plus one line per chain (task `pen_chain`,
steps done / total). Summary for the slides: `python eval/trial_log.py summary --markdown`.

Rehearsal without hardware: `python -m orchestrator hmi --open --env haller --vocabulary
orchestrator/skills_haller_pen.yaml --robot fake --policy fake --planner claude
--verifier manual` (run_single_arm's fake arm and a fake policy; mark steps done by hand; logs go to
`runs/trials_mock.jsonl`, never to the graded log).

## Run it

All commands from `ws4/` (they put `ws4/` on `sys.path`). `pip install -r orchestrator/requirements.txt` (Python 3.10+;
the Anthropic SDK is pinned to 1.15.0, the version the Claude calls were checked against). Headless rendering uses
`MUJOCO_GL=egl` (`osmesa` without a GPU).

**(a) Laptop, no GPU, no key: sim + scripted stand-in skills + stub planner**

```bash
cd ws4 && python -m orchestrator hmi --open
```

Opens http://127.0.0.1:8080/. Type a goal (the default is "put the apple and the yellow block in the box, then close
the lid"), press **Plan**, check the lanes, press **Approve and run**. **E-STOP** is the red button or the spacebar;
then **Release** and **Resume** (or **Abort**). Headless: `python -m orchestrator run --yes --fast`.
Sim attempts go to `ws4/orchestrator/runs/trials_sim.jsonl`:

```bash
python eval/trial_log.py --log orchestrator/runs/trials_sim.jsonl summary --markdown
```

**(b) Same with the real Claude planner.** The key comes from `ANTHROPIC_API_KEY`, else from
`~/.config/ws4/anthropic.env` (one line `ANTHROPIC_API_KEY=...`, mode 600; override the path with `ANTHROPIC_API_KEY_FILE`).

```bash
python -m orchestrator hmi --open --planner claude                       # Opus 5.5 plans, sim ground truth verifies
python -m orchestrator hmi --open --planner claude --verifier shadow --image-size 480x640
                                                                         # + Claude checks logged against ground truth
python -m orchestrator hmi --open --planner claude --verifier claude --image-size 480x640   # Claude decides
```

`--budget-usd` (default 5) caps the session's Claude spend; every call's tokens and estimated cost are in the HMI and in
`runs/<run>/claude_calls.jsonl`. Measured: a plan takes 12 s cold, 6 s warm (USD 0.024 / 0.013), a Haiku check 1.5 s
(USD 0.0001). To re-measure (about USD 0.04): `python -m orchestrator.live_check`.

**(c) FLUX on the RunPod server** (ask Oscar before using the pod; model load is about 6.4 min per start)

```bash
ssh -N -L 8765:127.0.0.1:8765 root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519_osrdvs    # ws4/cloud/RUNPOD.md
python -m orchestrator hmi --open --policy remote --package so101 --planner claude --verifier claude \
    --image-size 480x640                                                  # FLUX driving the sim arms (zero-shot)
python -m orchestrator.measure_concurrency --backend remote --left-package so101 --right-package so101 --warmup 20
```

Per-arm LoRAs: `--left-package lora_left --right-package lora_right` (package names as served by `flux/remote.py`).

**(d) Real robot** (only when Oscar says it is connected; same flags as `flux/run_xlerobot.py`)

```bash
python -m orchestrator hmi --open --env xlerobot --policy remote --planner claude --verifier claude \
    --left-port /dev/ttyACM0 --right-port /dev/ttyACM1 --head-cam <realsense-serial> \
    --left-wrist-cam /dev/video2 --right-wrist-cam /dev/video4 --calibration-limits \
    --condition zero_shot --variant A0 --checkpoint black-forest-labs/flux-3-action-so101@0713a7a --layout L1
```

Robot attempts go to the graded log `ws4/eval/trials.jsonl` (`python eval/trial_log.py summary --markdown`).

**Other commands**

```bash
python -m orchestrator plan --goal "..." [--planner claude] [--out plan.json]   # plan only, no motion
python -m orchestrator validate plan.json                                         # validate a plan (e.g. from WS3)
python -m orchestrator run --plan-file plan.json --yes --fast                     # run an external plan
python -m orchestrator schema                                                     # the plan JSON schema
python -m pytest orchestrator/tests -q                                            # 58 CPU tests, ~30 s, no network
ORCH_LIVE=1 python -m pytest orchestrator/tests/test_claude_live.py -q -s         # live Claude calls, ~USD 0.04
```

## Files

| Path | What |
|---|---|
| `skills.yaml` | The vocabulary: exact FLUX prompts, allowed arms, success checks, timeouts, verifier model, sim bindings. **Placeholder until the demo task is final.** |
| `plan.py` | Plan dataclasses, JSON schema, parsing and validation (vocabulary, arms, DAG, per-arm order, hold deadlocks), lanes, concurrency |
| `planner.py` | `ClaudePlanner` (structured output, feedback loop, at most 2 rounds) and `StubPlanner` (offline, deterministic) |
| `claude.py`, `cost.py` | One structured-output call with refusal fallbacks (doc URLs inline); token and cost ledger with a session budget guard |
| `live_check.py`, `requirements.txt` | One live call per Claude feature with latency and cost; pinned dependencies |
| `verifier.py` | `ClaudeVerifier` (Haiku 5.5 or Sonnet 5.5 per skill), `SimTruthVerifier`, `ShadowVerifier`, `ManualVerifier` |
| `verify_eval.py` | offline accuracy, latency and cost of the success checks on recorded LeRobot frames (confusion matrix per variant) |
| `executor.py` | DAG scheduler and control loop: concurrency, verification cadence, timeouts, retries, replans, e-stop, make-way parking |
| `gate.py` | Inter-arm collision gate (chunk truncation, tick-level e-stop) on `kinematics/mj_collision.py` |
| `primitives.py` | Scripted `home`, `open_gripper`, `close_gripper` behind the FLUX backend interface; `hold` needs no backend |
| `sim_backend.py`, `scenes/pack_box.xml` | Laptop backend: pack-box scene, ground-truth predicates, scripted IK stand-ins for the FLUX skills |
| `session.py` | Wires a backend configuration (a) to (d), frames, plan approval, trial logging, run directories |
| `hmi/server.py`, `hmi/index.html` | Localhost HMI (stdlib HTTP + one page) |
| `__main__.py` | CLI |
| `measure_concurrency.py` | Two arms on one shared model vs two models, sync vs async |
| `tests/` | CPU tests; `test_claude_live.py` runs only with `ANTHROPIC_API_KEY` |

`runs/` (git-ignored) holds one directory per run: approved plans, `events.jsonl`, `claude_calls.jsonl`, `result.json`.
