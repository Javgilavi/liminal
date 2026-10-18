# The liminal harness on the Haller (ws4 bridge)

The ws3 harness (`src/liminal`: behavior tree, task files, world state, perception) drives the green Haller through
the ws4 stack. **The tree decides what runs, when and on which arm; ws4 executes every motion.** Nothing in liminal
opens a serial port or sends a motor command on the Haller.

```
liminal run (behavior tree: inspect, fallback, WHILE right holds lid_open, ...)
  |  each skill node
  v
Ws4ChainBackend (src/liminal/ws4_bridge/backend.py)   refuses a prompt, arm or package the LoRA was not served for
  |  one-step ws4 plan
  v
ws4 orchestrator Session (one process owns BOTH arms + 3 cameras)
  executor: tuck/home ramps, lid precheck, start gate, FLUX policy, Haiku success check, finish: hold / home
  safety: per-skill kit keep-out + envelope, other arm as a live capsule, stall / load / keep-out freezes,
          torque cap 60 %, 1 deg/tick, cutter slow zone; e-stop = freeze with torque on, grippers keep their grip
  |  one request per chunk (8 states + 8 commands + 2 snapshots), horizon 16
  v
policy server :8790 (ws4/cloud/lora_adapter.py serve, SSH tunnel) -> skill_open_box | skill_green_pen |
                                                                     skill_yellow_cutter | skill_red_pen
```

After each ws4 step, the harness's own perception looks again (`ws4_haiku`: ws4's measured Haiku prompts, one per
predicate, on the head frame) so `SkillNode` verifies `success` on fresh evidence, not on ws4's word. While the pen
arm works, the WHILE node re-checks `lid_open` every 5 s on top of ws4's lid precheck.

## Run it on the Haller

From the repo root (`~/liminal` on the robot laptop, this branch checked out), same Python as the ws4 runs:

```bash
ssh -N -L 8790:127.0.0.1:8790 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519_osrdvs root@<pod-ip> -p <ssh-port>
```

```bash
export WS4_TOKEN_CMD="ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token"
```

```bash
fuser /dev/haller_arm_right /dev/haller_arm_left
```

(must print nothing: stop the ws4 HMI, recorder and runner first). The Anthropic key is found as for the ws4 HMI.
Dry run first, real arms with torque never enabled and nothing sent (FLUX and the checks still run):

```bash
LIMINAL_WS4_SESSION='{"observe": true}' PYTHONPATH=src python -m liminal run configs/tasks/assemble_components_haller.yaml --robot configs/robot/haller_ws4.yaml --backend ws4 --policy configs/policy/haller_ws4.yaml --perception configs/perception/haller_haiku.yaml
```

Then the real run (both arms folded at rest, torque off, box closed, objects on the mat as in the P3 overview):

```bash
LIMINAL_WS4_SESSION='{"operator": "oscar", "layout": "P3"}' PYTHONPATH=src python -m liminal run configs/tasks/assemble_components_haller.yaml --robot configs/robot/haller_ws4.yaml --backend ws4 --policy configs/policy/haller_ws4.yaml --perception configs/perception/haller_haiku.yaml
```

- **E-stop**: the ws4 HMI opens at http://127.0.0.1:8080 (red button or spacebar; Release, then Resume or Abort).
  Ctrl-C in the liminal terminal also e-stops. Every stop freezes both arms with torque on; the lid stays held. At
  the end, both arms keep holding (torque on) as after an HMI run: support the lid and each arm before any torque
  change (runbook section 5).
- **Box arm already holding the lid** (handed over, e.g. after a ws4 HMI run): add `"hold_arms": ["right"]` to
  `LIMINAL_WS4_SESSION`. The first inspection sees the lid open, so the tree skips `open_box` and holds the box arm.
- The task order is green pen, yellow cutter, red pen (`assemble_components_haller.yaml`), the order the takes were
  recorded in. Their `assemble_components.yaml` (green, red, cutter) would run the red-pen and cutter LoRAs out of
  distribution.
- Each FLUX attempt is logged to `ws4/eval/trials.jsonl` like an HMI run (variant `liminal_bt+lora_chain`); the
  liminal event log goes to `runs/<time>_assemble_components_haller.jsonl`, ws4's to `ws4/orchestrator/runs/`.

## The box replay and the Opus planner

- `open_box` runs as ws4's scripted replay of recorded box take 59 (`skill_backends: {open_box: replay}` in
  `configs/robot/haller_ws4.yaml`; the FLUX box failed 0/4 live). The liminal log and the skill result say
  "scripted replay of recorded demo 59, not FLUX"; ws4 logs it as condition `scripted`. Green, cutter and red pen are
  FLUX on 8790. Last resort for the cutter: add `put_yellow_cutter_in_slot: replay` (take 67).
- Their LLM planner (`LLMPlanner`: goal to task tree, compile errors fed back for repair) runs on Opus 5.5 with the
  same key as the ws4 orchestrator: `ANTHROPIC_API_KEY`, else `ANTHROPIC_API_KEY=...` in the file named by
  `ANTHROPIC_API_KEY_FILE` (default `~/.config/ws4/anthropic.env`). The key is never printed or logged.

```bash
PYTHONPATH=src python -m liminal plan "open the box, then put the green pen, the yellow cutter and the red pen in their slots" --planner claude --model claude-opus-5-5 --skills configs/skills_haller.yaml
```

  It prints a task tree that went through `parse_task`; the bridge still refuses any prompt, arm or package that is
  not served. The robot runs use the reviewed `assemble_components_haller.yaml`.

## Arm naming (read this before touching any robot config)

Everything here uses the **robot's own sides**: `left` = the pen arm = `/dev/haller_arm_right` (board 5B14030445,
calibration `haller_right`, wrist USB 7.2); `right` = the box arm = `/dev/haller_arm_left` (board 5B79018460,
`haller_left`, wrist USB 7.3). The udev names and the VR recorder use the operator frame (`left_*` = robot right).
The harness's `configs/robot/xlerobot.yaml` had board 5B79018460 as `left`: with the task files, the pen skills would
have driven the box arm. Fixed on this branch (b84ba76).

## What is verified (this branch, no robot, no pod)

- `tests/test_ws4_bridge.py`: `assemble_components_haller` end to end through the bridge on ws4's fake arms
  (run_single_arm.FakeSO101) with a fake policy behind flux.remote's real HTTP server: open_box on the right arm, then
  green, cutter, red on the left, each as a ws4 step with its kit keep-out, start gate, lid precheck; the box arm
  holds the lid from open_box on and keeps torque at close; the `liminal run` CLI with the shipped configs (only
  `ClaudeVerifier.check` is stubbed); a liminal e-stop mid-step freezes ws4 (rig hold) and aborts the run; direct
  motor commands and primitives on ws4 arms are refused; wrong prompt, wrong arm, unknown skill and a held arm are
  refused before ws4 moves anything.
- `tests/test_remote_engine.py`: liminal's own `FluxPolicyBackend` with `engine: remote`
  (`configs/policy/haller_remote_registry.yaml`, the four served packages) runs the unchanged `assemble_components`
  task on mock arms against the HTTP server; each package served only its own prompt.
- Their suite with the new tests: 357 passed, 3 skipped (uv env, Python 3.12, plus numpy and Pillow that the `dev`
  extra lacks); in the robot laptop's Python 3.13: 359 passed, 1 failed (`test_haller_feedback_integration`, which only
  runs where `haller_hmi` is installed and needs their unapplied `integrations/haller/command-feedback.patch`);
  `ruff check src tests scripts` clean; `liminal validate configs/tasks/*.yaml` all ok. ws4:
  `python -m pytest -q ws4/single_arm/tests ws4/orchestrator/tests` 152 passed, 1 skipped.

## Not verified

- Nothing ran on the Haller, its cameras, the pod or the real Claude API. Before a hardware run: the observe dry run
  above (ports, cameras, tunnel, token, packages, Haiku), then one real run with a hand on the lid.
- The WHILE re-checks `lid_open` with Haiku every 5 s while the pen arm works, and their WHILE treats one unclear
  verdict as a broken invariant (e-stop, both arms freeze, the run fails). Measured offline on the recorded pen takes
  (`ws4/eval/while_lid_refresh.py`: 53 takes, 785 checks at the live cadence with the policy 5x slower than the
  demos): 0 "closed" verdicts, but 74 unclear ones while the pen arm hides the box, which would break the WHILE in
  37 of 53 takes. So `ws4_haiku` keeps the last confident lid verdict through an unclear view for up to 45 s
  (`keep_s` in `configs/perception/haller_haiku.yaml`; 0/53 breaks at 30 s, longest unclear stretch 25 s). A confident
  "closed" still breaks the WHILE at once. ws4's own chain checks the lid only before each pen skill. Not measured
  on live policy runs, whose occlusions may last longer than in the demos.
- Timing: the tree gives each skill 240 s; the bridge runs one ws4 attempt per skill (`max_attempts: 1`, attempts
  time out at 120 s). The tree has no retry node; add `retry` in the task file to retry a skill.
- liminal's native SO-101 path (`configs/robot/xlerobot.yaml`) has none of the Haller keep-outs, start gate or load
  freeze; its e-stop switches torque off unless `estop: freeze`. Use the bridge on the Haller.
- Their Qwen perception path is unchanged and still unmeasured on Haller frames (CUDA, the laptop has a 4 GB T550).
- `liminal serve --backend ws4` (their web UI) should run (arms read from the ws4 session, e-stop maps to ws4's) but
  was not exercised; its manual jogs would latch the e-stop, since ws4 arms refuse direct commands.
