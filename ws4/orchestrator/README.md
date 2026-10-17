# ws4/orchestrator: Claude-driven orchestration of the two XLeRobot arms

The operator types a long task. Claude (or an offline stub) turns it into a plan of per-arm FLUX skills from a fixed
vocabulary; a human approves it in a localhost HMI; a DAG executor runs it on both arms, concurrently where the plan
allows, verifies every step with a vision check, retries or replans, and logs the attempt for the graded evaluation.
Design, plan format, measurements and open items: [`../research/orchestrator.md`](../research/orchestrator.md).

## Sunday demo: the green Haller, one arm, the red pen (run this)

Task (final, `skills.yaml`): the robot's LEFT arm puts the red pen in its slot in the cardboard box and takes it out
again, on a chain Claude plans. The robot-RIGHT arm stays parked and holding; this process never commands it and only
reads its joints (read-only, torque untouched) before each FLUX attempt, for the keep-out's `other_arm` zone.
FLUX: the per-arm SO-101 package (zero-shot) or tonight's LoRA, served from the RTX PRO 6000 pod (`so101-flux-serve`).
The arm side is `ws4/single_arm/run_single_arm.py`, imported read-only: torque-safe connect, camera parity with the
VR recording (head unrotated, wrist rotated 180, default fourcc), joint and step limits, LeRobot's goal cap, the FK
keep-out, a 60% torque limit; the orchestrator adds its stall and keep-out stops every tick, and every FLUX step first
ramps to the demos' start pose `[0, 0, 0, 0, 0, 100]` through the same guard, checked against the station zones and the
parked arm only (`KeepOut.without_envelope`; the demonstrated envelope bounds the policy, and unfolding from the folded
rest leaves it for a moment). One keep-out refusal on that ramp stops and holds, as `move_to` does; observe mode skips
the ramp (nothing is sent). Before the ramp it reads the parked arm's pose on `/dev/haller_arm_left`
(`keepout.read_arm_joints`) so the `other_arm` zone knows where it is. An unknown pose blocks the attempt (fail closed: the midline alone misses some
reaches toward that arm): if another process holds that port (someone may be driving the arm), or the read fails and
no pose younger than 180 s was read before, the e-stop latches with the reason, the arm holds at home and FLUX does not
start (nor the ramp); fix it, then **Release** and **Resume**. `--no-other-arm` skips the read for deliberate tests only and is
written into every attempt's notes (`other_arm=...`). Each step ends on a Claude Haiku 5.5 check of the head frame.

```bash
# 0. One process per arm: stop the haller HMI and the VR recorder first (this must print nothing).
fuser /dev/ttyACM* /dev/haller_arm_right /dev/haller_arm_left
# 1. Tunnel to the policy server. Pod host and SSH port: ask the orchestrator session or read them in the RunPod console.
ssh -N -L 8765:127.0.0.1:8765 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519_osrdvs root@<pod-ip> -p <ssh-port>
# 2. Dry run on the real arm: observe mode, torque never enabled, nothing sent (check cameras, plan, checks).
cd ws4 && python -m orchestrator hmi --open --env haller --observe --planner claude \
    --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token"
# 3. The demo, zero-shot (graded attempts go to ws4/eval/trials.jsonl).
cd ws4 && python -m orchestrator hmi --open --env haller --planner claude --package so101 \
    --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <ssh-port> root@<pod-ip> cat /workspace/flux_token" \
    --condition zero_shot --variant A0 --checkpoint black-forest-labs/flux-3-action-so101@0713a7a --layout L1 \
    --operator oscar
# 3b. With the LoRA: same command with the package name the pod serves it under and its HF revision, e.g.
#     --package <lora-package> --condition post_training --variant lora_A --checkpoint <hf-repo>@<rev>
```

Then in the HMI, type the goal, for example `put the pen in the box, take it out, put it back` (or `... 3 times`),
press **Plan**, check the lane, **Approve and run**. **E-STOP** (button or spacebar) latches the runner and freezes the
arm at its measured pose; **Release**, then **Resume** or **Abort**. Keep a hand near the power cable as with every
single-arm run. At the end the arm keeps holding with torque on unless it is back at its launch pose and torque was off
at launch (same rule as `run_single_arm.py`).

Logging: one `trial_log.py` line per FLUX attempt (task = the skill: `put_pen_in_slot`, `take_pen_out`; steps 1/1 when
the check passed, the failure reason otherwise, interventions counted) plus one line per chain (task `pen_chain`,
steps done / total). Summary for the slides: `python eval/trial_log.py summary --markdown`.

Rehearsal without hardware: `python -m orchestrator hmi --open --env haller --robot fake --policy fake --planner claude
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
