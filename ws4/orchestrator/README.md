# ws4/orchestrator: Claude-driven orchestration of the two XLeRobot arms

The operator types a long task. Claude (or an offline stub) turns it into a plan of per-arm FLUX skills from a fixed
vocabulary; a human approves it in a localhost HMI; a DAG executor runs it on both arms, concurrently where the plan
allows, verifies every step with a vision check, retries or replans, and logs the attempt for the graded evaluation.
Design, plan format, measurements and open items: [`../research/orchestrator.md`](../research/orchestrator.md).

## Run it

All commands from `ws4/` (they put `ws4/` on `sys.path`). Python 3.10+, `numpy`, `pyyaml`, `Pillow`, `mujoco` for the
sim; `anthropic>=1.15` only for the Claude planner and verifier. Headless rendering uses `MUJOCO_GL=egl`
(`osmesa` without a GPU).

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

**(b) Same with the real Claude planner** (export the key in your shell first; the code reads `ANTHROPIC_API_KEY` only)

```bash
python -m orchestrator hmi --open --planner claude                       # Opus 5.5 plans, sim ground truth verifies
python -m orchestrator hmi --open --planner claude --verifier shadow --image-size 480x640
                                                                         # + Claude checks logged against ground truth
python -m orchestrator hmi --open --planner claude --verifier claude --image-size 480x640   # Claude decides
```

`--budget-usd` (default 5) caps the session's Claude spend; every call's tokens and estimated cost are in the HMI and in
`runs/<run>/claude_calls.jsonl`.

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
python -m pytest orchestrator/tests -q                                            # 49 CPU tests, ~30 s
```

## Files

| Path | What |
|---|---|
| `skills.yaml` | The vocabulary: exact FLUX prompts, allowed arms, success checks, timeouts, verifier model, sim bindings. **Placeholder until the demo task is final.** |
| `plan.py` | Plan dataclasses, JSON schema, parsing and validation (vocabulary, arms, DAG, per-arm order, hold deadlocks), lanes, concurrency |
| `planner.py` | `ClaudePlanner` (structured output, feedback loop, at most 2 rounds) and `StubPlanner` (offline, deterministic) |
| `claude.py`, `cost.py` | One structured-output call with refusal fallbacks; token and cost ledger with a session budget guard |
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
