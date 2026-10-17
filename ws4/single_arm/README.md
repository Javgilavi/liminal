# ws4/single_arm: FLUX SO-101 zero-shot on one SO-101 follower (cup stacking)

Owner: ws4-single-arm (side session of the WS4 orchestrator). Everything here ran on the real arm on 2026-10-17
unless marked UNVERIFIED. Graded attempts live in `ws4/eval/trials.jsonl` (`task=stack_cups`, `condition=zero_shot`).

## Report

**Done and verified on hardware**
- Red SO-101 follower brought up on this laptop: CH343 board on `/dev/ttyACM0`, IDs 1 to 6 at 1 Mbps, STS3215
  model 777, firmware 3.10. Calibrated as `red_follower` (file archived in `calibration/`).
- The arm's servos are the **12 V STS3215** version (Oscar opened the arm and read the labels). On the 5 V brick the
  shoulder saturated at 92% torque about 11 degrees above the folded pose and tripped overload protection. On 12 V
  it tracks within 2 degrees at 6 to 9% load. The two attempts run at 5 V are logged as variant `A0-5V`.
- Cameras: scene `/dev/video6` ("Global Shutter Camera", 32e4:5234) fixed high-angle top-down; wrist `/dev/video4`
  (Jieli 5258:4a55) taped to the wrist. Both 640x480 MJPG at a measured 30 fps.
- `run_single_arm.py`: one SO-101 driven by `ws4/flux` (`DualArmRunner`, one per-arm target, the second slot is a
  never-commanded phantom) against the RunPod policy server through `flux.remote.RemoteBackend`, with four safety
  layers, per-attempt video and trace, and automatic `trial_log` entries. 4 CPU tests.
- Zero-shot A0 (released package, unchanged) on the dedicated RunPod pod (RTX PRO 6000, guidance 1.0 + compile):
  0.45 to 0.6 s per chunk round trip. Results: see "Zero-shot results".
- A0r statistics (our own normalization) from 3 hand-guided episodes, handed to ws4-datasets to serve as `so101r`.

**Not verified / open**
- A0r on the robot: `so101r` is served on the pod (port 8766, built from revision 0713a7af, quantiles read back
  and matched), but the arm needs its elbow servo repaired or replaced first.
- Guidance 1.0 vs the package default 3.0 on motion quality (the pod serves 1.0 only).
- Whether the wrist camera orientation matches the SO-101 corpus (it is taped on; the jaw shows at the bottom).

**Next 3 steps**
1. Repair or replace the elbow servo (12 V STS3215 C047; set ID 3 with only that motor connected, then recalibrate
   `red_follower`), rerun `probe_bus.py` and the lift test, then run A0r with the same protocol (5 attempts per
   prompt over L1 to L3, `--server http://127.0.0.1:8766 --package so101r --variant A0r`).
2. Record 40 to 60 teleop demos on this arm (leader arm or the VR stack adapted to one `so_follower`), same
   calibration, same cameras, for the post-training LoRA.
3. Train the LoRA from the A0r-normalized base and evaluate it under the same protocol (`condition=post_training`).

## Hardware as used

| Item | Value |
|---|---|
| Arm | SO-101 follower, red PLA, stock SO-101 gripper, 6x STS3215 **12 V** (labels read by Oscar) |
| Supply | 12 V brick (measured 12.2 to 12.5 V idle, 10.6 V minimum during FLUX attempts). A 5 V brick is wrong for these servos |
| Controller | CH343 USB serial `1a86:55d3`, `/dev/ttyACM0`, board serial 5AE6053278 |
| Calibration | `red_follower`: `~/.cache/huggingface/lerobot/calibration/robots/so_follower/red_follower.json`, copy in [calibration/red_follower.json](calibration/red_follower.json) (sha256 `a8e9f8d1...`) |
| Rest pose | folded, [calibration/red_follower_rest.json](calibration/red_follower_rest.json); lift and elbow sit on their end stops |
| Scene camera | `/dev/video6`, fixed about 50 to 70 cm above the table, steep top-down view: both cups centred, gripper enters from the bottom ([frames/scene_final_L1.jpg](frames/scene_final_L1.jpg)) |
| Wrist camera | `/dev/video4`, taped to the wrist |
| Policy server | RunPod `so101-flux-serve`, RTX PRO 6000 96 GB, `flux.remote`, package `so101` = `black-forest-labs/flux-3-action-so101@0713a7af67...`, `guidance_scale=1.0`, `guidance_scale_action=1.0`, `compile_model=true` (read from the server's argv) |

## Commands, power-on to a FLUX run

All from `~/liminal` in the `(base)` conda env (LeRobot 0.5.1, `use_degrees=True`).

1. Power: check the servo labels first. 7.4 V STS3215 take a 5 V supply, 12 V STS3215 a 12 V supply.
2. Bus check, read-only:
   ```bash
   python ws4/single_arm/probe_bus.py --port /dev/ttyACM0
   ```
   Must print IDs 1 to 6 at 1000000 bps, every Present_Voltage inside its limits, and `OK`.
3. Calibrate once (no motion; middle pose, then sweep every joint except wrist roll end to end):
   ```bash
   lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=red_follower
   ```
4. Look at the joints and cameras with torque off:
   ```bash
   python ws4/single_arm/show_joints.py --once --frames ws4/single_arm/frames/check --scene-cam /dev/video6
   ```
5. Dry runs: fake arm and fake policy, then the real arm with torque never enabled:
   ```bash
   python ws4/single_arm/run_single_arm.py --robot fake --backend fake --seconds 5 --yes
   python ws4/single_arm/run_single_arm.py --observe --scene-cam /dev/video6 --backend fake --seconds 10 --yes
   ```
6. First torque test, shoulder pan only (scripted, about 10 degrees):
   ```bash
   python ws4/single_arm/run_single_arm.py --scene-cam /dev/video6 --backend fake --fake-velocity 0.1 0 0 0 0 0 --fake-gripper -1 --seconds 6
   ```
   Lift test (the one that exposed the 5 V problem): `--fake-velocity 0 0.5 0 0 0 0 --fake-delay 0 --seconds 4`.
7. Tunnel to the policy server (port from the pod; it changes on every restart):
   ```bash
   ssh -N -L 8765:127.0.0.1:8765 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519_osrdvs root@<pod-ip> -p <port>
   ```
8. Zero-shot attempts (the token is read by the runner over SSH, never printed or put in argv):
   ```bash
   python ws4/single_arm/run_single_arm.py --scene-cam /dev/video6 --backend remote \
     --token-cmd "ssh -i ~/.ssh/id_ed25519_osrdvs -p <port> root@<pod-ip> cat /workspace/flux_token" \
     --task "stack the cups" --task "put the left cup into the right cup" --attempts 2 --layout L1 --variant A0 \
     --max-step 1.0 --max-relative-target 20 --seconds 40
   ```
   Enter starts an attempt. Space, q or Enter freezes the arm; then r returns slowly to rest or x quits holding.
   s / f end early. After each attempt it asks for steps done (0 to 3) and interventions, and appends the trial.
9. If the arm is left holding somewhere: `python ws4/single_arm/run_single_arm.py --scene-cam blank --home`.

## Safety layers (outermost last)

1. `SafetyGuard` (`ws4/flux/safety.py`): joint limits from the calibration minus 5 degrees, widened to include the
   measured rest pose (it sits on the lift and elbow end stops), and 1 degree per tick (30 deg/s) on every joint,
   5% per tick on the gripper. Non-finite commands hold.
2. LeRobot's own `ensure_safe_goal_position` with `max_relative_target=20` degrees against the measured position.
   5 degrees starved the shoulder of torque at LeRobot's P=16 (1172 of 1200 ticks capped, arm stalled).
3. FK keep-out ([keepout.py](keepout.py), added after the elbow damage): every commanded target goes through forward
   kinematics (`ws4/kinematics`, stock flange TCP and the wrist point). A target is refused when it goes deeper than
   the measured pose into (a) the table plane plus 2 cm, (b) a 12 cm cylinder around the pan axis below shoulder
   height plus 5 cm, or (c) a fold-back where the gripper points back at the base within 25 cm of the axis. Refused
   ticks hold the measured pose, are logged per tick and counted in the trial notes; 1.5 s of refusals ends the
   attempt. Replays: the two damaging A0 attempts are refused at 0.97 s and 1.27 s (elbow load passed 50% at 7 s
   and 8 s); Oscar's clean demos have 0 refusals. `table_z` is UNVERIFIED until `keepout.py --touch`.
4. Stall freeze: any arm joint more than 8 degrees behind its command for 0.5 s ends the attempt, and every stop holds
   at the MEASURED pose (added after the elbow damage).
5. Servo `Torque_Limit` 60% on the arm joints (RAM register, resets at power-off); LeRobot already sets 50% on the
   gripper. Softens table or cup collisions now that the 12 V arm is strong.
6. Operator keys and `touch ws4/single_arm/STOP`; any bus or server error freezes the arm instead of crashing; an
   arm that is not at rest is never dropped (torque stays on, with a warning). A hand near the power cable.

## Zero-shot results

Protocol: three fixed layouts (L1 as in [frames/scene_final_L1.jpg](frames/scene_final_L1.jpg); L2 both cups about
10 cm to the robot's left; L3 cups one behind the other in line with the arm), two prompts alternating ("stack the
cups", "put the left cup into the right cup"), 40 s per attempt, start and end at the folded rest pose. Steps:
grasp a cup, lift it clear of the table, release it inside the other cup. Every attempt has a video and a trace
under `runs/`, linked from the trial log.

| task | condition | variant | success | steps done (mean) | human help | top failures | checkpoint |
|---|---|---|---|---|---|---|---|
| stack_cups | zero_shot | A0 | 0/3 (0%) | 0.0/3 | 3/3 | moved away from the cups; operator stop to avoid collision x3 | black-forest-labs/flux-3-action-so101@0713a7a |
| stack_cups | zero_shot | A0-5V | 0/2 (0%) | 0.0/3 | 0/2 | stopped before: grasp a cup x2 | black-forest-labs/flux-3-action-so101@0713a7a |

State on 2026-10-17 17:35 CEST: A0 halted after 3 attempts at 12 V because the elbow servo was damaged (see below).

### What happened (graded material)

- **A0-5V (2 attempts, not a FLUX result).** 12 V servos on a 5 V brick: the shoulder could not lift the arm.
- **A0 (released package).** FLUX drives the shoulder lift up at the guard's 1 deg/tick limit on every tick (191
  degrees of commanded lift travel in each of the first two attempts) and wrist flex into its +106 limit, while pan
  swings left. The gripper never approaches a cup; the operator stops each attempt before a collision with the table
  or the robot's own base. Both prompts give nearly the same trajectory (mean per-joint difference 6 to 20 degrees).
- **Why (diagnosis).** The package normalizes with one community dataset's statistics in the pre-2025 SO-100 degree
  convention (state q01..q99: lift 85 to 184, elbow 61 to 179; `ws4/flux/checkpoint_facts.md`). In today's LeRobot
  degrees our folded arm reads lift -100 and elbow 96, far outside that range, so the model sees an impossible state
  and pushes the shoulder regardless of prompt or image. This is a units/convention failure, not evidence about the
  model's skill on the task.
- **Elbow damage (17:30).** After the third 12 V attempt the elbow servo (ID 3) rattled with power off. The traces
  show the elbow pushing against an obstruction while the wrist folded toward the robot's base: 92% load with the
  overload status bit in attempt 2 (before the torque cap existed), and more than 8 degrees behind its command for
  9.8 s at its 60% cap in attempt 3; temperature stayed under 30 C. The 60% cap kept the load below the servo's own
  80% overload trip, so it kept pushing. Fixed in the runner: a stall freeze (any arm joint more than 8 degrees
  behind its command for 0.5 s) and every stop now holds at the measured pose.
- **A0 vs A0r offline (same observation, no robot).** Rest state and the first frames of A0 attempt 3, sent to both
  packages on the pod:

  | package | prompt | shoulder lift after 32 ticks | fastest per-tick step |
  |---|---|---|---|
  | so101 (A0) | stack the cups | -102 to +1.6 | lift 4.6, elbow 3.9, wrist flex 3.7 deg |
  | so101 (A0) | put the left cup into the right cup | -102 to -17.7 | lift 4.1 deg |
  | so101r (A0r) | stack the cups | -102 to -87.7 | lift 1.5, elbow 2.8 deg |
  | so101r (A0r) | put the left cup into the right cup | -102 to -93.6 | lift 1.3, elbow 2.3 deg |

  With our own statistics the runaway shoulder disappears; the first chunk is a modest lift and elbow unfold close
  to the 1 deg/tick guard. One observation from a 320x240 video frame, not a hardware result.
- **A0r.** Same weights, history quantiles from our own arm (3 hand-guided episodes, `handguided/`,
  `record_handguided.py --stats`). Wrist roll never moved in the demos, so its range is widened to 20 deg / 1 deg per
  tick. UNVERIFIED how close hand-guided deltas are to teleop deltas.

## Lessons for anyone running FLUX on an SO-101

- Read the servo voltage before choosing the brick. The bus does not tell you (7.4 V and 12 V both report model 777;
  ours read Max_Voltage_Limit 140), and 12 V servos on 5 V look like a weak, stalling arm, not a dead one.
- On this arm a `Goal_Position` write alone switches torque on. Set the goal to the measured position before enabling
  torque, or the arm jumps to a stale goal.
- One corrupted status packet aborted LeRobot's `enable_torque` (no retries by default). Use `num_retry`.
- The global shutter scene camera needs about 3 s before its first frame (LeRobot's warm-up default is 1 s).
- The released SO-101 package's quantiles assume legacy degrees. With today's calibration, re-normalize (A0r) before
  judging zero-shot behaviour, and use the same normalization as the base for a LoRA.
- Feetech overload protection (Overload_Torque 80, Protection_Time 200, Protective_Torque 20 on these servos) drops a
  stalled joint to 20% torque after about 2 s, which looks like a joint that freezes (status bit 32).

## Recipe for the teammate (your own station)

1. Check the follower's servo labels (7.4 V: 5 V brick; 12 V: 12 V brick) and plug in the matching supply.
2. `python ws4/single_arm/probe_bus.py --port <port>`: six IDs, voltages in range, `OK`.
3. `lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=<your_arm>` and keep that file for
   recording, training and deployment alike.
4. Mount a scene camera fixed high above the table looking steeply down (arm and objects in view) and a wrist camera;
   640x480 MJPG. Check both with `show_joints.py --once --frames ... --scene-cam ... --wrist-cam ...`.
5. Fold the arm into rest, run the fake dry runs and the pan test (steps 5 and 6 above, with `--robot-id <your_arm>`,
   `--port`, `--wrist-cam`).
6. Get the policy server port and token from the orchestrator, open the tunnel, run step 8 with your prompt.
7. Before zero-shot judging, record 3 minutes of hand-guided motion (`record_handguided.py`, torque off) and ask for an
   A0r package built from `record_handguided.py --stats`.

## Files

| File | What |
|---|---|
| `probe_bus.py` | read-only bus check: IDs, baud, voltage against each servo's limits |
| `show_joints.py` | joints in LeRobot degrees with torque off; optional camera frames |
| `run_single_arm.py` | the runner (modes: fake / observe / real, `--home`), recording and trial logging |
| `record_handguided.py` | torque-off hand-guided recording and A0r quantiles |
| `keepout.py` | FK keep-out zones (table, base, fold-back); `--touch` calibrates the table height |
| `joint_test.py` | one joint, small slow motion at a low torque cap: does the output follow (gear check)? |
| `calibration/` | `red_follower.json` (as used), `red_follower_rest.json` |
| `frames/` | camera frames Oscar confirmed (`scene_final_L1.jpg`) and earlier placements |
| `handguided/` | the hand-guided episodes and `stats_red_follower.json` |
| `runs/<session>/attempt<NN>/` | `video.mp4` (scene / wrist), `trace.npz` (state, command, guard per tick), `meta.json` |
| `tests/` | fake arm, local fake policy server, STOP file, stall freeze, LeRobot cap, bus errors; keep-out replays of the damaging attempts and the clean demos (`tests/fixtures/`) |

`ws4/flux` was used read-only; nothing in it was changed by this workstream.
