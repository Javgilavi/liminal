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

## Haller (XLeRobot tower): FLUX on the robot-left arm (red pen into its slot in a cardboard box)

The red arm was retired (elbow servo damaged). The team demo moved to Oscar's green Haller station; demos come from VR
teleop (ws4-vr2, haller_ws HMI), this workstream provides calibration, safety and the FLUX runs.

| Item | Value (measured 2026-10-17 unless marked) |
|---|---|
| Servos | STS3215 7.4 V C001, Min/Max_Voltage_Limit 4.0/8.0 V, supply measured 7.2 to 7.3 V. Never 12 V |
| Ports (haller ids are operator-frame) | `/dev/haller_arm_right` = robot LEFT arm (task arm), `/dev/haller_arm_left` = robot RIGHT arm |
| Calibration | read-only EEPROM harvest (`ws4/teleop/calib.py harvest`), no servo writes: `haller_right.json` (robot left), `haller_left.json` (robot right) in `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`, copies in `calibration/` |
| Scene camera | D455 colour `/dev/video12` (`/dev/haller_cam_mast`), YUYV 640x480 at 30 fps |
| Wrist cameras | Autodarts (OV2710), both report serial SN0001, so address them by USB path: robot-left (task) arm `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0` (was `/dev/video4`, sees the cup), robot-right arm `...usb-0:7.3...` (`/dev/video6`); mounted upside down |
| Camera parity | the VR recording (ws4-vr2, haller_ws `hmi/backend/config.flux-real.yaml` at `ad21419`, lines 94-103 and 109-122; `haller_hmi/cameras.py:58-62, 142-146, 166-168`) reads both cameras through LeRobot `OpenCVCamera(path, fps=30, 640x480)` with default fourcc, no colour gains and no V4L2 writes; the task wrist cam is rotated 180 deg (`cv2.ROTATE_180`, mounted upside down), the head is not. The runner matches with `--scene-fourcc none --wrist-fourcc none --wrist-rotate 180` (LeRobot `Cv2Rotation.ROTATE_180`, the same `cv2.rotate`) and no `--parity-side`. Takes before `ad21419` were unrotated and moved to `ws4test/` |
| Keep-out | `calibration/haller_right_workspace.json` from `keepout.py --haller left`, fitted to the VR demos: fin-ray TCP; table split by point (TCP z >= -0.005, since the pen is grasped lying on the mat with the TCP down to 0.002; wrist z >= +0.05, demos never below 0.124); station table z = 0 (UNVERIFIED until `--touch`); torso box + 2 cm; own-base cylinder 6 cm (turntable only) and fold-back zone off, because takes 17 and 20 grasp the pen 7 cm in front of the base with the gripper pointing back; mast; midline y >= -0.11 as a backstop (demos reach -0.074 measured / -0.084 commanded); the parked robot-right arm as link capsules (`ws4/kinematics/collision.py`) at its measured joints, re-read before each attempt with `--other-arm-port /dev/haller_arm_left`, 4 cm surface clearance (demos keep >= 8.5 cm). 0 refusals on all 24 good pen-in takes (layouts A and B, 15,833 steps) and the 9 pen-out takes (regression fixtures in `tests/fixtures/`) |

Torque-safe connect: observe mode never writes a register or changes torque; a real run only switches torque off when
calibration or registers differ, and refuses that while the arm holds a pose unless `--allow-torque-off`.

```bash
python ws4/single_arm/run_single_arm.py --observe --port /dev/haller_arm_right --robot-id haller_right \
  --scene-cam /dev/haller_cam_mast --scene-fourcc none \
  --wrist-cam /dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0 --wrist-fourcc none --wrist-rotate 180 \
  --keepout-config ws4/single_arm/calibration/haller_right_workspace.json --other-arm-port /dev/haller_arm_left \
  --start-pose 0 0 0 0 0 100 --backend fake --seconds 6 --yes
```

Every FLUX attempt on the Haller starts where the VR demos start: haller home `[0, 0, 0, 0, 0, 100]` in LeRobot
degrees (arm joints at their calibration middle, gripper open), reached by a slow ramp that runs through the joint
limits and the keep-out on every step (`--start-pose 0 0 0 0 0 100`); the target and the measured start pose go into
each attempt's `meta.json`. Home is clear of every keep-out zone (TCP at station (0.362, 0.137, 0.317)). Not the sim
park pose `HOME_DEG [0, -10, 10, 75, 0, 30]`. At session end torque is switched off only if it was off at launch.

**Task and split (19:15):** the recorded task is "put the red pen in its slot in the cardboard box" (exact string from
ws4-vr2's final dataset), possibly with a second dataset "take the red pen out of the box and put it on the mat". The
held-out split (last 3 to 4 takes of each layout, 10 in total) is taken from ws4-vr2's take log once recording ends;
the block-in-cup plan (episodes 17-19, 37-39, 56-59) was dropped with that task.

Only one process may hold `/dev/haller_arm_*`: stop the haller HMI first (`fuser /dev/ttyACM*` must be empty).

### Offline data-format diagnostic: which normalization for the LoRA base (2026-10-17, no robot)

Question: with the released SO-101 weights unchanged, which joint convention / normalization lets FLUX best predict
Oscar's own pen demos? Tool: [offline_diag.py](offline_diag.py), run on RunPod `so101-diag-4090` (RTX 4090 24 GB,
EU-CZ-1, Qwen on the CPU).

- Data: snapshot of `ws4/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box` at 19:52, episodes 0 to 9 (layout A,
  none held out by the final split), converted per arm (`convert_bimanual --to per_arm --side right --src-units
  degrees --task-suffix ""`). 200 frames sampled with a full 8-tick history (seed 0).
- Each request is built exactly as the runner builds it: states s[t-7..t], the command issued before each
  observation (recorded action a[k-1]), image snapshots at window ticks 0 and 7. The first 32 of the 42 predicted
  absolute commands are compared with the recorded actions a[t..t+31]. Guidance 1.0 (video and action), fixed
  `inference_seed` 42, `compile_model` off; 0.68 s per prediction.
- Variants (same weights): **A** package statistics on our LeRobot degrees; **B** A0r, quantiles recomputed on these
  episodes; **C** the experimental adapter of
  [flux-action issue #2](https://github.com/black-forest-labs/flux-action/issues/2) (model = scale x ours + offset,
  scales [+1, -1, +1, +1, -1] x 4095/4096, offsets [7.998, 90.220, 84.111, -4.482, -0.396] deg) with package
  statistics; **D** sign flips only (shoulder_lift, wrist_roll) with quantiles recomputed on the flipped data.
- Selection metric, as the orchestrator asked: moving frames only (the recorded chunk moves an arm joint by more than
  5 deg or the gripper by more than 15 %; 151 of 200). Direction agreement counts a joint where its recorded move
  exceeds 2 deg (gripper 5 %) and compares signs of predicted and recorded displacement over the chunk. Relative
  displacement error = median |predicted - recorded displacement| / |recorded displacement|.

| variant | MAE pan, lift, elbow, wrist flex, roll (deg); gripper (%) | direction agreement | median rel. displacement error |
|---|---|---|---|
| hold (repeat current command) | 1.9, 4.4, 4.7, 2.9, 2.0; 4.4 | - | 100 % |
| A package stats | 4.3, **19.9**, 10.1, 11.2, 3.7; 6.6 | 42, 57, 53, 58, 34, 78 % | 195, 498, 242, 291, 177, 56 % |
| B our quantiles (A0r) | 1.9, 3.2, 3.8, 3.7, 1.8; 8.4 | 80, 79, 68, 60, 73, 72 % | 46, 73, 86, 100, 68, 102 % |
| C issue-2 adapter | 4.4, **16.8**, 6.0, 10.1, 1.9; 6.5 | 41, 43, 70, 64, 77, 82 % | 208, 476, 127, 283, 56, 57 % |
| D flips + our quantiles | 1.9, 3.2, 3.9, 3.5, 1.6; 7.6 | **80, 83, 70, 62, 81, 80 %** | 52, 76, 87, 99, 58, 100 % |

What it says:
- **A and C fail for the same reason the red arm's A0 failed:** inputs land outside the package's normalization range
  and the model drives the shoulder (lift MAE 20 and 17 deg, displacement errors of 200 to 500 %), at chance-level
  direction agreement on lift. The issue-2 offsets belong to that reporter's calibration and do not transfer.
- **B and D anticipate the demonstrated motion zero-shot**: direction agreement 60 to 83 % on every joint (chance is
  about 50 %), lift and elbow errors below the hold baseline, displacement magnitudes still under- or overshot (about
  50 to 100 % relative error), wrist flex and gripper not better than holding.
- **D vs B:** D is ahead on direction for 5 of 6 joints (lift 83 vs 79 %, roll 81 vs 73 %, gripper 80 vs 72 %) and
  on wrist flex, roll and gripper MAE; B is ahead on pan displacement. One sample seed: the robustness runs (guidance
  3, second seed) are below.

Robustness (same moving-frame metric):
- Seed 1, 200 other frames (168 moving): B direction 86, 83, 70, 62, 83, 74 %; D 84, 81, 70, 60, 88, 70 %; MAE
  within 0.1 to 0.2 deg of each other. Across both seeds neither is consistently ahead.
- Guidance 3: not completed (stopped to free the GPU for variant E).
- Variant **E** (Oscar's mapping lift_model = 90 - lift, elbow_model = elbow + 90, package statistics), seed 0:
  MAE 4.3, 16.7, 5.9, 9.6, 3.2; 6.4; direction 41, 43, 69, 64, 33, 82 %; displacement error 205, 483, 134, 286,
  158, 54 %. Nearly identical to C: lift direction at chance and five times too large.

**Decision: the pen LoRA base is B** (package weights + history quantiles of the training takes,
`a0r_stats.py` + `flux.package_tools.renormalize_package`). B and D tie, and B needs no transform: the dataset trains
as recorded and deployment needs no `--scale`. D would need lift and roll negated in the training data and at
deployment (`--scale 1 -1 1 1 -1 1`, supported and tested). C or E would need runtime transforms and lose anyway.

### Real-arm zero-shot on the Haller (2026-10-17 evening, robot-left task arm)

Served from RunPod `so101-serve-4090` with [serve_lowmem.py](serve_lowmem.py) (the stock `flux.remote` server's
load peak does not fit 24 GB), guidance 1.0, 0.92 to 0.94 s per chunk round trip. Every attempt starts at haller
home through a station-zone ramp; policy commands pass the keep-out (station zones, parked-arm capsules, and for the
pen task the demonstrated envelope), the 1 deg/tick guard, the stall freeze and the 60 % torque cap. Soft limits
(5 deg from the stops) and the load freeze were added after the first cup runs. Attempts were scored from their
videos where the steps prompt got no digit. The prompt now takes a typed answer plus Enter in normal terminal mode,
drops keys pressed during the attempt and ignores Ctrl-C (tested in a pty).

| task | variant | attempts | result | what happened |
|---|---|---|---|---|
| put_pen_in_slot | A0-E (stock + Oscar's mapping) | 3 | 0/3 each | approached the pen; #3 closed beside it, then hovered with the pen between the open fingers when 40 s ran out |
| grasp_cup | A0-E | 3 | 0/2 each | never approached the cup (two runs under the pen envelope), or folded to the end stops and swung outward |
| grasp_cup | A0r-B | 3 | 0/2 each | #1 reached the cup (beside the fingers at 8 s), did not close, then ducked toward the torso, held by the keep-out; #2 and #3 (soft limits + load freeze on) drove wrist flex into its 93.5 deg soft limit and held it there for two thirds of the 60 s, gripper curled back toward the robot body, closed on nothing, cup untouched |
| grasp_cup | A0r-D (lift + roll flipped) | 2 | 0/2 each | lift ran backward to -48 and -71 deg (robot frame, demos -8..74) until the keep-out froze it (mast and base); wrist flex again at its 93.5 deg soft limit; #2 stopped by Oscar at 26.5 s |

**Is the shoulder lift mis-mapped in B? No.** In B #2 and #3 lift stayed between 4 and 62 deg, inside the
demonstrated -8..74, and never ran toward a stop. With lift flipped (D, same quantile recipe on the flipped data)
lift ran out of the demonstrated range to -48 and -71 deg and kept pushing until the keep-out held it: the closed-loop
runaway a wrong sign produces (E's flipped lift did the same). Offline agreed, weakly: on the 53 moving lift samples
where the joint was still or reversing (no momentum to copy), B predicts the lift direction 72 % of the time and D
62 %. So the issue-2 adapter's lift flip (legacy SO-100 lift = 90 - new lift) does not hold for this package with
A0r quantiles. The joint that runs away under both mappings is wrist flex: commanded at its 93.5 deg soft limit for
most of each attempt (B 1,192 and 1,177 of 1,800 ticks, D 1,372 of 1,800), beyond the demos' 78.5 deg maximum, with
the gripper curled back toward the robot. Offline flex direction is near chance for B and D alike (58 to 61 % on still
or reversing frames). Whether flex is a convention problem or the zero-shot model not knowing the cup is UNVERIFIED;
the LoRA on B is the next test.

Load measured on the Haller during these runs never exceeded 29 % of max torque, also when E folded the arm to its
end stops; the red arm's damaging attempts reached 92 % (uncapped) and a sustained 60 % (at the cap).


### Post-training eval on the Haller pen task (kit ready 2026-10-17 22:15, not yet run on the arm)

Protocol (orchestrator): for every LoRA checkpoint, 4 start layouts x 3 attempts = 12, from haller home, verbatim
prompt "put the red pen in its slot in the cardboard box", every guard, guidance 1.0 as served. Logged as task
`put_pen_in_slot`, condition `post_training`, variant `lora_<microsteps>`, steps done of 3 (reach, grasp, insert),
interventions. Baseline row: zero-shot A0r-B on the same layouts, 1 attempt each. Layouts and scoring rules:
[layouts/README.md](layouts/README.md) (references = frame 0 of the 12 held-out takes, attempt k of a layout uses
held-out take k).

1. When the orchestrator says the arm is ours (someone else used the robot): read-only preflight. It checks ports
   free, both arms' EEPROM against the calibration files and the files against the repo copies, the task arm's
   servos and supply, and the scene and wrist camera against reference frames (colour, wrist orientation, mast
   camera shift on the robot's base plate). No servo writes, no motion.

   ```bash
   python ws4/single_arm/preflight.py --ref-dir ws4/single_arm/frames/reference_2026-10-17_night
   ```

   It picks take 103 from ws4-vr2's lossless references (layout D, nobody in view; `--ref-take` for another).
   Without `--ref-dir` it compares against held-out take 20's start frames. Offline check: take 20 vs take 103
   gives brightness 0.97 and a base-plate shift of +0.9 / +4.9 px (the mast camera held still through the
   recording); a 180 deg rotation is flagged on both cameras. Do it with the arm at home if possible: the
   references are at home, and each eval attempt repeats the base-plate check at home (`start_check.jpg`).
2. Server (orchestrator plan, 17 Oct night): real-serve-6000 (`ssh root@157.157.221.177 -p 15377`, RTX PRO 6000
   96 GB). ws4-datasets runs `/workspace/serve/pen_serve.sh 005000` = one server on port 8790 with `base` =
   `/workspace/packages/so101_haller_pen_B84` (stock so101 + B quantiles of the 84 train takes) and `lora_5000` =
   base + the lora_pen84 EMA adapter at 5,000 microsteps (folded into the weights in memory at load by
   `cloud/lora_adapter.py`, same merge as `real_lora.sh merge`), guidance 1.0 + compile, pen prompt pre-encoded.
   Later checkpoints (10k, 15k, 20k) replace `lora_<N>` on the same port after about 5 min of downtime;
   ws4-datasets asks first and holds while an arm session runs. The two
   rows then differ only by the adapter. Token in `/workspace/flux_token` there. Port 8780 is ws4-imagine's test
   server: leave it alone.
3. One arm session, as Oscar approved (17 Oct night, "ok i ll follow your recommendations"): checks, then 5
   attempts each of `base` (variant `A0r-B84`, zero_shot), `median` (`A0r-M`, zero_shot) and the checkpoint
   (`LoRA-5k`, post_training). All rows use layouts A B C D A (held-out takes 20, 36, 86, 101, 21), 40 s each and
   the same prompt and guards. One command runs all three rows over one tunnel:

   ```bash
   POD_HOST=157.157.221.177 POD_PORT=15377 LORA=lora_5000 bash ws4/single_arm/eval_pen.sh session
   ```

   If `median` is not served yet when the session starts, the rows become B84, LoRA-5k, median. Before the median
   row the wrapper checks the server again and skips it with a message if it is still missing (then `eval_pen.sh
   median` later). Tested against the live server with `base` + `lora_500` and no `median`: both rows ran and
   median was skipped. `base` or the LoRA missing, or the server down, stops before the arm moves.

   The wrapper checks that the packages are served before the arm moves. Between the rows the arm parks and the
   wrapper waits for Enter; the next row then reconnects, ramps to home and repeats the home check (well under a
   minute, no server restart). The trial-log checkpoint strings are ws4-datasets' (the B84 package path; the adapter
   path on B84 with its microsteps). `baseline` or `lora` instead of `session` runs one row; `START_AT=<N>` resumes
   the first row; `DRY_RUN=1` prints the runner commands; `FAKE_ROBOT=1` rehearses against the real server with the
   fake arm (nothing logged, no arm port). `--imagine` stays off for these graded rows.

   **Median row (A0r-M)** (`eval_pen.sh median` runs it alone; `ROWS="baseline lora"` leaves it out):
   package `median` = `/workspace/packages/so101_haller_pen_M`, stock weights with the median statistics of
   current-calibration SO-101 datasets (multimodalart Space 09edebb). A teammate's own script "basically worked"
   with it; through this wrapper it gets the same layouts, budget, prompt, guards, checks and logging. It runs
   only if ws4-datasets serves `median` (the wrapper checks before the arm moves). How M meets our arm:
   - Gripper: M's gripper spans 0..40 (state q99 40.3, action q99 40.7), ours 0..100 with open = 100. The guards
     never fight it: gripper limits 0..100, 5 %/tick, and the keep-out, envelope, stall and load freeze ignore
     the gripper. But M never opens past about 41, and our home gripper (100) normalizes to +4.0, so the first
     chunk sees an out-of-range gripper history until the gripper closes to 41 (12 ticks at 5 %/tick, then the
     8-tick history refills: about 0.7 s). Two opt-in fixes, never both: `M_START_GRIP=40` (start inside M's
     range) or `M_GRIP_SCALE=0.4` (model = 0.4 x robot gripper: our 100 is M's 40, M's open 40 becomes our 100;
     variant `A0r-M-g0.4`). Default: start at 100 like every other row. The notes record which was used.
   - Arm: M's ranges come from other arms and tasks. Home wrist flex 0 normalizes to -1.6 (M's flex 23..96), our
     reach goes to lift 73 (M's q99 43) and roll 66 (M's q99 18). B84's ranges are our own demos' (flex -18..76,
     lift -25..73, roll -56..66).

   The wrapper refuses busy ports, opens the tunnel, checks the packages are served (6-D, before touching the arm).
   The runner then ramps to haller home and runs the **home check** (`home_check.py`): its live head and wrist
   frames and joints against ws4-vr2's at-home references (take 103): brightness, mast camera shift on the base
   plate, wrist orientation, the task arm's position in the head view (a different calibration draws home
   elsewhere) and joints vs the recorded home (6 deg). PASS asks Enter. Otherwise you type `go` to run anyway or
   `park`. It saves `home_check.jpg` / `.json` in the session folder. Offline on the recorded frames: the arm at
   home in layouts A to D overlaps the reference 0.74 to 0.92 (pass); a 25 px displacement gives 0.46 (flagged).
   Then the layouts run in blocks. Before each attempt the placement page (http://127.0.0.1:8899) runs until Enter.
   After each attempt the runner asks steps and interventions (typed, Enter), logs one line and returns to home.
   At the end it parks at rest with torque off. Quit with Ctrl-C at an Enter prompt; resume with
   `START_AT=<attempt> HOME_REF=none` (same layout and reference order). Attempt cap 40 s (`ATTEMPT_S`), 2.5x the
   longest held-out take (16 s).
4. Keep-out: `calibration/haller_right_workspace_final.json` = the station zones plus a demonstrated envelope from
   the 84 train takes of all four layouts (joints -32..58, -52..84, -77..74, -40..105, -77..82 deg; 1,574 TCP
   voxels). Replays: station zones refuse 0 of 43,015 steps over all 96 good takes, and the envelope refuses 0 steps
   of the 12 held-out takes. The old `haller_right_workspace.json` envelope came from layouts A and B only and would
   have blocked C and D. Wrist flex: only train take 13 (layout A) goes above the 93.5 deg policy soft limit (56
   frames, max 95), so the soft limits leave the demonstrated motion intact.

Tested without the arm: 8 unit tests (`tests/test_eval_kit.py`: layout plan, references, composite, base-plate shift
on real frames, parity flags, placement page, logged fields, home check pass and fail) and pty rehearsals with the
fake robot and a local fake server: the LoRA sequence (resume, placement page fetched, typed scores, log rows with
layout, condition, checkpoint) and the baseline (home check, `go` after a failed check, 5 rows A B C D A with the
right references). The rehearsal caught one crash (a numpy bool in home_check.json), fixed. Not yet run on the arm
or against the real server.

### Imported by ws4-orchestrator (stable through Sunday)

`ws4/orchestrator` drives the Haller task arm with these, imported read-only; only keyword arguments with defaults
may be added: `connect_so101(args)`, `SingleArmRobot(so101, "left", send=, max_relative_target=)`,
`build_guard(args, slot, calib_path, rest)`, `KeepOutGuard(cfg, keepout, slot, measured)`, `read_token(args)`,
`warm_up(backend, robot, target, prompts)`, `move_to(robot, goal, guard_cfg, deg_per_tick, keys, keepout)`,
`hold_here(robot)`, `disconnect_so101(robot, torque_off)`, `FakeSO101`, `keepout.KeepOut`. Callers must read the arm
every tick before `guard.filter` (the keep-out compares against `robot.last_q`), call `hold_here` on any stop, and
switch torque off at the end only if `so101.ws4_launch_torque` is False.

## Hardware as used (red arm)

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
| `haller_identify.py` | which port / wrist camera moves when one arm is moved by hand (read-only) |
| `a0r_stats.py` | A0r quantiles from a recorded LeRobot dataset (per-arm or 12-D with `--arm`) |
| `keepout_demos.py` | replay recorded demos through the keep-out: every demonstrated motion must pass (0 refusals) |
| `compare_frames.py` | recorded dataset frame vs the runner's live camera path: orientation, colour, brightness |
| `joint_test.py` | one joint, small slow motion at a low torque cap: does the output follow (gear check)? |
| `eval_pen.sh` | pen eval rows (`session` = B84 baseline then LoRA, 5 attempts each): tunnel, server check, home check |
| `preflight.py` | read-only checks before an eval: ports, EEPROM vs calibration files, servo health, camera parity |
| `placement.py` | placement page (reference start vs live head camera) and the base-plate shift check |
| `home_check.py` | at home before the first attempt: the runner's live frames and joints vs the at-home references |
| `make_layout_refs.py` | held-out takes' start frames into `layouts/` (references, overview, index) |
| `layouts/` | the 12 reference start frames (head, wrist), `overview.jpg`, `index.json`, placement and scoring guide |
| `offline_diag.py`, `serve_lowmem.py` | offline data-format diagnostic (variants A to E); flux.remote server for 24 GB GPUs |
| `calibration/haller_*.json` | Haller calibration (EEPROM harvest) and keep-out workspaces (`_final`: envelope from the 84 train takes) |
| `calibration/` | `red_follower.json` (as used), `red_follower_rest.json` |
| `frames/` | camera frames Oscar confirmed (`scene_final_L1.jpg`) and earlier placements |
| `handguided/` | the hand-guided episodes and `stats_red_follower.json` |
| `runs/<session>/attempt<NN>/` | `video.mp4` (scene / wrist), `trace.npz` (state, command, guard per tick), `meta.json` |
| `tests/` | fake arm, local fake policy server, STOP file, stall freeze, LeRobot cap, bus errors; keep-out replays of the damaging attempts and the clean demos (`tests/fixtures/`) |

`ws4/flux` was used read-only; nothing in it was changed by this workstream.
