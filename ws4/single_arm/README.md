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

### Red-pen benchmark, 18 Oct (first real-arm inserts)

Full table, setup and caveats: [benchmarks/red_pen_2026-10-18.md](benchmarks/red_pen_2026-10-18.md) (regenerate with
`python ws4/single_arm/benchmark.py`). On the rebuilt pen84 table, FLUX RAW 5k (B84 + `lora_pen84` 005000) at an
action horizon of 16 inserted the pen in 3 of 5 attempts (9/15 steps); at the package horizon of 32 the best
checkpoint (RAW 5k) reached 8/15 steps without an insert, ACT pen84 100k 8/15 without an insert, and the lid-held
scene 0/15 (FLUX held still: the pen84 LoRA never saw that scene, and the wrist camera's view of the room decided
hold or reach). Run folders are backed up to `~/haller_rescue/single_arm_2026-10-18/` and the real-serve-6000 volume
(`/workspace/single_arm_runs/`).

Options added for it:

- `align_station.py`: camera-only live page (http://127.0.0.1:8899) that puts the station back on a recording's spot:
  head and wrist next to the reference take with a blend, the wrist room offset/scale/inliers, brightness and a hint.
- `WS4_ACTION_STEPS=N` (or `--action-steps N`): execute N commands of each 42-command chunk before replanning
  (default 32). 16 gave the inserts; without prefetching the arm waits about 0.6 s per chunk (about 50 % at 16).
- `ALIGN=1` in `eval_skill.sh` (or `--align` with `--home-check`): after the ramp home, the runner itself serves the
  alignment page (same numbers and hint as `align_station.py`, on the placement port 8899) from its own cameras while
  it holds the task arm at home, against the skill's home reference (`HOME_TAKE=N` picks the take, default the last
  held-out one). Enter = done, then the usual home check; q / Ctrl-C = stop and park. Works for either arm.
- `WS4_PREFETCH_TICKS=P` (or `--prefetch-ticks P`, off by default and for graded runs until the red-pen RAW 5k block
  is rerun with it): request the next chunk when P commands are left and keep executing; the new chunk replaces the
  queue from the index of the commands executed since its request (`flux/dual_arm_runner.py`, tests in
  `flux/tests/test_runner.py`). On the fake arm at 0.65 s latency: horizon 32 + P 32 never waits and replans about
  every 0.7 s (8 chunks in 6 s), against 38 % waiting (4 chunks) without prefetch and 50 % (5) at horizon 16.
- `WS4_LIFT_STALL_DEG=D` (or `--lift-stall-deg D`): the shoulder lift's stall threshold. The 7.4 V lift trails its
  command by 8-10 deg under load near the 60 % torque cap, which stopped 9 of 15 attempts at 8 deg; the load freeze
  still catches a real push.
- Ramps: a lift short of its goal by up to 8 deg counts as arrived (other joints 6), and a ramp refused by the
  keep-out (after an insert the straight return path entered the torso zone) falls back to one joint at a time in an
  order the keep-out accepts at every step.

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
   (`LoRA-5k-raw`, post_training). All rows use layouts A B C D A (held-out takes 20, 36, 86, 101, 21), 40 s each and
   the same prompt and guards. One command runs all three rows over one tunnel:

   ```bash
   POD_HOST=157.157.221.177 POD_PORT=15377 LORA=lora_10000_raw bash ws4/single_arm/eval_pen.sh session
   ```

   LoRA adapter, pre-registered (orchestrator, 18 Oct): the LoRA row uses whichever adapter (raw or EMA) of the
   latest checkpoint is best on the 12 held-out takes at session start. At 10k (served since about 01:50) that is
   raw: arm MAE 3.15 deg, direction 88.6 %, gripper direction 92 % (EMA 10k 3.19 / 87.2 / 84; raw 5k 3.24 / 83.3 /
   62): package `lora_10000_raw`, variant `LoRA-10k-raw`. The wrapper
   prints the choice before the first attempt and writes it into every LoRA trial's notes. `LORA=lora_<N>` is the
   EMA adapter (`LoRA-<N/1000>k`), `lora_<N>_raw` the raw one (`checkpoints/<N>/pretrained_model`).

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

### Four-skill chain: per-skill evals on both arms (tooling ready 18 Oct 00:45, no new skill built yet)

Final demo (Oscar, 18 Oct): the robot RIGHT arm opens the white box by its handle, then the robot LEFT arm puts the
green pen, the yellow cutter and the red pen into their slots. [skills.py](skills.py) holds the table:

| skill | task arm | port / calibration | wrist camera (dataset key) | trial task | graded steps |
|---|---|---|---|---|---|
| box | robot right | /dev/haller_arm_left, haller_left | USB 7.3 (`left_wrist`) | open_box | reach / grasp the handle, open the box |
| green | robot left | /dev/haller_arm_right, haller_right | USB 7.2 (`right_wrist`) | put_green_pen_in_slot | reach / grasp the green pen, insert |
| cutter | robot left | same | same | put_cutter_in_slot | reach / grasp the cutter, insert |
| pen | robot left | same | same | put_pen_in_slot | reach / grasp the pen, insert |

Both wrist cameras are mounted upside down and recorded rotated 180 (haller_ws `config.flux-real.yaml`, left wrist
checked 18 Oct), so the runner rotates both by 180. Every skill runs the same safety stack: 1 deg/tick, the task arm's
station keep-out with the parked other arm as capsules (read before every attempt), the skill's demo envelope, the
keep-out, stall and load freezes, policy soft limits, the 60 % torque cap.

1. When a skill's dataset is finalised (ws4-vr2 sends the repo id):

   ```bash
   python ws4/single_arm/build_skill.py box
   ```

   It snapshots the dataset (it refuses one whose writers are still open) and takes the split from
   `ws4_split.json`, or else the review.json rejects plus the last 3 good takes held out (marked provisional). It
   replays every good take through the task arm's station keep-out. It fits the envelope on the train takes and
   checks it on the held-out ones, and writes the held-out start frames. While the laptop records (until about
   06:00 CEST on 18 Oct) run it as `nice -n 19 ionice -c3 python ws4/single_arm/build_skill.py <skill> --no-refs`
   (no video decoding: keep-out and envelope only); eval_skill.sh refuses such a skill until the full build has
   written the references. Output: `skills/<skill>/skill.json`,
   `workspace.json`, `layouts/`. Rerun after more takes land. Tested on the finalised pen data: it reproduces the pen
   kit exactly (0 station refusals over 96 takes, the same 1,574-voxel envelope, 0 held-out refusals).
2. Preflight with the skill's arm and references: `python ws4/single_arm/preflight.py --skill box`.
3. Eval rows (packages on 8790 from ws4-datasets): `SKILL=box POD_HOST=157.157.221.177 POD_PORT=15377
   LORA=<package> LORA_CHECKPOINT="<what it is>" bash ws4/single_arm/eval_skill.sh session`. Rows default to `lora`
   for the new skills (`ROWS`, `BASE`, `MEDIAN` add the others). `eval_pen.sh` is `eval_skill.sh` with `SKILL=pen`
   and is unchanged in what it runs. A skill whose demos the keep-out refused does not run
   (`ALLOW_KEEPOUT_REFUSALS=1` after a review). The runner's `--task-side right` points the home check at the
   image-right arm.

The other arm: the runner reads it read-only before the home check and before every attempt. Its live pose places
its keep-out capsules, and the runner never moves it. It refuses to start when any arm joint of the other arm is
more than 5 deg (`PARKED_TOL`) outside the per-joint min..max of that arm over the skill's demos. Its place in both
camera views is part of what was learned. Box skill: the pen arm tucked at [-1.1, -102.5, 95.9, 37.7, -11.1, 2.4].
Pen-arm skills in the chain: the box arm HOLDS the lid open (the lid closes if it leaves), at a pose that varies a
little per run, so their range comes from their own demos: `build_skill.py` records min/max per joint, and a green or
cutter skill without that range does not run. The current red-pen dataset was recorded with the box arm tucked
(2.8..3.1, -102.9..-102.6, 96.7, 78.3..78.5, 12.8..13.0 over the 96 good takes); a re-recorded red pen gets its
range from the rebuild. During tonight's cup runs the box arm sat at pan 85 deg, so the current pen session refuses
until it is back at pan ~3.

Robot-right arm keep-out: `KeepOut.for_haller("right")` mirrors the left arm's station zones (bases at y = -/+0.137 m,
torso and mast centred, midline y <= +0.11 m). Its home and folded park pose are clear, and a reach toward the parked
left arm is refused by the capsules. The SO-101 itself is not mirror symmetric: on 5,000 random poses, the right arm
at a mirrored pose differs from the left by up to 3.6 cm in the torso, mast and midline depths. So the right-arm zones
are UNVERIFIED until the box demos replay with 0 refusals in `build_skill.py`. The right-arm home-check box (where the
arm sits in the head view) is the mirror of the left one, also UNVERIFIED until the box references exist.

**Box skill, built 18 Oct 01:45 (no references yet).** `ws4/haller_open_the_box`, 75 takes, final split
(drop 3, 7, 16, 19, 21, 28, 41, 46, 47, 70; held-out P2 36-38, P3 72-74). The first build failed its own demos (183
of 52,800 steps refused). Five takes reach TCP y 0.131 m measured / 0.142 m commanded, past the right arm's +0.11 m
midline. Takes 4-6 come within the capsule model of the pen arm tucked the early way. Approved by the orchestrator:
the box skill uses takes 7-74 only (the pen arm parked at ~[3, -83.6, 81.9, 82.8, -10.2, 0.9], pan 1.1..5.4 after a
bump in take 7), with the right-arm midline at +0.17 m for box only. Both are recorded as overrides in skill.json with
the reason. Result: 0 refusals over the 59 good takes with the station zones (43,366 steps), over the 53 train takes
with zones and envelope (39,666) and over the 6 held-out takes (3,700). The dilated TCP envelope keeps at least 2.04
cm (2.4 cm sampled) from the parked pen arm's capsules at both ends of its pan range. In the demos the box arm never
came closer than 4.4 cm to the pen arm. The capsule check (4 cm) is unchanged. The full build after 06:00 adds the
references:
`python ws4/single_arm/build_skill.py box --exclude 0 1 2 3 4 5 6 --midline 0.17 --reason "<same>"`.

**Green skill, built 18 Oct 03:10 (no references yet).** `ws4/haller_put_the_green_pen_in_the_green_slot`, 66
takes (drop 14, 18, 25, 44, 50; held-out 63-65 at P3). The box arm holds the lid in every frame (pan 12.0..12.3, the
rest fixed). The first build refused 9,866 of 53,042 steps: the green slot is in the box on the robot's right, so
every take reaches TCP y -0.179 m measured / -0.198 m commanded, past the pen arm's -0.11 m midline. Approved by
the orchestrator:
- midline -0.23 m for green;
- takes 3, 15, 23, 41, 47 left out of the safety envelope only (box-arm clearance, mast and table outliers; still in
  the LoRA data);
- `--env-gap 0.02`, which trims envelope voxels near the lid-holding box arm. The threshold includes the sampling
  bound, so the trimmed envelope keeps at least 2.05 cm (2.23 cm sampled on a 12^3 grid).

Result: 0 refused over the 56 good takes (48,456 steps) and the 3 held-out takes with the envelope (2,288). The 5
excluded takes still trip green's zones 62 times (box arm 30, mast 28, table 4).
Standing rule for the cutter (orchestrator): the same procedure within these bounds without asking: midline down to
-0.23 m, up to 10 % of takes out of the envelope, 2 cm trim. Every override is recorded and reported.

Red pen in the final setup: the current red-pen envelope (104 old takes, box at A-D, box arm tucked) does not cover the
box at its current position. Near the slots, 54 % of the green commands fall outside its TCP envelope (up to 5.3 cm),
pan goes up to 10 deg past its joint envelope, and 34 % cross its midline. With the midline at -0.23 it still refuses
11,298 of 53,042 green steps. About 10 red-pen takes in the final setup are recommended, built like green.

**Cutter skill, built 18 Oct 04:10 (no references yet).** `ws4/haller_put_the_yellow_cutter_in_the_yellow_slot`, 70
takes (drop 10, 27, 29, 30, 38, 44, 45, 50; held-out 67-69 at P3), box arm holding the lid. The standing rule gives
midline -0.21 m (demos reach -0.154 m measured / -0.175 m commanded), takes 3, 5, 31, 52 out of the envelope (mast
outliers, 6.5 %) and the 2 cm trim (34 voxels; at least 2.12 cm guaranteed). That still left the 4 cm box-arm
clearance refusing 52 of 58 takes: the pen arm's forearm and wrist pass the lid-holding box arm's shoulder and upper
arm next to the yellow slot (1.7 cm measured, 1.3 cm commanded). The orchestrator decided on a 1.0 cm clearance for
the cutter only, with a **slow zone**:
- whenever the arm's capsules are within 4 cm of the other arm's (measured), `KeepOutGuard` caps arm steps at 0.5 deg
  per tick from the previous command;
- `run_attempt` freezes on 50 % of the torque cap for 0.2 s instead of 70 % for 0.3 s.

Runner flags `--slow-zone 0.04 --slow-step 0.5 --slow-load-sustain 0.5 --slow-load-ticks 6`, recorded in skill.json
`runner_args` and passed by eval_skill.sh; off for every other skill. Result: 0 refused over the 58 good takes with
the station zones (49,950 steps) and the held-out takes with the envelope (2,364). Train with the envelope: 2 of
47,586, both measured states in take 66 (commands pass). The slow zone covers 4.4 % of the demo ticks, in 31 of 58
takes. Fake-robot tests: in the zone a 10 deg command moves 0.5 deg, and a 55 % load freezes there but not outside.

**Box-arm gate for the pen-arm chain skills (18 Oct 04:55).** The box arm holds the lid wherever the FLUX box skill
ended, and that varies run to run. Box demos' end poses (59 good takes, last second): lift -25..25, elbow -11..41,
roll 45..85 deg (p5..p95). The green/cutter recordings held at roll 29.9 and the red-pen session at roll 14.6, both
below every box end, so a ±5 deg range around a recorded hold would refuse nearly every real run. Approved by the
orchestrator, [gate.py](gate.py) replaces it for green, cutter and pen_lid:
- a loose joint sanity range: box ends plus the skill's recorded hold, each widened by 5 deg;
- a geometric gate with the live box-arm pose: refuse when any envelope voxel lies inside the box arm's capsules, or
  when more than 1 % of the skill's saved demo poses (about 3,000, `demo_poses.npz`) fall under its other-arm
  clearance (green 4 cm, cutter 1 cm).

It is recorded in skill.json `other_arm_gate`, checked at the home check and before every attempt, and passed by
eval_skill.sh. The per-tick capsule check and the cutter's slow zone stay the safety guard. Verdicts:
- the recorded hold and the median box end: allowed for both;
- box take 74's end: refused for both (envelope voxels inside the box arm);
- take 38's end: refused for the cutter only;
- the red-pen hold: refused for both.

`BOX_END="q1 .. q6" bash ws4/single_arm/chain_dryrun.sh` (or `gate.py report q1 .. q6`) shows the verdict per pen
skill for a box result before a demo run.

**Handover from the HMI to the runner (pen-arm chain skills, box arm holding the lid).** The HMI's `down.sh` drops
torque on both arms (recorder notes, 18 Oct 02:37), and the runner takes the box arm over with `--hold-other-arm`
(eval_skill.sh sets it for green, cutter and pen_lid). It caps Torque_Limit, writes Goal_Position = the box arm's
present position, then enables torque and verifies. It never commands that arm anywhere else, and leaves it holding
(torque on) when it exits. First real handover 18 Oct 03:41: it took the box arm over and held it (torque on,
goal = present), then the start check refused correctly: the box arm had dropped to a folded pose when down.sh cut
torque ([-2.5, -95.2, 97.9, 65.7, 14.7], lift 56 deg outside the pen_lid range). Re-placing it: support it,
`python ws4/single_arm/torque_off.py --port /dev/haller_arm_left --robot-id haller_left` (typed confirmation),
place it holding the lid as in the takes (`show_joints.py` streams its joints; `gate.py report q1..q6` gives the
verdict), then start the session again and hold it until the runner prints `held`.
1. Oscar supports the lid and both arms by hand.
2. `bash ~/haller_ws/scripts/quest-teleop/down.sh`, then check the ports are free:
   `fuser $(readlink -f /dev/haller_arm_right /dev/haller_arm_left) || echo free`.
3. Fold the pen arm down onto its rest by hand. The runner parks it back at its launch pose at the end and switches
   its torque off there, so the launch pose must be the folded rest, not mid-air.
4. Still supporting the lid: `python ws4/single_arm/preflight.py --skill green` (read-only, about 30 s: ports, EEPROM
   vs calibration files for both arms, servo health, cameras).
5. Start the session command. Its first runner line is `other arm (/dev/haller_arm_left) held at [...]: torque on,
   goal = its own pose`: from then on the box arm holds the lid and Oscar can let go. The pen arm then connects at its
   rest (torque on), the home check ramps it to home, and the box-arm gate prints its verdict with the live pose.
   REFUSED means the box arm holds the lid somewhere the skill's demos never had it: re-place it through the HMI
   and start again.
6. At the end the pen arm parks at rest (torque off). The box arm keeps holding the lid (torque on): support the lid
   before powering it off or starting the HMI again.

Tested without the arm: 8 tests in `tests/test_skills.py` (table, right-arm keep-out, mirrored home check, refs index,
provisional split, eval_skill commands for the box on the right arm, the refusal gate). There was also a fake-robot run of a
test box skill (built from the pen data with the right arm's columns) against the live 8790 server: right-arm flags,
`open_box` steps, and the home check reporting the missing left-wrist reference instead of crashing.

### Box fallback: scripted replay of a recorded take (18 Oct, morning)

The FLUX box LoRA (`skill_open_box`) went 0/4 on the arm (runs/20261018_082015, orchestrator
runs/20261018_081707_open_the_box): it crawls from rest, the wrist rolls about 5 deg where the demos roll about 50, and
the tcp_env keep-out stops it; offline from the recorded frame 0 it barely moves either. `--backend replay`
([replay.py](replay.py)) feeds one recorded take's box-arm actions (`left_*` columns) to the runner as policy chunks,
so every command passes the same guard, keep-out with the box envelope, stall / keep-out / load freezes, torque cap and
start check as a FLUX command. The take is time-warped onto the step limit first (each recorded step split into ticks
of at most `--max-step` deg, 5 % on the gripper, linear in joint space, from the arm's pose at the start), so the arm
follows the recorded path, only slower where the operator moved faster. When the last command has been executed the
attempt ends with `replay finished: scripted replay of recorded demo N, holding where the take ends`; the arm keeps
holding (x, the default) or returns on r. meta.json carries `replay` and `checkpoint: scripted replay of recorded demo
N (...)`; scored attempts go to trials.jsonl with condition `scripted` and variant `scripted-replay-ep<N>`, never
with the policy rows. Prefetch is forced off (it would skip recorded steps).

The gripper keeps its timing along the arm's path. The VR trigger jumps the recorded command 0 <-> 100 in one frame
and the real gripper then took about a second to follow while the arm moved on (take 59: trigger at frame 17, shut on
the handle at frame 45). The first version ramped that jump with the arm standing still and then slowed the arm:
offline servo model, the gripper shut at arm frame 20, on air, and re-opened 22 frames early; the first live box
replay (take 59, 10:05) ran to its end with the lid still closed. Now, while the recorded gripper state moves toward
its command, the replay commands that state 2 frames ahead (VERIFY the lead on the arm); once it has settled, the
recorded command again, so a closed grip still squeezes as in the take. Same model: shut at frame 44, re-opened at
154 (recorded 45, 154); takes 40, 42, 72, 73 within 2 frames.

Default take 72 (P3, held out): its first head frame is `skills/box/layouts/P3_ep072_head.jpg`, so the placement
page and the home check show exactly where its box and handle were. Ranked over the 32 good P3 takes (10:30, first
frames decoded from the snapshot, template match of the box and of its handle tab against the P3 references): every
take's box and handle sit within 6 px of the references, so having the take's own reference decided. 72 passes all
three pen-skill gates at its end (demo gaps 4.0 / 1.5 / 7.2 cm), 0 refusals from box home, 441 ticks (15 s); 73
(`P3_ep073_head.jpg`) is the second choice, 74 fails the green and cutter gates, 59 has no kit reference.
`python ws4/single_arm/replay.py 72` summarises a take.

```bash
python ws4/single_arm/run_single_arm.py --backend replay --replay-episode 72 \
  --port /dev/haller_arm_left --robot-id haller_left --other-arm-port /dev/haller_arm_right --other-arm-id haller_right \
  --other-arm-lo 1.1 -83.7 81.8 82.8 -10.4 0.9 --other-arm-hi 5.4 -83.5 81.9 82.8 -10.1 0.9 --other-arm-tol 5 \
  --scene-cam /dev/haller_cam_mast --scene-fourcc none \
  --wrist-cam /dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.3:1.0-video-index0 --wrist-fourcc none --wrist-rotate 180 \
  --task-side right --keepout-config ws4/single_arm/skills/box/workspace.json --start-pose 0 0 0 0 0 100 \
  --task "open the box by pulling its handle" --trial-task open_box --layouts P3 \
  --layout-ref ws4/single_arm/skills/box/layouts --home-check ws4/single_arm/skills/box/layouts \
  --seconds 60 --max-step 1.0 --max-relative-target 20 --lift-stall-deg 25
```

**Cutter (last-resort fallback, same day):** the augmented FLUX cutter grasped 4/5 but inserted 0/5 (it swings the
wrist roll instead of carrying the cutter). `--replay-skill cutter` (skills.py: the cutter dataset, the robot-left arm's
`right_*` columns, default take 67) replays a cutter take the same way, with the cutter kit's flags (slow zone, box-arm
gate, `--hold-other-arm`). Within the slow zone plus 1 cm of the box arm's live pose the take is warped at the slow
step (0.5 deg/tick), so the slow zone never has to clip it. Take 67 (P3, held out) has its first head frame in the kit
(`skills/cutter/layouts/P3_ep067_head.jpg`): with `--layout-ref` / `--home-check` on the kit's layouts the placement
page and the home check use THAT frame, so the cutter can be placed where the take had it. Of the three takes with a
reference it keeps the most room to the box arm (2.9 cm with the box arm where box take 59 ends, 1 cm clearance),
spends the fewest ticks in the slow zone, starts at home with the green pen in its slot and ends exactly at home; 0
refusals by the cutter workspace (757 ticks, 25 s at 1 deg/tick). The red pen's gate only looks at the box arm, which
the replay never moves (allowed with the box arm at take 59's end).

```bash
python ws4/single_arm/run_single_arm.py --backend replay --replay-skill cutter \
  --port /dev/haller_arm_right --robot-id haller_right --other-arm-port /dev/haller_arm_left --other-arm-id haller_left \
  --other-arm-lo -2.5 -38.9 -25.9 -30.6 24.9 0 --other-arm-hi 26.9 31.5 53.4 46.9 93.9 100 --other-arm-tol 0 \
  --other-arm-demo-poses ws4/single_arm/skills/cutter/demo_poses.npz --other-arm-gate-max-below 0.01 --hold-other-arm \
  --scene-cam /dev/haller_cam_mast --scene-fourcc none \
  --wrist-cam /dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0 --wrist-fourcc none --wrist-rotate 180 \
  --task-side left --keepout-config ws4/single_arm/skills/cutter/workspace.json --start-pose 0 0 0 0 0 100 \
  --task "put the yellow cutter in the yellow slot" --trial-task put_cutter_in_slot \
  --steps "reach the cutter" "grasp the cutter" "insert it in the yellow slot" --layouts P3 \
  --layout-ref ws4/single_arm/skills/cutter/layouts --home-check ws4/single_arm/skills/cutter/layouts \
  --seconds 90 --max-step 1.0 --max-relative-target 20 --lift-stall-deg 25 \
  --slow-zone 0.04 --slow-step 0.5 --slow-load-sustain 0.5 --slow-load-ticks 6
```

`--prime-ticks K` with `--backend remote` replays the first K ticks (rounded up to the action horizon) and then lets
FLUX take over with that history, for trying a primed FLUX box start; such attempts are logged with the variant
`<variant>+primed-ep<N>-<K>`. Tests: `tests/test_replay.py` (fake arm, a synthetic parquet; take 72 against the box
workspace when the dataset is in the cache).

### Imported by ws4-orchestrator (stable through Sunday)

`ws4/orchestrator` drives the Haller task arm with these, imported read-only; only keyword arguments with defaults
may be added: `connect_so101(args)`, `SingleArmRobot(so101, "left", send=, max_relative_target=)`,
`build_guard(args, slot, calib_path, rest)`, `KeepOutGuard(cfg, keepout, slot, measured)`, `read_token(args)`,
`warm_up(backend, robot, target, prompts)`, `move_to(robot, goal, guard_cfg, deg_per_tick, keys, keepout)`,
`hold_here(robot)`, `disconnect_so101(robot, torque_off)`, `FakeSO101`, `keepout.KeepOut`, `ARRIVE_TOL` (5 values,
lift 8), `joint_order(keepout, start, goal, step=1.0)`, `gate.check(ko, other_q, demo_poses, max_below, samples=4)`,
`gate.load_poses(path)`, and per skill `skills/<box|green|cutter|pen_lid>/skill.json` (`task`, `workspace`,
`runner_args`, `other_arm_gate`) with its `workspace.json`. Callers must read the arm
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
   ticks hold the measured pose, are logged per tick and counted in the trial notes. Refusals on 45 of the last 60
   policy ticks (1.5 s within 2 s) end the attempt (`KeepoutStop`, `--keepout-ticks` / `--keepout-window`); holds
   between chunks neither count nor clear. Until 17 Oct 23:15 this was "45 in a row", which the holds and each new
   chunk's first, shallower commands always reset: replayed on every recorded Haller attempt, the longest run was 41
   and none stopped (found by the ACT baseline session). The window rule would have ended 9 of the 10 attempts that
   had refusals, 7 to 43 s in; the one with none is untouched, and the demos have 0 refusals. Replays: the two damaging A0 attempts are refused at 0.97 s and 1.27 s (elbow load passed 50% at 7 s
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
