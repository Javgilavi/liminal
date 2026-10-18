# VR teleop recording for FLUX on the XLeRobot station (ws4-teleop)

Sat 17 Oct 2026, ~15:15 CEST. Code and commands: [ws4/teleop/README.md](../teleop/README.md). Code citations are
`haller_ws@b46b827` (Oscar's origin/main, checked out locally as branch `ws4-hackathon`), LeRobot 0.5.1 in
`~/venvs/haller-hmi` (records) and LeRobot main `146d847` (0.6.2) in `~/ws4/lerobot/.venv` (trains FLUX).

## Report

**Done and verified in sim on this laptop:** Quest teleop recording through Oscar's haller_ws HMI produces datasets
in the FLUX contract (30 fps, v3.0, degrees, AV1 640x480 head + two wrists, one task per dataset) with zero edits to
haller_ws. A scripted headset (exact WebXR wire format) recorded 3 episodes; `validate.py` passes them; `to_flux.py`
splits them into per-arm FLUX SO-101 sets and a 12-D set with robot-frame sides; LeRobot's merge precheck accepts
them against the converted public zonglin11 and cortx-labs sets; a real LeRobot-main aggregate plus
`LeRobotDataset` load and decode of every episode works; ws4/flux's A0r quantile code reads them. haller's
`vr_smoke.py` passes 48/48 against the ws4 launcher. 32 CPU tests, ruff clean.

**Not verified yet:** anything with the Quest Pro or the real arms. adb and the current udev rules need a sudo step
from Oscar; the 2026-09-11 calibration files are not on this laptop.

**Next 3 steps:** (1) Oscar: `sudo apt install adb`, install the udev rules, plug in the Quest, then a 2 to 3 take
sim drive from the headset. (2) Robot on this laptop: `calib.py harvest` + `alias`, first-hardware-run checklist,
5 to 10 real takes, validate, convert (A0r statistics). (3) Freeze the prompt list with the team, then 40 to 60
takes per subtask.

## Plan (written first, kept as the record of the decision)

**Path: haller_ws HMI recorder, not vr-teleop-kit + `lerobot-record`.** Both arms are followers and there are no
leader arms, so stock `lerobot-record` cannot capture a bimanual take (haller_ws docs/setup/dataset-collection.md,
"The bimanual path"). The HMI already has the Quest page, per-hand dead-man, E-STOP, in-headset take control and an
in-process recorder that reads the session's committed targets as `action` and measured joints as
`observation.state` from one tick (recorder.py module docstring). The kit's IK is already vendored into the HMI
(`haller_hmi/vr_teleop/kit_teleop.py`), so cloning vr-teleop-kit adds nothing for recording.

**What changes:** only configuration and a launcher in ws4/teleop. `backend.py` imports the unchanged
`haller_hmi.server` app, binds the recorder codec and the headset session rate, and serves it; `up.sh --sim|--real`
starts it plus the Next frontend and an `adb reverse` USB link (http://localhost is a secure context, so WebXR needs
no certificate, no Caddy and no venue Wi-Fi). Post-processing (validate, side swap, per-arm split, load check)
lives in ws4/teleop and reuses ws4/data.

**Risks, in order:** (1) the first hardware run of the unified build with the changed hand pairing (Oscar's
QUICKSTART); (2) calibration files: wrong ones get written into the servos; (3) AV1 encode keeping up with 3 real
cameras next to the 30 Hz loop; (4) the recorder's 0.5% fps gate refusing takes on the real rig; (5) camera parity
between recording and deployment.

## Findings (each one changes what gets recorded)

### 1. haller_ws as it stands would record 60 fps h264

- The recorder hard-codes h264: `recorder.py:901` `vcodec: str = "h264"`, and `server.py:331` builds the recorder
  without passing it. h264 cannot be merged with the AV1 public core set (LeRobot `datasets/feature_utils.py:143-171`;
  ws4/data/merge_datasets.py).
- fps is not `telemetry.hz`, despite dataset-collection.md: the recorder freezes `fps = round(measured tick rate)`
  (`recorder.py:1977` `_freeze_fps`), and during a headset session the tick bus publishes one sample per control
  tick (`server.py:126` builds the session without `sample_hz`). The headset page starts sessions without `hz`
  (`hmi/frontend/components/VRTeleopPanel.tsx:1018`), so the default `hz: float = 60.0` (`human_teleop.py:238`)
  applies: 60 fps datasets. Measured: the first smoke take was refused at "59.534 Hz against fps 60".
- Fix without editing haller_ws: `backend.py` binds `vcodec="libsvtav1"` into `haller_hmi.server.DatasetRecorder`
  and sets the session's own `_hz_override` (`human_teleop.py:743` `effective_hz = self._hz_override or hz`).

### 2. The recorder's 0.5% rate gate is tight; 30.08 Hz centres it

- `FPS_FAITHFUL_FRACTION = 0.005` (`recorder.py:189`). The session loop sleeps `period - elapsed`
  (`human_teleop.py:1890`), so every tick overshoots by the sleep latency. Measured at arm time (sim, AV1 recording):
  target 30.0 gave 29.855 to 29.883 Hz (0.4 to 0.46% slow, one refusal-level margin away); target 30.12 gave
  30.06. Default is now 30.08 (`backend.py SESSION_HZ`), override with `WS4_SESSION_HZ`.
- In-take sampling (from `observation.wall_clock`) averaged 30.01 to 30.04 Hz, p99 interval 33.7 ms. Every take has
  one 155 to 182 ms gap after its first frame (the episode opening); `validate.py` warns on it. Harmless for training
  (one frame per take), worth knowing.
- UNVERIFIED on the real rig, where the loop also reads two serial buses. `validate.py` prints `measured_hz`.

### 3. AV1 at 30 fps with three 640x480 cameras keeps up on this laptop

- Benchmark of LeRobot 0.5.1's own `StreamingVideoEncoder` (the class the recorder uses), 3 cameras, 30 s,
  real-time feed, noisy synthetic frames: libsvtav1 preset 12 (LeRobot default, `video_utils.py:86`) dropped 0
  frames, about 3.4 cores; h264 about 1.5 cores. i7-1260P, 16 threads.
- End to end in sim (MuJoCo rendering 4 cameras + AV1 + backend): 0 skipped ticks, video frames = parquet rows.
- Why `validate.py` counts video frames per episode: when the encoder queue is full LeRobot drops the frame with only
  a log warning (`video_utils.py:820` `except queue.Full`), which silently desynchronises video and parquet.
- Decision: record AV1 directly. `transcode.py` (tested) is the fallback for h264 or 1280x720 takes.

### 4. haller's side names are the operator's, not the robot's

- `config.desktop-real.yaml:88` "WHOSE LEFT? The operator's, standing in front of the bench facing the arms";
  confirmed by Oscar today. haller `left` is the robot's RIGHT arm. ws4/sim (`station_geometry.yaml`: left = +y),
  the public XLeRobot sets and the head-camera image use the robot's sides.
- Recording keeps haller's ids, because the headset's hand pairing is computed from them (frontend `lib/stance.ts`).
  `to_flux.py` swaps: robot left = haller right columns + `right_wrist` video. Verified column by column on the sim
  take and in `tests/test_to_flux.py`. Port / camera / calibration table: ws4/teleop/README.md.
- Consequence for deployment (`ws4/flux/run_xlerobot.py`): `left_arm_config.port=/dev/haller_arm_right`, robot-left
  wrist camera `/dev/haller_cam_wrist_right`. Told the other agents in the shared notes.

### 5. Calibration: one file per physical arm, and the right files are missing here

- Since 2026-09-11 the configs use `haller_left.json` / `haller_right.json`, harvested from the servos' EEPROM and
  named for the side (config.desktop-real.yaml arm notes). This laptop only has May's `haller_follower.json` and
  `haller_leader.json`. Oscar's notes: `haller_follower.json` is a bad calibration for the arm now on
  `/dev/haller_arm_left`, and on mismatch the HMI writes the file into the servos
  (`arm.py:121` `_write_calibration_to_motors`, wired as `calibrate` at `arm.py:296`).
- `calib.py harvest` reads Homing_Offset / Min / Max_Position_Limit (LeRobot `read_calibration`, read only, torque
  left as is) into the side-named files; `check` compares file and EEPROM; `alias` writes `xlerobot_left.json` /
  `xlerobot_right.json` (what `run_xlerobot.py --robot-id xlerobot` loads) as copies; `snapshot` stores them in the
  dataset. `validate.py` fails a real recording whose `haller_joint_calibration` ticks differ from the files.
- The FLUX SO-101 package normalises with one community dataset in the legacy SO-100 degree convention
  (ws4/flux/checkpoint_facts.md), so A0r statistics from our own takes matter; ws4/flux `dataset_arrays` +
  `compute_quantiles` run on the per-arm output (checked).

### 6. Geometry: the teleop IK is right, the collision guard is not

- The teleop IK is the vendored kit solver on the so101_new_calib URDF (`kit_teleop.py:138` `URDF_FROM_MOUNT`, the
  frame LeRobot degrees are defined in), working in each arm's own base frame, so mounts do not enter it. Its tool
  point is the stock jaw `gripper_frame` (`vr_teleop/kit/so101_model.py:51`: (-0.0079, -0.0002, -0.0981) in the
  gripper body); the fin-ray TCP from ws4/sim (station_geometry.yaml v1.2) is (-0.0019, 0, -0.0908): 9 mm apart.
  It only moves the pivot of hand rotations; left unchanged.
- The collision guard sweeps capsules along `haller_hmi/so101_kinematics.py`, the old SO-ARM100 MJCF chain
  (`_CHAIN` line 98, `_TIP_LOCAL` line 114). Compared with ws4/kinematics (so101_new_calib + fin-ray TCP) at the
  same LeRobot degrees, relative to each shoulder-pan pivot: 140 mm apart at all-zero, 183 mm at (0, -60, 60, 0, 0),
  356 mm at (30, -30, 45, 20, 0), and shoulder_pan turns the opposite way. The guard's clearance is therefore not
  meaningful on the real rig whatever the mounts; Oscar already runs it off (`config.desktop-real.yaml:62`).
  `station-real.yaml` keeps it off and transcribes the station mounts into haller's frame for the advisory readout
  (CAD values, `verified: false` in station_geometry.yaml; not tape-measured).
- Gap: no live inter-arm guard on the station model. ws4/kinematics `MjCollision` could drive one from the session's
  `goal_deg` (via `/teleop/human`) with an E-STOP below a margin; not built. Until then the operator keeps the
  grippers apart, as Oscar's own config assumes.

### 7. Cameras

- Recorded: head = D455 colour node over V4L2/OpenCV (`/dev/haller_cam_mast`), wrists = the two Sonix USB gripper
  cams (`/dev/haller_cam_wrist_left|right`), all 640x480 at 30 fps. Oscar's config used 1280x720 for the wrists
  (`config.desktop-real.yaml:204`: 4:3 modes crop to 75% of the horizontal FOV). 640x480 chosen for merge parity
  with the public core set and because FLUX squares every view anyway; revert in the config if grasps suffer.
- Keys: `head`, `left_wrist`, `right_wrist` (haller sides). Per-arm FLUX keys after `to_flux.py`:
  `observation.images.scene` (= head) and `observation.images.wrist`; `_xle12`: head / left_wrist / right_wrist in
  robot sides. No `--rename_map` needed for training.
- Parity at deployment: the recorder rotates both gripper cams 180 degrees and multiplies RGB by per-camera gains
  (`cameras.py:142-183`), and writes fixed V4L2 exposure / gain / white balance on every open. LeRobot's OpenCV camera
  does none of it. `camera_parity.py` reproduces it for the runner (UNVERIFIED that ws4/flux adopts it).
- UNVERIFIED: the gripper-cam colour gains and exposure were solved at 1280x720 on 2026-09-11; the 640x480 mode is
  a crop of the same sensor, so they should carry over.

### 8. haller's sim is a pipeline check, not training data

- haller's MuJoCo arms use the old SO-ARM100 joint zero and report the gripper as the jaw angle in degrees
  (`sim/world.py:185`, versus the real arm's 0..100 at `arm.py:382`): the sim take's shoulder_lift range is
  -190..10 deg. `validate.py` flags sim rigs from `haller_joint_calibration.source` and says not to train on them.
  The ws4/sim station model (LeRobot units) is the sim to use for FLUX work.

## Verified vs unverified

| Claim | Status |
|---|---|
| AV1 640x480 30 fps 3-camera recording through the haller recorder, 0 drops | verified, sim rig on this laptop |
| vr_smoke 48/48 with the ws4 launcher (session 30 Hz, AV1) | verified |
| Per-arm + 12-D conversion with robot-frame sides; merge precheck vs public; LeRobot main aggregate + load + decode | verified on the sim take |
| A0r quantiles computable from the per-arm output | verified (ws4/flux/package_tools on CPU) |
| Arm-time rate within the 0.5% gate at 30.08 Hz | verified in sim only |
| AV1 keeps up with real cameras + serial buses | UNVERIFIED |
| Hand pairing / stance on hardware (QUICKSTART step 4) | UNVERIFIED, first live run |
| `calib.py harvest` against real servos | UNVERIFIED (fake-bus tested; uses LeRobot's own `read_calibration`) |
| USB `adb reverse` path to the Quest Pro | UNVERIFIED (adb not installed yet) |
| Station mounts in haller frame | CAD-derived, not measured; advisory only |
| Gripper-cam imaging settings at 640x480 | UNVERIFIED |

## Gaps and follow-ups

1. Live station-model collision monitor (ws4/kinematics `MjCollision` on `/teleop/human` goals, E-STOP below margin).
2. ws4/flux deploy runner: robot-side port mapping and `camera_parity.StationCameras`.
3. Measure the real-rig session rate on the first take and adjust `WS4_SESSION_HZ` if `validate.py` warns.
4. haller's recorder writes `observation.effort`, `observation.wall_clock` and `episode_uid`; `to_flux.py` drops them
   for FLUX. They stay in the raw recording (useful for grasp detection, see dataset-collection.md).
