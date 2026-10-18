# ws4/teleop: Quest Pro teleop recording for FLUX on the XLeRobot station

Owner: ws4-teleop. Background, evidence and open points: [../research/vr_teleop.md](../research/vr_teleop.md).

This drives both arms from the Quest Pro with Oscar's own haller_ws HMI (WebXR, per-hand dead-man, in-process
bimanual recorder) and adds only configuration and thin adapters, so the recordings come out in the FLUX
contract: **30 fps, LeRobot v3, joints in LeRobot degrees, gripper 0..100, AV1 640x480 for head + both wrists,
one fixed task string per dataset.** `to_flux.py` then splits each recording into per-arm FLUX SO-101 datasets.

| File | What it does |
|---|---|
| `up.sh`, `down.sh` | start / stop backend + frontend + USB link to the headset; `down.sh` waits for the dataset to finalise |
| `backend.py` | runs the unchanged haller backend with AV1 and a 30 Hz headset session (the two things a config cannot set) |
| `configs/station-sim.yaml`, `configs/station-real.yaml` | haller HMI configs reshaped to the contract |
| `validate.py` | contract check of a raw recording (fps, codec, size, keys, units, video frames vs rows, calibration, tasks) |
| `to_flux.py` | recording -> `<name>_left`, `<name>_right` (6-D FLUX SO-101) and `<name>_xle12`, robot-frame sides |
| `check_load.py` | opens datasets with the LeRobot that trains FLUX and decodes every episode's first and last frame |
| `calib.py` | read-only EEPROM harvest, file vs servo check, deploy copies `xlerobot_<side>.json`, snapshot into a dataset |
| `transcode.py` | fallback: h264 or 1280x720 recordings -> AV1 640x480, frame counts unchanged |
| `camera_parity.py` | the rotation / colour gains / V4L2 controls the recorder applies, for the deployment loop |
| `sim_take.py` | scripted headset (exact WebXR wire format) for rehearsals without the Quest |
| `collision_monitor.py` | inter-arm clearance on the station model from the live session goals, E-STOP below 30 mm; replay QA |
| `prompts.yaml` | frozen instruction strings (still a proposal) |

## Sides: read this once

haller_ws names arms from the **operator's** side, standing in front of the bench facing the robot. FLUX, ws4/sim
and the public XLeRobot data use the **robot's** sides. Recording keeps haller's ids (the headset's hand pairing
depends on them); `to_flux.py` swaps them.

| robot side (FLUX datasets, ws4) | haller id (HMI, recorder columns) | port | gripper cam | calibration |
|---|---|---|---|---|
| left | right | `/dev/haller_arm_right` | `/dev/haller_cam_wrist_right` | `haller_right.json` = `xlerobot_left.json` |
| right | left | `/dev/haller_arm_left` | `/dev/haller_cam_wrist_left` | `haller_left.json` = `xlerobot_right.json` |

## One-time setup on the robot laptop

In a normal terminal (Ctrl+Alt+T), not the Claude app's terminal panel, which blocks sudo ("no new privileges"):

```bash
bash ~/liminal/ws4/teleop/setup_host.sh
```

It installs adb (its udev rules open the Quest's USB node to the logged-in user) and the current haller udev rules.

Quest Pro: developer mode on, USB-C into the laptop, accept "Allow USB debugging" in the headset.

Calibration, with the arms powered and nothing else holding the ports. If the 2026-09-11 `haller_left.json` /
`haller_right.json` exist on the desktop, copy them into `~/.cache/huggingface/lerobot/calibration/robots/so_follower/`
instead of harvesting. Never point LeRobot at `haller_follower.json` / `haller_leader.json` for these arms.

```bash
~/venvs/haller-hmi/bin/python ws4/teleop/calib.py harvest
```

```bash
python3 -I ws4/teleop/calib.py alias
```

```bash
python3 -I ws4/teleop/calib.py status
```

`harvest` only reads the servos. `alias` writes the robot-side copies that `ws4/flux/run_xlerobot.py --robot-id
xlerobot` loads, so deployment uses exactly the recording calibration. If anyone recalibrates an arm, harvest again
and record into a new dataset name.

## Headset on the table, controllers only, everything else from the laptop

Works because the Quest Pro controllers track themselves. `quest_link.sh` keeps the headset awake off-face
(proximity sensor off until reboot). Put the headset on once to press Enter Passthrough on
`http://localhost:3001/teleop/vr`, then set it on the table **facing the way you face while driving**: on every grip
squeeze the backend takes "forward" from the headset's heading (haller_hmi/vr_teleop/core/frames.py `head_yaw`).
Do not move it during a session. Start and stop takes from the laptop (`http://localhost:3001/` -> DATA -> COLLECT ->
RECORDER card), not with A/X: A/X uses the task draft stored in the Quest browser, the laptop buttons use the
laptop's.

## Rehearse in sim (nothing physical moves)

```bash
ws4/teleop/up.sh --sim
```

Open `http://localhost:3001/teleop/vr` in the Quest browser, Enter Passthrough. The MuJoCo view floats in the HUD.
Sim takes check the pipeline only: haller's sim arm is not in LeRobot units, never train on them.

## Record real demos

1. Start everything. The script refuses if a device symlink or a calibration file is missing.

   ```bash
   ws4/teleop/up.sh --real
   ```

2. First real run of the day, before any recording (QUICKSTART-QUEST.md "First hardware run", adapted): a second
   person at the cockpit E-STOP (`http://localhost:3001/`, red button, or B/Y in the headset). In the headset pick
   the `dual` session, stance `behind` if you stand behind the robot. Read the caption under the session buttons
   (`L hand -> right · R hand -> left`). Squeeze ONE grip, hold still through the 1 s countdown, nudge the hand
   5 cm outward: the arm on that side of the robot must move outward. Then up, then forward. Trigger through its
   range. Only then the other hand. The collision guard is off (its geometry is wrong for this station): keep the
   two grippers apart yourself, with `collision_monitor.py` (below) as a second guard.
3. In the cockpit's Dataset tab, set the HF user (`Oskrt`) and paste the task string from `prompts.yaml`. It
   persists, so the headset can start takes from it. One task string per dataset.
4. Per episode: reset the scene, hold A or X ~0.5 s (HUD shows `● REC ep n`), do the task, hold A/X again, then
   left stick click = SAVE, right stick click = DISCARD. Discard failures. Aim for 5 to 10 episodes per task first
   (enough for the A0r statistics), then 40 to 60.
   Without the headset buttons: `curl -X POST localhost:8000/record/start -H 'content-type: application/json' -d
   '{"repo_id": "Oskrt/xle_s2_open_hold_lid", "task": "Open the box lid and hold it open"}'` and
   `curl -X POST localhost:8000/record/stop -H 'content-type: application/json' -d '{"save": true}'`.
5. Stop, and let it finish writing (up to 2 minutes, never Ctrl-C it):

   ```bash
   ws4/teleop/down.sh
   ```

Takes land in `~/.cache/huggingface/lerobot/<hf user>/<name>`.

## Inter-arm clearance monitor (next to `up.sh --real`)

haller's guard stays off (old SO-ARM100 chain, ../research/vr_teleop.md section 6). `collision_monitor.py` is a
second guard on the measured station model: it polls `GET /teleop/human` at 20 Hz, maps each headset slot through
the session pairing to its haller arm and then to the robot side (table above), and takes the signed distance
between the two arms' meshes in `ws4/sim/xlerobot_station.xml` (fin-ray fingers, about 0.5 ms per check). Below
30 mm it POSTs `/estop` once per session and logs why, on the terminal and in `/tmp/ws4-teleop/collision_monitor.jsonl`.
Start it in a second terminal once `up.sh --real` is up; the first run of the day, use `dry-run` (prints the
clearance twice a second, never e-stops) and bring the grippers together slowly to see the number fall.

```bash
python3 -I ws4/teleop/collision_monitor.py live
```

```bash
python3 -I ws4/teleop/collision_monitor.py dry-run
```

`--margin-mm` widens the margin, `--url` points at another backend. It checks the committed goals (what the arms
are sent, not where they are), and only while a session drives both arms (a one-arm session prints `partial`).
Mounts are CAD values, not tape-measured, so the person at the E-STOP stays. After an e-stop haller drops torque
and ends the session: separate the arms, recover in the cockpit, start a new session, and the monitor re-arms.

Data QA: minimum clearance per episode of a recording's `action` column, plus a per-episode `sides` check of action
against `observation.state` (exit 1 if any episode dips below the margin or is crossed). `crossed` / `solo crossed`
means haller's recorder filed each arm's goals under the other arm, which it does whenever the headset pairing is
crossed (stance `behind`, the frontend default): `goal_deg` is keyed by hand slot and the recorder reads it by arm
id. Do not feed those episodes to `to_flux.py`: swap `action[:, :6]` and `action[:, 6:]` in a `crossed` one,
re-record a `solo crossed` one.

```bash
python3 -I ws4/teleop/collision_monitor.py replay ~/.cache/huggingface/lerobot/Oskrt/xle_s2_open_hold_lid --json /tmp/ws4-teleop/clearance_qa.json
```

## Validate, convert, check, upload

```bash
python3 -I ws4/teleop/validate.py ~/.cache/huggingface/lerobot/Oskrt/xle_s2_open_hold_lid --prompts ws4/teleop/prompts.yaml
```

```bash
python3 -I ws4/teleop/calib.py snapshot ~/.cache/huggingface/lerobot/Oskrt/xle_s2_open_hold_lid
```

```bash
python3 -I ws4/teleop/to_flux.py ~/.cache/huggingface/lerobot/Oskrt/xle_s2_open_hold_lid ~/ws4_data/flux --check-against ws4/.cache/converted/zonglin11_left ws4/.cache/converted/cortx-labs_left
```

```bash
~/ws4/lerobot/.venv/bin/python -I ws4/teleop/check_load.py ~/ws4_data/flux/xle_s2_open_hold_lid_left ~/ws4_data/flux/xle_s2_open_hold_lid_right ~/ws4_data/flux/xle_s2_open_hold_lid_xle12
```

Upload privately (this publishes to Hugging Face, so it is Oscar's call; repeat for `_right`, `_xle12` and the raw
recording):

```bash
hf upload Oskrt/xle_s2_open_hold_lid_left ~/ws4_data/flux/xle_s2_open_hold_lid_left --repo-type dataset --private
```

Training (ws4/flux/GPU_RUNBOOK.md section 6) takes `--dataset.repo_id=Oskrt/xle_s2_open_hold_lid_left`. No
`--rename_map` is needed: the per-arm sets already use `observation.images.scene` / `observation.images.wrist`, and
`_xle12` uses `head` / `left_wrist` / `right_wrist`. A0r statistics (runbook section 4) come from the same per-arm
sets: `dataset_arrays("Oskrt/xle_s2_open_hold_lid_left")`, or `root=` the local folder.

The per-arm task string is `"<prompt> with the left arm"` (robot side); the deployment prompt must be identical.

## Tests

```bash
python3 -m pytest ws4/teleop/tests -q
```

```bash
ruff check ws4/teleop
```

56 CPU tests, no headset, robot, GPU or network (the MuJoCo ones in `test_collision_monitor.py` skip without
`mujoco`). End-to-end sim check (needs the haller-hmi venv, ~2 min):
`ws4/teleop/up.sh --sim`, then `~/venvs/haller-hmi/bin/python ws4/teleop/sim_take.py --repo-id ws4sim/check
--task "Move both arms through the workspace" --episodes 3 --reset-seed -1`, `ws4/teleop/down.sh`, then the
validate / to_flux / check_load commands above (`--no-calib` for sim).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `/record/start` 409 "measured ... Hz against fps 30 is ...% slow" | the session loop overshoots its period; `down.sh`, then `WS4_SESSION_HZ=30.12 ws4/teleop/up.sh --real` (default 30.08, measured in sim; raise it if slow, lower it if fast) |
| validate: `video N frames ... parquet M rows` | the AV1 encoder fell behind and LeRobot dropped frames (it only logs a warning). Record with `WS4_VCODEC=h264 ws4/teleop/up.sh --real` instead, then `transcode.py` it to AV1 before merging |
| validate: `fps 60` | the backend was started without `backend.py` (haller's own up.sh): the VR page asks for 60 Hz sessions |
| validate: `codec h264` | same, or `WS4_VCODEC=h264` was set: not mergeable with the public core set |
| the wrong arm answers a hand | stance / pairing, see QUICKSTART-QUEST.md; do not edit ids in the config, that double-corrects |
| `up.sh --real` says a `/dev/haller_*` link is missing | udev rules not installed (above) or the board / camera is unplugged |
| headset page loads but never connects | backend URL baked into the Next dev cache; `down.sh` then `up.sh` (it clears `.next/dev` when the URL changes) |
