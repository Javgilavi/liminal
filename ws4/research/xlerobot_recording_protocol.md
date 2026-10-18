# XLeRobot demo task and recording protocol (draft, orchestrator)

Status: proposal for Oscar and the team to decide on. Written 2026-10-17 while the WS4 agents run.
Facts come from LeRobot 0.5.1 source on this laptop and the BFL/flux-action docs (see ws4/CONTEXT.md).
Anything not checked on the real robot is tagged UNVERIFIED.

## Why record now

Every adaptation path needs XLeRobot demos, and recording them is the slowest step that no agent can do.
If we record **raw joint space for both arms plus all three cameras at 30 Hz**, one dataset feeds all paths:

| Path | What it takes from the raw recording |
|---|---|
| Per-arm SO-101 LoRA (works with the released checkpoint) | split into two 6-D datasets: head cam as `scene`, that arm's wrist cam as `wrist` |
| Native 12-D joint heads | use as is: `[left 6 \| right 6]`, 3 cameras |
| EE50 (Simone's plan) | forward kinematics on the joints (ws4/kinematics), gripper 0 to 100 mapped to 0..1 |

So recording does not wait for the per-arm vs 12-D vs EE50 decision.

## Recording contract

- Tool: `lerobot-record` with robot type `bi_so_follower` (in LeRobot 0.5.1). Each arm is an `SOFollower`
  with `use_degrees=True` by default, so joints are in degrees and the gripper is 0 to 100. That matches
  the FLUX SO-101 contract in flux-action `docs/setup.md` (state in degrees, gripper in percentage points).
- 30 fps (the FLUX SO-101 recipe is 30 Hz, and the LeRobot FLUX guide asks for a 30 Hz LeRobot v3 dataset).
- Features produced by `bi_so_follower`: state and action keys `left_<joint>.pos` and `right_<joint>.pos`
  (12 each), and camera keys prefixed by the arm whose config holds the camera.
- Cameras, all 640x480 at 30 fps, RGB only (FLUX does not take depth):
  - RealSense colour stream, attached to the left arm config as `head`, so it lands as
    `observation.images.left_head`. Rename it at training time with `--rename_map`.
  - Left gripper camera as `wrist` on the left arm: `observation.images.left_wrist`.
  - Right gripper camera as `wrist` on the right arm: `observation.images.right_wrist`.
- One instruction per episode (`--dataset.single_task`). Keep a fixed phrase list (below) and never
  paraphrase between episodes of the same subtask.
- Keep `--dataset.push_to_hub=false` until Oscar decides on visibility, then push privately.
- Save the calibration files used (LeRobot calibration directory for `xlerobot_left` / `xlerobot_right`)
  next to the dataset. The FLUX SO-101 normalization assumes LeRobot calibration conventions, so a
  re-calibration mid-recording silently changes the data.

### Before the first episode

1. `lerobot-find-port` for each arm, `lerobot-find-cameras opencv` and `lerobot-find-cameras realsense`.
2. Calibrate both followers with LeRobot (`lerobot-calibrate --robot.type=bi_so_follower ...`) and both
   leaders. **Gripper with the soft fingers:** calibrate the range as fully open to fully closed on
   nothing, and check that "closed on a 3 cm cube" lands well above 0 so the policy can see a grasp.
3. UNVERIFIED: in stock XLeRobot the left arm shares a motor bus with the head servos and the right arm
   with the base wheels. `SOFollower` only addresses motor IDs 1 to 6, so this should be fine, but run
   `lerobot-teleoperate` first and watch for ID conflicts or the head moving.
4. Set `max_relative_target` (for example 10 degrees per step) on both followers for the first sessions.
5. Fix the RealSense and the table: the head camera pose must not change between recording and deployment.
   Mark the robot and box positions on the table with tape.

### Command template

```bash
lerobot-record \
  --robot.type=bi_so_follower --robot.id=xlerobot \
  --robot.left_arm_config.port=/dev/ttyACM0 \
  --robot.right_arm_config.port=/dev/ttyACM1 \
  --robot.left_arm_config.cameras='{
    head:  {"type": "intelrealsense", "serial_number_or_name": "REALSENSE_SERIAL", "width": 640, "height": 480, "fps": 30},
    wrist: {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
  }' \
  --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": "/dev/video6", "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG"},
  }' \
  --teleop.type=bi_so_leader --teleop.id=xle_leader \
  --teleop.left_arm_config.port=/dev/ttyACM2 \
  --teleop.right_arm_config.port=/dev/ttyACM3 \
  --dataset.repo_id=${HF_USER}/xle_s2_open_and_hold_lid \
  --dataset.single_task="Open the box lid with the left arm and hold it open" \
  --dataset.num_episodes=50 --dataset.fps=30 \
  --dataset.episode_time_s=30 --dataset.reset_time_s=15 \
  --dataset.streaming_encoding=true --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --display_data=true
```

Ports, video indices and the RealSense serial are placeholders. One dataset per subtask keeps relabelling
and per-subtask LoRAs simple. They can be merged later with `lerobot-edit-dataset`.

## Teleoperation options

| Option | Data it gives | Fit |
|---|---|---|
| Two SO-101 leader arms (`bi_so_leader`) | joint space, both arms at once | best: matches FLUX's joint commands, enables real bimanual coordination |
| One leader arm | one arm per episode while the other holds a fixed pose | per-arm path only, no learned coordination |
| XLeVR VR teleop (XLeRobot repo) | end-effector targets | natural for EE50, but setup time and IK in the loop |

Open question for Oscar: how many leader arms can we use on the XLeRobot (the hackathon station has one)?

## Demo task proposal: "clear, open, pack, close"

Criteria: a long chain (scored higher), a real bimanual dependency, something the soft fin-ray fingers
are good at, a real-world framing (kitting, food packing), a fast reset, and success that a camera can check.
It also matches WS2 (move the obstacle to reach the target) and step 4 of Simone's plan.

| # | Subtask (instruction text) | Arm | Depends on | Checkable success |
|---|---|---|---|---|
| S1 | "Push the bottle away from the box" | left | none | box front is clear |
| S2 | "Open the box lid with the left arm and hold it open" | left | S1 | lid angle above threshold |
| S3 | "Put the apple in the box" | right | S2 still holding (WHILE) | apple inside the box volume |
| S4 | "Put the yellow block in the box" | right | S2 still holding | block inside the box |
| S5 | "Close the box lid" | left | S3, S4 | lid closed |
| S6 (stretch) | "Hand the cup to the right arm" | both | none | cup in the right gripper |

Each subtask is a separate FLUX call with its own instruction. The orchestrator (WS3) chains them, holds
the left arm during S3 and S4, and re-checks success with perception before moving on. With per-arm
FLUX, S2 to S4 is "one arm holds, the other acts", which works without learned coordination. A 12-D or
EE50 model could later merge S2 to S4 into one bimanual skill.

Alternative if the box and lid are unreliable: "sort three objects into two bins with one handover", which
needs no articulated prop.

## How much data, and the time split

- BFL used about 200 demonstrations for its SO-101 pick-and-place LoRA. The LeRobot FLUX preset holds out
  the last 20% of episodes per task for evaluation.
- Proposed first pass: **50 episodes per subtask for S2 and S3** (the coupled hold and insert), then train
  the first LoRA tonight while recording S1, S4 and S5.
- At about 25 s per episode plus 15 s reset, 50 episodes take about 35 minutes. All five subtasks at 50
  each is about 3 hours of teleoperation for two people (one teleoperates, one resets and logs failures).
- Variation per subtask: start positions on a 3x3 grid of zones, box yaw within about 20 degrees, 0 to 2
  distractors, two lighting conditions. Start each subtask from varied end states of the previous one,
  which is what makes chaining work. Record 10 to 15% recovery episodes (lid half open, apple dropped next
  to the box), as the BFL fine-tuning guide recommends normal starts, varied starts, disturbances and recovery.
- Justifying the split to the judges: public data (ws4-datasets report) covers generic bimanual SO-101 and
  handheld priors; our own episodes cover the station's viewpoint, the soft gripper and the exact props.
  We spend our own teleop time only where public data cannot substitute.

## Open questions

1. How many leader arms are available for the XLeRobot?
2. RealSense model and serial, wrist camera models (also asked by the ws4-sim agent).
3. Does the team want the XLeRobot task as the main hardware demo, or as the WS4 showcase next to the
   SO-101 station demo?
4. Box, lid and props: what do we actually have on site?
