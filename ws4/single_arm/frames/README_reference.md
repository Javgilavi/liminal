# Reference frames for the Haller camera parity check (2026-10-17 night)

Source of truth: the recorded dataset, not a live capture. The Haller was handed to a
teammate for the night right after recording, so the reference is the first frames of the
held-out takes, which went through the exact recording path.

Dataset: `~/.cache/huggingface/lerobot/ws4/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box`
(104 episodes, split in `ws4_split.json` at its root).

## Which frames

Held-out layout D takes 101, 102 and 103. At frame 0 of each, the task arm (haller `right`,
robot LEFT) is at home, state `[0, 2.5, 2, 0.4, 0, 99.3]` deg (gripper open), with the pen
on the black mat and the box in layout D. The arm holds still for:

| episode | frames | still at home |
|---|---|---|
| 101 | 304 | 0 to 15 |
| 102 | 256 | 0 to 20 |
| 103 | 244 | 0 to 19 |

Use frames 0 to 15 of any of them. Ep 103 is the cleanest: nobody walks through the head view.

## Files in `reference_2026-10-17_night/`

- `ep10{1,2,3}_f0000_head.png`: `observation.images.head`, D455 colour, unrotated
- `ep10{1,2,3}_f0000_task_wrist_rot180.png`: `observation.images.right_wrist`, wrist cam on
  USB path 7.2, already rotated 180 deg (as recorded)
- `manifest.json`: still-frame ranges, frame-0 state, sha256 per PNG

Decoded with LeRobot 0.5.1 (`LeRobotDataset(...)[i]`, the training path), saved lossless.
There is no "other wrist" (USB 7.3) reference: that camera was never recorded.

## What the live path must match

- lerobot `OpenCVCamera`, 640x480 at 30 fps, `connect(warmup=True)`, no V4L2 control writes
  or colour gains
- head: `/dev/haller_cam_mast` (D455 colour node), no rotation
- task wrist: `/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0`, then
  `Cv2Rotation.ROTATE_180`. Both wrist cams report serial SN0001, so the USB path is the only
  way to pick the right one.
- arm at home (above), box in layout D, room lit as in the evening session

At home the wrist cam looks out over the room, so people and the background change between
takes. Compare the gripper jaw region and overall colour statistics, not the background.
