# VR teleop: what I need from you (ws4-teleop, Sat 17 Oct ~14:45)

Status: the whole recording pipeline works in sim on this laptop. haller_ws backend + your Quest stack, a
30 fps session, AV1 video, 640x480 head + both wrists, then the per-arm FLUX split, a LeRobot merge and a load
in the LeRobot that trains FLUX. vr_smoke passes 48/48.

I moved `~/haller_ws` to a new local branch `ws4-hackathon` at `origin/main` (your 2026-09-11 commits:
corrected left/right udev names, USB gripper cams, `haller_left` / `haller_right` calibration ids). Your local
`main` is untouched. Nothing is committed or pushed.

## Please do (needs sudo, 2 minutes)

1. adb, for the headset over USB (`localhost` is a secure context, so no Caddy, no certificate, no venue Wi-Fi):

   ```bash
   sudo apt install -y adb
   ```

   Quest Pro: developer mode on, plug the USB-C cable into this laptop, accept "Allow USB debugging" in the
   headset.
2. Your udev rules on this laptop are the pre-2026-09-11 version (no `/dev/haller_arm_left` or
   `/dev/haller_cam_wrist_left` names). Install the current ones:

   ```bash
   sudo cp ~/haller_ws/scripts/99-haller-devices.rules /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

## Please answer (one line each is fine)

1. Calibration. The current config uses `haller_left.json` and `haller_right.json` (harvested from EEPROM on
   2026-09-11). They are not on this laptop; only the May files `haller_follower.json` / `haller_leader.json`
   are, and your own config notes say `haller_follower.json` would write a bad calibration into the left
   arm. Can you copy the two 09-11 files here, or should I read them from the servos (read only) when the robot
   is plugged in? Did anyone recalibrate today?
2. Is the robot going to be driven from this laptop (both arm boards, the D455 and both gripper cams on USB
   here)? Is it plugged in yet?
3. Which task (or tasks) do we record first, and the exact instruction strings? They get frozen per dataset.

When adb is in and the headset is plugged in, tell me in the session and I will start the sim drive.
