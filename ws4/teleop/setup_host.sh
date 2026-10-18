#!/usr/bin/env bash
# ws4/teleop/setup_host.sh: one-time sudo setup of the robot laptop. Run it in a normal terminal (Ctrl+Alt+T):
# the Claude app's terminal sets "no new privileges", so sudo cannot run there.
#   - adb
#   - a udev rule for Meta/Oculus USB devices (vendor 2833): Ubuntu's 51-android.rules lists 132 vendors but not
#     Oculus, so without it adb reports "no permissions" for the Quest
#   - haller_ws udev rules as of origin/main (2026-09-11: /dev/haller_arm_left|right, /dev/haller_cam_wrist_*)
set -euo pipefail
sudo apt install -y adb
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev", TAG+="uaccess"' \
  | sudo tee /etc/udev/rules.d/51-oculus-quest.rules >/dev/null
sudo cp "$HOME/haller_ws/scripts/99-haller-devices.rules" /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
adb kill-server >/dev/null 2>&1 || true
sleep 2
adb devices -l
echo "If the Quest shows 'unauthorized': put the headset on and accept 'Allow USB debugging' (tick 'always')."
echo "If it still shows 'no permissions': unplug and replug the USB-C cable, then run: adb devices -l"
