#!/usr/bin/env bash
# ws4/teleop/quest_usb.sh: give the logged-in user USB access to the Quest (Meta vendor 2833) for adb.
# Run in a normal terminal (Ctrl+Alt+T); it asks for the sudo password once.
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0660", GROUP="plugdev", TAG+="uaccess"' \
  | sudo tee /etc/udev/rules.d/51-oculus-quest.rules >/dev/null || { echo ">>> FAILED: sudo did not run"; exit 1; }
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=usb --action=add
adb kill-server >/dev/null 2>&1
sleep 2
if adb devices | grep -q $'\tdevice$'; then
  echo ">>> QUEST OK"
elif adb devices | grep -q unauthorized; then
  echo ">>> ALMOST: put the headset on and accept 'Allow USB debugging'"
else
  echo ">>> STILL NO PERMISSION: unplug and replug the Quest USB-C cable, then run this script again"
fi
