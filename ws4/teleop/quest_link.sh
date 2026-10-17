#!/usr/bin/env bash
# ws4/teleop/quest_link.sh: (re)connect the Quest over USB for ws4/teleop. Run after every replug or headset sleep,
# in a normal terminal (Ctrl+Alt+T). Asks for the sudo password the first time it installs the udev rule.
#   1. udev rule for Meta's USB vendor (2833), so the node is user-accessible on every reconnect
#   2. chmod of the current node, for this connection
#   3. adb reverse of the frontend and backend ports: the headset's localhost:3001 / :8000 are this laptop's
set -u
RULE=/etc/udev/rules.d/51-oculus-quest.rules
NODES=$(lsusb -d 2833: | awk '{print "/dev/bus/usb/"$2"/"substr($4,1,3)}')
[ -n "$NODES" ] || { echo ">>> NO QUEST ON USB: plug the USB-C cable in and wake the headset"; exit 1; }
if [ ! -f "$RULE" ]; then
  sudo sh -c "echo 'SUBSYSTEM==\"usb\", ATTR{idVendor}==\"2833\", MODE=\"0666\"' > $RULE && udevadm control --reload-rules" \
    || { echo ">>> FAILED: sudo did not run"; exit 1; }
fi
for n in $NODES; do
  [ -w "$n" ] || sudo chmod 666 "$n" || { echo ">>> FAILED: could not open $n"; exit 1; }
done
adb kill-server >/dev/null 2>&1
echo "waiting for the headset (accept 'Allow USB debugging' inside it if asked)..."
adb wait-for-device
# Keep the headset awake when it is not on a face (controllers-only driving, headset on the table). Lasts until
# the headset reboots; undo with: adb shell am broadcast -a com.oculus.vrpowermanager.automation_disable
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close >/dev/null 2>&1
adb reverse tcp:3001 tcp:3001 >/dev/null && adb reverse tcp:8000 tcp:8000 >/dev/null \
  && echo ">>> QUEST LINKED: open http://localhost:3001/teleop/vr in the Quest browser" \
  || echo ">>> adb reverse failed: run this script again"
