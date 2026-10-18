#!/usr/bin/env bash
# ws4/teleop/up.sh: bring up Quest teleop + the FLUX-contract recorder on THIS laptop.
#
#   ws4/teleop/up.sh --sim            # haller MuJoCo arms, nothing physical moves
#   ws4/teleop/up.sh --real           # both real arms (configs/station-real.yaml)
#   add --wifi to serve the headset over Wi-Fi through haller_ws scripts/quest-teleop/up.sh (needs caddy);
#   the default is USB: `adb reverse` makes the laptop's ports the headset's localhost, and http://localhost is a
#   secure context, so WebXR works with no certificate, no Caddy and no venue Wi-Fi.
#
# Stop with ws4/teleop/down.sh (it waits for the dataset to be finalised; never kill -9 the backend).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HALLER_WS="${HALLER_WS:-$HOME/haller_ws}"
RUN_DIR="${WS4_TELEOP_RUN_DIR:-/tmp/ws4-teleop}"
BACKEND_PORT="${HALLER_BACKEND_PORT:-8000}"
NEXT_PORT="${WS4_NEXT_PORT:-3001}"
VCODEC="${WS4_VCODEC:-libsvtav1}"
SESSION_HZ="${WS4_SESSION_HZ:-}"   # empty = backend.py default (30.08, see SESSION_HZ there)
MODE="" ; LINK="usb"
for a in "$@"; do
  case "$a" in
    --sim) MODE=sim ;;
    --real) MODE=real ;;
    --wifi) LINK=wifi ;;
    --usb) LINK=usb ;;
    *) echo "usage: up.sh --sim|--real [--usb|--wifi]" >&2; exit 2 ;;
  esac
done
[ -n "$MODE" ] || { echo "usage: up.sh --sim|--real [--usb|--wifi]" >&2; exit 2; }
CONFIG="$HERE/configs/station-$MODE.yaml"
mkdir -p "$RUN_DIR"
say() { printf '\033[1m[ws4-teleop]\033[0m %s\n' "$*"; }

if [ "$MODE" = real ]; then
  missing=0
  for dev in /dev/haller_arm_left /dev/haller_arm_right /dev/haller_cam_mast /dev/haller_cam_wrist_left \
             /dev/haller_cam_wrist_right; do
    [ -e "$dev" ] || { say "missing $dev (udev rules installed? board plugged in?)"; missing=1; }
  done
  python3 -I "$HERE/calib.py" status | sed 's/^/  /'
  for f in haller_left haller_right; do
    [ -f "$HOME/.cache/huggingface/lerobot/calibration/robots/so_follower/$f.json" ] || {
      say "no $f.json: LeRobot would ask to calibrate, or write a wrong file into the servos. Run calib.py harvest."
      missing=1; }
  done
  [ "$missing" = 0 ] || exit 1
fi

# ---- backend ----------------------------------------------------------------------------------------------------
if curl -sm 2 "http://127.0.0.1:$BACKEND_PORT/health" | grep -q '"ok"'; then
  say "a backend is already up on :$BACKEND_PORT; stop it with down.sh if it is not this config"
else
  say "starting backend ($MODE, $(basename "$CONFIG"), vcodec $VCODEC) on :$BACKEND_PORT"
  (
    cd "$HALLER_WS/hmi/backend"
    # activate-haller-hmi sources ROS 2 Jazzy: haller_hmi.ros_bridge imports rclpy at module top.
    nohup bash -c "source \"$HOME/venvs/haller-hmi/bin/activate-haller-hmi\" && \
      WS4_TELEOP_RUN_DIR=\"$RUN_DIR\" exec python \"$HERE/backend.py\" --config \"$CONFIG\" \
      --vcodec \"$VCODEC\" --port \"$BACKEND_PORT\" ${SESSION_HZ:+--session-hz $SESSION_HZ}" \
      >"$RUN_DIR/backend.log" 2>&1 &
  )
  for _ in $(seq 1 60); do
    sleep 1
    curl -sm 2 "http://127.0.0.1:$BACKEND_PORT/health" | grep -q '"ok"' && break
  done
  curl -sm 2 "http://127.0.0.1:$BACKEND_PORT/health" | grep -q '"ok"' \
    || { say "backend failed, see $RUN_DIR/backend.log"; tail -20 "$RUN_DIR/backend.log"; exit 1; }
  say "backend up: $(curl -s "http://127.0.0.1:$BACKEND_PORT/health")"
fi

if [ "$LINK" = wifi ]; then
  command -v caddy >/dev/null || { say "--wifi needs caddy (haller up.sh serves HTTPS through it)"; exit 1; }
  flag=--local; [ "$MODE" = sim ] && flag=--sim
  IP="$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')"
  say "handing over to haller up.sh $flag (it reuses the backend above), origin $IP"
  HALLER_DESKTOP_IP="$IP" exec "$HALLER_WS/scripts/quest-teleop/up.sh" "$flag"
fi

# ---- frontend (Next dev, backend URL baked at start) -------------------------------------------------------------
FRONT="$HALLER_WS/hmi/frontend"
URL="http://localhost:$BACKEND_PORT"
STAMP="$FRONT/.next/.haller-origin"
if curl -sm 2 "http://localhost:$NEXT_PORT" >/dev/null 2>&1; then
  [ "$(cat "$STAMP" 2>/dev/null)" = "$URL" ] || say "WARNING: :$NEXT_PORT is up with another backend URL baked in; down.sh, then up.sh"
else
  # Same rule as haller up.sh: Turbopack caches NEXT_PUBLIC_BACKEND_URL in .next/dev, so a changed URL needs
  # that cache gone, and the stamp records what is baked.
  if [ -d "$FRONT/.next/dev" ] && [ "$(cat "$STAMP" 2>/dev/null)" != "$URL" ]; then rm -rf "$FRONT/.next/dev"; fi
  mkdir -p "$FRONT/.next"; printf '%s\n' "$URL" >"$STAMP"
  say "starting Next.js on :$NEXT_PORT (backend $URL)"
  # setsid: its own process group (pnpm -> sh -> next-server), so down.sh can stop all of it at once.
  ( cd "$FRONT"
    NEXT_PUBLIC_BACKEND_URL="$URL" setsid nohup pnpm dev -p "$NEXT_PORT" >"$RUN_DIR/next.log" 2>&1 </dev/null &
    echo $! >"$RUN_DIR/next.pid" )
  for _ in $(seq 1 90); do sleep 1; curl -sm 2 "http://localhost:$NEXT_PORT" >/dev/null 2>&1 && break; done
fi

# ---- headset over USB --------------------------------------------------------------------------------------------
if command -v adb >/dev/null; then
  if adb devices | awk 'NR>1 && $2=="device"' | grep -q .; then
    adb reverse "tcp:$NEXT_PORT" "tcp:$NEXT_PORT" >/dev/null
    adb reverse "tcp:$BACKEND_PORT" "tcp:$BACKEND_PORT" >/dev/null
    say "adb reverse :$NEXT_PORT and :$BACKEND_PORT set on $(adb devices | awk 'NR==2{print $1}')"
  else
    say "adb sees no authorised headset: plug it in, accept 'Allow USB debugging', re-run up.sh"
  fi
else
  say "adb not installed (sudo apt install adb): no USB link to the headset"
fi
say "in the Quest browser open:  http://localhost:$NEXT_PORT/teleop/vr"
say "desktop cockpit:            http://localhost:$NEXT_PORT/   (Dataset tab: task string + HF user)"
