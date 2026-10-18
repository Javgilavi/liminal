#!/usr/bin/env bash
# ws4/teleop/down.sh: stop what up.sh started. The backend gets SIGTERM and up to 120 s: its lifespan shutdown
# writes the parquet footers and meta/episodes. Killing it harder there destroys the session's takes.
set -uo pipefail
RUN_DIR="${WS4_TELEOP_RUN_DIR:-/tmp/ws4-teleop}"
HALLER_WS="${HALLER_WS:-$HOME/haller_ws}"
say() { printf '\033[1m[ws4-teleop]\033[0m %s\n' "$*"; }

if [ -f "$RUN_DIR/next.pid" ]; then
  kill -TERM -- "-$(cat "$RUN_DIR/next.pid")" 2>/dev/null || kill -TERM "$(cat "$RUN_DIR/next.pid")" 2>/dev/null
  rm -f "$RUN_DIR/next.pid"; say "frontend stopped"
fi
[ -x "$HALLER_WS/scripts/quest-teleop/down.sh" ] && [ -f /tmp/haller-quest/caddy.pid ] \
  && "$HALLER_WS/scripts/quest-teleop/down.sh" >/dev/null 2>&1
command -v adb >/dev/null && adb reverse --remove-all >/dev/null 2>&1

PID="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['pid'])" "$RUN_DIR/backend.json" 2>/dev/null)"
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  if curl -sm 2 http://127.0.0.1:"${HALLER_BACKEND_PORT:-8000}"/record/status | grep -q '"recording":true'; then
    say "a take is still open; it will be saved by the shutdown (the recorder saves on teleop stop)"
  fi
  kill -TERM "$PID"
  for i in $(seq 1 120); do
    kill -0 "$PID" 2>/dev/null || { say "backend finalised and stopped after ${i}s"; exit 0; }
    [ $((i % 10)) = 0 ] && say "still finalising, do NOT kill it ($i s)"
    sleep 1
  done
  say "backend still running after 120 s; check $RUN_DIR/backend.log before doing anything else"
  exit 1
fi
say "no backend from up.sh running"
