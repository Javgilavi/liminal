#!/usr/bin/env bash
# Real-arm pen eval on the Haller (orchestrator protocol, approved by Oscar 17 Oct night): parity + calibration checks,
# then graded rows with the same layouts (A B C D A: held-out takes 20, 36, 86, 101, 21), 40 s budget, verbatim prompt
# and every guard. Rows: baseline = B84 base (A0r-B84, zero_shot); median = stock weights + median stats of
# current-calibration SO-101 datasets (A0r-M, zero_shot); lora = B84 + the LoRA adapter (LoRA-<N>k, post_training).
# Before each row the runner ramps to haller home and checks its live frames and joints against the at-home
# references (home_check.py). One trials.jsonl line per attempt (task put_pen_in_slot, steps reach / grasp / insert).
#
# Server: real-serve-6000, ws4-datasets' one server on port 8790 with `base`, `lora_<N>` and (if ws4-datasets adds
# it) `median`, so switching rows costs no server restart.
#
#   POD_HOST=157.157.221.177 POD_PORT=15377 LORA=lora_5000 bash ws4/single_arm/eval_pen.sh session
#       # Oscar's order: B84, median, LoRA. `median` not served yet: B84, LoRA, then median at the end if served by
#       # then (re-checked), else skipped with a message.
#   POD_HOST=... POD_PORT=... bash ws4/single_arm/eval_pen.sh baseline|median                        # one row
#   POD_HOST=... POD_PORT=... LORA=lora_5000 bash ws4/single_arm/eval_pen.sh lora
#
# Optional env: ROWS (session rows, "baseline median lora"), LAYOUTS ("A B C D A"), PER (attempts per listed layout, 1),
# ATTEMPT_S (40), START_AT (resume the first row at attempt N), HOME_REF (at-home references, default
# ws4/single_arm/frames/reference_2026-10-17_night; none = skip the home check), BASE / MEDIAN package names (base,
# median), *_CHECKPOINT and *_VARIANT trial-log overrides, REMOTE_PORT (8790), LOCAL_PORT (8776), TOKEN_FILE
# (/workspace/flux_token), NOTE (appended to every trial's notes).
# Median row gripper (M's gripper stats span 0..40, ours 0..100 with open = 100): M_START_GRIP (start-pose gripper,
# default 100 like every row; 40 starts inside M's range), M_GRIP_SCALE (e.g. 0.4: model = 0.4 x robot gripper, so
# our 0..100 maps onto M's 0..40; variant gets a -g<scale> suffix). Both are written into the trial notes.
# DRY_RUN=1 prints the runner commands and touches nothing. FAKE_ROBOT=1 runs against the real server with the fake
# arm: no arm port opened, nothing logged (pipeline rehearsal). Run preflight.py first. Placement page:
# http://127.0.0.1:8899 while the runner waits for Enter.
set -euo pipefail
MODE="${1:?usage: eval_pen.sh session|baseline|median|lora}"
: "${POD_HOST:?POD_HOST of the serving pod}" "${POD_PORT:?POD_PORT (ssh) of the serving pod}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
KEY="${KEY:-$HOME/.ssh/id_ed25519_osrdvs}"
REMOTE_PORT="${REMOTE_PORT:-8790}"
LOCAL_PORT="${LOCAL_PORT:-8776}"
ATTEMPT_S="${ATTEMPT_S:-40}"
START_AT="${START_AT:-1}"
TOKEN_FILE="${TOKEN_FILE:-/workspace/flux_token}"
HOME_REF="${HOME_REF:-ws4/single_arm/frames/reference_2026-10-17_night}"
LAYOUTS="${LAYOUTS:-A B C D A}"
PER="${PER:-1}"
BASE="${BASE:-base}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/packages/so101_haller_pen_B84 (A0r-B, 84 train takes, final split)}"
BASE_VARIANT="${BASE_VARIANT:-A0r-B84}"
MEDIAN="${MEDIAN:-median}"
MEDIAN_CHECKPOINT="${MEDIAN_CHECKPOINT:-/workspace/packages/so101_haller_pen_M (stock so101 + median stats of current-calibration SO-101 datasets, multimodalart Space 09edebb)}"
M_START_GRIP="${M_START_GRIP:-100}"
M_GRIP_SCALE="${M_GRIP_SCALE:-}"
MEDIAN_VARIANT="${MEDIAN_VARIANT:-A0r-M${M_GRIP_SCALE:+-g$M_GRIP_SCALE}}"
if [ -n "$M_GRIP_SCALE" ] && [ "$M_START_GRIP" != 100 ]; then  # the scale already maps our open 100 onto M's 40
  echo "use M_START_GRIP=40 or M_GRIP_SCALE=0.4, not both" >&2; exit 2
fi
case "$MODE" in
  session) read -r -a ROWS <<< "${ROWS:-baseline median lora}" ;;
  baseline|median|lora) ROWS=("$MODE") ;;
  *) echo "mode must be session, baseline, median or lora" >&2; exit 2 ;;
esac
if [[ " ${ROWS[*]} " == *" lora "* ]]; then
  : "${LORA:?LORA package name on the server, e.g. lora_5000}"
  n="${LORA#lora_}"
  [[ "$n" =~ ^[0-9]+$ ]] || { echo "LORA must look like lora_<microsteps>" >&2; exit 2; }
  LORA_CHECKPOINT="${LORA_CHECKPOINT:-/workspace/real_lora/lora_pen84/checkpoints/$(printf %06d "$n")/pretrained_model_ema on so101_haller_pen_B84 (EMA, $n microsteps)}"
  default_variant="LoRA-$n"
  (( n >= 1000 && n % 1000 == 0 )) && default_variant="LoRA-$((n / 1000))k"
  LORA_VARIANT="${LORA_VARIANT:-$default_variant}"
fi
HOME_ARGS=()
[ "$HOME_REF" != none ] && HOME_ARGS=(--home-check "$HOME_REF")
ROBOT_ARGS=(--port /dev/haller_arm_right --robot-id haller_right --other-arm-port /dev/haller_arm_left)
[ -n "${FAKE_ROBOT:-}" ] && ROBOT_ARGS=(--robot fake --no-log --no-video)
cd "$REPO"
RUN=(python ws4/single_arm/run_single_arm.py)
[ -n "${DRY_RUN:-}" ] && RUN=(echo python ws4/single_arm/run_single_arm.py)  # print, touch nothing
TOKEN_CMD="ssh -i $KEY -p $POD_PORT -o ConnectTimeout=10 root@$POD_HOST cat $TOKEN_FILE"
PACKAGES=()
for row in "${ROWS[@]}"; do
  case "$row" in
    baseline) PACKAGES+=("$BASE") ;;
    median) PACKAGES+=("$MEDIAN") ;;
    lora) PACKAGES+=("$LORA") ;;
    *) echo "unknown row $row (baseline, median, lora)" >&2; exit 2 ;;
  esac
done

check_packages() {  # contracts to stderr; prints the names the server does not know (404), fails on anything else
  python - "http://127.0.0.1:$LOCAL_PORT" "$TOKEN_CMD" "$@" <<'EOF'
import shlex, subprocess, sys
sys.path.insert(0, "ws4")
from flux.remote import RemoteBackend
server, cmd, *packages = sys.argv[1:]
token = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True).stdout.strip()
missing = []
for package in packages:
    try:
        c = RemoteBackend(server, package, token=token, timeout_s=30).contract
    except RuntimeError as e:
        if "404" not in str(e):
            raise
        print(f"not served: {package}", file=sys.stderr)
        missing.append(package)
        continue
    print(f"server ok: {package} -> action_dim {c.action_dim}, chunk {c.chunk_size}, cameras {c.camera_keys}",
          file=sys.stderr)
    assert c.action_dim == 6, f"{package} is not a per-arm SO-101 package"
print(" ".join(missing))
EOF
}

if [ -z "${DRY_RUN:-}" ]; then
  if [ -z "${FAKE_ROBOT:-}" ]; then
    busy=$(fuser "$(readlink -f /dev/haller_arm_right)" "$(readlink -f /dev/haller_arm_left)" 2>/dev/null || true)
    if [ -n "$busy" ]; then echo "PORT BUSY (pid $busy): stop the HMI or recorder first" >&2; exit 1; fi
  fi
  ssh -N -L "$LOCAL_PORT:127.0.0.1:$REMOTE_PORT" -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
    -i "$KEY" -p "$POD_PORT" -o ConnectTimeout=10 "root@$POD_HOST" &
  TUNNEL=$!
  trap 'kill $TUNNEL 2>/dev/null || true' EXIT
  for _ in $(seq 30); do ss -ltn | grep -q "127.0.0.1:$LOCAL_PORT " && break; sleep 0.5; done
  # Every package of the plan must be served before the arm is touched (the token stays inside the process). Only
  # `median` may be missing (ws4-datasets adds it after its offline check): its row then moves to the end.
  missing=$(check_packages "${PACKAGES[@]}")
  for pkg in $missing; do
    if [ "$pkg" != "$MEDIAN" ]; then echo "package $pkg is not served on $REMOTE_PORT: stop" >&2; exit 1; fi
  done
  if [ -n "$missing" ] && [ "${#ROWS[@]}" -gt 1 ]; then
    kept=(); for row in "${ROWS[@]}"; do [ "$row" != median ] && kept+=("$row"); done
    ROWS=("${kept[@]}" median)
    echo ">>> $MEDIAN is not served yet: rows become ${ROWS[*]} (median re-checked before its row)"
  elif [ -n "$missing" ]; then
    echo "package $MEDIAN is not served on $REMOTE_PORT yet" >&2; exit 1
  fi
fi

run_row() {  # row package checkpoint variant condition start_at start_grip note [extra runner args...]
  local row=$1 pkg=$2 ckpt=$3 variant=$4 condition=$5 start=$6 grip=$7 note=$8
  shift 8
  echo "=== $row row: package $pkg as $variant ($condition), layouts [$LAYOUTS] x $PER, ${ATTEMPT_S} s each, start at $start"
  "${RUN[@]}" "${ROBOT_ARGS[@]}" \
    --scene-cam /dev/haller_cam_mast --scene-fourcc none \
    --wrist-cam /dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0 --wrist-fourcc none --wrist-rotate 180 \
    --keepout-config ws4/single_arm/calibration/haller_right_workspace_final.json --start-pose 0 0 0 0 0 "$grip" \
    --backend remote --server "http://127.0.0.1:$LOCAL_PORT" --package "$pkg" --token-cmd "$TOKEN_CMD" \
    --task "put the red pen in its slot in the cardboard box" --trial-task put_pen_in_slot \
    --layouts $LAYOUTS --per-layout "$PER" --start-at "$start" --layout-ref ws4/single_arm/layouts "${HOME_ARGS[@]}" \
    --condition "$condition" --variant "$variant" --checkpoint "$ckpt" \
    --seconds "$ATTEMPT_S" --max-step 1.0 --max-relative-target 20 \
    --guidance-note "1.0 + compile as served ($pkg)" \
    --note "pen eval 17 Oct night, $row row: layouts [$LAYOUTS] from held-out start frames, haller home, keep-out with the final-split envelope${note:+; $note}${NOTE:+; $NOTE}" \
    "$@"
}

start="$START_AT"
for i in "${!ROWS[@]}"; do
  if [ "$i" -gt 0 ] && [ -t 0 ] && [ -z "${DRY_RUN:-}" ]; then
    read -r -p ">>> ${ROWS[$((i - 1))]} row done, arm parked. Enter = start the ${ROWS[$i]} row, Ctrl-C = stop here " _
  fi
  case "${ROWS[$i]}" in
    baseline) run_row baseline "$BASE" "$BASE_CHECKPOINT" "$BASE_VARIANT" zero_shot "$start" 100 "" ;;
    median)
      if [ -z "${DRY_RUN:-}" ] && [ "$i" -gt 0 ] && [ -n "$(check_packages "$MEDIAN")" ]; then
        echo ">>> $MEDIAN is still not served: median row skipped (run eval_pen.sh median when it is)"
        continue
      fi
      m_note="M gripper stats 0..40 vs ours 0..100: start gripper $M_START_GRIP"
      m_args=()
      if [ -n "$M_GRIP_SCALE" ]; then
        m_note="$m_note, gripper scale $M_GRIP_SCALE (model = $M_GRIP_SCALE x robot)"
        m_args=(--scale 1 1 1 1 1 "$M_GRIP_SCALE")
      fi
      run_row median "$MEDIAN" "$MEDIAN_CHECKPOINT" "$MEDIAN_VARIANT" zero_shot "$start" "$M_START_GRIP" "$m_note" \
        "${m_args[@]}" ;;
    lora) run_row lora "$LORA" "$LORA_CHECKPOINT" "$LORA_VARIANT" post_training "$start" 100 "" ;;
  esac
  start=1
done
