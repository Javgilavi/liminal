#!/usr/bin/env bash
# Real-arm eval rows of one Haller skill (skills.py: pen, green, cutter, box), same protocol for every skill: start
# layouts from the held-out takes' start frames, haller home, verbatim prompt, every guard (1 deg/tick, keep-out with
# the skill's demo envelope, stall and load freeze, 60 % torque cap). Before each row the runner ramps to home and
# checks its live frames and joints against the skill's at-home references (home_check.py). One trials.jsonl line per
# attempt (the skill's trial task and graded steps). Pen rows (approved by Oscar 17 Oct night): baseline = B84 base
# (A0r-B84), median = stock weights + median stats (A0r-M), lora = B84 + the LoRA adapter; ROWS default "baseline
# median lora". Other skills: ROWS default "lora"; their package names on 8790 come from ws4-datasets.
#
#   POD_HOST=157.157.221.177 POD_PORT=15377 LORA=lora_5000_raw bash ws4/single_arm/eval_pen.sh session   # pen
#   SKILL=box POD_HOST=... POD_PORT=... LORA=<package> LORA_CHECKPOINT="<what it is>" \
#       bash ws4/single_arm/eval_skill.sh session
#
# The skill must be built first (build_skill.py <skill>: snapshot, keep-out replay, envelope, references); a skill
# whose demos the keep-out refused is not run (ALLOW_KEEPOUT_REFUSALS=1 overrides after a review).
# Optional env: ROWS, LAYOUTS (skill default), PER (attempts per listed layout, 1), ATTEMPT_S (40), START_AT (resume
# the first row at attempt N), HOME_REF (skill default; none = skip the home check), BASE / MEDIAN / LORA package
# names, *_CHECKPOINT and *_VARIANT trial-log strings, PARKED_TOL (deg the other arm may sit outside its pose range in
# the skill's demos, 5; the runner refuses to start otherwise), LORA_CHOICE (pre-registered adapter choice, written into the
# LoRA trials), REMOTE_PORT (8790), LOCAL_PORT (8776), TOKEN_FILE (/workspace/flux_token), NOTE.
# Median row gripper (M's gripper stats span 0..40, ours 0..100 with open = 100): M_START_GRIP (start-pose gripper,
# default 100; 40 starts inside M's range) or M_GRIP_SCALE (e.g. 0.4: model = 0.4 x robot gripper), not both.
# DRY_RUN=1 prints the runner commands and touches nothing. FAKE_ROBOT=1 runs against the real server with the fake
# arm: no arm port opened, nothing logged. ALIGN=1 adds --align (live station alignment at the home check),
# HOME_TAKE=N picks the reference take for it and the home check (default: the last held-out take). HOLD_OTHER (default 1 for the chain's pen-arm skills): the runner takes the
# box arm over from the HMI and holds it at its pose (--hold-other-arm). Run preflight.py --skill <skill> first. Placement page:
# http://127.0.0.1:8899 while the runner waits for Enter.
set -euo pipefail
MODE="${1:?usage: eval_skill.sh session|baseline|median|lora}"
: "${POD_HOST:?POD_HOST of the serving pod}" "${POD_PORT:?POD_PORT (ssh) of the serving pod}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
SKILL="${SKILL:-pen}"
SKILL_ENV="$(python ws4/single_arm/skills.py env "$SKILL")"
eval "$SKILL_ENV"
if [ "$SKILL_REFS" != 1 ] && [ -z "${ALLOW_NO_REFS:-}" ]; then  # ALLOW_NO_REFS: fake-robot pipeline runs only
  echo "skill $SKILL was built with --no-refs: run build_skill.py $SKILL again (with references) first" >&2; exit 1
fi
if [ "$SKILL_VALID" != 1 ] && [ -z "${ALLOW_KEEPOUT_REFUSALS:-}" ]; then
  echo "skill $SKILL: the keep-out refused demonstrated motion (skills/$SKILL/skill.json): review first" >&2; exit 1
fi
KEY="${KEY:-$HOME/.ssh/id_ed25519_osrdvs}"
REMOTE_PORT="${REMOTE_PORT:-8790}"
LOCAL_PORT="${LOCAL_PORT:-8776}"
ATTEMPT_S="${ATTEMPT_S:-40}"
START_AT="${START_AT:-1}"
TOKEN_FILE="${TOKEN_FILE:-/workspace/flux_token}"
HOME_REF="${HOME_REF:-$SKILL_HOME_REF}"
LAYOUTS="${LAYOUTS:-$SKILL_LAYOUTS}"
PER="${PER:-1}"
if [ "$SKILL" = pen ]; then
  BASE="${BASE:-base}"
  BASE_CHECKPOINT="${BASE_CHECKPOINT:-/workspace/packages/so101_haller_pen_B84 (A0r-B, 84 train takes, final split)}"
  BASE_VARIANT="${BASE_VARIANT:-A0r-B84}"
  DEFAULT_ROWS="baseline median lora"
else
  BASE_VARIANT="${BASE_VARIANT:-A0r-B}"
  DEFAULT_ROWS="lora"
fi
MEDIAN="${MEDIAN:-median}"
MEDIAN_CHECKPOINT="${MEDIAN_CHECKPOINT:-/workspace/packages/so101_haller_pen_M (zero-shot, multimodalart 09edebb median new-calibration stats; gripper range 0..40)}"
M_START_GRIP="${M_START_GRIP:-100}"
M_GRIP_SCALE="${M_GRIP_SCALE:-}"
MEDIAN_VARIANT="${MEDIAN_VARIANT:-A0r-M${M_GRIP_SCALE:+-g$M_GRIP_SCALE}}"
if [ -n "$M_GRIP_SCALE" ] && [ "$M_START_GRIP" != 100 ]; then  # the scale already maps our open 100 onto M's 40
  echo "use M_START_GRIP=40 or M_GRIP_SCALE=0.4, not both" >&2; exit 2
fi
case "$MODE" in
  session) read -r -a ROWS <<< "${ROWS:-$DEFAULT_ROWS}" ;;
  baseline|median|lora) ROWS=("$MODE") ;;
  *) echo "mode must be session, baseline, median or lora" >&2; exit 2 ;;
esac
if [[ " ${ROWS[*]} " == *" baseline "* ]]; then
  : "${BASE:?BASE package name on the server for skill $SKILL}" "${BASE_CHECKPOINT:?BASE_CHECKPOINT for the trial log}"
fi
if [[ " ${ROWS[*]} " == *" lora "* ]]; then
  : "${LORA:?LORA package name on the server (lora_[<skill>_]<microsteps>[_raw], or any name with LORA_VARIANT)}"
  if [[ "$LORA" =~ ^lora_(.*_)?([0-9]+)(_raw|_ema)?$ ]]; then
    n="${BASH_REMATCH[2]}"
    kind=EMA
    adapter_dir=pretrained_model_ema
    if [ "${BASH_REMATCH[3]}" = _raw ]; then kind=raw; adapter_dir=pretrained_model; fi
    if [ "$SKILL" = pen ]; then
      LORA_CHECKPOINT="${LORA_CHECKPOINT:-/workspace/real_lora/lora_pen84/checkpoints/$(printf %06d "$n")/$adapter_dir on so101_haller_pen_B84 (${kind^^}, $n microsteps)}"
    fi
    default_variant="LoRA-$n"
    (( n >= 1000 && n % 1000 == 0 )) && default_variant="LoRA-$((n / 1000))k"
    [ "$kind" = raw ] && default_variant="$default_variant-raw"
    LORA_VARIANT="${LORA_VARIANT:-$default_variant}"
    choice="$kind adapter of checkpoint $n"
  else  # a package named by ws4-datasets (e.g. skill_red_pen): the variant must be given
    : "${LORA_VARIANT:?LORA_VARIANT for package $LORA (e.g. LoRA-10k-raw)}"
    choice="package $LORA ($LORA_VARIANT)"
  fi
  : "${LORA_CHECKPOINT:?LORA_CHECKPOINT for the trial log (from ws4-datasets)}"
  # Pre-registered rule (orchestrator, 18 Oct): the LoRA row uses whichever adapter (raw or EMA) of the latest
  # checkpoint is best on the held-out protocol at session start. The choice goes into every LoRA trial's notes and
  # is printed before the first attempt.
  LORA_CHOICE="${LORA_CHOICE:-$choice, chosen before the session as the best of raw/EMA on the held-out takes}"
  echo "pre-registered LoRA row: package $LORA, variant $LORA_VARIANT; $LORA_CHOICE"
fi
read -r -a SKILL_EXTRA <<< "$SKILL_RUNNER_ARGS"  # per-skill runner flags recorded in skill.json (cutter: slow zone)
# ALIGN=1: after the ramp home, a live alignment page vs the home reference until Enter (station back on the
# recording's spot). Horizon, lift stall and prefetch come from WS4_ACTION_STEPS / WS4_LIFT_STALL_DEG /
# WS4_PREFETCH_TICKS in the environment (run_single_arm.py reads them).
[ -n "${ALIGN:-}" ] && [ "$HOME_REF" != none ] && SKILL_EXTRA+=(--align)
HOME_ARGS=()
[ "$HOME_REF" != none ] && HOME_ARGS=(--home-check "$HOME_REF")
[ "$HOME_REF" != none ] && [ -n "${HOME_TAKE:-}" ] && HOME_ARGS+=(--home-check-take "$HOME_TAKE")  # default: last take
read -r -a SKILL_GATE <<< "$SKILL_GATE_ARGS"  # pen-arm chain skills: geometric box-arm gate (gate.py)
read -r -a PARKED_LO <<< "$SKILL_PARKED_LO"
read -r -a PARKED_HI <<< "$SKILL_PARKED_HI"
ROBOT_ARGS=(--port "$SKILL_PORT" --robot-id "$SKILL_ROBOT_ID" --other-arm-port "$SKILL_OTHER_PORT" --other-arm-id "$SKILL_OTHER_ID"
            --other-arm-lo "${PARKED_LO[@]}" --other-arm-hi "${PARKED_HI[@]}" --other-arm-tol "${PARKED_TOL:-$SKILL_PARKED_TOL}"
            "${SKILL_GATE[@]}")
if [ "${HOLD_OTHER:-$SKILL_HOLD_OTHER}" = 1 ]; then
  ROBOT_ARGS+=(--hold-other-arm)  # the runner takes the box arm over (goal = its pose, then torque on) and holds it
fi
if [ -n "${FAKE_ROBOT:-}" ]; then
  ROBOT_ARGS=(--robot fake --no-log --no-video)
  if [ "${HOLD_OTHER:-$SKILL_HOLD_OTHER}" = 1 ] && [ -n "$SKILL_PARKED_MEDIAN" ]; then
    read -r -a FAKE_OTHER <<< "$SKILL_PARKED_MEDIAN"
    ROBOT_ARGS+=(--other-arm-port fake --hold-other-arm --fake-other-q "${FAKE_OTHER[@]}")
  fi
fi
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
    --wrist-cam "$SKILL_WRIST_CAM" --wrist-fourcc none --wrist-rotate 180 --task-side "$SKILL_ARM" \
    --keepout-config "$SKILL_WORKSPACE" --start-pose 0 0 0 0 0 "$grip" \
    --backend remote --server "http://127.0.0.1:$LOCAL_PORT" --package "$pkg" --token-cmd "$TOKEN_CMD" \
    --task "$SKILL_TASK" --trial-task "$SKILL_TRIAL_TASK" --steps "${SKILL_STEPS[@]}" \
    --layouts $LAYOUTS --per-layout "$PER" --start-at "$start" --layout-ref "$SKILL_LAYOUT_REF" "${HOME_ARGS[@]}" \
    --condition "$condition" --variant "$variant" --checkpoint "$ckpt" \
    --seconds "$ATTEMPT_S" --max-step 1.0 --max-relative-target 20 \
    --guidance-note "1.0 + compile as served ($pkg)" \
    --note "$SKILL eval, $row row: layouts [$LAYOUTS] from held-out start frames, haller home, keep-out with the skill's demo envelope${note:+; $note}${NOTE:+; $NOTE}" \
    "${SKILL_EXTRA[@]}" "$@"
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
    lora) run_row lora "$LORA" "$LORA_CHECKPOINT" "$LORA_VARIANT" post_training "$start" 100 "$LORA_CHOICE" ;;
  esac
  start=1
done
