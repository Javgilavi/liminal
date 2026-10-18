#!/usr/bin/env bash
# Live ACT attempts on the Haller with the eval kit's flags (eval_skill.sh's row), but the policy in process
# (run_single_arm.py --backend act) instead of a FLUX server. Every guard as for FLUX: keep-out with the skill's
# demo envelope, soft limits, 1 deg/tick, ramps to haller home, home check, stall and load freeze, torque cap.
# Skills whose box arm holds the lid (pen_lid) take it over and hold it (--hold-other-arm); pen leaves it tucked.
# Keys during an attempt: space / q / Enter = STOP.
#
#   SKILL=pen bash ws4/act/eval_act_arm.sh     # original pen84 scene: layouts A B C D A, box arm tucked
#   SKILL=pen_lid OTHER_TOL=20 LAYOUTS="P3 P3 P3" bash ws4/act/eval_act_arm.sh   # lid held by the box arm
#
# Env: SKILL (pen), ACT_CKPT, VARIANT, LAYOUTS (the skill's), ATTEMPT_S (40), ACT_STEPS (32), OTHER_TOL (the
# skill's), HOME_REF (the skill's; none = skip), NOTE. FAKE_ROBOT=1 = fake arm, nothing logged.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
SKILL="${SKILL:-pen}"
eval "$(python ws4/single_arm/skills.py env "$SKILL")"
ACT_CKPT="${ACT_CKPT:-ws4/.cache/act/act_pen84_100k}"
VARIANT="${VARIANT:-ACT-pen84-100k}"
LAYOUTS="${LAYOUTS:-$SKILL_LAYOUTS}"
ATTEMPT_S="${ATTEMPT_S:-40}"
ACT_STEPS="${ACT_STEPS:-32}"  # re-plan every 32 commands: the horizon the offline eval scored, FLUX's cadence
OTHER_TOL="${OTHER_TOL:-$SKILL_PARKED_TOL}"  # pen_lid: Oscar's override was 20 (18 Oct)
HOME_REF="${HOME_REF:-$SKILL_HOME_REF}"
HOME_ARGS=()
[ "$HOME_REF" != none ] && HOME_ARGS=(--home-check "$HOME_REF")
read -r -a PARKED_LO <<< "$SKILL_PARKED_LO"
read -r -a PARKED_HI <<< "$SKILL_PARKED_HI"
read -r -a GATE <<< "$SKILL_GATE_ARGS"
read -r -a EXTRA <<< "$SKILL_RUNNER_ARGS"
ROBOT=(--port "$SKILL_PORT" --robot-id "$SKILL_ROBOT_ID" --other-arm-port "$SKILL_OTHER_PORT"
       --other-arm-id "$SKILL_OTHER_ID" --other-arm-lo "${PARKED_LO[@]}" --other-arm-hi "${PARKED_HI[@]}"
       --other-arm-tol "$OTHER_TOL" "${GATE[@]}")
[ "$SKILL_HOLD_OTHER" = 1 ] && ROBOT+=(--hold-other-arm)
if [ -n "${FAKE_ROBOT:-}" ]; then
  read -r -a FAKE_OTHER <<< "$SKILL_PARKED_MEDIAN"
  ROBOT=(--robot fake --no-log --no-video)
  [ "$SKILL_HOLD_OTHER" = 1 ] && ROBOT+=(--other-arm-port fake --hold-other-arm --fake-other-q "${FAKE_OTHER[@]}")
else
  busy=$(fuser "$(readlink -f "$SKILL_PORT")" "$(readlink -f "$SKILL_OTHER_PORT")" 2>/dev/null || true)
  if [ -n "$busy" ]; then echo "PORT BUSY (pid $busy): stop the HMI or recorder first" >&2; exit 1; fi
fi
python ws4/single_arm/run_single_arm.py "${ROBOT[@]}" \
  --scene-cam /dev/haller_cam_mast --scene-fourcc none \
  --wrist-cam "$SKILL_WRIST_CAM" --wrist-fourcc none --wrist-rotate 180 --task-side "$SKILL_ARM" \
  --keepout-config "$SKILL_WORKSPACE" --start-pose 0 0 0 0 0 100 \
  --backend act --act-checkpoint "$ACT_CKPT" --act-device auto --act-steps "$ACT_STEPS" \
  --task "$SKILL_TASK" --trial-task "$SKILL_TRIAL_TASK" --steps "${SKILL_STEPS[@]}" \
  --layouts $LAYOUTS --per-layout 1 --layout-ref "$SKILL_LAYOUT_REF" "${HOME_ARGS[@]}" \
  --condition post_training --variant "$VARIANT" \
  --seconds "$ATTEMPT_S" --max-step 1.0 --max-relative-target 20 \
  --guidance-note "ACT in process on the laptop GPU, act-steps $ACT_STEPS" \
  --note "$SKILL live, ACT $VARIANT, act-steps $ACT_STEPS, other-arm tol $OTHER_TOL${NOTE:+; $NOTE}" \
  "${EXTRA[@]}"
