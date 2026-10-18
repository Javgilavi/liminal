#!/usr/bin/env bash
# Pipeline dry run of the demo chain, skill by skill through eval_skill.sh: fake robot (no arm port, nothing logged),
# the real policy server, one short attempt per skill with that skill's prompt, steps, arm flags, keep-out and
# references. Any 6-D package exercises the pipeline; the skill-agnostic `median` is the default row.
#
#   POD_HOST=157.157.221.177 POD_PORT=15377 bash ws4/single_arm/chain_dryrun.sh
#
# Env: CHAIN ("box green cutter pen_lid"), ROWS (median), ATTEMPT_S (4), ALLOW_NO_REFS (1: skills built with
# --no-refs still run, their home check reports the missing references), plus anything eval_skill.sh takes.
set -euo pipefail
: "${POD_HOST:?POD_HOST}" "${POD_PORT:?POD_PORT}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
for skill in ${CHAIN:-box green cutter pen_lid}; do
  if ! env_out=$(python ws4/single_arm/skills.py env "$skill" 2>&1); then
    echo "=== $skill: not ready ($env_out)"; continue
  fi
  first=$(sed -n "s/^SKILL_LAYOUTS='\{0,1\}\([^ ']*\).*/\1/p" <<< "$env_out")
  echo "=== $skill: one attempt on layout $first"
  if ! SKILL="$skill" FAKE_ROBOT=1 ROWS="${ROWS:-median}" LAYOUTS="$first" PER=1 ATTEMPT_S="${ATTEMPT_S:-4}" \
      ALLOW_NO_REFS="${ALLOW_NO_REFS:-1}" bash ws4/single_arm/eval_skill.sh session < /dev/null; then
    echo "=== $skill: refused or failed (see above)"
  fi
done
