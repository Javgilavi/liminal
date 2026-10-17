# Haller controller integration

Haller owns the motors/cameras for the default
[Liminal real-robot runner](../../docs/official-inference.md). It is a separate
repository, not a package bundled here.

`command-feedback.patch` is exported against Haller commit
`b46b827be16f9f77238df99d826faeccb923e0c9`. It includes:

- Opt-in `command_feedback: 1` acknowledgements with actual committed targets.
- Observations paired with the most recent command and sequence number.
- Torque release for the commanded arm on disconnect.
- Policy targets passed without slew reshaping; manual moves retain their limiter.
- The corresponding real/sim driver signature changes and torque-release handling.

This supersedes the early feedback patch that retained policy speed limiting.
Calibrated travel and mode guards remain; the official Liminal runner rejects
out-of-travel chunks and stops if the controller alters a valid target.

For a clean checkout at the recorded base, from the Liminal root:

```bash
git -C ../haller_ws apply --check "$PWD/integrations/haller/command-feedback.patch"
git -C ../haller_ws apply "$PWD/integrations/haller/command-feedback.patch"
```

Do not apply twice. A different Haller revision may require porting the patch;
`--check` must pass before applying it. Stop/restart the controller only with the
arm supported and torque disengaged. The commissioned checkout is already
patched. This repository does not commit or modify that checkout.

Configure Haller's `top` scene view and `wrist_a` snapshot endpoint to match the
local rig profile. The patch covers the policy protocol/driver dependency, not
machine-specific camera setup. Read-only `./scripts/infer_robot.sh` checks the
feedback handshake before execution.

Optional cross-repository verification uses fake servos and in-memory streams:

```bash
PYTHONPATH="$PWD/../haller_ws/hmi/backend:$PWD/src" PYTHONNOUSERSITE=1 \
  .venv-robot/bin/pytest -q tests/test_haller_feedback_integration.py
```
