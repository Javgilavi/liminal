# Independent runtime review

Reviewed base: `origin/integrate/harness-main`, commit
`6975d2f87ec73559d48d9f2c6abb5e4bb0768bf2`. A separate read-only reviewer inspected
the actual replacement-task atomic skills, BT, policy executor, live worker and
hardware stop contracts, then reviewed the integration changes again.

| Finding | Resolution / evidence |
|---|---|
| FLUX loader/predict stubs disconnected from working live code | Shared motor-free `LeRobotEngine`, used by BT and legacy worker |
| Missing per-skill identity/compatibility | Registry validates pinned bundle hashes, provenance, base, targets, units, joint order, cameras and dimensions |
| Concurrent waiter bypassed inference quarantine | Full skill serialization plus health check under inference lock; concurrent timeout test |
| Cancellation released ownership without stopping generic skills | `SkillNode` finally stops while ownership is held; cancellation test |
| Failed recovery was ignored | Retry propagates recovery status; home recovery includes positive inspection checks |
| Known-false home allowed cable restore | Explicit home checks; negative inspection regression test |
| Old success/invariant facts could survive indefinitely | Fresh version/capture checks, out-of-order rejection and expiry monitor |
| History had one frame per chunk and no committed commands | Fresh per-skill windows, explicit measured/commanded histories, reset and provenance tests |
| Parking overwrote closing gripper target | Normal stop parks only non-gripper joints; fake-bus test |
| New generic error conversion swallowed safety exceptions | Explicit propagation of `EmergencyStop`/`ArmConflict` |
| Loading competed with active-hold evidence deadlines | Independent invariant refresh and explicit skill/load budgets |

The reviewer explicitly distinguished source/mock evidence from real policy and
hardware validation. See [showcase-runtime.md](../showcase-runtime.md) for setup,
reproduction commands and remaining hardware validation gates. No new physical
robot or trained-checkpoint trial was performed for this integration.

Final validation on this branch: 225 tests passed; one optional Haller integration
module was skipped because `haller_hmi` is unavailable. Ruff and all six shipped
task-file validations passed. Clear, blocked and recovery symbolic demonstrations
passed; compact results are in [showcase-results.json](showcase-results.json).
Tests used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to exclude unrelated ROS pytest
plugins installed on the host. No test required GPU, network or serial ports.

The reviewer verified the final driver-coordinate contract check and the
invariant-expiry/refresh race fix. Regression coverage includes conflicting but
internally consistent coordinate metadata and bimanual registry execution.
