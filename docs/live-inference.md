# Legacy standalone FLUX inference

Use [real SO101 inference](official-inference.md) for the current default and
[first validation results](validation/single-arm-v1.md) for recorded outcomes.

The initial 17 October tests used BFL's official standalone
`flux_action.inference.so101.load_policy`, implemented by
`scripts/flux_worker.py`, with the custom Haller/Liminal transport. This differs
from the later official **LeRobot `Flux3Policy` and saved processors** path.

The historical launcher is `scripts/infer_robot_haller.sh`, selecting
`configs/policy/live_right.yaml`. It is observation-only by default. The source is
retained for comparison and historical log analysis; current Haller policy
streaming no longer applies the 20°/s limiter present in the early recorded trials.
Consequently, running that launcher today does not reproduce the original motor
trajectory unchanged.

The standalone path contains joint-delta rebasing around acknowledged targets.
The official LeRobot path bypasses that code, integrates through the saved
postprocessor, and rejects invalid execution horizons instead of clipping them.
Both paths retain local telemetry/load monitoring and single-owner execution.
Never run them concurrently.

Early tests established inference and command delivery, with substantial
trajectory limiting and elbow tracking loss. They did not establish a successful
grasp. Exact later replay is limited where only startup JPEGs were saved.
