"""Liminal: long-horizon, dual-arm orchestration on top of a learned VLA policy.

Layering (top to bottom):

    ui            -> triggers tasks, shows world state, e-stop
    orchestration -> behavior tree with AFTER / WHILE / UNTIL dependencies
    perception    -> symbolic world state + event log, verifies skill effects
    safety        -> arm ownership, workspace zones, joint limits, e-stop
    policy        -> skill backends (FLUX 3 Action, scripted, mock)
    hardware/sim  -> per-arm controllers (SO-101 via LeRobot, MuJoCo, mock)

See docs/architecture.md for the full picture.
"""

__version__ = "0.1.0"
