"""RoboTTT-style test-time-training memory for FLUX 3 Action (WS4 prototype, CPU-testable).

See ``ws4/research/robottt_flux.md`` for the analysis and ``ws4/ttt/README.md`` for how to run things.
"""

from .temporal import (
    TemporalTTTBlock,
    masked_sequence_mean,
    sequence_action_forcing_timesteps,
    tbptt_slices,
)
from .ttt_layer import GatedTTT, TanhGate, TTTLayer, detach_state, fast_forward, fast_grad

__all__ = [
    "GatedTTT",
    "TTTLayer",
    "TanhGate",
    "TemporalTTTBlock",
    "detach_state",
    "fast_forward",
    "fast_grad",
    "masked_sequence_mean",
    "sequence_action_forcing_timesteps",
    "tbptt_slices",
]
