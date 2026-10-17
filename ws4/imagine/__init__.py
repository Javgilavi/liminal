"""Imagination for FLUX 3 Action: decode, show and use the future frames the model denoises with each chunk.

See ``ws4/research/imagination.md`` for the evidence and design, and ``GPU_RUNBOOK.md`` for the cloud steps.
Heavy dependencies (torch, lerobot, anthropic, PIL) are imported inside functions or submodules, so
``import ws4.imagine`` works on any laptop.
"""

__all__ = ["compose", "predict_with_video", "score"]
