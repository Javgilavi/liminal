"""Be a good neighbour on the robot laptop: lowest CPU priority and at most 4 threads for every heavy ws4/ttt job.

Real-time robot loops (30 Hz FLUX runner, VR recorder) share this machine. Every CLI here calls :func:`throttle`
first. ``WS4_TTT_THREADS`` changes the thread cap (default 4); ``WS4_TTT_NICE=0`` keeps the normal priority
(only for a machine with nothing else running, e.g. a pod).
"""

from __future__ import annotations

import os

MAX_THREADS = 4


def throttle() -> int:
    """``nice 19`` and cap BLAS/OpenMP/torch threads; returns the thread cap. Call before heavy imports run work."""
    n = int(os.environ.get("WS4_TTT_THREADS", MAX_THREADS))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(min(n, int(os.environ.get(var, n))))
    if os.environ.get("WS4_TTT_NICE", "1") != "0":
        try:
            os.nice(19 - os.nice(0))
        except OSError:
            pass
    try:
        import torch

        torch.set_num_threads(min(n, torch.get_num_threads()))
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass  # interop threads can only be set once per process
    return n
