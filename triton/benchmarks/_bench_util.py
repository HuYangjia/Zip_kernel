"""Shared benchmark helpers.

Why a dedicated module?
-----------------------
``bench_dense.py``, ``bench_sparse.py``, ``bench_linear.py`` and
``sweep_v9.py`` all used to carry a near-identical ``_time_ms`` helper with
the exact same signature.  We observed that the naive version

    for _ in range(n_iter): fn()
    return start.elapsed_time(end) / n_iter

systematically **over-reports** time for small kernels (<30 us) on idle
GPUs, for two reasons:

1. RTX 4090 down-clocks to <500 MHz when idle; 10 warm-up calls are not
   enough to get it back to boost frequency before ``start.record``.
2. A single 30-iter window is too short; CUDA runtime jitter can shift
   the average by 30 percent, which in turn makes the "bs=1 is slower
   than bs=16" artefact we debugged on 2026-04-22.

Fix: warm up longer, run *multiple* measurement windows, and keep the
**minimum** average as the reported time.  ``min`` is the de-facto
standard in microbenchmarks because it is closest to the inherent GPU
cost and least contaminated by transient OS/driver noise.

All helpers intentionally keep the old ``_time_ms(fn)`` signature so
existing callers can be switched over with a single import change.
"""
from __future__ import annotations

from typing import Callable

import torch

__all__ = ["time_ms"]


def time_ms(
    fn: Callable[[], None],
    n_warmup: int = 50,
    n_iter: int = 100,
    n_repeat: int = 3,
) -> float:
    """Return the GPU-side wall time of ``fn`` in milliseconds.

    Parameters
    ----------
    fn:
        Zero-argument callable that issues CUDA work.  Must be idempotent
        (safe to call repeatedly) and should not synchronize internally.
    n_warmup:
        Number of warm-up invocations executed *before* the first timing
        window.  50 is enough for the RTX 4090 to lock to boost clocks
        and for Triton/cuBLAS heuristics to settle.
    n_iter:
        Number of invocations per timing window.  The reported time is
        ``window_elapsed / n_iter`` (average inside the window).
    n_repeat:
        Number of timing windows.  The helper returns the **minimum**
        across windows to reject transient noise.
    """
    if n_warmup > 0:
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()

    best_ms = float("inf")
    for _ in range(max(1, n_repeat)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        torch.cuda.synchronize()
        best_ms = min(best_ms, start.elapsed_time(end) / n_iter)
    return best_ms
