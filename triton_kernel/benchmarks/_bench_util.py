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

r62 P2 addendum: L2-cache flush
--------------------------------
Tight-loop bench (``n_iter=100`` against the same tensor) lets any
problem ``<= L2 size`` run from L2 after the first miss, which inflates
the BF16 cuBLAS baseline in the compare path by up to 2x.  Callers that
need cold-cache HBM measurements (matching real LLM inference where each
weight is read once per layer) must pass ``flush_l2=True``.

All helpers intentionally keep the old ``_time_ms(fn)`` signature so
existing callers can be switched over with a single import change.
"""
from __future__ import annotations

from typing import Callable

import torch

__all__ = ["time_ms"]


# r62 P2: shared 96 MB L2-flush scratch (RTX 4090 L2 = 72 MB).
_L2_FLUSH_TENSOR = None


def _get_l2_flush_tensor():
    """Lazy-init global L2 flush scratch (96 MB, reused across calls)."""
    global _L2_FLUSH_TENSOR
    if _L2_FLUSH_TENSOR is None:
        if not torch.cuda.is_available():
            return None
        _L2_FLUSH_TENSOR = torch.empty(
            96 * 1024 * 1024, dtype=torch.int8, device="cuda"
        )
    return _L2_FLUSH_TENSOR


def _flush_l2():
    """Evict L2 by writing a 96 MB scratch buffer."""
    buf = _get_l2_flush_tensor()
    if buf is not None:
        buf.zero_()


def time_ms(
    fn: Callable[[], None],
    n_warmup: int = 50,
    n_iter: int = 100,
    n_repeat: int = 3,
    flush_l2: bool = False,
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
    flush_l2:
        If True, evict L2 (96 MB scratch write) *before* every invocation
        of ``fn`` inside the timing windows.  The per-flush cost is
        calibrated once and subtracted from the reported time so the
        returned value remains fn-only.  Use this for baselines that
        must reflect cold-cache HBM traffic (e.g. comparing an INT4
        kernel against cuBLAS BF16 matmul on <72 MB problems).
    """
    if n_warmup > 0:
        for _ in range(n_warmup):
            if flush_l2:
                _flush_l2()
            fn()
        torch.cuda.synchronize()

    flush_ms = 0.0
    if flush_l2:
        # Calibrate flush-only cost (amortised across windows).
        torch.cuda.synchronize()
        s0 = torch.cuda.Event(enable_timing=True)
        e0 = torch.cuda.Event(enable_timing=True)
        s0.record()
        for _ in range(n_iter):
            _flush_l2()
        e0.record()
        torch.cuda.synchronize()
        flush_ms = s0.elapsed_time(e0) / n_iter

    best_ms = float("inf")
    for _ in range(max(1, n_repeat)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            if flush_l2:
                _flush_l2()
            fn()
        end.record()
        torch.cuda.synchronize()
        # Subtract the calibrated flush cost to isolate fn's contribution.
        best_ms = min(best_ms, start.elapsed_time(end) / n_iter - flush_ms)
    return best_ms
