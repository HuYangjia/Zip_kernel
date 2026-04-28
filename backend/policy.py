"""Per-kernel backend selection policy.

A *policy* is a function ``(kernel_name, shape_ctx) -> 'triton' | 'cuda'``.
The built-in ``auto`` policy uses a conservative hand-rolled decision
table that prefers CUDA only when we have strong evidence it wins
(see Phase 1 benchmark notes in ``research/analysis_20260422_next_steps.md``).

Users can override in three ways:

1. Environment variable ``HKUST_V9_BACKEND`` before process start:
   - ``HKUST_V9_BACKEND=triton`` : force Triton for every kernel
   - ``HKUST_V9_BACKEND=cuda``   : force CUDA where available (else Triton)
   - ``HKUST_V9_BACKEND=auto``   : use the built-in policy (default)

2. Programmatic override with :func:`set_backend_policy`, which accepts
   either a string (``'auto'`` / ``'triton'`` / ``'cuda'``) or a custom
   callable.

3. Per-kernel override via :func:`set_backend_policy` with a ``dict``
   mapping kernel name -> backend string.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Union

from .registry import (
    ALL_KERNELS,
    KERNEL_ACTIVATION_QUANT,
    KERNEL_DENSE_GEMM,
    KERNEL_FUSED_DENSE_SPARSE,
    KERNEL_SPARSE_GEMM,
    BackendKernel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shape context passed into the policy function
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapeContext:
    """Minimal shape descriptor used by policy decisions.

    Populated by :func:`kernel.backend.dispatcher.select_impl` from the
    runtime call arguments.  Using a dedicated struct (vs. threading raw
    ints through the policy signature) keeps the policy API stable as
    we add more features (e.g. dtype, hp_ratio buckets).
    """

    T: int = -1           # flattened batch (batch * seq_len)
    d_out: int = -1
    d_in: int = -1
    n_hp_blocks: int = 0  # 0 for activation_quant / dense-only paths
    n_groups: int = -1


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------

# A policy is any callable:  (kernel_name, ShapeContext) -> str
PolicyFn = Callable[[str, ShapeContext], str]


def _auto_policy(kernel_name: str, ctx: ShapeContext) -> str:
    """Default hand-rolled decision table, recalibrated to Round-46
    (bench_20260427_224405) on RTX 4090 (SM89).

    History:
      - Round 8/9 calibrated against cuBLAS FP16, which produced a
        conservative "prefer Triton at mid/large T" table.
      - Rounds 38-46 rewrote the CUDA kernels (activation_quant vector
        scatter, dense MMA kBm=64 gate, fused_dense_sparse kBm=64 BSR
        remap, sparse kGrpBuf=128 opt-in shmem, R46 unified decode
        pipeline) and the CUDA path now dominates Triton across every
        tracked shape / T regime on the canonical bench.

    Round-46 authoritative measurements (bench_20260427_224405,
    CUDA speedup vs Triton, > 1.0 means CUDA wins):

      activation_quant    T=1 .. T=1024  : 3.07x .. 4.77x
      dense_gemm          T=1 (4k/4k)    : 4.39x
      dense_gemm          T=1 (4k/11k)   : 1.93x
      dense_gemm          T=1 (11k/4k)   : 3.47x
      dense_gemm          T=8/16/64/128  : 1.85x / 1.85x / 1.47x / 1.45x
      dense_gemm          T=512/1024     : 1.81x / 1.89x
      sparse_gemm         T=1 .. T=128   : 3.72x .. 3.80x
      sparse_gemm         T=512 / 1024   : 2.78x / 1.61x
      fused_dense_sparse  T=1 (4k/4k)    : 4.91x
      fused_dense_sparse  T=1 (4k/11k)   : 2.20x
      fused_dense_sparse  T=1 (11k/4k)   : 3.98x
      fused_dense_sparse  T=8/16/64/128  : 2.27x / 2.03x / 1.63x / 1.59x
      fused_dense_sparse  T=512 / 1024   : 1.81x / 1.82x

    End-to-end v9_linear (Round-46):
      T=1   (4k/4k)   : 3.96x    (Triton 155.40us -> CUDA 39.28us)
      T=1   (4k/11k)  : 3.21x
      T=1   (11k/4k)  : 4.15x
      T=8   (4k/4k)   : 3.28x
      T=16  (4k/4k)   : 3.02x
      T=64  (4k/4k)   : 2.48x
      T=128 (4k/4k)   : 2.32x
      T=512 (4k/4k)   : 1.91x
      T=1024(4k/4k)   : 1.76x

    Policy summary (Round 46): every kernel prefers CUDA on every
    shape we have data for.  No shape in the canonical bench regresses
    under "force cuda" vs "force triton".  We keep a structured per-
    kernel table (instead of a bare ``return 'cuda'``) so future shape-
    specific blacklists / fallbacks can be grafted in without rewiring.

    Known gaps (not regressions, just unmeasured regions):
      - T > 1024 prefill never benched here; _forward_prefill applies
        its own W4A16 fallback for very large dense-only shapes, so the
        policy choice only matters when hp_blocks > 0.
      - hp_ratio other than 0.05 is not in the canonical bench set.
        R42/R43/R45 sweeps on hp=0 and hp=0.05 show CUDA winning in
        both cases; the policy therefore does not branch on hp.
    """
    if kernel_name == KERNEL_ACTIVATION_QUANT:
        # 3.07x-4.77x across every T; always CUDA.
        return "cuda"

    if kernel_name == KERNEL_DENSE_GEMM:
        # R46: CUDA wins 1.45x-4.39x across every benched T (1, 8, 16,
        # 64, 128, 512, 1024).  Old Round-9 "T>=8 => triton" rule was
        # calibrated against a cuBLAS-FP16 comparator and is obsolete
        # for the quant pipeline, where Triton is the true baseline.
        return "cuda"

    if kernel_name == KERNEL_SPARSE_GEMM:
        # R46: CUDA wins 1.61x-3.80x across T=1..1024.  Old Round-7
        # "T>16 => triton" rule predates the kGrpBuf=128 opt-in shmem
        # fix and the kBn tuning; remove it.
        return "cuda"

    if kernel_name == KERNEL_FUSED_DENSE_SPARSE:
        # R46: CUDA wins 1.59x-4.91x across T=1..1024.  The dispatcher
        # decides fused-vs-split on top of this; the policy here only
        # controls whether the fused kernel itself is CUDA or Triton.
        return "cuda"

    return "triton"

def _force_triton_policy(kernel_name: str, ctx: ShapeContext) -> str:
    return "triton"


def _force_cuda_policy(kernel_name: str, ctx: ShapeContext) -> str:
    return "cuda"


_NAMED_POLICIES: Dict[str, PolicyFn] = {
    "auto": _auto_policy,
    "triton": _force_triton_policy,
    "cuda": _force_cuda_policy,
}


# ---------------------------------------------------------------------------
# Active policy (module-level, mutable via ``set_backend_policy``)
# ---------------------------------------------------------------------------

def _resolve_env_policy() -> PolicyFn:
    name = os.environ.get("HKUST_V9_BACKEND", "auto").lower()
    if name in _NAMED_POLICIES:
        logger.info("Backend policy from environment: %s", name)
        return _NAMED_POLICIES[name]
    logger.warning(
        "Unknown HKUST_V9_BACKEND=%r; falling back to 'auto'", name
    )
    return _auto_policy


_active_policy: PolicyFn = _resolve_env_policy()


def current_policy() -> PolicyFn:
    return _active_policy


def set_backend_policy(
    policy: Union[str, PolicyFn, Dict[str, str]],
) -> None:
    """Override the per-kernel backend selection.

    Accepts:
      - ``'auto'``   : restore the default hand-rolled table
      - ``'triton'`` : force Triton everywhere
      - ``'cuda'``   : force CUDA everywhere (with Triton fallback on
                       kernels whose CUDA impl is unavailable)
      - ``dict``     : per-kernel override map, e.g.
                       ``{'activation_quant': 'cuda', 'dense_gemm': 'triton'}``;
                       unspecified kernels use the ``auto`` default.
      - ``callable`` : full custom policy ``(name, ShapeContext) -> str``.

    Thread-safety: the policy is a single module-level function
    reference, swapped atomically; concurrent reads during a swap will
    see either the old or the new policy, never a torn state.  No lock
    is needed for correctness.
    """
    global _active_policy

    if isinstance(policy, str):
        key = policy.lower()
        if key not in _NAMED_POLICIES:
            raise ValueError(
                f"Unknown backend policy name {policy!r}; "
                f"expected one of {sorted(_NAMED_POLICIES.keys())}"
            )
        _active_policy = _NAMED_POLICIES[key]
        logger.info("Backend policy set to %s", key)
        return

    if isinstance(policy, dict):
        # Validate up front so a bad dict fails at set-time, not at call-time.
        for k, v in policy.items():
            if k not in ALL_KERNELS:
                raise ValueError(f"Unknown kernel name {k!r} in policy dict")
            if v not in ("triton", "cuda"):
                raise ValueError(
                    f"Policy value for {k!r} must be 'triton' or 'cuda', got {v!r}"
                )
        override = dict(policy)

        def _dict_policy(kernel_name: str, ctx: ShapeContext) -> str:
            if kernel_name in override:
                return override[kernel_name]
            return _auto_policy(kernel_name, ctx)

        _active_policy = _dict_policy
        logger.info("Backend policy set to per-kernel override: %s", override)
        return

    if callable(policy):
        _active_policy = policy
        logger.info("Backend policy set to custom callable %r", policy)
        return

    raise TypeError(
        f"Policy must be str, dict, or callable; got {type(policy).__name__}"
    )


# ---------------------------------------------------------------------------
# Status report (diagnostic)
# ---------------------------------------------------------------------------


def get_backend_status() -> Dict[str, object]:
    """Return a diagnostic dict of current backend availability.

    Useful for logging once at program start-up to confirm which
    backend is in use.  Not perf-critical.
    """
    return {
        "cuda_available": BackendKernel.cuda_available(),
        "cuda_import_error": BackendKernel.cuda_import_error(),
        "cuda_kernels": list(BackendKernel.cuda_available_kernels()),
        "active_policy": getattr(_active_policy, "__name__", repr(_active_policy)),
    }


__all__ = [
    "ShapeContext",
    "PolicyFn",
    "set_backend_policy",
    "current_policy",
    "get_backend_status",
]
