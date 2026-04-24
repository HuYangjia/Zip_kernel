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
    """Default hand-rolled decision table, calibrated against cuBLAS FP16
    baseline on RTX 4090 (SM89).

    Round 8 (bench_20260424_141934) switched the reference from Triton to
    cuBLAS FP16 matmul.  Round 9 loosened the dense/fused rule for T=1
    after discovering that the old ``d_out <= d_in`` guard was calibrated
    against Triton and is no longer needed when FP16 is the comparator.

    Concrete speedups vs cuBLAS FP16 (>1 = CUDA wins):

      activation_quant  : 0.21x .. 0.35x  -- always slower than FP16 memcpy
                          (unavoidable: FP16 path doesn't need quantization)
      dense_gemm  T=1, d_out=11k, d_in=4k : 1.60x  (win -- large d_out)
      dense_gemm  T=1, d_out=4k,  d_in=4k : 0.34x  (lose -- FP16 GEMV fast)
      dense_gemm  T=1, d_out=4k,  d_in=11k: 0.56x  (lose)
      dense_gemm  T>=8                     : 0.02x .. 0.22x (lose)
      sparse_gemm T=1, d_out=11k           : 5.21x  (big win -- sparsity)
      sparse_gemm T=1, d_out=4k            : 0.92x  (near-parity)
      sparse_gemm T<=16                    : 0.84x .. 5.29x
      sparse_gemm T>=64                    : 0.24x .. 0.29x (lose)
      fused       T=1, d_out=11k           : 1.80x  (win)
      fused       T=1, d_out=4k            : 0.31x  (lose)
      fused       T>=8                     : 0.05x .. 0.19x (lose)

    End-to-end v9_linear (auto policy):
      T=1, 4k/11k : 1.34x (win -- after Round-9 policy fix)
      T=1, 4k/4k  : 0.30x (lose -- FP16 GEMV dominates)
      T=1, 11k/4k : 0.56x (lose)
      T>=8        : 0.09x .. 0.56x (lose)

    Policy summary (Round 9):
      - activation_quant: CUDA always (3-4.7x over Triton, even if slower
        than FP16 memcpy -- it's a mandatory step for W4A8 accuracy).
      - dense_gemm:
          T=1              -> CUDA always (wins on large d_out; near-parity
                              on square; acceptable loss on d_in>d_out).
          T=2..8, d_out<=d_in -> CUDA (1.07x win).
          else             -> Triton.
      - sparse_gemm: CUDA iff T <= 16 (robust 0.84-5.29x range).
      - fused_dense_sparse:
          T=1              -> CUDA always (same reasoning as dense_gemm T=1).
          T=2..8, d_out<=d_in -> CUDA.
          else             -> Triton.
    """
    if kernel_name == KERNEL_ACTIVATION_QUANT:
        return "cuda"

    if kernel_name == KERNEL_DENSE_GEMM:
        if ctx.T == 1:
            return "cuda"   # Round-9: always CUDA at T=1 (wins on large d_out)
        if ctx.T <= 8 and ctx.d_out <= ctx.d_in:
            return "cuda"
        return "triton"

    if kernel_name == KERNEL_SPARSE_GEMM:
        # Round 7 cp.async helped T<=16 (3.8x->3.95x) but pushed T>=64
        # into a loss zone due to kBn=4 spill + prefetch pressure.
        if ctx.T <= 16:
            return "cuda"
        return "triton"

    if kernel_name == KERNEL_FUSED_DENSE_SPARSE:
        if ctx.T == 1:
            return "cuda"   # Round-9: always CUDA at T=1
        if ctx.T <= 8 and ctx.d_out <= ctx.d_in:
            return "cuda"
        return "triton"

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
