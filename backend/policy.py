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
    """Default hand-rolled decision table, calibrated from measured
    CUDA-vs-Triton benchmarks on RTX 4090 (SM89) at 2026-04-24.

    Measurement source: ``kernel/cuda_kernel/benchmarks/bench_cuda_vs_triton.py``
    run ``bench_20260424_122507``.  Concrete speedups (CUDA vs Triton,
    >1 means CUDA is faster):

      activation_quant     : 2.62x .. 4.70x across all T in [1, 1024]
      dense_gemm  T=1      : 1.17x    (win)
      dense_gemm  T=1 d_out=11k : 0.75x (lose; 11k * 1 path hits a
                                        Triton sweet spot)
      dense_gemm  T>=8     : 0.02x .. 0.27x (CUDA SIMT path scales
                                              badly -- see Phase 5 plan)
      sparse_gemm T=1      : 3.60x .. 3.80x
      sparse_gemm T>=16    : 0.12x .. 0.63x
      fused        T=1     : ~1.25x small-d_out, ~0.95x large-d_out
      fused        T>=8    : 0.04x .. 0.39x

    End-to-end v9_linear_forward at T=1 gets 2.75-2.82x from the
    combined activation_quant+sparse+fused CUDA path; that's the
    primary win for LLM decoding.

    Policy summary:
      - activation_quant: ALWAYS CUDA (wins everywhere).
      - dense_gemm: CUDA only at T=1 AND d_out <= d_in (avoid the 11k
        loss case); Triton otherwise.
      - sparse_gemm: CUDA only at T=1; Triton for T>=8 (the SIMT loop
        over kBn columns does not ILP well at wider tiles -- Phase 5
        MMA rewrite should flip this).
      - fused_dense_sparse: CUDA only at T=1 AND d_out <= d_in; Triton
        otherwise.

    For kernels whose CUDA implementation is still a stub the dispatcher
    will detect the missing impl and fall back to Triton regardless of
    what this function returns.
    """
    if kernel_name == KERNEL_ACTIVATION_QUANT:
        # 2.6x-4.7x across every shape tested.  Even at T=2048 the
        # CUDA two-pass-in-one-launch still wins on launch overhead.
        return "cuda"

    if kernel_name == KERNEL_DENSE_GEMM:
        # Win margin is narrow (~17%) even at the sweet spot, and flips
        # into a loss at d_out >> d_in (11k vs 4k case) because our SIMT
        # loop runs 86 M-tiles sequentially while Triton's TC pipeline
        # overlaps them.  Restrict CUDA to the strict decode case where
        # the SIMT launch-overhead advantage dominates.
        if ctx.T == 1 and ctx.d_out <= ctx.d_in:
            return "cuda"
        return "triton"

    if kernel_name == KERNEL_SPARSE_GEMM:
        # CUDA wins by 3.6-3.8x at T=1 (tiny launch overhead + no
        # autotune).  At T>=8 the per-BSR-block shared-mem round trip
        # starts dominating; Triton's persistent tile cache is better.
        if ctx.T == 1:
            return "cuda"
        return "triton"

    if kernel_name == KERNEL_FUSED_DENSE_SPARSE:
        # Same shape as dense_gemm.
        if ctx.T == 1 and ctx.d_out <= ctx.d_in:
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
