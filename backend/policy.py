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
    """Default hand-rolled decision table.

    The table is intentionally conservative: when in doubt, return
    Triton (the reference path we know is fast enough).  CUDA wins
    are only claimed where we have microbench evidence from Phase 1.

    For kernels whose CUDA implementation is still a stub (phase 2+)
    the dispatcher will detect the missing impl and fall back to Triton
    regardless of what this function returns.
    """
    if kernel_name == KERNEL_ACTIVATION_QUANT:
        # CUDA wins across the whole shape range: the Triton kernel's
        # two-pass design (max then quant) plus autotune-dispatch cost
        # are both eliminated in the CUDA version.
        return "cuda"

    if kernel_name == KERNEL_DENSE_GEMM:
        if ctx.T <= 32:
            # Decode regime: CUDA wins on launch overhead (no autotune).
            return "cuda"
        if ctx.T >= 1024:
            # Prefill: v9_linear already switches to W4A16 fallback above
            # this threshold, so this branch only fires when hp>0 forces
            # the int4 path.  Triton's autotuned m16n8k16 kernel is
            # competitive here; don't risk regressing until Phase 2 CUDA
            # GEMM lands with its own microbench.
            return "triton"
        return "cuda"

    if kernel_name == KERNEL_SPARSE_GEMM:
        # Sparse kernel launches ``nrow`` programs even when every BSR
        # row is empty; the CUDA version uses a persistent kernel with
        # a work queue, so it wins hardest at low hp_ratio.  Always
        # prefer CUDA when available.
        return "cuda"

    if kernel_name == KERNEL_FUSED_DENSE_SPARSE:
        # Prefill + hp>0 only.  Same trade-off as dense_gemm.
        if ctx.T >= 1024:
            return "triton"
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
