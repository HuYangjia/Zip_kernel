"""Backend registry.

Discovers which backends are available at import time and provides a
single ``BackendKernel`` facade that callers can use to access any
sub-kernel (quant / dense_gemm / sparse_gemm / fused_dense_sparse) on
either backend, without ever importing the two backend packages
directly.

Why a runtime registry (vs. a static if/else)?
----------------------------------------------
The CUDA extension can fail to build or load for many reasons:
  - Non-SM89 GPU (we compile ``-gencode=arch=compute_89,code=sm_89``)
  - ``nvcc`` not on PATH during ``torch.utils.cpp_extension.load``
  - CUDA toolkit version mismatch with the torch build
In any of these cases we want the package to *still* import cleanly and
simply report ``cuda_available == False``.  A static ``from .cuda_kernel
import *`` at module top would have crashed the whole package import
instead, taking down the Triton path with it.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel names (canonical, used as dict keys across policy / dispatcher)
# ---------------------------------------------------------------------------

KERNEL_ACTIVATION_QUANT = "activation_quant"
KERNEL_DENSE_GEMM = "dense_gemm"
KERNEL_SPARSE_GEMM = "sparse_gemm"
KERNEL_FUSED_DENSE_SPARSE = "fused_dense_sparse"

ALL_KERNELS = (
    KERNEL_ACTIVATION_QUANT,
    KERNEL_DENSE_GEMM,
    KERNEL_SPARSE_GEMM,
    KERNEL_FUSED_DENSE_SPARSE,
)


# ---------------------------------------------------------------------------
# Backend probing (done once at import time)
# ---------------------------------------------------------------------------

_triton_impls: Dict[str, Callable[..., Any]] = {}
_cuda_impls: Dict[str, Callable[..., Any]] = {}
_cuda_import_error: str | None = None


def _load_triton_backend() -> None:
    """Populate ``_triton_impls`` from ``kernel.triton_kernel``.

    This is the reference backend and must always succeed.
    """
    from kernel.triton_kernel.activation_quant import quantize_activation_s4
    from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4
    from kernel.triton_kernel.fused_dense_sparse_gemm import fused_dense_sparse_gemm
    from kernel.triton_kernel.sparse_s4s4_gemm import sparse_gemm_s4_s4

    _triton_impls[KERNEL_ACTIVATION_QUANT] = quantize_activation_s4
    _triton_impls[KERNEL_DENSE_GEMM] = dense_gemm_u4_s4
    _triton_impls[KERNEL_SPARSE_GEMM] = sparse_gemm_s4_s4
    _triton_impls[KERNEL_FUSED_DENSE_SPARSE] = fused_dense_sparse_gemm


def _load_cuda_backend() -> None:
    """Populate ``_cuda_impls`` from ``kernel.cuda_kernel``.

    Failures here are non-fatal; the dispatcher will fall back to
    Triton for every kernel whose CUDA implementation is unavailable.
    """
    global _cuda_import_error

    # Hard opt-out: ``HKUST_V9_DISABLE_CUDA=1`` skips JIT compilation
    # entirely (useful for CI hosts with CUDA toolkit but non-SM89 GPU,
    # or for bisecting Triton-only regressions).
    if os.environ.get("HKUST_V9_DISABLE_CUDA", "0") == "1":
        _cuda_import_error = "disabled via HKUST_V9_DISABLE_CUDA=1"
        logger.info("CUDA backend disabled via environment variable")
        return

    try:
        from kernel.cuda_kernel import ops as cuda_ops

        # Each op module exposes a callable named ``<kernel>_cuda`` which
        # either is the real implementation, or ``None`` if the kernel is
        # stubbed (not yet implemented in this phase).
        for name in ALL_KERNELS:
            fn = getattr(cuda_ops, f"{name}_cuda", None)
            if callable(fn):
                _cuda_impls[name] = fn
    except Exception as exc:  # noqa: BLE001 -- JIT build can raise anything
        _cuda_import_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "CUDA backend not available (%s); falling back to Triton for all kernels",
            _cuda_import_error,
        )


# Populate at import time so callers can inspect availability before
# making policy decisions.  Triton must succeed; CUDA is best-effort.
_load_triton_backend()
_load_cuda_backend()


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


class BackendKernel:
    """Static facade over per-kernel backend lookup.

    Usage::

        from kernel.backend import BackendKernel
        BackendKernel.activation_quant(x, perm)   # picks backend per policy
        BackendKernel.dense_gemm(...)             # same

    The per-call backend choice is made by ``kernel.backend.policy``,
    which takes tensor shapes as input.  Callers that want to hardwire
    one backend should use :func:`set_backend_policy`.
    """

    # --- introspection ---------------------------------------------------

    @staticmethod
    def cuda_available() -> bool:
        """True iff ``cuda_kernel`` loaded successfully (any kernel)."""
        return len(_cuda_impls) > 0

    @staticmethod
    def cuda_available_kernels() -> tuple[str, ...]:
        """Tuple of kernel names with a working CUDA implementation."""
        return tuple(sorted(_cuda_impls.keys()))

    @staticmethod
    def cuda_import_error() -> str | None:
        """Reason the CUDA backend is unavailable, or ``None`` if it loaded."""
        return _cuda_import_error

    # --- raw backend access (bypasses policy; used by tests) --------------

    @staticmethod
    def triton_impl(name: str) -> Callable[..., Any]:
        return _triton_impls[name]

    @staticmethod
    def cuda_impl(name: str) -> Callable[..., Any] | None:
        return _cuda_impls.get(name)

    # --- policy-aware convenience wrappers --------------------------------
    # (Implemented in dispatcher.py via ``select_impl``; the wrappers
    # below are thin shims so application code can use attribute access.)

    @staticmethod
    def activation_quant(*args, **kwargs):
        from .dispatcher import select_impl
        return select_impl(KERNEL_ACTIVATION_QUANT, args, kwargs)(*args, **kwargs)

    @staticmethod
    def dense_gemm(*args, **kwargs):
        from .dispatcher import select_impl
        return select_impl(KERNEL_DENSE_GEMM, args, kwargs)(*args, **kwargs)

    @staticmethod
    def sparse_gemm(*args, **kwargs):
        from .dispatcher import select_impl
        return select_impl(KERNEL_SPARSE_GEMM, args, kwargs)(*args, **kwargs)

    @staticmethod
    def fused_dense_sparse(*args, **kwargs):
        from .dispatcher import select_impl
        return select_impl(KERNEL_FUSED_DENSE_SPARSE, args, kwargs)(*args, **kwargs)


__all__ = [
    "BackendKernel",
    "ALL_KERNELS",
    "KERNEL_ACTIVATION_QUANT",
    "KERNEL_DENSE_GEMM",
    "KERNEL_SPARSE_GEMM",
    "KERNEL_FUSED_DENSE_SPARSE",
]
