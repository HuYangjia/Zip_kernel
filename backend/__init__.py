"""V9 Linear backend dispatch layer.

This package is the **only** module external callers should import from.
It owns the runtime decision of whether to execute a given sub-kernel on
the Triton backend (``kernel.triton_kernel``) or the CUDA backend
(``kernel.cuda_kernel``), per-kernel and per-shape.

Public API
----------
- ``v9_linear_forward``        : shape-agnostic entry, auto-dispatch decode/prefill
- ``v9_linear_forward_decode`` : explicit decode path
- ``v9_linear_forward_prefill``: explicit prefill path
- ``v9_linear_fakequant``      : reference fp16 forward for correctness checks
- ``set_backend_policy``       : override the per-kernel backend selection
- ``get_backend_status``       : report CUDA availability and current policy
- ``v9_linear_forward_cuda_graph``: opt-in CUDA Graph replay entry (R49 Step 1;
                                    behaviour-compatible with ``v9_linear_forward``)
- ``set_cuda_graph_policy``    : enable/disable/force the CUDA Graph path

Isolation guarantee
-------------------
- ``kernel.triton_kernel`` never imports from ``kernel.cuda_kernel`` and
  vice versa; only this package knows about both backends.  When the
  CUDA extension fails to build (e.g. non-SM89 machines) the dispatcher
  transparently falls back to the Triton implementation for every
  kernel, so the public API keeps working.
"""

from __future__ import annotations

# R50 L4: Allow importing pure-python sub-modules
# (``kernel.backend.weight_loader``) from triton-less hosts (Mac
# dev boxes). The eager dispatcher/graph_cache imports below would
# otherwise explode with ``ModuleNotFoundError: triton`` at import
# time. We soft-fail the triton-dependent surface exactly like
# ``kernel.cuda_kernel.__init__`` does for the JIT build.

try:
    from .dispatcher import (
        v9_linear_fakequant,
        v9_linear_forward,
        v9_linear_forward_decode,
        v9_linear_forward_prefill,
    )
    from .graph_cache import (
        clear_cuda_graph_cache,
        cuda_graph_cache_stats,
        get_cuda_graph_policy,
        prewarm_cuda_graph_cache,
        set_cuda_graph_policy,
        v9_linear_forward_cuda_graph,
    )
    from .policy import get_backend_status, set_backend_policy
    from .registry import BackendKernel

    _TRITON_BACKEND_AVAILABLE = True
    _TRITON_BACKEND_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as _exc:  # pragma: no cover - triton-less hosts only
    import warnings

    _TRITON_BACKEND_AVAILABLE = False
    _TRITON_BACKEND_IMPORT_ERROR = _exc

    # Provide placeholder names so ``from kernel.backend import X`` fails
    # only at use-site, not at import-time, for triton-dependent symbols.
    def _missing_triton(*_a, **_k):  # pragma: no cover - trivial
        raise RuntimeError(
            "kernel.backend triton-dependent path is unavailable on this "
            f"host ({type(_TRITON_BACKEND_IMPORT_ERROR).__name__}: "
            f"{_TRITON_BACKEND_IMPORT_ERROR}). Pure-python helpers under "
            "kernel.backend.weight_loader remain importable."
        )

    v9_linear_forward = _missing_triton  # type: ignore[assignment]
    v9_linear_forward_decode = _missing_triton  # type: ignore[assignment]
    v9_linear_forward_prefill = _missing_triton  # type: ignore[assignment]
    v9_linear_fakequant = _missing_triton  # type: ignore[assignment]
    v9_linear_forward_cuda_graph = _missing_triton  # type: ignore[assignment]
    set_cuda_graph_policy = _missing_triton  # type: ignore[assignment]
    get_cuda_graph_policy = _missing_triton  # type: ignore[assignment]
    prewarm_cuda_graph_cache = _missing_triton  # type: ignore[assignment]
    cuda_graph_cache_stats = _missing_triton  # type: ignore[assignment]
    clear_cuda_graph_cache = _missing_triton  # type: ignore[assignment]
    set_backend_policy = _missing_triton  # type: ignore[assignment]
    get_backend_status = _missing_triton  # type: ignore[assignment]
    BackendKernel = None  # type: ignore[assignment]

    warnings.warn(
        "kernel.backend triton-dependent surface disabled "
        f"({type(_exc).__name__}: {_exc}); pure-python helpers "
        "(weight_loader, CutlassV9Tensors) remain importable.",
        RuntimeWarning,
        stacklevel=2,
    )

from .weight_loader import (
    CutlassPackValidationError,
    CutlassV9Tensors,
    pack_v9_weights_for_cutlass,
)

__all__ = [
    "v9_linear_forward",
    "v9_linear_forward_decode",
    "v9_linear_forward_prefill",
    "v9_linear_fakequant",
    "set_backend_policy",
    "get_backend_status",
    "BackendKernel",
    # R49 Step 1 — CUDA Graph opt-in path (see graph_cache.py)
    "v9_linear_forward_cuda_graph",
    "set_cuda_graph_policy",
    "get_cuda_graph_policy",
    "prewarm_cuda_graph_cache",
    "cuda_graph_cache_stats",
    "clear_cuda_graph_cache",
    # R50 L4.1 — CUTLASS INT4 weight-loader adapter (see weight_loader.py)
    "pack_v9_weights_for_cutlass",
    "CutlassV9Tensors",
    "CutlassPackValidationError",
]
