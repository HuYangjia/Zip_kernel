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
]
