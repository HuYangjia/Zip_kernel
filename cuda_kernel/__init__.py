"""V9 CUDA kernel package (RTX 4090 / SM89).

This package is a drop-in replacement for selected sub-kernels in
``kernel.triton_kernel``.  It is **never** imported directly by user
code; instead :mod:`kernel.backend` loads it opportunistically and
falls back to Triton if the JIT compilation step fails (e.g. on a
machine without nvcc, or with a non-SM89 GPU).

Build strategy
--------------
We use :func:`torch.utils.cpp_extension.load` for just-in-time
compilation.  On first import the ``.cu`` / ``.cc`` sources under
``csrc/`` are compiled with ``-gencode=arch=compute_89,code=sm_89``
and the resulting ``.so`` is cached under ``~/.cache/torch_extensions``.
Subsequent imports reuse the cache.

This choice deliberately avoids a ``setup.py build_ext`` step so that
iterating on kernels does not require reinstalling the package.  The
cost is one ~30 s JIT compile on first run after a ``.cu`` change.
"""

from __future__ import annotations

# Importing ``ops`` triggers the JIT build; leave it eager so that
# :mod:`kernel.backend.registry` can detect build failures at import
# time and downgrade the CUDA backend cleanly.
#
# Kernel set (all Tensor-Core based):
#   activation_quant (fused per-token quant)
#   dense_gemm_mma_int8 / dense_gemm_mma_int4
#   sparse_gemm_mma_int8 / sparse_gemm_mma_int4
#   fused_dense_sparse_mma_int8 / fused_dense_sparse_mma_int4
#
# The dp4a (SIMT) implementations that preceded this phase have been
# retired -- on SM89 the INT8 Tensor Core path supersedes them by
# ~4x peak throughput.
from . import ops  # noqa: F401

__all__ = ["ops"]
