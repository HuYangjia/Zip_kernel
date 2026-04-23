"""CUDA kernel Python bindings.

Builds the C++/CUDA extension via ``torch.utils.cpp_extension.load``
on first import, then exposes one Python-level wrapper per kernel with
a signature that matches the corresponding Triton wrapper in
:mod:`kernel.triton_kernel`.  This signature match is what allows
:mod:`kernel.backend.dispatcher` to swap backends transparently.

Stub semantics
--------------
Kernels that are not yet implemented in this phase set their
``<name>_cuda`` symbol to ``None``.  ``kernel.backend.registry`` checks
for ``callable(fn)`` and skips stubs automatically, so the dispatcher
transparently routes those calls back to Triton without any explicit
branching elsewhere.

Phase 1 implementation status
-----------------------------
- activation_quant        : implemented (see csrc/activation_quant/)
- dense_gemm              : STUB (Phase 2)
- sparse_gemm             : STUB (Phase 3)
- fused_dense_sparse      : STUB (Phase 4)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import torch

from kernel.triton_kernel.pack_utils import BCOL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Build the extension once on import
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_CSRC = _HERE / "csrc"

_SOURCES = [
    str(_CSRC / "bindings.cc"),
    str(_CSRC / "activation_quant" / "activation_quant.cu"),
    # --- Phase 2+ sources are declared here but their .cu files will
    # --- contain empty host launchers until the real kernels land, so
    # --- the extension still links.
    str(_CSRC / "dense_gemm" / "dense_gemm.cu"),
    str(_CSRC / "sparse_gemm" / "sparse_gemm.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_dense_sparse.cu"),
]

_NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    # SM89 = RTX 4090 / Ada Lovelace.  We deliberately do NOT emit PTX
    # (``-gencode=arch=compute_89,code=sm_89`` only, no ``,compute_89``
    # trailing fragment) to keep the .so small; running on a different
    # arch will fail loudly rather than silently JIT-recompile PTX.
    "-gencode=arch=compute_89,code=sm_89",
    # NB: --use_fast_math is deliberately NOT set.  It would enable
    # --prec-div=false which turns ``x / s`` into an approximate
    # reciprocal-multiply, breaking the bit-exact parity we require
    # against the Triton reference in activation_quant.  We can opt
    # individual GEMM kernels into fast-math later via pragmas.
    "--fmad=true",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

_CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fvisibility=hidden",
]


def _build_extension():
    """JIT-build the CUDA extension.  Returns the loaded module.

    Raises on failure; caller (:mod:`kernel.backend.registry`) catches
    all exceptions and downgrades the CUDA backend to unavailable.
    """
    from torch.utils.cpp_extension import load

    include_dirs = [str(_CSRC)]
    build_dir = os.environ.get(
        "HKUST_V9_CUDA_BUILD_DIR",
        str(Path.home() / ".cache" / "hkust_v9_cuda"),
    )
    os.makedirs(build_dir, exist_ok=True)

    logger.info(
        "Building V9 CUDA extension (SM89); build dir = %s", build_dir
    )
    mod = load(
        name="hkust_v9_cuda",
        sources=_SOURCES,
        extra_cflags=_CXX_FLAGS,
        extra_cuda_cflags=_NVCC_FLAGS,
        extra_include_paths=include_dirs,
        build_directory=build_dir,
        verbose=os.environ.get("HKUST_V9_CUDA_VERBOSE", "0") == "1",
    )
    return mod


# Eager build at import time.  On SM89 this is a 20-40 s one-shot cost
# (cached by ninja); on non-SM89 machines this raises and
# :mod:`kernel.backend.registry` downgrades to Triton-only silently.
_ext = _build_extension()


# ---------------------------------------------------------------------------
# Python-level wrappers
# ---------------------------------------------------------------------------


def activation_quant_cuda(
    X_fp16: torch.Tensor,
    perm: torch.Tensor,
    bcol: int = BCOL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused per-token SINT4 activation quantization (CUDA).

    Signature is byte-compatible with
    :func:`kernel.triton_kernel.activation_quant.quantize_activation_s4`:
    the wrapper accepts 2D or 3D ``X_fp16`` and returns the same three
    tensors (``X_s4``, ``scale_x``, ``sum_X``) with identical dtype /
    shape / memory-layout contract.

    The CUDA kernel always handles the permuted gather on-chip; the
    Triton version's ``_L2_THRASH_THRESHOLD_ELEMS`` pre-permute path is
    unnecessary because our gather is tile-local and uses LDG.128 with
    better L1 hit rate.  We still respect the contract that a ``perm ==
    arange`` input produces identity-order output.
    """
    assert X_fp16.is_cuda, "activation_quant_cuda requires a CUDA tensor"
    assert X_fp16.dtype == torch.float16, "X must be fp16"
    assert perm.dtype in (torch.int32, torch.int64), "perm must be int32/int64"

    original_shape = X_fp16.shape
    if X_fp16.dim() == 3:
        T, D = original_shape[0] * original_shape[1], original_shape[2]
    elif X_fp16.dim() == 2:
        T, D = original_shape
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {original_shape}")

    if D % bcol != 0:
        raise ValueError(f"d_in ({D}) must be divisible by bcol ({bcol})")
    if D % 2 != 0:
        raise ValueError(f"d_in ({D}) must be even for 4-bit packing")

    X_2d = X_fp16.reshape(T, D).contiguous()
    perm_i32 = perm.to(torch.int32).contiguous()

    n_groups = D // bcol
    device = X_2d.device
    X_s4 = torch.empty((T, D // 2), dtype=torch.int8, device=device)
    scale_x = torch.empty((T,), dtype=torch.float16, device=device)
    sum_X = torch.empty((T, n_groups), dtype=torch.int32, device=device)

    _ext.activation_quant_launch(
        X_2d, perm_i32, X_s4, scale_x, sum_X,
        int(T), int(D), int(bcol),
    )
    return X_s4, scale_x, sum_X


# ---------------------------------------------------------------------------
# Stubs (Phase 2+) -- set to None so the registry skips them and the
# dispatcher falls back to Triton for these kernels.
# ---------------------------------------------------------------------------


def dense_gemm_cuda(
    W_low_packed: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    sum_X: torch.Tensor,
    scale_x: torch.Tensor,
) -> torch.Tensor:
    """Dense UINT4 x SINT4 GEMM (CUDA).

    Signature matches :func:`kernel.triton_kernel.dense_u4s4_gemm.dense_gemm_u4_s4`
    exactly: accepts packed LE SINT4 weights/activations plus the
    per-group dequant metadata and returns ``Y_low`` of shape
    ``(d_out, T)`` in fp16.
    """
    assert W_low_packed.is_cuda and X_s4.is_cuda
    assert W_low_packed.dtype == torch.int8 and X_s4.dtype == torch.int8
    d_out, d_in_half = W_low_packed.shape
    T = X_s4.shape[0]
    d_in = d_in_half * 2
    n_groups = d_in // BCOL

    assert X_s4.shape == (T, d_in_half)
    assert scale_u4.shape == (d_out, n_groups)
    assert zero_u4.shape == (d_out, n_groups)
    assert sum_X.shape == (T, n_groups)
    assert scale_x.shape == (T,)

    W_low_packed = W_low_packed.contiguous()
    X_s4 = X_s4.contiguous()
    scale_u4 = scale_u4.contiguous().to(torch.float16)
    zero_u4 = zero_u4.contiguous().to(torch.float16)
    sum_X = sum_X.contiguous().to(torch.int32)
    scale_x = scale_x.contiguous().to(torch.float16)

    Y_low = torch.empty((d_out, T), dtype=torch.float16,
                        device=W_low_packed.device)
    _ext.dense_gemm_launch(
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, Y_low
    )
    return Y_low


def sparse_gemm_cuda(
    W_high_blocks_packed: torch.Tensor,
    hp_row_offsets: torch.Tensor,
    hp_col_indices: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    scale_x: torch.Tensor,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """BSR sparse SINT4 x SINT4 GEMM (CUDA).

    Signature matches
    :func:`kernel.triton_kernel.sparse_s4s4_gemm.sparse_gemm_s4_s4`.
    Returns a freshly-allocated ``Y_high`` of shape ``(d_out, T)`` in
    fp16, zero-initialized for block-rows with no high-precision blocks.
    """
    assert X_s4.is_cuda
    T = X_s4.shape[0]

    Y_high = torch.zeros((d_out, T), dtype=torch.float16, device=X_s4.device)
    if W_high_blocks_packed.shape[0] == 0:
        return Y_high

    W_high_blocks_packed = W_high_blocks_packed.contiguous()
    hp_row_offsets = hp_row_offsets.contiguous().to(torch.int32)
    hp_col_indices = hp_col_indices.contiguous().to(torch.int32)
    X_s4 = X_s4.contiguous()
    scale_u4 = scale_u4.contiguous().to(torch.float16)
    scale_x = scale_x.contiguous().to(torch.float16)

    _ext.sparse_gemm_launch(
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, scale_x, Y_high,
        int(d_out), int(d_in),
    )
    return Y_high


def fused_dense_sparse_cuda(
    W_low_packed: torch.Tensor,
    W_high_blocks_packed: torch.Tensor,
    hp_row_offsets: torch.Tensor,
    hp_col_indices: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    sum_X: torch.Tensor,
    scale_x: torch.Tensor,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Fused dense + sparse GEMM (CUDA).

    Semantics:  ``Y_total[m, n] = Y_low[m, n] + 16 * Y_high[m, n]``.
    Signature matches
    :func:`kernel.triton_kernel.fused_dense_sparse_gemm.fused_dense_sparse_gemm`.
    """
    assert W_low_packed.is_cuda and X_s4.is_cuda
    assert W_low_packed.dtype == torch.int8 and X_s4.dtype == torch.int8
    T = X_s4.shape[0]
    d_in_half = d_in // 2
    n_groups = d_in // BCOL
    assert X_s4.shape == (T, d_in_half)
    assert scale_u4.shape == (d_out, n_groups)
    assert zero_u4.shape == (d_out, n_groups)
    assert sum_X.shape == (T, n_groups)
    assert scale_x.shape == (T,)

    W_low_packed = W_low_packed.contiguous()
    X_s4 = X_s4.contiguous()
    scale_u4 = scale_u4.contiguous().to(torch.float16)
    zero_u4 = zero_u4.contiguous().to(torch.float16)
    sum_X = sum_X.contiguous().to(torch.int32)
    scale_x = scale_x.contiguous().to(torch.float16)

    # Normalise empty BSR: the CUDA launcher also handles this, but we
    # materialise the zero tensor on this side so stride(2)==1 holds.
    if W_high_blocks_packed.numel() == 0:
        W_high_blocks_packed = torch.zeros(
            (0, 128, BCOL // 2),
            dtype=torch.int8,
            device=W_low_packed.device,
        )
    W_high_blocks_packed = W_high_blocks_packed.contiguous()
    hp_row_offsets = hp_row_offsets.contiguous().to(torch.int32)
    hp_col_indices = hp_col_indices.contiguous().to(torch.int32)

    Y_total = torch.empty((d_out, T), dtype=torch.float16,
                          device=W_low_packed.device)
    _ext.fused_dense_sparse_launch(
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4,
        scale_u4, zero_u4, sum_X, scale_x,
        Y_total,
        int(d_out), int(d_in),
    )
    return Y_total


__all__ = [
    "activation_quant_cuda",
    "dense_gemm_cuda",
    "sparse_gemm_cuda",
    "fused_dense_sparse_cuda",
]
