"""Naive CUDA kernel Python bindings (reference baseline for ``ops.py``).

Builds a *separate* torch extension ``hkust_v9_cuda_naive`` on first
import so it coexists with the optimised ``hkust_v9_cuda`` without
symbol collisions.  The extension exposes four kernels only:

  * activation_quant_naive   (per-token SINT4 quant + pack + sum_X)
  * dense_gemm_naive         (UINT4 x SINT4 tiled GEMM, INT4 Tensor Core)
  * sparse_gemm_naive        (SINT4 x SINT4 BSR GEMM, INT4 Tensor Core)
  * reduce_sum_naive         (Y_total = Y_low + Y_high)

The Python wrappers here mirror the *pre-quantised* two-step path in
``ops.py`` but with three key differences:

  1. ``dense_gemm_naive`` and ``sparse_gemm_naive`` do NOT produce a
     fused Y_total; each writes its own buffer, and the caller must
     invoke ``reduce_sum_naive`` to get Y_total.
  2. There is NO T=1 GEMV specialisation, NO P0 fused quant path, and
     NO CUTLASS dispatch.  One code path for every shape.
  3. Every tensor layout matches the optimised kernel byte-for-byte
     (W_low uint4 packing, W_high_blocks BSR, scale/zero fp16 per
     group, sum_X int32) so the two backends can be parity-compared on
     the exact same random inputs.

Intended consumer: ``kernel.bench.layer.qwen3_w4a4_ops_naive``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import torch

from kernel.triton_kernel.pack_utils import BCOL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build the extension once on import
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_CSRC = _HERE / "csrc_naive"

_SOURCES = [
    str(_CSRC / "bindings_naive.cc"),
    str(_CSRC / "activation_quant_naive.cu"),
    str(_CSRC / "dense_gemm_naive.cu"),
    str(_CSRC / "sparse_gemm_naive.cu"),
    str(_CSRC / "reduce_sum_naive.cu"),
]

_NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-gencode=arch=compute_89,code=sm_89",
    "--fmad=true",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

_CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fvisibility=hidden",
]


def _build_extension():
    from torch.utils.cpp_extension import load

    build_dir = os.environ.get(
        "HKUST_V9_CUDA_NAIVE_BUILD_DIR",
        str(Path.home() / ".cache" / "hkust_v9_cuda_naive"),
    )
    os.makedirs(build_dir, exist_ok=True)

    logger.info(
        "Building NAIVE V9 CUDA extension (SM89); build dir = %s", build_dir
    )
    mod = load(
        name="hkust_v9_cuda_naive",
        sources=_SOURCES,
        extra_cflags=_CXX_FLAGS,
        extra_cuda_cflags=_NVCC_FLAGS,
        build_directory=build_dir,
        verbose=os.environ.get("HKUST_V9_CUDA_NAIVE_VERBOSE", "0") == "1",
    )
    return mod


_ext = _build_extension()


# ---------------------------------------------------------------------------
# activation_quant
# ---------------------------------------------------------------------------


def activation_quant_naive(
    X_fp16: torch.Tensor,
    perm: torch.Tensor,
    bcol: int = BCOL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Naive per-token SINT4 activation quantization (CUDA)."""
    assert X_fp16.is_cuda and X_fp16.dtype == torch.float16
    assert perm.dtype in (torch.int32, torch.int64)

    if X_fp16.dim() == 3:
        T, D = X_fp16.shape[0] * X_fp16.shape[1], X_fp16.shape[2]
    elif X_fp16.dim() == 2:
        T, D = X_fp16.shape
    else:
        raise ValueError(f"X must be 2D or 3D, got {X_fp16.shape}")

    assert D % bcol == 0 and D % 2 == 0
    X_2d = X_fp16.reshape(T, D).contiguous()
    perm_i32 = perm.to(torch.int32).contiguous()

    n_groups = D // bcol
    device = X_2d.device
    X_s4    = torch.empty((T, D // 2),      dtype=torch.int8,  device=device)
    scale_x = torch.empty((T,),             dtype=torch.float16, device=device)
    sum_X   = torch.empty((T, n_groups),    dtype=torch.int32, device=device)

    _ext.activation_quant_naive_launch(
        X_2d, perm_i32, X_s4, scale_x, sum_X,
        int(T), int(D), int(bcol),
    )
    return X_s4, scale_x, sum_X


# ---------------------------------------------------------------------------
# dense GEMM
# ---------------------------------------------------------------------------


def dense_gemm_naive(
    W_low: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    sum_X: torch.Tensor,
    scale_x: torch.Tensor,
) -> torch.Tensor:
    """Naive UINT4 × SINT4 dense GEMM (CUDA, INT4 Tensor Core)."""
    assert W_low.is_cuda and W_low.dtype == torch.int8
    assert X_s4.is_cuda and X_s4.dtype == torch.int8
    d_out, d_in_half = W_low.shape
    T = X_s4.shape[0]
    d_in = d_in_half * 2

    W_low_c    = W_low.contiguous()
    X_s4_c     = X_s4.contiguous()
    scale_u4_c = scale_u4.contiguous().to(torch.float16)
    zero_u4_c  = zero_u4.contiguous().to(torch.float16)
    sum_X_c    = sum_X.contiguous().to(torch.int32)
    scale_x_c  = scale_x.contiguous().to(torch.float16)

    Y_low = torch.empty((d_out, T), dtype=torch.float16, device=W_low.device)
    _ext.dense_gemm_naive_launch(
        W_low_c, X_s4_c, scale_u4_c, zero_u4_c, sum_X_c, scale_x_c, Y_low
    )
    return Y_low


# ---------------------------------------------------------------------------
# sparse GEMM
# ---------------------------------------------------------------------------


def sparse_gemm_naive(
    W_high_blocks: torch.Tensor,
    hp_row_offsets: torch.Tensor,
    hp_col_indices: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    scale_x: torch.Tensor,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Naive SINT4 × SINT4 BSR GEMM (CUDA, INT4 Tensor Core)."""
    T = X_s4.shape[0]
    Y_high = torch.zeros((d_out, T), dtype=torch.float16, device=X_s4.device)

    # Empty-BSR short-circuit: launcher still accepts zero blocks but
    # we avoid an unnecessary kernel launch.  (Match ops.py behaviour.)
    if W_high_blocks.shape[0] == 0:
        return Y_high

    _ext.sparse_gemm_naive_launch(
        W_high_blocks.contiguous(),
        hp_row_offsets.contiguous().to(torch.int32),
        hp_col_indices.contiguous().to(torch.int32),
        X_s4.contiguous(),
        scale_u4.contiguous().to(torch.float16),
        scale_x.contiguous().to(torch.float16),
        Y_high,
        int(d_out), int(d_in),
    )
    return Y_high


# ---------------------------------------------------------------------------
# reduce sum
# ---------------------------------------------------------------------------


def reduce_sum_naive(Y_low: torch.Tensor, Y_high: torch.Tensor) -> torch.Tensor:
    """Naive element-wise add: Y_total = Y_low + Y_high."""
    assert Y_low.shape == Y_high.shape
    Y_total = torch.empty_like(Y_low)
    _ext.reduce_sum_naive_launch(
        Y_low.contiguous(), Y_high.contiguous(), Y_total
    )
    return Y_total


__all__ = [
    "activation_quant_naive",
    "dense_gemm_naive",
    "sparse_gemm_naive",
    "reduce_sum_naive",
]
