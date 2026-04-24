"""CUDA kernel Python bindings.

Builds the C++/CUDA extension via ``torch.utils.cpp_extension.load``
on first import, then exposes one Python-level wrapper per kernel with
a signature that matches the corresponding Triton wrapper in
:mod:`kernel.triton_kernel`.  This signature match is what allows
:mod:`kernel.backend.dispatcher` to swap backends transparently.

Kernel set
----------
- activation_quant (CUDA, fused per-token quant)
- dense_gemm_cuda_int8   : mma.m16n8k32.s8.s8.s32 (INT8 Tensor Core)
- dense_gemm_cuda_int4   : mma.m16n8k64.s4.s4.s32 (INT4 Tensor Core)
- sparse_gemm_cuda_int8  : ditto for BSR s4xs4
- sparse_gemm_cuda_int4  : ditto, INT4 MMA
- fused_dense_sparse_cuda_int8 / _int4

For backward compatibility with existing callers
(:mod:`kernel.backend.dispatcher`), ``dense_gemm_cuda`` /
``sparse_gemm_cuda`` / ``fused_dense_sparse_cuda`` are aliased to the
INT8 variant (the performance-robust default on SM89; INT4 MMA on Ada
is deprecated hardware path).
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
    # MMA variants (Tensor Core); the dp4a .cu files in each subdir are
    # left as empty stubs and deliberately excluded from the source
    # list so that we don't spend compile time on them.
    str(_CSRC / "dense_gemm"         / "dense_gemm_mma_int8.cu"),
    str(_CSRC / "dense_gemm"         / "dense_gemm_mma_int4.cu"),
    str(_CSRC / "sparse_gemm"        / "sparse_gemm_mma_int8.cu"),
    str(_CSRC / "sparse_gemm"        / "sparse_gemm_mma_int4.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_dense_sparse_mma_int8.cu"),
    str(_CSRC / "fused_dense_sparse" / "fused_dense_sparse_mma_int4.cu"),
]

_NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    # SM89 = RTX 4090 / Ada Lovelace.
    "-gencode=arch=compute_89,code=sm_89",
    "--fmad=true",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--ptxas-options=-v",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    # mma.s4 is deprecated starting with PTX 8.7 (the compiler we use
    # still accepts it and ptxas still maps it to Tensor Core on Ada,
    # but emits a warning); suppress to keep build logs clean.
    "-Wno-deprecated-declarations",
]

_CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fvisibility=hidden",
]


def _build_extension():
    from torch.utils.cpp_extension import load

    include_dirs = [str(_CSRC)]
    build_dir = os.environ.get(
        "HKUST_V9_CUDA_BUILD_DIR",
        str(Path.home() / ".cache" / "hkust_v9_cuda"),
    )
    os.makedirs(build_dir, exist_ok=True)

    logger.info(
        "Building V9 CUDA extension (SM89, MMA); build dir = %s", build_dir
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


_ext = _build_extension()


# ---------------------------------------------------------------------------
# activation_quant (unchanged)
# ---------------------------------------------------------------------------


def activation_quant_cuda(
    X_fp16: torch.Tensor,
    perm: torch.Tensor,
    bcol: int = BCOL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused per-token SINT4 activation quantization (CUDA)."""
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
# Common argument prep shared by all GEMM variants.
# ---------------------------------------------------------------------------


def _prepare_dense_args(
    W_low_packed: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    sum_X: torch.Tensor,
    scale_x: torch.Tensor,
):
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
    return (W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, Y_low)


# ---------------------------------------------------------------------------
# Dense GEMM -- INT8 and INT4 MMA variants
# ---------------------------------------------------------------------------


def dense_gemm_cuda_int8(
    W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
) -> torch.Tensor:
    """Dense UINT4 x SINT4 GEMM via mma.m16n8k32.s8 (CUDA, SM89)."""
    args = _prepare_dense_args(W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x)
    _ext.dense_gemm_mma_int8_launch(*args)
    return args[-1]


def dense_gemm_cuda_int4(
    W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
) -> torch.Tensor:
    """Dense UINT4 x SINT4 GEMM via mma.m16n8k64.s4 (CUDA, SM89)."""
    args = _prepare_dense_args(W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x)
    _ext.dense_gemm_mma_int4_launch(*args)
    return args[-1]


# Default alias: INT8 MMA (robust on SM89).
dense_gemm_cuda = dense_gemm_cuda_int8


# ---------------------------------------------------------------------------
# Sparse GEMM
# ---------------------------------------------------------------------------


def _prepare_sparse_args(
    W_high_blocks_packed, hp_row_offsets, hp_col_indices,
    X_s4, scale_u4, scale_x, d_out, d_in,
):
    assert X_s4.is_cuda
    T = X_s4.shape[0]

    Y_high = torch.zeros((d_out, T), dtype=torch.float16, device=X_s4.device)
    if W_high_blocks_packed.shape[0] == 0:
        return None, Y_high  # signal: empty BSR; skip launch

    W_high_blocks_packed = W_high_blocks_packed.contiguous()
    hp_row_offsets = hp_row_offsets.contiguous().to(torch.int32)
    hp_col_indices = hp_col_indices.contiguous().to(torch.int32)
    X_s4 = X_s4.contiguous()
    scale_u4 = scale_u4.contiguous().to(torch.float16)
    scale_x = scale_x.contiguous().to(torch.float16)

    return (
        (W_high_blocks_packed, hp_row_offsets, hp_col_indices,
         X_s4, scale_u4, scale_x, Y_high,
         int(d_out), int(d_in)),
        Y_high,
    )


def sparse_gemm_cuda_int8(
    W_high_blocks_packed, hp_row_offsets, hp_col_indices,
    X_s4, scale_u4, scale_x, d_out, d_in,
) -> torch.Tensor:
    """BSR sparse SINT4 x SINT4 GEMM via mma.m16n8k32.s8 (CUDA, SM89)."""
    prepared = _prepare_sparse_args(
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, scale_x, d_out, d_in
    )
    args, Y_high = prepared
    if args is None:
        return Y_high
    _ext.sparse_gemm_mma_int8_launch(*args)
    return Y_high


def sparse_gemm_cuda_int4(
    W_high_blocks_packed, hp_row_offsets, hp_col_indices,
    X_s4, scale_u4, scale_x, d_out, d_in,
) -> torch.Tensor:
    """BSR sparse SINT4 x SINT4 GEMM via mma.m16n8k64.s4 (CUDA, SM89)."""
    prepared = _prepare_sparse_args(
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, scale_x, d_out, d_in
    )
    args, Y_high = prepared
    if args is None:
        return Y_high
    _ext.sparse_gemm_mma_int4_launch(*args)
    return Y_high


sparse_gemm_cuda = sparse_gemm_cuda_int8


# ---------------------------------------------------------------------------
# Fused dense+sparse GEMM
# ---------------------------------------------------------------------------


def _prepare_fused_args(
    W_low_packed, W_high_blocks_packed, hp_row_offsets, hp_col_indices,
    X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
):
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
    return (
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4,
        scale_u4, zero_u4, sum_X, scale_x,
        Y_total,
        int(d_out), int(d_in),
    ), Y_total


def fused_dense_sparse_cuda_int8(
    W_low_packed, W_high_blocks_packed, hp_row_offsets, hp_col_indices,
    X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
) -> torch.Tensor:
    """Fused dense+sparse GEMM via mma.m16n8k32.s8 (CUDA, SM89)."""
    args, Y_total = _prepare_fused_args(
        W_low_packed, W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
    )
    _ext.fused_dense_sparse_mma_int8_launch(*args)
    return Y_total


def fused_dense_sparse_cuda_int4(
    W_low_packed, W_high_blocks_packed, hp_row_offsets, hp_col_indices,
    X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
) -> torch.Tensor:
    """Fused dense+sparse GEMM via mma.m16n8k64.s4 (CUDA, SM89)."""
    args, Y_total = _prepare_fused_args(
        W_low_packed, W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
    )
    _ext.fused_dense_sparse_mma_int4_launch(*args)
    return Y_total


fused_dense_sparse_cuda = fused_dense_sparse_cuda_int8


__all__ = [
    "activation_quant_cuda",
    # GEMM default aliases (INT8 MMA)
    "dense_gemm_cuda",
    "sparse_gemm_cuda",
    "fused_dense_sparse_cuda",
    # Explicit INT8 MMA entry points
    "dense_gemm_cuda_int8",
    "sparse_gemm_cuda_int8",
    "fused_dense_sparse_cuda_int8",
    # Explicit INT4 MMA entry points
    "dense_gemm_cuda_int4",
    "sparse_gemm_cuda_int4",
    "fused_dense_sparse_cuda_int4",
]
