"""Kernel (1c) - Split-K Dense UINT4 x SINT4 GEMM for tiny-T decode.

Design rationale (see research/p4_splitk_dense_design.md):

For decode shapes with ``T <= 16`` and ``d_out <= 14336``, the plain
dense kernel launches a grid of only ``ceil(d_out / BM)`` programs
(e.g. 32 programs for ``d_out=4096, BM=128``).  RTX 4090 has 128 SMs,
so ~75% of them sit idle while the kernel runs.  HBM bandwidth
utilisation is only ~13%.  It is **not** HBM-bound -- it is
SM-occupancy bound.

This kernel splits the K axis into ``SPLIT_K`` chunks, producing a
grid of ``(cdiv(d_out, BM) * cdiv(T, BN), SPLIT_K)`` programs that
together cover all 128 SMs.  Each program accumulates its share of
the group-wise dequantisation sum into an FP32 partial, and a
separate reduce kernel folds the ``SPLIT_K`` partials into the
final FP16 output.

Shape conventions (inputs identical to dense_gemm_u4_s4):
  - W_low_packed : (d_out, d_in // 2) int8   SINT4 packed, little-endian
  - X_s4         : (T,     d_in // 2) int8   SINT4 packed, little-endian
  - scale_u4     : (d_out, n_groups)  fp16
  - zero_u4      : (d_out, n_groups)  fp16   (pre-subtracted 8)
  - sum_X        : (T,     n_groups)  int32
  - scale_x      : (T,)                fp16
  - Y_out        : (T,     d_out)      fp16   (final output, same as to_out)

Intermediate:
  - Y_partial    : (SPLIT_K, T, d_out) fp32

SPLIT_K policy
--------------
``SPLIT_K`` is chosen deterministically in the wrapper from ``(d_out, T, d_in)``
rather than being autotuned.  Triton's autotune key would need to include
SPLIT_K, splitting the cache and adding compile churn.  We pick:

    SPLIT_K = 1            if grid_mn * n_groups < SM_count * 2
    SPLIT_K = n_groups_div if n_groups is divisible and grid_mn * split >= SM_count
    SPLIT_K clamped to {1, 2, 4, 8}

so the main kernel sees a simple compile-time constant.

Numerical note
--------------
FP32 addition is not associative.  Reordering the per-group sum across
SPLIT_K partitions perturbs the result by ~n_groups * ULP(final), typically
<1e-4 absolute on LLM activations.  The reduce kernel explicitly does
``sum_{g in this split} + sum_{split} via FP32 adds`` so the only
reordering is at the split boundary.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .dense_u4s4_gemm import _unpack_packed_s4_rowmajor
from .pack_utils import BCOL


# ---------------------------------------------------------------------------
# Main split-K kernel
# ---------------------------------------------------------------------------
# Autotune covers only BM/BN/num_warps/num_stages (SPLIT_K fixed per call).
@triton.autotune(
    configs=[
        triton.Config({"BM": 64,  "BN": 16,  "BK": 128, "GROUP_SIZE_M": 1}, num_warps=2, num_stages=3),
        triton.Config({"BM": 128, "BN": 16,  "BK": 128, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 32,  "BK": 128, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 64,  "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64,  "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    ],
    key=["d_out", "d_in", "T", "SPLIT_K"],
)
@triton.jit
def dense_gemm_splitk_kernel(
    W_low_ptr,
    X_s4_ptr,
    scale_u4_ptr,
    zero_u4_ptr,
    sum_X_ptr,
    Y_partial_ptr,           # (SPLIT_K, T, d_out) fp32
    d_out, d_in, T,
    stride_w_m, stride_w_k,
    stride_x_n, stride_x_k,
    stride_su_m, stride_su_g,
    stride_zu_m, stride_zu_g,
    stride_sx_n, stride_sx_g,
    stride_yp_s, stride_yp_t, stride_yp_d,
    N_GROUPS: tl.constexpr,
    BCOL_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Grid: (cdiv(d_out, BM), cdiv(T, BN), SPLIT_K)
    split_id = tl.program_id(2)
    pid_m_raw = tl.program_id(0)
    pid_n_raw = tl.program_id(1)
    num_pid_m = tl.cdiv(d_out, BM)
    num_pid_n = tl.cdiv(T, BN)

    # L2-swizzle same as parent kernel
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    pid = pid_m_raw * num_pid_n + pid_n_raw
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < d_out
    mask_n = offs_n < T

    # FP32 accumulator for this split's contribution.
    # Note: we do NOT multiply by scale_x here -- scale_x is applied in the
    # reduce kernel so each split's partial is independent of the global scale.
    y_acc = tl.zeros((BM, BN), dtype=tl.float32)

    BK_HALF: tl.constexpr = BK // 2
    tl.static_assert(BK == BCOL_K, "BK must equal BCOL_K (group size)")

    offs_k_half = tl.arange(0, BK_HALF)

    n_k_iters = tl.cdiv(d_in, BK)
    # Split the K loop: program ``split_id`` handles groups
    # [split_id, split_id + SPLIT_K, split_id + 2*SPLIT_K, ...].
    # Stride-SPLIT_K stride keeps each split's groups interleaved across HBM,
    # which is the standard split-K trick to distribute HBM traffic evenly.
    for k_block in range(split_id, n_k_iters, SPLIT_K):
        k_byte_start = k_block * BK_HALF
        k_bytes = k_byte_start + offs_k_half
        k_bytes_mask = k_bytes < (d_in // 2)

        w_ptrs = W_low_ptr + offs_m[:, None] * stride_w_m + k_bytes[None, :] * stride_w_k
        w_packed = tl.load(
            w_ptrs,
            mask=mask_m[:, None] & k_bytes_mask[None, :],
            other=0,
        )
        x_ptrs = X_s4_ptr + offs_n[:, None] * stride_x_n + k_bytes[None, :] * stride_x_k
        x_packed = tl.load(
            x_ptrs,
            mask=mask_n[:, None] & k_bytes_mask[None, :],
            other=0,
        )

        w_tile = _unpack_packed_s4_rowmajor(w_packed, BM, BK)
        x_tile = _unpack_packed_s4_rowmajor(x_packed, BN, BK)

        x_tile_t = tl.trans(x_tile)
        acc_g = tl.dot(w_tile, x_tile_t, out_dtype=tl.int32)

        g = k_block
        scale_g = tl.load(
            scale_u4_ptr + offs_m * stride_su_m + g * stride_su_g,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        zero_g = tl.load(
            zero_u4_ptr + offs_m * stride_zu_m + g * stride_zu_g,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        sum_X_g = tl.load(
            sum_X_ptr + offs_n * stride_sx_n + g * stride_sx_g,
            mask=mask_n,
            other=0,
        ).to(tl.float32)

        corrected = acc_g.to(tl.float32) - zero_g[:, None] * sum_X_g[None, :]
        # NOTE: scale_x deliberately applied in the reduce pass, not here.
        y_acc += corrected * scale_g[:, None]

    # Write FP32 partial into (SPLIT_K, T, d_out) -- we transpose (BM, BN)
    # to (BN, BM) in registers and store into the (T, d_out) slice, so the
    # reduce kernel can read stride-1 along d_out.
    y_tile_t = tl.trans(y_acc)                            # (BN, BM) fp32
    yp_ptrs = (
        Y_partial_ptr
        + split_id * stride_yp_s
        + offs_n[:, None] * stride_yp_t
        + offs_m[None, :] * stride_yp_d
    )
    tl.store(yp_ptrs, y_tile_t, mask=mask_n[:, None] & mask_m[None, :])


# ---------------------------------------------------------------------------
# Reduce kernel: (SPLIT_K, T, d_out) fp32 -> (T, d_out) fp16
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 1,  "BLOCK_D": 256}, num_warps=2),
        triton.Config({"BLOCK_T": 1,  "BLOCK_D": 512}, num_warps=4),
        triton.Config({"BLOCK_T": 4,  "BLOCK_D": 128}, num_warps=2),
        triton.Config({"BLOCK_T": 4,  "BLOCK_D": 256}, num_warps=4),
        triton.Config({"BLOCK_T": 16, "BLOCK_D": 128}, num_warps=4),
    ],
    key=["T", "d_out", "SPLIT_K"],
)
@triton.jit
def splitk_reduce_kernel(
    Y_partial_ptr,           # (SPLIT_K, T, d_out) fp32
    scale_x_ptr,             # (T,) fp16
    Y_out_ptr,               # (T, d_out) fp16
    T, d_out,
    stride_yp_s, stride_yp_t, stride_yp_d,
    stride_yo_t, stride_yo_d,
    SPLIT_K: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_d = offs_d < d_out

    sx = tl.load(scale_x_ptr + offs_t, mask=mask_t, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
    # Static loop over SPLIT_K -- constexpr so Triton fully unrolls it.
    for s in tl.static_range(SPLIT_K):
        yp_ptrs = (
            Y_partial_ptr
            + s * stride_yp_s
            + offs_t[:, None] * stride_yp_t
            + offs_d[None, :] * stride_yp_d
        )
        part = tl.load(
            yp_ptrs,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        )
        acc += part

    # Apply global scale_x here (same as non-split kernel's epilogue).
    y = acc * sx[:, None]

    yo_ptrs = Y_out_ptr + offs_t[:, None] * stride_yo_t + offs_d[None, :] * stride_yo_d
    tl.store(yo_ptrs, y.to(tl.float16), mask=mask_t[:, None] & mask_d[None, :])


# ---------------------------------------------------------------------------
# SPLIT_K policy
# ---------------------------------------------------------------------------
# RTX 4090 has 128 SMs; we target >= 2 waves of coverage.
_SM_TARGET = 128


def _choose_split_k(d_out: int, T: int, d_in: int, bm_hint: int = 128, bn_hint: int = 16) -> int:
    """Pick SPLIT_K in {1, 2, 4, 8} so that grid_mn * SPLIT_K >= _SM_TARGET
    and SPLIT_K divides n_groups (to keep the K loop balanced).

    Heuristic tuned for RTX 4090; will be re-evaluated in bench_dense_splitk.
    """
    n_groups = d_in // BCOL
    grid_m = (d_out + bm_hint - 1) // bm_hint
    grid_n = (T + bn_hint - 1) // bn_hint
    grid_mn = grid_m * grid_n

    # Already enough coverage -> no split needed.
    if grid_mn >= _SM_TARGET:
        return 1

    # Candidate SPLIT_K values that evenly divide n_groups.
    candidates = [8, 4, 2, 1]
    for sk in candidates:
        if n_groups % sk != 0:
            continue
        if n_groups // sk < 2:
            # Each split must cover at least 2 groups to amortise startup.
            continue
        if grid_mn * sk >= _SM_TARGET:
            return sk

    # Fallback: no clean divisor found, try largest SPLIT_K that still divides.
    for sk in candidates:
        if n_groups % sk == 0 and n_groups // sk >= 2:
            return sk
    return 1


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def dense_gemm_u4_s4_splitk(
    W_low_packed: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    sum_X: torch.Tensor,
    scale_x: torch.Tensor,
    T: int | None = None,
    d_out: int | None = None,
    split_k: int | None = None,
) -> torch.Tensor:
    """Split-K Dense UINT4 x SINT4 GEMM -> (T, d_out) fp16.

    Drop-in alternative to ``dense_gemm_u4_s4_to_out`` for decode shapes
    where the non-split grid is too small to cover all SMs.
    """
    assert W_low_packed.is_cuda and X_s4.is_cuda
    assert W_low_packed.dtype == torch.int8 and X_s4.dtype == torch.int8
    _d_out, d_in_half = W_low_packed.shape
    _T, d_in_half_x = X_s4.shape
    assert d_in_half == d_in_half_x
    d_in = d_in_half * 2
    if T is None:
        T = _T
    else:
        assert T == _T
    if d_out is None:
        d_out = _d_out
    else:
        assert d_out == _d_out

    bcol = BCOL
    n_groups = d_in // bcol
    assert scale_u4.shape == (d_out, n_groups)
    assert zero_u4.shape == (d_out, n_groups)
    assert sum_X.shape == (T, n_groups)
    assert scale_x.shape == (T,)

    if split_k is None:
        split_k = _choose_split_k(d_out, T, d_in)
    else:
        assert split_k in (1, 2, 4, 8), f"split_k must be in {{1,2,4,8}}, got {split_k}"
        assert n_groups % split_k == 0, f"n_groups={n_groups} not divisible by split_k={split_k}"

    W_low_packed = W_low_packed.contiguous()
    X_s4 = X_s4.contiguous()
    scale_u4 = scale_u4.contiguous().to(torch.float16)
    zero_u4 = zero_u4.contiguous().to(torch.float16)
    sum_X = sum_X.contiguous().to(torch.int32)
    scale_x = scale_x.contiguous().to(torch.float16)

    device = W_low_packed.device

    # FP32 partials.  For SPLIT_K=1 this is a single surface = d_out*T*4 bytes,
    # still small enough to land in L2 for decode (<= 1 MiB).
    Y_partial = torch.empty((split_k, T, d_out), dtype=torch.float32, device=device)
    Y_out = torch.empty((T, d_out), dtype=torch.float16, device=device)

    grid_main = lambda META: (
        triton.cdiv(d_out, META["BM"]),
        triton.cdiv(T, META["BN"]),
        split_k,
    )
    dense_gemm_splitk_kernel[grid_main](
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, Y_partial,
        d_out, d_in, T,
        W_low_packed.stride(0), W_low_packed.stride(1),
        X_s4.stride(0), X_s4.stride(1),
        scale_u4.stride(0), scale_u4.stride(1),
        zero_u4.stride(0), zero_u4.stride(1),
        sum_X.stride(0), sum_X.stride(1),
        Y_partial.stride(0), Y_partial.stride(1), Y_partial.stride(2),
        N_GROUPS=n_groups,
        BCOL_K=bcol,
        SPLIT_K=split_k,
    )

    grid_reduce = lambda META: (
        triton.cdiv(T, META["BLOCK_T"]),
        triton.cdiv(d_out, META["BLOCK_D"]),
    )
    splitk_reduce_kernel[grid_reduce](
        Y_partial, scale_x, Y_out,
        T, d_out,
        Y_partial.stride(0), Y_partial.stride(1), Y_partial.stride(2),
        Y_out.stride(0), Y_out.stride(1),
        SPLIT_K=split_k,
    )
    return Y_out


__all__ = [
    "dense_gemm_splitk_kernel",
    "splitk_reduce_kernel",
    "dense_gemm_u4_s4_splitk",
    "_choose_split_k",
]
