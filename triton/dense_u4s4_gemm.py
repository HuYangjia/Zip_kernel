"""Kernel (1) - Dense UINT4 x SINT4 GEMM for the V9 low-bit layer.

Both operands are treated as SINT4 in the MMA path (the UINT4 -> SINT4 offset
is absorbed offline into `zero_u4` by `pack_utils.pack_v9_weights`).

Shape conventions
-----------------
- W_low_packed : (d_out, d_in // 2) int8   SINT4 packed, little-endian
- X_s4         : (T,     d_in // 2) int8   SINT4 packed, little-endian
- scale_u4     : (d_out, n_groups)  fp16
- zero_u4      : (d_out, n_groups)  fp16   (pre-subtracted 8)
- sum_X        : (T,     n_groups)  int32
- scale_x      : (T,)                fp16
- Y_low        : (d_out, T)          fp16   (output; row-major d_out x T)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .pack_utils import BCOL


# ---------------------------------------------------------------------------
# Unpack helper (evaluated at compile time per tile)
# ---------------------------------------------------------------------------

@triton.jit
def _unpack_packed_s4_rowmajor(packed, BM: tl.constexpr, BK: tl.constexpr):
    """Unpack (BM, BK//2) int8 tile into (BM, BK) SINT4 int8 tile.

    Layout: byte holds (high << 4) | (low & 0x0F); element order in the
    output is [low_0, high_0, low_1, high_1, ...].

    Implemented without `tl.interleave` (which is not present in all Triton
    versions) by broadcasting the packed tile and masking two planes.
    """
    BK_HALF: tl.constexpr = BK // 2
    # low (BM, BK_HALF) and high (BM, BK_HALF) int32 in [0, 15]
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Sign-extend to [-8, 7]
    low = tl.where(low >= 8, low - 16, low)
    high = tl.where(high >= 8, high - 16, high)
    # Stitch: out[:, 2*k]     = low[:, k]
    #         out[:, 2*k + 1] = high[:, k]
    # Triton has no native interleave in all versions; emulate via a stacked
    # reshape trick: build (BM, BK_HALF, 2) and then reshape to (BM, BK).
    # tl.join is the standard primitive for this since Triton 2.1.
    stacked = tl.join(low, high)           # (BM, BK_HALF, 2) int8
    out = tl.reshape(stacked, (BM, BK))    # (BM, BK) int8
    return out


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # --- decode regime (small N/T): BN must be tiny so we don't waste
        #     threads on N-padding.  M tiles stay medium/large for SM coverage.
        triton.Config({"BM": 64,  "BN": 16,  "BK": 128, "GROUP_SIZE_M": 1}, num_warps=2, num_stages=3),
        triton.Config({"BM": 128, "BN": 16,  "BK": 128, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 32,  "BK": 128, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=3),
        # --- mid regime (16 <= N <= 128): square-ish tiles benefit from L2
        #     swizzle.  GROUP_SIZE_M=8 roughly matches RTX 4090's 72 MiB L2.
        triton.Config({"BM": 64,  "BN": 64,  "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64,  "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        # --- prefill regime (N >= 256): large square tiles, deeper pipeline.
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
    ],
    key=["d_out", "d_in", "T"],
)
@triton.jit
def dense_gemm_kernel(
    W_low_ptr,          # (d_out, d_in // 2) int8 packed SINT4
    X_s4_ptr,           # (T,     d_in // 2) int8 packed SINT4
    scale_u4_ptr,       # (d_out, n_groups) fp16
    zero_u4_ptr,        # (d_out, n_groups) fp16
    sum_X_ptr,          # (T,     n_groups) int32
    scale_x_ptr,        # (T,)    fp16
    Y_low_ptr,          # (d_out, T) fp16
    d_out, d_in, T,
    stride_w_m, stride_w_k,
    stride_x_n, stride_x_k,
    stride_su_m, stride_su_g,
    stride_zu_m, stride_zu_g,
    stride_sx_n, stride_sx_g,
    stride_y_m, stride_y_n,
    N_GROUPS: tl.constexpr,
    BCOL_K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ------------------------------------------------------------------
    # Grouped program-ID swizzle for L2 locality.
    #
    # Without swizzle, program (pid_m, pid_n) visits memory in a raster
    # pattern.  A group of ``GROUP_SIZE_M`` adjacent M-blocks is reused
    # across all N-blocks -> we want those ``GROUP_SIZE_M * num_pid_n``
    # programs to be scheduled together so W-tiles stay hot in L2.
    # This is the standard Triton matmul tutorial swizzle, adapted to a
    # (pid_m, pid_n) 2-D launch grid rather than a 1-D flattening.
    # ------------------------------------------------------------------
    pid_m_raw = tl.program_id(0)
    pid_n_raw = tl.program_id(1)
    num_pid_m = tl.cdiv(d_out, BM)
    num_pid_n = tl.cdiv(T,     BN)

    # Flatten + re-index so every ``GROUP_SIZE_M`` consecutive M rows are
    # processed across the full N dimension before moving on.
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

    # scale_x[j] is constant across the whole K loop
    sx = tl.load(scale_x_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    # Final accumulator in FP32 for dequant output.
    y_acc = tl.zeros((BM, BN), dtype=tl.float32)

    BK_HALF: tl.constexpr = BK // 2
    # In this design BK == BCOL_K so each K iteration covers one group.
    tl.static_assert(BK == BCOL_K, "BK must equal BCOL_K (group size) for per-group epilogue")

    # half-byte column offsets: each int8 byte holds two int4 elements
    offs_k_half = tl.arange(0, BK_HALF)

    n_k_iters = tl.cdiv(d_in, BK)
    for k_block in range(0, n_k_iters):
        k_byte_start = k_block * BK_HALF
        k_bytes = k_byte_start + offs_k_half
        k_bytes_mask = k_bytes < (d_in // 2)

        # Load packed W tile (BM, BK_HALF)
        w_ptrs = W_low_ptr + offs_m[:, None] * stride_w_m + k_bytes[None, :] * stride_w_k
        w_packed = tl.load(
            w_ptrs,
            mask=mask_m[:, None] & k_bytes_mask[None, :],
            other=0,
        )
        # Load packed X tile (BN, BK_HALF)
        x_ptrs = X_s4_ptr + offs_n[:, None] * stride_x_n + k_bytes[None, :] * stride_x_k
        x_packed = tl.load(
            x_ptrs,
            mask=mask_n[:, None] & k_bytes_mask[None, :],
            other=0,
        )

        # Unpack to SINT4 int8 tiles (BM, BK) and (BN, BK)
        w_tile = _unpack_packed_s4_rowmajor(w_packed, BM, BK)   # (BM, BK) int8
        x_tile = _unpack_packed_s4_rowmajor(x_packed, BN, BK)   # (BN, BK) int8

        # tl.dot expects (M, K) x (K, N) -> (M, N). We have W as (BM, BK)
        # and X as (BN, BK); transpose X to (BK, BN) via tl.trans.
        x_tile_t = tl.trans(x_tile)                             # (BK, BN) int8
        acc_g = tl.dot(w_tile, x_tile_t, out_dtype=tl.int32)

        # Per-group dequant.  Each K iteration corresponds to group index k_block.
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

        # acc_g[m,n] := acc_g - zero_u4[m] * sum_X[n]
        corrected = acc_g.to(tl.float32) - zero_g[:, None] * sum_X_g[None, :]
        # Multiply by scale_u4[m] * scale_x[n]
        y_acc += corrected * scale_g[:, None] * sx[None, :]

    # Write back (d_out, T) in fp16
    y_ptrs = Y_low_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    tl.store(y_ptrs, y_acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def dense_gemm_u4_s4(
    W_low_packed: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    sum_X: torch.Tensor,
    scale_x: torch.Tensor,
) -> torch.Tensor:
    """Dense UINT4 x SINT4 GEMM (low-bit layer), returning FP16 Y_low (d_out, T)."""
    assert W_low_packed.is_cuda and X_s4.is_cuda
    assert W_low_packed.dtype == torch.int8 and X_s4.dtype == torch.int8
    d_out, d_in_half = W_low_packed.shape
    T, d_in_half_x = X_s4.shape
    assert d_in_half == d_in_half_x, "d_in mismatch between W and X"
    d_in = d_in_half * 2

    bcol = BCOL
    n_groups = d_in // bcol
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

    Y_low = torch.empty((d_out, T), dtype=torch.float16, device=W_low_packed.device)

    grid = lambda META: (triton.cdiv(d_out, META["BM"]), triton.cdiv(T, META["BN"]))
    dense_gemm_kernel[grid](
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, Y_low,
        d_out, d_in, T,
        W_low_packed.stride(0), W_low_packed.stride(1),
        X_s4.stride(0), X_s4.stride(1),
        scale_u4.stride(0), scale_u4.stride(1),
        zero_u4.stride(0), zero_u4.stride(1),
        sum_X.stride(0), sum_X.stride(1),
        Y_low.stride(0), Y_low.stride(1),
        N_GROUPS=n_groups,
        BCOL_K=bcol,
    )
    return Y_low


__all__ = ["dense_gemm_kernel", "dense_gemm_u4_s4"]
