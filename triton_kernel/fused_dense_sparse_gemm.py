"""Fused Dense + Sparse (UINT4 x SINT4 + SINT4 x SINT4) GEMM for V9.

Replaces the pair (dense_u4s4_gemm, sparse_s4s4_gemm) when both paths
are needed (``hp_ratio > 0``).  One Triton program computes a single
``(BM, BN)`` output tile of::

    Y_total[m, n] = Y_low[m, n] + 16 * Y_high[m, n]

in **FP32 accumulators**, then stores the tile to a single ``(d_out, T)``
output buffer.  This saves, versus running the two kernels back-to-back:

- one kernel launch (~5-10us)
- one full ``(d_out, T)`` FP16 store (=d_out*T*2 bytes of HBM write)
- one full ``(d_out, T)`` FP16 read in the downstream combine+transpose
  pass (because we now feed a single tensor instead of Y_low + Y_high)

Structurally:

- Dense branch: unchanged K-loop over all ``n_k_iters`` groups; per-group
  dequant with scale_u4[m, g], zero_u4[m, g], sum_X[n, g], scale_x[n].
- Sparse branch: a second short loop over the BSR high-precision blocks
  for this block-row.  Each block contributes to
  ``y_acc += 16 * scale_u4[m, bc] * scale_x[n] * dot(W_high, X_s4_bc)``.

Shape conventions (same as the two source kernels):
- W_low_packed         : (d_out, d_in // 2)              int8
- W_high_blocks_packed : (n_hp_blocks, BROW, BCOL // 2)  int8
- hp_row_offsets       : (nrow + 1,)                     int32
- hp_col_indices       : (n_hp_blocks,)                  int32
- X_s4                 : (T, d_in // 2)                  int8
- scale_u4             : (d_out, n_groups)               fp16
- zero_u4              : (d_out, n_groups)               fp16   (pre-subtracted 8)
- sum_X                : (T, n_groups)                   int32
- scale_x              : (T,)                            fp16
- Y_total              : (d_out, T)                      fp16   <== single output
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .pack_utils import BROW, BCOL


# ---------------------------------------------------------------------------
# Shared unpack helper (same as dense / sparse kernels).
# ---------------------------------------------------------------------------

@triton.jit
def _unpack_packed_s4_rowmajor(packed, BM: tl.constexpr, BK: tl.constexpr):
    BK_HALF: tl.constexpr = BK // 2
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    low = tl.where(low >= 8, low - 16, low)
    high = tl.where(high >= 8, high - 16, high)
    stacked = tl.join(low, high)
    return tl.reshape(stacked, (BM, BK))


# ---------------------------------------------------------------------------
# Fused kernel
# ---------------------------------------------------------------------------
#
# Tile layout invariants (must hold statically):
#   - BM <= BROW_K and BROW_K % BM == 0, OR BM >= BROW_K and BM % BROW_K == 0
#     so each (BM) block lives in exactly one BSR row (hence ``br`` is
#     unique for the whole tile).
#   - BK == BCOL_K so each K iteration corresponds to one group g.
#
# We only carry configs that respect the first invariant (BM in {64, 128,
# 256}); with BROW=BCOL=128 these are all legal.
#
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # --- mid/prefill regime: BSR is on 128-row blocks, so BM in {64, 128, 256}
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 256, "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 256, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        # --- small/decode regime: narrow BN, still 64-row BM (two per BSR row)
        triton.Config({"BM": 64,  "BN": 64,  "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64,  "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 32,  "BK": 128, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 16,  "BK": 128, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 16,  "BK": 128, "GROUP_SIZE_M": 1}, num_warps=2, num_stages=3),
    ],
    key=["d_out", "d_in", "T"],
)
@triton.jit
def fused_dense_sparse_kernel(
    # Dense branch inputs
    W_low_ptr,              # (d_out, d_in // 2) int8
    X_s4_ptr,               # (T,     d_in // 2) int8
    scale_u4_ptr,           # (d_out, n_groups)  fp16  (shared!)
    zero_u4_ptr,            # (d_out, n_groups)  fp16
    sum_X_ptr,              # (T,     n_groups)  int32
    scale_x_ptr,            # (T,)               fp16
    # Sparse branch inputs
    W_high_blocks_ptr,      # (n_hp, BROW, BCOL // 2) int8
    hp_row_offsets_ptr,     # (nrow + 1,) int32
    hp_col_indices_ptr,     # (n_hp,)     int32
    # Output
    Y_total_ptr,            # (d_out, T) fp16
    # Shape
    d_out, d_in, T,
    # Strides
    stride_w_m, stride_w_k,
    stride_x_n, stride_x_k,
    stride_su_m, stride_su_g,
    stride_zu_m, stride_zu_g,
    stride_sx_n, stride_sx_g,
    stride_wb_blk, stride_wb_r, stride_wb_k,
    stride_y_m, stride_y_n,
    # Meta
    N_GROUPS: tl.constexpr,
    BROW_K: tl.constexpr,
    BCOL_K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ------------------------------------------------------------------
    # Grouped PID swizzle for L2 locality (same as dense kernel).
    # ------------------------------------------------------------------
    pid_m_raw = tl.program_id(0)
    pid_n_raw = tl.program_id(1)
    num_pid_m = tl.cdiv(d_out, BM)
    num_pid_n = tl.cdiv(T,     BN)
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

    # scale_x[n] is constant across the whole K loop -> hoist.
    sx = tl.load(scale_x_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    # FP32 accumulator for Y_total = Y_low + 16 * Y_high.
    y_acc = tl.zeros((BM, BN), dtype=tl.float32)

    BK_HALF: tl.constexpr = BK // 2
    tl.static_assert(BK == BCOL_K,
                     "BK must equal BCOL_K (group size) for per-group epilogue")
    # Sparse branch needs BM to fit inside exactly one BSR row.
    tl.static_assert(BROW_K % BM == 0 or BM % BROW_K == 0,
                     "BM and BROW_K must be commensurable")

    offs_k_half = tl.arange(0, BK_HALF)

    # ==================================================================
    # DENSE branch : full K sweep over the entire d_in.
    # ==================================================================
    n_k_iters = tl.cdiv(d_in, BK)
    for k_block in range(0, n_k_iters):
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
        w_tile = _unpack_packed_s4_rowmajor(w_packed, BM, BK)   # (BM, BK)
        x_tile = _unpack_packed_s4_rowmajor(x_packed, BN, BK)   # (BN, BK)
        x_tile_t = tl.trans(x_tile)                             # (BK, BN)
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
        y_acc += corrected * scale_g[:, None] * sx[None, :]

    # ==================================================================
    # SPARSE branch : short K-loop over BSR blocks of this block-row,
    # contributes 16 * scale_u4[m, bc] * scale_x[n] * dot(W_high, X_s4_bc).
    # Falls through with zero contribution if the block-row is empty.
    # ==================================================================
    br = (pid_m * BM) // BROW_K
    row_in_blk = offs_m - br * BROW_K
    mask_row_in_blk = (row_in_blk >= 0) & (row_in_blk < BROW_K)

    start = tl.load(hp_row_offsets_ptr + br)
    end = tl.load(hp_row_offsets_ptr + br + 1)

    for block_idx in range(start, end):
        bc = tl.load(hp_col_indices_ptr + block_idx)

        # Load W_high_blocks[block_idx, row_in_blk, :] for BM rows.
        wb_ptrs = (
            W_high_blocks_ptr
            + block_idx * stride_wb_blk
            + row_in_blk[:, None] * stride_wb_r
            + offs_k_half[None, :] * stride_wb_k
        )
        wb_packed = tl.load(
            wb_ptrs,
            mask=mask_m[:, None] & mask_row_in_blk[:, None],
            other=0,
        )

        # Load X_s4 slice for group bc (runtime index, can't share with
        # the dense path's load because bc is data-dependent).
        k_byte_start2 = bc * BK_HALF
        k_bytes2 = k_byte_start2 + offs_k_half
        k_bytes_mask2 = k_bytes2 < (d_in // 2)
        xb_ptrs = X_s4_ptr + offs_n[:, None] * stride_x_n + k_bytes2[None, :] * stride_x_k
        xb_packed = tl.load(
            xb_ptrs,
            mask=mask_n[:, None] & k_bytes_mask2[None, :],
            other=0,
        )

        wb_tile = _unpack_packed_s4_rowmajor(wb_packed, BM, BCOL_K)   # (BM, BCOL_K)
        xb_tile = _unpack_packed_s4_rowmajor(xb_packed, BN, BCOL_K)   # (BN, BCOL_K)
        xb_tile_t = tl.trans(xb_tile)                                 # (BCOL_K, BN)

        acc_block = tl.dot(wb_tile, xb_tile_t, out_dtype=tl.int32)

        scale_bc = tl.load(
            scale_u4_ptr + offs_m * stride_su_m + bc * stride_su_g,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        # NOTE: factor of 16 (Y_high contribution weight), matches
        #       v9_linear combine: Y = Y_low + 16 * Y_high.
        y_acc += 16.0 * acc_block.to(tl.float32) * scale_bc[:, None] * sx[None, :]

    # ==================================================================
    # Store Y_total (d_out, T) fp16
    # ==================================================================
    y_ptrs = Y_total_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    tl.store(y_ptrs, y_acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def fused_dense_sparse_gemm(
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
    """Fused dense + sparse GEMM, returns ``Y_total`` fp16 of shape ``(d_out, T)``.

    Semantics:
        Y_total[m, n] = Y_low[m, n] + 16 * Y_high[m, n]

    where Y_low is the dense UINT4xSINT4 result and Y_high is the BSR
    SINT4xSINT4 result.
    """
    assert W_low_packed.is_cuda and X_s4.is_cuda
    assert W_low_packed.dtype == torch.int8 and X_s4.dtype == torch.int8
    T, d_in_half_x = X_s4.shape
    d_in_half = d_in // 2
    assert d_in_half_x == d_in_half, "d_in mismatch between X and declared d_in"

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

    Y_total = torch.empty((d_out, T), dtype=torch.float16, device=W_low_packed.device)

    # When there are no high-precision blocks we can still call the fused
    # kernel -- the sparse branch will iterate 0 times and we get exactly
    # the dense result.  In that case the caller would usually prefer to
    # use ``dense_gemm_u4_s4`` directly (avoids the per-tile BSR bookkeeping)
    # but we accept the degenerate input for completeness.
    if W_high_blocks_packed.numel() == 0:
        # Build a zero-length but strided tensor so we can still pass strides.
        W_high_blocks_packed = torch.zeros(
            (0, BROW, bcol // 2), dtype=torch.int8, device=W_low_packed.device
        )
    W_high_blocks_packed = W_high_blocks_packed.contiguous()
    hp_row_offsets = hp_row_offsets.contiguous().to(torch.int32)
    hp_col_indices = hp_col_indices.contiguous().to(torch.int32)

    grid = lambda META: (
        triton.cdiv(d_out, META["BM"]),
        triton.cdiv(T, META["BN"]),
    )
    fused_dense_sparse_kernel[grid](
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x,
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        Y_total,
        d_out, d_in, T,
        W_low_packed.stride(0), W_low_packed.stride(1),
        X_s4.stride(0), X_s4.stride(1),
        scale_u4.stride(0), scale_u4.stride(1),
        zero_u4.stride(0), zero_u4.stride(1),
        sum_X.stride(0), sum_X.stride(1),
        W_high_blocks_packed.stride(0),
        W_high_blocks_packed.stride(1),
        W_high_blocks_packed.stride(2),
        Y_total.stride(0), Y_total.stride(1),
        N_GROUPS=n_groups,
        BROW_K=BROW,
        BCOL_K=bcol,
    )
    return Y_total


__all__ = ["fused_dense_sparse_kernel", "fused_dense_sparse_gemm"]
