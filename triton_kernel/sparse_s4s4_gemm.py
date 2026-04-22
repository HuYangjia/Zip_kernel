"""Kernel (2) - 2D block-sparse SINT4 x SINT4 GEMM for the V9 high-bit layer.

Output-row-tile scheduling strategy (see triton_kernel_prompt.md sec 3):
each Triton program owns one output tile Y_high[r0:r1, j0:j1] and walks a
short K-loop over the high-precision blocks belonging to block row `br`.
No scatter, no atomics.

Shape conventions
-----------------
- W_high_blocks_packed : (n_hp_blocks, brow, bcol // 2) int8 SINT4 packed
- hp_row_offsets       : (nrow + 1,) int32
- hp_col_indices       : (n_hp_blocks,) int32
- X_s4                 : (T, d_in // 2) int8 packed SINT4
- scale_u4             : (d_out, n_groups) fp16
- scale_x              : (T,) fp16
- Y_high               : (d_out, T) fp16
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .pack_utils import BROW, BCOL


@triton.jit
def _unpack_packed_s4_rowmajor(packed, BM: tl.constexpr, BK: tl.constexpr):
    """Unpack a (BM, BK // 2) int8 tile of packed SINT4 values into (BM, BK).

    See `dense_u4s4_gemm._unpack_packed_s4_rowmajor` for details.
    """
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    low = tl.where(low >= 8, low - 16, low)
    high = tl.where(high >= 8, high - 16, high)
    stacked = tl.join(low, high)
    return tl.reshape(stacked, (BM, BK))


@triton.autotune(
    configs=[
        triton.Config({"BM": 128, "BN": 128}, num_warps=4, num_stages=2),
        triton.Config({"BM": 128, "BN": 128}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128}, num_warps=4, num_stages=2),
    ],
    key=["d_out", "d_in", "T"],
)
@triton.jit
def sparse_gemm_kernel(
    W_high_blocks_ptr,      # (n_hp, brow, bcol // 2) int8
    hp_row_offsets_ptr,     # (nrow + 1,) int32
    hp_col_indices_ptr,     # (n_hp,) int32
    X_s4_ptr,               # (T, d_in // 2) int8
    scale_u4_ptr,           # (d_out, n_groups) fp16
    scale_x_ptr,            # (T,) fp16
    Y_high_ptr,             # (d_out, T) fp16
    d_out, d_in, T,
    stride_wb_blk, stride_wb_r, stride_wb_k,
    stride_x_n, stride_x_k,
    stride_su_m, stride_su_g,
    stride_y_m, stride_y_n,
    BROW_K: tl.constexpr, BCOL_K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < d_out
    mask_n = offs_n < T

    # Block row this program belongs to.
    # We require BM <= BROW_K (typically BM == BROW == 128).
    br = (pid_m * BM) // BROW_K
    # Row offset within this block row
    row_in_blk = offs_m - br * BROW_K
    mask_row_in_blk = (row_in_blk >= 0) & (row_in_blk < BROW_K)

    start = tl.load(hp_row_offsets_ptr + br)
    end = tl.load(hp_row_offsets_ptr + br + 1)

    # Static check: BM must evenly divide BROW_K so that row_in_blk is in
    # [0, BROW_K) for every program.  autotune only supplies 64 or 128 here.
    tl.static_assert(BROW_K % BM == 0 or BM % BROW_K == 0,
                     "BM and BROW_K must be commensurable")

    sx = tl.load(scale_x_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    y_acc = tl.zeros((BM, BN), dtype=tl.float32)

    BK_HALF: tl.constexpr = BCOL_K // 2
    offs_k_half = tl.arange(0, BK_HALF)

    # K-loop over the (potentially empty) list of high-precision blocks
    # belonging to this block-row.  If start == end the loop executes 0 times
    # and we fall straight through to the final tl.store (writes zeros).
    for block_idx in range(start, end):
        bc = tl.load(hp_col_indices_ptr + block_idx)

        # Load W_high_blocks[block_idx, row_in_blk, :] for the BM rows in this tile.
        # Only rows that actually belong to this block row are meaningful.
        w_ptrs = (
            W_high_blocks_ptr
            + block_idx * stride_wb_blk
            + row_in_blk[:, None] * stride_wb_r
            + offs_k_half[None, :] * stride_wb_k
        )
        w_packed = tl.load(
            w_ptrs,
            mask=mask_m[:, None] & mask_row_in_blk[:, None],
            other=0,
        )

        # Load X_s4 slice for this bc column group.
        k_byte_start = bc * BK_HALF
        k_bytes = k_byte_start + offs_k_half
        k_bytes_mask = k_bytes < (d_in // 2)
        x_ptrs = X_s4_ptr + offs_n[:, None] * stride_x_n + k_bytes[None, :] * stride_x_k
        x_packed = tl.load(
            x_ptrs,
            mask=mask_n[:, None] & k_bytes_mask[None, :],
            other=0,
        )

        w_tile = _unpack_packed_s4_rowmajor(w_packed, BM, BCOL_K)   # (BM, BCOL_K) int8
        x_tile = _unpack_packed_s4_rowmajor(x_packed, BN, BCOL_K)   # (BN, BCOL_K) int8
        x_tile_t = tl.trans(x_tile)                                 # (BCOL_K, BN)

        acc_block = tl.dot(w_tile, x_tile_t, out_dtype=tl.int32)

        # Per-block dequant: scale_u4[i, bc] * scale_x[j] * acc_block.
        scale_bc = tl.load(
            scale_u4_ptr + offs_m * stride_su_m + bc * stride_su_g,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        y_acc += acc_block.to(tl.float32) * scale_bc[:, None] * sx[None, :]

    # Write back (zero if the block row has no high-precision blocks).
    y_ptrs = Y_high_ptr + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
    tl.store(y_ptrs, y_acc.to(tl.float16), mask=mask_m[:, None] & mask_n[None, :])


def sparse_gemm_s4_s4(
    W_high_blocks_packed: torch.Tensor,
    hp_row_offsets: torch.Tensor,
    hp_col_indices: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    scale_x: torch.Tensor,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Block-sparse SINT4 x SINT4 GEMM, returns Y_high (d_out, T) fp16."""
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

    n_groups = scale_u4.shape[1]

    grid = lambda META: (triton.cdiv(d_out, META["BM"]), triton.cdiv(T, META["BN"]))
    sparse_gemm_kernel[grid](
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, scale_x, Y_high,
        d_out, d_in, T,
        W_high_blocks_packed.stride(0),
        W_high_blocks_packed.stride(1),
        W_high_blocks_packed.stride(2),
        X_s4.stride(0), X_s4.stride(1),
        scale_u4.stride(0), scale_u4.stride(1),
        Y_high.stride(0), Y_high.stride(1),
        BROW_K=BROW,
        BCOL_K=BCOL,
    )
    return Y_high


__all__ = ["sparse_gemm_kernel", "sparse_gemm_s4_s4"]
