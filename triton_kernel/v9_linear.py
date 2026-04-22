"""V9 end-to-end Linear forward wrapper.

Combines activation quantization + Kernel (1) + Kernel (2) into a single
Python entry point.  Also provides a pure-PyTorch FakeQuant reference for
correctness testing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .activation_quant import quantize_activation_s4
from .dense_u4s4_gemm import dense_gemm_u4_s4
from .pack_utils import BCOL, V9WeightContainer, unpack_s4_le
from .sparse_s4s4_gemm import sparse_gemm_s4_s4


# ---------------------------------------------------------------------------
# Fused combine + transpose kernel.
#
# Replaces the two separate traversals
#     Y_low.add_(Y_high, alpha=16.0)
#     Y_out = Y_low.transpose(0, 1).contiguous()
# which together touch the full (d_out, T) fp16 surface **twice**
# (one load+store for the add, one load+store for the contiguous copy),
# with a single pass that reads Y_low and Y_high row-by-row from the
# (d_out, T) layout and stores directly into the (T, d_out) output layout.
#
# Kernel grid: (cdiv(T, BT), cdiv(d_out, BD))   -- one program per output tile.
# Each program:
#     * loads a (BT, BD) tile from Y_low.T   (reading Y_low with stride (T,1))
#     * optionally loads the same tile from Y_high.T
#     * writes to Y_out[t0:t1, d0:d1] with stride (d_out, 1)   -- COALESCED
#
# This layout choice is critical: the coalesced writes are on the output
# dimension (d_out) which is contiguous in Y_out, so BD consecutive threads
# emit a single sector, while the (d_out, T) input reads are non-coalesced
# but cache-friendly because each warp reads BT rows of BD contiguous cols.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BT": 32,  "BD": 256}, num_warps=4),
        triton.Config({"BT": 64,  "BD": 128}, num_warps=4),
        triton.Config({"BT": 32,  "BD": 512}, num_warps=8),
        triton.Config({"BT": 64,  "BD": 256}, num_warps=8),
        triton.Config({"BT": 128, "BD": 128}, num_warps=8),
    ],
    key=["T", "d_out", "HAS_HIGH"],
)
@triton.jit
def _combine_transpose_kernel(
    Y_low_ptr,          # (d_out, T) fp16
    Y_high_ptr,         # (d_out, T) fp16 -- may alias Y_low if HAS_HIGH=False
    Y_out_ptr,          # (T, d_out) fp16
    T, d_out,
    stride_low_d, stride_low_t,
    stride_high_d, stride_high_t,
    stride_out_t, stride_out_d,
    BT: tl.constexpr, BD: tl.constexpr,
    HAS_HIGH: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_t = offs_t < T
    mask_d = offs_d < d_out

    # Load Y_low[d, t] -> tile shape (BD, BT).
    low_ptrs = Y_low_ptr + offs_d[:, None] * stride_low_d + offs_t[None, :] * stride_low_t
    low_val = tl.load(low_ptrs, mask=mask_d[:, None] & mask_t[None, :], other=0.0)

    if HAS_HIGH:
        high_ptrs = (
            Y_high_ptr
            + offs_d[:, None] * stride_high_d
            + offs_t[None, :] * stride_high_t
        )
        high_val = tl.load(
            high_ptrs, mask=mask_d[:, None] & mask_t[None, :], other=0.0
        )
        # fp16 add in fp32 to avoid subnormal rounding, then back to fp16.
        out_val = (low_val.to(tl.float32) + 16.0 * high_val.to(tl.float32)).to(tl.float16)
    else:
        out_val = low_val

    # Transpose on write: (BD, BT) tile -> Y_out[t, d] stride (d_out, 1)
    # so that BD consecutive threads along the last axis hit contiguous memory.
    out_tile = tl.trans(out_val)                          # (BT, BD)
    out_ptrs = (
        Y_out_ptr
        + offs_t[:, None] * stride_out_t
        + offs_d[None, :] * stride_out_d
    )
    tl.store(out_ptrs, out_tile, mask=mask_t[:, None] & mask_d[None, :])


def _combine_transpose(
    Y_low: torch.Tensor,
    Y_high: torch.Tensor | None,
    d_out: int,
    T: int,
) -> torch.Tensor:
    """Fused combine + transpose: returns (T, d_out) fp16.

    If ``Y_high`` is None, this degenerates to a pure transpose, replacing
    the previous ``Y_low.transpose(0,1).contiguous()`` call with a single
    pass that is slightly faster (no intermediate add) and keeps the kernel
    launch count identical.

    Small-T fast path
    -----------------
    The Triton kernel pays a fixed ~55-65us launch + autotune-dispatch
    overhead.  PyTorch's native ``.t().contiguous()`` is a highly tuned
    memcpy kernel that beats our fused kernel on surfaces below ~4M
    elements (8 MiB fp16).  Measured on RTX 4090 with HAS_HIGH=False:

        surf        torch    triton   winner
        262K elem   11.6us   62.0us   torch (5.3x)
        2M elem     27.2us   52.6us   torch (1.9x)
        8M elem     104us    62.1us   triton (1.7x)

    With HAS_HIGH=True the crossover is similar: torch's ``add_`` + native
    transpose sequence stays ahead of our fused kernel until ~4M elements.

    So we fall back to torch when ``T * d_out <= 4M``.
    """
    assert Y_low.is_cuda and Y_low.dtype == torch.float16
    # Threshold tuned empirically; see docstring for the microbench table.
    # Above 4M elements the fused Triton kernel amortises its launch cost
    # and wins by eliminating one full pass over the surface; below 4M the
    # launch overhead dominates.
    SMALL_SURFACE = 4 * 1024 * 1024  # elements (= 8 MiB fp16)
    if T * d_out <= SMALL_SURFACE:
        if Y_high is None:
            return Y_low.transpose(0, 1).contiguous()
        # Accumulate in-place into Y_low (saves one temp alloc), then transpose.
        # NB: Y_low is a fresh buffer returned by dense_gemm_u4_s4, so mutating
        # it is safe within v9_linear_forward.
        Y_low.add_(Y_high, alpha=16.0)
        return Y_low.transpose(0, 1).contiguous()

    Y_out = torch.empty((T, d_out), dtype=torch.float16, device=Y_low.device)
    if Y_high is None:
        y_high_ptr = Y_low          # harmless alias; kernel ignores it when HAS_HIGH=False
        stride_h_d, stride_h_t = Y_low.stride(0), Y_low.stride(1)
        has_high = False
    else:
        assert Y_high.shape == Y_low.shape and Y_high.dtype == torch.float16
        y_high_ptr = Y_high
        stride_h_d, stride_h_t = Y_high.stride(0), Y_high.stride(1)
        has_high = True
    grid = lambda META: (triton.cdiv(T, META["BT"]), triton.cdiv(d_out, META["BD"]))
    _combine_transpose_kernel[grid](
        Y_low, y_high_ptr, Y_out,
        T, d_out,
        Y_low.stride(0), Y_low.stride(1),
        stride_h_d, stride_h_t,
        Y_out.stride(0), Y_out.stride(1),
        HAS_HIGH=has_high,
    )
    return Y_out


def v9_linear_forward(X_fp16: torch.Tensor, W: V9WeightContainer) -> torch.Tensor:
    """V9 Linear forward.  Returns Y_fp16 with shape matching X on all-but-last dim.

    Internal pipeline:
      (1) per-token SINT4 activation quantization (fused kernel)
      (2) dense UINT4 x SINT4 GEMM   -> Y_low  (d_out, T)
      (3) block-sparse SINT4 x SINT4 GEMM -> Y_high (d_out, T)   [only if hp>0]
      (4) **fused**: Y_out[t, d] = Y_low[d, t] + 16 * Y_high[d, t]
          (single-pass combine + transpose, see ``_combine_transpose_kernel``)

    Rationale (vs. earlier implementation)
    --------------------------------------
    The former epilogue did two *independent* traversals of the whole
    ``(d_out, T)`` fp16 surface::

        Y_low.add_(Y_high, alpha=16.0)               # 1 load + 1 store
        Y_out = Y_low.transpose(0, 1).contiguous()   # 1 load + 1 store

    ``(d_out * T)`` fp16 is not small: at ``d_out = d_in = 4096, bs = 2048``
    that is 16 MiB touched **four times** end-to-end.  The new fused path
    keeps the dense kernel's output layout (critical for store coalescing
    in the inner GEMM -- a prior attempt to make dense write directly into
    a ``(T, d_out)`` view regressed bs=2048 shapes by ~120% because it
    spread consecutive N-tile stores across 2*d_out-byte strides), and
    performs *one* pass that reads ``Y_low`` (and optionally ``Y_high``),
    combines, and stores directly into the ``(T, d_out)`` final layout.

    Net effect: ~2x fewer bytes touched in stage 4, and when ``W.n_hp_blocks
    == 0`` the ``Y_high`` load is compiled out entirely via a
    ``HAS_HIGH`` ``constexpr`` switch.
    """
    assert X_fp16.is_cuda and X_fp16.dtype == torch.float16

    original_shape = X_fp16.shape
    d_in = W.d_in
    d_out = W.d_out
    if X_fp16.shape[-1] != d_in:
        raise ValueError(
            f"X last dim ({X_fp16.shape[-1]}) must match d_in ({d_in})"
        )

    X_2d = X_fp16.reshape(-1, d_in)
    T = X_2d.shape[0]

    # (1) Activation quantization
    X_s4, scale_x, sum_X = quantize_activation_s4(X_2d, W.perm, bcol=BCOL)

    # (2) Dense low-bit GEMM -- keep natural (d_out, T) output for coalesced
    #     stores inside the inner GEMM.
    Y_low = dense_gemm_u4_s4(
        W.W_low_packed, X_s4,
        W.scale_u4, W.zero_u4,
        sum_X, scale_x,
    )

    # (3) Sparse high-bit GEMM
    Y_high: torch.Tensor | None = None
    if W.n_hp_blocks > 0:
        Y_high = sparse_gemm_s4_s4(
            W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, scale_x,
            d_out=d_out, d_in=d_in,
        )

    # (4) Fused combine + transpose (single pass over the d_out x T surface).
    Y_out = _combine_transpose(Y_low, Y_high, d_out=d_out, T=T)

    out_shape = original_shape[:-1] + (d_out,)
    return Y_out.reshape(out_shape)


# ---------------------------------------------------------------------------
# Reference: fakequant Linear reconstructed from the packed container
# ---------------------------------------------------------------------------

def reconstruct_w_fakequant_fp16(W: V9WeightContainer) -> torch.Tensor:
    """Rebuild the fp16 dequantized weight (permuted column order) from a V9 pack.

    Useful for cross-checking kernel outputs.  Returns (d_out, d_in) fp16.
    """
    d_out, d_in = W.d_out, W.d_in
    device = W.scale_u4.device

    # Unpack low-bit SINT4 weights -> integer [-8, 7]
    w_low_s4 = unpack_s4_le(W.W_low_packed, signed=True).to(torch.float32)
    zero_fp = W.zero_u4.to(torch.float32)                # already pre-subtracted 8
    scale_fp = W.scale_u4.to(torch.float32)

    bcol = BCOL
    n_groups = d_in // bcol

    # Y_low_contrib per group: (w_low - zero) * scale
    # Broadcast over the bcol columns.
    scale_expand = scale_fp.repeat_interleave(bcol, dim=1)       # (d_out, d_in)
    zero_expand = zero_fp.repeat_interleave(bcol, dim=1)         # (d_out, d_in)
    w_fp_low = (w_low_s4 - zero_expand) * scale_expand

    # Add 16 * W_high contributions.
    w_fp_high = torch.zeros_like(w_fp_low)
    if W.n_hp_blocks > 0:
        w_high_s4 = unpack_s4_le(W.W_high_blocks_packed, signed=True).to(torch.float32)
        # Iterate blocks (Python loop is OK for reference path)
        hp_row_offsets = W.hp_row_offsets.cpu().tolist()
        hp_col_indices = W.hp_col_indices.cpu().tolist()
        nrow = (d_out + W.block_shape[0] - 1) // W.block_shape[0]
        for br in range(nrow):
            s, e = hp_row_offsets[br], hp_row_offsets[br + 1]
            for idx in range(s, e):
                bc = hp_col_indices[idx]
                r0, r1 = br * W.block_shape[0], min((br + 1) * W.block_shape[0], d_out)
                c0, c1 = bc * W.block_shape[1], min((bc + 1) * W.block_shape[1], d_in)
                tile = w_high_s4[idx, : r1 - r0, : c1 - c0]
                # Scale for this block is scale_u4[r0:r1, bc] (bc == group index).
                sc = scale_fp[r0:r1, bc: bc + 1]
                w_fp_high[r0:r1, c0:c1] += 16.0 * tile * sc

    w_fp = (w_fp_low + w_fp_high).to(torch.float16)
    return w_fp


def v9_linear_fakequant(X_fp16: torch.Tensor, W: V9WeightContainer) -> torch.Tensor:
    """Reference Linear forward using the dequantized fp16 weight.

    NB: this consumes the same V9 container so cross-checks are apples-to-apples.
    Reconstruction is expensive; for benchmarking against stock FP16 Linear,
    pass a pre-built `W_fakequant_fp16` instead.
    """
    d_in = W.d_in
    original_shape = X_fp16.shape
    X_2d = X_fp16.reshape(-1, d_in)

    # Permute input columns
    X_perm = X_2d[:, W.perm.to(torch.long)]

    # Reconstruct fakequant weight and quantize activation in fp16 for apples-to-apples.
    W_fp = reconstruct_w_fakequant_fp16(W)          # (d_out, d_in)

    # Quantize activation just like the kernel does (per-token symmetric SINT4),
    # so the reference reflects the same algorithm, not the FP16 upper bound.
    max_abs = X_perm.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale_x = (max_abs / 7.0).to(torch.float16).to(torch.float32)
    q = torch.clamp(torch.round(X_perm.to(torch.float32) / scale_x), -8.0, 7.0)
    X_dequant = (q * scale_x).to(torch.float16)

    Y_2d = X_dequant @ W_fp.t()
    out_shape = original_shape[:-1] + (W.d_out,)
    return Y_2d.reshape(out_shape)


__all__ = ["v9_linear_forward", "v9_linear_fakequant", "reconstruct_w_fakequant_fp16"]
