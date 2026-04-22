"""V9 end-to-end Linear forward wrapper.

Combines activation quantization + Kernel (1) + Kernel (2) into a single
Python entry point.  Also provides a pure-PyTorch FakeQuant reference for
correctness testing.
"""

from __future__ import annotations

import torch

from .activation_quant import quantize_activation_s4
from .dense_u4s4_gemm import dense_gemm_u4_s4
from .pack_utils import BCOL, V9WeightContainer, unpack_s4_le
from .sparse_s4s4_gemm import sparse_gemm_s4_s4


def v9_linear_forward(X_fp16: torch.Tensor, W: V9WeightContainer) -> torch.Tensor:
    """V9 Linear forward.  Returns Y_fp16 with shape matching X on all-but-last dim.

    Internal pipeline:
      (1) per-token SINT4 activation quantization (fused kernel)
      (2) dense UINT4 x SINT4 GEMM  -> Y_low (d_out, T)
      (3) block-sparse SINT4 x SINT4 GEMM -> Y_high (d_out, T)
      (4) Y = Y_low + 16 * Y_high
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

    # (2) Dense low-bit GEMM
    Y_low = dense_gemm_u4_s4(
        W.W_low_packed, X_s4,
        W.scale_u4, W.zero_u4,
        sum_X, scale_x,
    )

    # (3) Sparse high-bit GEMM (skip if no hp blocks)
    if W.n_hp_blocks > 0:
        Y_high = sparse_gemm_s4_s4(
            W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, scale_x,
            d_out=d_out, d_in=d_in,
        )
        # (4a) In-place combine: Y_low <- Y_low + 16 * Y_high.
        # Avoids the temp (d_out, T) fp16 tensor a naive `Y_low + 16*Y_high`
        # would allocate, saving ~one full output read+write per call.
        Y_low.add_(Y_high, alpha=16.0)
    Y = Y_low

    # (4b) Y is (d_out, T) ; turn back to (..., d_out) with the original leading dims.
    Y_out = Y.transpose(0, 1).contiguous()              # (T, d_out)
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
