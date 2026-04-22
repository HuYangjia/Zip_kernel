"""Fused dequant kernel: V9 packed SINT4 weight -> FP16 row-major weight.

Purpose (Phase B-2)
-------------------
For the prefill regime, sweep analysis showed our online UINT4 x SINT4
GEMM runs at 1.27x of cuBLAS FP16 at median (HBM BW util 1.6-7%, i.e.
TC-occupancy limited). A practical escape hatch is a W4A16 fallback:
dequant the weight once on-device to FP16, then hand the GEMM to
cuBLAS. This only pays off when (dequant_ms + fp16_gemm_ms) <
int4_gemm_ms; for small T the dequant cost dominates, but for large
T (e.g. prefill bs >= 512) it can amortise.

The existing ``reconstruct_w_fakequant_fp16`` helper uses eager PyTorch
``repeat_interleave`` and takes ~2 ms for 4096x4096 (HBM roof would put
it at ~0.04 ms), so we need a dedicated Triton kernel.

What this kernel does
---------------------
Reads the packed SINT4 weight tile ``(d_out, d_in // 2) int8`` plus
per-(row, group) scale/zero in FP16, and writes the dequantised FP16
weight ``(d_out, d_in)``. The formula is the inverse of the online
GEMM's epilogue:

    w_fp16[m, k] = (w_s4[m, k] + 8 - (zero_u4_raw[m, g] - 8))
                   * scale_u4[m, g]
                 = (w_s4[m, k] + 8 - zero_u4_stored[m, g])
                   * scale_u4[m, g]

where g = k // BCOL, ``w_s4`` is the SINT4 value already unpacked from
the low-nibble pair (stored pre-subtracted-by-8 by ``pack_v9_weights``),
and ``zero_u4_stored`` is the stored value which already equals the
original UINT4 zero minus 8.

Constants match the online GEMM (BCOL=128, so each row has
``d_in // 128`` groups); we deliberately do NOT re-factor the algebra
because any deviation here would silently drift from the
``dense_gemm_u4_s4`` epilogue.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .pack_utils import BCOL


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BK": 512}, num_warps=8, num_stages=3),
        triton.Config({"BM": 32,  "BK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BM": 256, "BK": 256}, num_warps=8, num_stages=3),
    ],
    key=["d_out", "d_in"],
)
@triton.jit
def _dequant_u4_to_fp16_kernel(
    W_low_ptr,       # (d_out, d_in // 2) int8 (SINT4 nibble pairs, pre-offset by -8)
    scale_ptr,       # (d_out, n_groups) fp16
    zero_ptr,        # (d_out, n_groups) fp16  (stored value = zero_u4_raw - 8)
    W_out_ptr,       # (d_out, d_in) fp16
    d_out, d_in,
    stride_wlow_m, stride_wlow_k,
    stride_sc_m,   stride_sc_g,
    stride_zr_m,   stride_zr_g,
    stride_out_m,  stride_out_k,
    N_GROUPS: tl.constexpr,
    BCOL_K:   tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = pid_k * BK + tl.arange(0, BK)
    mask_m = offs_m < d_out
    mask_k = offs_k < d_in

    # BK must be a multiple of BCOL_K for the per-group broadcast to be clean.
    # We encode that assumption here rather than at the Python wrapper so a
    # bad BK config is caught at compile time rather than silently producing
    # wrong values.
    tl.static_assert(BK % BCOL_K == 0, "BK must be a multiple of BCOL_K")

    # -- load the packed SINT4 weight tile --------------------------------------
    BK_HALF: tl.constexpr = BK // 2
    offs_k_half = pid_k * BK_HALF + tl.arange(0, BK_HALF)
    mask_k_half = offs_k_half < (d_in // 2)

    w_ptrs = W_low_ptr + offs_m[:, None] * stride_wlow_m + offs_k_half[None, :] * stride_wlow_k
    w_packed = tl.load(
        w_ptrs, mask=mask_m[:, None] & mask_k_half[None, :], other=0,
    )

    # Unpack: low nibble (even k), high nibble (odd k). Values are already
    # in [-8, 7] (SINT4) thanks to the packing convention.
    low  = (w_packed << 4).to(tl.int8) >> 4           # arithmetic shift
    high = w_packed >> 4
    stacked = tl.join(low, high)                       # (BM, BK_HALF, 2)
    w_s4 = tl.reshape(stacked, (BM, BK))               # (BM, BK) int8

    # -- load per-(row, group) scale and zero ----------------------------------
    # Each K-block covers (BK // BCOL_K) groups, so we gather BK // BCOL_K
    # scale/zero columns per program.
    N_GROUPS_PER_BLOCK: tl.constexpr = BK // BCOL_K
    offs_g = pid_k * N_GROUPS_PER_BLOCK + tl.arange(0, N_GROUPS_PER_BLOCK)
    mask_g = offs_g < N_GROUPS

    sc_ptrs = scale_ptr + offs_m[:, None] * stride_sc_m + offs_g[None, :] * stride_sc_g
    zr_ptrs = zero_ptr  + offs_m[:, None] * stride_zr_m + offs_g[None, :] * stride_zr_g
    scale = tl.load(sc_ptrs, mask=mask_m[:, None] & mask_g[None, :], other=0.0)
    zero  = tl.load(zr_ptrs, mask=mask_m[:, None] & mask_g[None, :], other=0.0)

    # Broadcast scale/zero across the BCOL_K elements of each group.
    # w_s4 is (BM, BK). We treat it as (BM, N_GROUPS_PER_BLOCK, BCOL_K) and
    # scale/zero are (BM, N_GROUPS_PER_BLOCK) -> broadcast the last dim.
    w_s4_g = tl.reshape(w_s4, (BM, N_GROUPS_PER_BLOCK, BCOL_K))
    w_fp  = (w_s4_g.to(tl.float16) - zero[:, :, None]) * scale[:, :, None]
    w_out = tl.reshape(w_fp, (BM, BK))

    # -- store -----------------------------------------------------------------
    out_ptrs = W_out_ptr + offs_m[:, None] * stride_out_m + offs_k[None, :] * stride_out_k
    tl.store(out_ptrs, w_out, mask=mask_m[:, None] & mask_k[None, :])


def dequant_u4_to_fp16(W: "V9WeightContainer") -> torch.Tensor:
    """Dequantise V9 packed SINT4 weight to a dense FP16 matrix.

    Returns a ``(d_out, d_in)`` fp16 tensor suitable for cuBLAS FP16 GEMM.
    The SINT4 low-precision path is handled here; if the container carries
    sparse high-precision blocks (``n_hp_blocks > 0``) they are NOT added
    back -- callers that need full equivalence with the online V9 GEMM
    must add the sparse contribution separately.
    """
    from .pack_utils import V9WeightContainer  # local to avoid cycles
    assert isinstance(W, V9WeightContainer)
    d_out = int(W.d_out)
    d_in = int(W.d_in)
    n_groups = d_in // BCOL
    assert W.scale_u4.shape == (d_out, n_groups), (
        f"scale_u4 shape mismatch: {tuple(W.scale_u4.shape)} vs {(d_out, n_groups)}"
    )
    assert W.zero_u4.shape == (d_out, n_groups)

    W_out = torch.empty(d_out, d_in, dtype=torch.float16, device=W.W_low_packed.device)

    grid = lambda META: (
        triton.cdiv(d_out, META["BM"]),
        triton.cdiv(d_in,  META["BK"]),
    )
    _dequant_u4_to_fp16_kernel[grid](
        W.W_low_packed, W.scale_u4, W.zero_u4, W_out,
        d_out, d_in,
        W.W_low_packed.stride(0), W.W_low_packed.stride(1),
        W.scale_u4.stride(0),     W.scale_u4.stride(1),
        W.zero_u4.stride(0),      W.zero_u4.stride(1),
        W_out.stride(0),          W_out.stride(1),
        N_GROUPS=n_groups,
        BCOL_K=BCOL,
    )
    return W_out


__all__ = ["dequant_u4_to_fp16"]
