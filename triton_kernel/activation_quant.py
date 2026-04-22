"""Fused per-token SINT4 activation quantization Triton kernel.

Produces three outputs from a FP16 activation tensor in one kernel launch:
  - X_s4    : (batch*seq, d_in // 2) int8, 4-bit little-endian packed SINT4
  - scale_x : (batch*seq,)            fp16, per-token symmetric scale
  - sum_X   : (batch*seq, n_groups)   int32, per-group sum of SINT4 activations

See requirements.md section 2 and triton_kernel_prompt.md section 4.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as tl_libdevice

from .pack_utils import BCOL


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ----- Small-T regime (decode, T <= 16) ------------------------
        # One/two warps is enough; match BT to typical token counts to
        # avoid launching tons of partially-idle blocks.
        triton.Config({"BT": 16,  "BD": 256},  num_warps=2, num_stages=2),
        triton.Config({"BT": 16,  "BD": 512},  num_warps=2, num_stages=3),
        triton.Config({"BT": 32,  "BD": 256},  num_warps=2, num_stages=2),
        triton.Config({"BT": 32,  "BD": 512},  num_warps=4, num_stages=3),
        # ----- Medium regime (16 < T <= 128) ---------------------------
        # These BT values only make sense when we actually have enough
        # tokens to fill them; autotune will automatically avoid them
        # for small T because the empty-tile overhead dominates.
        triton.Config({"BT": 64,  "BD": 512},  num_warps=4, num_stages=2),
        triton.Config({"BT": 64,  "BD": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BT": 128, "BD": 512},  num_warps=4, num_stages=2),
        triton.Config({"BT": 128, "BD": 1024}, num_warps=8, num_stages=2),
        # ----- Large-T regime (T >= 256) -------------------------------
        # For the Llama-2 FFN shapes (d_in=11008) the kernel is
        # load-bandwidth bound, so we bias toward BD=2048 with one extra
        # pipeline stage -- this double-buffers the wide loads against
        # the divide+rint work of Pass 2.  We intentionally keep BT<=128
        # to guarantee enough program-level parallelism for SM occupancy
        # even when T is only ~32-64.
        triton.Config({"BT": 64,  "BD": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BT": 64,  "BD": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BT": 128, "BD": 2048}, num_warps=8, num_stages=3),
    ],
    key=["T", "D", "N_GROUPS"],
)
@triton.jit
def quantize_activation_kernel(
    X_ptr, perm_ptr,
    X_s4_ptr, scale_x_ptr, sum_X_ptr,
    T, D,                              # batch*seq, d_in
    stride_xt, stride_xd,              # X strides
    stride_qt, stride_qd,              # X_s4 strides (packed, last dim = D // 2)
    stride_st,                         # sum_X strides (T, n_groups)
    stride_sg,
    N_GROUPS: tl.constexpr,
    BCOL_K: tl.constexpr,              # group size along d_in
    BT: tl.constexpr,                  # tokens per program
    BD: tl.constexpr,                  # tile along d_in for streaming pass
):
    pid_t = tl.program_id(0)
    t_start = pid_t * BT
    offs_t = t_start + tl.arange(0, BT)
    mask_t = offs_t < T

    # ------------------------------------------------------------------
    # Pass 1: compute per-token max(|X|) in permuted order.
    # Streaming along d_in with tile size BD so we never cache the whole row.
    # ------------------------------------------------------------------
    max_abs = tl.zeros((BT,), dtype=tl.float32)
    for d_start in range(0, D, BD):
        offs_d = d_start + tl.arange(0, BD)
        mask_d = offs_d < D
        # Gather permuted column indices (int32)
        perm_idx = tl.load(perm_ptr + offs_d, mask=mask_d, other=0).to(tl.int32)
        # Load X[t, perm_idx]
        x_ptrs = X_ptr + offs_t[:, None] * stride_xt + perm_idx[None, :] * stride_xd
        x_tile = tl.load(
            x_ptrs,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        tile_max = tl.max(tl.abs(x_tile), axis=1)
        max_abs = tl.maximum(max_abs, tile_max)

    # scale_x = max / 7   (symmetric SINT4; clamp denom away from zero)
    # Important: both the stored scale and the value used for quantization
    # must go through the same fp16 rounding, otherwise the kernel output
    # differs bitwise from a reference that rounds scale to fp16 first.
    # Also: use x / scale (single rounding) rather than x * (1/scale) (two
    # roundings) to match numpy/torch's `torch.round(x / s)` reference.
    scale_fp32 = max_abs / 7.0
    scale_fp16 = scale_fp32.to(tl.float16)           # round-to-fp16
    scale = scale_fp16.to(tl.float32)                # back to fp32 for math
    # For zero rows: use 1.0 in the divide so we don't get NaN, but zero out
    # the result via a mask below.
    scale_safe = tl.where(scale > 0.0, scale, 1.0)
    scale_is_zero = scale <= 0.0

    tl.store(scale_x_ptr + offs_t, scale_fp16, mask=mask_t)

    # ------------------------------------------------------------------
    # Pass 2: quantize, pack (little-endian 4-bit), and accumulate sum_X
    # per group.  We walk d_in in tiles of BCOL_K (== group size) so the
    # per-group reduction is trivial.
    #
    # Implementation note: Triton 3.x forbids Python-style slicing on
    # constexpr dims (e.g. `q[:, :, 0]` or `q[:, 0::2]`).  To obtain the
    # even/odd columns needed for 4-bit packing we issue two separate loads
    # using strided offsets (2*i for low nibble, 2*i+1 for high nibble).
    # This yields two (BT, BCOL_K//2) tiles directly, so no reshape/slice
    # is required.
    # ------------------------------------------------------------------
    offs_h = tl.arange(0, BCOL_K // 2)
    for g in range(0, N_GROUPS):
        d_start = g * BCOL_K
        # Even (low-nibble) column indices within this group: 2*h
        offs_d_lo = d_start + 2 * offs_h
        # Odd  (high-nibble) column indices within this group: 2*h + 1
        offs_d_hi = d_start + 2 * offs_h + 1
        mask_d_lo = offs_d_lo < D
        mask_d_hi = offs_d_hi < D

        # Gather permuted column indices separately for even/odd cols.
        perm_lo = tl.load(perm_ptr + offs_d_lo, mask=mask_d_lo, other=0).to(tl.int32)
        perm_hi = tl.load(perm_ptr + offs_d_hi, mask=mask_d_hi, other=0).to(tl.int32)

        # Load FP16 activations for even / odd columns.
        x_lo = tl.load(
            X_ptr + offs_t[:, None] * stride_xt + perm_lo[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d_lo[None, :],
            other=0.0,
        ).to(tl.float32)
        x_hi = tl.load(
            X_ptr + offs_t[:, None] * stride_xt + perm_hi[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d_hi[None, :],
            other=0.0,
        ).to(tl.float32)

        # q = clamp(round(x / scale), -8, 7)  -- inlined for lo/hi.
        # Use division (single fp32 rounding) to match torch.round(x / s),
        # and libdevice.rint for IEEE round-half-to-even (matches torch.round).
        q_lo = x_lo / scale_safe[:, None]
        q_lo = tl_libdevice.rint(q_lo)
        q_lo = tl.minimum(tl.maximum(q_lo, -8.0), 7.0)
        q_lo_i32 = q_lo.to(tl.int32)
        # Zero rows must produce q=0, not garbage from dividing by the 1.0 fallback.
        q_lo_i32 = tl.where(scale_is_zero[:, None], 0, q_lo_i32)
        q_lo_i32 = tl.where(mask_d_lo[None, :], q_lo_i32, 0)

        q_hi = x_hi / scale_safe[:, None]
        q_hi = tl_libdevice.rint(q_hi)
        q_hi = tl.minimum(tl.maximum(q_hi, -8.0), 7.0)
        q_hi_i32 = q_hi.to(tl.int32)
        q_hi_i32 = tl.where(scale_is_zero[:, None], 0, q_hi_i32)
        q_hi_i32 = tl.where(mask_d_hi[None, :], q_hi_i32, 0)

        # sum_X[t, g] = sum_k q_i32[t, k] over the full group (low+high).
        g_sum = tl.sum(q_lo_i32, axis=1) + tl.sum(q_hi_i32, axis=1)
        tl.store(sum_X_ptr + offs_t * stride_st + g * stride_sg, g_sum, mask=mask_t)

        # Pack two consecutive SINT4 values into one int8 byte (little-endian).
        # After q & 0x0F we have the 4-bit two's-complement pattern.
        low = q_lo_i32 & 0x0F
        high = q_hi_i32 & 0x0F
        packed = ((high << 4) | low) & 0xFF
        # Convert 0..255 -> signed int8
        packed_i8 = tl.where(packed >= 128, packed - 256, packed).to(tl.int8)

        # Store packed bytes.  X_s4 layout: (T, D // 2).  Column offset is
        # (d_start // 2) + [0 .. BCOL_K/2).
        byte_offs = (d_start // 2) + offs_h
        byte_mask = byte_offs < (D // 2)
        qs_ptrs = X_s4_ptr + offs_t[:, None] * stride_qt + byte_offs[None, :] * stride_qd
        tl.store(
            qs_ptrs,
            packed_i8,
            mask=mask_t[:, None] & byte_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def quantize_activation_s4(
    X_fp16: torch.Tensor,
    perm: torch.Tensor,
    bcol: int = BCOL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused activation quantization wrapper.

    BT / BD / num_warps are picked by Triton autotune (keyed on (T, D, N_GROUPS)),
    so the first call at a new shape pays a short auto-tuning cost and then
    caches the best config.

    Args:
        X_fp16: (batch, seq_len, d_in) or (T, d_in) fp16 tensor.
        perm:   (d_in,) int32 permutation (act-order).
        bcol:   group size (default 128).

    Returns:
        X_s4    : (T, d_in // 2) int8 packed SINT4
        scale_x : (T,) fp16
        sum_X   : (T, n_groups) int32
    """
    assert X_fp16.is_cuda, "quantize_activation_s4 requires a CUDA tensor"
    assert X_fp16.dtype == torch.float16, "X must be fp16"
    assert perm.dtype in (torch.int32, torch.int64), "perm must be int"

    original_shape = X_fp16.shape
    if X_fp16.dim() == 3:
        T = original_shape[0] * original_shape[1]
        D = original_shape[2]
    elif X_fp16.dim() == 2:
        T, D = original_shape
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {original_shape}")

    if D % bcol != 0:
        raise ValueError(f"d_in ({D}) must be divisible by bcol ({bcol})")
    if D % 2 != 0:
        raise ValueError(f"d_in ({D}) must be even for 4-bit packing")

    X_2d = X_fp16.reshape(T, D).contiguous()
    perm = perm.to(torch.int32).contiguous()

    n_groups = D // bcol
    device = X_2d.device

    X_s4 = torch.empty((T, D // 2), dtype=torch.int8, device=device)
    scale_x = torch.empty((T,), dtype=torch.float16, device=device)
    sum_X = torch.empty((T, n_groups), dtype=torch.int32, device=device)

    # autotune picks BT/BD/num_warps; grid depends on BT so pass a callable.
    grid = lambda META: (triton.cdiv(T, META["BT"]),)
    quantize_activation_kernel[grid](
        X_2d, perm,
        X_s4, scale_x, sum_X,
        T, D,
        X_2d.stride(0), X_2d.stride(1),
        X_s4.stride(0), X_s4.stride(1),
        sum_X.stride(0), sum_X.stride(1),
        N_GROUPS=n_groups,
        BCOL_K=bcol,
    )

    return X_s4, scale_x, sum_X


__all__ = ["quantize_activation_kernel", "quantize_activation_s4"]
