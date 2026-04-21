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

from .pack_utils import BCOL


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

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
    scale = max_abs / 7.0
    scale_safe = tl.where(scale > 0.0, scale, 1.0)
    inv_scale = 1.0 / scale_safe
    # zero rows stay zero
    inv_scale = tl.where(scale > 0.0, inv_scale, 0.0)

    tl.store(scale_x_ptr + offs_t, scale.to(tl.float16), mask=mask_t)

    # ------------------------------------------------------------------
    # Pass 2: quantize, pack (little-endian 4-bit), and accumulate sum_X
    # per group.  We walk d_in in tiles of BCOL_K (== group size) so the
    # per-group reduction is trivial.
    # ------------------------------------------------------------------
    for g in range(0, N_GROUPS):
        d_start = g * BCOL_K
        offs_d = d_start + tl.arange(0, BCOL_K)
        mask_d = offs_d < D
        perm_idx = tl.load(perm_ptr + offs_d, mask=mask_d, other=0).to(tl.int32)
        x_ptrs = X_ptr + offs_t[:, None] * stride_xt + perm_idx[None, :] * stride_xd
        x_tile = tl.load(
            x_ptrs,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # q = clamp(round(x * inv_scale), -8, 7)
        # Explicit round-half-to-nearest-even via floor(x + 0.5*sign).  Avoids
        # relying on tl.extra.cuda.libdevice.rint which moved paths across
        # Triton versions (libdevice.rint in 2.2, tl.math.round in 3.x).
        q = x_tile * inv_scale[:, None]
        q = tl.where(q >= 0.0, tl.math.floor(q + 0.5), tl.math.ceil(q - 0.5))
        q = tl.minimum(tl.maximum(q, -8.0), 7.0)
        q_i32 = q.to(tl.int32)
        # Apply the mask (OOB columns -> 0 so they don't affect sum_X).
        q_i32 = tl.where(mask_d[None, :], q_i32, 0)

        # sum_X[t, g] = sum_k q_i32[t, k]
        g_sum = tl.sum(q_i32, axis=1)
        tl.store(sum_X_ptr + offs_t * stride_st + g * stride_sg, g_sum, mask=mask_t)

        # Pack two consecutive SINT4 values into one int8 byte (little-endian).
        # After q_i32 & 0x0F we have the 4-bit two's-complement pattern.
        q_bits = q_i32 & 0x0F
        # Split along the last dim into even/odd pairs.
        # BCOL_K is a constexpr power-of-two so reshape is legal.
        q_reshaped = tl.reshape(q_bits, (BT, BCOL_K // 2, 2))
        low = q_reshaped[:, :, 0]
        high = q_reshaped[:, :, 1]
        packed = ((high << 4) | low) & 0xFF
        # Convert 0..255 -> signed int8
        packed_i8 = tl.where(packed >= 128, packed - 256, packed).to(tl.int8)

        # Store packed bytes.  X_s4 layout: (T, D // 2).  Column offset is
        # (d_start // 2) + [0 .. BCOL_K/2).
        byte_offs = (d_start // 2) + tl.arange(0, BCOL_K // 2)
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
    BT: int = 32,
    BD: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused activation quantization wrapper.

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

    grid = (triton.cdiv(T, BT),)
    quantize_activation_kernel[grid](
        X_2d, perm,
        X_s4, scale_x, sum_X,
        T, D,
        X_2d.stride(0), X_2d.stride(1),
        X_s4.stride(0), X_s4.stride(1),
        sum_X.stride(0), sum_X.stride(1),
        N_GROUPS=n_groups,
        BCOL_K=bcol,
        BT=BT,
        BD=BD,
        num_warps=4,
    )

    return X_s4, scale_x, sum_X


__all__ = ["quantize_activation_kernel", "quantize_activation_s4"]
