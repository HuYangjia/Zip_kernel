"""Pure-torch mirror of :mod:`kernel.triton_kernel.activation_quant`.

The production activation-quant kernel is Triton (GPU-only).  Parity
tests that want to run on CPU dev hosts need an equivalent pure-torch
implementation with *bitwise-compatible* rounding semantics.

Contract (mirrored from activation_quant.py lines ~230-280)
-----------------------------------------------------------
Given FP16 activation ``X`` of shape ``(T, d_in)`` and permutation
``perm`` of shape ``(d_in,)``:

    X_perm[t, d]  = X[t, perm[d]]
    scale_fp32[t] = max(|X_perm[t, :]|) / 7.0
    scale_fp16[t] = scale_fp32[t].to(fp16)                 # single fp16 round
    scale_x[t]    = scale_fp16[t]                          # stored value
    s[t]          = scale_fp16[t].to(fp32)                 # used for quant
    s_safe[t]     = s[t] if s[t] > 0 else 1.0
    q[t, d]       = clamp(rint(X_perm[t, d] / s_safe[t]), -8, 7)
                    with q zeroed out where s[t] == 0
    X_s4          = pack_s4_le(q)                          # (T, d_in // 2) int8
    sum_X[t, g]   = Σ_{k in g} q[t, k]                     # per-group int32 sum

where ``rint`` is IEEE-754 round-half-to-even (same as torch.round).

Notes
-----
* Division (not multiplication by 1/s) is used on purpose — matches
  the kernel (single rounding step) and ``torch.round(x / s)``.
* The kernel has an L2-thrash workaround that pre-permutes X and
  swaps ``perm`` for identity when ``T*D > 32 Mi elems``.  That does
  not alter the mathematical output, so this reference ignores it.
"""

from __future__ import annotations

from typing import Tuple

import torch

from kernel.triton_kernel.pack_utils import BCOL, pack_s4_le


def quantize_activation_s4_reference(
    X_fp16: torch.Tensor,
    perm: torch.Tensor,
    bcol: int = BCOL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch equivalent of
    :func:`kernel.triton_kernel.activation_quant.quantize_activation_s4`.

    Runs on any device (CPU or CUDA); produces identical outputs up to
    the rounding guarantees described in the module docstring.

    Args:
        X_fp16: ``(T, d_in)`` or ``(batch, seq, d_in)`` fp16 tensor.
        perm:   ``(d_in,)`` int32 or int64 permutation.
        bcol:   group size along d_in (default 128 = BCOL).

    Returns:
        X_s4    : ``(T, d_in // 2)`` int8 packed SINT4.
        scale_x : ``(T,)``             fp16.
        sum_X   : ``(T, n_groups)``    int32.
    """
    if X_fp16.dtype != torch.float16:
        raise TypeError(f"X must be fp16, got {X_fp16.dtype}")
    if perm.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"perm must be int32 / int64, got {perm.dtype}")

    # Flatten to 2D (T, d_in).
    if X_fp16.dim() == 3:
        T, D = X_fp16.shape[0] * X_fp16.shape[1], X_fp16.shape[2]
    elif X_fp16.dim() == 2:
        T, D = X_fp16.shape
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {tuple(X_fp16.shape)}")

    if D % bcol != 0:
        raise ValueError(f"d_in ({D}) must be divisible by bcol ({bcol})")
    if D % 2 != 0:
        raise ValueError(f"d_in ({D}) must be even for 4-bit packing")

    X_2d = X_fp16.reshape(T, D).contiguous()
    perm_long = perm.to(torch.int64)

    # X_perm[t, d] = X[t, perm[d]] — the kernel gathers permuted columns
    # inside the kernel; we do it up-front here.
    X_perm = X_2d.index_select(1, perm_long).to(torch.float32)   # (T, D) fp32

    # Per-token max(|x|) / 7 -> fp16 -> fp32 (bitwise matches kernel).
    max_abs = X_perm.abs().amax(dim=1)                           # (T,) fp32
    scale_fp32 = max_abs / 7.0
    scale_fp16 = scale_fp32.to(torch.float16)
    s = scale_fp16.to(torch.float32)

    is_zero = s <= 0.0
    s_safe = torch.where(is_zero, torch.ones_like(s), s)

    # Quantize: round-half-to-even via torch.round, then clamp.
    q = torch.round(X_perm / s_safe[:, None]).clamp(min=-8.0, max=7.0)
    q_i32 = q.to(torch.int32)
    # Zero-scale rows must produce q=0, not the fallback 1.0 output.
    q_i32 = torch.where(is_zero[:, None], torch.zeros_like(q_i32), q_i32)

    # Per-group SINT4 sum (groups span bcol columns of d_in).
    n_groups = D // bcol
    # reshape (T, n_groups, bcol) -> sum over last axis -> (T, n_groups)
    sum_X = q_i32.view(T, n_groups, bcol).sum(dim=-1).to(torch.int32)

    # Pack to little-endian SINT4 int8.
    X_s4 = pack_s4_le(q_i32.to(torch.int8))                      # (T, D // 2) int8

    return X_s4, scale_fp16, sum_X


__all__ = ["quantize_activation_s4_reference"]
