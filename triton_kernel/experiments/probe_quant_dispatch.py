"""One-off experiment: measure autotune-dispatcher overhead in quant kernel.

Idea: build two variants of the same kernel body:
    1. The current autotuned `quantize_activation_kernel`
    2. A bypass variant with fixed BT/BD/num_warps/num_stages

Compare their decode-shape (T=1, T=16) latency.  Delta = dispatcher cost.

Run on server with:
    PYTHONPATH=/root python /root/Zip_kernel/triton_kernel/experiments/probe_quant_dispatch.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as tl_libdevice

from kernel.triton_kernel.pack_utils import BCOL
from kernel.triton_kernel.benchmarks._bench_util import time_ms
from kernel.triton_kernel.activation_quant import (
    quantize_activation_s4,
    quantize_activation_kernel,
)


# Bypass: same kernel body but no @triton.autotune wrapper.  We inline
# the essential subset of the body that matches the autotuned version.
# (Copy-paste is fine for a one-off probe.)
@triton.jit
def quantize_activation_kernel_fixed(
    X_ptr, perm_ptr,
    X_s4_ptr, scale_x_ptr, sum_X_ptr,
    T, D,
    stride_xt, stride_xd,
    stride_qt, stride_qd,
    stride_st, stride_sg,
    N_GROUPS: tl.constexpr,
    BCOL_K: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    pid_t = tl.program_id(0)
    t_start = pid_t * BT
    offs_t = t_start + tl.arange(0, BT)
    mask_t = offs_t < T
    max_abs = tl.zeros((BT,), dtype=tl.float32)
    for d_start in range(0, D, BD):
        offs_d = d_start + tl.arange(0, BD)
        mask_d = offs_d < D
        perm_idx = tl.load(perm_ptr + offs_d, mask=mask_d, other=0).to(tl.int32)
        x_ptrs = X_ptr + offs_t[:, None] * stride_xt + perm_idx[None, :] * stride_xd
        x_tile = tl.load(x_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        tile_max = tl.max(tl.abs(x_tile), axis=1)
        max_abs = tl.maximum(max_abs, tile_max)
    scale_fp32 = max_abs / 7.0
    scale_fp16 = scale_fp32.to(tl.float16)
    scale = scale_fp16.to(tl.float32)
    scale_safe = tl.where(scale > 0.0, scale, 1.0)
    scale_is_zero = scale <= 0.0
    tl.store(scale_x_ptr + offs_t, scale_fp16, mask=mask_t)
    offs_h = tl.arange(0, BCOL_K // 2)
    for g in range(0, N_GROUPS):
        d_start = g * BCOL_K
        offs_d_lo = d_start + 2 * offs_h
        offs_d_hi = d_start + 2 * offs_h + 1
        mask_d_lo = offs_d_lo < D
        mask_d_hi = offs_d_hi < D
        perm_lo = tl.load(perm_ptr + offs_d_lo, mask=mask_d_lo, other=0).to(tl.int32)
        perm_hi = tl.load(perm_ptr + offs_d_hi, mask=mask_d_hi, other=0).to(tl.int32)
        x_lo = tl.load(
            X_ptr + offs_t[:, None] * stride_xt + perm_lo[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d_lo[None, :], other=0.0,
        ).to(tl.float32)
        x_hi = tl.load(
            X_ptr + offs_t[:, None] * stride_xt + perm_hi[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d_hi[None, :], other=0.0,
        ).to(tl.float32)
        q_lo = x_lo / scale_safe[:, None]
        q_lo = tl_libdevice.rint(q_lo)
        q_lo = tl.minimum(tl.maximum(q_lo, -8.0), 7.0)
        q_lo_i32 = q_lo.to(tl.int32)
        q_lo_i32 = tl.where(scale_is_zero[:, None], 0, q_lo_i32)
        q_lo_i32 = tl.where(mask_d_lo[None, :], q_lo_i32, 0)
        q_hi = x_hi / scale_safe[:, None]
        q_hi = tl_libdevice.rint(q_hi)
        q_hi = tl.minimum(tl.maximum(q_hi, -8.0), 7.0)
        q_hi_i32 = q_hi.to(tl.int32)
        q_hi_i32 = tl.where(scale_is_zero[:, None], 0, q_hi_i32)
        q_hi_i32 = tl.where(mask_d_hi[None, :], q_hi_i32, 0)
        g_sum = tl.sum(q_lo_i32, axis=1) + tl.sum(q_hi_i32, axis=1)
        tl.store(sum_X_ptr + offs_t * stride_st + g * stride_sg, g_sum, mask=mask_t)
        low = q_lo_i32 & 0x0F
        high = q_hi_i32 & 0x0F
        packed = ((high << 4) | low) & 0xFF
        packed_i8 = tl.where(packed >= 128, packed - 256, packed).to(tl.int8)
        byte_offs = (d_start // 2) + offs_h
        byte_mask = byte_offs < (D // 2)
        qs_ptrs = X_s4_ptr + offs_t[:, None] * stride_qt + byte_offs[None, :] * stride_qd
        tl.store(qs_ptrs, packed_i8, mask=mask_t[:, None] & byte_mask[None, :])


def main():
    torch.manual_seed(0)
    print(f"{'T':>5} {'D':>6} {'autotune_us':>14} {'fixed_us':>12} {'Δ_us':>8} {'Δ_%':>7}")
    for T, D in [(1, 4096), (1, 11008), (1, 14336), (16, 4096), (16, 11008), (64, 4096)]:
        n_groups = D // BCOL
        X = torch.randn(T, D, dtype=torch.float16, device='cuda') * 0.5
        perm = torch.randperm(D, dtype=torch.int32, device='cuda')

        def run_autotune():
            quantize_activation_s4(X, perm, bcol=BCOL)

        X_s4 = torch.empty((T, D // 2), dtype=torch.int8, device='cuda')
        scale_x = torch.empty((T,), dtype=torch.float16, device='cuda')
        sum_X = torch.empty((T, n_groups), dtype=torch.int32, device='cuda')

        # Best config based on autotune experience for small-T: BT=16, BD=512, warps=2, stages=3
        BT_FIXED, BD_FIXED = 16, 512
        grid = (triton.cdiv(T, BT_FIXED),)

        def run_fixed():
            quantize_activation_kernel_fixed[grid](
                X, perm, X_s4, scale_x, sum_X, T, D,
                X.stride(0), X.stride(1),
                X_s4.stride(0), X_s4.stride(1),
                sum_X.stride(0), sum_X.stride(1),
                N_GROUPS=n_groups, BCOL_K=BCOL,
                BT=BT_FIXED, BD=BD_FIXED,
                num_warps=2, num_stages=3,
            )

        # warmup both
        for _ in range(50):
            run_autotune(); run_fixed()
        torch.cuda.synchronize()
        t_at = time_ms(run_autotune, n_warmup=100, n_iter=200, n_repeat=5) * 1000
        t_fix = time_ms(run_fixed, n_warmup=100, n_iter=200, n_repeat=5) * 1000
        delta = t_at - t_fix
        pct = 100 * delta / t_at if t_at > 0 else 0
        print(f"{T:>5} {D:>6} {t_at:>14.2f} {t_fix:>12.2f} {delta:>+8.2f} {pct:>+6.1f}%")


if __name__ == "__main__":
    main()
