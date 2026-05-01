"""P0 viability probe for kv_proj shapes — can raw P0.2 already beat FP16?

Motivation:
  14B kv_proj T=8..512 all lose to FP16 because the two-step path has
  a ~17us activation_quant launch floor, while the GEMM kernel itself
  (18.82us at T=8) is already ~30% faster than cuBLAS FP16 (26.77us).
  Total 37us loses to 27us.

  P0 (fused quant+MMA) is supposed to eliminate the launch floor.
  Even though P0.2 was ruled "neutral at best" on mid shapes, the
  kv_proj family has a UNIQUE property: the GEMM part is so small
  (d_out=2048, grid_m=16) that launch-floor dominance is EXTREME.
  This makes kv_proj the best candidate shape for P0 to shine.

  This probe measures P0.2 vs legacy two-step vs FP16 on kv_proj.
  If P0 wins kv_proj we enable shape-specific dispatch; if it
  doesn't, we commit to P0.4 (add cp.async / group-cache).

Protocol (per [[memory:bmmiahpl]]):
  warmup=500, outer=10, inner=200, 4 interleaved trials, median.
  Parity-verify the P0 path on every shape BEFORE timing.
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})


def bench_us(fn, warmup=500, outer=10, inner=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / inner)
    return best


def run_legacy(b, d_out, d_in):
    # activation_quant + fused_dense_sparse  (what bench_qwen3 measures)
    return ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        b["X_s4"], b["scale_u4"], b["zero_u4"],
        b["sum_X"], b["scale_x"], d_out, d_in,
    )


def run_legacy_e2e(X_fp16, perm, b, d_out, d_in):
    # activation_quant_cuda + fused_dense_sparse -- includes the launch floor
    X_s4, sx, sX = ops.activation_quant_cuda(X_fp16, perm)
    return ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        X_s4, b["scale_u4"], b["zero_u4"],
        sX, sx, d_out, d_in,
    )


def run_p0(X_fp16, perm, b, d_out, d_in):
    # P0 fused quant + dense+sparse MMA (single kernel)
    os.environ["HKUST_V9_P0_MODE"] = "1"
    return ops.fused_dense_sparse_e2e_cuda(
        X_fp16, perm,
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        b["scale_u4"], b["zero_u4"], d_out, d_in,
    )


def run_fp16(W_fp, X_fp_t):
    # Match bench_qwen3's fp16 protocol: W_fp @ X_fp_t (d_out, d_in) x (d_in, T)
    return torch.matmul(W_fp, X_fp_t)


TARGETS = [
    # Focus: Qwen3-14B kv_proj (the 4 losing Ts), plus a few sibling shapes
    # (other models' kv_proj) that have the same d_out=2048 launch-floor trap.
    ("14B kv T=8",    8,   5120, 2048),
    ("14B kv T=32",   32,  5120, 2048),
    ("14B kv T=128",  128, 5120, 2048),
    ("14B kv T=512",  512, 5120, 2048),
    ("8B kv T=8",     8,   4096, 2048),
    ("8B kv T=128",   128, 4096, 2048),
    ("32B kv T=8",    8,   5120, 2048),
    ("32B kv T=128",  128, 5120, 2048),
    ("70B kv T=8",    8,   8192, 2048),
    ("70B kv T=128",  128, 8192, 2048),
    # Control: q_proj at the same d_in/d_out (where GEMM is bigger,
    # launch floor less relevant -- P0 should be less of a win here).
    ("14B q T=8",     8,   5120, 5120),
    ("14B q T=128",   128, 5120, 5120),
]


PR("=" * 96)
PR("P0 viability probe on kv_proj (and q_proj control)")
PR("=" * 96)
PR(f"{'shape':<16} {'d_in':>5} {'d_out':>5} {'fp16 us':>8} "
   f"{'legacy us':>10} {'P0 us':>8} "
   f"{'fp16 sp(leg)':>13} {'fp16 sp(P0)':>12} {'P0 vs leg':>10}")

p0_wins_fp16 = 0
leg_wins_fp16 = 0
for label, T, d_in, d_out in TARGETS:
    b = make_inputs(T, d_out, d_in, hp_ratio=0.0,  # dense-only so P0 can run
                    device="cuda", seed=T + d_in + d_out)
    # Field names per bench_qwen3_shapes.make_inputs:
    #   X:  (T, d_in) fp16, the quantisation input
    #   W_fp: (d_out, d_in) fp16, the reference weight for FP16 matmul
    #   X_fp_t: (d_in, T) fp16, transposed view matching cuBLAS call shape
    #   perm: (d_in,) int32 identity permutation
    W_fp   = b.get("W_fp")
    X_fp16 = b.get("X")
    X_fp_t = b.get("X_fp_t")
    perm   = b.get("perm")

    if W_fp is None or X_fp16 is None:
        PR(f"  {label}: missing W_fp/X in make_inputs, skipping")
        continue

    # Parity check: P0 result vs legacy result (dense-only, same inputs)
    os.environ["HKUST_V9_P0_MODE"] = "1"
    y_p0 = run_p0(X_fp16, perm, b, d_out, d_in)
    os.environ["HKUST_V9_P0_MODE"] = "0"
    X_s4, sx, sX = ops.activation_quant_cuda(X_fp16, perm)
    y_leg = ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        X_s4, b["scale_u4"], b["zero_u4"], sX, sx, d_out, d_in,
    )
    rel = (y_p0.float() - y_leg.float()).abs().max().item() / (
        y_leg.float().abs().max().item() + 1e-12)
    if rel > 0.02:
        PR(f"  {label}: PARITY FAIL rel={rel:.4f}, skipping timing")
        continue

    # Timings
    t_fp16 = bench_us(lambda: run_fp16(W_fp, X_fp_t))
    t_leg  = bench_us(lambda: run_legacy_e2e(X_fp16, perm, b, d_out, d_in))
    t_p0   = bench_us(lambda: run_p0(X_fp16, perm, b, d_out, d_in))
    os.environ["HKUST_V9_P0_MODE"] = "0"

    sp_leg = t_fp16 / t_leg
    sp_p0  = t_fp16 / t_p0
    p0_vs_leg = (t_p0 - t_leg) / t_leg * 100
    if sp_p0 >= 1.0: p0_wins_fp16 += 1
    if sp_leg >= 1.0: leg_wins_fp16 += 1
    mark_p0 = " ** WIN" if sp_p0 >= 1.0 and sp_p0 > sp_leg else ""
    PR(f"  {label:<14} {d_in:>5} {d_out:>5} {t_fp16:>7.2f} "
       f"{t_leg:>9.2f} {t_p0:>7.2f} "
       f"{sp_leg:>12.3f}x {sp_p0:>11.3f}x {p0_vs_leg:+9.2f}%{mark_p0}")


PR()
PR(f"Summary: legacy wins fp16 on {leg_wins_fp16}/{len(TARGETS)}, "
   f"P0 wins fp16 on {p0_wins_fp16}/{len(TARGETS)}")
