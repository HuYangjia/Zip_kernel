"""C.7 validation: strict A/B test before/after the dispatcher change.

Compares:
  (1) DEFAULT path (now: C.7 new rule active)
  (2) env-forced sk=1 baseline (pre-C.7 behaviour)
  (3) env-forced sk=2 (what we expect C.7 to do for region D)
  (4) FP16 cuBLAS with L2 flush (ground truth baseline)

We expect on target (14B gu T >= 2048):
  default == forced sk=2 << sk=1 baseline
  default speedup > 1.0x vs fp16

We also verify on 6 winner shapes:
  default ≈ sk=1 baseline (no change, within 2%)
  default ≈ winner behaviour expected
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


def bench_fp16_flushed(W_fp, X_fp_t, warmup=200, outer=5, inner=100,
                      flush_mb=96):
    flush = torch.empty(flush_mb * 1024 * 256, dtype=torch.int8, device="cuda")
    def _flush_once():
        flush.zero_()
    for _ in range(warmup):
        _flush_once(); torch.matmul(W_fp, X_fp_t)
    torch.cuda.synchronize()
    best = float('inf')
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            _flush_once(); torch.matmul(W_fp, X_fp_t)
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / inner)
    for _ in range(warmup):
        _flush_once()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(inner):
        _flush_once()
    e.record()
    torch.cuda.synchronize()
    flush_us = s.elapsed_time(e) * 1000.0 / inner
    return max(0.0, best - flush_us)


def set_env(sk=None):
    os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)
    if sk is not None:
        os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(sk)


def run_e2e(X, perm, b, d_out, d_in):
    X_s4, sx, sX = ops.activation_quant_cuda(X, perm)
    return ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        X_s4, b["scale_u4"], b["zero_u4"], sX, sx, d_out, d_in,
    )


# ===================================================================
# TARGET shapes (C.7 should change behaviour, default should == sk=2)
# ===================================================================
TARGETS = [
    ("14B gu T=2048",  2048, 5120, 34816),
    ("14B gu T=4096",  4096, 5120, 34816),
    ("14B gu T=8192",  8192, 5120, 34816),
    # Also verify T=512 which C.7 gate now allows too
    ("14B gu T=512",   512,  5120, 34816),
]

# ===================================================================
# GUARD shapes (C.7 should NOT change behaviour; default must == sk=1)
# ===================================================================
GUARDS = [
    # 32B gu (d_out=55296): excluded by d_out <= 44000
    ("32B gu T=2048 (guard)",   2048, 5120, 55296),
    # 70B gu (d_in=8192): excluded by d_in == 5120
    ("70B gu T=2048 (guard)",   2048, 8192, 57344),
    # 8B gu (d_in=4096): excluded by d_in == 5120
    ("8B gu T=2048 (guard)",    2048, 4096, 24576),
    # 4B q (d_in=2560): excluded, should stay 1.3x winner
    ("4B q T=2048 (guard)",     2048, 2560, 4096),
    # 14B q (d_out=5120 < 32768): excluded from region D, in region A (sk=2 via C.6v2)
    ("14B q T=2048 (guard)",    2048, 5120, 5120),
    # 32B dn (d_in=27648 >= 16384): caught by C.6v2 region B, should stay sk=2
    ("32B dn T=2048 (guard)",   2048, 27648, 5120),
]


PR("=" * 120)
PR("C.7 validation A/B — verify dispatcher behaviour before/after the new rule")
PR("=" * 120)

PR()
PR("### TARGET shapes (C.7 should now route to sk=2; default must match forced sk=2)")
PR(f"  {'shape':<22}  {'fp16':>7}  {'default':>8} {'force sk=1':>11} {'force sk=2':>11}  "
   f"{'default_sp':>10}  {'sk1_sp':>7}  {'sk2_sp':>7}  {'fix?':>6}")

for label, T, d_in, d_out in TARGETS:
    b = make_inputs(T, d_out, d_in, hp_ratio=0.0,
                    device="cuda", seed=T + d_in + d_out)
    W_fp = b["W_fp"]; X_fp_t = b["X_fp_t"]; X = b["X"]; perm = b["perm"]
    t_fp16 = bench_fp16_flushed(W_fp, X_fp_t)

    set_env(None);   t_def = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in))
    set_env(1);      t_sk1 = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in))
    set_env(2);      t_sk2 = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in))
    set_env(None)

    # Sanity: default should match sk=2 to within 3%
    default_matches_sk2 = abs(t_def - t_sk2) / t_sk2 < 0.03
    fixed = t_fp16 / t_def >= 1.0 and t_fp16 / t_sk1 < 1.0
    PR(f"  {label:<22}  {t_fp16:>6.2f}  {t_def:>7.2f} {t_sk1:>10.2f} {t_sk2:>10.2f}  "
       f"{t_fp16/t_def:>9.3f}x  {t_fp16/t_sk1:>6.3f}x  {t_fp16/t_sk2:>6.3f}x  "
       f"{'✓' if default_matches_sk2 and fixed else '✗':>6}")

PR()
PR("### GUARD shapes (C.7 must not change their behaviour)")
PR(f"  {'shape':<26}  {'fp16':>7}  {'default':>8} {'force sk=1':>11} {'force sk=2':>11}  "
   f"{'default_sp':>10}  {'sk1_sp':>7}  {'sk2_sp':>7}  {'note':<30}")

for label, T, d_in, d_out in GUARDS:
    b = make_inputs(T, d_out, d_in, hp_ratio=0.0,
                    device="cuda", seed=T + d_in + d_out)
    W_fp = b["W_fp"]; X_fp_t = b["X_fp_t"]; X = b["X"]; perm = b["perm"]
    t_fp16 = bench_fp16_flushed(W_fp, X_fp_t)

    set_env(None);   t_def = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in))
    set_env(1);      t_sk1 = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in))
    set_env(2);      t_sk2 = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in))
    set_env(None)

    # Note whether default matches sk=1 or sk=2 (C.6v2 may already have set sk=2 for some)
    d1 = abs(t_def - t_sk1) / t_sk1
    d2 = abs(t_def - t_sk2) / t_sk2
    if d1 < 0.03 and d2 > 0.03:
        note = "default = sk=1 (OK)"
    elif d2 < 0.03 and d1 > 0.03:
        note = "default = sk=2 (C.6v2 rule, OK)"
    elif d1 < 0.03 and d2 < 0.03:
        note = "sk=1 == sk=2 (ambiguous)"
    else:
        note = "DEFAULT DIVERGED?!"
    PR(f"  {label:<26}  {t_fp16:>6.2f}  {t_def:>7.2f} {t_sk1:>10.2f} {t_sk2:>10.2f}  "
       f"{t_fp16/t_def:>9.3f}x  {t_fp16/t_sk1:>6.3f}x  {t_fp16/t_sk2:>6.3f}x  "
       f"{note}")

PR()
PR("=" * 120)
PR("If TARGETS all show ✓ and GUARDS show no 'DEFAULT DIVERGED', C.7 is safe to ship.")
PR("=" * 120)
