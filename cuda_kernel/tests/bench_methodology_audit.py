"""Option III: bench methodology audit — why does r68 prefill bench report
14B/32B kv T=2048 at sp=0.96x, while strict L2-flush probe shows 1.03x?

Hypothesis: bench_qwen3_shapes._flush_l2 has a calibration bug that
undercounts the "flush cost" when subtracting it from FP16 measurements,
making FP16 appear artificially fast (and thus W4A4 appear 0.96x instead
of its real 1.03x).

This script reproduces the exact bench_qwen3 FP16 measurement using
`tools.profile.full_bench_vs_bf16._bench_cuda_event` in both tight-loop
mode (no flush) and with-flush mode, and prints the raw numbers alongside
our probe's median-of-3-trials protocol, to isolate the discrepancy.

We pick 3 "suspicious" shapes where probe and bench disagree:
  - 14B kv T=2048: bench 0.96x,   probe 1.03x  (delta 7%)
  - 32B kv T=2048: bench 0.97x,   probe 1.03x  (delta 6%)
  - 14B kv T=4096: bench 1.01x,   probe ?       (control)
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs, bench_us

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})


def probe_bench(fn, warmup=500, outer=10, inner=200, n_trials=3):
    """Our probe-style strict measurement."""
    trials = []
    for _ in range(n_trials):
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
        trials.append(best)
    return statistics.median(trials)


def probe_with_flush(fn, warmup=500, outer=10, inner=200, flush_mb=96):
    """Probe with L2 flush (mimicking bench_qwen3)."""
    flush = torch.empty(flush_mb * 1024 * 256, dtype=torch.int8, device="cuda")
    def _flush_once(): flush.zero_()
    for _ in range(warmup):
        _flush_once(); fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            _flush_once(); fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / inner)
    # Measure flush cost separately
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
    return best, flush_us  # return both raw-with-flush and flush-cost


SUSPECTS = [
    ("Qwen3-14B kv T=2048", 2048, 5120, 2048),
    ("Qwen2.5-32B kv T=2048", 2048, 5120, 2048),
    ("Qwen3-14B kv T=4096", 4096, 5120, 2048),  # control
    # Add winner too
    ("Qwen3-8B gu T=2048",  2048, 4096, 24576),
]


PR("=" * 150)
PR("Bench methodology audit: FP16 flush-cost calibration — is it accurate?")
PR("=" * 150)
PR()
PR(f"  {'shape':<26}  {'cuda (probe, no flush)':>22}  {'cuda (probe, flush)':>22}  "
   f"{'fp16 bench_qwen3':>18}  {'fp16 (probe no flush)':>22}  "
   f"{'fp16 (probe flush)':>22}  {'flush cost':>10}  {'fp16 NET (flush-cost)':>20}")

for label, T, d_in, d_out in SUSPECTS:
    b = make_inputs(T, d_out, d_in, hp_ratio=0.0,
                    device="cuda", seed=T + d_in + d_out)
    W_fp = b["W_fp"]; X_fp_t = b["X_fp_t"]; X = b["X"]; perm = b["perm"]

    def run_fp16():
        return torch.matmul(W_fp, X_fp_t)

    def run_cuda():
        X_s4, sx, sX = ops.activation_quant_cuda(X, perm)
        return ops.fused_dense_sparse_cuda_int4(
            b["W_low_packed"], b["W_high_packed"],
            b["hp_row_offsets"], b["hp_col_indices"],
            X_s4, b["scale_u4"], b["zero_u4"], sX, sx, d_out, d_in,
        )

    # 1. CUDA (probe, no flush) — the "real" number
    cuda_noflush = probe_bench(run_cuda, warmup=500, outer=10, inner=200, n_trials=3)
    # 2. CUDA (probe, flush) — with flush (noise check)
    cuda_flush, flush_c1 = probe_with_flush(run_cuda)
    # 3. FP16 (bench_qwen3 method, with its own flush) — what's in the bench
    fp16_bench = bench_us(run_fp16, warmup=50, outer=3, inner=100, flush_l2=True)
    # 4. FP16 (probe, no flush) — L2 warm
    fp16_noflush = probe_bench(run_fp16, warmup=500, outer=10, inner=200, n_trials=3)
    # 5. FP16 (probe, flush) — with flush cost NOT subtracted
    fp16_flush, flush_c = probe_with_flush(run_fp16)
    # 6. FP16 NET (flush - flush_cost) — our probe's calibrated baseline
    fp16_net = max(0.0, fp16_flush - flush_c)

    PR(f"  {label:<26}  {cuda_noflush:>21.2f}  {cuda_flush:>21.2f}  "
       f"{fp16_bench:>17.2f}  {fp16_noflush:>21.2f}  "
       f"{fp16_flush:>21.2f}  {flush_c:>9.2f}  {fp16_net:>19.2f}")

PR()
PR("Interpretation:")
PR("  - If 'fp16 bench_qwen3' << 'fp16 (probe flush)', bench is undercounting flush cost.")
PR("  - If 'fp16 bench_qwen3' ≈ 'fp16 (probe no flush)', bench isn't actually flushing L2.")
PR("  - If 'fp16 bench_qwen3' ≈ 'fp16 NET (flush-cost)', bench is accurate.")
