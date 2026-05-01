"""C.6 v2 verify — widened A/B test covering both WIN targets and
REGRESS shapes that C.6-v1 incorrectly activated on.

Protocol (per [[memory:bmmiahpl]]):
  warmup=500, outer=10, inner=200, 4 interleaved trials, median.

For each shape we measure:
  A path: force sk=1 (pre-C.6 behaviour)
  B path: default (C.6-v2 auto-picks sk based on region A/B/C gates)

Expected:
  - C.6 WIN shapes (in regions A/B/C): B faster than A by 5-19%.
  - C.6 EXCLUDED shapes (the ones that REGRESSED under v1): B == A
    because the v2 gate doesn't fire, both paths end at sk=1.
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})


def run_bundle(b, d_out, d_in):
    return ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        b["X_s4"], b["scale_u4"], b["zero_u4"],
        b["sum_X"], b["scale_x"], d_out, d_in,
    )


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


def set_sk(v):
    if v is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(v)


T = 512

# C.6-v2 WIN targets: all 8 shapes the revised rule fires on
WIN_TARGETS = [
    # Region A: d_out <= 5120 AND d_in in [5120, 8192]
    ("Qwen3-14B", "q",   5120,  5120, "A"),
    ("Qwen3-14B", "kv",  5120,  2048, "A"),
    ("Qwen3-14B", "o",   5120,  5120, "A"),
    ("Qwen2.5-32B", "kv", 5120, 2048, "A"),
    ("Qwen3-70B", "kv",  8192,  2048, "A+C"),
    # Region B: d_in >= 16384
    ("Qwen3-14B",  "dn", 17408, 5120, "B"),
    ("Qwen2.5-32B","dn", 27648, 5120, "B"),
    ("LLaMA3-70B", "dn", 28672, 8192, "B"),
    # Region C: d_out <= 2048 AND d_in >= 6144 (1.7B dn)
    ("Qwen3-1.7B", "dn", 6144,  2048, "C"),
]

# Shapes that C.6-v1 REGRESSED on; C.6-v2 must NOT fire on them
EXCLUDED = [
    ("Qwen3-8B",   "q",   4096,  4096, "d_in=4096 ⇒ all regions fail"),
    ("Qwen3-8B",   "kv",  4096,  2048, "d_in=4096 ⇒ excluded"),
    ("Qwen3-8B",   "o",   4096,  4096, "d_in=4096 ⇒ excluded"),
    ("Qwen3-8B",   "gu",  4096, 24576, "d_in=4096 + big d_out ⇒ excluded"),
    ("Qwen3-8B",   "dn", 12288,  4096, "d_out>2048 + d_in<16384 ⇒ excluded"),
    ("Qwen3-4B",   "o",   4096,  2560, "d_in=4096 ⇒ excluded"),
    ("Qwen3-4B",   "dn",  9728,  2560, "d_out>2048 + d_in<16384 ⇒ excluded"),
    ("Qwen2.5-32B","gu",  5120, 55296, "d_out>5120 ⇒ region A fails"),
    ("Qwen3-70B",  "q",   8192,  8192, "d_out>5120 ⇒ A fails, d_out>2048 ⇒ C fails"),
    ("Qwen3-70B",  "o",   8192,  8192, "d_out>5120 ⇒ excluded"),
    ("Qwen3-70B",  "gu",  8192, 57344, "d_out huge ⇒ excluded"),
]


def bench_shape(d_in, d_out):
    b = make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda",
                    seed=T + d_in + d_out)
    def _run():
        run_bundle(b, d_out, d_in)
    us_sk1, us_c6 = [], []
    for _ in range(4):
        set_sk(1)
        us_sk1.append(bench_us(_run))
        set_sk(None)
        us_c6.append(bench_us(_run))
    set_sk(None)
    return statistics.median(us_sk1), statistics.median(us_c6)


PR("=" * 90)
PR("C.6-v2 verify — WIN targets")
PR("=" * 90)
PR(f"{'shape':<28} {'region':>7} {'sk=1 us':>9} {'C6v2 us':>9} {'delta':>8}")
wins = []
for model, proj, d_in, d_out, region in WIN_TARGETS:
    m1, mc = bench_shape(d_in, d_out)
    delta = (mc - m1) / m1 * 100
    mark = "  WIN" if delta < -3 else ("  FAIL" if delta > 3 else "  neutral")
    if delta < -3:
        wins.append(-delta)
    PR(f"  {model}/{proj} {d_in}→{d_out:<6}  {region:>5}  {m1:>8.2f}  "
       f"{mc:>8.2f}  {delta:+7.2f}%{mark}")


PR()
PR("=" * 90)
PR("C.6-v2 excluded shapes (v1 regressed; v2 must NOT fire ⇒ sk=1≈C6 default)")
PR("=" * 90)
PR(f"{'shape':<28} {'sk=1 us':>9} {'C6v2 us':>9} {'delta':>8}  {'status':<10}")
regress = 0
for model, proj, d_in, d_out, note in EXCLUDED:
    m1, mc = bench_shape(d_in, d_out)
    delta = (mc - m1) / m1 * 100
    if abs(delta) < 3:
        mark = "OK"
    elif delta < -3:
        mark = "unexpected_win"
    else:
        mark = "REGRESS!"
        regress += 1
    PR(f"  {model}/{proj} {d_in}→{d_out:<6}  {m1:>8.2f}  {mc:>8.2f}  "
       f"{delta:+7.2f}%  {mark:<10}  ({note})")


PR()
PR("=" * 90)
PR("SUMMARY")
PR("=" * 90)
if wins:
    PR(f"C.6-v2 WIN: {len(wins)}/{len(WIN_TARGETS)}  "
       f"median uplift: {statistics.median(wins):+.2f}%  "
       f"mean: {statistics.mean(wins):+.2f}%")
PR(f"C.6-v2 EXCLUDED regress: {regress}/{len(EXCLUDED)}")
