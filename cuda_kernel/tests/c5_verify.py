"""C.5 verify: bench r66 (main, kbm gate old) vs C.5 (kbm gate new).

Target WIN shapes (probe predicted kBm=64 is best):
  - Qwen3-8B q/o T=128 (4096→4096)  predicted -14%/-6%
  - Qwen3-4B q/o T=128 (2560→4096, 4096→2560) predicted -12%/-10%

Guard NO-REGRESS shapes (boundary shapes that should NOT be affected):
  - R52-owned band:  d_out <= 2048 (Qwen3-14B kv 5120→2048, 1.7B down 6144→2048)
  - Out of new band: d_in > 4096 (LLaMA-70B kv 8192→2048, Qwen3-14B gate_up 5120→34816)
  - Other T:         T=32 gu (not T=128, should not hit new rule)

We bench the CURRENT HEAD (C.5 new gate active) directly; C.5 gate only
changes dispatcher behaviour so just run bench_us and compare against
r66's published bench.json numbers.
"""
import json
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs


dev = torch.device("cuda:0")


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


# Load r66 baseline
with open("/root/Zip_kernel/kernel/cuda_kernel/logs/r66_path_c/bench.json") as f:
    r66 = json.load(f)
r66_us = {}
for r in r66["records"]:
    if r.get("kernel") == "end_to_end":
        key = (r["model"], r["proj"], r["T"], r["d_in"], r["d_out"])
        r66_us[key] = r["cuda_us"]


TARGETS_WIN = [
    ("Qwen3-8B",  "q_proj",      128, 4096, 4096),
    ("Qwen3-8B",  "o_proj",      128, 4096, 4096),
    ("Qwen3-4B",  "q_proj",      128, 2560, 4096),
    ("Qwen3-4B",  "o_proj",      128, 4096, 2560),
]

TARGETS_GUARD = [
    # R52 band: d_out <= 2048 (should still take kBm=64, unchanged)
    ("Qwen3-14B",   "kv_proj",    128, 5120, 2048),
    ("Qwen3-1.7B",  "down_proj",  128, 6144, 2048),
    # Just outside new rule: d_in > 4096
    ("LLaMA3-70B",  "kv_proj",    128, 8192, 2048),
    # Just outside new rule: d_out > 4096 (wide d_out already on kBm=64 via C.3)
    ("Qwen3-14B",   "gate_up_proj", 128, 5120, 34816),
    # Qwen3-8B gu T=128 (d_in=4096 d_out=24576): still on kBm=128 (d_out > 4096)
    ("Qwen3-8B",    "gate_up_proj", 128, 4096, 24576),
    # Very small shape: Qwen3-0.6B (d_in=1024, out of new rule)
    ("Qwen3-0.6B",  "q_proj",     128, 1024, 2048),
    # Different T: ensure T!=128 not affected
    ("Qwen3-8B",    "q_proj",      32, 4096, 4096),
    ("Qwen3-8B",    "gate_up_proj", 32, 4096, 24576),
]


def prep(T, d_in, d_out):
    return make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda",
                       seed=T + d_in + d_out)


print("=" * 90)
print("C.5 verify: WIN targets")
print("=" * 90)
print(f"{'shape':<40} {'r66_us':>8} {'C5_us':>8} {'delta':>8}")
win_gains = []
for model, proj, T, d_in, d_out in TARGETS_WIN:
    b = prep(T, d_in, d_out)
    trials = [bench_us(lambda: run_bundle(b, d_out, d_in)) for _ in range(3)]
    c5 = statistics.median(trials)
    r66 = r66_us.get((model, proj, T, d_in, d_out), float("nan"))
    delta_pct = (c5 - r66) / r66 * 100 if r66 == r66 else float("nan")
    flag = "  WIN" if delta_pct < -2 else ("  REGRESS" if delta_pct > 2 else "")
    if delta_pct < -2:
        win_gains.append(-delta_pct)
    print(f"  {model}/{proj} T={T} {d_in}→{d_out:<6}  {r66:>7.2f}  {c5:>7.2f}  "
          f"{delta_pct:+7.2f}%{flag}")

print()
print("=" * 90)
print("C.5 verify: GUARD shapes (must not regress)")
print("=" * 90)
print(f"{'shape':<40} {'r66_us':>8} {'C5_us':>8} {'delta':>8}")
regress_count = 0
for model, proj, T, d_in, d_out in TARGETS_GUARD:
    b = prep(T, d_in, d_out)
    trials = [bench_us(lambda: run_bundle(b, d_out, d_in)) for _ in range(3)]
    c5 = statistics.median(trials)
    r66 = r66_us.get((model, proj, T, d_in, d_out), float("nan"))
    delta_pct = (c5 - r66) / r66 * 100 if r66 == r66 else float("nan")
    flag = ""
    if delta_pct > 3:
        regress_count += 1
        flag = "  REGRESS"
    print(f"  {model}/{proj} T={T} {d_in}→{d_out:<6}  {r66:>7.2f}  {c5:>7.2f}  "
          f"{delta_pct:+7.2f}%{flag}")

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"WIN shapes gain ≥2%: {len(win_gains)}/{len(TARGETS_WIN)}  "
      f"avg uplift: {statistics.mean(win_gains) if win_gains else 0:.2f}%")
print(f"GUARD shapes regress >3%: {regress_count}/{len(TARGETS_GUARD)}")
