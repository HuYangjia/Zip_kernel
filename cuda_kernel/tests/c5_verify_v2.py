"""C.5 verify v2: true in-process A/B using HKUST_V9_FUSED_FORCE_KBM.

The v1 comparison against r66's published bench.json was contaminated
by GPU environment drift (all shapes showed -30% or more, impossible
from a dispatcher change that only affects 4 shapes).

This v2 isolates the dispatcher delta by:
  - path A: force kBm=128 (r66's old behaviour for target shapes)
  - path B: force kBm=64  (C.5's new behaviour for target shapes)
Both paths benched interleaved in the same process on the same GPU,
giving a clean uplift measurement.

For guard shapes (which C.5 does NOT change), we check that switching
kBm in this way captures the dispatcher's *intent*: guard shapes
should pick the r66 choice naturally, so env-forcing kBm=128 vs
kBm=64 should show whatever they were already doing is near-optimal.
"""
import os
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


def set_env_kbm(val):
    if val is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = str(val)


TARGETS = [
    # (model, proj, T, d_in, d_out, C5_target?)
    ("Qwen3-8B",  "q_proj",      128, 4096, 4096, True),
    ("Qwen3-8B",  "o_proj",      128, 4096, 4096, True),
    ("Qwen3-4B",  "q_proj",      128, 2560, 4096, True),
    ("Qwen3-4B",  "o_proj",      128, 4096, 2560, True),
    # Guards: should NOT benefit from kBm=64 (or already on best)
    ("Qwen3-14B", "kv_proj",     128, 5120, 2048, False),
    ("Qwen3-14B", "gate_up_proj", 128, 5120, 34816, False),
    ("Qwen3-8B",  "gate_up_proj", 128, 4096, 24576, False),
    ("LLaMA3-70B","kv_proj",     128, 8192, 2048, False),
    ("Qwen3-0.6B","q_proj",      128, 1024, 2048, False),
]


print("=" * 90)
print("C.5 verify v2 — in-process A/B (kBm=128 vs kBm=64)")
print("=" * 90)
print(f"{'shape':<44} {'kBm=128':>9} {'kBm=64':>9} {'delta':>8} {'c5 tgt?':>8}")

c5_wins = []
guard_deltas = []
for model, proj, T, d_in, d_out, c5_tgt in TARGETS:
    b = make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda",
                    seed=T + d_in + d_out)

    def _run():
        run_bundle(b, d_out, d_in)

    us_128, us_64 = [], []
    for _ in range(4):
        set_env_kbm(128)
        us_128.append(bench_us(_run))
        set_env_kbm(64)
        us_64.append(bench_us(_run))
    set_env_kbm(None)

    m128 = statistics.median(us_128)
    m64 = statistics.median(us_64)
    delta = (m64 - m128) / m128 * 100
    flag = "C5 TGT" if c5_tgt else "guard"
    if c5_tgt and delta < -5:
        c5_wins.append(-delta)
    if not c5_tgt:
        guard_deltas.append(delta)

    print(f"  {model}/{proj} T={T} {d_in}→{d_out:<6}  "
          f"{m128:>8.2f}us  {m64:>8.2f}us  {delta:+7.2f}%  {flag:>8}")

print()
print("=" * 90)
print("SUMMARY (same-process, same-GPU, interleaved 4× A/B)")
print("=" * 90)
if c5_wins:
    print(f"C.5 target shapes win ≥5%: {len(c5_wins)}/4  "
          f"avg uplift (kBm=64 vs 128): {statistics.mean(c5_wins):.2f}%")
if guard_deltas:
    best_guard = min(guard_deltas)
    worst_guard = max(guard_deltas)
    print(f"Guard shapes delta (kBm=64 vs 128) range: "
          f"[{best_guard:+.2f}%, {worst_guard:+.2f}%]")
    print("Interpretation: negative delta on a guard means kBm=64 is ACTUALLY")
    print("better there too; C.5 may be too conservative (could widen more).")
    print("Positive delta on a guard means kBm=128 IS the right choice (C.5 OK).")
