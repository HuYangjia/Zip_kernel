"""C.6 verify — in-process A/B for the T=512 deep-K split_k=2 rule.

Protocol (per [[memory:bmmiahpl]]):
  warmup=500, outer=10, inner=200, 4 interleaved trials, median.

A vs B:
  A path (r67 behaviour): force SPLITK=1 via HKUST_V9_FUSED_FORCE_SPLITK=1
  B path (C.6 active):    default dispatcher (C.6 auto-picks sk=2 when T>=256 && n_g>=32)

Targets: 4 WIN shapes from the fast probe + 4 GUARD shapes that
should NOT be affected (small T or small n_groups).
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})

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


# T=512 shapes where probe predicted C.6 should activate
TARGETS_WIN = [
    ("Qwen3-14B", "gate_up_proj",  5120, 34816),   # +19.02% predicted
    ("Qwen2.5-32B", "down_proj",  27648,  5120),   # +18.94% predicted
    ("Qwen3-14B", "kv_proj",       5120,  2048),   # +5.80%
    ("Qwen3-14B", "q_proj",        5120,  5120),   # +5.63%
]

# Guards: C.6 should NOT activate (wrong T or wrong n_groups)
TARGETS_GUARD = [
    ("Qwen3-8B",  "q_proj",   128, 4096,  4096),   # T=128 ⇒ skip C.6
    ("Qwen3-8B",  "q_proj",    32, 4096,  4096),   # T=32  ⇒ skip C.6
    ("Qwen3-0.6B","q_proj",   512, 1024,  2048),   # T=512 but n_g=8 < 32
    ("Qwen3-1.7B","q_proj",   512, 2048,  2048),   # T=512 but n_g=16 < 32
]

T_WIN = 512


def set_env_sk(val):
    if val is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(val)


PR("=" * 90)
PR("C.6 verify v1 — in-process A/B (sk=1 forced vs C.6 auto)")
PR("=" * 90)
PR()
PR(f"{'shape':<44} {'sk=1 us':>9} {'C6 us':>9} {'delta':>8}")

win_gains = []
for entry in TARGETS_WIN:
    model, proj, d_in, d_out = entry
    T = T_WIN
    b = make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda",
                    seed=T + d_in + d_out)
    def _run():
        run_bundle(b, d_out, d_in)
    us_sk1, us_c6 = [], []
    for _ in range(4):
        set_env_sk(1)
        us_sk1.append(bench_us(_run))
        set_env_sk(None)
        us_c6.append(bench_us(_run))
    set_env_sk(None)
    m1 = statistics.median(us_sk1)
    mc = statistics.median(us_c6)
    delta = (mc - m1) / m1 * 100
    mark = "  WIN" if delta < -3 else ""
    if mark:
        win_gains.append(-delta)
    PR(f"  {model}/{proj} T={T} {d_in}→{d_out:<6}  "
       f"{m1:>8.2f}  {mc:>8.2f}  {delta:+7.2f}%{mark}")

PR()
PR("=" * 90)
PR("C.6 guards (C.6 should NOT fire ⇒ sk=1 and default should be equal)")
PR("=" * 90)
PR(f"{'shape':<44} {'sk=1 us':>9} {'C6 us':>9} {'delta':>8}")
regress = 0
for entry in TARGETS_GUARD:
    model, proj, T, d_in, d_out = entry
    b = make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda",
                    seed=T + d_in + d_out)
    def _run():
        run_bundle(b, d_out, d_in)
    us_sk1, us_c6 = [], []
    for _ in range(4):
        set_env_sk(1)
        us_sk1.append(bench_us(_run))
        set_env_sk(None)
        us_c6.append(bench_us(_run))
    set_env_sk(None)
    m1 = statistics.median(us_sk1)
    mc = statistics.median(us_c6)
    delta = (mc - m1) / m1 * 100
    if abs(delta) < 3:
        mark = "  OK"
    elif delta < -3:
        mark = "  UNEXPECTED WIN (investigate)"
    else:
        mark = "  REGRESS"
        regress += 1
    PR(f"  {model}/{proj} T={T} {d_in}→{d_out:<6}  "
       f"{m1:>8.2f}  {mc:>8.2f}  {delta:+7.2f}%{mark}")

PR()
PR("=" * 90)
PR("SUMMARY")
PR("=" * 90)
if win_gains:
    PR(f"C.6 WIN: {len(win_gains)}/{len(TARGETS_WIN)}  "
       f"median uplift: {statistics.median(win_gains):+.2f}%  "
       f"mean: {statistics.mean(win_gains):+.2f}%")
PR(f"C.6 GUARD regress: {regress}/{len(TARGETS_GUARD)}")
