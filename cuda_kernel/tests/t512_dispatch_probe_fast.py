"""T=512 dispatcher FAST probe (Step 2A, compressed).

Reduced from v1 (12×12) to 6×6 for rapid turnaround (~3 min).
Only measures the most likely knobs.  If any shape shows a clean
>=5% win from a non-default config, we'll follow up with a full
probe on that config.

print(..., flush=True) for line-buffered output under redirection.
"""
import os
import statistics
import sys

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})

dev = torch.device("cuda:0")

def prep(T, d_in, d_out):
    return make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda",
                       seed=T + d_in + d_out)

def run(b, d_out, d_in):
    return ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        b["X_s4"], b["scale_u4"], b["zero_u4"],
        b["sum_X"], b["scale_x"], d_out, d_in,
    )

def bench_us(fn, warmup=200, outer=5, inner=100):
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


# Compressed target set (6 shapes)
TARGETS = [
    ("G1 8B kv",      4096,  2048),   # mid kv, sp=0.79
    ("G1 14B kv",     5120,  2048),   # sp=0.84
    ("G2 14B gu",     5120, 34816),   # big d_out compute-bound, sp=0.79
    ("G2 70B gu",     8192, 57344),   # sp=0.71
    ("G3 32B dn",    27648,  5120),   # big d_in, sp=0.71
    ("G4 14B q",      5120,  5120),   # mid q, sp=0.99
]

# Compressed config space (6 configs — most promising knobs only)
CONFIGS = [
    ("default",        None, None, None, None),
    ("sk=2",           None, None, None, "2"),
    ("kBm=64",         None, "64", None, None),
    ("kBn=32",         "32", None, None, None),
    ("cache=1",        None, None, "1",  None),
    ("cache=0",        None, None, "0",  None),
]

def apply_env(kbn, kbm, cache, sk):
    for k in ("HKUST_V9_FUSED_FORCE_KBN", "HKUST_V9_FUSED_FORCE_KBM",
              "HKUST_V9_FUSED_FORCE_CACHE", "HKUST_V9_FUSED_FORCE_SPLITK"):
        os.environ.pop(k, None)
    if kbn   is not None: os.environ["HKUST_V9_FUSED_FORCE_KBN"]    = kbn
    if kbm   is not None: os.environ["HKUST_V9_FUSED_FORCE_KBM"]    = kbm
    if cache is not None: os.environ["HKUST_V9_FUSED_FORCE_CACHE"]  = cache
    if sk    is not None: os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = sk


T = 512
PR("=" * 90)
PR(f"T=512 FAST probe — {len(TARGETS)} shapes × {len(CONFIGS)} configs (single trial each)")
PR("=" * 90)

all_results = []
for ti, (label, d_in, d_out) in enumerate(TARGETS):
    PR(f"\n## [{ti+1}/{len(TARGETS)}] {label}  d_in={d_in} d_out={d_out}")
    b = prep(T, d_in, d_out)
    def _run():
        run(b, d_out, d_in)

    us_by_cfg = {}
    for cfg_lbl, kbn, kbm, cache, sk in CONFIGS:
        apply_env(kbn, kbm, cache, sk)
        us = bench_us(_run)
        us_by_cfg[cfg_lbl] = us
        PR(f"    {cfg_lbl:<14}  {us:>8.2f}us")

    d = us_by_cfg["default"]
    sorted_cfgs = sorted(us_by_cfg.items(), key=lambda kv: kv[1])
    best_cfg, best_us = sorted_cfgs[0]
    uplift = (d - best_us) / d * 100
    mark = "  WIN" if (best_cfg != "default" and uplift > 2) else ""
    PR(f"  -> best='{best_cfg}'  {best_us:.2f}us  uplift={uplift:+.2f}%{mark}")

    all_results.append({"label": label, "d_in": d_in, "d_out": d_out,
                        "default_us": d, "best_cfg": best_cfg, "best_us": best_us,
                        "uplift": uplift})

PR()
PR("=" * 90)
PR("SUMMARY")
PR("=" * 90)
PR(f"{'shape':<16} {'default':>8} {'best_cfg':<12} {'best_us':>8} {'uplift':>8}")
wins = 0
total_uplift = 0.0
for r in all_results:
    mark = "  WIN" if (r["best_cfg"] != "default" and r["uplift"] > 2) else ""
    PR(f"  {r['label']:<14} {r['default_us']:>7.2f}  {r['best_cfg']:<12} "
       f"{r['best_us']:>7.2f}  {r['uplift']:+7.2f}%{mark}")
    if mark:
        wins += 1
        total_uplift += r["uplift"]
PR(f"\nWins ≥2%: {wins}/{len(all_results)}  "
   f"avg uplift: {total_uplift/max(wins,1):+.2f}%")

from collections import Counter
c = Counter(r["best_cfg"] for r in all_results if r["uplift"] > 2)
if c:
    PR("\nWinner config distribution:")
    for cfg, n in c.most_common():
        PR(f"  {cfg:<14}  {n} shape(s)")
