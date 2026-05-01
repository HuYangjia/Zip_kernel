"""T=512 extended probe — hunt for more C.6-style wins.

Strategy: after C.6 (T>=256 + n_g>=32 + sk==1 → sk=2), check if
any additional T=512 shapes benefit from sk=2 BUT were missed by
the n_g>=32 or T>=256 gate.  Specifically probe:

  1. Shapes with n_g in [16, 31]  (excluded by n_g>=32) — would
     relaxing the gate to n_g>=16 capture more wins?
  2. Small T=512 shapes (Qwen3-0.6B, 1.7B) — double-check guard
     is correct (no wins forced by sk=2).
  3. All T=512 loser shapes in the full bench we haven't probed
     yet (8B dn, 70B kv, 4B dn, etc).

Method: for each shape, measure
  (a) C.6 default  (dispatcher picks via the new C.6 rule)
  (b) sk=1 forced  (simulate pre-C.6 behaviour)
  (c) sk=2 forced  (see if C.6 missed this shape)

Single trial each, but use strong warmup=300 outer=5 inner=150 for
reliability.  print(flush=True) for live output.
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})


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


def bench_us(fn, warmup=300, outer=5, inner=150):
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

# All T=512 loser shapes from r67 bench (sp < 1.0x), plus a few
# near-1x borderlines.  Grouped by n_groups bucket for analysis.
SHAPES = [
    # (model, proj, d_in, d_out, n_groups, notes)
    # Small n_g (<32) — expected NOT to benefit from sk=2
    ("0.6B", "q",   1024, 2048, 8,  "ng=8"),
    ("0.6B", "o",   2048, 1024, 16, "ng=16"),
    ("0.6B", "dn",  3072, 1024, 24, "ng=24"),
    ("0.6B", "gu",  1024, 6144, 8,  "ng=8"),
    ("1.7B", "q",   2048, 2048, 16, "ng=16"),
    ("1.7B", "dn",  6144, 2048, 48, "ng=48 !C.6 gate"),
    ("1.7B", "gu",  2048, 12288, 16, "ng=16"),
    ("4B",   "q",   2560, 4096, 20, "ng=20"),
    ("4B",   "o",   4096, 2560, 32, "ng=32"),
    ("4B",   "kv",  2560, 2048, 20, "ng=20"),
    ("4B",   "dn",  9728, 2560, 76, "ng=76 !C.6 gate"),
    ("4B",   "gu",  2560, 19456, 20, "ng=20"),
    # n_g=32 (C.6 gate boundary)
    ("8B",   "q",   4096, 4096, 32, "ng=32 !C.6"),
    ("8B",   "kv",  4096, 2048, 32, "ng=32 !C.6"),
    ("8B",   "o",   4096, 4096, 32, "ng=32 !C.6"),
    ("8B",   "gu",  4096, 24576, 32, "ng=32 !C.6"),
    ("8B",   "dn",  12288, 4096, 96, "ng=96 !C.6"),
    # 14B n_g=40
    ("14B",  "q",   5120, 5120, 40, "ng=40 !C.6 verified"),
    ("14B",  "kv",  5120, 2048, 40, "ng=40 !C.6 verified"),
    ("14B",  "o",   5120, 5120, 40, "ng=40 !C.6"),
    ("14B",  "gu",  5120, 34816, 40, "ng=40 !C.6 verified"),
    ("14B",  "dn",  17408, 5120, 136, "ng=136 !C.6"),
    # 32B same as 14B proj but d_out from different model
    ("32B",  "kv",  5120, 2048, 40, "ng=40 !C.6"),
    ("32B",  "gu",  5120, 55296, 40, "ng=40 !C.6"),
    ("32B",  "dn",  27648, 5120, 216, "ng=216 !C.6 verified"),
    # 70B larger d_in
    ("70B",  "q",   8192, 8192, 64, "ng=64 !C.6"),
    ("70B",  "kv",  8192, 2048, 64, "ng=64 !C.6"),
    ("70B",  "o",   8192, 8192, 64, "ng=64 !C.6"),
    ("70B",  "gu",  8192, 57344, 64, "ng=64 !C.6"),
    ("70B",  "dn", 28672, 8192, 224, "ng=224 !C.6"),
]


PR("=" * 96)
PR(f"T=512 EXTENDED probe — {len(SHAPES)} shapes × (default / sk=1 / sk=2) measured")
PR("=" * 96)
PR()
PR(f"{'shape':<26} {'n_g':>4} {'default':>9} {'sk=1':>9} {'sk=2':>9} "
   f"{'best':>10} {'sk2 vs sk1':>11} {'c6?':>5}")

rows = []
for model, proj, d_in, d_out, n_g, note in SHAPES:
    b = prep(T, d_in, d_out)
    def _run():
        run(b, d_out, d_in)

    set_sk(None);  us_def = bench_us(_run)
    set_sk(1);     us_sk1 = bench_us(_run)
    set_sk(2);     us_sk2 = bench_us(_run)
    set_sk(None)

    label = f"{model}/{proj} {d_in}→{d_out}"
    # Is C.6 active? (T>=256 && n_g>=32 && n_g%2==0 && sk rule gave 1)
    # We can infer C.6 active by checking if |default - sk=2| < 2% (and different from sk=1)
    c6_active = abs(us_def - us_sk2) / us_sk2 < 0.02 and abs(us_def - us_sk1) / us_sk1 > 0.02
    sk2_vs_sk1 = (us_sk2 - us_sk1) / us_sk1 * 100

    rows.append({"label": label, "n_g": n_g, "us_def": us_def,
                 "us_sk1": us_sk1, "us_sk2": us_sk2,
                 "sk2_vs_sk1": sk2_vs_sk1, "c6_active": c6_active,
                 "n_g": n_g})

    best = min(us_def, us_sk1, us_sk2)
    c6_mark = "YES" if c6_active else "no"
    PR(f"  {label:<24}  {n_g:>3}  {us_def:>8.2f}  {us_sk1:>8.2f}  {us_sk2:>8.2f}  "
       f"{best:>9.2f}  {sk2_vs_sk1:+10.2f}%  {c6_mark:>5}")


PR()
PR("=" * 96)
PR("ANALYSIS")
PR("=" * 96)

# 1. Are there shapes where sk=2 wins but C.6 doesn't fire?
missed = [r for r in rows
          if (not r["c6_active"]) and r["sk2_vs_sk1"] < -3]
PR(f"\n[1] Shapes where sk=2 wins >3% vs sk=1, but C.6 does NOT fire:")
if missed:
    PR(f"  (candidates for widening C.6 gate)")
    for r in sorted(missed, key=lambda x: x["sk2_vs_sk1"]):
        PR(f"    {r['label']:<26} n_g={r['n_g']:<4} sk=2 vs sk=1: {r['sk2_vs_sk1']:+.2f}%")
else:
    PR(f"  (none — C.6 gate already captures all the available sk=2 wins)")

# 2. C.6 active shapes — how much did we win?
fired = [r for r in rows if r["c6_active"]]
PR(f"\n[2] Shapes where C.6 fires ({len(fired)}/{len(rows)}):")
if fired:
    uplifts = [-r["sk2_vs_sk1"] for r in fired if r["sk2_vs_sk1"] < 0]
    PR(f"  median sk=2 uplift vs sk=1: {statistics.median(uplifts):+.2f}%")
    PR(f"  range: {min(uplifts):+.2f}% .. {max(uplifts):+.2f}%")

# 3. Shapes where sk=2 HURTS (would be bad to widen C.6)
hurts = [r for r in rows if r["sk2_vs_sk1"] > 3]
PR(f"\n[3] Shapes where sk=2 REGRESSES >3% vs sk=1:")
if hurts:
    PR(f"  (do NOT widen C.6 to include these)")
    for r in sorted(hurts, key=lambda x: -x["sk2_vs_sk1"]):
        PR(f"    {r['label']:<26} n_g={r['n_g']:<4} sk=2 vs sk=1: {r['sk2_vs_sk1']:+.2f}%")
else:
    PR(f"  (none — sk=2 is always safe or neutral at T=512)")
