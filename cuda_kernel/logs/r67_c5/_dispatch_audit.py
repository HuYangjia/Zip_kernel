"""Dispatch audit: which kernel path each shape in the r67 bench lands on,
and how well each path performs.

Paths:
  P1. T=1 + d_in <= 20480       → fused_gemv_decode (dedicated GEMV)
  P2. T=1 + d_in > 20480        → fused_dense_sparse_mma_int4 with T=1 special
  P3. T>=2                      → fused_dense_sparse_mma_int4 (INT4 MMA)
       Sub-paths within P3 (inside the kernel's dispatcher):
         P3a. T<=8            → kBn=8
         P3b. T in [16, 64]   → kBn=32/16/8 wave-aware
         P3c. T>=64 + large   → kBn=64
         plus kBm=64 vs 128 gating (R41/R44/R52/C.3/C.5 rules)
         plus group_cache on/off (r61 Stage F)
         plus split_k (r62 F2)
"""
import json
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
R67 = json.loads((ROOT / "logs/r67_c5/bench.json").read_text())


def dispatch_path(T, d_in, d_out):
    """Classify which dispatch branch handles this shape."""
    if T == 1:
        if d_in <= 20480:
            return "P1: gemv_decode"
        else:
            return "P2: mma_int4 (T=1 oversize)"
    # T >= 2 → mma_int4.  Further classify by T sub-bucket (matches
    # the dispatcher's pick() in fused_dense_sparse_mma_int4.cu).
    if T <= 8:
        return "P3a: mma_int4 T<=8 (kBn=8)"
    if T <= 32:
        return "P3b: mma_int4 T=16..32"
    if T <= 64:
        return "P3c: mma_int4 T=48..64 mid"
    if T <= 128:
        return "P3d: mma_int4 T=128"
    return "P3e: mma_int4 T>=256 (compute-bound)"


# Collect end_to_end records
records = []
for r in R67["records"]:
    if r.get("kernel") != "end_to_end":
        continue
    T, d_in, d_out = r["T"], r["d_in"], r["d_out"]
    path = dispatch_path(T, d_in, d_out)
    records.append({
        "path": path,
        "model": r["model"], "proj": r["proj"],
        "T": T, "d_in": d_in, "d_out": d_out,
        "fp16_us": r["fp16_us"], "cuda_us": r["cuda_us"],
        "sp": r["cuda_speedup_vs_fp16"],
    })

by_path = defaultdict(list)
for r in records:
    by_path[r["path"]].append(r)


print("=" * 96)
print("r67 (main, post-C.5) — dispatch audit")
print("=" * 96)
print()
print(f"{'path':<40} {'count':>5} {'median sp':>10} {'p25 sp':>8} {'p75 sp':>8} "
      f"{'wins≥1':>6} {'bigwin≥2':>8}")
for path in sorted(by_path.keys()):
    rs = by_path[path]
    sps = sorted(r["sp"] for r in rs)
    med = statistics.median(sps)
    p25 = sps[len(sps) // 4]
    p75 = sps[3 * len(sps) // 4]
    wins = sum(1 for sp in sps if sp >= 1.0)
    bigw = sum(1 for sp in sps if sp >= 2.0)
    print(f"  {path:<38} {len(rs):>5} {med:>9.3f}x {p25:>7.3f}x {p75:>7.3f}x "
          f"{wins:>5}/{len(rs)} {bigw:>6}/{len(rs)}")

# Per-path detailed sub-report
for path in sorted(by_path.keys()):
    rs = by_path[path]
    print()
    print("=" * 96)
    print(f"{path} — {len(rs)} shapes")
    print("=" * 96)
    # Sort by sp (worst first)
    rs_sorted = sorted(rs, key=lambda r: r["sp"])
    print(f"  {'shape':<42} {'fp16 us':>8} {'cuda us':>8} {'speedup':>8}")
    for r in rs_sorted:
        print(f"  {r['model']}/{r['proj']} T={r['T']} {r['d_in']}→{r['d_out']:<6} "
              f"{r['fp16_us']:>7.1f}  {r['cuda_us']:>7.1f}  {r['sp']:>7.2f}x")

    # Path-level diagnosis
    sps = [r["sp"] for r in rs]
    med = statistics.median(sps)
    losers = [r for r in rs if r["sp"] < 1.0]
    print()
    print(f"  summary: median {med:.2f}x, losers (sp<1.0) = {len(losers)}/{len(rs)}")
    if losers:
        worst = min(losers, key=lambda r: r["sp"])
        print(f"  worst   : {worst['model']}/{worst['proj']} T={worst['T']} "
              f"{worst['d_in']}→{worst['d_out']}  sp={worst['sp']:.2f}x")

# Cross-cut by T bucket (T bucket and path correlate)
print()
print("=" * 96)
print("Cross-cut by T bucket")
print("=" * 96)
by_t = defaultdict(list)
for r in records:
    by_t[r["T"]].append(r)
print(f"{'T':>4} {'n':>4} {'median sp':>10} {'worst sp':>9} {'loser shapes':>13}")
for T in sorted(by_t):
    rs = by_t[T]
    sps = [r["sp"] for r in rs]
    med = statistics.median(sps)
    worst = min(sps)
    losers = sum(1 for sp in sps if sp < 1.0)
    print(f"{T:>4} {len(rs):>4} {med:>9.3f}x {worst:>8.3f}x {losers:>6}/{len(rs)}")
