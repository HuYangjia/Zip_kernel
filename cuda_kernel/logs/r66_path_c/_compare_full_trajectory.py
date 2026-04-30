"""Four-way comparison: r63 → r64 (C.1+C.2) → r65 (+C.3) → r66 (+C.4).

C.4 changes ONLY affect the dispatcher for T ∈ {48, 64, 96}, which
are NOT in the 140-shape bench (T ∈ {1, 32, 128, 512}).  So r66
should match r65 at the macro level; we verify this + confirm no
regression from the C.2b T-bound tightening on T=32 shapes.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHES = {
    "r63": ROOT / "logs/r63_combined/bench.json",
    "r64": ROOT / "logs/r64_path_c/bench.json",
    "r65": ROOT / "logs/r65_path_c/bench.json",
    "r66": ROOT / "logs/r66_path_c/bench.json",
}

def load(p):
    return {(r["model"], r["proj"], r["T"]): r
            for r in json.loads(p.read_text())["records"]
            if r.get("kernel") == "end_to_end"}

srcs = {k: load(v) for k, v in BENCHES.items()}
keys = sorted(set.intersection(*(set(s) for s in srcs.values())))

def sp(r):
    return r.get("cuda_speedup_vs_fp16", r["fp16_us"] / r["cuda_us"] if r["cuda_us"] else 0)

print(f"# Four-way comparison  N = {len(keys)} shapes overlap\n")
for lbl, src in srcs.items():
    vals = [sp(src[k]) for k in keys]
    med = statistics.median(vals)
    mean = statistics.mean(vals)
    wins = sum(1 for v in vals if v > 1.0)
    big = sum(1 for v in vals if v > 2.0)
    peak = max(vals)
    print(f"  {lbl:<5}  median {med:.3f}×  mean {mean:.3f}×  "
          f"wins>1× {wins:>3}/{len(keys)}  big>2× {big:>2}  peak {peak:.2f}×")

# Paired deltas
print()
print("## Paired stage-by-stage deltas (median):")
labels = ["r63", "r64", "r65", "r66"]
for i in range(1, 4):
    a, b = labels[i - 1], labels[i]
    med_a = statistics.median(sp(srcs[a][k]) for k in keys)
    med_b = statistics.median(sp(srcs[b][k]) for k in keys)
    print(f"  {a} → {b}:  {med_a:.3f}×  →  {med_b:.3f}×  ({med_b - med_a:+.3f})")

# Sanity: r65 → r66 should be ≈ 0 (C.4 doesn't touch T∈{1,32,128,512})
print()
print("## Sanity: r65 → r66 by T (C.4 should not affect these T)")
from collections import defaultdict
by_T = defaultdict(list)
for k in keys:
    by_T[k[2]].append(k)
print(f"  {'T':>4}  {'n':>3}  {'Δmedian sp':>10}")
for T in sorted(by_T):
    ks = by_T[T]
    d = statistics.median(sp(srcs["r66"][k]) for k in ks) - \
        statistics.median(sp(srcs["r65"][k]) for k in ks)
    print(f"  {T:>4}  {len(ks):>3}  {d:>+10.4f}×")

# Biggest r65 → r66 changes
deltas = [(k, sp(srcs["r66"][k]) - sp(srcs["r65"][k])) for k in keys]
print()
print("## Biggest r65 → r66 changes (should be noise only)")
for k, d in sorted(deltas, key=lambda x: -abs(x[1]))[:6]:
    s65, s66 = sp(srcs["r65"][k]), sp(srcs["r66"][k])
    print(f"  {k[0]:<12} {k[1]:<14} T={k[2]:>3}  "
          f"sp {s65:.2f}→{s66:.2f}× ({d:+.3f})")

# Final cumulative r63 → r66 summary
print()
print("## Cumulative top improvements (r63 → r66):")
cd = [(k, sp(srcs["r66"][k]) - sp(srcs["r63"][k])) for k in keys]
for k, d in sorted(cd, key=lambda x: -x[1])[:10]:
    s63, s66 = sp(srcs["r63"][k]), sp(srcs["r66"][k])
    print(f"  {k[0]:<12} {k[1]:<14} T={k[2]:>3}  sp {s63:.2f}×→{s66:.2f}×  ({d:+.3f})")
