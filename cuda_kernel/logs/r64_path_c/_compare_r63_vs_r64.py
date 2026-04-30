"""Compare r63_combined (pre-C.1+C.2) vs r64_path_c (post-C.1+C.2)
on the 140-shape overlap.  Answer: did C.1 + C.2 bring real improvements?

Outputs:
  §1 — overall summary (median speedup, #wins)
  §2 — biggest improvements (top 10)
  §3 — biggest regressions (top 10)
  §4 — change buckets: speedup Δ distribution
  §5 — by T aggregation
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # cuda_kernel/
R63 = ROOT / "logs/r63_combined/bench.json"
R64 = ROOT / "logs/r64_path_c/bench/qwen3_20260430_153701/bench.json"

def load_e2e(p):
    data = json.loads(p.read_text())
    return {(r["model"], r["proj"], r["T"]): r
            for r in data["records"] if r.get("kernel") == "end_to_end"}

r63 = load_e2e(R63)
r64 = load_e2e(R64)
shared = sorted(set(r63) & set(r64))

def sp(r):
    return r.get("cuda_speedup_vs_fp16", r["fp16_us"]/r["cuda_us"] if r["cuda_us"] else 0)

rows = []
for k in shared:
    a, b = r63[k], r64[k]
    rows.append({
        "model": k[0], "proj": k[1], "T": k[2],
        "d_in": a["d_in"], "d_out": a["d_out"],
        "fp16_r63": a["fp16_us"], "fp16_r64": b["fp16_us"],
        "cuda_r63": a["cuda_us"], "cuda_r64": b["cuda_us"],
        "sp_r63": sp(a), "sp_r64": sp(b),
        "cuda_delta_pct": (b["cuda_us"] - a["cuda_us"]) / a["cuda_us"] * 100,
        "sp_delta": sp(b) - sp(a),
    })

print(f"# Overlap: {len(rows)} shapes")
print(f"# Source r63: {R63.relative_to(ROOT)}")
print(f"# Source r64: {R64.relative_to(ROOT)}")
print()

# §1 — overall summary
print("## §1 Overall summary")
median_sp_63 = statistics.median(r["sp_r63"] for r in rows)
median_sp_64 = statistics.median(r["sp_r64"] for r in rows)
wins_63 = sum(1 for r in rows if r["sp_r63"] > 1.0)
wins_64 = sum(1 for r in rows if r["sp_r64"] > 1.0)
print(f"  Median speedup:   r63 {median_sp_63:.3f}×  →  r64 {median_sp_64:.3f}×  "
      f"(Δ {(median_sp_64-median_sp_63):+.3f}×, "
      f"{(median_sp_64/median_sp_63-1)*100:+.1f}%)")
print(f"  Wins (sp > 1.0):  r63 {wins_63}/{len(rows)}  →  r64 {wins_64}/{len(rows)}"
      f"  (Δ {wins_64-wins_63:+d})")

# §2 — biggest improvements
print()
print("## §2 Top 10 biggest speedup improvements (r64 vs r63)")
print(f"  {'model':<12} {'proj':<14} {'T':>4} {'d_in':>5} {'d_out':>6}  "
      f"{'sp_r63':>7}  {'sp_r64':>7}  {'Δsp':>7}  {'cuda_Δ%':>7}")
for r in sorted(rows, key=lambda r: -r["sp_delta"])[:10]:
    print(f"  {r['model']:<12} {r['proj']:<14} {r['T']:>4} {r['d_in']:>5} {r['d_out']:>6}  "
          f"{r['sp_r63']:>6.2f}× {r['sp_r64']:>6.2f}× {r['sp_delta']:>+6.2f}× "
          f"{r['cuda_delta_pct']:>+6.1f}%")

# §3 — regressions
print()
print("## §3 Top 10 biggest speedup regressions")
regs = sorted(rows, key=lambda r: r["sp_delta"])[:10]
print(f"  {'model':<12} {'proj':<14} {'T':>4} {'d_in':>5} {'d_out':>6}  "
      f"{'sp_r63':>7}  {'sp_r64':>7}  {'Δsp':>7}  {'cuda_Δ%':>7}  {'fp16_Δ%':>7}")
for r in regs:
    fp16_d = (r["fp16_r64"] - r["fp16_r63"]) / r["fp16_r63"] * 100
    print(f"  {r['model']:<12} {r['proj']:<14} {r['T']:>4} {r['d_in']:>5} {r['d_out']:>6}  "
          f"{r['sp_r63']:>6.2f}× {r['sp_r64']:>6.2f}× {r['sp_delta']:>+6.2f}× "
          f"{r['cuda_delta_pct']:>+6.1f}% {fp16_d:>+6.1f}%")

# §4 — delta distribution
print()
print("## §4 Δspeedup distribution")
buckets = [
    ("big_improve (>+0.10×)",  [r for r in rows if r["sp_delta"] > 0.10]),
    ("mid_improve (+0.03..+0.10)", [r for r in rows if 0.03 < r["sp_delta"] <= 0.10]),
    ("neutral (±0.03)",        [r for r in rows if abs(r["sp_delta"]) <= 0.03]),
    ("mid_regress (-0.03..-0.10)", [r for r in rows if -0.10 <= r["sp_delta"] < -0.03]),
    ("big_regress (<-0.10×)",  [r for r in rows if r["sp_delta"] < -0.10]),
]
for name, bs in buckets:
    print(f"  {name:<32} {len(bs):>3} / {len(rows)}")

# §5 — by T
print()
print("## §5 Aggregated by T (cuda_us median change — kernel-only, not affected by fp16 drift)")
by_T = defaultdict(list)
for r in rows:
    by_T[r["T"]].append(r)
print(f"  {'T':>4}  {'n':>3}  {'cuda Δ% median':>16}  {'sp r63':>7}  {'sp r64':>7}  {'Δ wins':>8}")
for T in sorted(by_T):
    vs = by_T[T]
    cd = statistics.median(r["cuda_delta_pct"] for r in vs)
    s63 = statistics.median(r["sp_r63"] for r in vs)
    s64 = statistics.median(r["sp_r64"] for r in vs)
    w63 = sum(1 for r in vs if r["sp_r63"] > 1.0)
    w64 = sum(1 for r in vs if r["sp_r64"] > 1.0)
    print(f"  {T:>4}  {len(vs):>3}  {cd:>+14.2f}%   {s63:>6.2f}× {s64:>6.2f}× "
          f"{w64-w63:>+7d}")

# §6 — Targeted C.2b fix verification
print()
print("## §6 Targeted C.1/C.2 wins verification (shapes we targeted)")
targets = [
    ("Qwen3-14B", "gate_up_proj", 32, "C.2b: was +34% oversight"),
    ("Qwen3-4B",  "gate_up_proj", 32, "C.2b: was +17% oversight"),
    ("Qwen3-8B",  "gate_up_proj", 32, "C.2b: was +6.6% oversight"),
    ("Qwen3-1.7B","down_proj",   128, "C.1: was +10% gain"),
    ("Qwen3-14B", "q_proj",      128, "C.1: was +4% gain"),
    ("Qwen3-14B", "o_proj",      128, "C.1: was +4% gain"),
]
for m, p, T, note in targets:
    k = (m, p, T)
    if k in r63 and k in r64:
        a, b = r63[k], r64[k]
        print(f"  {m:<12} {p:<14} T={T:>3}  cuda {a['cuda_us']:>7.2f}→{b['cuda_us']:>7.2f}us "
              f"({(b['cuda_us']-a['cuda_us'])/a['cuda_us']*100:+.1f}%)  "
              f"sp {sp(a):.2f}→{sp(b):.2f}×  — {note}")
    else:
        print(f"  {m} {p} T={T}  MISSING")
