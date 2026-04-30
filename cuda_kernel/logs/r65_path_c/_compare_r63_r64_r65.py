"""Three-way comparison: r63_combined (baseline) → r64 (C.1+C.2) → r65 (+C.3).

Shows the cumulative effect of Path C dispatcher changes.
Focused on real shapes where we targeted a fix, plus the
macro summary (median speedup, big wins count, peak).
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # cuda_kernel/
R63 = ROOT / "logs/r63_combined/bench.json"
R64 = ROOT / "logs/r64_path_c/bench.json"
R65 = ROOT / "logs/r65_path_c/bench.json"

def load(p):
    d = json.loads(p.read_text())
    return {(r["model"], r["proj"], r["T"]): r
            for r in d["records"] if r.get("kernel") == "end_to_end"}

r63 = load(R63); r64 = load(R64); r65 = load(R65)
keys = sorted(set(r63) & set(r64) & set(r65))

def sp(r):
    return r.get("cuda_speedup_vs_fp16", r["fp16_us"]/r["cuda_us"] if r["cuda_us"] else 0)

# §0 Macro summary
print(f"# Three-way comparison (N = {len(keys)} shapes)")
print()
for label, src in [("r63 (baseline)", r63), ("r64 (C.1+C.2)", r64), ("r65 (+C.3)", r65)]:
    vals = [sp(src[k]) for k in keys]
    median = statistics.median(vals)
    mean = statistics.mean(vals)
    wins = sum(1 for v in vals if v > 1.0)
    big = sum(1 for v in vals if v > 2.0)
    peak = max(vals)
    peak_k = max(keys, key=lambda k: sp(src[k]))
    print(f"  {label:<18} median {median:.3f}×  mean {mean:.3f}×  "
          f"wins>1× {wins}/{len(keys)}  big>2× {big}  peak {peak:.2f}× ({peak_k[0]} {peak_k[1]} T={peak_k[2]})")

# §1 Targeted shapes: full trajectory r63 → r64 → r65
print()
print("## Targeted shape trajectory")
targets = [
    ("Qwen3-14B",  "gate_up_proj", 128, "C.3 target"),
    ("Qwen3-14B",  "gate_up_proj",  32, "C.2b target"),
    ("Qwen3-4B",   "gate_up_proj",  32, "C.2b target"),
    ("Qwen3-8B",   "gate_up_proj",  32, "C.2b target"),
    ("Qwen3-8B",   "gate_up_proj", 128, "C.3 off-target (must not regress)"),
    ("Qwen3-4B",   "gate_up_proj", 128, "C.3 off-target (must not regress)"),
    ("Qwen3-1.7B", "down_proj",    128, "C.1 target"),
    ("Qwen3-14B",  "q_proj",       128, "C.1 target"),
    ("Qwen3-14B",  "o_proj",       128, "C.1 target"),
]
print(f"  {'shape':<32} {'T':>4}  {'r63':>8} {'r64':>8} {'r65':>8}  "
      f"{'sp r63':>7} {'sp r64':>7} {'sp r65':>7}  note")
for m, p, T, note in targets:
    k = (m, p, T)
    if k not in r63 or k not in r64 or k not in r65:
        continue
    a, b, c = r63[k], r64[k], r65[k]
    print(f"  {m:<12} {p:<18} {T:>4}  "
          f"{a['cuda_us']:>7.2f}  {b['cuda_us']:>7.2f}  {c['cuda_us']:>7.2f}  "
          f"{sp(a):>6.2f}× {sp(b):>6.2f}× {sp(c):>6.2f}×  {note}")

# §2 r64 → r65 delta (isolates C.3's impact)
print()
print("## Biggest r64 → r65 changes (isolates C.3 impact)")
deltas = []
for k in keys:
    s64, s65 = sp(r64[k]), sp(r65[k])
    cd = (r65[k]["cuda_us"] - r64[k]["cuda_us"]) / r64[k]["cuda_us"] * 100
    deltas.append((k, s64, s65, s65-s64, cd))

print("  Top 10 improvements (r65 > r64):")
for k, s64, s65, d, cd in sorted(deltas, key=lambda x: -x[3])[:10]:
    print(f"    {k[0]:<12} {k[1]:<18} T={k[2]:>3}  "
          f"sp {s64:.2f}→{s65:.2f}× ({d:+.3f})  cuda {cd:+.1f}%")
print()
print("  Top 10 regressions (r65 < r64):")
for k, s64, s65, d, cd in sorted(deltas, key=lambda x: x[3])[:10]:
    print(f"    {k[0]:<12} {k[1]:<18} T={k[2]:>3}  "
          f"sp {s64:.2f}→{s65:.2f}× ({d:+.3f})  cuda {cd:+.1f}%")

# §3 Cumulative r63 → r65 delta
print()
print("## Cumulative r63 → r65 (big improvements)")
cdeltas = []
for k in keys:
    s63, s65 = sp(r63[k]), sp(r65[k])
    cdeltas.append((k, s63, s65, s65-s63))
for k, s63, s65, d in sorted(cdeltas, key=lambda x: -x[3])[:10]:
    print(f"    {k[0]:<12} {k[1]:<18} T={k[2]:>3}  sp {s63:.2f}×→{s65:.2f}×  ({d:+.3f})")
