"""r66_today vs r67: drift-free comparison.

Both benches ran 2026-05-01 within 10 minutes of each other on the
same GPU (same clock state).  This isolates C.5 effect from drift.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R66_TODAY = json.loads((ROOT / "logs/r66_today/bench.json").read_text())
R67       = json.loads((ROOT / "logs/r67_c5/bench.json").read_text())


def index(bench):
    out = {}
    for r in bench["records"]:
        if r.get("kernel") != "end_to_end":
            continue
        key = (r["model"], r["proj"], r["T"], r["d_in"], r["d_out"])
        out[key] = r
    return out


A = index(R66_TODAY)
B = index(R67)

C5_TARGETS = {
    ("Qwen3-8B", "q_proj", 128, 4096, 4096),
    ("Qwen3-8B", "o_proj", 128, 4096, 4096),
    ("Qwen3-4B", "q_proj", 128, 2560, 4096),
    ("Qwen3-4B", "o_proj", 128, 4096, 2560),
}

print("=" * 90)
print("r66_today vs r67 (C.5) — drift-free same-GPU comparison")
print("=" * 90)

sp66, sp67 = [], []
wins66, wins67 = 0, 0
deltas = []
for k in sorted(A.keys()):
    if k not in B:
        continue
    a, b = A[k], B[k]
    sp66.append(a["cuda_speedup_vs_fp16"])
    sp67.append(b["cuda_speedup_vs_fp16"])
    if a["cuda_speedup_vs_fp16"] >= 1.0: wins66 += 1
    if b["cuda_speedup_vs_fp16"] >= 1.0: wins67 += 1
    dp = (b["cuda_us"] - a["cuda_us"]) / a["cuda_us"] * 100
    deltas.append((dp, k, a["cuda_us"], b["cuda_us"],
                   a["cuda_speedup_vs_fp16"], b["cuda_speedup_vs_fp16"]))

print(f"Total: {len(deltas)}")
print(f"Median sp:  r66_today = {statistics.median(sp66):.4f}x   "
      f"r67 = {statistics.median(sp67):.4f}x   "
      f"Δ = {statistics.median(sp67)-statistics.median(sp66):+.4f}")
print(f"Mean sp:    r66_today = {statistics.mean(sp66):.4f}x   "
      f"r67 = {statistics.mean(sp67):.4f}x   "
      f"Δ = {statistics.mean(sp67)-statistics.mean(sp66):+.4f}")
print(f"Wins ≥1.0×: r66_today = {wins66}/{len(deltas)}    "
      f"r67 = {wins67}/{len(deltas)}    Δ = {wins67-wins66:+d}")

print()
print("## C.5 target shapes")
print(f"{'shape':<44} {'r66t us':>8} {'r67 us':>8} {'Δ us':>8} {'r66 sp':>7} {'r67 sp':>7}")
c5_gains = []
for dp, k, us66, us67, s66, s67 in deltas:
    if k in C5_TARGETS:
        c5_gains.append(-dp)
        print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
              f"{us66:>7.2f}  {us67:>7.2f}  {dp:+7.2f}%  {s66:.2f}x  {s67:.2f}x")
if c5_gains:
    print(f"  C.5 median uplift: {statistics.median(c5_gains):+.2f}%  "
          f"mean: {statistics.mean(c5_gains):+.2f}%")

print()
print("## Top 10 improvements (r67 faster)")
for dp, k, us66, us67, s66, s67 in sorted(deltas)[:10]:
    tag = "  [C.5]" if k in C5_TARGETS else ""
    print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
          f"{us66:>7.2f}  {us67:>7.2f}  {dp:+6.2f}%{tag}")

print()
print("## Top 10 regressions (r67 slower)")
for dp, k, us66, us67, s66, s67 in sorted(deltas, key=lambda x: -x[0])[:10]:
    tag = "  [C.5]" if k in C5_TARGETS else ""
    print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
          f"{us66:>7.2f}  {us67:>7.2f}  {dp:+6.2f}%{tag}")

concerns = [x for x in deltas if x[0] > 3.0]
print()
print(f"## Regressions > 3%: {len(concerns)}")
for dp, k, us66, us67, s66, s67 in sorted(concerns, key=lambda x: -x[0]):
    tag = "  [C.5]" if k in C5_TARGETS else ""
    print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
          f"{us66:>7.2f}  {us67:>7.2f}  {dp:+6.2f}%{tag}")

print()
print("## By T bucket")
print(f"{'T':>4} {'n':>4} {'med r66t sp':>11} {'med r67 sp':>11} {'med Δ us':>10}")
by_t = {}
for dp, k, us66, us67, s66, s67 in deltas:
    by_t.setdefault(k[2], []).append((dp, s66, s67))
for T in sorted(by_t):
    rows = by_t[T]
    m66 = statistics.median([r[1] for r in rows])
    m67 = statistics.median([r[2] for r in rows])
    md = statistics.median([r[0] for r in rows])
    print(f"{T:>4} {len(rows):>4}  {m66:>10.3f}x  {m67:>10.3f}x  {md:+9.2f}%")
