"""Compare r66 (pre-C.5) vs r67 (post-C.5) bench.  Isolate C.5 gain."""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R66 = json.loads((ROOT / "logs/r66_path_c/bench.json").read_text())
R67 = json.loads((ROOT / "logs/r67_c5/bench.json").read_text())


def index(bench):
    out = {}
    for r in bench["records"]:
        if r.get("kernel") != "end_to_end":
            continue
        key = (r["model"], r["proj"], r["T"], r["d_in"], r["d_out"])
        out[key] = r
    return out


A = index(R66)
B = index(R67)

print("=" * 90)
print("r66 → r67 (C.5 dispatcher widen) comparison")
print("=" * 90)

speedup66 = []
speedup67 = []
wins66 = 0
wins67 = 0
big_wins67 = 0
deltas = []  # (delta_us_pct, shape_key)

for k in sorted(A.keys()):
    if k not in B:
        continue
    a = A[k]
    b = B[k]
    sp66 = a["cuda_speedup_vs_fp16"]
    sp67 = b["cuda_speedup_vs_fp16"]
    us66 = a["cuda_us"]
    us67 = b["cuda_us"]
    speedup66.append(sp66)
    speedup67.append(sp67)
    if sp66 >= 1.0:
        wins66 += 1
    if sp67 >= 1.0:
        wins67 += 1
    if sp67 >= 2.0:
        big_wins67 += 1
    delta_pct = (us67 - us66) / us66 * 100
    deltas.append((delta_pct, k, us66, us67, sp66, sp67))

# C.5 target shapes
C5_TARGETS = {
    ("Qwen3-8B", "q_proj", 128, 4096, 4096),
    ("Qwen3-8B", "o_proj", 128, 4096, 4096),
    ("Qwen3-4B", "q_proj", 128, 2560, 4096),
    ("Qwen3-4B", "o_proj", 128, 4096, 2560),
}

print()
print(f"Total shapes compared: {len(deltas)}")
print(f"Median speedup:    r66 = {statistics.median(speedup66):.4f}x   "
      f"r67 = {statistics.median(speedup67):.4f}x   "
      f"Δ = {statistics.median(speedup67)-statistics.median(speedup66):+.4f}")
print(f"Mean speedup:      r66 = {statistics.mean(speedup66):.4f}x   "
      f"r67 = {statistics.mean(speedup67):.4f}x   "
      f"Δ = {statistics.mean(speedup67)-statistics.mean(speedup66):+.4f}")
print(f"Wins (sp≥1.0):     r66 = {wins66}/{len(deltas)}    "
      f"r67 = {wins67}/{len(deltas)}   Δ = {wins67-wins66:+d}")
print(f"Big wins (≥2.0×):  r67 = {big_wins67}/{len(deltas)}")
print()

# C.5 target shapes
print("## C.5 target shapes")
print(f"{'shape':<40} {'r66 us':>8} {'r67 us':>8} {'Δ us':>7} {'r66 sp':>7} {'r67 sp':>7}")
for dp, k, us66, us67, sp66, sp67 in deltas:
    if k in C5_TARGETS:
        print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
              f"{us66:>7.2f}  {us67:>7.2f}  {dp:+6.2f}%  {sp66:.2f}x  {sp67:.2f}x")

# Top 10 biggest improvements (most negative delta)
print()
print("## Top 10 improvements (ignoring T=1 which is a different kernel path)")
dt_t1_excl = [x for x in deltas if x[1][2] != 1]
for dp, k, us66, us67, sp66, sp67 in sorted(dt_t1_excl)[:10]:
    tag = "  [C.5]" if k in C5_TARGETS else ""
    print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
          f"{us66:>7.2f}  {us67:>7.2f}  {dp:+6.2f}%  {sp66:.2f}x→{sp67:.2f}x{tag}")

# Top 10 regressions (most positive delta)
print()
print("## Top 10 regressions (including T=1; watch for any >3%)")
for dp, k, us66, us67, sp66, sp67 in sorted(deltas, key=lambda x: -x[0])[:10]:
    tag = "  [C.5]" if k in C5_TARGETS else ""
    print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
          f"{us66:>7.2f}  {us67:>7.2f}  {dp:+6.2f}%  {sp66:.2f}x→{sp67:.2f}x{tag}")

print()
print("## Regressions > 3% (potential concerns)")
concerns = [x for x in deltas if x[0] > 3.0]
if concerns:
    for dp, k, us66, us67, sp66, sp67 in sorted(concerns, key=lambda x: -x[0]):
        print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
              f"{us66:>7.2f}  {us67:>7.2f}  {dp:+6.2f}%  {sp66:.2f}x→{sp67:.2f}x")
    print(f"  total: {len(concerns)} shapes")
else:
    print("  (none — clean upgrade)")

# By T bucket
print()
print("## By T bucket")
print(f"{'T':>4} {'n':>4} {'med r66 sp':>11} {'med r67 sp':>11} {'med Δ us':>10}")
by_t = {}
for dp, k, us66, us67, sp66, sp67 in deltas:
    by_t.setdefault(k[2], []).append((dp, sp66, sp67))
for T in sorted(by_t):
    rows = by_t[T]
    m66 = statistics.median([r[1] for r in rows])
    m67 = statistics.median([r[2] for r in rows])
    md = statistics.median([r[0] for r in rows])
    print(f"{T:>4} {len(rows):>4}  {m66:>10.3f}x  {m67:>10.3f}x  {md:+9.2f}%")
