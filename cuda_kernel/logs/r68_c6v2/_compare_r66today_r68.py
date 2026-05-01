"""Three-way drift-free comparison:
  r66_today : main @ r66 (no C.5, no C.6), same-day bench
  r67       : main @ r67 (C.5 in), cross-day — used only as sanity
  r68_c6v2  : main @ r67 + C.6-v2, same-day (15 min after r66_today)

Since r66_today (2026-05-01 15:18) and r68_c6v2 (2026-05-01 16:29)
are same-day-same-GPU, the delta between them IS the combined
C.5 + C.6-v2 effect + roughly 1h of drift (usually <1% on 4090).
"""
import json
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
R66T = json.loads((ROOT / "logs/r66_today/bench.json").read_text())
R68  = json.loads((ROOT / "logs/r68_c6v2/bench.json").read_text())


def index(bench):
    out = {}
    for r in bench["records"]:
        if r.get("kernel") != "end_to_end":
            continue
        key = (r["model"], r["proj"], r["T"], r["d_in"], r["d_out"])
        out[key] = r
    return out


A = index(R66T)
B = index(R68)

# C.5 targets
C5_TARGETS = {
    ("Qwen3-8B", "q_proj", 128, 4096, 4096),
    ("Qwen3-8B", "o_proj", 128, 4096, 4096),
    ("Qwen3-4B", "q_proj", 128, 2560, 4096),
    ("Qwen3-4B", "o_proj", 128, 4096, 2560),
}

# C.6-v2 targets (the 8 shapes where v2 gate activates)
C6_TARGETS = {
    ("Qwen3-14B",  "q_proj",      512, 5120,  5120),
    ("Qwen3-14B",  "kv_proj",     512, 5120,  2048),
    ("Qwen3-14B",  "o_proj",      512, 5120,  5120),
    ("Qwen2.5-32B","kv_proj",     512, 5120,  2048),
    ("LLaMA3-70B", "kv_proj",     512, 8192,  2048),
    ("Qwen3-14B",  "down_proj",   512, 17408, 5120),
    ("Qwen2.5-32B","down_proj",   512, 27648, 5120),
    ("LLaMA3-70B", "down_proj",   512, 28672, 8192),
    ("Qwen3-1.7B", "down_proj",   512, 6144,  2048),
}


print("=" * 96)
print("r66_today → r68_c6v2 (C.5 + C.6-v2 combined, same-day)")
print("=" * 96)

sp66, sp68 = [], []
wins66 = wins68 = bigw68 = 0
deltas = []
for k in sorted(A.keys()):
    if k not in B:
        continue
    a, b = A[k], B[k]
    sp66.append(a["cuda_speedup_vs_fp16"])
    sp68.append(b["cuda_speedup_vs_fp16"])
    if a["cuda_speedup_vs_fp16"] >= 1.0: wins66 += 1
    if b["cuda_speedup_vs_fp16"] >= 1.0: wins68 += 1
    if b["cuda_speedup_vs_fp16"] >= 2.0: bigw68 += 1
    dp = (b["cuda_us"] - a["cuda_us"]) / a["cuda_us"] * 100
    deltas.append((dp, k, a["cuda_us"], b["cuda_us"],
                   a["cuda_speedup_vs_fp16"], b["cuda_speedup_vs_fp16"]))

print(f"Total shapes: {len(deltas)}")
print(f"Median sp:  r66_today = {statistics.median(sp66):.4f}x   "
      f"r68 = {statistics.median(sp68):.4f}x   "
      f"Δ = {statistics.median(sp68)-statistics.median(sp66):+.4f}")
print(f"Mean sp:    r66_today = {statistics.mean(sp66):.4f}x   "
      f"r68 = {statistics.mean(sp68):.4f}x   "
      f"Δ = {statistics.mean(sp68)-statistics.mean(sp66):+.4f}")
print(f"Wins ≥1.0×: r66_today = {wins66}/{len(deltas)}    r68 = {wins68}/{len(deltas)}  "
      f"Δ = {wins68-wins66:+d}")
print(f"Big wins ≥2.0×: r68 = {bigw68}/{len(deltas)}")

print()
print("## C.5 target shapes (4)")
print(f"{'shape':<40} {'r66t us':>8} {'r68 us':>8} {'Δ':>8}")
for dp, k, us66, us68, s66, s68 in deltas:
    if k in C5_TARGETS:
        print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
              f"{us66:>7.2f}  {us68:>7.2f}  {dp:+7.2f}%")

print()
print("## C.6-v2 target shapes (9)")
print(f"{'shape':<40} {'r66t us':>8} {'r68 us':>8} {'Δ':>8}")
for dp, k, us66, us68, s66, s68 in deltas:
    if k in C6_TARGETS:
        print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
              f"{us66:>7.2f}  {us68:>7.2f}  {dp:+7.2f}%")

print()
print("## Top 10 regressions (watch for >3%)")
for dp, k, us66, us68, s66, s68 in sorted(deltas, key=lambda x: -x[0])[:10]:
    tag = ""
    if k in C5_TARGETS: tag = "  [C.5]"
    if k in C6_TARGETS: tag = "  [C.6]"
    print(f"  {k[0]}/{k[1]} T={k[2]} {k[3]}→{k[4]:<6}  "
          f"{us66:>7.2f}  {us68:>7.2f}  {dp:+7.2f}%  {s66:.2f}x→{s68:.2f}x{tag}")

concerns = [x for x in deltas if x[0] > 3.0]
print(f"\nRegressions > 3%: {len(concerns)}")

# By T bucket
print()
print("## By T bucket")
print(f"{'T':>4} {'n':>4} {'med r66t':>9} {'med r68':>9} {'Δ us':>9}")
by_t = defaultdict(list)
for dp, k, us66, us68, s66, s68 in deltas:
    by_t[k[2]].append((dp, s66, s68))
for T in sorted(by_t):
    rows = by_t[T]
    m66 = statistics.median([r[1] for r in rows])
    m68 = statistics.median([r[2] for r in rows])
    md = statistics.median([r[0] for r in rows])
    print(f"{T:>4} {len(rows):>4} {m66:>8.3f}x {m68:>8.3f}x {md:+9.2f}%")
