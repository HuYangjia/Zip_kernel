"""r63 aggregate: combine r62_f2_final (0.6B/1.7B/4B/8B) + r63 (14B/32B/70B).

Emits:
  - by-model summary
  - by-T summary
  - scaling trend (median speedup vs model size)
"""
import json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs')

BENCH_FILES = [
    ROOT / 'r62_f2_final/qwen3_20260430_122555/bench.json',
    ROOT / 'r63_large_models/qwen3_20260430_124225/bench.json',
]

# Model params for scaling trend
MODEL_PARAMS = {
    "Qwen3-0.6B": 0.6,
    "Qwen3-1.7B": 1.7,
    "Qwen3-4B":   4.0,
    "Qwen3-8B":   8.0,
    "Qwen3-14B":  14.0,
    "Qwen2.5-32B": 32.0,
    "LLaMA3-70B":  70.0,
}

# Collect all e2e records
all_recs = []
for bf in BENCH_FILES:
    data = json.loads(bf.read_text())
    for r in data['records']:
        if r.get('kernel') == 'end_to_end':
            all_recs.append(r)

print(f"# Combined {len(all_recs)} shapes across {len(MODEL_PARAMS)} models")
print()

# By model
by_m = defaultdict(list)
for r in all_recs:
    by_m[r['model']].append(r['cuda_speedup_vs_fp16'])

print(f"{'model':<14} {'params':>8} {'N':>3} {'median':>8} {'mean':>8} {'wins':>8} {'peak':>7}")
for m in MODEL_PARAMS:
    sp = by_m.get(m, [])
    if not sp: continue
    wins = sum(1 for s in sp if s >= 1.0)
    print(f"{m:<14} {MODEL_PARAMS[m]:>7.1f}B {len(sp):>3} "
          f"{statistics.median(sp):>7.2f}x {statistics.mean(sp):>7.2f}x "
          f"{wins:>3}/{len(sp):>3}  {max(sp):>5.2f}x")

print()
print("# By T (across all 7 models)")
by_t = defaultdict(list)
for r in all_recs:
    by_t[r['T']].append(r['cuda_speedup_vs_fp16'])
print(f"{'T':>4} {'N':>3} {'median':>8} {'mean':>8} {'wins':>8}")
for t in sorted(by_t):
    sp = by_t[t]
    wins = sum(1 for s in sp if s >= 1.0)
    print(f"{t:>4} {len(sp):>3} {statistics.median(sp):>7.2f}x {statistics.mean(sp):>7.2f}x {wins}/{len(sp)}")

# Overall
print()
print("# Overall (140 shapes)")
all_sp = [r['cuda_speedup_vs_fp16'] for r in all_recs]
print(f"  N={len(all_sp)}  median={statistics.median(all_sp):.3f}x  mean={statistics.mean(all_sp):.3f}x")
print(f"  wins>=1.00x: {sum(1 for s in all_sp if s>=1.00)}/{len(all_sp)}")
print(f"  wins>=1.10x: {sum(1 for s in all_sp if s>=1.10)}/{len(all_sp)}")
print(f"  wins>=2.00x: {sum(1 for s in all_sp if s>=2.00)}/{len(all_sp)}")
print(f"  losses<0.90x: {sum(1 for s in all_sp if s<0.90)}/{len(all_sp)}")
print(f"  max={max(all_sp):.3f}x  min={min(all_sp):.3f}x")

# Big-model subset (14B+): shows the high-M regime
print()
print("# Large-model subset (14B + 32B + 70B, 60 shapes) — the real production target")
big_sp = [r['cuda_speedup_vs_fp16'] for r in all_recs
         if r['model'] in ('Qwen3-14B', 'Qwen2.5-32B', 'LLaMA3-70B')]
wins_big = sum(1 for s in big_sp if s >= 1.0)
print(f"  N={len(big_sp)}  median={statistics.median(big_sp):.3f}x  mean={statistics.mean(big_sp):.3f}x")
print(f"  wins: {wins_big}/{len(big_sp)}")
print(f"  peak: {max(big_sp):.3f}x")
