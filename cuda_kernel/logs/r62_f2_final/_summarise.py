import json, statistics
from collections import defaultdict
b = json.load(open('/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r62_f2_final/qwen3_20260430_122555/bench.json'))
recs = [x for x in b['records'] if x['kernel']=='end_to_end']

by_model = defaultdict(list)
for r in recs:
    by_model[r['model']].append(r['cuda_speedup_vs_fp16'])

print(f"{'model':<15} {'N':>3} {'median':>8} {'mean':>8} {'wins':>8} {'max':>7} {'min':>7}")
for m in ['Qwen3-0.6B','Qwen3-1.7B','Qwen3-4B','Qwen3-8B']:
    sp = by_model[m]
    wins = sum(1 for s in sp if s >= 1.0)
    print(f'{m:<15} {len(sp):>3} {statistics.median(sp):>7.2f}x {statistics.mean(sp):>7.2f}x {wins}/{len(sp):>3}  {max(sp):>6.2f}x {min(sp):>6.2f}x')

print()
by_T = defaultdict(list)
for r in recs: by_T[r['T']].append(r['cuda_speedup_vs_fp16'])
print(f"{'T':>4} {'N':>3} {'median':>8} {'mean':>8} {'wins':>5}")
for t in sorted(by_T):
    sp = by_T[t]
    print(f'{t:>4} {len(sp):>3} {statistics.median(sp):>7.2f}x {statistics.mean(sp):>7.2f}x {sum(1 for s in sp if s>=1.0)}/{len(sp)}')
