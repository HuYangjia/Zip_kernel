"""Compare r47 (04-28 tight-loop FP16) vs r63 (04-30 cold-cache FP16)
on the overlap shapes for small Qwen3 models (0.6B + 1.7B).

Goal: confirm that the small-model regression is ENTIRELY explained by the
FP16 baseline becoming cold-cache (slower baseline at R47 -> faster baseline
at r63 -> speedup drops even though our INT4 kernel time is unchanged or
marginally better).
"""
import json
from pathlib import Path

R47 = Path('/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/qwen3_20260428_111515/bench.json')
R63 = Path('/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r63_combined/bench.json')

def key(r): return (r['model'], r['proj'], r['T'])

def e2e_map(path):
    data = json.loads(path.read_text())
    return {key(r): r for r in data['records']
            if r.get('kernel') == 'end_to_end'}

r47 = e2e_map(R47)
r63 = e2e_map(R63)

# Focus on Qwen3-0.6B and Qwen3-1.7B shapes that exist in both
shared = sorted(set(r47).intersection(r63),
                key=lambda k: (k[0], k[1], k[2]))

print(f"Overlap shapes: {len(shared)}")
print(f"R47 Ts: {sorted({k[2] for k in r47})}")
print(f"R63 Ts: {sorted({k[2] for k in r63})}")

print()
print(f"{'model':<12} {'proj':<14} {'T':>4} "
      f"{'R47_fp16':>8} {'R63_fp16':>8} {'fp16_Δ%':>8} "
      f"{'R47_cuda':>8} {'R63_cuda':>8} {'cuda_Δ%':>8} "
      f"{'R47_sp':>7} {'R63_sp':>7} {'sp_Δ%':>7}")
for k in shared:
    if k[0] not in ('Qwen3-0.6B', 'Qwen3-1.7B'): continue
    a, b = r47[k], r63[k]
    fa, fb = a['fp16_us'], b['fp16_us']
    ca, cb = a['cuda_us'], b['cuda_us']
    sa = a.get('cuda_speedup_vs_fp16', fa / ca if ca else 0)
    sb = b.get('cuda_speedup_vs_fp16', fb / cb if cb else 0)
    print(f"{k[0]:<12} {k[1]:<14} {k[2]:>4} "
          f"{fa:>7.2f} {fb:>7.2f} {(fb-fa)/fa*100:>+7.1f}% "
          f"{ca:>7.2f} {cb:>7.2f} {(cb-ca)/ca*100:>+7.1f}% "
          f"{sa:>6.2f}x {sb:>6.2f}x {(sb-sa)/sa*100:>+6.1f}%")

# Summary: how much did FP16 change vs how much did cuda change?
print()
small = [k for k in shared if k[0] in ('Qwen3-0.6B', 'Qwen3-1.7B')]
fp16_drops = [(r63[k]['fp16_us'] - r47[k]['fp16_us']) / r47[k]['fp16_us'] * 100
              for k in small]
cuda_drops = [(r63[k]['cuda_us'] - r47[k]['cuda_us']) / r47[k]['cuda_us'] * 100
              for k in small]

import statistics
print(f"Small-model overlap: {len(small)} shapes")
print(f"  FP16 time change: median {statistics.median(fp16_drops):+.1f}%, "
      f"mean {statistics.mean(fp16_drops):+.1f}%")
print(f"  CUDA time change: median {statistics.median(cuda_drops):+.1f}%, "
      f"mean {statistics.mean(cuda_drops):+.1f}%")

# Breakdown by proj
print()
from collections import defaultdict
byp = defaultdict(list)
for k in small:
    byp[k[1]].append((
        (r63[k]['fp16_us'] - r47[k]['fp16_us']) / r47[k]['fp16_us'] * 100,
        (r63[k]['cuda_us'] - r47[k]['cuda_us']) / r47[k]['cuda_us'] * 100))
print(f"{'proj':<14} {'n':>3} {'FP16 Δ median':>14} {'CUDA Δ median':>14}")
for p, vs in byp.items():
    fp = statistics.median([v[0] for v in vs])
    cu = statistics.median([v[1] for v in vs])
    print(f"{p:<14} {len(vs):>3} {fp:>+13.1f}% {cu:>+13.1f}%")
