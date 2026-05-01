"""Compare the r68 prefill bench vs historical bench at T=1024 overlap."""
import json
from collections import Counter

HIST = "/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/qwen3_20260428_111515/bench.json"
PREF = "/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r68_prefill_survey/bench.json"

d_hist = json.load(open(HIST))
d_pref = json.load(open(PREF))

e2e_hist = [r for r in d_hist['records']
            if r.get('kernel') == 'end_to_end' and r['T'] == 1024]
print(f"Historical T=1024 (2026-04-28) shapes: {len(e2e_hist)}")
c = Counter(r['model'] for r in e2e_hist)
for m, n in sorted(c.items()):
    print(f"  {m}: {n} shapes")

pref_T1024 = {(r['model'], r['proj']): r for r in d_pref['records']
              if r.get('kernel') == 'end_to_end' and r['T'] == 1024}

print()
print("T=1024 comparison (historical 2026-04-28 vs r68_prefill 2026-05-01):")
print(f"  {'model':<14} {'proj':<14} "
      f"{'cuda_hist':>10} {'cuda_r68':>10} {'Δ%':>7}  "
      f"{'fp16_hist':>10} {'fp16_r68':>10} {'Δ%':>7}  "
      f"{'sp_hist':>8} {'sp_r68':>8}")
for r in sorted(e2e_hist, key=lambda x: (x['model'], x['proj'])):
    key = (r['model'], r['proj'])
    if key in pref_T1024:
        r2 = pref_T1024[key]
        du = (r2['cuda_us'] - r['cuda_us']) / r['cuda_us'] * 100
        df = (r2['fp16_us'] - r['fp16_us']) / r['fp16_us'] * 100
        print(f"  {r['model']:<14} {r['proj']:<14} "
              f"{r['cuda_us']:>9.1f} {r2['cuda_us']:>9.1f} {du:+6.1f}%  "
              f"{r['fp16_us']:>9.1f} {r2['fp16_us']:>9.1f} {df:+6.1f}%  "
              f"{r['cuda_speedup_vs_fp16']:>7.3f}x "
              f"{r2['cuda_speedup_vs_fp16']:>7.3f}x")
