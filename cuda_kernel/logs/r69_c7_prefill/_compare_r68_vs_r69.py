"""r68 vs r69 C.7 impact comparison.

Loads both bench.json files (140 shapes each at T in {1024,2048,4096,8192})
and prints:
  §A Global: median speedup before/after, win count before/after
  §B Per-model breakdown
  §C Target family (14B gu): per-T shape-level deltas
  §D Guard families (32B gu, 70B gu, 8B gu, etc.): per-T deltas
  §E Full regression check: any shape that got >3% slower after C.7?
"""
import json
import statistics
from collections import defaultdict

R68 = "/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r68_prefill_survey/bench.json"
R69 = "/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r69_c7_prefill/bench.json"

d68 = json.load(open(R68))
d69 = json.load(open(R69))

def index(data):
    return {(r['model'], r['proj'], r['T'], r['d_in'], r['d_out']): r
            for r in data['records']
            if r.get('kernel') == 'end_to_end' and r['model'] != 'Qwen3-0.6B'}

i68 = index(d68); i69 = index(d69)
common = set(i68) & set(i69)
print(f"Common (model, proj, T, d_in, d_out) shapes: {len(common)}")

# ========================================================
# §A Global summary
# ========================================================
sps_68 = [i68[k]['cuda_speedup_vs_fp16'] for k in common]
sps_69 = [i69[k]['cuda_speedup_vs_fp16'] for k in common]

wins_68 = sum(1 for s in sps_68 if s >= 1.0)
wins_69 = sum(1 for s in sps_69 if s >= 1.0)
med_68  = statistics.median(sps_68)
med_69  = statistics.median(sps_69)
mean_68 = statistics.mean(sps_68)
mean_69 = statistics.mean(sps_69)

print()
print("§A  Global summary (r68 → r69 C.7)")
print(f"  median speedup:  {med_68:.4f}×  →  {med_69:.4f}×   Δ = {(med_69-med_68):+.4f}")
print(f"  mean   speedup:  {mean_68:.4f}×  →  {mean_69:.4f}×   Δ = {(mean_69-mean_68):+.4f}")
print(f"  wins (sp>=1.0):  {wins_68}/{len(common)}  →  {wins_69}/{len(common)}   "
      f"Δ = {wins_69-wins_68:+d}")

# ========================================================
# §B Per-model breakdown
# ========================================================
print()
print("§B  Per-model breakdown")
print(f"  {'model':<14}  {'med r68':>8}  {'med r69':>8}  {'Δmed':>8}  "
      f"{'wins r68':>9}  {'wins r69':>9}  {'Δwins':>7}")

models = sorted({k[0] for k in common}, key=lambda m: m)
for m in models:
    mk = [k for k in common if k[0] == m]
    s68 = [i68[k]['cuda_speedup_vs_fp16'] for k in mk]
    s69 = [i69[k]['cuda_speedup_vs_fp16'] for k in mk]
    w68 = sum(1 for s in s68 if s >= 1.0)
    w69 = sum(1 for s in s69 if s >= 1.0)
    print(f"  {m:<14}  {statistics.median(s68):>7.3f}×  "
          f"{statistics.median(s69):>7.3f}×  "
          f"{statistics.median(s69)-statistics.median(s68):+8.4f}  "
          f"{w68:>5}/{len(mk):<3}  {w69:>5}/{len(mk):<3}  "
          f"{w69-w68:+5d}")

# ========================================================
# §C Target (14B gu) shapes
# ========================================================
print()
print("§C  TARGET family (Qwen3-14B gate_up_proj) — C.7 region D")
print(f"  {'T':>5}  {'fp16 r68':>9}  {'cuda r68':>9}  {'sp r68':>7}  "
      f"{'fp16 r69':>9}  {'cuda r69':>9}  {'sp r69':>7}  {'Δcuda%':>7}  {'Δsp':>6}")

for T in [1024, 2048, 4096, 8192]:
    k = ('Qwen3-14B', 'gate_up_proj', T, 5120, 34816)
    if k not in common:
        continue
    r68 = i68[k]; r69 = i69[k]
    dcuda = (r69['cuda_us'] - r68['cuda_us']) / r68['cuda_us'] * 100
    dsp = r69['cuda_speedup_vs_fp16'] - r68['cuda_speedup_vs_fp16']
    marker = " ✓" if r69['cuda_speedup_vs_fp16'] >= 1.0 and r68['cuda_speedup_vs_fp16'] < 1.0 else ""
    print(f"  {T:>5}  {r68['fp16_us']:>8.1f}  {r68['cuda_us']:>8.1f}  "
          f"{r68['cuda_speedup_vs_fp16']:>6.3f}×  "
          f"{r69['fp16_us']:>8.1f}  {r69['cuda_us']:>8.1f}  "
          f"{r69['cuda_speedup_vs_fp16']:>6.3f}×  "
          f"{dcuda:+6.2f}%  {dsp:+5.3f}{marker}")

# ========================================================
# §D Guard shapes (should be unchanged)
# ========================================================
print()
print("§D  GUARD shapes (C.7 must NOT touch these)")
GUARDS = [
    ('Qwen3-8B',    'gate_up_proj', 4096, 24576),  # winner
    ('Qwen3-4B',    'q_proj',       2560,  4096),  # winner
    ('Qwen2.5-32B', 'gate_up_proj', 5120, 55296),  # loser, must stay same
    ('LLaMA3-70B',  'gate_up_proj', 8192, 57344),  # loser, must stay same
    ('Qwen3-14B',   'q_proj',       5120,  5120),  # neutral
    ('Qwen2.5-32B', 'down_proj',   27648,  5120),  # C.6v2 region B, unchanged
]

print(f"  {'shape':<44}  {'T':>5}  {'cuda r68':>9}  {'cuda r69':>9}  "
      f"{'Δcuda%':>7}  {'sp r68':>7}  {'sp r69':>7}  {'Δsp':>6}  {'status':<15}")
for m, p, din, dout in GUARDS:
    label = f"{m} {p} {din}->{dout}"
    for T in [1024, 2048, 4096, 8192]:
        k = (m, p, T, din, dout)
        if k not in common:
            continue
        r68 = i68[k]; r69 = i69[k]
        dcuda = (r69['cuda_us'] - r68['cuda_us']) / r68['cuda_us'] * 100
        dsp = r69['cuda_speedup_vs_fp16'] - r68['cuda_speedup_vs_fp16']
        if abs(dcuda) > 3:
            status = "CHANGED!"
        elif abs(dcuda) > 1:
            status = "minor"
        else:
            status = "unchanged ✓"
        print(f"  {label:<44}  {T:>5}  {r68['cuda_us']:>8.1f}  "
              f"{r69['cuda_us']:>8.1f}  {dcuda:+6.2f}%  "
              f"{r68['cuda_speedup_vs_fp16']:>6.3f}×  "
              f"{r69['cuda_speedup_vs_fp16']:>6.3f}×  {dsp:+5.3f}  {status}")

# ========================================================
# §E Regression check - list ALL shapes that got >3% slower or >2% lower sp
# ========================================================
print()
print("§E  Full regression check — any shape >3% slower cuda_us or >0.03 lower sp in r69?")
regressions = []
for k in common:
    r68 = i68[k]; r69 = i69[k]
    dcuda = (r69['cuda_us'] - r68['cuda_us']) / r68['cuda_us'] * 100
    dsp = r69['cuda_speedup_vs_fp16'] - r68['cuda_speedup_vs_fp16']
    if dcuda > 3 or dsp < -0.03:
        regressions.append((k, r68, r69, dcuda, dsp))

if not regressions:
    print("  NO REGRESSIONS DETECTED.")
else:
    print(f"  Found {len(regressions)} potential regressions (may be timing noise):")
    for k, r68, r69, dcuda, dsp in sorted(regressions, key=lambda x: x[3], reverse=True):
        print(f"    {k[0]:<14} {k[1]:<14} T={k[2]:<5} d_in={k[3]:<5} d_out={k[4]:<6}  "
              f"cuda {r68['cuda_us']:.1f}→{r69['cuda_us']:.1f} ({dcuda:+.2f}%)  "
              f"sp {r68['cuda_speedup_vs_fp16']:.3f}→{r69['cuda_speedup_vs_fp16']:.3f} ({dsp:+.3f})")

# ========================================================
# §F Improvements — shapes that got faster
# ========================================================
print()
print("§F  Top improvements — shapes >3% faster in r69")
improvements = []
for k in common:
    r68 = i68[k]; r69 = i69[k]
    dcuda = (r69['cuda_us'] - r68['cuda_us']) / r68['cuda_us'] * 100
    dsp = r69['cuda_speedup_vs_fp16'] - r68['cuda_speedup_vs_fp16']
    if dcuda < -3:
        improvements.append((k, r68, r69, dcuda, dsp))

print(f"  Found {len(improvements)} improvements:")
for k, r68, r69, dcuda, dsp in sorted(improvements, key=lambda x: x[3]):
    print(f"    {k[0]:<14} {k[1]:<14} T={k[2]:<5} d_in={k[3]:<5} d_out={k[4]:<6}  "
          f"cuda {r68['cuda_us']:.1f}→{r69['cuda_us']:.1f} ({dcuda:+.2f}%)  "
          f"sp {r68['cuda_speedup_vs_fp16']:.3f}→{r69['cuda_speedup_vs_fp16']:.3f} ({dsp:+.3f})")
