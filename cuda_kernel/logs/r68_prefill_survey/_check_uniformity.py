"""Analyse the "uniformity" puzzle:
Why do all models at T >= 1024 cluster around 1.0x-1.3x speedup?

Hypothesis: the user observed that prefill speedups look "flat" across
models and batch sizes.  Verify:
  - Per-proj breakdown: are projs consistent or masked by median?
  - Per-(model, proj) T-curves: do they all plateau?
  - FP16 and CUDA efficiency numbers: do they both stabilise?
"""
import json
import statistics
from collections import defaultdict

PREF = json.load(open(
    "/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r68_prefill_survey/bench.json"))
rows = [r for r in PREF['records']
        if r.get('kernel') == 'end_to_end' and r['model'] != 'Qwen3-0.6B']

# Roofline
HBM = 1008e9 * 0.85
FP16_TC = 165.2e12 * 0.85
INT4_TC = 660.6e12 * 0.85
BCOL = 128

def cuda_roof(T, di, do):
    ng = di // BCOL
    if T == 1:
        b = 0.5*di*do + 2*di + 4*do*ng + 2*do
        return max(b/HBM, 2*di*do/INT4_TC) * 1e6
    bq = 2*T*di + 0.5*T*di + 2*T + 4*T*ng
    tq = bq / HBM * 1e6
    bg = 0.5*di*do + 0.5*T*di + 4*do*ng + 4*T*ng + 2*T + 2*T*do
    tg = max(2*T*di*do/INT4_TC, bg/HBM) * 1e6
    return tq + tg

def fp16_roof(T, di, do):
    return max(2*T*di*do/FP16_TC, 2*(di*do+T*di+T*do)/HBM) * 1e6

for r in rows:
    r['cuda_eff'] = cuda_roof(r['T'], r['d_in'], r['d_out']) / r['cuda_us']
    r['fp16_eff'] = fp16_roof(r['T'], r['d_in'], r['d_out']) / r['fp16_us']
    r['sp'] = r['cuda_speedup_vs_fp16']

# ============================================================
# §A  Variance WITHIN each (model, T) across its 5 projs
# ============================================================
print("§A  Speedup spread across 5 projs within each (model, T)")
print(f"  {'model':<14} {'T':>5}  {'min':>6} {'med':>6} {'max':>6}  "
      f"{'spread':>8}  {'projs':<50}")
models = ['Qwen3-1.7B','Qwen3-4B','Qwen3-8B','Qwen3-14B','Qwen2.5-32B','LLaMA3-70B']
for m in models:
    for T in [1024, 2048, 4096, 8192]:
        rs = [r for r in rows if r['model']==m and r['T']==T]
        sps = sorted([(r['sp'], r['proj']) for r in rs])
        if not sps:
            continue
        lo, med, hi = sps[0][0], sps[len(sps)//2][0], sps[-1][0]
        spread = (hi - lo) / med * 100
        proj_str = " ".join(f"{p[:2]}={s:.2f}" for s, p in sps)
        print(f"  {m:<14} {T:>5}  {lo:>5.2f}x {med:>5.2f}x {hi:>5.2f}x  "
              f"{spread:>6.1f}%  {proj_str}")
    print()

# ============================================================
# §B  Variance ACROSS T within each (model, proj)
# ============================================================
print()
print("§B  Speedup spread across 4 prefill Ts within each (model, proj)")
print(f"  {'model':<14} {'proj':<14}  "
      f"{'T=1024':>7} {'T=2048':>7} {'T=4096':>7} {'T=8192':>7}  {'spread':>8}")
projs = ['q_proj', 'kv_proj', 'o_proj', 'gate_up_proj', 'down_proj']
for m in models:
    for p in projs:
        rs = [r for r in rows if r['model']==m and r['proj']==p]
        if not rs:
            continue
        by_T = {r['T']: r['sp'] for r in rs}
        row = f"  {m:<14} {p:<14}  "
        vals = []
        for T in [1024, 2048, 4096, 8192]:
            v = by_T.get(T)
            if v is None:
                row += f"{'-':>7} "
            else:
                row += f"{v:>6.2f}x "
                vals.append(v)
        if vals:
            spread = (max(vals) - min(vals)) / statistics.median(vals) * 100
            row += f" {spread:>6.1f}%"
        print(row)
    print()

# ============================================================
# §C  Top 10 biggest winners and losers at T=2048
# ============================================================
print()
print("§C  T=2048 individual-shape outliers (so we see real variance)")
t2048 = sorted([r for r in rows if r['T']==2048], key=lambda r: r['sp'])
print("  Bottom 5 (worst):")
for r in t2048[:5]:
    print(f"    {r['model']:<14} {r['proj']:<14} {r['d_in']:>5}->{r['d_out']:<6}  "
          f"sp={r['sp']:.2f}x  cuda_eff={r['cuda_eff']*100:.0f}%  "
          f"fp16_eff={r['fp16_eff']*100:.0f}%")
print("  Top 5 (best):")
for r in t2048[-5:]:
    print(f"    {r['model']:<14} {r['proj']:<14} {r['d_in']:>5}->{r['d_out']:<6}  "
          f"sp={r['sp']:.2f}x  cuda_eff={r['cuda_eff']*100:.0f}%  "
          f"fp16_eff={r['fp16_eff']*100:.0f}%")
