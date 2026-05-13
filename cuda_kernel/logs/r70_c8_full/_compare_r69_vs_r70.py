"""r69 C.7 vs r70 C.8 impact comparison.

C.8 changes (all in fused_dense_sparse_mma_int4.cu):
  * C.8.1(a): kBm 64 -> 128 for d_in<=5120 & d_out>=8192 & T>=128
              (target: 32B gu, 70B gu)
  * C.8.1(b): kBm 128 -> 64  for d_in>=6144 & d_out<=2560
              (target: 70B kv, 1.7B dn, 4B dn)
  * C.8.2   : split_k = 2 for d_in>=8192 & d_out<=2560 & T>=1024
              (target: 4B dn)

This script prints:
  §A Global:    median/mean sp, wins (r69 -> r70)
  §B Per-model breakdown
  §C C.8 TARGET shapes (5 loser shapes)
  §D C.8 GUARD shapes (already winners, must NOT regress)
  §E Regressions >3% slower cuda_us or sp drop > 0.03
  §F Improvements  >3% faster cuda_us
"""
import json
import statistics
from pathlib import Path

R69 = "/root/Zip_kernel/kernel/cuda_kernel/logs/r69_c7_prefill/bench.json"
# r70_c8_full/qwen3_<ts>/bench.json — resolve by glob
_r70_root = Path("/root/Zip_kernel/kernel/cuda_kernel/logs/r70_c8_full")
_cands = sorted(_r70_root.glob("qwen3_*/bench.json"))
assert _cands, f"no r70 bench.json under {_r70_root}"
R70 = str(_cands[-1])
print(f"r69 json: {R69}")
print(f"r70 json: {R70}")

d69 = json.load(open(R69))
d70 = json.load(open(R70))

def index(data):
    # Only end-to-end records; drop 0.6B (noise-dominated tiny).
    return {(r['model'], r['proj'], r['T'], r['d_in'], r['d_out']): r
            for r in data['records']
            if r.get('kernel') == 'end_to_end' and r['model'] != 'Qwen3-0.6B'}

i69 = index(d69); i70 = index(d70)
common = set(i69) & set(i70)
print(f"Common (model, proj, T, d_in, d_out) shapes: {len(common)}   "
      f"(r69={len(i69)}, r70={len(i70)})")

# ========================================================
# §A Global summary
# ========================================================
sps_69 = [i69[k]['cuda_speedup_vs_fp16'] for k in common]
sps_70 = [i70[k]['cuda_speedup_vs_fp16'] for k in common]

wins_69 = sum(1 for s in sps_69 if s >= 1.0)
wins_70 = sum(1 for s in sps_70 if s >= 1.0)
med_69  = statistics.median(sps_69)
med_70  = statistics.median(sps_70)
mean_69 = statistics.mean(sps_69)
mean_70 = statistics.mean(sps_70)

print()
print("=" * 90)
print("§A  Global summary (r69 C.7 -> r70 C.8)")
print("=" * 90)
print(f"  median speedup:  {med_69:.4f}x  ->  {med_70:.4f}x   delta = {(med_70-med_69):+.4f}")
print(f"  mean   speedup:  {mean_69:.4f}x  ->  {mean_70:.4f}x   delta = {(mean_70-mean_69):+.4f}")
print(f"  wins (sp>=1.0):  {wins_69}/{len(common)}  ->  {wins_70}/{len(common)}   delta = {wins_70-wins_69:+d}")

# ========================================================
# §B Per-model breakdown
# ========================================================
print()
print("=" * 90)
print("§B  Per-model breakdown")
print("=" * 90)
print(f"  {'model':<14}  {'med r69':>8}  {'med r70':>8}  {'d_med':>8}  "
      f"{'wins r69':>9}  {'wins r70':>9}  {'d_wins':>7}")

models = sorted({k[0] for k in common})
for m in models:
    mk = [k for k in common if k[0] == m]
    s69 = [i69[k]['cuda_speedup_vs_fp16'] for k in mk]
    s70 = [i70[k]['cuda_speedup_vs_fp16'] for k in mk]
    w69 = sum(1 for s in s69 if s >= 1.0)
    w70 = sum(1 for s in s70 if s >= 1.0)
    print(f"  {m:<14}  {statistics.median(s69):>7.3f}x  "
          f"{statistics.median(s70):>7.3f}x  "
          f"{statistics.median(s70)-statistics.median(s69):+8.4f}  "
          f"{w69:>5}/{len(mk):<3}  {w70:>5}/{len(mk):<3}  "
          f"{w70-w69:+5d}")

# ========================================================
# §C C.8 TARGETs (5 loser shapes)
# ========================================================
print()
print("=" * 90)
print("§C  C.8 TARGET shapes (5 losers expected to improve)")
print("=" * 90)
TARGETS = [
    ('Qwen2.5-32B', 'gate_up_proj', 2048, 5120, 55296),  # C.8.1(a)
    ('LLaMA3-70B',  'gate_up_proj', 2048, 8192, 57344),  # C.8.1(a)
    ('LLaMA3-70B',  'kv_proj',      1024, 8192,  2048),  # C.8.1(b)
    ('Qwen3-1.7B',  'down_proj',    1024, 6144,  2048),  # C.8.1(b)
    ('Qwen3-4B',    'down_proj',    1024, 9728,  2560),  # C.8.1(b)+C.8.2
]
print(f"  {'model':<14} {'proj':<14} {'T':>5} {'d_in':>6} {'d_out':>6}  "
      f"{'cuda69':>8} {'cuda70':>8} {'d_cuda%':>8}  "
      f"{'sp69':>6} {'sp70':>6} {'d_sp':>6}")
print("  " + "-" * 88)
for m, p, T, din, dout in TARGETS:
    k = (m, p, T, din, dout)
    if k not in common:
        print(f"  [MISSING] {m} {p} T={T} {din}->{dout}")
        continue
    r69 = i69[k]; r70 = i70[k]
    dcuda = (r70['cuda_us'] - r69['cuda_us']) / r69['cuda_us'] * 100
    dsp = r70['cuda_speedup_vs_fp16'] - r69['cuda_speedup_vs_fp16']
    marker = " WIN" if r70['cuda_speedup_vs_fp16'] >= 1.0 and r69['cuda_speedup_vs_fp16'] < 1.0 else ""
    print(f"  {m:<14} {p:<14} {T:>5} {din:>6} {dout:>6}  "
          f"{r69['cuda_us']:>8.1f} {r70['cuda_us']:>8.1f} {dcuda:+7.2f}%  "
          f"{r69['cuda_speedup_vs_fp16']:>5.3f}x {r70['cuda_speedup_vs_fp16']:>5.3f}x "
          f"{dsp:+5.3f}{marker}")

# ========================================================
# §D C.8 GUARD shapes — must NOT regress
# ========================================================
print()
print("=" * 90)
print("§D  C.8 GUARD shapes (existing winners / neutrals must NOT regress)")
print("=" * 90)
GUARDS = [
    # Winners that the C.8.1(a) rule could accidentally touch:
    ('Qwen3-8B',    'gate_up_proj', 2048, 4096, 24576),  # d_in=4096 <=5120, d_out=24576 >=8192  - MATCHES C.8.1(a)!
    ('Qwen3-14B',   'gate_up_proj', 2048, 5120, 34816),  # d_in=5120, d_out=34816  - MATCHES C.8.1(a)!
    # Winners that C.8.1(b) might accidentally touch:
    ('Qwen3-4B',    'o_proj',       1024, 4096, 2560),   # d_in=4096 <6144 no - safe
    # Pure guards:
    ('Qwen3-8B',    'q_proj',       2048, 4096,  4096),
    ('Qwen2.5-32B', 'down_proj',    1024, 27648, 5120),  # huge d_in
    ('LLaMA3-70B',  'down_proj',    1024, 28672, 8192),  # huge d_in
    ('Qwen3-14B',   'kv_proj',      1024, 5120,  2048),  # d_in<6144 no C.8 match
    ('Qwen3-4B',    'q_proj',       1024, 2560,  4096),  # d_in=2560 no C.8 match
]
print(f"  {'model':<14} {'proj':<14} {'T':>5} {'d_in':>6} {'d_out':>6}  "
      f"{'cuda69':>8} {'cuda70':>8} {'d_cuda%':>8}  "
      f"{'sp69':>6} {'sp70':>6} {'status'}")
print("  " + "-" * 88)
for m, p, T, din, dout in GUARDS:
    k = (m, p, T, din, dout)
    if k not in common:
        print(f"  [MISSING] {m} {p} T={T} {din}->{dout}")
        continue
    r69 = i69[k]; r70 = i70[k]
    dcuda = (r70['cuda_us'] - r69['cuda_us']) / r69['cuda_us'] * 100
    dsp = r70['cuda_speedup_vs_fp16'] - r69['cuda_speedup_vs_fp16']
    if dcuda > 3:
        status = "REGRESSION!"
    elif dcuda > 1:
        status = "minor slow"
    elif abs(dcuda) <= 1:
        status = "unchanged"
    else:
        status = "faster"
    print(f"  {m:<14} {p:<14} {T:>5} {din:>6} {dout:>6}  "
          f"{r69['cuda_us']:>8.1f} {r70['cuda_us']:>8.1f} {dcuda:+7.2f}%  "
          f"{r69['cuda_speedup_vs_fp16']:>5.3f}x {r70['cuda_speedup_vs_fp16']:>5.3f}x  {status}")

# ========================================================
# §E Regressions across ALL shapes
# ========================================================
print()
print("=" * 90)
print("§E  Full regression check — any shape >3% slower cuda_us OR sp drop >0.03")
print("=" * 90)
regressions = []
for k in common:
    r69 = i69[k]; r70 = i70[k]
    dcuda = (r70['cuda_us'] - r69['cuda_us']) / r69['cuda_us'] * 100
    dsp = r70['cuda_speedup_vs_fp16'] - r69['cuda_speedup_vs_fp16']
    if dcuda > 3 or dsp < -0.03:
        regressions.append((k, r69, r70, dcuda, dsp))
if not regressions:
    print("  NO REGRESSIONS DETECTED (threshold: >3% cuda_us or >0.03 sp drop).")
else:
    print(f"  Found {len(regressions)} potential regressions (may include timing noise):")
    for k, r69, r70, dcuda, dsp in sorted(regressions, key=lambda x: x[3], reverse=True):
        print(f"    {k[0]:<14} {k[1]:<14} T={k[2]:<5} {k[3]:>5}->{k[4]:<6}  "
              f"cuda {r69['cuda_us']:.1f}->{r70['cuda_us']:.1f} ({dcuda:+.2f}%)  "
              f"sp {r69['cuda_speedup_vs_fp16']:.3f}->{r70['cuda_speedup_vs_fp16']:.3f} ({dsp:+.3f})")

# ========================================================
# §F Top improvements
# ========================================================
print()
print("=" * 90)
print("§F  Top improvements (cuda_us >3% faster)")
print("=" * 90)
improvements = []
for k in common:
    r69 = i69[k]; r70 = i70[k]
    dcuda = (r70['cuda_us'] - r69['cuda_us']) / r69['cuda_us'] * 100
    dsp = r70['cuda_speedup_vs_fp16'] - r69['cuda_speedup_vs_fp16']
    if dcuda < -3:
        improvements.append((k, r69, r70, dcuda, dsp))
print(f"  Found {len(improvements)} improvements:")
for k, r69, r70, dcuda, dsp in sorted(improvements, key=lambda x: x[3])[:30]:
    print(f"    {k[0]:<14} {k[1]:<14} T={k[2]:<5} {k[3]:>5}->{k[4]:<6}  "
          f"cuda {r69['cuda_us']:.1f}->{r70['cuda_us']:.1f} ({dcuda:+.2f}%)  "
          f"sp {r69['cuda_speedup_vs_fp16']:.3f}->{r70['cuda_speedup_vs_fp16']:.3f} ({dsp:+.3f})")

print()
print("=" * 90)
print("END OF REPORT")
print("=" * 90)
