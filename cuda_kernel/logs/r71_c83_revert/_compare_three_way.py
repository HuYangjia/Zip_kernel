"""Three-way comparison: r69 (C.7 baseline) vs r70 (C.8 full) vs r71 (C.8.3 revert).

Validates that C.8.3 revert recovers r69 performance on the 5 regressed shapes
while preserving any C.8.1(a)/C.8.2 gains.
"""
import json
import statistics
from pathlib import Path

R69 = "/root/Zip_kernel/kernel/cuda_kernel/logs/r69_c7_prefill/bench.json"
R70 = "/root/Zip_kernel/kernel/cuda_kernel/logs/r70_c8_full/qwen3_20260502_141424/bench.json"
_r71_root = Path("/root/Zip_kernel/kernel/cuda_kernel/logs/r71_c83_revert")
_cands = sorted(_r71_root.glob("qwen3_*/bench.json"))
assert _cands, f"no r71 bench.json under {_r71_root}"
R71 = str(_cands[-1])
print(f"r69: {R69}")
print(f"r70: {R70}")
print(f"r71: {R71}")

def index(data):
    return {(r['model'], r['proj'], r['T'], r['d_in'], r['d_out']): r
            for r in data['records']
            if r.get('kernel') == 'end_to_end' and r['model'] != 'Qwen3-0.6B'}

i69 = index(json.load(open(R69)))
i70 = index(json.load(open(R70)))
i71 = index(json.load(open(R71)))

# Common across all 3 — r69 has T={1024,2048,4096,8192}; r70/r71 have T={1,8,128,512,1024}
common_all = set(i69) & set(i70) & set(i71)
common_70_71 = set(i70) & set(i71)
print(f"Shapes common to all 3: {len(common_all)} (only T=1024 overlap)")
print(f"Shapes common to r70 & r71: {len(common_70_71)}")

# ========================================================
# §A  Global summary - r70 vs r71 (same shape set)
# ========================================================
print()
print("=" * 90)
print("§A  Global summary: r70 (C.8) vs r71 (C.8.3 revert)")
print("=" * 90)
sps70 = [i70[k]['cuda_speedup_vs_fp16'] for k in common_70_71]
sps71 = [i71[k]['cuda_speedup_vs_fp16'] for k in common_70_71]
w70 = sum(1 for s in sps70 if s >= 1.0)
w71 = sum(1 for s in sps71 if s >= 1.0)
print(f"  median sp: {statistics.median(sps70):.4f}x -> {statistics.median(sps71):.4f}x  "
      f"d = {statistics.median(sps71)-statistics.median(sps70):+.4f}")
print(f"  mean   sp: {statistics.mean(sps70):.4f}x -> {statistics.mean(sps71):.4f}x  "
      f"d = {statistics.mean(sps71)-statistics.mean(sps70):+.4f}")
print(f"  wins:      {w70}/{len(common_70_71)} -> {w71}/{len(common_70_71)}  "
      f"d = {w71-w70:+d}")

# ========================================================
# §B  C.8 TARGET shapes — three-way view
# ========================================================
print()
print("=" * 90)
print("§B  C.8 TARGET shapes — three-way (was regressed under C.8, expected to recover)")
print("=" * 90)
# Only shapes visible in common_all or common_70_71
TARGETS = [
    # (model, proj, din, dout, available_Ts)
    ("LLaMA3-70B",  "kv_proj",   8192,  2048),
    ("Qwen3-1.7B",  "down_proj", 6144,  2048),
    ("Qwen3-4B",    "down_proj", 9728,  2560),
    ("Qwen3-1.7B",  "kv_proj",   2048,  2048),
    ("LLaMA3-70B",  "o_proj",    8192,  8192),
]
print(f"  {'model':<14} {'proj':<12} {'T':>5} {'d_in':>6} {'d_out':>6}  "
      f"{'cuda69':>8} {'cuda70':>8} {'cuda71':>8}  "
      f"{'sp69':>7} {'sp70':>7} {'sp71':>7}  {'status'}")
print("  " + "-" * 115)
for m, p, din, dout in TARGETS:
    for T in [512, 1024]:
        k = (m, p, T, din, dout)
        r69 = i69.get(k); r70 = i70.get(k); r71 = i71.get(k)
        if r71 is None and r70 is None:
            continue
        c69 = f"{r69['cuda_us']:>8.1f}" if r69 else "      -"
        s69 = f"{r69['cuda_speedup_vs_fp16']:.3f}x" if r69 else "   -"
        c70 = f"{r70['cuda_us']:>8.1f}" if r70 else "      -"
        s70 = f"{r70['cuda_speedup_vs_fp16']:.3f}x" if r70 else "   -"
        c71 = f"{r71['cuda_us']:>8.1f}" if r71 else "      -"
        s71 = f"{r71['cuda_speedup_vs_fp16']:.3f}x" if r71 else "   -"
        # status: compare r71 vs r70 — expect recovery
        if r70 and r71:
            dsp = r71['cuda_speedup_vs_fp16'] - r70['cuda_speedup_vs_fp16']
            dc  = (r71['cuda_us'] - r70['cuda_us']) / r70['cuda_us'] * 100
            status = f"dsp={dsp:+.3f}  dc={dc:+.2f}%"
        else:
            status = ""
        print(f"  {m:<14} {p:<12} {T:>5} {din:>6} {dout:>6}  {c69} {c70} {c71}  "
              f"{s69:>7} {s70:>7} {s71:>7}  {status}")

# ========================================================
# §C  C.8 GUARD - was the 4B dn T=1024 C.8.2 split_k=2 win preserved?
# ========================================================
print()
print("=" * 90)
print("§C  C.8.2 preservation check — 4B dn T=1024 should keep split_k=2 gain")
print("=" * 90)
k = ("Qwen3-4B", "down_proj", 1024, 9728, 2560)
r69 = i69.get(k); r70 = i70.get(k); r71 = i71.get(k)
if r69 and r70 and r71:
    print(f"  r69 C.7       : cuda={r69['cuda_us']:.1f}us  sp={r69['cuda_speedup_vs_fp16']:.3f}x")
    print(f"  r70 C.8 (both): cuda={r70['cuda_us']:.1f}us  sp={r70['cuda_speedup_vs_fp16']:.3f}x")
    print(f"  r71 C.8.3     : cuda={r71['cuda_us']:.1f}us  sp={r71['cuda_speedup_vs_fp16']:.3f}x")
    if r71['cuda_speedup_vs_fp16'] > r69['cuda_speedup_vs_fp16'] + 0.01:
        print("  -> C.8.2 split_k=2 gain PRESERVED in C.8.3")
    else:
        print("  -> C.8.2 gain appears to be lost (likely because it required C.8.1(b)'s kBm=64)")

# ========================================================
# §D  Regressions r69 -> r71 on T=1024 overlap
# ========================================================
print()
print("=" * 90)
print("§D  Regressions r69 -> r71 on T=1024 (any shape >3% slower OR sp drop >0.03)")
print("=" * 90)
regs = []
for k in common_all:
    r69, r71 = i69[k], i71[k]
    dcuda = (r71['cuda_us'] - r69['cuda_us']) / r69['cuda_us'] * 100
    dsp = r71['cuda_speedup_vs_fp16'] - r69['cuda_speedup_vs_fp16']
    if dcuda > 3 or dsp < -0.03:
        regs.append((k, r69, r71, dcuda, dsp))
if not regs:
    print(f"  NO REGRESSIONS on {len(common_all)} common T=1024 shapes.")
else:
    print(f"  Found {len(regs)} potential regressions:")
    for k, r69, r71, dcuda, dsp in sorted(regs, key=lambda x: x[3], reverse=True):
        print(f"    {k[0]:<14} {k[1]:<14} T={k[2]:<5} {k[3]:>5}->{k[4]:<6}  "
              f"cuda {r69['cuda_us']:.1f}->{r71['cuda_us']:.1f} ({dcuda:+.2f}%)  "
              f"sp {r69['cuda_speedup_vs_fp16']:.3f}->{r71['cuda_speedup_vs_fp16']:.3f} ({dsp:+.3f})")

# ========================================================
# §E  Improvements r69 -> r71
# ========================================================
print()
print("=" * 90)
print("§E  Improvements r69 -> r71 on T=1024 (>3% faster)")
print("=" * 90)
imps = []
for k in common_all:
    r69, r71 = i69[k], i71[k]
    dcuda = (r71['cuda_us'] - r69['cuda_us']) / r69['cuda_us'] * 100
    dsp = r71['cuda_speedup_vs_fp16'] - r69['cuda_speedup_vs_fp16']
    if dcuda < -3:
        imps.append((k, r69, r71, dcuda, dsp))
print(f"  {len(imps)} improvements:")
for k, r69, r71, dcuda, dsp in sorted(imps, key=lambda x: x[3])[:30]:
    print(f"    {k[0]:<14} {k[1]:<14} T={k[2]:<5} {k[3]:>5}->{k[4]:<6}  "
          f"cuda {r69['cuda_us']:.1f}->{r71['cuda_us']:.1f} ({dcuda:+.2f}%)  "
          f"sp {r69['cuda_speedup_vs_fp16']:.3f}->{r71['cuda_speedup_vs_fp16']:.3f} ({dsp:+.3f})")

print()
print("=" * 90)
print("END OF REPORT")
print("=" * 90)
