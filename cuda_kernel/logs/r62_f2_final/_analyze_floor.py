"""Analyse the '30us floor' pattern in r62 F2 final bench."""
import json
b = json.load(open('/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r62_f2_final/qwen3_20260430_122555/bench.json'))
recs = [x for x in b['records']]

# Pull out quant_us and fused_us per shape
by_shape = {}
for r in recs:
    key = (r['model'], r['proj'], r['T'])
    by_shape.setdefault(key, {})[r['kernel']] = r

print(f"{'model':<12} {'proj':<14} {'T':>4} {'d_in':>5} {'d_out':>6} {'qnt_us':>7} {'fused':>7} {'e2e':>7} {'cuda/fp16':>10}")
pairs = []
for (m, p, T), ks in by_shape.items():
    e2e = ks.get('end_to_end')
    qnt = ks.get('activation_quant')
    fsd = ks.get('fused_dense_sparse')
    if e2e is None: continue
    q_us = qnt['cuda_us'] if qnt else 0
    f_us = fsd['cuda_us'] if fsd else 0
    c_us = e2e['cuda_us']
    spd = e2e['cuda_speedup_vs_fp16']
    pairs.append((q_us, f_us, c_us, spd, m, p, T, e2e['d_in'], e2e['d_out']))

# Sort by increasing cuda_us to see floor
pairs.sort()
print("# Lowest cuda_us — shows the fixed overhead floor")
for q, f, c, s, m, p, T, di, do in pairs[:20]:
    print(f"{m:<12} {p:<14} {T:>4} {di:>5} {do:>6} {q:>6.2f}us {f:>6.2f}us {c:>6.2f}us {s:>9.2f}x")

print()
print(f"# 'Stuck' at 30us region — quant_us/fused_us split analysis")
stuck = [x for x in pairs if 28 <= x[2] <= 36]
print(f"{len(stuck)} shapes in [28-36]us band out of {len(pairs)}")
print(f"  avg q_us: {sum(x[0] for x in stuck)/max(len(stuck),1):.2f}us")
print(f"  avg f_us: {sum(x[1] for x in stuck)/max(len(stuck),1):.2f}us")
print(f"  avg e2e:  {sum(x[2] for x in stuck)/max(len(stuck),1):.2f}us")

# Most loss
print()
print("# Worst speedup (where both q_us and f_us dominate roofline)")
pairs.sort(key=lambda x: x[3])
for q, f, c, s, m, p, T, di, do in pairs[:10]:
    print(f"{m:<12} {p:<14} {T:>4} {di:>5} {do:>6} {q:>6.2f}us {f:>6.2f}us {c:>6.2f}us {s:>9.2f}x  (quant fraction: {q/c*100:.0f}%)")
