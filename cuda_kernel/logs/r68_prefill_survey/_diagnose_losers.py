"""Precise per-shape diagnosis for the 3 loser families.

For each of the 3 target families, we break down:
  1. cuda_us vs fp16_us (what the user sees)
  2. cuda_eff (how far we are from INT4 roof)
  3. n_cta_m / n_cta_n / SM wave count (grid shape)
  4. compute-bound or mem-bound? (where does roof come from)
  5. sub-kernel breakdown: activation_quant_us vs GEMM_us

Output: actionable per-shape table for decision-making.
"""
import json
import statistics
from collections import defaultdict

# Load both benches; prefill has T=1024..8192, survey has T=1..512.
SURVEY = json.load(open(
    "/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r68_multiT_survey/bench.json"))
PREFILL = json.load(open(
    "/Users/yangjiahu/Desktop/workspace/HKUST/kernel/cuda_kernel/logs/r68_prefill_survey/bench.json"))

# Collect both e2e and sub-kernel (dense_gemm / fused_dense_sparse) records
all_recs = []
for src, data in [("survey", SURVEY), ("prefill", PREFILL)]:
    for r in data["records"]:
        r["_src"] = src
        all_recs.append(r)

# Build index: (model, proj, T, kernel) -> record
idx = {}
for r in all_recs:
    if r.get('model') == 'Qwen3-0.6B':
        continue
    key = (r['model'], r['proj'], r['T'], r.get('kernel'))
    idx[key] = r

# Roofline helpers
HBM = 1008e9 * 0.85
INT4_TC = 660.6e12 * 0.85
FP16_TC = 165.2e12 * 0.85
BCOL = 128
SM_COUNT = 128  # RTX 4090

def sm_wave_info(T, d_in, d_out, kBm=128, kBn=64):
    """Compute grid and SM-wave utilisation."""
    n_cta_m = (d_out + kBm - 1) // kBm
    n_cta_n = (T + kBn - 1) // kBn
    n_cta = n_cta_m * n_cta_n
    waves_full = n_cta // SM_COUNT
    tail = n_cta % SM_COUNT
    util_last = tail / SM_COUNT if tail else 1.0
    return n_cta_m, n_cta_n, n_cta, waves_full, util_last

def cuda_roof_us(T, d_in, d_out):
    ng = d_in // BCOL
    if T == 1:
        b = 0.5*d_in*d_out + 2*d_in + 4*d_out*ng + 2*d_out
        return max(b/HBM, 2*d_in*d_out/INT4_TC) * 1e6
    bq = 2*T*d_in + 0.5*T*d_in + 2*T + 4*T*ng
    tq = bq / HBM * 1e6
    bg = 0.5*d_in*d_out + 0.5*T*d_in + 4*d_out*ng + 4*T*ng + 2*T + 2*T*d_out
    tg = max(2*T*d_in*d_out / INT4_TC, bg / HBM) * 1e6
    return tq + tg, tq, tg

def fp16_roof_us(T, d_in, d_out):
    return max(2*T*d_in*d_out/FP16_TC, 2*(d_in*d_out+T*d_in+T*d_out)/HBM) * 1e6

def bound_kind(T, d_in, d_out):
    ng = d_in // BCOL
    bg = 0.5*d_in*d_out + 0.5*T*d_in + 4*d_out*ng + 4*T*ng + 2*T + 2*T*d_out
    t_mem = bg / HBM * 1e6
    t_cmp = 2*T*d_in*d_out/INT4_TC * 1e6
    return "compute" if t_cmp > t_mem else "mem", t_cmp, t_mem


# ============================================================
# Family A: gate_up_proj LARGE model (14B / 32B / 70B)
# ============================================================
FAMILY_A_SHAPES = [
    ("14B gu", 5120, 34816),
    ("32B gu", 5120, 55296),
    ("70B gu", 8192, 57344),
]
# Ts for each family: only the prefill-range ones where it loses
TS = [512, 1024, 2048, 4096, 8192]

print("=" * 110)
print("Family A: gate_up_proj LARGE model (14B/32B/70B) — primary loser")
print("=" * 110)
print(f"{'shape':<10} {'T':>5}  {'cuda_us':>9} {'fp16_us':>9} {'sp':>6}  "
      f"{'cuda_eff':>9}  {'grid':>10}  {'bound':>7}  "
      f"{'tcmp':>7} {'tmem':>7}  {'quant_us':>9}")

for label, d_in, d_out in FAMILY_A_SHAPES:
    m = label.split()[0]
    model = {"14B":"Qwen3-14B", "32B":"Qwen2.5-32B", "70B":"LLaMA3-70B"}[m]
    for T in TS:
        k_e2e = (model, 'gate_up_proj', T, 'end_to_end')
        k_aq  = (model, 'gate_up_proj', T, 'activation_quant')
        if k_e2e not in idx:
            continue
        r = idx[k_e2e]
        aq_us = idx.get(k_aq, {}).get('cuda_us', 0)
        n_cta_m, n_cta_n, n_cta, wf, ul = sm_wave_info(T, d_in, d_out)
        kind, tcmp, tmem = bound_kind(T, d_in, d_out)
        cuda_roof, tq, tg = cuda_roof_us(T, d_in, d_out)
        ce = cuda_roof / r['cuda_us']
        print(f"{label:<10} {T:>5}  {r['cuda_us']:>8.1f} "
              f"{r['fp16_us']:>8.1f} {r['cuda_speedup_vs_fp16']:>5.2f}x  "
              f"{ce*100:>7.0f}%  {n_cta_m}x{n_cta_n}({n_cta}){' '*max(0,10-len(str(n_cta_m)+'x'+str(n_cta_n)+'('+str(n_cta)+')'))}"
              f"  {kind:<7}  {tcmp:>6.0f} {tmem:>6.0f}  "
              f"{aq_us:>8.1f}")
print()

# ============================================================
# Family B: kv_proj LARGE model (14B / 32B / 70B)
# ============================================================
FAMILY_B_SHAPES = [
    ("14B kv", 5120, 2048),
    ("32B kv", 5120, 2048),
    ("70B kv", 8192, 2048),
]

print("=" * 110)
print("Family B: kv_proj LARGE model (14B/32B/70B) — secondary loser")
print("=" * 110)
print(f"{'shape':<10} {'T':>5}  {'cuda_us':>9} {'fp16_us':>9} {'sp':>6}  "
      f"{'cuda_eff':>9}  {'grid':>10}  {'bound':>7}  "
      f"{'tcmp':>7} {'tmem':>7}  {'quant_us':>9}")

for label, d_in, d_out in FAMILY_B_SHAPES:
    m = label.split()[0]
    model = {"14B":"Qwen3-14B", "32B":"Qwen2.5-32B", "70B":"LLaMA3-70B"}[m]
    for T in TS:
        k_e2e = (model, 'kv_proj', T, 'end_to_end')
        k_aq  = (model, 'kv_proj', T, 'activation_quant')
        if k_e2e not in idx:
            continue
        r = idx[k_e2e]
        aq_us = idx.get(k_aq, {}).get('cuda_us', 0)
        n_cta_m, n_cta_n, n_cta, wf, ul = sm_wave_info(T, d_in, d_out)
        kind, tcmp, tmem = bound_kind(T, d_in, d_out)
        cuda_roof, tq, tg = cuda_roof_us(T, d_in, d_out)
        ce = cuda_roof / r['cuda_us']
        print(f"{label:<10} {T:>5}  {r['cuda_us']:>8.1f} "
              f"{r['fp16_us']:>8.1f} {r['cuda_speedup_vs_fp16']:>5.2f}x  "
              f"{ce*100:>7.0f}%  {n_cta_m}x{n_cta_n}({n_cta}){' '*max(0,10-len(str(n_cta_m)+'x'+str(n_cta_n)+'('+str(n_cta)+')'))}"
              f"  {kind:<7}  {tcmp:>6.0f} {tmem:>6.0f}  "
              f"{aq_us:>8.1f}")
print()

# ============================================================
# Family C: down_proj SMALL model (1.7B / 4B) — quick check
# ============================================================
FAMILY_C_SHAPES = [
    ("1.7B dn", 6144, 2048),
    ("4B dn",   9728, 2560),
]
print("=" * 110)
print("Family C: down_proj SMALL model (1.7B/4B) — tertiary loser")
print("=" * 110)
print(f"{'shape':<10} {'T':>5}  {'cuda_us':>9} {'fp16_us':>9} {'sp':>6}  "
      f"{'cuda_eff':>9}  {'grid':>10}  {'bound':>7}  "
      f"{'tcmp':>7} {'tmem':>7}  {'quant_us':>9}")

for label, d_in, d_out in FAMILY_C_SHAPES:
    m = label.split()[0]
    model = {"1.7B":"Qwen3-1.7B", "4B":"Qwen3-4B"}[m]
    for T in TS:
        k_e2e = (model, 'down_proj', T, 'end_to_end')
        k_aq  = (model, 'down_proj', T, 'activation_quant')
        if k_e2e not in idx:
            continue
        r = idx[k_e2e]
        aq_us = idx.get(k_aq, {}).get('cuda_us', 0)
        n_cta_m, n_cta_n, n_cta, wf, ul = sm_wave_info(T, d_in, d_out)
        kind, tcmp, tmem = bound_kind(T, d_in, d_out)
        cuda_roof, tq, tg = cuda_roof_us(T, d_in, d_out)
        ce = cuda_roof / r['cuda_us']
        print(f"{label:<10} {T:>5}  {r['cuda_us']:>8.1f} "
              f"{r['fp16_us']:>8.1f} {r['cuda_speedup_vs_fp16']:>5.2f}x  "
              f"{ce*100:>7.0f}%  {n_cta_m}x{n_cta_n}({n_cta}){' '*max(0,10-len(str(n_cta_m)+'x'+str(n_cta_n)+'('+str(n_cta)+')'))}"
              f"  {kind:<7}  {tcmp:>6.0f} {tmem:>6.0f}  "
              f"{aq_us:>8.1f}")
