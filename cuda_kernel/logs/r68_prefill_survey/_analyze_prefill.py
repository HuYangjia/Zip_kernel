"""Prefill-scenario analysis: combine r68_multiT (T=1..512) + r68_prefill
(T=1024..8192) and produce a unified report covering the full T range
that matters for real LLM inference.

Output sections:
  §A. Global T-sweep for every model (table + per-model trend)
  §B. Cross-model comparison at each T (why bigger models fare better)
  §C. Per-shape deep-dive at T=2048 (representative prefill point)
  §D. Gap analysis: cuda_eff vs fp16_eff, bound by compute/mem,
      root-cause tag per shape
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load both bench runs
survey = json.loads((ROOT / "logs/r68_multiT_survey/bench.json").read_text())
prefill = json.loads((ROOT / "logs/r68_prefill_survey/bench.json").read_text())

all_recs = []
for src, data in [("survey", survey), ("prefill", prefill)]:
    for r in data["records"]:
        if r.get("kernel") == "end_to_end" and r.get("model") != "Qwen3-0.6B":
            r["_src"] = src
            all_recs.append(r)

# Dedupe: if the same (model,proj,T,d_in,d_out) exists in both, prefer prefill
# (newer).  For T in {1,8,32,128,512}, only survey has it; for T in
# {1024,2048,4096,8192}, only prefill has it.
index = {}
for r in all_recs:
    key = (r["model"], r["proj"], r["T"], r["d_in"], r["d_out"])
    if key not in index or r["_src"] == "prefill":
        index[key] = r
rows = list(index.values())
print(f"Loaded {len(rows)} unique (model,proj,T) e2e records from both runs")

# ============================================================
# RTX 4090 roofline constants (match qwen3_roofline_report.py)
# ============================================================
HBM_BW_GBS = 1008.0
FP16_TC_TFLOPS = 165.2
INT4_TC_TOPS = 660.6
ACHIEVABLE_FRACTION = 0.85
eff_hbm = HBM_BW_GBS * ACHIEVABLE_FRACTION
eff_fp16 = FP16_TC_TFLOPS * ACHIEVABLE_FRACTION * 1e12
eff_int4 = INT4_TC_TOPS * ACHIEVABLE_FRACTION * 1e12
BCOL = 128


def fp16_roof_us(T, d_in, d_out):
    flops = 2.0 * T * d_in * d_out
    t_cmp = flops / eff_fp16
    t_mem = 2.0 * (d_in * d_out + T * d_in + T * d_out) / (eff_hbm * 1e9)
    return max(t_cmp, t_mem) * 1e6


def cuda_roof_us(T, d_in, d_out):
    n_groups = d_in // BCOL
    if T == 1:
        bytes_fused = 0.5 * d_in * d_out + 2.0 * d_in + 4.0 * d_out * n_groups + 2.0 * d_out
        t_mem = bytes_fused / (eff_hbm * 1e9)
        t_cmp = 2.0 * d_in * d_out / eff_int4
        return max(t_cmp, t_mem) * 1e6
    bytes_quant = 2.0 * T * d_in + 0.5 * T * d_in + 2.0 * T + 4.0 * T * n_groups
    t_quant = bytes_quant / (eff_hbm * 1e9) * 1e6
    bytes_gemm = (0.5 * d_in * d_out + 0.5 * T * d_in + 4.0 * d_out * n_groups
                  + 4.0 * T * n_groups + 2.0 * T + 2.0 * T * d_out)
    t_mem = bytes_gemm / (eff_hbm * 1e9)
    t_cmp = 2.0 * T * d_in * d_out / eff_int4
    t_gemm = max(t_cmp, t_mem) * 1e6
    return t_quant + t_gemm


def gemm_bound(T, d_in, d_out):
    t_cmp = 2.0 * T * d_in * d_out / eff_int4 * 1e6
    n_groups = d_in // BCOL
    bytes_gemm = (0.5 * d_in * d_out + 0.5 * T * d_in + 4.0 * d_out * n_groups
                  + 4.0 * T * n_groups + 2.0 * T + 2.0 * T * d_out)
    t_mem = bytes_gemm / (eff_hbm * 1e9) * 1e6
    return "compute" if t_cmp >= t_mem else "mem"


PARAMS_B = {"Qwen3-1.7B": 1.7, "Qwen3-4B": 4.0, "Qwen3-8B": 8.0,
            "Qwen3-14B": 14.0, "Qwen2.5-32B": 32.0, "LLaMA3-70B": 70.0}


def row_stats(r):
    T, d_in, d_out = r["T"], r["d_in"], r["d_out"]
    fp16_r = fp16_roof_us(T, d_in, d_out)
    cuda_r = cuda_roof_us(T, d_in, d_out)
    r["fp16_roof"] = fp16_r
    r["cuda_roof"] = cuda_r
    r["fp16_eff"] = fp16_r / r["fp16_us"] if r["fp16_us"] > 0 else 0.0
    r["cuda_eff"] = cuda_r / r["cuda_us"] if r["cuda_us"] > 0 else 0.0
    r["sp"] = r["cuda_speedup_vs_fp16"]
    r["bound"] = gemm_bound(T, d_in, d_out)
    return r


for r in rows:
    row_stats(r)

# ============================================================
# Output
# ============================================================
out = []


def P(s=""):
    out.append(s)


P("# r68 prefill-scenario unified analysis (T=1..8192)")
P("")
P(f"Combined records: {len(rows)}")
models = sorted({r['model'] for r in rows}, key=lambda m: PARAMS_B.get(m, 999))
Ts = sorted({r['T'] for r in rows})
P(f"Models: {models}")
P(f"Ts: {Ts}")
P("")

# ============================================================
# §A. Global T-sweep (every model at every T): median speedup + cuda_eff
# ============================================================
P("## §A. Global T-sweep (median speedup / median cuda_eff across 5 projs per (model,T))")
P("")
P("### Speedup vs FP16")
P("")
header = "| model | params |" + " | ".join(f"T={T}" for T in Ts) + " |"
P(header)
P("|---|---:|" + "|".join(["---:"] * len(Ts)) + "|")
for m in models:
    cells = [f"{m}", f"{PARAMS_B.get(m, 0):.1f}B"]
    for T in Ts:
        rs = [r for r in rows if r['model'] == m and r['T'] == T]
        if rs:
            cells.append(f"{statistics.median(r['sp'] for r in rs):.2f}×")
        else:
            cells.append("-")
    P("| " + " | ".join(cells) + " |")
P("")

P("### CUDA efficiency (cuda_roof / cuda_us)")
P("")
P(header)
P("|---|---:|" + "|".join(["---:"] * len(Ts)) + "|")
for m in models:
    cells = [f"{m}", f"{PARAMS_B.get(m, 0):.1f}B"]
    for T in Ts:
        rs = [r for r in rows if r['model'] == m and r['T'] == T]
        if rs:
            cells.append(f"{statistics.median(r['cuda_eff'] for r in rs)*100:.0f}%")
        else:
            cells.append("-")
    P("| " + " | ".join(cells) + " |")
P("")

# ============================================================
# §B. Cross-model comparison at each T
# ============================================================
P("## §B. Cross-model comparison at each T (how does speedup scale with model size?)")
P("")
P("For each T, we list median speedup by model (ordered by param count).")
P("If speedup rises with model size, the kernel is benefiting from bigger tile utilisation.")
P("If it stays flat or drops, FP16 cuBLAS also scales and we're not gaining.")
P("")
P("| T | " + " | ".join(m.replace("Qwen3-", "").replace("Qwen2.5-", "")
                         .replace("LLaMA3-", "") for m in models) + " |")
P("|---:|" + "|".join(["---:"] * len(models)) + "|")
for T in Ts:
    cells = [str(T)]
    for m in models:
        rs = [r for r in rows if r['model'] == m and r['T'] == T]
        cells.append(f"{statistics.median(r['sp'] for r in rs):.2f}×"
                     if rs else "-")
    P("| " + " | ".join(cells) + " |")
P("")

# ============================================================
# §C. Per-shape deep-dive at T=2048 (representative mid-prefill)
# ============================================================
if 2048 in set(r['T'] for r in rows):
    P("## §C. Per-shape deep-dive at T=2048 (representative prefill point)")
    P("")
    P("| model | proj | shape | fp16_us | cuda_us | speedup | cuda_eff | "
      "bound | gap_reason |")
    P("|---|---|---|---:|---:|---:|---:|:---:|---|")
    tgt = sorted(
        [r for r in rows if r['T'] == 2048],
        key=lambda r: (PARAMS_B.get(r['model'], 999),
                       ['q_proj','kv_proj','o_proj','gate_up_proj','down_proj']
                       .index(r['proj']) if r['proj'] in
                       ['q_proj','kv_proj','o_proj','gate_up_proj','down_proj']
                       else 99)
    )
    for r in tgt:
        # Diagnose gap
        reason = []
        if r['cuda_eff'] < 0.30:
            reason.append("low cuda_eff (kernel sub-par)")
        if r['fp16_eff'] > 1.05:
            reason.append("fp16 exceeds vendor roof (L2 reuse)")
        if r['bound'] == "mem" and r['cuda_eff'] > 0.70:
            reason.append("mem-bound / near ceiling")
        if r['bound'] == "compute" and r['cuda_eff'] < 0.40:
            reason.append("compute-bound / MMA starvation")
        if r['sp'] >= 1.5:
            reason.append("clear win")
        reason_str = "; ".join(reason) if reason else "-"
        P(f"| {r['model']} | {r['proj']} | {r['d_in']}→{r['d_out']} | "
          f"{r['fp16_us']:.1f} | {r['cuda_us']:.1f} | {r['sp']:.2f}× | "
          f"{r['cuda_eff']*100:.0f}% | {r['bound']} | {reason_str} |")
    P("")

# ============================================================
# §D. Explain the gap
# ============================================================
P("## §D. Root-cause analysis — why model-size and T influence speedup")
P("")
P("### D.1 Why does speedup rise with model size? (e.g. 1.7B 0.78× → 8B 1.35×)")
P("")
P("Hypothesis 1: **Grid utilisation**.  Bigger d_out = more m-tiles, "
  "better SM utilisation.  At T=2048, kv_proj has the same d_out=2048 "
  "across all models (same m-grid size), so this only explains q/gu/dn.")
P("")
P("Hypothesis 2: **Launch-overhead amortisation**.  Each cuda kernel has "
  "~15us activation_quant launch floor.  Small-shape kernels "
  "(cuda_us < 40us) are 40-50% launch overhead.  Bigger models have "
  "bigger work per launch so overhead fraction drops.")
P("")
P("Check: compute `launch_overhead_fraction = 15us / cuda_us`.")
P("")
for m in models:
    rs = [r for r in rows if r['model'] == m and r['T'] == 2048]
    if not rs:
        continue
    cuda_med = statistics.median(r['cuda_us'] for r in rs)
    sp_med = statistics.median(r['sp'] for r in rs)
    overhead_frac = 15.0 / cuda_med * 100
    P(f"- {m:<14} T=2048: cuda_us={cuda_med:.1f}, overhead={overhead_frac:.0f}%, "
      f"median sp={sp_med:.2f}×")
P("")

P("### D.2 Why does speedup drop as T grows?")
P("")
P("At T=1 median sp ≈ 2.0× (W4A4 gemv kills fp16 gemv).  At T=4096+ "
  "median sp ≈ 0.9-1.0× (fp16 tensorcore gets full benefit, W4A4 also "
  "compute-bound).  The transition is at T ≈ 128-512 where cuBLAS "
  "switches from gemv-optimised to tensor-core GEMM internally.")
P("")
for T in Ts:
    rs = [r for r in rows if r['T'] == T]
    if not rs:
        continue
    sp_med = statistics.median(r['sp'] for r in rs)
    ceff_med = statistics.median(r['cuda_eff'] for r in rs)
    feff_med = statistics.median(r['fp16_eff'] for r in rs)
    compute_bound = sum(1 for r in rs if r['bound'] == 'compute')
    P(f"- T={T:>5}: median sp={sp_med:.2f}×  cuda_eff={ceff_med*100:.0f}%  "
      f"fp16_eff={feff_med*100:.0f}%  compute-bound={compute_bound}/{len(rs)}")
P("")

# ============================================================
# Write report
# ============================================================
report_path = ROOT / "logs/r68_prefill_survey/prefill_analysis.md"
report_path.write_text("\n".join(out))
print(f"\nWrote {report_path}")
print(f"Report length: {len(out)} lines, {sum(len(l) for l in out)} chars")
