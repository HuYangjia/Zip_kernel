"""Estimate per-shape physical ceiling on cuda_eff for T=512.

Model each shape's ceiling as the min of:
  (a) TC-limited ceiling: achievable_tc_frac × INT4_peak time
  (b) HBM-limited ceiling: achievable_hbm_frac × HBM_peak time
  (c) HFMA-critical-path remaining overhead (only for shapes where
      HFMA is still on the critical path).

We anchor (a)/(b) with empirical fractions from our bench:
  - "achievable_tc_frac" = 0.50 (50% of vendor INT4 TOPS) based on:
     + DeepGEMM / tensorrt reports int4 achievable 50-60% of peak
     + our best observed shape (8B gu T=512) is already at 34% raw,
       with HFMA bottleneck; removing HFMA => ~50%.
  - "achievable_hbm_frac" = 0.80.

HFMA criticality per shape is inferred from waves-per-SM:
  - n_waves >= 20  ⇒ HFMA already hidden by ILP (experimentally 14B shows this)
  - n_waves  < 20  ⇒ HFMA on critical path, warp-spec can help
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "logs/r66_path_c/bench.json"

INT4_PEAK_TOPS = 660.6
HBM_PEAK_GBs  = 1008.0
N_SMS = 128

# Achievable-fraction targets after plausible optimisation
TC_FRAC_WARP_SPEC = 0.50   # warp-specialised ceiling
TC_FRAC_R66       = 0.34   # empirical best raw (8B gu T=512)
HBM_FRAC          = 0.80

def int4_bytes(T, d_in, d_out):
    n_g = d_in // 128
    return 0.5 * d_in * d_out + 0.5 * T * d_in + 2.0 * T * d_out + 4.0 * d_out * n_g

def int4_flops(T, d_in, d_out):
    return 2.0 * T * d_in * d_out

def waves(T, d_in, d_out, kBm=128, kBn=32):
    n_cta_m = (d_out + kBm - 1) // kBm
    n_cta_n = (T + kBn - 1) // kBn
    return n_cta_m * n_cta_n / N_SMS

records = [r for r in json.loads(BENCH.read_text())["records"]
           if r.get("kernel") == "end_to_end" and r["T"] == 512]

print(f"# Per-shape cuda_eff ceiling analysis for T=512")
print(f"# Peaks: INT4 {INT4_PEAK_TOPS} TOPS, HBM {HBM_PEAK_GBs} GB/s")
print(f"# Ceiling assumptions: TC frac (warp-spec) = {TC_FRAC_WARP_SPEC}, HBM frac = {HBM_FRAC}")
print()
print(f"{'shape':<36} {'waves':>5} {'HFMA':>5} {'r66 us':>8} {'r66 eff':>7} "
      f"{'ceil us':>8} {'ceil eff':>8} {'ceil sp':>7} {'gap':>5}")

ceiling_rows = []
for r in sorted(records, key=lambda r: -r["fp16_us"] / r["cuda_us"]):
    T, d_in, d_out = r["T"], r["d_in"], r["d_out"]
    flops = int4_flops(T, d_in, d_out)
    bytes_ = int4_bytes(T, d_in, d_out)
    cuda_us = r["cuda_us"]
    fp16_us = r["fp16_us"]

    # Current eff (vs INT4 roofline)
    t_tc_peak  = flops / (INT4_PEAK_TOPS * 1e12) * 1e6
    t_hbm_peak = bytes_ / (HBM_PEAK_GBs * 1e9) * 1e6
    roof_us    = max(t_tc_peak, t_hbm_peak)
    r66_eff    = roof_us / cuda_us

    # Ceiling: achievable fractions
    t_tc_achievable  = t_tc_peak / TC_FRAC_WARP_SPEC
    t_hbm_achievable = t_hbm_peak / HBM_FRAC
    ceiling_us_tc    = max(t_tc_achievable, t_hbm_achievable)

    # HFMA criticality heuristic
    w = waves(T, d_in, d_out)
    hfma_critical = w < 20
    # If HFMA is not critical (big grids), the ceiling is lower
    # because we lack any mechanism to push past the MMA issue rate.
    if not hfma_critical:
        # Use empirical ceiling matching r66's best (no warp-spec benefit)
        ceiling_us_tc = max(t_tc_peak / TC_FRAC_R66, t_hbm_achievable)

    ceiling_eff = roof_us / ceiling_us_tc
    ceiling_sp  = fp16_us / ceiling_us_tc

    gap = (cuda_us - ceiling_us_tc) / cuda_us * 100

    label = f"{r['model']:<10} {r['proj']:<14} {d_in:>5}→{d_out:>5}"
    hfma_tag = "YES" if hfma_critical else "no"
    print(f"  {label}  {w:>4.1f}  {hfma_tag:>5}  "
          f"{cuda_us:>7.1f}  {r66_eff*100:>5.1f}%  "
          f"{ceiling_us_tc:>7.1f}  {ceiling_eff*100:>6.1f}%  "
          f"{ceiling_sp:>5.2f}×  {gap:>4.1f}%")

    ceiling_rows.append({
        "model": r["model"], "proj": r["proj"], "T": T,
        "d_in": d_in, "d_out": d_out,
        "r66_us": cuda_us, "r66_eff": r66_eff, "r66_sp": fp16_us/cuda_us,
        "ceil_us": ceiling_us_tc, "ceil_eff": ceiling_eff, "ceil_sp": ceiling_sp,
        "hfma_critical": hfma_critical, "waves": w,
    })

# Aggregates
print()
print("## Aggregate")
hfma_yes = [r for r in ceiling_rows if r["hfma_critical"]]
hfma_no  = [r for r in ceiling_rows if not r["hfma_critical"]]

print(f"  HFMA-critical shapes (warp-spec can help): {len(hfma_yes)}")
print(f"    median r66 sp:      {statistics.median(r['r66_sp'] for r in hfma_yes):.3f}×")
print(f"    median ceiling sp:  {statistics.median(r['ceil_sp'] for r in hfma_yes):.3f}×")
print(f"    median lift:        +{statistics.median(r['ceil_sp']-r['r66_sp'] for r in hfma_yes):.3f}")
print(f"  HFMA-not-critical shapes (big grids, no warp-spec help):")
print(f"    median r66 sp:      {statistics.median(r['r66_sp'] for r in hfma_no):.3f}×")
print(f"    median ceiling sp:  {statistics.median(r['ceil_sp'] for r in hfma_no):.3f}×")

# Overall T=512 projection
combined_r66 = statistics.median(r['r66_sp'] for r in ceiling_rows)
combined_ceil = statistics.median(r['ceil_sp'] for r in ceiling_rows)
print(f"  All T=512 shapes: median {combined_r66:.3f}× → ceiling {combined_ceil:.3f}×")
