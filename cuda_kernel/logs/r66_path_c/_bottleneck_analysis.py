"""Bottleneck analysis: infer whether T=128/512 shapes are compute-bound
or HBM-bound from r66 bench data, using roofline back-calculation.

No ncu needed — we use measured cuda_us + known FLOPS/bytes to compute
actual TC utilisation and actual HBM bandwidth utilisation, then
compare against RTX 4090 peaks.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "logs/r66_path_c/bench.json"

# RTX 4090 vendor peaks (SM89)
INT4_TC_PEAK_TOPS = 660.6      # INT4 tensor-core
FP16_TC_PEAK_TFLOPS = 165.2    # FP16 tensor-core (baseline reference)
HBM_BW_GBs = 1008.0            # GDDR6X

def load(p):
    return [r for r in json.loads(p.read_text())["records"]
            if r.get("kernel") == "end_to_end"]

def int4_bytes(T, d_in, d_out):
    """Total HBM traffic for the INT4 kernel (estimate, excluding scales)."""
    # W: d_in * d_out * 0.5 (int4 packed)
    # X: T * d_in * 0.5   (int4 packed, post-quant)
    # Y: T * d_out * 2    (fp16 output)
    # scales/zeros: d_out * n_g * 4  (fp16 scale + fp16 zero)
    n_g = d_in // 128
    bytes_ = 0.5 * d_in * d_out + 0.5 * T * d_in + 2.0 * T * d_out + 4.0 * d_out * n_g
    return bytes_

def int4_flops(T, d_in, d_out):
    """Total MAC → FLOPS (counting MAC = 2 FLOPS as standard)."""
    return 2.0 * T * d_in * d_out

records = load(BENCH)

print(f"# Bottleneck analysis  (source: {BENCH.relative_to(ROOT)})")
print(f"# RTX 4090 peaks: INT4 {INT4_TC_PEAK_TOPS} TOPS, HBM {HBM_BW_GBs} GB/s")
print()

# Per-T aggregation
by_T = defaultdict(list)
for r in records:
    T, d_in, d_out = r["T"], r["d_in"], r["d_out"]
    cuda_us = r["cuda_us"]
    flops = int4_flops(T, d_in, d_out)
    bytes_ = int4_bytes(T, d_in, d_out)
    tc_tops = flops / cuda_us / 1e6       # us → TOPS
    hbm_gbs = bytes_ / cuda_us / 1e3      # us → GB/s
    tc_util = tc_tops / INT4_TC_PEAK_TOPS
    hbm_util = hbm_gbs / HBM_BW_GBs
    # Roofline classification
    # A workload is compute-bound if FLOPS/peak > BYTES/peak
    t_compute = flops / (INT4_TC_PEAK_TOPS * 1e12) * 1e6    # us
    t_hbm     = bytes_ / (HBM_BW_GBs * 1e9) * 1e6           # us
    bound = "compute" if t_compute > t_hbm else "HBM"
    by_T[T].append({
        **{k: r[k] for k in ("model", "proj", "T", "d_in", "d_out", "cuda_us", "cuda_speedup_vs_fp16")},
        "tc_util": tc_util,
        "hbm_util": hbm_util,
        "bound": bound,
        "t_compute_roof_us": t_compute,
        "t_hbm_roof_us": t_hbm,
    })

for T in sorted(by_T):
    rows = by_T[T]
    compute_share = sum(1 for r in rows if r["bound"] == "compute") / len(rows)
    print(f"## T = {T}   ({len(rows)} shapes, {compute_share*100:.0f}% roofline compute-bound)")
    print(f"  {'shape':<36} {'us':>7}  {'TC%':>5}  {'HBM%':>5}  {'bound':>7}  {'sp':>5}")
    # Sort by TC util descending
    for r in sorted(rows, key=lambda x: -x["tc_util"]):
        print(f"  {r['model']:<10} {r['proj']:<14} "
              f"{r['d_in']:>5}→{r['d_out']:>5}  "
              f"{r['cuda_us']:>6.1f}  "
              f"{r['tc_util']*100:>4.1f}%  "
              f"{r['hbm_util']*100:>4.1f}%  "
              f"{r['bound']:>7}  "
              f"{r.get('cuda_speedup_vs_fp16', 0):>4.2f}×")
    # Aggregates
    med_tc = statistics.median(r["tc_util"] for r in rows)
    med_hbm = statistics.median(r["hbm_util"] for r in rows)
    print(f"  [median TC util {med_tc*100:.1f}%, median HBM util {med_hbm*100:.1f}%]")
    print()

# Focused summary: compute- vs HBM-bound within T=128 and T=512
print("## Decision summary (T=128 and T=512 only)")
for T in (128, 512):
    rows = by_T[T]
    # How many shapes have HBM util >= 60% (HBM near-saturated)?
    hbm_sat = [r for r in rows if r["hbm_util"] >= 0.60]
    # How many have TC util >= 40% (TC near-saturated — but still has room)?
    tc_near = [r for r in rows if r["tc_util"] >= 0.40]
    # How many are "slow" (speedup < 1.0) — what bottleneck do they hit?
    losers = [r for r in rows if r.get("cuda_speedup_vs_fp16", 1.0) < 1.0]
    print(f"  T={T}:")
    print(f"    {len(rows)} total, {len(hbm_sat)} with HBM ≥ 60% util, "
          f"{len(tc_near)} with TC ≥ 40% util")
    print(f"    {len(losers)} losers (sp<1.0), their median TC util: "
          f"{statistics.median(r['tc_util'] for r in losers)*100 if losers else 0:.1f}%, "
          f"median HBM util: "
          f"{statistics.median(r['hbm_util'] for r in losers)*100 if losers else 0:.1f}%")
