"""Diagnose *why* CUDA efficiency drops as T grows.

Hypotheses to test numerically (hard data only, no hand-waving):

H1. Bound-type switch: at T=1/32 the kernel is HBM-bound (INT4 halves
    the W traffic → easy to saturate); at T=128/512 the kernel crosses
    into compute-bound (MAC = 2·T·d_in·d_out grows linearly with T),
    where the 4090 INT4 peak 660 TOPS requires ~100% MMA issue to match,
    which our kernel can't reach (mma pipeline starvation bd78lejo).

H2. activation_quant overhead fraction *decreases* at large T (kernel
    body gets bigger, fixed 16us floor gets amortised).  So H1 must
    be the dominant cause.

H3. Wave-fill varies by T: small T ⇒ fewer N-tiles ⇒ wave-starved
    (bad for compute-bound kernel but fine for mem-bound where a few
    SMs can saturate HBM anyway).  Large T ⇒ full wave but compute-
    bound ⇒ MMA pipeline is the bottleneck.

H4. L2 reuse: at small T, one W-tile serves only few tokens; at large
    T, one W-tile serves many tokens → higher arithmetic intensity →
    more compute-bound.

H5. FP16 baseline efficiency *increases* at T=512 (cuBLAS hits 113%
    median eff vs its own roof, i.e. near TC peak).  So cuBLAS is
    getting better AND we're getting worse → 2x effect on speedup.
"""
import json, statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
BF = HERE / "bench.json"
data = json.loads(BF.read_text())

# ------------------------------------------------------------------
# Recompute rooflines so we can attach bound-type per row (we don't
# trust the json's bound field — not always populated).
# ------------------------------------------------------------------
HBM_BW  = 1008e9 / 1e6          # GB/s → bytes/us
FP16_TC = 165.2e12 / 1e6        # TFLOPS → ops/us
INT4_TC = 660.6e12 / 1e6
ACHIEVE = 0.85
EFF_HBM  = HBM_BW  * ACHIEVE
EFF_FP16 = FP16_TC * ACHIEVE
EFF_INT4 = INT4_TC * ACHIEVE
BCOL = 128

def fp16_roof(T, d_in, d_out):
    flops = 2.0 * T * d_in * d_out
    bytes_ = 2.0 * (d_in*d_out + T*d_in + T*d_out)
    return max(flops / EFF_FP16, bytes_ / EFF_HBM)

def cuda_roof(T, d_in, d_out):
    n_g = d_in // BCOL
    # quant (T>=2)
    bytes_q = 2*T*d_in + 0.5*T*d_in + 2*T + 4*T*n_g
    t_q = bytes_q / EFF_HBM if T >= 2 else 0.0
    # gemm
    if T == 1:
        bytes_g = 0.5*d_in*d_out + 2*d_in + 4*d_out*n_g + 2*d_out
        comp    = 2.0 * d_in * d_out / EFF_INT4
    else:
        bytes_g = (0.5*d_in*d_out + 0.5*T*d_in + 4*d_out*n_g
                   + 4*T*n_g + 2*T + 2*T*d_out)
        comp    = 2.0 * T * d_in * d_out / EFF_INT4
    t_g = max(comp, bytes_g / EFF_HBM)
    return dict(t_q=t_q, t_g=t_g, t_total=t_q + t_g,
                bytes_g=bytes_g, comp_g=comp)

def fp16_bound(T, d_in, d_out):
    flops = 2.0 * T * d_in * d_out
    bytes_ = 2.0 * (d_in*d_out + T*d_in + T*d_out)
    return "compute" if flops / EFF_FP16 >= bytes_ / EFF_HBM else "mem"

def cuda_gemm_bound(T, d_in, d_out):
    roof = cuda_roof(T, d_in, d_out)
    return "compute" if roof["comp_g"] >= roof["bytes_g"] / EFF_HBM else "mem"

# ------------------------------------------------------------------
# Build per-shape rows with added diagnostics
# ------------------------------------------------------------------
rows = []
for r in data["records"]:
    if r.get("kernel") != "end_to_end":
        continue
    T, d_in, d_out = r["T"], r["d_in"], r["d_out"]
    fp16_us = r["fp16_us"]
    cuda_us = r["cuda_us"]
    fr = fp16_roof(T, d_in, d_out)
    cr = cuda_roof(T, d_in, d_out)
    fp16_eff = fr / fp16_us
    cuda_eff = cr["t_total"] / cuda_us
    quant_frac = cr["t_q"] / cr["t_total"] if cr["t_total"] > 0 else 0
    rows.append({
        "model": r["model"], "proj": r["proj"], "T": T,
        "d_in": d_in, "d_out": d_out,
        "fp16_us": fp16_us, "cuda_us": cuda_us,
        "fp16_roof": fr, "cuda_roof": cr["t_total"],
        "fp16_eff": fp16_eff, "cuda_eff": cuda_eff,
        "fp16_bound": fp16_bound(T, d_in, d_out),
        "cuda_bound": cuda_gemm_bound(T, d_in, d_out),
        "quant_roof_frac": quant_frac,
        "n_cta_m": (d_out + 127) // 128,
        "n_cta_n32": (T + 31) // 32,
        "n_cta_n64": (T + 63) // 64,
    })

# ------------------------------------------------------------------
# §1 — cuda_eff by T, split by bound type (tests H1)
# ------------------------------------------------------------------
print("# §1 cuda_eff by T, split by cuda-kernel's own compute/mem bound")
print()
print(f"{'T':>4}  {'N':>3}  {'mem-bound':>28}  {'compute-bound':>28}")
print(f"{'':4}  {'':3}  {'n':>3} {'median eff':>10} {'max eff':>8}  {'n':>3} {'median eff':>10} {'max eff':>8}")
by_T_bound = defaultdict(lambda: defaultdict(list))
for r in rows:
    by_T_bound[r["T"]][r["cuda_bound"]].append(r["cuda_eff"])
for T in sorted(by_T_bound):
    mem = by_T_bound[T]["mem"]
    com = by_T_bound[T]["compute"]
    tot = len(mem) + len(com)
    print(f"{T:>4}  {tot:>3}  "
          f"{len(mem):>3} {statistics.median(mem)*100 if mem else 0:>9.1f}% "
          f"{max(mem)*100 if mem else 0:>7.1f}%  "
          f"{len(com):>3} {statistics.median(com)*100 if com else 0:>9.1f}% "
          f"{max(com)*100 if com else 0:>7.1f}%")

# ------------------------------------------------------------------
# §2 — quant overhead roof fraction by T (tests H2)
# ------------------------------------------------------------------
print()
print("# §2 activation_quant roofline fraction by T (quant_roof / total_roof)")
print(f"{'T':>4}  {'n':>3}  {'median frac':>12} {'min':>8} {'max':>8}")
byT_qf = defaultdict(list)
for r in rows:
    byT_qf[r["T"]].append(r["quant_roof_frac"])
for T in sorted(byT_qf):
    vs = byT_qf[T]
    print(f"{T:>4}  {len(vs):>3}  {statistics.median(vs)*100:>11.1f}% "
          f"{min(vs)*100:>7.1f}% {max(vs)*100:>7.1f}%")

# ------------------------------------------------------------------
# §3 — FP16 eff by T (tests H5, why speedup fraction worsens too)
# ------------------------------------------------------------------
print()
print("# §3 FP16 (cuBLAS) efficiency by T  — the BASELINE getting better")
print(f"{'T':>4}  {'n':>3}  {'median eff':>10} {'mean':>8} {'max':>8}")
byT_fp = defaultdict(list)
for r in rows:
    byT_fp[r["T"]].append(r["fp16_eff"])
for T in sorted(byT_fp):
    vs = byT_fp[T]
    print(f"{T:>4}  {len(vs):>3}  {statistics.median(vs)*100:>9.1f}% "
          f"{statistics.mean(vs)*100:>7.1f}% {max(vs)*100:>7.1f}%")

# ------------------------------------------------------------------
# §4 — Wave-fill indicator (H3): how many N-tiles vs 128 SMs
# ------------------------------------------------------------------
print()
print("# §4 Wave fill: how many (M×N) CTAs vs 128 SMs (kBn=32 default path)")
print(f"{'T':>4}  {'n':>3}  {'median_Mtiles':>13} {'median_Ntiles':>13} {'median_waves':>13}")
byT_wv = defaultdict(list)
for r in rows:
    byT_wv[r["T"]].append((r["n_cta_m"], r["n_cta_n32"],
                           r["n_cta_m"] * r["n_cta_n32"] / 128.0))
for T in sorted(byT_wv):
    vs = byT_wv[T]
    mm = statistics.median([v[0] for v in vs])
    mn = statistics.median([v[1] for v in vs])
    mw = statistics.median([v[2] for v in vs])
    print(f"{T:>4}  {len(vs):>3}  {mm:>12.0f} {mn:>12.0f} {mw:>12.2f}")

# ------------------------------------------------------------------
# §5 — Arithmetic Intensity (AI) by T: GEMM-only
#     AI = 2·T·d_in·d_out / bytes_g  (flops per HBM byte)
#     At 4090 roofline-knee: AI_knee = INT4_peak / HBM_peak = 660/1008 ≈ 656 flops/B
# ------------------------------------------------------------------
print()
print("# §5 Arithmetic Intensity by T  (AI_knee @ 4090/INT4 = 656 flops/B)")
print(f"{'T':>4}  {'n':>3}  {'median AI':>10} {'max AI':>10} {'%>knee':>8}")
byT_ai = defaultdict(list)
for r in rows:
    T, d_in, d_out = r["T"], r["d_in"], r["d_out"]
    cr = cuda_roof(T, d_in, d_out)
    ai = 2.0 * T * d_in * d_out / cr["bytes_g"]
    byT_ai[T].append(ai)
knee = INT4_TC / HBM_BW
for T in sorted(byT_ai):
    vs = byT_ai[T]
    over = sum(1 for v in vs if v >= knee)
    print(f"{T:>4}  {len(vs):>3}  {statistics.median(vs):>10.0f} "
          f"{max(vs):>10.0f} {over}/{len(vs):>3}")

# ------------------------------------------------------------------
# §6 — composite: biggest cuda_eff droppers T=32→128 on same (model,proj)
# ------------------------------------------------------------------
print()
print("# §6 Biggest cuda_eff drops when going T=32 → T=128 (same model, proj)")
by_mp = defaultdict(dict)
for r in rows:
    by_mp[(r["model"], r["proj"])][r["T"]] = r
drops = []
for k, tmap in by_mp.items():
    if 32 in tmap and 128 in tmap:
        d = tmap[128]["cuda_eff"] - tmap[32]["cuda_eff"]
        drops.append((d, k, tmap[32]["cuda_eff"], tmap[128]["cuda_eff"],
                      tmap[32]["cuda_bound"], tmap[128]["cuda_bound"]))
drops.sort()
print(f"{'model':<14} {'proj':<14} {'eff32':>6} {'eff128':>6} {'delta':>8} "
      f"{'bnd32':>6} {'bnd128':>8}")
for d, (m, p), e32, e128, b32, b128 in drops[:10]:
    print(f"{m:<14} {p:<14} {e32*100:>5.1f}% {e128*100:>5.1f}% "
          f"{d*100:>+7.1f}% {b32:>6} {b128:>8}")
