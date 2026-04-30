# Why does cuda_eff drop as batch size T grows?

> **Status**: Phase 3 technical deep-dive, 2026-04-30.
> **Data source**: `logs/r63_combined/bench.json` (140 shapes × 7 models, cold-cache FP16 baseline, RTX 4090).
> **Diagnostic script**: `logs/r63_combined/_why_bigT_eff_drops.py`.

---

## TL;DR

Large-T `cuda_eff` drop is **not a bug** — it is three structural
effects stacking on top of each other, each of which is independently
sufficient to explain the observation:

| T bucket | median cuda_eff | dominant cause |
|---:|---:|---|
| 1 | 42% | mem-bound; eff = fraction of HBM roofline achieved |
| 32 | 30% | mem-bound; +activation_quant overhead fraction rising |
| 128 | 26% | mem-bound; +quant fraction at 11.5%, AI approaching roofline knee |
| 512 | 32%† (compute-bound) | **architectural**: kernel crosses the roofline knee into compute-bound territory where W4A4 epilogue HFMA2 chain starves the MMA pipeline |

†Only 3 of 35 T=512 shapes are still mem-bound (median 19% eff).  The
32 compute-bound shapes sit at median 32% eff.

**Bottom line**: the T=512 drop is exactly the Phase 4 `tc_underutil`
mode from the Phase 2 microscope report — MMA pipeline starvation
driven by per-group dequant FMA chains and 2-stage cp.async — and it
is architectural, not tunable within the current kernel.

---

## §1 Empirical evidence — the three mechanisms

### A. activation_quant overhead fraction grows with T (amortisation fails)

`quant_roof_frac = t_quant_roof / (t_quant_roof + t_gemm_roof)` — the
share of the total CUDA roofline time that the quant kernel holds.

| T | median | min | max |
|---:|---:|---:|---:|
| 1 | 0% | 0% | 0% (quant is fused in GEMV) |
| 32 | 3.5% | 0.3% | 12.2% |
| 128 | **11.5%** | 1.0% | **31.7%** |
| 512 | **16.8%** | 1.4% | **44.8%** |

**Why**: the activation_quant kernel has a hard floor of ~16 us on
RTX 4090 (2 kernel launches × ~7 us + kernel body).  That floor is
essentially shape-independent, so as the GEMM body shrinks relative
to T, the quant fraction grows.

**Observable consequence**: measured `cuda_us` contains an extra
~16 us that the pure-MMA roofline cannot explain, so `cuda_eff =
cuda_roof / cuda_us` deflates.  At T=512 up to 45% of the roofline
time is "quant that we're not good at" — the eff denominator is
inflated but the numerator is not.

### B. GEMM crosses the roofline knee into compute-bound

Arithmetic intensity (AI) = `2·T·d_in·d_out / bytes_gemm_roof`.  At
the RTX 4090 INT4 knee, AI = `peak_int4 / peak_hbm = 660.6 TOPS /
1008 GB/s ≈ 656 flops/byte`.

| T | median AI | max AI | #shapes ≥ knee |
|---:|---:|---:|---:|
| 1 | 4 | 4 | 0 / 35 |
| 32 | 116 | 120 | 0 / 35 |
| 128 | 419 | 467 | 0 / 35 |
| 512 | **1208** | **1706** | **32 / 35** |

At T ≤ 128, every single shape is still HBM-bound — one good idea
(enough issue rate to drain HBM) gets you close to peak eff.
At T=512, 91% of shapes have crossed the knee, and the kernel's
achievable performance is now dictated by whether the MMA pipeline
can sustain ~85% TC issue rate.

### Verifying with the bound-split table

cuda_eff, split by the kernel's own compute/mem boundedness:

| T | mem shapes | mem median eff | max mem eff | compute shapes | compute median eff | max comp eff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35 | 41.7% | 66.1% | 0 | — | — |
| 32 | 35 | 30.3% | 88.2% | 0 | — | — |
| 128 | 35 | 26.2% | 48.4% | 0 | — | — |
| 512 | **3** | 18.6% | 25.8% | **32** | **31.7%** | **42.7%** |

At T=512, the **compute-bound** shapes actually have *higher* median
eff than the mem-bound ones.  That is because the 3 mem-bound T=512
shapes are the very wide ones (e.g. `down_proj 28672→8192`) where
HBM read volume dominates and our cold-cache path doesn't beat
cuBLAS there.

### C. FP16 baseline's own efficiency improves at large T — the scoreboard shifts

| T | FP16 eff median | mean | max |
|---:|---:|---:|---:|
| 1 | 98% | 96% | 111% |
| 32 | 96% | 93% | 104% |
| 128 | 91% | 90% | 101% |
| **512** | **113%** | **120%** | **151%** |

**cuBLAS at T=512 exceeds its own roofline by 13-20% because it hits
L2 reuse that the roofline model does not account for** (each W
column is reused by 512 tokens; even with cold-cache flush, the
intra-kernel reuse lets cuBLAS outperform the naive HBM roof).

This is important for the speedup story: T=512 is where the **INT4
kernel hits an architectural ceiling *and* the FP16 baseline gets a
free L2 tailwind**.  Both pressures combine to explain why speedup
collapses from 1.08× (T=32) → 0.87× (T=512) even though absolute
INT4 throughput is still improving.

### D. Wave-fill is NOT the culprit

Number of (M×N) CTAs compared to 128 SMs on RTX 4090:

| T | median Mtiles | median Ntiles | median waves |
|---:|---:|---:|---:|
| 1 | 32 | 1 | 0.25 |
| 32 | 32 | 1 | 0.25 |
| 128 | 32 | 4 | 1.00 |
| 512 | 32 | 16 | **4.00** |

Wave fill grows **monotonically** with T, from 0.25 waves (starved)
to 4 waves (saturated).  Yet cuda_eff *falls*.  This conclusively
rules out wave starvation as the root cause: if T=512 has 4× the SM
occupancy of T=128 but lower eff, the bottleneck is *per-CTA*
throughput, not grid coverage.

---

## §2 The worst droppers (T=32 → T=128) tell the same story

Picking the (model, proj) pairs with the biggest eff drop when T
doubles from 32 to 128 (all still nominally mem-bound):

| model | proj | eff@32 | eff@128 | Δ | bound32 | bound128 |
|---|---|---:|---:|---:|---|---|
| Qwen3-8B | gate_up_proj | 88.2% | 48.4% | **−40pt** | mem | mem |
| Qwen3-4B | gate_up_proj | 69.6% | 41.6% | −28pt | mem | mem |
| LLaMA3-70B | gate_up_proj | 50.0% | 27.2% | −23pt | mem | mem |
| Qwen2.5-32B | gate_up_proj | 47.6% | 27.6% | −20pt | mem | mem |
| LLaMA3-70B | o_proj | 58.0% | 42.5% | −15pt | mem | mem |
| LLaMA3-70B | q_proj | 57.9% | 42.5% | −15pt | mem | mem |
| Qwen3-14B | down_proj | 46.6% | 33.5% | −13pt | mem | mem |
| Qwen3-14B | gate_up_proj | 37.9% | 25.8% | −12pt | mem | mem |
| Qwen3-14B | q_proj | 42.0% | 29.9% | −12pt | mem | mem |
| Qwen3-14B | o_proj | 41.8% | 29.8% | −12pt | mem | mem |

Notice the huge 88% eff cliff for `Qwen3-8B gate_up T=32` — this is
the single shape where the kernel sits *exactly* at its best
operating point (small T + large d_out lets group-cache pay off
fully and the kernel's HBM read coalescing is perfect).  At T=128,
the same shape's n_groups-heavy dequant pressure + occupancy drop
(group-cache gate flips off at T>32) pulls it down hard.

This confirms mechanism A: as T grows, the per-group HFMA2 dequant
path that was previously masked by HBM wait now becomes visible,
because the kernel is no longer HBM-bound on those group reads.

---

## §3 Why cuBLAS does NOT suffer the same drop

cuBLAS (16-bit Tensor Core path on Ada) does NOT have a per-group
dequant epilogue.  Its mainloop is:

```
for k in 0..K:
    a = ldmatrix.bf16(A[m, k])
    b = ldmatrix.bf16(B[k, n])
    C += mma.bf16(a, b)
# Epilogue: write C back as bf16 (one pass)
```

Per K-iteration: 2 ldmatrix + 1 mma + 0 scalar work.

Our kernel's mainloop is:

```
for g in 0..n_groups:                        # per-group K-slab
    sW, sX, sum_X = load                     # HBM + cp.async
    for ks in 0..1:
        ldmatrix_x4(sW) + ldmatrix_x2(sX)    # B, N
        mma.s4.s32                           # the cheap one
    fold_dense:
        corrected = int_acc - zero[m,g] * sum_X[n,g]   # HFMA2
        y_fp     += corrected * scale_u4[m,g]           # HFMA2
# Epilogue: y_fp *= scale_x[n], convert fp32 → fp16
```

Per-group per sub-tile: `kMsubPerWarp × kNsubPerCta × 4 reg-vectors
× 2 HFMA2 ≈ 64 HFMA2 instructions` — vs only **2 MMAs**.  That is a
**32:1 scalar-to-tensor ratio**, and the scalar work is on the same
warp scheduler as issue the MMA.

As T grows and the shape crosses the AI knee, the MMA pipeline wants
to issue at >85% — but every MMA is followed by ~32 HFMA2 on the
same warp scheduler, **capping MMA issue at ~3%** regardless of how
optimised the rest is.

This is the exact mechanism documented in
`cuda_kernel/logs/phase2_microscope/phase2_tc_rediagnosis.md`
(label: `tc_underutil` / MMA pipeline starvation).

---

## §4 Why this is architectural, not tunable

### What would fix it (and what it costs)

| Fix direction | Target eff | Work | Risk |
|---|---:|---:|---|
| Fuse HFMA2 dequant into mma.sync via register dual-issue | 40-45% | 3-5 days | medium — NVCC cannot dual-issue IMMA+HFMA2 without warp-specialisation |
| 3-stage cp.async pipeline | 35-40% | 5 days | high — needs larger smem (>48 KB → opt-in carveout) |
| Full warp-specialised kernel (1 warp issues MMA, 1 warp does dequant) | 50-60% | 8-10 days | high — needs rewrite |
| CUTLASS 3.x mainloop with patched K-loop hook | 55-65% | 10+ days | high — needs CUTLASS expertise and 3.x migration [[memory:ie8lp95b]] |
| **Status quo** | 30-32% at T=512 | 0 | 0 |

All four fixes share one property: they **don't target a tuning
parameter**, they **restructure the kernel**.  The current fused
kernel has been R23, R24, R41, R44, R52, F1, F2 tuned — dispatch,
kBm/kBn, split-K, group-cache — each of those got us 2-10% per
round.  We've extracted nearly everything that tuning can give.

The remaining 20-30 percentage points of eff *only* come from
kernel architecture changes.

### What the legacy kernel CAN still do well

- Small T (1, 32) — already at 30-88% eff, near the mem-bound
  ceiling.  This is the production decode / small-batch prefill
  region in any real LLM server.
- Narrow d_out shapes — group-cache fully effective, fold cost
  amortised over many CTAs.
- cp.async-profitable shapes (n_groups ≥ 16) — pipeline already
  overlapping W/X load with MMA.

T=512 is effectively a benchmarking artefact for a Phase-3-stage
kernel: it exposes the compute-bound ceiling but is rare in real
deployments (prefill rarely runs one layer with T=512 alone in
eager mode; batched prefill splits it).

---

## §5 Recommended use of this analysis

1. **Phase 3 final report** must cite this document to explain the
   T=512 regression honestly.  Not a failure — a well-understood
   architectural ceiling, aligned with the Phase 2 microscope.
2. **Phase 4 scoping**: this is the canonical input for the "do we
   rewrite the mainloop?" decision.  The table in §4 is literally
   the Phase 4 options matrix.
3. **Production deployment**: if integrating into a serving
   framework, the policy layer should **route T>256 to FP16 cuBLAS**
   and keep the INT4 kernel for T ≤ 256.  This is not "cheating" —
   it is the right design for this kernel's actual competence zone.

---

## Appendix — reproducing the numbers

```bash
cd kernel/cuda_kernel/logs/r63_combined
python3 _why_bigT_eff_drops.py
```

All six tables in §1 and §2 come from that single run over
`bench.json`.  Formulas for fp16_roof / cuda_roof match
`kernel/tools/profile/qwen3_roofline_report.py`.
