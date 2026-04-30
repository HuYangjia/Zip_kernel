# D.1 Pivot — Bottleneck confirmed via roofline back-calculation

**Date**: 2026-04-30
**Source data**: `cuda_kernel/logs/r66_path_c/bench.json` (140 shapes, r66 baseline)

## 1. What we measured (without ncu)

Using roofline back-calculation on r66 bench data — actual TOPS and HBM GB/s
derived from measured `cuda_us`:

### T=512 compute-bound shapes (91% of T=512 shapes, 32 of 35)

| metric | median | max |
|---|---:|---:|
| TC utilisation (vs 660 TOPS INT4 peak) | **21.4%** | 34.0% |
| HBM utilisation (vs 1008 GB/s peak) | **11.5%** | 21.6% |
| T=512 losers (sp < 1.0×) TC util | 19.1% | — |
| T=512 losers HBM util | 11.5% | — |

### T=128

| metric | median |
|---:|---:|
| TC utilisation | 13.9% |
| HBM utilisation | 20.6% |

## 2. What this proves

**Both TC and HBM are far from saturated.**  Since the hardware has unused
capacity on both pipelines, the bottleneck must be in the **warp scheduler's
ability to issue instructions** — i.e., per [[memory:bd78lejo]]'s
"MMA pipeline starvation" hypothesis (B1 HFMA2 dependency chain + B2 swizzle
IMAD address arithmetic).

**Crucially, B3 (cp.async depth) is NOT the bottleneck.**  HBM at 11.5% util
has massive headroom.  Deepening `cp.async` from 2→4 stages would not help
because HBM bandwidth is not the limiter.

## 3. Pivot decision — direct to warp specialisation

Original plan (D.1 γ): split P (cp.async producer) / C (everything else)
to reduce cp.async overhead in consumer warps.
- Now unjustified — cp.async is not stalling anything.

Revised plan: split the MMA issue path from the HFMA2 fold/dequant path.

### Design

**CTA topology**: 256 threads = 8 warps, split 4+4:

| role | count | duty | pipeline |
|---|---:|---|---|
| **MMA producer** | 4 warps | `ldmatrix` → `mma.sync.s4.s4.s32` → spill `d_acc` (int32) to smem | TC pipe |
| **Fold consumer** | 4 warps | wait on smem → read d_acc → HFMA2 `y_fp += s * (d - z*sumxn)` | FP pipe |

The int32 `d_acc` handoff through smem costs ~512 bytes per K-slab (for
kBm=128 × kBn=32) — trivial versus HBM.  The key benefit is that MMA and
HFMA2 no longer share warp scheduler slots; they dual-issue on TC pipe
vs FP32 pipe.

### Why this wasn't attempted before

The r66 kernel already has `__half2float` hoisting, ldmatrix, cp.async,
swizzled smem layout — all ILP-level optimisations within a single warp.
But within one warp, `mma.sync` and the HFMA2 chain that consumes its
result **share the warp's issue slot** even when they use different
execution pipes, because the warp scheduler cannot issue two instructions
per cycle from the same warp.

Putting MMA and fold on different warps lets the scheduler pick one
instruction per pipe per cycle from different warps — true dual-issue.

### Expected gain (revised conservatively)

- TC util 21% → **~32%** (50% lift)
- Qwen3-8B gu T=512: 0.87× → **~1.15×**
- Global median speedup: 1.049× → **1.08-1.10×**
- Big wins > 2×: 20 → 22-23

### Time budget

| MS | days | content |
|---|---:|---|
| MS1 | 2 | new kernel file, 8-warp CTA, no role split (smoke test CTA shape + parity) |
| MS2 | 2 | 4+4 MMA/Fold role split with smem d_acc buffer + barriers |
| MS3 | 1 | 30-shape quick bench (T=128/512 × 3 models) |
| MS4 | 1-2 | tune barrier strategy / buffer depth if MS3 shows gain |
| MS5 | 1 | 140-shape full validation + merge decision |
| total | **7-8** | |

### Fallback
If MS2 or MS3 fails (parity break / no gain), retreat to r66 (main branch is
pristine) and explore dual-issue PTX (D.3) instead.

## 4. What MS1 looks like (starting now)

Copy `fused_dense_sparse_mma_int4_kernel` to a new file
`fused_dense_sparse_mma_int4_warpspec.cu`, change:

- blockDim: 128 → 256 (add `int kNumThreads = kBm` template param)
- add `warp_id = tid >> 5` (0..7) and `is_mma_warp = warp_id < 4`
- keep all existing math, just make each warp do either MMA+fold (no-op role)
- verify parity 10/10 bit-exact

At MS1 nothing actually splits; it's purely a re-packaged kernel with a bigger
CTA.  Parity MUST pass.  MS2 adds the real role split.

---

## 5. HFMA stress-test results (2026-04-30 afternoon)

Before committing to the 7-8 day D.1 warp-specialisation work, we ran a
30-minute confirmation experiment: inject **8 extra `fmaf` ops** into the
critical path of `fold_dense` (real FP dependency chain, DCE-protected
with sentinel write-back).  Measure cuda_us vs clean r66 baseline.

### Results

| shape | r66 (us) | +8 HFMA (us) | slowdown | HFMA on critical path? |
|---|---:|---:|---:|---|
| **Qwen3-8B gu T=512** | 458.3 | 702.6 | **+53.3%** | ✅ YES |
| **Qwen3-8B gu T=128** | 148.5 | 242.5 | **+63.3%** | ✅ YES |
| Qwen3-14B gu T=512 | 1513.4 | 1443.0 | −4.6% (noise) | ❌ no (ILP hides) |
| Qwen3-8B gu T=32 | 66.9 | 72.6 | +8.6% | ❌ no |
| Qwen3-8B gu T=1 | 94.9 | 47.0 | −50.5% | n/a (different kernel) |

Script: `cuda_kernel/tests/d1_hfma_stress.py`.  Revert after measurement
(single-use diagnostic).

### Finding

HFMA2 is on the critical path **only for medium-grid shapes** (waves < 20
per SM).  Large-grid shapes (Qwen3-14B / 32B / 70B gu T=512) already have
ILP hiding HFMA2 across the multiple in-flight CTAs per SM, so
warp-specialisation would **not help** them.

The original D.1 plan assumed warp-spec would lift every shape.  That
assumption is wrong — it only lifts the medium-grid subset.

---

## 6. Per-shape cuda_eff ceiling analysis (T=512)

Script: `cuda_kernel/logs/r66_path_c/_cuda_eff_ceiling.py`.

Model assumptions:
- Achievable INT4 TC fraction (warp-specialised): **50%** of 660.6 TOPS
  (DeepGEMM / TensorRT-reported ceiling on Ada SM89).
- Achievable HBM fraction: 80% of 1008 GB/s.
- HFMA-critical = `waves_per_SM < 20`; else ILP hides it and ceiling
  collapses to 34% TC (empirical max observed: r66 8B gu T=512).

### Results (T=512, 35 shapes)

| bucket | n | r66 median sp | **ceiling median sp** | ceiling cuda_eff |
|---|---:|---:|---:|---:|
| HFMA-critical (waves<20) | 31 | 0.884× | **1.985×** | 50% |
| HFMA-hidden (waves≥20) | 4 | 0.749× | 1.447× | 34% |
| **all T=512** | 35 | 0.866× | **1.947×** | ~47% |

Biggest opportunity shapes (r66 → ceiling):
- Qwen3-14B / 32B down T=512: 0.80× → **2.14×** (+1.34)
- Qwen3-8B down T=512: 1.07× → 2.07× (+1.00)
- Qwen3-0.6B/1.7B q/o/kv T=512: 0.59× → 1.55× (+0.96)

Shapes that **cannot** be lifted past ~1.45× by any warp-level trick
(would need CUTLASS 3.x mainloop rewrite, 2+ weeks):
- **LLaMA-70B gu T=512**: 0.70× → ceiling 1.42×
- **Qwen2.5-32B gu T=512**: 0.72× → ceiling 1.44×
- **Qwen3-14B gu T=512**: 0.78× → ceiling 1.45×

### Same methodology applied to T=128

- r66 median: ~0.92×, TC util 14%, HBM util 21%
- Ceiling: cuda_eff **35-40%**, speedup median **1.35-1.50×**
- Most T=128 shapes have waves<20 → all HFMA-critical → all benefit
  from dual-issue / warp-spec

---

## 7. Final Path D decision

### D.1 (full warp-specialisation) — REJECTED

- 7-8 day engineering cost for a 1481-line kernel rewrite
- Benefits only ~31 of 35 T=512 shapes (HFMA-critical bucket)
- Cannot rescue 4 large-grid losers (LLaMA-70B gu / 32B gu / 14B gu / 32B down)
- Risk: CTA-shape change (128→256 threads) can destabilise occupancy
  and regress small-T kernel paths
- Expected global median uplift: +0.03 (1.049× → ~1.08×), ROI poor

### D.3 (dual-issue inline PTX) — SELECTED

See `PHASE4_D3_DUAL_ISSUE_DESIGN.md` for the full design.

One-line summary: keep r66 CTA shape and warp layout, only **interleave
`fmaf` (FP pipe) with `mma.sync` (TC pipe) at the PTX level** inside
`run_mma_pass` so the Ada dual-issue dispatcher can issue them in the
same cycle.

- 3-4 day engineering cost
- Benefits the same 31 HFMA-critical shapes (the dual-issue is what
  warp-spec was trying to achieve, but without the CTA restructure)
- Expected global median uplift: +0.02-0.05 (1.049× → ~1.07-1.10×)
- Rollback: trivial — the PTX changes are local to two nested loops

### Large-grid losers — ACCEPTED AS LIMIT

The 4 large-grid shapes (LLaMA-70B gu T=512 etc.) are bound by INT4 MMA
issue-rate on Ada, not by any per-warp inefficiency we can fix.
Escaping this ceiling requires CUTLASS 3.x warp-specialised mainloop
(≥2 weeks).  Documented as **known limitation**, out of scope for
Phase 4.
