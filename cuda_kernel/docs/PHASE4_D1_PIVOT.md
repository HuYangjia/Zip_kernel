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
