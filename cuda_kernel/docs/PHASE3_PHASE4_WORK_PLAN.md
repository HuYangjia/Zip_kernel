# Phase 3 / Phase 4 Multi-Path Work Plan

> **Status**: living document, updated after every iteration.
> **Baseline**: r63_combined (140 shapes, median 1.02× speedup vs FP16, cold-cache).
> **Strategy**: route different T ranges to different kernels; optimise each within its competence zone rather than a single kernel covering all.

---

## 0. Architectural premise

Evidence: `docs/why_bigT_eff_drops.md` — a single W4A4 MMA kernel
cannot simultaneously optimise mem-bound (T ≤ 128, AI < 656 flops/B)
and compute-bound (T ≥ 256, AI > 656 flops/B) regions because the
required trade-offs (group-cache size, cp.async stages, warp
specialisation, mma issue cadence) are mutually exclusive.

Therefore we split by T:

| Path | T range | Current kernel | Current median eff | Target eff |
|---|---|---|---:|---:|
| **A** | T = 1 | `fused_quant_gemv` / `fused_gemv_decode` | 42% | 50% (minor) |
| ~~**B**~~ | T ∈ [2, 16] | ~~`fused_gemv_smallT`~~ **REJECTED** | — | — |
| **C** | T ∈ [2, 256] | `fused_dense_sparse_mma_int4` | 26-30% | **45-55%** |
| **D** | T ≥ 257 | same kernel (compute-bound) | 30-32% | **50-60%** |

### Why Path B was rejected (2026-04-30)
Benchmarked `fused_gemv_smallT` vs main MMA at T ∈ {2,4,8,16} across 7
production shapes (`tests/bench_smallT_revisit.py`).  smallT is
slower everywhere except the 3 smallest-shape cells where it
ties (within 3%) because both paths are dominated by the
`activation_quant` floor.  Conclusion: R16 rejection still stands;
smallT remains archived on-disk per VALIDATION_LOG norms but not
on default dispatch.

---

## 1. TODO list (multi-iteration, checked off as done)

### Path C — main MMA kernel refinement (Step 1, low-risk high-ROI)

- [x] **C.1** Re-tune `use_group_cache` gate at T=128 ✅ DONE (2026-04-30)
  - Decision: widened gate to `T=128 && n_groups ∈ (32, 64] && n_cta_m ≤ 64`.
  - Wins: Qwen3-1.7B dn T=128 −10.2%, Qwen3-14B q/o T=128 −4.0%/−4.1%.
  - Key lesson: initial attempt (`n_groups ≤ 64`) regressed 5 shapes at
    n_groups=32 by 6-13% because they preferred cache OFF at T=128.
    See failure log F-C1a and `logs/r64_path_c/c1_group_cache_sweep.json`.
  - Commit: `da6fb02`.
- [x] **C.1+C.2 validation** — 140-shape full bench ✅ DONE (2026-04-30)
  - Artefact: `logs/r64_path_c/bench.json` + `roofline_report.md` + `_compare_r63_vs_r64.py`.
  - Headline wins (r63 → r64):
    - Peak speedup: 3.25× → **3.57×** 🏆 (Qwen3-8B gate_up T=32)
    - Big wins (≥ 2×): 19 → 20
    - Wins (≥ 1×): 72 → 73
    - Qwen3-8B median: 1.34× → 1.35×
    - Qwen3-14B gate_up T=32: 1.45× → **2.01×** (+0.56 abs, 299.79us → 216.16us, −27.9%)
    - Qwen3-4B  gate_up T=32: 2.52× → **2.93×** (+0.42 abs)
    - Qwen3-1.7B down T=128:  0.73× → **0.80×** (+0.08 abs)
  - No architectural regression.  Δspeedup bucket: 5 big improve / 11 mid improve / 102 neutral / 17 mid regress / 5 big regress, where the "regress" bucket is dominated by T=1 decode shapes (cuda +8–11%, fp16 −0.3% — GPU tenant/clock drift, NOT dispatcher-related since T=1 uses a different kernel `fused_quant_gemv`).
  - Median Δ (r64 − r63) = −0.005×: within today's GPU drift envelope (T=32 median cuda_us +3.58% vs r63 on the same machine).  Not a real regression.
- [x] **C.2** Introduce kBn=16 + refine dispatcher for mid-T ✅ DONE (2026-04-30)
  - Template: kBn=16 now instantiates on both kbm_pick paths (128/64).
  - Dispatcher additions (all data-driven from `logs/r64_path_c/c2_kbn_sweep.json`):
    - **C.2**: `T ∈ [16,64] && d_out ≥ 4096 && waves_at(32) ≥ 16 → kBn=32` (avoid over-fragmentation).
    - **C.2b**: `T ∈ (8,64] && d_out ≥ 4096 && n_groups ≤ 63 && waves_at(32) ≥ 64 → kBn=32` (avoid under-filled kBn=64).
    - **C.2c-1**: Stage E large-ng kBn=64 force now also requires `T ≥ 64` (avoid under-filled kBn=64 at T=32 with deep n_g).
  - Wins (cumulative vs pre-C.1 baseline):
    - Qwen3-14B gate_up T=32: 267.37us → 189.07us (**-29.3%** 🏆)
    - Qwen3-4B  gate_up T=32:  35.55us →  29.73us (-16.4%)
    - Qwen3-8B  gate_up T=32:  56.81us →  53.03us (-6.7%)
  - Residual oversights (accepted as known):
    - Qwen3-14B dn T=32 (17408→5120, n_g=136): still +12% k32 vs auto (large-ng guard still too coarse).
    - Qwen3-8B  kv T=32 (4096→2048,  n_g=32):  still +10% k16 vs auto (d_out<4096 fallback).
    - Qwen3-4B  dn T=32 (9216→2560,  n_g=72):  still +6% k64 vs auto (deep-n_g mid-d_out edge).
  - See failure log F-C2c for the over-broad kBn=16 rule that was reverted.
  - Commits: `da6fb02` (kBn=16 instantiate), `a740d00` (final dispatcher).
- [x] **C.3** Qwen3-8B / Qwen3-14B gate_up T=128 attack ✅ iteration 1 DONE (2026-04-30, pending r65 full bench)
  - Diagnosis: 4-axis sweep (kBm / kBn+cache / split_k / joint) on
    4 shapes found that at T=128 + **d_out ≥ ~32768**, kBm=64 beats
    kBm=128 by 17%. 8B gu T=128 (d_out=24576) actually PREFERS kBm=128
    (kBm=64 regresses by +28%).  The cliff lies between d_out=24576
    and d_out=34816.
  - Patch: expanded `kbm64_gate_default` with
    `(T == 128 && d_out >= 32768 && d_in <= 8192)`.
  - Targeted microsweep verification (post-patch):
    - Qwen3-14B gu T=128 (5120→34816): auto 420.46us → **349.70us** (**−17.0%**) ✅
    - Qwen3-8B  gu T=128 (4096→24576): auto 128.55us (unchanged, does not touch the new branch)
    - Qwen3-4B  gu T=128 (2560→18432): auto  72.65us (unchanged)
    - Qwen3-8B  gu T=512 (4096→24576): auto 427us (T=512 not in branch)
  - Parity 10/10 still passing.  Commit: `0b3fdda`.
  - r65 full 140-shape bench kicked off to validate no global regression.
- [ ] **C.4** Cover T ∈ {48, 64, 96} — currently unbenched holes
- [x] **C.1+C.2 validation** — 140-shape full bench ✅ DONE (2026-04-30)
Prerequisite: Path C complete.

- [ ] **D.0** Decision: in-place template flag vs new TU
  - Recommendation: new TU `fused_dense_sparse_mma_int4_ws.cu` to
    protect the hand-tuned legacy kernel.
- [ ] **D.1** Warp role design
  - Producer warps (W/X load + dequant prep): 1 warp
  - Consumer warps (MMA issue + accumulate): 3 warps
  - Coordination: smem named barrier / cp.async mbarrier
- [ ] **D.2** Register budget analysis (target <200 regs/thread after change)
- [ ] **D.3** Smem budget (3-4 stage cp.async requires opt-in to 100KB carveout)
- [ ] **D.4** Prototype on 1 shape (Qwen3-8B gate_up T=512)
  - Parity must pass before ANY perf comparison
  - Target eff: 45%+ (vs current 32%)
- [ ] **D.5** Scale to all T≥128 shapes; if any regress > 5%, fallback
  to legacy path via dispatcher gate
- [ ] **D.6** 140-shape bench + report as `r65_path_d_ws`

### Documentation (always)

- [ ] **DOC.1** Update VALIDATION_LOG.md for each iteration (C.1 → D.6)
- [ ] **DOC.2** Update Phase 3 final report after Path C complete
- [ ] **DOC.3** Write Phase 4 scope + results after Path D complete (or fail)
- [ ] **DOC.4** Preserve all failed experiments on-disk (per project norm)
  with explicit `// DISABLED: <reason>` markers in code

---

## 2. Progress log (append-only, most recent at top)

### 2026-04-30 — C.3 kBm=64 for huge d_out at T=128 + full r65 validation
- Problem: Qwen3-14B gate_up T=128 (5120→34816) ran at 0.96× speedup
  despite sitting in a region where dispatcher heuristics should give
  full wave; microsweep showed kBm=64 beat kBm=128 by 17%.
- Patch: extend kBm=64 gate with
  `(T == 128 && d_out >= 32768 && d_in <= 8192)`.  Surgically narrow:
  excludes Qwen3-8B gu (d_out=24576 prefers kBm=128) and LLaMA-70B
  (d_in=28672).
- Targeted microsweep verification: 14B gu T=128 420us → 349us (-17%).
- r65 full 140-shape bench confirms cumulative wins:
  - median speedup: 1.021× → 1.042×  (+2.1% total over r63)
  - wins > 1×:        72 → 76  (+4)
  - Qwen3-14B gu T=128: 0.96× → **1.19×** (+0.23 abs)
  - Qwen3-14B gu T=32:  1.45× → 2.01× (+0.56 abs) — sticky from r64
  - peak remains 3.56× (Qwen3-8B gu T=32)
- Sanity check on suspect "regressions" (0.6B q/kv T=32): trial spread
  41% (max-min = 7us around a 17us median), confirming these are pure
  GPU drift / activation_quant launch-floor jitter, NOT dispatcher
  regressions.  The targeted wins have trial spread <1%, i.e. rock
  solid.
- Commit: `0b3fdda` (C.3 patch), `9024c6e` (14B-aware clamp revert).
- Artefacts: `logs/r65_path_c/bench.json` + `roofline_report.md` +
  `_compare_r63_r64_r65.py`.

### 2026-04-30 — C.2 kBn=16 + mid-T dispatcher refinement
- kBn=16 template instantiated on both kbm=128/64 paths (no kernel body change, `kNsubPerCta = (kBn+7)/8` already generic).
- Dispatcher gained three new rules (all data-driven):
  - C.2   : `T ∈ [16,64] && d_out ≥ 4096 → kBn=32` (avoid over-fragmentation to kBn=8)
  - C.2b  : `T ∈ (8,64] && d_out ≥ 4096 && n_g ≤ 63 → kBn=32` (avoid under-filled kBn=64)
  - C.2c-1: Stage E's n_g≥64 force-kBn=64 now needs `T ≥ 64`
- Measurement methodology: trial-randomised in-process sweep (N=5 interleaved trials, median per mode) — the initial single-shot in-process sweep had 40% noise from GPU clock/L2 transients.  Subprocess isolation considered but ~40 min too slow.
- Biggest win: Qwen3-14B gate_up T=32 from 267us → 189us (-29.3%).  Cumulative four shapes with >5% gain; no regressions on the 30-shape parity/perf suite.
- Failed C.2c-2 (over-broad kBn=16 rule) reverted — see F-C2c.
- Commits: `da6fb02`..`a740d00`.

### 2026-04-30 — C.1 group-cache gate widened
- 25-shape T=128 sweep (`logs/r64_path_c/c1_group_cache_sweep.json`)
  identified n_groups ∈ (32, 64] + small grid_M as a previously
  unreachable cache-on regime.
- Gate widened to include that regime.  Parity 10/10 still passing.
- Measured gains: Qwen3-1.7B dn T=128 -10.2%, Qwen3-14B q/o -4.0%/-4.1%.
- First iteration regressed 5 shapes (n_g=32, 6-13% slower) → narrowed
  the lower bound to `n_groups > kGrpBuf (=32)`; second iteration clean.
- Commit: `da6fb02`.

### 2026-04-30 — Path B rejection + work plan creation
- Path B (smallT) re-benched at T∈{2,4,8,16}; unanimously slower than
  main MMA in production shapes (1.24×–9.05× regression); REJECTED.
- This document created as the single source of truth for Path C/D
  iterative work.
- Baseline frozen: `logs/r63_combined/roofline_report.md`
  (median 1.02×, 72/140 wins, 19/140 big wins ≥2×).

---

## 3. Failure tracking

Per project norm: failed experiments are not deleted, only disabled.
Each entry includes: what was tried, why it failed, numerical
evidence, lesson.

### F-C2c (2026-04-30) — over-broad kBn=16 fallback rule
- **Tried**: `T ∈ (8,64] && n_groups ≥ 16 && d_out ≥ 2048 → kBn=16`
  as a fix for Qwen3-8B kv T=32 (4096→2048) which benefits from kBn=16.
- **Status**: regressed 2 shapes while fixing 1.
  - Qwen3-0.6B o T=32 (2048→1024, n_g=16): kBn=64 wins +27.5% (small
    n_cta_m=8 + deep intra-SM residency makes kBn=64 actually optimal).
  - Qwen3-4B dn T=32 (9216→2560, n_g=72): kBn=64 wins +6.8% for the
    same reason, plus deep K reuse.
- **Root cause**: the `d_out ≥ 2048` lower bound was too permissive;
  small d_out shapes with heavy per-CTA residency prefer kBn=64, not
  kBn=16, because the whole shape fits in a handful of SMs.
- **Lesson**: "n_groups ≥ 16" is NOT enough signal for a kBn=16
  preference; interact with `n_cta_m` and per-CTA work volume.
- **Resolution**: reverted the rule (commit `a740d00`).  The 8B kv
  T=32 +10% oversight remains as an accepted residual (see C.2 DONE
  list).

### F-C1a (2026-04-30) — group-cache gate too wide (first attempt)
- **Tried**: widen use_group_cache to `T=128 && n_groups ≤ kMaxWindowedGroups (=64) && n_cta_m ≤ 64`.
- **Status**: regression on 5 shapes with n_groups=32.
  - Qwen3-8B q_proj   T=128: 32.29 → 34.29us (+6%)
  - Qwen3-8B o_proj   T=128: 32.29 → 34.30us (+6%)
  - Qwen3-8B kv_proj  T=128: 21.00 → 23.81us (+13%)
  - Qwen3-4B o_proj   T=128: 24.70 → 27.38us (+10.8%)
  - Qwen3-4B q_proj   T=128: less dramatic but also not an improvement
- **Root cause**: at n_groups = 32, the scale/zero-buffer smem footprint
  (16 KB) actively harms SM occupancy (drops from 3 to 2 CTAs/SM on
  kBm=128), even though the cache path IS runtime-enabled.  Those
  shapes had been benefiting from the non-cached variant's lighter
  smem.  The windowed-cache-only regime (33..64) has larger n_groups
  so the cache amortises over more reuse, tilting the balance the
  other way.
- **Lesson**: smem-gated optimisations need a LOWER bound (`n_groups > kGrpBuf`),
  not just an upper bound.  Always bench the gate's neighbour regime
  too (n_g = 32 here) before shipping the change.
- **Resolution**: narrowed to `n_groups > kGrpBuf && n_groups ≤ kMaxWindowedGroups`.

### F-P0 (2026-04-30) — activation_quant fusion into MMA prologue
- **Tried**: fuse activation_quant into fused_dense_sparse_mma_int4
  prologue (new TU `fused_quant_dense_sparse_mma_int4.cu`).
- **Status**: parity PASS (10/10 shapes bit-exact); perf FAIL
  (0.05×–0.59× speedup across all 14 measured shapes).
- **Root cause**: grid = (d_out/kBm, T/kBn) replicates quant work
  `d_out/kBm` times.  For d_out=24576 that's 192× redundant quant.
  Architectural — no amount of tuning fixes it within this grid
  layout.
- **Lesson**: any "fuse prologue into MMA kernel" approach requires
  a persistent-along-M grid or cooperative-group grid sync; both are
  orthogonal refactors not worth the complexity.
- **File preserved**: `csrc/fused_dense_sparse/fused_quant_dense_sparse_mma_int4.cu`
  kept with full implementation for reference, removed from default
  dispatch.  Planned action: DISABLE but preserve.
- **Docs**: `docs/P0_QUANT_FUSION_SPIKE.md` + this entry.

### F-B (2026-04-30) — smallT kernel for T∈[2,16]
- **Tried**: default-dispatch to `fused_gemv_smallT` for T∈[2,16].
- **Status**: perf FAIL (1.24×–9.05× regression in production shapes).
- **Root cause**: dp4a + 1-warp-per-row kernel already saturates at
  T=1 (no N reduction parallelism).  At T≥2 the outer T-loop
  serialises dp4a passes while main MMA's N=8 sub-tile amortises over
  T cols in a single pass.
- **Lesson**: always re-benchmark archived rejected kernels against
  the CURRENT default dispatcher before assuming the landscape has
  shifted — it hadn't.
- **File preserved**: `csrc/fused_dense_sparse/fused_gemv_smallT.cu`
  on disk, not on dispatch path.
- **Bench**: `cuda_kernel/tests/bench_smallT_revisit.py`.

---

## 4. Non-goals (explicit)

1. **cuBLAS FP16 path as fallback** — vetoed by user; CUDA-only
   target per project scope.
2. **Triton kernel optimisation** — archived, not benchmarked.
3. **Per-token / per-tensor dynamic scale changes** — changes the
   quantisation contract, out of Kernel scope.
4. **SM90 (Hopper) async WGMMA** — kernel is SM89 (Ada) target, not
   re-targeting.

---

## 5. Definitions / formulas reference

- `HBM_BW = 1008 GB/s`, `FP16_TC_peak = 165.2 TFLOPS`,
  `INT4_TC_peak = 660.6 TOPS`, `ACHIEVABLE_FRACTION = 0.85`.
- `AI_knee = INT4_peak / HBM_BW = 656 flops/byte`.  Shapes with
  AI ≥ 656 are compute-bound on INT4 TC.
- `cuda_eff = cuda_roof_us / cuda_us`  (denominator = INT4 roofline).
- `fp16_eff = fp16_roof_us / fp16_us`  (denominator = FP16 roofline).
- Absolute TOPS comparison: `TOPS = 2·T·d_in·d_out / (kernel_us · 1e6)`.
- Roofline formulas same as `kernel/tools/profile/qwen3_roofline_report.py`.

