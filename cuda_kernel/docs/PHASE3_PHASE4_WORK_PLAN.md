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

- [ ] **C.1** Re-tune `use_group_cache` gate at T=128
  - Current gate: `n_groups ≤ 8 OR (n_groups ≤ 32 AND T ≤ 32)`.
  - Hypothesis: cache helps T=128 for low-grid_M shapes too.
  - Sweep: (T=128, cache∈{on,off}) × (n_groups∈{4,8,16,32}) × (d_out∈{1024,2048,4096,8192,14336}).
  - Action: either widen gate or keep it; write decision + data to VALIDATION_LOG.
- [ ] **C.2** Introduce kBn=16 as a new tile size
  - Current set: {8, 32, 64}; gap at T=128 where kBn=32 gives only 4 N-tiles.
  - Instantiate template with kBn=16, hook into dispatcher sweep.
  - Sweep all mid shapes; adopt kBn=16 for shapes where it wins ≥5%.
- [ ] **C.3** Qwen3-8B gate_up_proj T=128 (4096→24576) specific attack
  - Today: 48% eff (Path C's single worst drop).
  - Grid_M = 192 CTAs → group-cache effectively unused.
  - Experiments: (a) windowed cache on large grid_M, (b) split-K tuning,
    (c) kBm=64 variant with cache.
  - Target: 65%+ eff on this shape (≥0.20× speedup gain).
- [ ] **C.4** Cover T ∈ {48, 64, 96} — currently unbenched holes
  - r62 F2 dispatcher has explicit gates for these but we never measured.
  - Add them to bench_qwen3_shapes default T list; capture roofline.
- [ ] **C.5** Re-run 140-shape bench + update roofline report under
  name `r64_path_c_refined`.

### Path D — warp-specialised kernel rewrite (Step 2, high-risk high-reward)

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

