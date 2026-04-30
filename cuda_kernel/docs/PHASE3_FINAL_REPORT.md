# Phase 3 Final Report — W4A4 INT4 kernel for SM89 (RTX 4090)

**Status**: Phase 3 delivered.  Qwen3-8B cold-cache median speedup
**1.335× vs BF16 cuBLAS**, **15 / 20 shapes winning**.

## 1. Scope

Phase 3 targeted the HBM/compute efficiency of the fused W4A4 INT4
kernel (`fused_dense_sparse_mma_int4`) on SM89, along Qwen3-8B's 20
canonical projection shapes (5 projections × 4 batch / T values).

Optimisation work covered **r60 Stage I** → **r62 F2** (approximately
two weeks of iteration on a single RTX 4090 autodl GPU).  The source
of truth for every single experiment, including failures, is
[VALIDATION_LOG.md](../VALIDATION_LOG.md).

## 2. Final Qwen3-8B end-to-end numbers

Benchmark: `kernel/cuda_kernel/benchmarks/bench_qwen3_shapes.py`,
`--ts 1 32 128 512`, `hp_ratio=0.05`, cold-cache FP16 baseline
(L2-flushed before each BF16 timing sample).

| | Start of Phase 3 | Phase 3 final (r62 F2) |
|---|---:|---:|
| median speedup vs FP16 | 0.82× | **1.335×** |
| mean speedup vs FP16 | 0.96× | **1.489×** |
| shapes winning (≥1.00×) | 5 / 20 | **15 / 20** |
| shapes with clear win (≥1.10×) | 3 / 20 | **14 / 20** |
| shapes with clear loss (<0.90×) | 13 / 20 | **3 / 20** |
| peak single-shape speedup | 1.78× | **3.25×** (gate_up_proj T=32) |

Full per-shape table at
[../logs/r62_f2_v2/qwen3_20260430_113807/bench.md](../logs/r62_f2_v2/qwen3_20260430_113807/bench.md).

## 2.1 Full 80-shape bench across Qwen3 family + roofline analysis

Delivered as r62 F2 Final: 4 models × 4 batch sizes = **80 shapes** with
cold-cache FP16 baseline *and* RTX 4090 roofline cross-reference.

Artefacts:
- [../logs/r62_f2_final/SUMMARY.md](../logs/r62_f2_final/SUMMARY.md)
- [../logs/r62_f2_final/qwen3_20260430_122555/bench.md](../logs/r62_f2_final/qwen3_20260430_122555/bench.md)
- [../logs/r62_f2_final/qwen3_20260430_122555/roofline_report.md](../logs/r62_f2_final/qwen3_20260430_122555/roofline_report.md)

| metric | value |
|---|---:|
| total shapes | 80 |
| median speedup vs FP16 | **0.90×** |
| mean speedup vs FP16 | **1.05×** |
| wins (≥ 1.00×) | **35 / 80** (44 %) |
| big wins (≥ 2.00×) | **8 / 80** |
| peak | **3.25×** (Qwen3-8B gate_up_proj T=32) |

Per-model:

| model | median | wins | peak |
|---|---:|---:|---:|
| Qwen3-0.6B | 0.36× | 2 / 20 | 1.74× |
| Qwen3-1.7B | 0.76× | 8 / 20 | 2.35× |
| Qwen3-4B | 1.01× | 10 / 20 | 2.52× |
| **Qwen3-8B** | **1.34×** | **15 / 20** | **3.25×** |

Per-T across all models (decode = T=1 is the most important case for
real inference):

| T | median | wins | comment |
|---:|---:|---:|---|
| **1** | **1.55×** | **17 / 20** | decode, all models beat FP16 comfortably |
| 32 | 0.68× | 7 / 20 | small-T prefill hit by overhead floor |
| 128 | 0.68× | 4 / 20 | same |
| 512 | 0.85× | 7 / 20 | GEMM re-dominates, recovery starts |

### Key insight from roofline report §5 — the ~33 us overhead floor

The worst 15 shapes by `cuda_efficiency` all share a single pattern:
`cuda_us ≈ 30-34 us` regardless of problem size, while `cuda_roof` is
only 1.5-4.2 us.  Decomposition (see `_analyze_floor.py`) shows that
**`activation_quant` contributes ~16 us of that floor** on every shape
— essentially `2 × 7-8 us kernel launch overhead`.  Of the 45 measured
losses, **32 sit inside this 28-36 us band** where the quant launch
cost dominates over the kernel time itself.

### Physics ceiling

Critical roofline result (§6): **80 / 80** shapes have
`cuda_roof < fp16_roof`.  Every shape has room for a W4A4 win at the
ceiling — **0 shapes are physics-bound losses**.  All 45 measured
losses are therefore pure *implementation* gap.

### INT4 efficiency (cuda_eff)

```
by T   : T=1 → 39%, T=32 → 19%, T=128 → 22%, T=512 → 31% (median)
by proj: gate_up 45%, down 30%, q/o 25-27%, kv 18%  (median)
peak   : 88% (Qwen3-8B gate_up T=32) — kernel can reach hardware limit
```

## 3. Contributions by stage

| stage | what shipped | median speedup | wins |
|---|---|---:|---:|
| r60 Stage I | Split-K on n_groups axis | 0.82× | 5 / 20 |
| r61 F | Occupancy-aware scale/zero cache gate | ~0.91× (tight-loop) | 7 / 20 |
| r61 G | Widen R44 dispatch gate + fix kBn demote | ~0.91× (tight-loop) | 7 / 20 |
| **r62 P2** | **Bench methodology: cold-cache FP16 baseline** | **1.21×** | **11 / 20** |
| r62 F4 | CUTLASS 2.11 migration (A1.5 / A3 / H) — **REJECTED** | n/a | 0 / 12 |
| **r62 F2** | **Data-driven dispatcher rewrite** | **1.335×** | **15 / 20** |

r62 P2 is the single largest *methodology* fix (exposed that 4 claimed
kernel regressions were really L2-warm cuBLAS cheating).  r62 F2 is
the single largest *optimisation* fix (dispatcher was under-using the
existing split-K path on 5 / 20 shapes).

## 4. Architecture summary

The production kernel is `fused_dense_sparse_mma_int4_kernel` at
`csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`.

Key dimensions:
- **MMA atom**: `mma.m16n8k64.s4.s4.s32` (I-L2, frozen).
- **Tile variants**: (kBm, kBn) ∈ {(128, 64), (128, 32), (128, 8),
  (64, 64), (64, 32), (64, 8)}, all template-instantiated at compile
  time.  split_k ∈ {1, 2, 4}, `gridDim.z = split_k`.
- **Workspace**: int32 fp32 partial buffer only allocated when
  `split_k > 1`.
- **Epilogue**: in-place `y = (acc - zero·sumX) · scale_u4 · scale_x`,
  vectorised `__half2` stores to `Y_total`.
- **Shared memory**: 2-stage `cp.async` double buffer of W + X with
  optional group-cache of scale/zero/sumX (on for ng ≤ 8 or
  ng ≤ 32 ∧ T ≤ 32, off otherwise per occupancy analysis).

Dispatcher (r62 F2):
```
hp_empty = (hp_nnz == 0)
kbm = 64  if R52 gate matches (T-band × d_out-band) else 128
sk  = ceil(128 / (n_cta_m_at_128 * ceil(T/64)))   // fill 1 RTX 4090 wave
       clamped to {1,2,4} with n_groups % sk == 0 and T >= 8
kbn = pick_from_waves_at(kBn) * sk                 // wave-aware, sk-aware
demote(kbn→8) if (kbm=64, T∈[32,96], d_out≤2048, kbn≥32)  // R44
```
Sparse branch inside the kernel is gated on `split_k_idx == 0` so the
BSR contribution is added exactly once across K-splits; the legacy
`hp_nnz == 0` restriction on split-K has been lifted.

## 5. Remaining losses and known limits

Three `kv_proj` shapes at `d_out = 2048` remain < 1× vs FP16 cuBLAS:

| shape | speedup | observation |
|---|---:|---|
| kv_proj T=32 | 0.60× | dispatch_sweep confirms auto = best; kernel at intrinsic ceiling |
| kv_proj T=128 | 0.65× | recovered from 0.54× via F2 split-K |
| kv_proj T=512 | 0.78× | tile already optimal per sweep |

These are **not dispatcher bugs** — the `dispatch_sweep_after2.md`
report shows `auto` within 2 % of the best forced config for all three.
Closing them would require one of:

1. **Custom CUTLASS EpilogueVisitor** (`LinearCombinationDequantizeW4A4`):
   3-5 days, success probability ~45 %, targeted ceiling ~1.5× median.
2. **Shared-memory layout rewrite with ldmatrix + XOR swizzle**:
   r61 Stage C attempted this — all three sub-experiments (C.1a
   ldmatrix-only, C.1b ldmatrix+swizzle, split-K extension) regressed
   on all tested shapes due to the 16-byte-alignment constraint of
   `ldmatrix` on our 64-byte stride.  Blocked without a deeper
   rewrite.
3. **Shape-specific kernel (kBm = 64, split-N)**: feasible in 1-2 days
   but targets only 3 of 20 shapes, expected ceiling ~0.85×.

## 6. Negative-result archive (engineering hygiene)

Per the project's experiment-management conventions, failed optimisations
are kept in tree / in log, not deleted:

- **r62 F4 Stage A1.5** — CUTLASS GemmBatched + 3D workspace.  0/12 wins,
  worst 0.11× (1.5 GB workspace dominates HBM).
- **r62 F4 Stage A3** — CUTLASS per-group Gemm + 2D workspace.  0/12 wins,
  worst 0.07× (32 × 7us launch overhead dominates).
- **r62 F4 Stage H** — 4-byte `cp.async.ca` for bank-conflict relief.
  Parity regressed (rel_err 0.47-0.58 at ng ≥ 16); cp.async.ca
  violates per-group tile atomicity.
- **r61 Stage C** — ldmatrix + XOR swizzle.  Parity-safe but 6-66 %
  performance regression; 16-byte alignment blocks full deswizzle.

Each of the above is logged under `../VALIDATION_LOG.md` with
enough detail to either resume or reject more decisively in Phase 4.

## 7. Reproduction

```bash
# Parity (from repo root):
python -m pytest kernel/cuda_kernel/tests/test_parity.py -q

# Dispatch sweep (20 shapes × 19 configs, ~5 min):
PYTHONPATH=. python -m kernel.tools.profile.dispatch_sweep \
    --output kernel/cuda_kernel/logs/r62_f2/dispatch_sweep.md \
    --json   kernel/cuda_kernel/logs/r62_f2/dispatch_sweep.json

# Qwen3-8B e2e bench (cold-cache, all 20 shapes):
PYTHONPATH=. python -m kernel.cuda_kernel.benchmarks.bench_qwen3_shapes \
    --models Qwen3-8B --ts 1 32 128 512 \
    --out-root kernel/cuda_kernel/logs/r62_f2_v2
```

## 8. Next phase preview (Phase 4 candidate)

Based on the full 80-shape roofline analysis (§2.1), the priority
ordering has shifted significantly since Phase 3 kick-off:

1. **🥇 `activation_quant` fusion into the main kernel prologue**
   - **Evidence**: ~16 us floor on 32/80 shapes, killing T=32/128 on
     small models.  Of the 45 measured losses, 32 are trapped under
     this floor.
   - **Expected gain**: removing ~16 us from e2e puts ~20-25 losing
     shapes above 1.0×, lifts overall median from 0.90× toward ~1.10×.
   - **Risk**: medium (requires carrying the per-token max-abs
     reduction through the fused kernel's prologue and packing int4 in
     shared memory).  Estimated 2-3 days.

2. **🥈 `kv_proj` targeted tile (Phase 3 r62 F5 candidate)**
   - 3 remaining Qwen3-8B losses all `kv_proj` (0.60× / 0.65× / 0.78×).
   - Dispatch_sweep already confirms auto = best; kernel-internal
     bottleneck, not dispatch.
   - 1-2 days, ~70 % success probability, bounded to 3 shapes.

3. **🥉 Custom CUTLASS EpilogueVisitor (F4v2)** — REJECTED.
   Memory `ie8lp95b` documents the finding: CUTLASS 2.11 visitor
   infrastructure is present but W4A4 per-group dequant cannot be
   expressed in an epilogue (it must happen inside the MMA K-loop at
   every 128-col tile).  Only a CUTLASS mainloop patch would work, at
   ≥ 1 week cost and ~35 % success probability — not worth the ROI
   against option 1.

4. **SM 90 / H100 port** — orthogonal to Ada-specific tuning; cuTE's
   `WarpSpecializedCooperative` schedule may natively support per-tile-k
   epilogue hooks that could revisit option 3 at lower risk.  Deferred
   as Phase 5+ material.
