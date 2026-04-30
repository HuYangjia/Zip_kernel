# Phase 4 D.3 — Dual-Issue PTX Design Document

**Status**: design → implementation starting 2026-04-30
**Branch**: `phase4-warp-specialised` (will create D.3 subbranch or append)
**Base**: r66 (Path C final, main = `fused_dense_sparse_mma_int4.cu`)
**Predecessor**: D.1 warp-specialisation (rejected — see `PHASE4_D1_PIVOT.md`)

## 0. One-paragraph summary

Ada SM89 has a **dual-issue dispatcher** per warp scheduler that can
issue one Tensor-Core instruction (`mma.sync`) and one FP32 instruction
(`fmaf`) in the same cycle — but only if the compiler schedules them
such that their dependencies permit it and they land on different
pipe types within the same dispatch window.  Today, nvcc emits the
HFMA-fold chain **after** all `mma.sync` of a K-slab, creating a
strict MMA → HFMA sequential dependency that forces sequential issue.
D.3 **manually interleaves** `fmaf` and `mma.sync` at the PTX level so
the hardware dispatcher can pair them.

## 1. Evidence base (must read before implementation)

| source | claim | file |
|---|---|---|
| Roofline | T=512 TC util 21% + HBM util 11% → scheduler stall | `logs/r66_path_c/_bottleneck_analysis.py` |
| HFMA stress | +8 fmaf ops → 8B shapes +53-63% slowdown → HFMA on critical path | `tests/d1_hfma_stress.py` |
| HFMA stress | +8 fmaf ops → 14B gu T=512 −4.6% (noise) → large grids hide HFMA | ditto |
| Ceiling analysis | HFMA-critical shapes can reach TC 50% (vs current 21%) | `logs/r66_path_c/_cuda_eff_ceiling.py` |

## 2. What we change

File: `csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`

The target is the **dense-branch K-loop** inside `run_mma_pass`
(roughly lines 580-640 of r66).  Current shape (pseudocode):

```cpp
for (int im = 0; im < kMsubPerWarp; ++im) {
    for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
        mma_m16n8k64_s4s4s32(a_regs[im], b_regs[in_sub], d_acc[im][in_sub]);
    }
}
// then a separate loop:
for (int im = 0; im < kMsubPerWarp; ++im) {
    auto pr = prefetch_fn(...);
    for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
        for (int r = 0; r < 4; ++r) {
            fold_fn(d_acc[im][in_sub][r], ..., pr);  // 3-4 fmaf ops each
        }
    }
}
```

After D.3 (pseudocode, dense fold interleaved into mma loop via a
compile-time flag `kInterleaveFold`):

```cpp
// Prefetch per-row scalars BEFORE the mma loop (cheap, hoistable)
#pragma unroll
for (int im = 0; im < kMsubPerWarp; ++im)
    pr_cache[im] = prefetch_fn(...);   // pre-compute z0/z1, s0/s1

// Interleaved mma + fold of the PREVIOUS iteration's d_acc
for (int im = 0; im < kMsubPerWarp; ++im) {
    for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
        mma_m16n8k64_s4s4s32(a_regs[im], b_regs[in_sub], d_acc_next[im][in_sub]);
        // Immediately after issuing the mma, start folding the PREVIOUS
        // iteration's result (no WAW/RAW hazard — different accumulator
        // register set, and the fold targets y_fp not d_acc_next).
        if (im > 0 || in_sub > 0) {
            fold_fn(d_acc_prev[...], ..., pr_cache[...]);
        }
    }
}
// Tail: fold the last pending d_acc
fold_fn(d_acc_next_last, ..., pr_cache[...]);
```

Key insight: `mma.sync` takes ~8-16 cycles latency; `fmaf` takes 4
cycles and lands on the FP32 pipe.  If we place a `fmaf` immediately
after an `mma.sync` whose result isn't consumed by that `fmaf`, the
dispatcher can co-issue them on cycle N (TC pipe) and cycle N+1 (FP
pipe, or same cycle depending on pair slot).

## 3. Why this is different from D.1 warp-spec

| | D.1 warp-spec | D.3 dual-issue |
|---|---|---|
| CTA threads | 128 → 256 | unchanged |
| Warp count | 4 → 8, split 4+4 roles | unchanged, 4 warps |
| Shared-mem layout | new `d_acc` spill buffer | unchanged |
| Sync primitives | mbarrier / named_barrier | unchanged |
| Source change | ~500 LOC rewrite | ~100 LOC, local |
| Risk | occupancy / parity break | PTX syntax / scheduler hint |
| Rollback | revert kernel file | toggle `kInterleaveFold = false` |
| Time | 7-8 days | 3-4 days |

## 4. Plan of attack

### MS-0 (0.5 day, done) — Decision + docs
- ✅ HFMA stress diagnostic
- ✅ ceiling analysis
- ✅ this design doc

### MS-1 (1 day) — Infrastructure
- Add template param `bool kInterleaveFold = false`
- Create a **double-buffered** accumulator `d_acc[2][kMsub][kNsub][4]`
  so fold of iteration `g-1` can overlap with mma of iteration `g`
- **No behavioural change yet** — `kInterleaveFold=false` path is
  bit-exact with r66
- Parity test: 10/10 on Qwen3-8B 20 shapes

### MS-2 (1 day) — Interleaved loop
- Implement `kInterleaveFold=true` path in `run_mma_pass`
- Prefetch `pr_cache` before the fused loop
- Fold previous iteration's d_acc inside the current mma loop
- Parity test: 10/10 on Qwen3-8B 20 shapes
- Critical: verify no register spills (`ptxas -v`)

### MS-3 (0.5 day) — Bench gate
- Bench on 3 representative HFMA-critical shapes:
  - Qwen3-8B gu T=512 (ref 458us, expected target ~330-380us, +20-30%)
  - Qwen3-8B gu T=128 (ref 148us, expected target ~105-125us)
  - Qwen3-8B o_proj T=512 (ref 86us, expected target ~70us)
- **Go/no-go gate**:
  - Go if median speedup across these 3 shapes ≥ +15%
  - No-go if ≤ +5% or any parity failure → revert, write negative result

### MS-4 (1 day, conditional on MS-3 pass) — Full bench & tuning
- 140-shape full bench via `benchmarks/bench_phase4_all.py`
- Tune `kInterleaveFold` default: on vs off per kBn
- Commit only if global median lifts by ≥ +0.02

### MS-5 (0.5 day) — Merge or archive
- If Go: merge to main, update `Phase 3 final report` with Phase 4 result
- If No-go: keep branch alive with negative-result note in
  `VALIDATION_LOG.md`, main stays at r66

Total: **3-4 days max**, strict gate at MS-3.

## 5. Known risks

1. **Register spill** — the double-buffered `d_acc[2]` doubles
   accumulator registers per warp.  With `kMsubPerWarp=2, kNsubPerCta=4,
   r=4` → 32 int32 regs × 2 buffers = 64 regs/warp just for d_acc.  Must
   check `ptxas -v` does NOT spill.  Mitigation: fall back to single
   buffer + tail-aware scheduling.
2. **nvcc de-interleaves** — even with hand-placed `fmaf` after
   `mma.sync`, nvcc can reorder to a canonical form.  Mitigation: use
   `asm volatile` for the mma emission + `__threadfence_block()`-free
   ordering barriers, or escalate to full inline PTX blocks.
3. **No measurable gain** — possibility that nvcc already achieves
   some dual-issue via its own scheduler.  MS-3 gate catches this in
   0.5 day, limiting total wasted time to ~2.5 days.

## 6. Expected outcome (honest)

**Median target**: 1.049× → **1.07-1.10×** global.
**Best case**: 1.049× → 1.12× if dual-issue approaches 2× MMA throughput
on the HFMA-critical bucket.
**Worst case (MS-3 gate fails)**: 2.5 days spent on a negative result,
documented in VALIDATION_LOG.md, main branch unchanged.

## 7. Out of scope

- Large-grid losers (LLaMA-70B gu T=512 and friends) — accepted as
  Phase 4 limit, requires CUTLASS 3.x mainloop rewrite
- T=1 / T=32 — already optimal path, not touched
- Anything changing kernel dispatcher or CTA shape
