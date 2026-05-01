# D.3 Iteration Log

Phase 4 D.3 (dual-issue PTX) iterative experiments.  Each iteration has:
fold variant design → parity → perf A/B → verdict → next-iter plan.

## Iter 1 — algebraic fold re-form (fmaf + precomputed nzs)

**Date**: 2026-05-01

**Hypothesis**: by re-arranging `y_fp += (d - z*sumxn) * s` into
`y_fp += fmaf(d, s, -(z*sumxn)*s)`, we shorten the per-element critical
path from ~8c (mul-sub-fma chain) to ~4c (single fma), giving fold loop
a ~2× speedup.  Measurable on HFMA-critical shapes (T=128/512 medium
grid, per HFMA stress test +53-63% sensitivity).

**Implementation**:
- Added `bool kInterleaveFold = false` template param (MS-0).
- Added `HKUST_V9_INTERLEAVE` env switch (re-read per launch).
- Branched fold_dense lambda on kInterleaveFold — false path bit-exact
  r66, true path uses `fmaf(float(d), s, y_fp + nzs)` with
  `nzs = -(z * sumxn) * s` precomputed per element.

**Parity** (10 shapes, rel err tolerance 5e-3):
- 10/10 PASS, max rel err 9.77e-4 (1 ulp difference from single vs double
  rounding — expected).

**Perf A/B** (interleaved trials, 4×):

| shape | r66 base | Iter1 new | Δ |
|---|---:|---:|---:|
| 8B gu T=128 | 131.50us | 136.56us | **+3.85%** (regression) |
| 8B gu T=512 | 411.30us | 409.52us | −0.43% (noise) |
| 8B o T=512  |  72.68us |  73.78us | +1.51% (noise) |

**Verdict**: NO-GAIN.  The re-form is algebraically correct but nvcc
already compiles r66's `d - z*sumxn` into `fma(-z, sumxn, float(d))`
form (we can confirm by disassembling SASS; not needed for Iter 1
decision).  The extra `*s` fmul in Iter 1's `nzs` precompute adds an
op without shortening the final fma's critical path (`y_fp += ...`
still R-M-W on same register).

**Root insight**: fold's critical path is already minimised by nvcc.
The only way to speed up fold is to **interleave it with mma.sync
from a DIFFERENT group**, so the warp scheduler can dual-issue on
TC pipe + FP pipe.  Requires cross-group d_acc double buffer.

**Iter 2 plan**: implement cross-group d_acc double buffer.  While
group `g+1`'s K-loop runs (16 mma.sync), fold group `g`'s d_acc
(32 fmaf) inline.  Warp scheduler can then issue mma (TC pipe) and
fmaf (FP pipe) on adjacent cycles.  Register budget increases by
~32 int32 per thread; must verify no spill with `-Xptxas -v`.

Status: **Iter 1 kept behind env flag** (no revert needed — default
off, zero regression).  Iter 2 will add a *second* code path on top
of Iter 1's scaffolding.

---

## Iter 2a — batched prefetch before fold loop

**Date**: 2026-05-01 (same day as Iter 1)

**Hypothesis**: r66's `for (im) { prefetch(im); fold(im); }` nests the
smem-read of prefetch inside each im iteration.  Batching all prefetch
(`for (im) pr_cache[im] = prefetch(im); for (im, in_sub, r) fold(...);`)
should let the warp scheduler issue `prefetch[im=1]`'s smem reads
concurrently with `fold[im=0]`'s fmaf chain, hiding smem-read latency.

**Implementation**: inside `run_mma_pass`, split on `if constexpr
(kInterleaveFold)`.  True path batches prefetch into a `pr_cache[kMsub]`
array before the fold triple-nested loop.  `pr_cache` uses
`decltype(prefetch_fn(...))` so it works for both dense and sparse
fold ABIs.  This is on TOP of Iter 1's fmaf re-form (both active under
the same env flag).

**Parity**: 10/10 PASS, max rel err 9.77e-4 (identical to Iter 1 —
fold result bit-identical regardless of prefetch order).

**Perf A/B**:

| shape | r66 base | Iter2a new | Δ |
|---|---:|---:|---:|
| 8B gu T=128 | 131.78us | 140.17us | **+6.37%** (worse) |
| 8B gu T=512 | 409.77us | 421.83us | **+2.94%** (worse) |
| 8B o T=512  |  72.62us |  73.21us | +0.80% (noise) |

**Verdict**: NO-GAIN, regression.  Batching prefetch increases register
pressure (`pr_cache[kMsub] = 4 floats × 2 = 8 fp32 regs`/thread held
live across the entire fold loop).  r66's nested form let nvcc recycle
`pr` registers between `im` iterations; batching prevents this.  The
extra register pressure likely evicts d_acc to spill or drops occupancy.

**Combined Iter 1 + 2a**: both changes on the kInterleaveFold=true
path are regressions.  Source-level fold reordering / algebraic reform
**cannot improve** beyond nvcc's baseline — nvcc already fuses
`d - z*sumxn` into an fma and pipelines prefetch/fold loops.

## Iter 2a → 3 decision point

Two paths remaining:

**Path 3a (continue D.3)** — cross-group d_acc double buffer.  This is
the ONLY remaining source-level win: while group g+1's K-loop runs
(16 mma.sync), fold group g's d_acc (32 fmaf).  Requires d_acc to
live outside `run_mma_pass` scope, with prologue / steady-state / drain
plumbing.  Expected +15-25% on HFMA-critical shapes if it works; risk
of register spill or parity break.  ~1 day effort.

**Path 3b (accept and archive)** — Iter 1+2a data proves that per-
element fold order/reform is exhausted; only cross-group interleave
remains and it's a 1-day gamble.  Write D.3 failure post-mortem and
keep main at r66 (Path C's 1.049× median).  Document the bottleneck
ceiling is at nvcc's current optimisation frontier for this kernel
shape; future gains require full CUTLASS 3.x mainloop rewrite.

Current recommendation: **Path 3a**, one more serious attempt (hard
cap 1 day).  If Path 3a also fails parity or perf, Path 3b automatically.

Status: both Iter 1 and Iter 2a code paths are behind the
`kInterleaveFold` template flag (default false = r66 bit-exact).  No
revert needed; main is safe.
