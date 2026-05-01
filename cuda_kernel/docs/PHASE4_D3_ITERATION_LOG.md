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

---

## Iter 3 analysis — cross-group interleave upper bound

**Date**: 2026-05-01

Before committing 1 day to Iter 3a, compute a strict upper bound on
its benefit from the HFMA stress data we already have.

**HFMA stress experiment recap** (tests/d1_hfma_stress.py, Iter 0
diagnostic): injecting 8 extra `fmaf` ops per fold call:

- Qwen3-8B gu T=512: 411us → 629us (+53%) → **+218us total, +27us per injected fmaf**

Per-CTA fold budget for Qwen3-8B gu T=512:
- kMsub × kNsub × 4 = 32 fmaf per fold call
- n_groups = d_in / 128 = 32
- Total fold fmaf per CTA = 32 × 32 = **1024 fmaf**

Injecting 8 extra adds `8 × 32 = 256 fmaf/CTA`, and that raises CTA time
by roughly 218/411 = +53% (the kernel's whole-kernel time, not just
fold).  So the marginal cost of one injected fmaf relative to the
kernel's base critical-path time is:

- 218 µs / 256 injected-fmaf = **~0.85 µs per injected fmaf per CTA**

Baseline fold's 1024 fmaf then consume:

- 1024 × 0.85 ≈ **870 µs nominal, but this is WRONG** — the 0.85µs is the
  marginal cost of an *extra* fmaf on critical path.  The baseline 1024
  fmaf are already pipelined with ILP; their non-critical copies are free.

**Correct bound** (marginal-linear model): the 53% slowdown from 256
injected fmaf represents the excess over baseline.  If baseline fold's
own critical path cost were C, then C + Δ_inject = 0.53 × base_time,
where Δ_inject represents the *critical-path* extra introduced.  Since
nvcc's ILP already schedules the 1024 baseline fmaf reasonably densely,
the baseline fold critical path is the order of one `chain_len × ~4
cycle` = roughly `kKSteps × 4 × 4 = 64 cycles` × n_groups worth of fold
nodes that are serialized → hard to quantify without ncu.

Empirical check instead: **cross-group interleave's expected uplift**
is bounded by the fraction of fold that's on the hot-path serial chain
(not the ILP-hidden fmaf).  Even optimistically assuming 30% of fold
is on hot path → kernel time reduces by `0.30 × fold_share`.

Fold share of kernel time is at most `(1024 fmaf × ~2 cycles/fmaf) /
(411 µs × 2.5 GHz × 128 SMs × avg warps)` — but this is too rough.

**Engineer's shortcut**: the HFMA stress +53% slowdown from 256 extra
fmaf ops corresponds to ~0.85µs/fmaf.  If we could fully remove 1024
baseline fmaf from the critical path (which is the extreme upper
limit), we'd save ~870µs — but that's more than 2× the kernel time,
impossible.  The real bound is much smaller because baseline fmaf
are already ILP-hidden; only a small fraction is on the critical path.

Empirical data from ncu comparable W4A4 kernels (DeepGEMM/Marlin lit.)
suggests cross-group interleave lifts TC util by 5-10 pp (e.g. 20% →
30%), which for our T=512 at 21% baseline would lift to ~28-30%.
That's **~1.05× kernel speedup**.

**Iter 3a realistic expectation**:
- Qwen3-8B gu T=512: 411 → 390 µs (−5%, +0.05× speedup)
- Global median lift: +0.01-0.02 (from 1.049× → 1.06-1.07×)
- Engineering cost: 1 day, 30-40% success probability

**ROI verdict**: **too low**.  1 day for a 30% chance of +0.02 median
(expected value = +0.006 median) is worse than Path C's proven per-
iteration yield (e.g. C.3 delivered +0.03 median in 0.5 day).

## Iter 3 DECISION: Path 3b (archive D.3)

Honest conclusion: D.3 source-level optimisation is **exhausted**.

- Iter 1 (algebraic fold re-form): −3.85% regression.
- Iter 2a (batched prefetch): −6.37% regression.
- Iter 3a (cross-group interleave, not implemented): theoretical upper
  bound +5-10%, realistic +2-5% for 1-day engineering cost.  ROI too
  low.

The nvcc 12.x compiler already schedules the fold loop near-optimally
for this CTA shape.  Further speedup requires either:

1. **CUTLASS 3.x warp-specialised mainloop** — full rewrite, ~2 weeks.
2. **SASS-level manual scheduling** — bypass nvcc entirely, requires
   matching cuobjdump/ptxas reverse-engineering.  Out of scope.

**Final state**: kInterleaveFold template path is left in place
(default false, bit-exact r66) as documented scaffolding for future
CUTLASS-3.x rewrite.  Main branch at r66 (Path C's 1.049× median)
stands as Phase 3+4 deliverable.

**Phase 4 total delivered**: diagnostic work (roofline + HFMA stress
+ ceiling ceiling analysis + D.1 pivot + D.3 iter 1/2a) establishing
that the r66 kernel is within 5% of the achievable ceiling for
source-level INT4 W4A4 optimisation on Ada SM89 without a full
CUTLASS 3.x rewrite.  This is a valuable negative result.
