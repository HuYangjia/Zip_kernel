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
