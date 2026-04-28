# epilogue_fma_bound cluster: pressure-test report

Budget per timing: warmup=500, outer=20, inner=200.  Each shape ran 5 independent trials with base/scale_one interleaved per trial.  Verdict is NOISE when median |delta| < 0.5% (trivially zero), OR when median |delta| < 3% AND the [p05, p95] interval straddles 0.  Otherwise REAL_SIGNAL.

| shape | T | d_in | d_out | median delta | [p05, p95] | verdict |
|---|---|---|---|---|---|---|
| `mid_T128_kv_2560_2048` | 128 | 2560 | 2048 | +0.27% | [-0.07%, +0.46%] | **NOISE** |
| `worst_T8_q_4096_4096` | 8 | 4096 | 4096 | -0.16% | [-0.26%, +0.57%] | **NOISE** |
| `decode_T1_kv_2560_2048` | 1 | 2560 | 2048 | +0.14% | [-0.04%, +0.44%] | **NOISE** |
| `large_T1024_gu_4096_24576` | 1024 | 4096 | 24576 | +0.01% | [-0.18%, +0.02%] | **NOISE** |

## Conclusion

- 4 / 4 shapes verdict = NOISE
- 0 / 4 shapes verdict = REAL_SIGNAL

**Every tested shape -- including the three highest-|delta| reps from the Phase 2 re-run -- shows an epilogue signal indistinguishable from noise under the (warmup=500, outer=20, inner=200, 5 trials) budget.**

This corroborates the `epilogue_fma_bound` cluster collapse and closes out the last audit item before R49 can commit to the two-cluster roadmap (`tc_underutil`, `launch_sparse`).  See `../xzero_probe/xzero_probe_report.md` for the sibling audit that prompted the warmup bump in the first place.

## Methodology notes

- Each trial reruns both `base` and `scale_one` so transient clock drift cancels to first order.  Within a trial the two variants are back-to-back (no sleeps).
- Per-variant timing uses the project standard `time_forward_us(warmup, outer, inner)` which returns `min over outer of (mean over inner of per-iter us)`.  The budgets here are ~2.5x the stronger Phase 2 re-run.
- `scale_one` is constructed via `_weights_with_identity_scales(W)` (same helper as `microbench_bisection`) which replaces `scale_u4` with ones and `zero_u4` with zeros; the rest of W is shared, so the only algorithmic difference is the epilogue dequant FMA degenerating.
- Inputs are rebuilt via `build_shape_inputs(tag)`; X is a fresh random draw once per shape and reused across trials.  This is intentional: we want to isolate the epilogue lever, not convolve with X-dependent HBM scheduling.
