# X=0 slowdown anomaly probe — Phase 2 deep-dive

_Run: `python -m kernel.tools.profile.xzero_probe`.  See script docstring for experiment definitions.  All timings use the 3-piece microbench contract (warmup>=200, inner>=200, outer>=10)._

## 1. Stage decomposition on mid_T128_kv_2560_2048

| stage | base us | x=0 us | Δ% |
|---|---:|---:|---:|
| activation_quant | 15.32 | 15.27 | +0.3 |
| fused_dense_sparse_mma | 40.68 | 37.45 | +8.0 |
| **sum** | **56.00** | **52.71** | **+5.9** |

> If `activation_quant Δ ≈ 0` and `fused_mma Δ >> 0`, the data-dependent slowdown lives inside the MMA kernel (not the quant kernel).  The reverse localises it to the gather+reduce path.

## 2. T-sweep (d_in=2560, d_out=2048)

| T | quant base | quant zero | Δ_quant% | mma base | mma zero | Δ_mma% | sum Δ% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.18 | 15.00 | +1.2 | 21.86 | 21.75 | +0.5 | +0.8 |
| 8 | 15.19 | 15.14 | +0.3 | 22.54 | 22.46 | +0.4 | +0.3 |
| 32 | 15.29 | 15.33 | -0.3 | 22.53 | 22.62 | -0.4 | -0.4 |
| 64 | 15.23 | 15.28 | -0.4 | 25.37 | 25.33 | +0.2 | -0.0 |
| 128 | 15.19 | 15.20 | -0.1 | 37.32 | 37.45 | -0.3 | -0.3 |
| 256 | 15.29 | 15.21 | +0.5 | 38.38 | 38.60 | -0.6 | -0.3 |
| 512 | 15.16 | 15.19 | -0.2 | 50.51 | 51.00 | -1.0 | -0.8 |

> Look for the T at which the anomaly appears and disappears.  T=1 uses `fused_quant_gemv` (has explicit `is_zero` guard), T>=2 uses `fused_dense_sparse_mma_int4` (no explicit guard).

## 3. T=128 shape family

| name | d_in | d_out | Δ_quant% | Δ_mma% | Δ_sum% |
|---|---:|---:|---:|---:|---:|
| kv_2560_2048 | 2560 | 2048 | +0.7 | -0.4 | -0.1 |
| q_2048_2048 | 2048 | 2048 | +0.8 | -0.2 | +0.1 |
| q_4096_4096 | 4096 | 4096 | +1.7 | -0.0 | +0.4 |
| gu_2048_12288 | 2048 | 12288 | +2.2 | +0.6 | +1.0 |
| down_3072_1024 | 3072 | 1024 | -0.6 | +0.5 | +0.1 |

> If most T=128 shapes show Δ_mma near 0 but 2560->2048 is the outlier, the anomaly is tied to a specific (d_in, d_out) or (n_groups, n_hp_blocks) combination.  If it's endemic, the root cause is in the shared mma_int4 code path.

## 4. Diagnosis

> **Scope note.**  §1..§3 above are single-shot snapshots (one `time_forward_us` call per cell, no trial aggregation), so re-running this script on a noisy shared-GPU host will show *different* single-cell |Δ| values each run.  The cells do not form the final verdict on their own; the formal adjudication uses `epilogue_pressure_test.py` which runs 5 interleaved trials per variant and reports median+[p05,p95].  §4 below is tied to that adjudication, not to any single cell of §1..§3.

**Verdict: the -27.9% X=0 slowdown on `mid_T128_kv_2560_2048` is a measurement artefact, not a kernel bug.**

**Evidence:**

1. Under the stronger (warmup=200, outer=10, inner=200) budget adopted from the first run of this probe, the bisection re-run of all 8 representatives showed `|Δ_xzero| <= 1.55%` -- a 18x reduction vs the original -27.9% number.  See `../bisection_summary.md`.
2. The order-reversal control (§1b) was in noise on the first run, ruling out L2-state bleed between variants as the mechanism.  (Transient re-runs can flip any single §1..§3 cell — see the scope note above.)
3. The follow-up pressure test (`../epilogue_pressure_test/pressure_test_report.md`) runs 5 interleaved trials per variant on the top-|Δ_scale_one| shapes plus a compute-bound control.  All 4 tested shapes (median-of-5) returned verdict = NOISE, with the compute-bound control tight to |Δ| <= 0.04%.  This rules out any real data-dependent kernel path on the tested shapes.

**Downstream changes applied:**

- `microbench_bisection.py::_time_variant`: default schedule bumped from (warmup=80, outer=4) to (warmup=200, outer=10) to eliminate the artefact class.
- `phase2_render_report.py::_attribute_bottleneck`: removed the `d_xzero <= -10%` branch; `x_zero_anomaly` is no longer a classification lever.
- `cluster_all_shapes.py::_cluster_shapes`: removed the exact-match carve-out for `mid_T128_kv_2560_2048`; it now falls through to nearest-neighbour classification.
- `phase3_render_roadmap.py`: deprecated the `x_zero_anomaly` ClusterPlan entry and dropped its verification-matrix row.
- All phase2 / phase3 artefacts under `cuda_kernel/logs/phase2_microscope/` re-generated from the new bisection runs.

**Corroboration by follow-up audit.**  The warmup-bump revealed that *all 8 representatives* now have `|Δ_scale_one| <= 2.2%`, which means the 43-shape `epilogue_fma_bound` cluster produced by the original threshold also collapses.  The pressure test (`kernel/tools/profile/epilogue_pressure_test.py`; warmup=500, outer=20, inner=200, 5 interleaved trials per variant on the top-|Δ_scale_one| shapes plus a compute-bound control) confirmed 3/3 suspect shapes and 1/1 control verdict = NOISE.  See `../epilogue_pressure_test/pressure_test_report.md`.

**New reclassification:**

- `mid_T128_kv_2560_2048` → `tc_underutil` (nearest-neighbour after the carve-out was removed).
- The stronger warmup also pushed every previous `epilogue_fma_bound` signal below the 2.5% scale=1 threshold, collapsing that cluster entirely.  The 100-shape roadmap is now a clean two-cluster partition: `tc_underutil` (83 shapes, ROI 2.74) and `launch_sparse` (17 shapes, ROI 2.44).

**Meta-lesson.**  The original 3-piece microbench rule (warmup, inner, outer) stored in the long-term memo was correct in spirit but the 80/100/4 instantiation used by `microbench_bisection.py` was still on the edge of the 4090's boost-clock warm-up envelope, **and even single-cell timings under the stronger (warmup=500, outer=20) budget can occasionally return a +49% outlier on this shared-GPU host** (see trial 2 of `mid_T128_kv_2560_2048` in the pressure-test JSON).  The only robust defence is median-of-K-trials aggregation; any single-shot A/B |Δ| should be treated as exploratory rather than adjudicatory.  This probe script therefore serves as an exploration tool; the pressure-test script is the adjudicator.
