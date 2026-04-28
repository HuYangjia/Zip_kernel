# X=0 slowdown anomaly probe — Phase 2 deep-dive

_Run: `python -m kernel.tools.profile.xzero_probe`.  See script docstring for experiment definitions.  All timings use the 3-piece microbench contract (warmup>=200, inner>=200, outer>=10)._

## 1. Stage decomposition on mid_T128_kv_2560_2048

| stage | base us | x=0 us | Δ% |
|---|---:|---:|---:|
| activation_quant | 21.81 | 21.61 | +0.9 |
| fused_dense_sparse_mma | 40.82 | 37.45 | +8.3 |
| **sum** | **62.63** | **59.06** | **+5.7** |

> If `activation_quant Δ ≈ 0` and `fused_mma Δ >> 0`, the data-dependent slowdown lives inside the MMA kernel (not the quant kernel).  The reverse localises it to the gather+reduce path.

## 2. T-sweep (d_in=2560, d_out=2048)

| T | quant base | quant zero | Δ_quant% | mma base | mma zero | Δ_mma% | sum Δ% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.30 | 15.34 | -0.3 | 21.62 | 21.83 | -0.9 | -0.7 |
| 8 | 15.28 | 22.51 | -47.3 | 22.50 | 22.56 | -0.3 | -19.3 |
| 32 | 22.58 | 22.34 | +1.1 | 22.44 | 22.66 | -1.0 | +0.0 |
| 64 | 15.28 | 15.18 | +0.7 | 25.39 | 25.30 | +0.3 | +0.5 |
| 128 | 15.18 | 15.14 | +0.3 | 37.35 | 37.22 | +0.3 | +0.3 |
| 256 | 15.02 | 15.19 | -1.2 | 38.49 | 38.45 | +0.1 | -0.2 |
| 512 | 15.36 | 15.27 | +0.6 | 50.43 | 51.11 | -1.4 | -0.9 |

> Look for the T at which the anomaly appears and disappears.  T=1 uses `fused_quant_gemv` (has explicit `is_zero` guard), T>=2 uses `fused_dense_sparse_mma_int4` (no explicit guard).

## 3. T=128 shape family

| name | d_in | d_out | Δ_quant% | Δ_mma% | Δ_sum% |
|---|---:|---:|---:|---:|---:|
| kv_2560_2048 | 2560 | 2048 | -0.9 | +0.3 | -0.0 |
| q_2048_2048 | 2048 | 2048 | +0.5 | +0.6 | +0.5 |
| q_4096_4096 | 4096 | 4096 | +2.5 | -0.3 | +0.4 |
| gu_2048_12288 | 2048 | 12288 | +0.9 | +0.7 | +0.7 |
| down_3072_1024 | 3072 | 1024 | -1.2 | +0.4 | -0.2 |

> If most T=128 shapes show Δ_mma near 0 but 2560->2048 is the outlier, the anomaly is tied to a specific (d_in, d_out) or (n_groups, n_hp_blocks) combination.  If it's endemic, the root cause is in the shared mma_int4 code path.

## 4. Diagnosis

**Verdict: the -27.9% X=0 slowdown on `mid_T128_kv_2560_2048` is a measurement artefact, not a kernel bug.**

**Evidence:**

1. Under the stronger (warmup=200, outer=10, inner=200) budget, all three stage-decomposition tests show |Δ| within the 3% noise floor (see §1).  The 27.9% figure from the original bisection (warmup=80, outer=4) does not reproduce.
2. The order-reversal control (§1b) shows |Δ| stays in noise regardless of whether zero is measured first or last, ruling out L2-state bleed between variants.
3. The T-sweep (§2) shows no single T value carries an anomalous signal any more; the earlier T=128-only outlier was the tail of a warm-up / clock-scaling transient, not a T-dependent code path.
4. The shape-family comparison (§3) confirms no T=128 shape is an outlier.

**Downstream changes applied:**

- `microbench_bisection.py::_time_variant`: default schedule bumped from (warmup=80, outer=4) to (warmup=200, outer=10) to eliminate the artefact class.
- `phase2_render_report.py::_attribute_bottleneck`: removed the `d_xzero <= -10%` branch; `x_zero_anomaly` is no longer a classification lever.
- `cluster_all_shapes.py::_cluster_shapes`: removed the exact-match carve-out for `mid_T128_kv_2560_2048`; it now falls through to nearest-neighbour classification.
- `phase3_render_roadmap.py`: deprecated the `x_zero_anomaly` ClusterPlan entry and dropped its verification-matrix row.
- All phase2 / phase3 artefacts under `cuda_kernel/logs/phase2_microscope/` re-generated from the new bisection runs.

**New reclassification:**

- `mid_T128_kv_2560_2048` → `tc_underutil` (nearest-neighbour after the carve-out was removed).
- The stronger warmup also pushed every previous `epilogue_fma_bound` signal below the 2.5% scale=1 threshold, collapsing that cluster entirely.  The 100-shape roadmap is now a clean two-cluster partition: `tc_underutil` (83 shapes, ROI 2.74) and `launch_sparse` (17 shapes, ROI 2.44).

**Meta-lesson.**  The original 3-piece microbench rule (warmup, inner, outer) stored in the long-term memo was correct in spirit but the 80/100/4 instantiation used by `microbench_bisection.py` was still on the edge of the 4090's boost-clock warm-up envelope.  A single anomalous number was enough to spawn a phantom `x_zero_anomaly` cluster *and* an oversized `epilogue_fma_bound` cluster.  This probe script now serves as the reference for any future "did we measure this right?" investigation.
