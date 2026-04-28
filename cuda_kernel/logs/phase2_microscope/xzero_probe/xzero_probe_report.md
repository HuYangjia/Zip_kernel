# X=0 slowdown anomaly probe — Phase 2 deep-dive

_Run: `python -m kernel.tools.profile.xzero_probe`.  See script docstring for experiment definitions.  All timings use the 3-piece microbench contract (warmup>=200, inner>=200, outer>=10)._

## 1. Stage decomposition on mid_T128_kv_2560_2048

| stage | base us | x=0 us | Δ% |
|---|---:|---:|---:|
| activation_quant | 22.30 | 22.02 | +1.3 |
| fused_dense_sparse_mma | 40.81 | 37.41 | +8.3 |
| **sum** | **63.11** | **59.42** | **+5.8** |

> If `activation_quant Δ ≈ 0` and `fused_mma Δ >> 0`, the data-dependent slowdown lives inside the MMA kernel (not the quant kernel).  The reverse localises it to the gather+reduce path.

## 2. T-sweep (d_in=2560, d_out=2048)

| T | quant base | quant zero | Δ_quant% | mma base | mma zero | Δ_mma% | sum Δ% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.86 | 22.15 | -1.3 | 21.89 | 21.70 | +0.9 | -0.2 |
| 8 | 21.97 | 21.94 | +0.1 | 22.55 | 22.41 | +0.6 | +0.4 |
| 32 | 21.42 | 21.27 | +0.7 | 22.54 | 22.55 | -0.1 | +0.3 |
| 64 | 21.15 | 21.11 | +0.2 | 25.30 | 25.30 | -0.0 | +0.1 |
| 128 | 21.48 | 21.21 | +1.2 | 37.34 | 37.18 | +0.4 | +0.7 |
| 256 | 21.00 | 20.93 | +0.3 | 38.37 | 38.51 | -0.4 | -0.1 |
| 512 | 21.13 | 21.17 | -0.2 | 50.45 | 51.03 | -1.1 | -0.9 |

> Look for the T at which the anomaly appears and disappears.  T=1 uses `fused_quant_gemv` (has explicit `is_zero` guard), T>=2 uses `fused_dense_sparse_mma_int4` (no explicit guard).

## 3. T=128 shape family

| name | d_in | d_out | Δ_quant% | Δ_mma% | Δ_sum% |
|---|---:|---:|---:|---:|---:|
| kv_2560_2048 | 2560 | 2048 | -0.8 | +0.2 | -0.2 |
| q_2048_2048 | 2048 | 2048 | -0.6 | -0.5 | -0.5 |
| q_4096_4096 | 4096 | 4096 | +0.9 | -0.2 | +0.1 |
| gu_2048_12288 | 2048 | 12288 | +1.3 | +0.8 | +0.9 |
| down_3072_1024 | 3072 | 1024 | -0.3 | +0.3 | +0.0 |

> If most T=128 shapes show Δ_mma near 0 but 2560->2048 is the outlier, the anomaly is tied to a specific (d_in, d_out) or (n_groups, n_hp_blocks) combination.  If it's endemic, the root cause is in the shared mma_int4 code path.

## 4. Diagnosis (to fill after running)

Based on §1, the slowdown is localised to `<stage>`.  §2 shows the effect is `<T-dependent / T-independent>` which points to `<dispatch path>`.  §3 shows the effect is `<shape-specific / endemic>`.  Root cause: `<hypothesis>`.  Next step: `<code fix or further probe>`.
