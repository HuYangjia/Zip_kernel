# Phase 2 microbench bisection summary

Each row is a representative shape.  `base` is a fresh build;
`l2_hot / x_zero / scale_one` are the three no-code-change
experiments.  `Δ%` = (base - variant) / base * 100 (positive
= faster).  |Δ| >= 3 % is considered meaningful (below that
is dominated by timer jitter even with min-of-means).

| shape | T | base_us | l2_hot_us | Δ_l2% | x_zero_us | Δ_xzero% | scale1_us | Δ_scale1% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_T1_kv_2560_2048 | 1 | 43.10 | 42.75 | +0.8 | 42.54 | +1.3 | 42.62 | +1.1 |
| decode_T1_q_2048_2048 | 1 | 42.02 | 41.62 | +1.0 | 41.56 | +1.1 | 41.67 | +0.9 |
| large_T1024_gu_4096_24576 | 1024 | 1127.25 | 1128.92 | -0.1 | 1125.67 | +0.1 | 1128.01 | -0.1 |
| mid_T128_kv_2560_2048 | 128 | 78.86 | 78.34 | +0.7 | 77.64 | +1.5 | 77.14 | +2.2 |
| prefill_T1024_down_3072_1024 | 1024 | 79.17 | 79.27 | -0.1 | 78.53 | +0.8 | 79.30 | -0.2 |
| prefill_T512_gu_2048_12288 | 512 | 161.16 | 161.91 | -0.5 | 161.24 | -0.1 | 161.94 | -0.5 |
| worst_T8_kv_1024_2048 | 8 | 52.29 | 51.78 | +1.0 | 51.72 | +1.1 | 51.87 | +0.8 |
| worst_T8_q_4096_4096 | 8 | 75.80 | 74.79 | +1.3 | 74.98 | +1.1 | 74.28 | +2.0 |

## Interpretation cheat-sheet

- **Δ_l2 >= 3 %**  → kernel is HBM-weight bound (L2 hot fetch helped).
- **Δ_xzero >= 3 %** → data-dependent latency on activation path (rare).
- **Δ_scale1 >= 3 %** → epilogue FMA is a real consumer; scale/zero path dominates tail.
- all three small and base close to roofline mem → memory-bound with no headroom.
- all three small and base far from roofline      → compute-bound / occupancy-bound (TC_underutil).
