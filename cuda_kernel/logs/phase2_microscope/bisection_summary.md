# Phase 2 microbench bisection summary

Each row is a representative shape.  `base` is a fresh build;
`l2_hot / x_zero / scale_one` are the three no-code-change
experiments.  `Δ%` = (base - variant) / base * 100 (positive
= faster).  |Δ| >= 3 % is considered meaningful (below that
is dominated by timer jitter even with min-of-means).

| shape | T | base_us | l2_hot_us | Δ_l2% | x_zero_us | Δ_xzero% | scale1_us | Δ_scale1% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_T1_kv_2560_2048 | 1 | 68.26 | 67.62 | +0.9 | 66.94 | +1.9 | 67.65 | +0.9 |
| decode_T1_q_2048_2048 | 1 | 46.54 | 45.52 | +2.2 | 45.50 | +2.2 | 45.55 | +2.1 |
| large_T1024_gu_4096_24576 | 1024 | 1126.72 | 1129.16 | -0.2 | 1125.94 | +0.1 | 1127.82 | -0.1 |
| mid_T128_kv_2560_2048 | 128 | 86.49 | 85.84 | +0.8 | 110.58 | -27.9 | 85.92 | +0.7 |
| prefill_T1024_down_3072_1024 | 1024 | 82.44 | 81.38 | +1.3 | 81.81 | +0.8 | 79.98 | +3.0 |
| prefill_T512_gu_2048_12288 | 512 | 160.72 | 161.35 | -0.4 | 160.68 | +0.0 | 161.36 | -0.4 |
| worst_T8_kv_1024_2048 | 8 | 82.94 | 81.14 | +2.2 | 80.91 | +2.5 | 80.21 | +3.3 |
| worst_T8_q_4096_4096 | 8 | 82.74 | 81.06 | +2.0 | 80.05 | +3.3 | 80.29 | +3.0 |

## Interpretation cheat-sheet

- **Δ_l2 >= 3 %**  → kernel is HBM-weight bound (L2 hot fetch helped).
- **Δ_xzero >= 3 %** → data-dependent latency on activation path (rare).
- **Δ_scale1 >= 3 %** → epilogue FMA is a real consumer; scale/zero path dominates tail.
- all three small and base close to roofline mem → memory-bound with no headroom.
- all three small and base far from roofline      → compute-bound / occupancy-bound (TC_underutil).
