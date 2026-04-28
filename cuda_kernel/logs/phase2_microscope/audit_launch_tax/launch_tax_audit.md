# Launch-tax audit under tightened timer

_Generated 2026-04-28T11:49:44.757567Z; K=5 trials × (warmup=200, outer=10, inner=100); alternating plain/graph order per trial._

## Motivation

The Phase 2 roadmap assigned 17 shapes to the ``launch_sparse`` cluster based on ``launch_tax_pct >= 50%`` measured under ``(warmup=50, outer=3, inner=100)`` — the same budget that was later proven to fabricate the ``x_zero_anomaly`` and ``epilogue_fma_bound`` clusters.  This audit re-measures the verdict under the tightened schedule before R49 commits to CUDA-Graph based optimisation.

## Verdict rule

* ``CONFIRMED``: median launch_tax_pct >= 50% — stays in cluster.
* ``DEGRADED``: 30% <= median launch_tax_pct < 50% — stays, lower ROI.
* ``REJECTED``: median launch_tax_pct < 30% — reclassify.
* ``UNSTABLE`` flag: max-min range >= 20pp — audit-only (still applies the median-based verdict).

## Summary

| verdict | n |
|---|---:|
| CONFIRMED | 14 |
| DEGRADED | 2 |
| REJECTED | 1 |
| CAPTURE_FAILED | 0 |

## Per-shape verdict

| tag | T | d_in | d_out | plain (us) | graph (us) | tax (us) | tax% median | [min,max] | verdict | flags |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `audit_4B_gu_T1_2560_19456` | 1 | 2560 | 19456 | 48.53 | 47.62 | 0.89 | 1.8% | [1.6, 2.1] | REJECTED | - |
| `audit_1p7B_gu_T1_2048_12288` | 1 | 2048 | 12288 | 44.47 | 27.45 | 17.03 | 38.3% | [38.0, 38.7] | DEGRADED | - |
| `audit_1p7B_gu_T8_2048_12288` | 8 | 2048 | 12288 | 52.73 | 28.31 | 24.42 | 46.3% | [46.2, 46.5] | DEGRADED | - |
| `audit_1p7B_kv_T8_2048_2048` | 8 | 2048 | 2048 | 52.70 | 26.03 | 26.67 | 50.6% | [50.4, 50.8] | CONFIRMED | - |
| `audit_1p7B_q_T8_2048_2048` | 8 | 2048 | 2048 | 52.70 | 26.02 | 26.68 | 50.6% | [50.6, 50.7] | CONFIRMED | - |
| `audit_1p7B_o_T8_2048_2048` | 8 | 2048 | 2048 | 52.93 | 26.04 | 26.89 | 50.8% | [50.6, 51.0] | CONFIRMED | - |
| `audit_0p6B_o_T8_2048_1024` | 8 | 2048 | 1024 | 53.00 | 25.55 | 27.45 | 51.8% | [51.5, 63.3] | CONFIRMED | - |
| `audit_4B_q_T1_2560_4096` | 1 | 2560 | 4096 | 44.09 | 17.81 | 26.29 | 59.6% | [59.6, 59.6] | CONFIRMED | - |
| `audit_0p6B_gu_T8_1024_6144` | 8 | 1024 | 6144 | 52.21 | 16.68 | 35.52 | 68.0% | [68.0, 68.2] | CONFIRMED | - |
| `audit_1p7B_q_T1_2048_2048` | 1 | 2048 | 2048 | 44.38 | 11.95 | 32.42 | 73.0% | [73.0, 73.2] | CONFIRMED | - |
| `audit_1p7B_o_T1_2048_2048` | 1 | 2048 | 2048 | 44.56 | 12.01 | 32.55 | 73.1% | [73.0, 73.1] | CONFIRMED | - |
| `audit_1p7B_kv_T1_2048_2048` | 1 | 2048 | 2048 | 44.55 | 11.99 | 32.56 | 73.1% | [73.0, 73.2] | CONFIRMED | - |
| `audit_0p6B_gu_T1_1024_6144` | 1 | 1024 | 6144 | 44.31 | 10.82 | 33.48 | 75.5% | [75.5, 75.6] | CONFIRMED | - |
| `audit_0p6B_kv_T8_1024_2048` | 8 | 1024 | 2048 | 76.89 | 17.02 | 59.88 | 77.9% | [77.8, 78.0] | CONFIRMED | - |
| `audit_0p6B_q_T8_1024_2048` | 8 | 1024 | 2048 | 78.99 | 17.07 | 61.92 | 78.4% | [77.9, 78.5] | CONFIRMED | - |
| `audit_0p6B_kv_T1_1024_2048` | 1 | 1024 | 2048 | 64.42 | 8.52 | 55.90 | 86.8% | [86.8, 86.9] | CONFIRMED | - |
| `audit_0p6B_q_T1_1024_2048` | 1 | 1024 | 2048 | 66.14 | 8.51 | 57.64 | 87.2% | [87.1, 87.3] | CONFIRMED | - |

## Cluster stability

Of 17 cluster members: 14 CONFIRMED, 2 DEGRADED, 1 REJECTED.

Cluster is **partially stable**: CUDA-Graph work remains worthwhile for the confirmed subset, but ROI must be recomputed over the reduced shape count.

