# r61 Stage F — occupancy-driven gate tightening

Full-shape bench of the CUDA INT4 W4A4 fused kernel (r54 stage-B.1) against a BF16 cuBLAS `torch.matmul` baseline on the same device, with Roofline efficiency computed per the canonical formulas in `kernel/tools/profile/roofline_delta.py` (RTX 4090 vendor peaks, ACHIEVABLE = 0.85).

| shape (d_out×d_in×T) | ng | INT4 (μs) | BF16 (μs) | speed-up | roof_INT4 (μs) | INT4 eff | roof_FP16 (μs) | BF16 eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024×1024×128 | 8 | 11.80 | 12.55 | 1.06× | 1.42 | 12% | 3.06 | 24% |
| 2048×2048×128 | 16 | 17.15 | 13.76 | 0.80× | 4.14 | 24% | 11.01 | 80% |
| 4096×4096×128 | 32 | 41.22 | 33.23 | 0.81× | 13.48 | 33% | 41.61 | 125% |
| 1024×4096×128 | 32 | 14.89 | 12.84 | 0.86× | 4.76 | 32% | 11.32 | 88% |
| 4096×1024×128 | 8 | 14.26 | 10.33 | 0.72× | 4.29 | 30% | 11.32 | 110% |
| 2048×4096×128 | 32 | 22.72 | 18.89 | 0.83× | 7.67 | 34% | 21.42 | 113% |
| 4096×2048×128 | 16 | 19.89 | 17.18 | 0.86× | 7.35 | 37% | 21.42 | 125% |
| 4096×4096×32 | 32 | 34.80 | 21.30 | 0.61× | 11.17 | 32% | 39.77 | 187% |
| 4096×4096×1 | 32 | 21.00 | 16.98 | 0.81× | 10.42 | 50% | 39.18 | 231% |
| 4096×14336×128 | 112 | 84.85 | 151.13 | 1.78× | 44.13 | 52% | 142.58 | 94% |
| 14336×4096×128 | 32 | 67.08 | 150.46 | 2.24× | 42.55 | 63% | 142.58 | 95% |

## Aggregate

- median INT4/BF16 speed-up: **0.83×** (min 0.61× / max 2.24×)
- median INT4 roofline efficiency: **32.7%** (min 12.0% / max 63.4%)
- median BF16 roofline efficiency: **109.6%** (min 24.4% / max 230.7%)

