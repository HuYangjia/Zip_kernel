# r61 Stage G — dispatch gate fix

Full-shape bench of the CUDA INT4 W4A4 fused kernel (r54 stage-B.1) against a BF16 cuBLAS `torch.matmul` baseline on the same device, with Roofline efficiency computed per the canonical formulas in `kernel/tools/profile/roofline_delta.py` (RTX 4090 vendor peaks, ACHIEVABLE = 0.85).

| shape (d_out×d_in×T) | ng | INT4 (μs) | BF16 (μs) | speed-up | roof_INT4 (μs) | INT4 eff | roof_FP16 (μs) | BF16 eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024×1024×128 | 8 | 11.83 | 20.49 | 1.73× | 1.42 | 12% | 3.06 | 15% |
| 2048×2048×128 | 16 | 17.19 | 20.14 | 1.17× | 4.14 | 24% | 11.01 | 55% |
| 4096×4096×128 | 32 | 37.44 | 32.81 | 0.88× | 13.48 | 36% | 41.61 | 127% |
| 1024×4096×128 | 32 | 16.28 | 20.46 | 1.26× | 4.76 | 29% | 11.32 | 55% |
| 4096×1024×128 | 8 | 14.09 | 15.68 | 1.11× | 4.29 | 30% | 11.32 | 72% |
| 2048×4096×128 | 32 | 22.47 | 21.83 | 0.97× | 7.67 | 34% | 21.42 | 98% |
| 4096×2048×128 | 16 | 19.61 | 16.94 | 0.86× | 7.35 | 37% | 21.42 | 126% |
| 4096×4096×32 | 32 | 28.23 | 21.34 | 0.76× | 11.17 | 40% | 39.77 | 186% |
| 4096×4096×1 | 32 | 20.75 | 16.75 | 0.81× | 10.42 | 50% | 39.18 | 234% |
| 4096×14336×128 | 112 | 83.74 | 151.58 | 1.81× | 44.13 | 53% | 142.58 | 94% |
| 14336×4096×128 | 32 | 65.96 | 152.18 | 2.31× | 42.55 | 65% | 142.58 | 94% |

## Aggregate

- median INT4/BF16 speed-up: **1.11×** (min 0.76× / max 2.31×)
- median INT4 roofline efficiency: **36.0%** (min 12.0% / max 64.5%)
- median BF16 roofline efficiency: **94.1%** (min 14.9% / max 234.0%)

