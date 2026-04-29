# Full-shape bench vs BF16 + Roofline

Full-shape bench of the CUDA INT4 W4A4 fused kernel (r54 stage-B.1) against a BF16 cuBLAS `torch.matmul` baseline on the same device, with Roofline efficiency computed per the canonical formulas in `kernel/tools/profile/roofline_delta.py` (RTX 4090 vendor peaks, ACHIEVABLE = 0.85).

| shape (d_out×d_in×T) | ng | INT4 (μs) | BF16 (μs) | speed-up | roof_INT4 (μs) | INT4 eff | roof_FP16 (μs) | BF16 eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024×1024×128 | 8 | 11.78 | 12.61 | 1.07× | 1.42 | 12% | 3.06 | 24% |
| 2048×2048×128 | 16 | 16.79 | 13.76 | 0.82× | 4.14 | 25% | 11.01 | 80% |
| 4096×4096×128 | 32 | 36.71 | 33.20 | 0.90× | 13.48 | 37% | 41.61 | 125% |
| 1024×4096×128 | 32 | 18.17 | 12.79 | 0.70× | 4.76 | 26% | 11.32 | 89% |
| 4096×1024×128 | 8 | 14.25 | 10.34 | 0.73× | 4.29 | 30% | 11.32 | 110% |
| 2048×4096×128 | 32 | 26.21 | 18.90 | 0.72× | 7.67 | 29% | 21.42 | 113% |
| 4096×2048×128 | 16 | 20.06 | 17.14 | 0.85× | 7.35 | 37% | 21.42 | 125% |
| 4096×4096×32 | 32 | 34.73 | 21.27 | 0.61× | 11.17 | 32% | 39.77 | 187% |
| 4096×4096×1 | 32 | 20.96 | 16.94 | 0.81× | 10.42 | 50% | 39.18 | 231% |
| 4096×14336×128 | 112 | 84.82 | 151.06 | 1.78× | 44.13 | 52% | 142.58 | 94% |
| 14336×4096×128 | 32 | 69.93 | 150.15 | 2.15× | 42.55 | 61% | 142.58 | 95% |

## Aggregate

- median INT4/BF16 speed-up: **0.82×** (min 0.61× / max 2.15×)
- median INT4 roofline efficiency: **32.2%** (min 12.1% / max 60.8%)
- median BF16 roofline efficiency: **109.5%** (min 24.3% / max 231.3%)

