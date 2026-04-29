# Final bench r57 vs BF16 + Roofline (RTX 4090)

Full-shape bench of the CUDA INT4 W4A4 fused kernel (r54 stage-B.1) against a BF16 cuBLAS `torch.matmul` baseline on the same device, with Roofline efficiency computed per the canonical formulas in `kernel/tools/profile/roofline_delta.py` (RTX 4090 vendor peaks, ACHIEVABLE = 0.85).

| shape (d_out×d_in×T) | ng | INT4 (μs) | BF16 (μs) | speed-up | roof_INT4 (μs) | INT4 eff | roof_FP16 (μs) | BF16 eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024×1024×128 | 8 | 11.32 | 17.21 | 1.52× | 1.42 | 13% | 3.06 | 18% |
| 2048×2048×128 | 16 | 15.08 | 16.67 | 1.11× | 4.14 | 27% | 11.01 | 66% |
| 4096×4096×128 | 32 | 37.18 | 33.35 | 0.90× | 13.48 | 36% | 41.61 | 125% |
| 1024×4096×128 | 32 | 27.72 | 17.15 | 0.62× | 4.76 | 17% | 11.32 | 66% |
| 4096×1024×128 | 8 | 14.37 | 13.78 | 0.96× | 4.29 | 30% | 11.32 | 82% |
| 2048×4096×128 | 32 | 28.07 | 18.94 | 0.67× | 7.67 | 27% | 21.42 | 113% |
| 4096×2048×128 | 16 | 20.09 | 17.21 | 0.86× | 7.35 | 37% | 21.42 | 124% |
| 4096×4096×32 | 32 | 34.76 | 21.37 | 0.61× | 11.17 | 32% | 39.77 | 186% |
| 4096×4096×1 | 32 | 21.43 | 17.03 | 0.80× | 10.42 | 49% | 39.18 | 230% |
| 4096×14336×128 | 112 | 140.82 | 151.10 | 1.07× | 44.13 | 31% | 142.58 | 94% |
| 14336×4096×128 | 32 | 69.51 | 150.62 | 2.17× | 42.55 | 61% | 142.58 | 95% |

## Aggregate

- median INT4/BF16 speed-up: **0.90×** (min 0.61× / max 2.17×)
- median INT4 roofline efficiency: **31.3%** (min 12.5% / max 61.2%)
- median BF16 roofline efficiency: **94.7%** (min 17.8% / max 230.0%)

