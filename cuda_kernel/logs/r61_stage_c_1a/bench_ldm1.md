# r61 Stage C.1a (LDMATRIX=1)

Full-shape bench of the CUDA INT4 W4A4 fused kernel (r54 stage-B.1) against a BF16 cuBLAS `torch.matmul` baseline on the same device, with Roofline efficiency computed per the canonical formulas in `kernel/tools/profile/roofline_delta.py` (RTX 4090 vendor peaks, ACHIEVABLE = 0.85).

| shape (d_out×d_in×T) | ng | INT4 (μs) | BF16 (μs) | speed-up | roof_INT4 (μs) | INT4 eff | roof_FP16 (μs) | BF16 eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024×1024×128 | 8 | 12.50 | 12.71 | 1.02× | 1.42 | 11% | 3.06 | 24% |
| 2048×2048×128 | 16 | 18.88 | 13.76 | 0.73× | 4.14 | 22% | 11.01 | 80% |
| 4096×4096×128 | 32 | 42.60 | 33.10 | 0.78× | 13.48 | 32% | 41.61 | 126% |
| 1024×4096×128 | 32 | 18.74 | 12.89 | 0.69× | 4.76 | 25% | 11.32 | 88% |
| 4096×1024×128 | 8 | 14.60 | 10.34 | 0.71× | 4.29 | 29% | 11.32 | 110% |
| 2048×4096×128 | 32 | 26.60 | 18.88 | 0.71× | 7.67 | 29% | 21.42 | 113% |
| 4096×2048×128 | 16 | 20.88 | 17.14 | 0.82× | 7.35 | 35% | 21.42 | 125% |
| 4096×4096×32 | 32 | 34.96 | 21.26 | 0.61× | 11.17 | 32% | 39.77 | 187% |
| 4096×4096×1 | 32 | 21.43 | 16.94 | 0.79× | 10.42 | 49% | 39.18 | 231% |
| 4096×14336×128 | 112 | 88.66 | 151.06 | 1.70× | 44.13 | 50% | 142.58 | 94% |
| 14336×4096×128 | 32 | 71.05 | 150.28 | 2.12× | 42.55 | 60% | 142.58 | 95% |

## Aggregate

- median INT4/BF16 speed-up: **0.78×** (min 0.61× / max 2.12×)
- median INT4 roofline efficiency: **31.6%** (min 11.4% / max 59.9%)
- median BF16 roofline efficiency: **109.5%** (min 24.1% / max 231.3%)

