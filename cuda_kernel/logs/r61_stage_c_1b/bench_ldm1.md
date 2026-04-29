# r61 Stage C.1b (LDMATRIX=1 + swizzle)

Full-shape bench of the CUDA INT4 W4A4 fused kernel (r54 stage-B.1) against a BF16 cuBLAS `torch.matmul` baseline on the same device, with Roofline efficiency computed per the canonical formulas in `kernel/tools/profile/roofline_delta.py` (RTX 4090 vendor peaks, ACHIEVABLE = 0.85).

| shape (d_out×d_in×T) | ng | INT4 (μs) | BF16 (μs) | speed-up | roof_INT4 (μs) | INT4 eff | roof_FP16 (μs) | BF16 eff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024×1024×128 | 8 | 13.96 | 12.80 | 0.92× | 1.42 | 10% | 3.06 | 24% |
| 2048×2048×128 | 16 | 27.00 | 13.75 | 0.51× | 4.14 | 15% | 11.01 | 80% |
| 4096×4096×128 | 32 | 60.81 | 33.23 | 0.55× | 13.48 | 22% | 41.61 | 125% |
| 1024×4096×128 | 32 | 21.64 | 12.84 | 0.59× | 4.76 | 22% | 11.32 | 88% |
| 4096×1024×128 | 8 | 17.16 | 10.36 | 0.60× | 4.29 | 25% | 11.32 | 109% |
| 2048×4096×128 | 32 | 33.16 | 18.96 | 0.57× | 7.67 | 23% | 21.42 | 113% |
| 4096×2048×128 | 16 | 31.77 | 17.20 | 0.54× | 7.35 | 23% | 21.42 | 125% |
| 4096×4096×32 | 32 | 47.41 | 21.32 | 0.45× | 11.17 | 24% | 39.77 | 187% |
| 4096×4096×1 | 32 | 35.22 | 17.02 | 0.48× | 10.42 | 30% | 39.18 | 230% |
| 4096×14336×128 | 112 | 141.85 | 151.19 | 1.07× | 44.13 | 31% | 142.58 | 94% |
| 14336×4096×128 | 32 | 93.42 | 150.58 | 1.61× | 42.55 | 46% | 142.58 | 95% |

## Aggregate

- median INT4/BF16 speed-up: **0.57×** (min 0.45× / max 1.61×)
- median INT4 roofline efficiency: **23.1%** (min 10.2% / max 45.5%)
- median BF16 roofline efficiency: **109.3%** (min 23.9% / max 230.3%)

