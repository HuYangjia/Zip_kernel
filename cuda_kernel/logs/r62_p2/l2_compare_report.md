# Full-shape bench vs BF16 + Roofline — L2-flush comparison

Side-by-side bench with and without L2-cache flush before each inner launch.  The tight-loop mode (no flush) inflates BF16 cuBLAS because a ≤72 MB problem hits L2 after the first miss; the flushed mode forces every launch to re-fetch from HBM, matching a real LLM inference workload (one weight read per layer).

| shape (d_out×d_in×T) | ng | INT4 tight | INT4 cold | BF16 tight | BF16 cold | speedup tight | speedup cold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024×1024×128 | 8 | 11.77 | 12.22 | 16.21 | 4.87 | 1.38× | 0.40× |
| 2048×2048×128 | 16 | 17.14 | 14.02 | 15.68 | 13.28 | 0.91× | 0.95× |
| 4096×4096×128 | 32 | 37.34 | 37.25 | 32.74 | 42.55 | 0.88× | 1.14× |
| 1024×4096×128 | 32 | 14.57 | 9.80 | 16.31 | 13.49 | 1.12× | 1.38× |
| 4096×1024×128 | 8 | 13.98 | 13.42 | 13.09 | 12.75 | 0.94× | 0.95× |
| 2048×4096×128 | 32 | 22.26 | 22.46 | 18.52 | 23.81 | 0.83× | 1.06× |
| 4096×2048×128 | 16 | 19.59 | 19.98 | 16.89 | 22.23 | 0.86× | 1.11× |
| 4096×4096×32 | 32 | 28.19 | 35.72 | 20.99 | 39.62 | 0.74× | 1.11× |
| 4096×4096×1 | 32 | 20.73 | 33.81 | 16.68 | 40.20 | 0.80× | 1.19× |
| 4096×14336×128 | 112 | 83.73 | 94.71 | 151.10 | 164.82 | 1.80× | 1.74× |
| 14336×4096×128 | 32 | 65.90 | 74.40 | 151.84 | 159.36 | 2.30× | 2.14× |

## Aggregate deltas

- median speed-up: tight=0.91× → cold=1.11× (Δ +0.20×)
- BF16 slowdown from L2 flush: median 1.41× (16.9→23.8us)
- INT4 slowdown from L2 flush: median 1.08× (20.7→22.5us)

