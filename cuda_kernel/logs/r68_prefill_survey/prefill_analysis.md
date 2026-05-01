# r68 prefill-scenario unified analysis (T=1..8192)

Combined records: 270
Models: ['Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B', 'Qwen3-14B', 'Qwen2.5-32B', 'LLaMA3-70B']
Ts: [1, 8, 32, 128, 512, 1024, 2048, 4096, 8192]

## §A. Global T-sweep (median speedup / median cuda_eff across 5 projs per (model,T))

### Speedup vs FP16

| model | params |T=1 | T=8 | T=32 | T=128 | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 1.50× | 0.34× | 0.34× | 0.38× | 0.58× | 0.99× | 1.13× | 1.14× | 1.15× |
| Qwen3-4B | 4.0B | 1.65× | 0.77× | 0.77× | 0.78× | 0.97× | 1.13× | 1.20× | 1.23× | 1.26× |
| Qwen3-8B | 8.0B | 2.14× | 1.14× | 1.15× | 0.91× | 1.35× | 1.31× | 1.30× | 1.32× | 1.32× |
| Qwen3-14B | 14.0B | 2.23× | 1.78× | 1.57× | 1.06× | 0.90× | 0.98× | 0.97× | 0.99× | 1.03× |
| Qwen2.5-32B | 32.0B | 2.02× | 1.78× | 1.57× | 1.02× | 0.88× | 1.04× | 1.00× | 1.00× | 1.06× |
| LLaMA3-70B | 70.0B | 2.01× | 1.55× | 1.87× | 1.14× | 1.02× | 1.08× | 1.07× | 1.09× | 1.13× |

### CUDA efficiency (cuda_roof / cuda_us)

| model | params |T=1 | T=8 | T=32 | T=128 | T=512 | T=1024 | T=2048 | T=4096 | T=8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 1.7B | 35% | 8% | 9% | 12% | 31% | 43% | 45% | 48% | 47% |
| Qwen3-4B | 4.0B | 41% | 19% | 20% | 25% | 31% | 41% | 43% | 45% | 44% |
| Qwen3-8B | 8.0B | 56% | 30% | 32% | 29% | 41% | 45% | 45% | 45% | 46% |
| Qwen3-14B | 14.0B | 60% | 46% | 42% | 31% | 33% | 36% | 33% | 32% | 33% |
| Qwen2.5-32B | 32.0B | 59% | 46% | 42% | 29% | 33% | 36% | 33% | 32% | 33% |
| LLaMA3-70B | 70.0B | 59% | 42% | 51% | 30% | 35% | 35% | 35% | 35% | 35% |

## §B. Cross-model comparison at each T (how does speedup scale with model size?)

For each T, we list median speedup by model (ordered by param count).
If speedup rises with model size, the kernel is benefiting from bigger tile utilisation.
If it stays flat or drops, FP16 cuBLAS also scales and we're not gaining.

| T | 1.7B | 4B | 8B | 14B | 32B | 70B |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.50× | 1.65× | 2.14× | 2.23× | 2.02× | 2.01× |
| 8 | 0.34× | 0.77× | 1.14× | 1.78× | 1.78× | 1.55× |
| 32 | 0.34× | 0.77× | 1.15× | 1.57× | 1.57× | 1.87× |
| 128 | 0.38× | 0.78× | 0.91× | 1.06× | 1.02× | 1.14× |
| 512 | 0.58× | 0.97× | 1.35× | 0.90× | 0.88× | 1.02× |
| 1024 | 0.99× | 1.13× | 1.31× | 0.98× | 1.04× | 1.08× |
| 2048 | 1.13× | 1.20× | 1.30× | 0.97× | 1.00× | 1.07× |
| 4096 | 1.14× | 1.23× | 1.32× | 0.99× | 1.00× | 1.09× |
| 8192 | 1.15× | 1.26× | 1.32× | 1.03× | 1.06× | 1.13× |

## §C. Per-shape deep-dive at T=2048 (representative prefill point)

| model | proj | shape | fp16_us | cuda_us | speedup | cuda_eff | bound | gap_reason |
|---|---|---|---:|---:|---:|---:|:---:|---|
| Qwen3-1.7B | q_proj | 2048→2048 | 107.9 | 95.4 | 1.13× | 45% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-1.7B | kv_proj | 2048→2048 | 108.1 | 95.6 | 1.13× | 45% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-1.7B | o_proj | 2048→2048 | 108.1 | 95.7 | 1.13× | 45% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-1.7B | gate_up_proj | 2048→12288 | 632.2 | 469.4 | 1.35× | 42% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-1.7B | down_proj | 6144→2048 | 302.9 | 340.5 | 0.89× | 38% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen3-4B | q_proj | 2560→4096 | 258.8 | 210.5 | 1.23× | 44% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-4B | kv_proj | 2560→2048 | 143.6 | 119.6 | 1.20× | 45% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-4B | o_proj | 4096→2560 | 274.4 | 235.8 | 1.16× | 43% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-4B | gate_up_proj | 2560→19456 | 1286.6 | 916.5 | 1.40× | 41% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-4B | down_proj | 9728→2560 | 658.1 | 688.4 | 0.96× | 35% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen3-8B | q_proj | 4096→4096 | 434.5 | 322.6 | 1.35× | 46% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-8B | kv_proj | 4096→2048 | 225.9 | 186.6 | 1.21× | 46% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-8B | o_proj | 4096→4096 | 423.7 | 326.2 | 1.30× | 45% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-8B | gate_up_proj | 4096→24576 | 2569.1 | 1770.5 | 1.45× | 43% | compute | fp16 exceeds vendor roof (L2 reuse) |
| Qwen3-8B | down_proj | 12288→4096 | 1345.0 | 1301.6 | 1.03× | 34% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen3-14B | q_proj | 5120→5120 | 651.7 | 671.6 | 0.97× | 33% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen3-14B | kv_proj | 5120→2048 | 262.6 | 272.8 | 0.96× | 39% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen3-14B | o_proj | 5120→5120 | 669.7 | 667.9 | 1.00× | 33% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen3-14B | gate_up_proj | 5120→34816 | 4607.1 | 6024.0 | 0.76× | 22% | compute | low cuda_eff (kernel sub-par); fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen3-14B | down_proj | 17408→5120 | 2388.7 | 2436.9 | 0.98× | 31% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen2.5-32B | q_proj | 5120→5120 | 672.2 | 671.1 | 1.00× | 33% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen2.5-32B | kv_proj | 5120→2048 | 263.1 | 270.0 | 0.97× | 40% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen2.5-32B | o_proj | 5120→5120 | 672.6 | 669.4 | 1.00× | 33% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen2.5-32B | gate_up_proj | 5120→55296 | 7298.8 | 10192.7 | 0.72× | 21% | compute | low cuda_eff (kernel sub-par); fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| Qwen2.5-32B | down_proj | 27648→5120 | 3790.6 | 3613.0 | 1.05× | 33% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| LLaMA3-70B | q_proj | 8192→8192 | 1775.7 | 1545.1 | 1.15× | 35% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| LLaMA3-70B | kv_proj | 8192→2048 | 410.2 | 493.0 | 0.83× | 35% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| LLaMA3-70B | o_proj | 8192→8192 | 1795.6 | 1549.5 | 1.16× | 35% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| LLaMA3-70B | gate_up_proj | 8192→57344 | 12153.3 | 17409.0 | 0.70× | 20% | compute | low cuda_eff (kernel sub-par); fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |
| LLaMA3-70B | down_proj | 28672→8192 | 6330.4 | 5889.2 | 1.07× | 32% | compute | fp16 exceeds vendor roof (L2 reuse); compute-bound / MMA starvation |

## §D. Root-cause analysis — why model-size and T influence speedup

### D.1 Why does speedup rise with model size? (e.g. 1.7B 0.78× → 8B 1.35×)

Hypothesis 1: **Grid utilisation**.  Bigger d_out = more m-tiles, better SM utilisation.  At T=2048, kv_proj has the same d_out=2048 across all models (same m-grid size), so this only explains q/gu/dn.

Hypothesis 2: **Launch-overhead amortisation**.  Each cuda kernel has ~15us activation_quant launch floor.  Small-shape kernels (cuda_us < 40us) are 40-50% launch overhead.  Bigger models have bigger work per launch so overhead fraction drops.

Check: compute `launch_overhead_fraction = 15us / cuda_us`.

- Qwen3-1.7B     T=2048: cuda_us=95.7, overhead=16%, median sp=1.13×
- Qwen3-4B       T=2048: cuda_us=235.8, overhead=6%, median sp=1.20×
- Qwen3-8B       T=2048: cuda_us=326.2, overhead=5%, median sp=1.30×
- Qwen3-14B      T=2048: cuda_us=671.6, overhead=2%, median sp=0.97×
- Qwen2.5-32B    T=2048: cuda_us=671.1, overhead=2%, median sp=1.00×
- LLaMA3-70B     T=2048: cuda_us=1549.5, overhead=1%, median sp=1.07×

### D.2 Why does speedup drop as T grows?

At T=1 median sp ≈ 2.0× (W4A4 gemv kills fp16 gemv).  At T=4096+ median sp ≈ 0.9-1.0× (fp16 tensorcore gets full benefit, W4A4 also compute-bound).  The transition is at T ≈ 128-512 where cuBLAS switches from gemv-optimised to tensor-core GEMM internally.

- T=    1: median sp=1.99×  cuda_eff=56%  fp16_eff=100%  compute-bound=0/30
- T=    8: median sp=1.27×  cuda_eff=34%  fp16_eff=96%  compute-bound=0/30
- T=   32: median sp=1.17×  cuda_eff=33%  fp16_eff=96%  compute-bound=0/30
- T=  128: median sp=1.00×  cuda_eff=29%  fp16_eff=91%  compute-bound=0/30
- T=  512: median sp=0.96×  cuda_eff=33%  fp16_eff=110%  compute-bound=30/30
- T= 1024: median sp=1.08×  cuda_eff=36%  fp16_eff=112%  compute-bound=30/30
- T= 2048: median sp=1.06×  cuda_eff=36%  fp16_eff=113%  compute-bound=30/30
- T= 4096: median sp=1.08×  cuda_eff=36%  fp16_eff=113%  compute-bound=30/30
- T= 8192: median sp=1.11×  cuda_eff=36%  fp16_eff=110%  compute-bound=30/30
