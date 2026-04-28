# Roofline theoretical vs measured report

Source: `logs/qwen3_bench/qwen3_20260428_111515/bench.json`

GPU model: RTX 4090 (vendor spec, ACHIEVABLE_FRACTION=0.85)


## §1 Hardware constants and formulas

| Parameter | Value | Note |
| --- | --- | --- |
| HBM bandwidth | 1008 GB/s | RTX 4090 vendor spec |
| FP16/BF16 TC peak | 165.2 TFLOPS | boost clock, no sparsity |
| INT8 TC peak | 330.4 TOPS | boost clock |
| INT4 TC peak | 660.6 TOPS | boost clock |
| ACHIEVABLE_FRACTION | 0.85 | engineering derating |

**Formulas** (all time in microseconds, eff_* = peak * ACHIEVABLE_FRACTION):

- **FP16 roofline** — `t = max(flops / eff_fp16, bytes / eff_hbm)` where `flops = 2*T*d_in*d_out`, `bytes = 2*(d_in*d_out + T*d_in + T*d_out)`.
- **CUDA `activation_quant` (T>=2)** — pure mem-bound, `bytes = 2*T*d_in + 0.5*T*d_in + 2*T + 4*T*n_groups`.
- **CUDA `fused_dense_sparse` (T>=2)** — `t = max(2*T*d_in*d_out / eff_int4, bytes / eff_hbm)` with `bytes = 0.5*d_in*d_out + 0.5*T*d_in + 4*d_out*n_groups + 2*T*d_out`.
- **CUDA end-to-end (T>=2)** — `t_cuda_roof = t_quant + t_gemm` (serial sum, because ops.py has no stream overlap).
- **CUDA T=1 fused** — single stage roofline with quant traffic merged into GEMV: `bytes = 2*d_in + 0.5*d_in*d_out + 4*d_out*n_groups + 2*d_out`.

### Known systematic biases

1. **Kernel launch overhead** (5-10us/launch) is *not* counted in any roofline; the model will systematically under-estimate achievable time for T<=8 small shapes.
2. **L2 cache reuse** — back-to-back benches may let W stay in L2, making real mem time 20-40% below the pure-HBM roofline. Our roofline is therefore a conservative *upper bound* on achievable time — it may actually be *too pessimistic* relative to what the kernel achieves with L2 reuse.
3. **Tensor-Core utilisation** — 165.2 / 660.6 TFLOPS are vendor peaks; cuBLAS / hand-written kernels on irregular / unaligned shapes typically reach 85-95%. `ACHIEVABLE_FRACTION < 1.0` is the mandatory conservative term.
4. **Reduce in activation_quant** — a CTA-wide max-abs reduce is not strictly pure mem-bound, but its cost is <1us and is neglected.
5. **Epilogue FMA cost** — each output point runs n_groups dequant FMAs on CUDA Cores; we fold this into the INT4 TC peak which is an *optimistic* simplification. In reality the GEMM stage is often CUDA-Core-FMA-bound, and this is the main reason measured `cuda_efficiency` falls below 1.0.

> If `cuda_efficiency > 1.0` for some row, the row is tagged `⚠ L2/roof low-bound` in §4 — this indicates L2 hit or model pessimism.

## §2 FP16 efficiency distribution (by T)

`fp16_efficiency = fp16_roof_us / fp16_us` — how close cuBLAS gets to 4090's physical limit.

| T | n | min | median | max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 55% | 103% | 167% |
| 8 | 20 | 42% | 105% | 293% |
| 128 | 20 | 51% | 93% | 136% |
| 512 | 20 | 92% | 105% | 118% |
| 1024 | 20 | 90% | 108% | 116% |

## §3 CUDA efficiency distribution

`cuda_efficiency = cuda_roof_us / cuda_us` — how close our W4A4 kernel gets to its own roofline.

### §3.1 By T

| T | n | min | median | max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 17% | 39% | 65% |
| 8 | 20 | 5% | 13% | 95% |
| 128 | 20 | 9% | 13% | 39% |
| 512 | 20 | 19% | 26% | 36% |
| 1024 | 20 | 27% | 34% | 40% |

### §3.2 By proj

| proj | n | min | median | max |
| :--- | ---: | ---: | ---: | ---: |
| q_proj | 20 | 5% | 23% | 55% |
| kv_proj | 20 | 5% | 21% | 40% |
| o_proj | 20 | 5% | 23% | 55% |
| gate_up_proj | 20 | 15% | 37% | 95% |
| down_proj | 20 | 6% | 25% | 58% |

## §4 Per-shape detail tables (core section, 100 rows)

One subtable per Qwen3 model. One row per (proj, T) covering all bench.json end_to_end records.

### Qwen3-0.6B

| proj | T | shape | fp16_us | fp16_roof_us | fp16_eff | cuda_us | cuda_roof_us | cuda_eff | cuda/fp16 actual (roof) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| q_proj | 1 | 1024→2048 | 8.78 | 4.90 | 56% | 7.55 | 1.31 | 17% | 0.86x (0.27x) |
| q_proj | 8 | 1024→2048 | 9.21 | 4.95 | 54% | 26.53 | 1.37 | 5% | 2.88x (0.28x) |
| q_proj | 128 | 1024→2048 | 9.44 | 5.81 | 62% | 26.84 | 2.38 | 9% | 2.84x (0.41x) |
| q_proj | 512 | 1024→2048 | 15.53 | 15.29 | 98% | 28.79 | 5.60 | 19% | 1.85x (0.37x) |
| q_proj | 1024 | 1024→2048 | 29.82 | 30.59 | 103% | 39.22 | 10.75 | 27% | 1.32x (0.35x) |
| kv_proj | 1 | 1024→2048 | 8.77 | 4.90 | 56% | 7.39 | 1.31 | 18% | 0.84x (0.27x) |
| kv_proj | 8 | 1024→2048 | 9.18 | 4.95 | 54% | 26.97 | 1.37 | 5% | 2.94x (0.28x) |
| kv_proj | 128 | 1024→2048 | 9.44 | 5.81 | 62% | 26.90 | 2.38 | 9% | 2.85x (0.41x) |
| kv_proj | 512 | 1024→2048 | 15.61 | 15.29 | 98% | 28.87 | 5.60 | 19% | 1.85x (0.37x) |
| kv_proj | 1024 | 1024→2048 | 29.78 | 30.59 | 103% | 39.26 | 10.75 | 27% | 1.32x (0.35x) |
| o_proj | 1 | 2048→1024 | 8.91 | 4.90 | 55% | 7.43 | 1.31 | 18% | 0.83x (0.27x) |
| o_proj | 8 | 2048→1024 | 11.92 | 4.95 | 42% | 26.62 | 1.38 | 5% | 2.23x (0.28x) |
| o_proj | 128 | 2048→1024 | 11.44 | 5.81 | 51% | 28.69 | 2.53 | 9% | 2.51x (0.44x) |
| o_proj | 512 | 2048→1024 | 16.68 | 15.29 | 92% | 36.69 | 6.92 | 19% | 2.20x (0.45x) |
| o_proj | 1024 | 2048→1024 | 34.08 | 30.59 | 90% | 47.27 | 13.85 | 29% | 1.39x (0.45x) |
| gate_up_proj | 1 | 1024→6144 | 15.17 | 14.70 | 97% | 9.51 | 3.92 | 41% | 0.63x (0.27x) |
| gate_up_proj | 8 | 1024→6144 | 9.15 | 14.82 | 162% | 26.76 | 4.04 | 15% | 2.93x (0.27x) |
| gate_up_proj | 128 | 1024→6144 | 15.82 | 16.83 | 106% | 26.78 | 6.20 | 23% | 1.69x (0.37x) |
| gate_up_proj | 512 | 1024→6144 | 44.01 | 45.88 | 104% | 60.74 | 13.10 | 22% | 1.38x (0.29x) |
| gate_up_proj | 1024 | 1024→6144 | 80.00 | 91.76 | 115% | 97.41 | 26.05 | 27% | 1.22x (0.28x) |
| down_proj | 1 | 3072→1024 | 8.73 | 7.35 | 84% | 9.96 | 1.96 | 20% | 1.14x (0.27x) |
| down_proj | 8 | 3072→1024 | 11.92 | 7.42 | 62% | 35.87 | 2.06 | 6% | 3.01x (0.28x) |
| down_proj | 128 | 3072→1024 | 11.50 | 8.57 | 74% | 41.68 | 3.65 | 9% | 3.62x (0.43x) |
| down_proj | 512 | 3072→1024 | 23.36 | 22.94 | 98% | 49.76 | 10.38 | 21% | 2.13x (0.45x) |
| down_proj | 1024 | 3072→1024 | 46.95 | 45.88 | 98% | 68.01 | 20.77 | 31% | 1.45x (0.45x) |

### Qwen3-1.7B

| proj | T | shape | fp16_us | fp16_roof_us | fp16_eff | cuda_us | cuda_roof_us | cuda_eff | cuda/fp16 actual (roof) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| q_proj | 1 | 2048→2048 | 8.63 | 9.80 | 114% | 7.43 | 2.61 | 35% | 0.86x (0.27x) |
| q_proj | 8 | 2048→2048 | 11.96 | 9.87 | 83% | 27.18 | 2.70 | 10% | 2.27x (0.27x) |
| q_proj | 128 | 2048→2048 | 12.87 | 11.01 | 86% | 33.76 | 4.14 | 12% | 2.62x (0.38x) |
| q_proj | 512 | 2048→2048 | 27.85 | 30.59 | 110% | 46.44 | 10.75 | 23% | 1.67x (0.35x) |
| q_proj | 1024 | 2048→2048 | 55.95 | 61.17 | 109% | 61.90 | 21.50 | 35% | 1.11x (0.35x) |
| kv_proj | 1 | 2048→2048 | 8.55 | 9.80 | 115% | 7.45 | 2.61 | 35% | 0.87x (0.27x) |
| kv_proj | 8 | 2048→2048 | 11.88 | 9.87 | 83% | 26.91 | 2.70 | 10% | 2.27x (0.27x) |
| kv_proj | 128 | 2048→2048 | 12.87 | 11.01 | 86% | 33.74 | 4.14 | 12% | 2.62x (0.38x) |
| kv_proj | 512 | 2048→2048 | 27.88 | 30.59 | 110% | 46.58 | 10.75 | 23% | 1.67x (0.35x) |
| kv_proj | 1024 | 2048→2048 | 55.95 | 61.17 | 109% | 62.01 | 21.50 | 35% | 1.11x (0.35x) |
| o_proj | 1 | 2048→2048 | 8.57 | 9.80 | 114% | 7.44 | 2.61 | 35% | 0.87x (0.27x) |
| o_proj | 8 | 2048→2048 | 11.99 | 9.87 | 82% | 26.99 | 2.70 | 10% | 2.25x (0.27x) |
| o_proj | 128 | 2048→2048 | 12.85 | 11.01 | 86% | 33.76 | 4.14 | 12% | 2.63x (0.38x) |
| o_proj | 512 | 2048→2048 | 27.87 | 30.59 | 110% | 46.56 | 10.75 | 23% | 1.67x (0.35x) |
| o_proj | 1024 | 2048→2048 | 55.97 | 61.17 | 109% | 61.90 | 21.50 | 35% | 1.11x (0.35x) |
| gate_up_proj | 1 | 2048→12288 | 55.23 | 58.78 | 106% | 25.48 | 15.64 | 61% | 0.46x (0.27x) |
| gate_up_proj | 8 | 2048→12288 | 20.77 | 59.01 | 284% | 27.64 | 15.89 | 58% | 1.33x (0.27x) |
| gate_up_proj | 128 | 2048→12288 | 46.74 | 63.03 | 135% | 53.63 | 20.20 | 38% | 1.15x (0.32x) |
| gate_up_proj | 512 | 2048→12288 | 156.94 | 183.52 | 117% | 153.68 | 48.99 | 32% | 0.98x (0.27x) |
| gate_up_proj | 1024 | 2048→12288 | 327.49 | 367.04 | 112% | 293.17 | 97.99 | 33% | 0.90x (0.27x) |
| down_proj | 1 | 6144→2048 | 29.94 | 29.39 | 98% | 20.81 | 7.82 | 38% | 0.69x (0.27x) |
| down_proj | 8 | 6144→2048 | 12.21 | 29.52 | 242% | 66.22 | 8.01 | 12% | 5.43x (0.27x) |
| down_proj | 128 | 6144→2048 | 24.64 | 31.82 | 129% | 89.87 | 11.20 | 12% | 3.65x (0.35x) |
| down_proj | 512 | 6144→2048 | 78.04 | 91.76 | 118% | 119.74 | 32.24 | 27% | 1.53x (0.35x) |
| down_proj | 1024 | 6144→2048 | 158.65 | 183.52 | 116% | 162.34 | 64.48 | 40% | 1.02x (0.35x) |

### Qwen3-4B

| proj | T | shape | fp16_us | fp16_roof_us | fp16_eff | cuda_us | cuda_roof_us | cuda_eff | cuda/fp16 actual (roof) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| q_proj | 1 | 2560→4096 | 24.72 | 24.49 | 99% | 13.22 | 6.52 | 49% | 0.53x (0.27x) |
| q_proj | 8 | 2560→4096 | 12.03 | 24.60 | 204% | 30.76 | 6.65 | 22% | 2.56x (0.27x) |
| q_proj | 128 | 2560→4096 | 20.12 | 26.47 | 132% | 46.53 | 8.89 | 19% | 2.31x (0.34x) |
| q_proj | 512 | 2560→4096 | 72.85 | 76.47 | 105% | 72.13 | 23.00 | 32% | 0.99x (0.30x) |
| q_proj | 1024 | 2560→4096 | 143.58 | 152.93 | 107% | 135.82 | 45.99 | 34% | 0.95x (0.30x) |
| kv_proj | 1 | 2560→2048 | 9.90 | 12.25 | 124% | 10.01 | 3.26 | 33% | 1.01x (0.27x) |
| kv_proj | 8 | 2560→2048 | 11.99 | 12.32 | 103% | 30.65 | 3.36 | 11% | 2.56x (0.27x) |
| kv_proj | 128 | 2560→2048 | 14.56 | 13.62 | 94% | 45.78 | 5.02 | 11% | 3.14x (0.37x) |
| kv_proj | 512 | 2560→2048 | 34.44 | 38.23 | 111% | 59.80 | 13.43 | 22% | 1.74x (0.35x) |
| kv_proj | 1024 | 2560→2048 | 72.80 | 76.47 | 105% | 73.57 | 26.87 | 37% | 1.01x (0.35x) |
| o_proj | 1 | 4096→2560 | 24.25 | 24.49 | 101% | 16.25 | 6.52 | 40% | 0.67x (0.27x) |
| o_proj | 8 | 4096→2560 | 12.09 | 24.60 | 203% | 46.45 | 6.67 | 14% | 3.84x (0.27x) |
| o_proj | 128 | 4096→2560 | 23.06 | 26.47 | 115% | 65.57 | 9.12 | 14% | 2.84x (0.34x) |
| o_proj | 512 | 4096→2560 | 74.16 | 76.47 | 103% | 97.31 | 25.32 | 26% | 1.31x (0.33x) |
| o_proj | 1024 | 4096→2560 | 142.29 | 152.93 | 107% | 180.43 | 50.64 | 28% | 1.27x (0.33x) |
| gate_up_proj | 1 | 2560→19456 | 106.20 | 116.32 | 110% | 49.12 | 30.93 | 63% | 0.46x (0.27x) |
| gate_up_proj | 8 | 2560→19456 | 115.50 | 116.67 | 101% | 49.75 | 31.32 | 63% | 0.43x (0.27x) |
| gate_up_proj | 128 | 2560→19456 | 136.15 | 122.84 | 90% | 119.48 | 37.86 | 32% | 0.88x (0.31x) |
| gate_up_proj | 512 | 2560→19456 | 375.29 | 363.22 | 97% | 292.83 | 94.70 | 32% | 0.78x (0.26x) |
| gate_up_proj | 1024 | 2560→19456 | 680.12 | 726.43 | 107% | 557.02 | 189.41 | 34% | 0.82x (0.26x) |
| down_proj | 1 | 9728→2560 | 55.65 | 58.16 | 105% | 38.36 | 15.47 | 40% | 0.69x (0.27x) |
| down_proj | 8 | 9728→2560 | 19.89 | 58.36 | 293% | 116.81 | 15.76 | 13% | 5.87x (0.27x) |
| down_proj | 128 | 9728→2560 | 45.43 | 61.80 | 136% | 152.73 | 20.61 | 13% | 3.36x (0.33x) |
| down_proj | 512 | 9728→2560 | 185.21 | 181.61 | 98% | 205.54 | 60.13 | 29% | 1.11x (0.33x) |
| down_proj | 1024 | 9728→2560 | 331.33 | 363.22 | 110% | 400.30 | 120.26 | 30% | 1.21x (0.33x) |

### Qwen3-8B

| proj | T | shape | fp16_us | fp16_roof_us | fp16_eff | cuda_us | cuda_roof_us | cuda_eff | cuda/fp16 actual (roof) |
| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| q_proj | 1 | 4096→4096 | 38.37 | 39.18 | 102% | 18.88 | 10.42 | 55% | 0.49x (0.27x) |
| q_proj | 8 | 4096→4096 | 14.62 | 39.32 | 269% | 47.66 | 10.59 | 22% | 3.26x (0.27x) |
| q_proj | 128 | 4096→4096 | 30.94 | 41.61 | 134% | 63.17 | 13.48 | 21% | 2.04x (0.32x) |
| q_proj | 512 | 4096→4096 | 115.03 | 122.35 | 106% | 103.43 | 36.79 | 36% | 0.90x (0.30x) |
| q_proj | 1024 | 4096→4096 | 212.62 | 244.69 | 115% | 191.77 | 73.59 | 38% | 0.90x (0.30x) |
| kv_proj | 1 | 4096→2048 | 19.21 | 19.60 | 102% | 13.92 | 5.22 | 37% | 0.72x (0.27x) |
| kv_proj | 8 | 4096→2048 | 12.04 | 19.70 | 164% | 45.47 | 5.36 | 12% | 3.78x (0.27x) |
| kv_proj | 128 | 4096→2048 | 19.43 | 21.42 | 110% | 62.47 | 7.67 | 12% | 3.21x (0.36x) |
| kv_proj | 512 | 4096→2048 | 53.10 | 61.17 | 115% | 81.70 | 21.49 | 26% | 1.54x (0.35x) |
| kv_proj | 1024 | 4096→2048 | 114.94 | 122.35 | 106% | 108.17 | 42.99 | 40% | 0.94x (0.35x) |
| o_proj | 1 | 4096→4096 | 23.50 | 39.18 | 167% | 18.91 | 10.42 | 55% | 0.80x (0.27x) |
| o_proj | 8 | 4096→4096 | 14.66 | 39.32 | 268% | 47.72 | 10.59 | 22% | 3.25x (0.27x) |
| o_proj | 128 | 4096→4096 | 30.90 | 41.61 | 135% | 63.16 | 13.48 | 21% | 2.04x (0.32x) |
| o_proj | 512 | 4096→4096 | 113.99 | 122.35 | 107% | 103.40 | 36.79 | 36% | 0.91x (0.30x) |
| o_proj | 1024 | 4096→4096 | 212.91 | 244.69 | 115% | 191.79 | 73.59 | 38% | 0.90x (0.30x) |
| gate_up_proj | 1 | 4096→24576 | 212.49 | 235.04 | 111% | 95.47 | 62.48 | 65% | 0.45x (0.27x) |
| gate_up_proj | 8 | 4096→24576 | 221.35 | 235.51 | 106% | 66.61 | 62.99 | 95% | 0.30x (0.27x) |
| gate_up_proj | 128 | 4096→24576 | 263.71 | 243.54 | 92% | 182.17 | 71.61 | 39% | 0.69x (0.29x) |
| gate_up_proj | 512 | 4096→24576 | 750.82 | 734.08 | 98% | 527.63 | 189.77 | 36% | 0.70x (0.26x) |
| gate_up_proj | 1024 | 4096→24576 | 1363.17 | 1468.16 | 108% | 1023.32 | 379.54 | 37% | 0.75x (0.26x) |
| down_proj | 1 | 12288→4096 | 109.70 | 117.53 | 107% | 53.98 | 31.25 | 58% | 0.49x (0.27x) |
| down_proj | 8 | 12288→4096 | 118.30 | 117.79 | 100% | 142.23 | 31.63 | 22% | 1.20x (0.27x) |
| down_proj | 128 | 12288→4096 | 133.84 | 122.38 | 91% | 194.44 | 38.00 | 20% | 1.45x (0.31x) |
| down_proj | 512 | 12288→4096 | 359.49 | 367.04 | 102% | 304.11 | 110.38 | 36% | 0.85x (0.30x) |
| down_proj | 1024 | 12288→4096 | 715.37 | 734.08 | 103% | 563.74 | 220.75 | 39% | 0.79x (0.30x) |

_Detail rows rendered: 100._

## §5 CUDA implementation-gap TOP-15 (worst cuda_efficiency)

Complement to §4 — zoom on the shapes furthest from their own roofline.

| rank | model | proj | T | shape | cuda_us | t_quant_roof | t_gemm_roof | cuda_roof | cuda_eff | gemm_bound |
| ---: | :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 1 | Qwen3-0.6B | kv_proj | 8 | 1024→2048 | 26.97 | 0.02 | 1.34 | 1.37 | 5% | mem |
| 2 | Qwen3-0.6B | q_proj | 8 | 1024→2048 | 26.53 | 0.02 | 1.34 | 1.37 | 5% | mem |
| 3 | Qwen3-0.6B | o_proj | 8 | 2048→1024 | 26.62 | 0.05 | 1.33 | 1.38 | 5% | mem |
| 4 | Qwen3-0.6B | down_proj | 8 | 3072→1024 | 35.87 | 0.07 | 1.98 | 2.06 | 6% | mem |
| 5 | Qwen3-0.6B | down_proj | 128 | 3072→1024 | 41.68 | 1.16 | 2.49 | 3.65 | 9% | mem |
| 6 | Qwen3-0.6B | o_proj | 128 | 2048→1024 | 28.69 | 0.77 | 1.76 | 2.53 | 9% | mem |
| 7 | Qwen3-0.6B | kv_proj | 128 | 1024→2048 | 26.90 | 0.39 | 1.99 | 2.38 | 9% | mem |
| 8 | Qwen3-0.6B | q_proj | 128 | 1024→2048 | 26.84 | 0.39 | 1.99 | 2.38 | 9% | mem |
| 9 | Qwen3-1.7B | q_proj | 8 | 2048→2048 | 27.18 | 0.05 | 2.65 | 2.70 | 10% | mem |
| 10 | Qwen3-1.7B | o_proj | 8 | 2048→2048 | 26.99 | 0.05 | 2.65 | 2.70 | 10% | mem |
| 11 | Qwen3-1.7B | kv_proj | 8 | 2048→2048 | 26.91 | 0.05 | 2.65 | 2.70 | 10% | mem |
| 12 | Qwen3-4B | kv_proj | 8 | 2560→2048 | 30.65 | 0.06 | 3.30 | 3.36 | 11% | mem |
| 13 | Qwen3-4B | kv_proj | 128 | 2560→2048 | 45.78 | 0.97 | 4.05 | 5.02 | 11% | mem |
| 14 | Qwen3-8B | kv_proj | 8 | 4096→2048 | 45.47 | 0.10 | 5.26 | 5.36 | 12% | mem |
| 15 | Qwen3-1.7B | down_proj | 8 | 6144→2048 | 66.22 | 0.15 | 7.87 | 8.01 | 12% | mem |

## §6 Roofline-side CUDA vs FP16  (theoretical ceiling comparison)

`cuda_roof_us / fp16_roof_us` — shows which shapes **W4A4 cannot beat FP16 even at the physical limit**. Values >1.0 mean the FP16 ceiling is *faster*.

### Qwen3-0.6B

| proj | T | shape | fp16_roof_us | cuda_roof_us | cuda_roof / fp16_roof |
| :--- | ---: | :---: | ---: | ---: | :---: |
| q_proj | 1 | 1024→2048 | 4.90 | 1.31 | 0.27x |
| q_proj | 8 | 1024→2048 | 4.95 | 1.37 | 0.28x |
| q_proj | 128 | 1024→2048 | 5.81 | 2.38 | 0.41x |
| q_proj | 512 | 1024→2048 | 15.29 | 5.60 | 0.37x |
| q_proj | 1024 | 1024→2048 | 30.59 | 10.75 | 0.35x |
| kv_proj | 1 | 1024→2048 | 4.90 | 1.31 | 0.27x |
| kv_proj | 8 | 1024→2048 | 4.95 | 1.37 | 0.28x |
| kv_proj | 128 | 1024→2048 | 5.81 | 2.38 | 0.41x |
| kv_proj | 512 | 1024→2048 | 15.29 | 5.60 | 0.37x |
| kv_proj | 1024 | 1024→2048 | 30.59 | 10.75 | 0.35x |
| o_proj | 1 | 2048→1024 | 4.90 | 1.31 | 0.27x |
| o_proj | 8 | 2048→1024 | 4.95 | 1.38 | 0.28x |
| o_proj | 128 | 2048→1024 | 5.81 | 2.53 | 0.44x |
| o_proj | 512 | 2048→1024 | 15.29 | 6.92 | 0.45x |
| o_proj | 1024 | 2048→1024 | 30.59 | 13.85 | 0.45x |
| gate_up_proj | 1 | 1024→6144 | 14.70 | 3.92 | 0.27x |
| gate_up_proj | 8 | 1024→6144 | 14.82 | 4.04 | 0.27x |
| gate_up_proj | 128 | 1024→6144 | 16.83 | 6.20 | 0.37x |
| gate_up_proj | 512 | 1024→6144 | 45.88 | 13.10 | 0.29x |
| gate_up_proj | 1024 | 1024→6144 | 91.76 | 26.05 | 0.28x |
| down_proj | 1 | 3072→1024 | 7.35 | 1.96 | 0.27x |
| down_proj | 8 | 3072→1024 | 7.42 | 2.06 | 0.28x |
| down_proj | 128 | 3072→1024 | 8.57 | 3.65 | 0.43x |
| down_proj | 512 | 3072→1024 | 22.94 | 10.38 | 0.45x |
| down_proj | 1024 | 3072→1024 | 45.88 | 20.77 | 0.45x |

### Qwen3-1.7B

| proj | T | shape | fp16_roof_us | cuda_roof_us | cuda_roof / fp16_roof |
| :--- | ---: | :---: | ---: | ---: | :---: |
| q_proj | 1 | 2048→2048 | 9.80 | 2.61 | 0.27x |
| q_proj | 8 | 2048→2048 | 9.87 | 2.70 | 0.27x |
| q_proj | 128 | 2048→2048 | 11.01 | 4.14 | 0.38x |
| q_proj | 512 | 2048→2048 | 30.59 | 10.75 | 0.35x |
| q_proj | 1024 | 2048→2048 | 61.17 | 21.50 | 0.35x |
| kv_proj | 1 | 2048→2048 | 9.80 | 2.61 | 0.27x |
| kv_proj | 8 | 2048→2048 | 9.87 | 2.70 | 0.27x |
| kv_proj | 128 | 2048→2048 | 11.01 | 4.14 | 0.38x |
| kv_proj | 512 | 2048→2048 | 30.59 | 10.75 | 0.35x |
| kv_proj | 1024 | 2048→2048 | 61.17 | 21.50 | 0.35x |
| o_proj | 1 | 2048→2048 | 9.80 | 2.61 | 0.27x |
| o_proj | 8 | 2048→2048 | 9.87 | 2.70 | 0.27x |
| o_proj | 128 | 2048→2048 | 11.01 | 4.14 | 0.38x |
| o_proj | 512 | 2048→2048 | 30.59 | 10.75 | 0.35x |
| o_proj | 1024 | 2048→2048 | 61.17 | 21.50 | 0.35x |
| gate_up_proj | 1 | 2048→12288 | 58.78 | 15.64 | 0.27x |
| gate_up_proj | 8 | 2048→12288 | 59.01 | 15.89 | 0.27x |
| gate_up_proj | 128 | 2048→12288 | 63.03 | 20.20 | 0.32x |
| gate_up_proj | 512 | 2048→12288 | 183.52 | 48.99 | 0.27x |
| gate_up_proj | 1024 | 2048→12288 | 367.04 | 97.99 | 0.27x |
| down_proj | 1 | 6144→2048 | 29.39 | 7.82 | 0.27x |
| down_proj | 8 | 6144→2048 | 29.52 | 8.01 | 0.27x |
| down_proj | 128 | 6144→2048 | 31.82 | 11.20 | 0.35x |
| down_proj | 512 | 6144→2048 | 91.76 | 32.24 | 0.35x |
| down_proj | 1024 | 6144→2048 | 183.52 | 64.48 | 0.35x |

### Qwen3-4B

| proj | T | shape | fp16_roof_us | cuda_roof_us | cuda_roof / fp16_roof |
| :--- | ---: | :---: | ---: | ---: | :---: |
| q_proj | 1 | 2560→4096 | 24.49 | 6.52 | 0.27x |
| q_proj | 8 | 2560→4096 | 24.60 | 6.65 | 0.27x |
| q_proj | 128 | 2560→4096 | 26.47 | 8.89 | 0.34x |
| q_proj | 512 | 2560→4096 | 76.47 | 23.00 | 0.30x |
| q_proj | 1024 | 2560→4096 | 152.93 | 45.99 | 0.30x |
| kv_proj | 1 | 2560→2048 | 12.25 | 3.26 | 0.27x |
| kv_proj | 8 | 2560→2048 | 12.32 | 3.36 | 0.27x |
| kv_proj | 128 | 2560→2048 | 13.62 | 5.02 | 0.37x |
| kv_proj | 512 | 2560→2048 | 38.23 | 13.43 | 0.35x |
| kv_proj | 1024 | 2560→2048 | 76.47 | 26.87 | 0.35x |
| o_proj | 1 | 4096→2560 | 24.49 | 6.52 | 0.27x |
| o_proj | 8 | 4096→2560 | 24.60 | 6.67 | 0.27x |
| o_proj | 128 | 4096→2560 | 26.47 | 9.12 | 0.34x |
| o_proj | 512 | 4096→2560 | 76.47 | 25.32 | 0.33x |
| o_proj | 1024 | 4096→2560 | 152.93 | 50.64 | 0.33x |
| gate_up_proj | 1 | 2560→19456 | 116.32 | 30.93 | 0.27x |
| gate_up_proj | 8 | 2560→19456 | 116.67 | 31.32 | 0.27x |
| gate_up_proj | 128 | 2560→19456 | 122.84 | 37.86 | 0.31x |
| gate_up_proj | 512 | 2560→19456 | 363.22 | 94.70 | 0.26x |
| gate_up_proj | 1024 | 2560→19456 | 726.43 | 189.41 | 0.26x |
| down_proj | 1 | 9728→2560 | 58.16 | 15.47 | 0.27x |
| down_proj | 8 | 9728→2560 | 58.36 | 15.76 | 0.27x |
| down_proj | 128 | 9728→2560 | 61.80 | 20.61 | 0.33x |
| down_proj | 512 | 9728→2560 | 181.61 | 60.13 | 0.33x |
| down_proj | 1024 | 9728→2560 | 363.22 | 120.26 | 0.33x |

### Qwen3-8B

| proj | T | shape | fp16_roof_us | cuda_roof_us | cuda_roof / fp16_roof |
| :--- | ---: | :---: | ---: | ---: | :---: |
| q_proj | 1 | 4096→4096 | 39.18 | 10.42 | 0.27x |
| q_proj | 8 | 4096→4096 | 39.32 | 10.59 | 0.27x |
| q_proj | 128 | 4096→4096 | 41.61 | 13.48 | 0.32x |
| q_proj | 512 | 4096→4096 | 122.35 | 36.79 | 0.30x |
| q_proj | 1024 | 4096→4096 | 244.69 | 73.59 | 0.30x |
| kv_proj | 1 | 4096→2048 | 19.60 | 5.22 | 0.27x |
| kv_proj | 8 | 4096→2048 | 19.70 | 5.36 | 0.27x |
| kv_proj | 128 | 4096→2048 | 21.42 | 7.67 | 0.36x |
| kv_proj | 512 | 4096→2048 | 61.17 | 21.49 | 0.35x |
| kv_proj | 1024 | 4096→2048 | 122.35 | 42.99 | 0.35x |
| o_proj | 1 | 4096→4096 | 39.18 | 10.42 | 0.27x |
| o_proj | 8 | 4096→4096 | 39.32 | 10.59 | 0.27x |
| o_proj | 128 | 4096→4096 | 41.61 | 13.48 | 0.32x |
| o_proj | 512 | 4096→4096 | 122.35 | 36.79 | 0.30x |
| o_proj | 1024 | 4096→4096 | 244.69 | 73.59 | 0.30x |
| gate_up_proj | 1 | 4096→24576 | 235.04 | 62.48 | 0.27x |
| gate_up_proj | 8 | 4096→24576 | 235.51 | 62.99 | 0.27x |
| gate_up_proj | 128 | 4096→24576 | 243.54 | 71.61 | 0.29x |
| gate_up_proj | 512 | 4096→24576 | 734.08 | 189.77 | 0.26x |
| gate_up_proj | 1024 | 4096→24576 | 1468.16 | 379.54 | 0.26x |
| down_proj | 1 | 12288→4096 | 117.53 | 31.25 | 0.27x |
| down_proj | 8 | 12288→4096 | 117.79 | 31.63 | 0.27x |
| down_proj | 128 | 12288→4096 | 122.38 | 38.00 | 0.31x |
| down_proj | 512 | 12288→4096 | 367.04 | 110.38 | 0.30x |
| down_proj | 1024 | 12288→4096 | 734.08 | 220.75 | 0.30x |

## §7 Conclusions and next steps

- Out of **100** shapes, **1** reach `cuda_efficiency >= 0.8` — these are already near the physical limit; further kernel tuning has diminishing ROI.
- **91** shapes sit at `cuda_efficiency < 0.5` — these have real implementation slack; cross-reference §5 for the worst offenders and their mem/compute bound to pick the next kernel to fix.
- **0** shapes have `cuda_roof > fp16_roof` — W4A4 loses even at the ceiling; these should fall back to FP16 via policy, no kernel work can rescue them.
- Measured today, **63** shapes actually lose to FP16; the delta `63 - 0` is the *implementation* gap (fixable), the rest is *physics* (unfixable).
- Recommended next moves: (a) for the §5 top offenders whose `gemm_bound == mem`, audit packed-weight layout / cache reuse; (b) for those with `gemm_bound == compute` and narrow d_out, revisit CTA sizing (see R41-R46 iteration); (c) for shapes listed as `✗ W4A4 ceiling slower` in §6, route via policy.py to FP16.
