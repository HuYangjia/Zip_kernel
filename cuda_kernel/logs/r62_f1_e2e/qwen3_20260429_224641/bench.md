# Qwen3 multi-scale kernel benchmark

- Timestamp: `20260429_224641`
- Device: `NVIDIA GeForce RTX 4090`
- PyTorch: `2.8.0+cu126`  Triton: `3.4.0`
- Baseline: cuBLAS FP16 matmul (`torch.matmul` on `fp16`)
- CUDA path: `activation_quant_cuda` + `fused_dense_sparse_cuda` (T=1 uses `fused_quant_gemv_cuda`, with automatic fallback on unsupported decode-group counts)
- Triton path: `quantize_activation_s4` + `fused_dense_sparse_gemm`
- hp_ratio: `0.05`  (block-sparse density)
- Stats: stable microbenchmark helper = 50 warmup, 100 inner, 3 repeats, min-of-means


## 1. End-to-end speedup vs FP16

Rows: projection. Cells: `fp16_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-8B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 4096->4096 | **2.18x** | **0.97x** | **0.81x** | **1.36x** |
| kv_proj | 4096->2048 | **1.52x** | **0.62x** | **0.54x** | **0.78x** |
| o_proj | 4096->4096 | **2.16x** | **0.97x** | **0.81x** | **1.36x** |
| gate_up_proj | 4096->24576 | **2.25x** | **3.25x** | **1.87x** | **1.54x** |
| down_proj | 12288->4096 | **2.10x** | **0.80x** | **0.86x** | **1.07x** |


## 2. End-to-end raw latencies (us)


### Qwen3-8B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 12288->4096 | 1 | 112.40 | 270.65 | 53.43 | 0.42x | 2.10x | 5.07x |
| down_proj | 12288->4096 | 32 | 123.35 | 316.98 | 153.40 | 0.39x | 0.80x | 2.07x |
| down_proj | 12288->4096 | 128 | 135.86 | 348.84 | 157.82 | 0.39x | 0.86x | 2.21x |
| down_proj | 12288->4096 | 512 | 328.87 | 640.84 | 308.06 | 0.51x | 1.07x | 2.08x |
| gate_up_proj | 4096->24576 | 1 | 216.25 | 136.86 | 96.05 | 1.58x | 2.25x | 1.42x |
| gate_up_proj | 4096->24576 | 32 | 237.97 | 181.95 | 73.26 | 1.31x | 3.25x | 2.48x |
| gate_up_proj | 4096->24576 | 128 | 274.28 | 306.73 | 147.00 | 0.89x | 1.87x | 2.09x |
| gate_up_proj | 4096->24576 | 512 | 712.94 | 996.16 | 462.80 | 0.72x | 1.54x | 2.15x |
| kv_proj | 4096->2048 | 1 | 20.53 | 136.76 | 13.51 | 0.15x | 1.52x | 10.13x |
| kv_proj | 4096->2048 | 32 | 21.30 | 138.13 | 34.54 | 0.15x | 0.62x | 4.00x |
| kv_proj | 4096->2048 | 128 | 23.14 | 137.12 | 42.55 | 0.17x | 0.54x | 3.22x |
| kv_proj | 4096->2048 | 512 | 46.91 | 154.65 | 60.13 | 0.30x | 0.78x | 2.57x |
| o_proj | 4096->4096 | 1 | 40.02 | 137.33 | 18.54 | 0.29x | 2.16x | 7.41x |
| o_proj | 4096->4096 | 32 | 40.14 | 136.81 | 41.50 | 0.29x | 0.97x | 3.30x |
| o_proj | 4096->4096 | 128 | 42.03 | 137.59 | 51.93 | 0.31x | 0.81x | 2.65x |
| o_proj | 4096->4096 | 512 | 117.49 | 219.75 | 86.65 | 0.53x | 1.36x | 2.54x |
| q_proj | 4096->4096 | 1 | 40.28 | 137.44 | 18.49 | 0.29x | 2.18x | 7.43x |
| q_proj | 4096->4096 | 32 | 40.14 | 136.26 | 41.31 | 0.29x | 0.97x | 3.30x |
| q_proj | 4096->4096 | 128 | 42.23 | 138.17 | 51.89 | 0.31x | 0.81x | 2.66x |
| q_proj | 4096->4096 | 512 | 117.09 | 219.85 | 86.37 | 0.53x | 1.36x | 2.55x |


## 3. Sub-kernel breakdown (us)


### Qwen3-8B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 116.36 | 24.87 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 112.35 | 144.81 | 43.63 | 0.78x | 2.57x | 3.32x |
| down_proj | 1 | sparse_gemm | 112.42 | 68.01 | 19.36 | 1.65x | 5.81x | 3.51x |
| down_proj | 1 | fused_dense_sparse | 112.43 | 157.23 | 45.43 | 0.72x | 2.47x | 3.46x |
| down_proj | 32 | activation_quant |  -  | 163.32 | 24.63 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 123.28 | 142.03 | 118.32 | 0.87x | 1.04x | 1.20x |
| down_proj | 32 | sparse_gemm | 123.32 | 69.05 | 19.19 | 1.79x | 6.43x | 3.60x |
| down_proj | 32 | fused_dense_sparse | 123.25 | 154.17 | 128.88 | 0.80x | 0.96x | 1.20x |
| down_proj | 128 | activation_quant |  -  | 165.16 | 25.07 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 135.89 | 168.91 | 144.39 | 0.80x | 0.94x | 1.17x |
| down_proj | 128 | sparse_gemm | 136.07 | 69.57 | 22.14 | 1.96x | 6.15x | 3.14x |
| down_proj | 128 | fused_dense_sparse | 135.86 | 182.41 | 132.41 | 0.74x | 1.03x | 1.38x |
| down_proj | 512 | activation_quant |  -  | 165.79 | 50.88 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 324.59 | 429.66 | 331.68 | 0.76x | 0.98x | 1.30x |
| down_proj | 512 | sparse_gemm | 325.89 | 69.79 | 36.30 | 4.67x | 8.98x | 1.92x |
| down_proj | 512 | fused_dense_sparse | 324.98 | 475.93 | 259.02 | 0.68x | 1.25x | 1.84x |
| gate_up_proj | 1 | activation_quant |  -  | 45.31 | 16.56 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 216.27 | 78.33 | 73.43 | 2.76x | 2.95x | 1.07x |
| gate_up_proj | 1 | sparse_gemm | 216.22 | 68.81 | 19.32 | 3.14x | 11.19x | 3.56x |
| gate_up_proj | 1 | fused_dense_sparse | 216.26 | 87.18 | 77.93 | 2.48x | 2.78x | 1.12x |
| gate_up_proj | 32 | activation_quant |  -  | 57.61 | 16.52 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 237.96 | 111.13 | 64.52 | 2.14x | 3.69x | 1.72x |
| gate_up_proj | 32 | sparse_gemm | 238.04 | 69.10 | 19.28 | 3.44x | 12.35x | 3.58x |
| gate_up_proj | 32 | fused_dense_sparse | 238.01 | 124.19 | 61.49 | 1.92x | 3.87x | 2.02x |
| gate_up_proj | 128 | activation_quant |  -  | 55.38 | 16.58 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 274.68 | 226.06 | 146.26 | 1.22x | 1.88x | 1.55x |
| gate_up_proj | 128 | sparse_gemm | 274.48 | 69.34 | 36.87 | 3.96x | 7.44x | 1.88x |
| gate_up_proj | 128 | fused_dense_sparse | 273.93 | 249.79 | 133.34 | 1.10x | 2.05x | 1.87x |
| gate_up_proj | 512 | activation_quant |  -  | 55.75 | 16.57 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 700.19 | 868.68 | 457.27 | 0.81x | 1.53x | 1.90x |
| gate_up_proj | 512 | sparse_gemm | 727.47 | 70.01 | 106.52 | 10.39x | 6.83x | 0.66x |
| gate_up_proj | 512 | fused_dense_sparse | 705.65 | 946.32 | 448.70 | 0.75x | 1.57x | 2.11x |
| kv_proj | 1 | activation_quant |  -  | 45.20 | 16.23 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 20.65 | 70.66 | 13.28 | 0.29x | 1.56x | 5.32x |
| kv_proj | 1 | sparse_gemm | 20.58 | 69.64 | 19.54 | 0.30x | 1.05x | 3.56x |
| kv_proj | 1 | fused_dense_sparse | 20.44 | 82.53 | 14.51 | 0.25x | 1.41x | 5.69x |
| kv_proj | 32 | activation_quant |  -  | 55.12 | 16.48 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 21.33 | 70.78 | 31.75 | 0.30x | 0.67x | 2.23x |
| kv_proj | 32 | sparse_gemm | 21.30 | 69.77 | 19.28 | 0.31x | 1.10x | 3.62x |
| kv_proj | 32 | fused_dense_sparse | 21.28 | 83.10 | 23.34 | 0.26x | 0.91x | 3.56x |
| kv_proj | 128 | activation_quant |  -  | 55.24 | 16.58 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 23.17 | 70.81 | 46.51 | 0.33x | 0.50x | 1.52x |
| kv_proj | 128 | sparse_gemm | 23.04 | 69.62 | 19.45 | 0.33x | 1.18x | 3.58x |
| kv_proj | 128 | fused_dense_sparse | 23.08 | 82.06 | 30.64 | 0.28x | 0.75x | 2.68x |
| kv_proj | 512 | activation_quant |  -  | 55.38 | 16.30 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 47.05 | 86.65 | 57.95 | 0.54x | 0.81x | 1.50x |
| kv_proj | 512 | sparse_gemm | 47.31 | 69.23 | 19.45 | 0.68x | 2.43x | 3.56x |
| kv_proj | 512 | fused_dense_sparse | 47.16 | 95.90 | 47.96 | 0.49x | 0.98x | 2.00x |
| o_proj | 1 | activation_quant |  -  | 43.97 | 16.23 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 39.90 | 70.22 | 15.82 | 0.57x | 2.52x | 4.44x |
| o_proj | 1 | sparse_gemm | 39.87 | 68.77 | 19.09 | 0.58x | 2.09x | 3.60x |
| o_proj | 1 | fused_dense_sparse | 39.84 | 82.85 | 16.54 | 0.48x | 2.41x | 5.01x |
| o_proj | 32 | activation_quant |  -  | 55.14 | 16.44 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 40.11 | 70.73 | 37.85 | 0.57x | 1.06x | 1.87x |
| o_proj | 32 | sparse_gemm | 40.14 | 69.39 | 19.48 | 0.58x | 2.06x | 3.56x |
| o_proj | 32 | fused_dense_sparse | 40.14 | 82.60 | 30.10 | 0.49x | 1.33x | 2.74x |
| o_proj | 128 | activation_quant |  -  | 55.39 | 16.41 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 42.02 | 71.13 | 47.44 | 0.59x | 0.89x | 1.50x |
| o_proj | 128 | sparse_gemm | 41.93 | 69.57 | 19.35 | 0.60x | 2.17x | 3.59x |
| o_proj | 128 | fused_dense_sparse | 41.96 | 83.43 | 40.28 | 0.50x | 1.04x | 2.07x |
| o_proj | 512 | activation_quant |  -  | 55.48 | 16.07 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 117.04 | 144.32 | 80.12 | 0.81x | 1.46x | 1.80x |
| o_proj | 512 | sparse_gemm | 117.53 | 68.43 | 24.00 | 1.72x | 4.90x | 2.85x |
| o_proj | 512 | fused_dense_sparse | 117.39 | 161.89 | 72.76 | 0.73x | 1.61x | 2.23x |
| q_proj | 1 | activation_quant |  -  | 46.50 | 16.33 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 40.74 | 71.03 | 15.75 | 0.57x | 2.59x | 4.51x |
| q_proj | 1 | sparse_gemm | 40.25 | 69.96 | 19.14 | 0.58x | 2.10x | 3.66x |
| q_proj | 1 | fused_dense_sparse | 40.35 | 82.44 | 16.45 | 0.49x | 2.45x | 5.01x |
| q_proj | 32 | activation_quant |  -  | 54.84 | 16.38 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 39.99 | 70.63 | 37.58 | 0.57x | 1.06x | 1.88x |
| q_proj | 32 | sparse_gemm | 40.19 | 69.53 | 19.67 | 0.58x | 2.04x | 3.53x |
| q_proj | 32 | fused_dense_sparse | 40.12 | 81.84 | 29.86 | 0.49x | 1.34x | 2.74x |
| q_proj | 128 | activation_quant |  -  | 55.04 | 16.53 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 42.25 | 70.75 | 47.31 | 0.60x | 0.89x | 1.50x |
| q_proj | 128 | sparse_gemm | 42.27 | 69.37 | 19.46 | 0.61x | 2.17x | 3.57x |
| q_proj | 128 | fused_dense_sparse | 42.27 | 83.04 | 40.06 | 0.51x | 1.06x | 2.07x |
| q_proj | 512 | activation_quant |  -  | 55.43 | 16.41 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 115.32 | 144.45 | 80.18 | 0.80x | 1.44x | 1.80x |
| q_proj | 512 | sparse_gemm | 116.36 | 70.17 | 23.88 | 1.66x | 4.87x | 2.94x |
| q_proj | 512 | fused_dense_sparse | 115.86 | 161.36 | 72.82 | 0.72x | 1.59x | 2.22x |


## 4. End-to-end speedup (CUDA over Triton)

Rows: projection. Cells: `triton_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-8B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 4096->4096 | **7.43x** | **3.30x** | **2.66x** | **2.55x** |
| kv_proj | 4096->2048 | **10.13x** | **4.00x** | **3.22x** | **2.57x** |
| o_proj | 4096->4096 | **7.41x** | **3.30x** | **2.65x** | **2.54x** |
| gate_up_proj | 4096->24576 | **1.42x** | **2.48x** | **2.09x** | **2.15x** |
| down_proj | 12288->4096 | **5.07x** | **2.07x** | **2.21x** | **2.08x** |


## 5. CUDA end-to-end bottleneck hint

For each shape, compare CUDA `activation_quant` against CUDA `fused_dense_sparse`. A larger `quant_share` means launch/prologue dominates; a larger `fused_share` means the main CUDA matmul kernel dominates.

| model | proj | T | shape | quant_us | fused_us | quant_share | fused_share | likely_bottleneck |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-8B | q_proj | 1 | 4096->4096 | 16.33 | 16.45 | 49.8% | 50.2% | quant/prologue dominated |
| Qwen3-8B | q_proj | 32 | 4096->4096 | 16.38 | 29.86 | 35.4% | 64.6% | quant/prologue dominated |
| Qwen3-8B | q_proj | 128 | 4096->4096 | 16.53 | 40.06 | 29.2% | 70.8% | mixed |
| Qwen3-8B | q_proj | 512 | 4096->4096 | 16.41 | 72.82 | 18.4% | 81.6% | main fused kernel dominated |
| Qwen3-8B | kv_proj | 1 | 4096->2048 | 16.23 | 14.51 | 52.8% | 47.2% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 32 | 4096->2048 | 16.48 | 23.34 | 41.4% | 58.6% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 128 | 4096->2048 | 16.58 | 30.64 | 35.1% | 64.9% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 512 | 4096->2048 | 16.30 | 47.96 | 25.4% | 74.6% | mixed |
| Qwen3-8B | o_proj | 1 | 4096->4096 | 16.23 | 16.54 | 49.5% | 50.5% | quant/prologue dominated |
| Qwen3-8B | o_proj | 32 | 4096->4096 | 16.44 | 30.10 | 35.3% | 64.7% | quant/prologue dominated |
| Qwen3-8B | o_proj | 128 | 4096->4096 | 16.41 | 40.28 | 28.9% | 71.1% | mixed |
| Qwen3-8B | o_proj | 512 | 4096->4096 | 16.07 | 72.76 | 18.1% | 81.9% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 1 | 4096->24576 | 16.56 | 77.93 | 17.5% | 82.5% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 32 | 4096->24576 | 16.52 | 61.49 | 21.2% | 78.8% | mixed |
| Qwen3-8B | gate_up_proj | 128 | 4096->24576 | 16.58 | 133.34 | 11.1% | 88.9% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 512 | 4096->24576 | 16.57 | 448.70 | 3.6% | 96.4% | main fused kernel dominated |
| Qwen3-8B | down_proj | 1 | 12288->4096 | 24.87 | 45.43 | 35.4% | 64.6% | quant/prologue dominated |
| Qwen3-8B | down_proj | 32 | 12288->4096 | 24.63 | 128.88 | 16.0% | 84.0% | main fused kernel dominated |
| Qwen3-8B | down_proj | 128 | 12288->4096 | 25.07 | 132.41 | 15.9% | 84.1% | main fused kernel dominated |
| Qwen3-8B | down_proj | 512 | 12288->4096 | 50.88 | 259.02 | 16.4% | 83.6% | main fused kernel dominated |
