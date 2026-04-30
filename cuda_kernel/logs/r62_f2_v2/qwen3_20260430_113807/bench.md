# Qwen3 multi-scale kernel benchmark

- Timestamp: `20260430_113807`
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
| q_proj | 4096->4096 | **2.19x** | **1.14x** | **0.92x** | **1.33x** |
| kv_proj | 4096->2048 | **1.52x** | **0.60x** | **0.65x** | **0.78x** |
| o_proj | 4096->4096 | **2.15x** | **1.13x** | **0.91x** | **1.34x** |
| gate_up_proj | 4096->24576 | **2.31x** | **3.25x** | **1.86x** | **1.54x** |
| down_proj | 12288->4096 | **2.12x** | **1.72x** | **1.28x** | **1.04x** |


## 2. End-to-end raw latencies (us)


### Qwen3-8B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 12288->4096 | 1 | 112.52 | 272.05 | 52.99 | 0.41x | 2.12x | 5.13x |
| down_proj | 12288->4096 | 32 | 123.48 | 318.91 | 71.71 | 0.39x | 1.72x | 4.45x |
| down_proj | 12288->4096 | 128 | 135.98 | 349.93 | 106.36 | 0.39x | 1.28x | 3.29x |
| down_proj | 12288->4096 | 512 | 322.48 | 647.57 | 309.94 | 0.50x | 1.04x | 2.09x |
| gate_up_proj | 4096->24576 | 1 | 216.34 | 134.78 | 93.54 | 1.61x | 2.31x | 1.44x |
| gate_up_proj | 4096->24576 | 32 | 238.26 | 182.23 | 73.26 | 1.31x | 3.25x | 2.49x |
| gate_up_proj | 4096->24576 | 128 | 273.69 | 307.99 | 147.50 | 0.89x | 1.86x | 2.09x |
| gate_up_proj | 4096->24576 | 512 | 706.47 | 987.69 | 458.44 | 0.72x | 1.54x | 2.15x |
| kv_proj | 4096->2048 | 1 | 20.55 | 133.37 | 13.51 | 0.15x | 1.52x | 9.87x |
| kv_proj | 4096->2048 | 32 | 21.23 | 135.50 | 35.50 | 0.16x | 0.60x | 3.82x |
| kv_proj | 4096->2048 | 128 | 23.12 | 133.88 | 35.45 | 0.17x | 0.65x | 3.78x |
| kv_proj | 4096->2048 | 512 | 47.29 | 155.49 | 60.48 | 0.30x | 0.78x | 2.57x |
| o_proj | 4096->4096 | 1 | 39.92 | 133.77 | 18.52 | 0.30x | 2.15x | 7.22x |
| o_proj | 4096->4096 | 32 | 40.11 | 134.49 | 35.48 | 0.30x | 1.13x | 3.79x |
| o_proj | 4096->4096 | 128 | 42.07 | 134.86 | 46.44 | 0.31x | 0.91x | 2.90x |
| o_proj | 4096->4096 | 512 | 115.65 | 219.92 | 86.45 | 0.53x | 1.34x | 2.54x |
| q_proj | 4096->4096 | 1 | 40.47 | 133.56 | 18.48 | 0.30x | 2.19x | 7.23x |
| q_proj | 4096->4096 | 32 | 40.31 | 135.12 | 35.42 | 0.30x | 1.14x | 3.81x |
| q_proj | 4096->4096 | 128 | 42.33 | 134.23 | 46.18 | 0.32x | 0.92x | 2.91x |
| q_proj | 4096->4096 | 512 | 115.08 | 219.96 | 86.37 | 0.52x | 1.33x | 2.55x |


## 3. Sub-kernel breakdown (us)


### Qwen3-8B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 115.72 | 24.74 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 112.57 | 145.29 | 43.93 | 0.77x | 2.56x | 3.31x |
| down_proj | 1 | sparse_gemm | 112.53 | 68.12 | 20.20 | 1.65x | 5.57x | 3.37x |
| down_proj | 1 | fused_dense_sparse | 112.55 | 158.28 | 45.69 | 0.71x | 2.46x | 3.46x |
| down_proj | 32 | activation_quant |  -  | 164.25 | 24.73 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 123.33 | 142.67 | 118.75 | 0.86x | 1.04x | 1.20x |
| down_proj | 32 | sparse_gemm | 123.45 | 67.68 | 20.02 | 1.82x | 6.17x | 3.38x |
| down_proj | 32 | fused_dense_sparse | 123.39 | 155.06 | 46.54 | 0.80x | 2.65x | 3.33x |
| down_proj | 128 | activation_quant |  -  | 165.82 | 25.21 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 135.93 | 169.60 | 144.12 | 0.80x | 0.94x | 1.18x |
| down_proj | 128 | sparse_gemm | 136.06 | 67.97 | 22.30 | 2.00x | 6.10x | 3.05x |
| down_proj | 128 | fused_dense_sparse | 135.97 | 183.15 | 80.25 | 0.74x | 1.69x | 2.28x |
| down_proj | 512 | activation_quant |  -  | 165.98 | 51.05 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 321.26 | 431.80 | 333.19 | 0.74x | 0.96x | 1.30x |
| down_proj | 512 | sparse_gemm | 325.30 | 68.56 | 36.53 | 4.75x | 8.91x | 1.88x |
| down_proj | 512 | fused_dense_sparse | 321.27 | 480.91 | 260.16 | 0.67x | 1.23x | 1.85x |
| gate_up_proj | 1 | activation_quant |  -  | 44.34 | 16.64 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 216.32 | 78.69 | 72.82 | 2.75x | 2.97x | 1.08x |
| gate_up_proj | 1 | sparse_gemm | 216.31 | 68.44 | 20.01 | 3.16x | 10.81x | 3.42x |
| gate_up_proj | 1 | fused_dense_sparse | 216.30 | 87.10 | 77.12 | 2.48x | 2.80x | 1.13x |
| gate_up_proj | 32 | activation_quant |  -  | 55.50 | 16.56 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 238.18 | 111.37 | 64.73 | 2.14x | 3.68x | 1.72x |
| gate_up_proj | 32 | sparse_gemm | 238.28 | 68.76 | 19.98 | 3.47x | 11.93x | 3.44x |
| gate_up_proj | 32 | fused_dense_sparse | 238.20 | 124.65 | 60.77 | 1.91x | 3.92x | 2.05x |
| gate_up_proj | 128 | activation_quant |  -  | 55.59 | 16.60 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 273.56 | 221.15 | 147.78 | 1.24x | 1.85x | 1.50x |
| gate_up_proj | 128 | sparse_gemm | 273.99 | 68.62 | 37.19 | 3.99x | 7.37x | 1.85x |
| gate_up_proj | 128 | fused_dense_sparse | 273.43 | 250.85 | 133.68 | 1.09x | 2.05x | 1.88x |
| gate_up_proj | 512 | activation_quant |  -  | 55.96 | 16.58 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 689.77 | 862.64 | 451.15 | 0.80x | 1.53x | 1.91x |
| gate_up_proj | 512 | sparse_gemm | 693.71 | 68.67 | 107.33 | 10.10x | 6.46x | 0.64x |
| gate_up_proj | 512 | fused_dense_sparse | 693.90 | 936.04 | 443.17 | 0.74x | 1.57x | 2.11x |
| kv_proj | 1 | activation_quant |  -  | 43.05 | 16.42 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 20.57 | 69.12 | 12.99 | 0.30x | 1.58x | 5.32x |
| kv_proj | 1 | sparse_gemm | 20.67 | 67.49 | 20.04 | 0.31x | 1.03x | 3.37x |
| kv_proj | 1 | fused_dense_sparse | 20.46 | 80.40 | 14.26 | 0.25x | 1.43x | 5.64x |
| kv_proj | 32 | activation_quant |  -  | 55.12 | 16.31 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 21.25 | 68.52 | 31.91 | 0.31x | 0.67x | 2.15x |
| kv_proj | 32 | sparse_gemm | 21.24 | 67.47 | 19.98 | 0.31x | 1.06x | 3.38x |
| kv_proj | 32 | fused_dense_sparse | 21.23 | 81.03 | 23.77 | 0.26x | 0.89x | 3.41x |
| kv_proj | 128 | activation_quant |  -  | 55.26 | 16.42 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 23.22 | 68.99 | 46.54 | 0.34x | 0.50x | 1.48x |
| kv_proj | 128 | sparse_gemm | 23.29 | 68.48 | 19.83 | 0.34x | 1.17x | 3.45x |
| kv_proj | 128 | fused_dense_sparse | 23.20 | 81.99 | 22.05 | 0.28x | 1.05x | 3.72x |
| kv_proj | 512 | activation_quant |  -  | 55.40 | 16.37 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 47.05 | 86.68 | 57.98 | 0.54x | 0.81x | 1.50x |
| kv_proj | 512 | sparse_gemm | 47.19 | 67.95 | 20.29 | 0.69x | 2.33x | 3.35x |
| kv_proj | 512 | fused_dense_sparse | 47.20 | 96.37 | 48.23 | 0.49x | 0.98x | 2.00x |
| o_proj | 1 | activation_quant |  -  | 42.86 | 16.18 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 39.85 | 68.73 | 15.94 | 0.58x | 2.50x | 4.31x |
| o_proj | 1 | sparse_gemm | 39.86 | 67.58 | 19.77 | 0.59x | 2.02x | 3.42x |
| o_proj | 1 | fused_dense_sparse | 39.91 | 80.87 | 16.65 | 0.49x | 2.40x | 4.86x |
| o_proj | 32 | activation_quant |  -  | 55.46 | 16.30 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 40.14 | 68.89 | 37.85 | 0.58x | 1.06x | 1.82x |
| o_proj | 32 | sparse_gemm | 40.04 | 68.00 | 20.16 | 0.59x | 1.99x | 3.37x |
| o_proj | 32 | fused_dense_sparse | 40.06 | 80.87 | 22.53 | 0.50x | 1.78x | 3.59x |
| o_proj | 128 | activation_quant |  -  | 55.31 | 16.50 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 41.96 | 69.25 | 47.51 | 0.61x | 0.88x | 1.46x |
| o_proj | 128 | sparse_gemm | 42.08 | 67.75 | 20.17 | 0.62x | 2.09x | 3.36x |
| o_proj | 128 | fused_dense_sparse | 42.10 | 82.01 | 34.48 | 0.51x | 1.22x | 2.38x |
| o_proj | 512 | activation_quant |  -  | 55.42 | 16.38 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 114.99 | 144.55 | 80.64 | 0.80x | 1.43x | 1.79x |
| o_proj | 512 | sparse_gemm | 115.20 | 67.93 | 24.05 | 1.70x | 4.79x | 2.82x |
| o_proj | 512 | fused_dense_sparse | 115.16 | 162.34 | 72.52 | 0.71x | 1.59x | 2.24x |
| q_proj | 1 | activation_quant |  -  | 43.55 | 16.41 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 40.80 | 68.18 | 15.83 | 0.60x | 2.58x | 4.31x |
| q_proj | 1 | sparse_gemm | 40.55 | 67.41 | 19.89 | 0.60x | 2.04x | 3.39x |
| q_proj | 1 | fused_dense_sparse | 40.43 | 81.44 | 16.58 | 0.50x | 2.44x | 4.91x |
| q_proj | 32 | activation_quant |  -  | 54.90 | 16.49 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 40.21 | 69.29 | 37.73 | 0.58x | 1.07x | 1.84x |
| q_proj | 32 | sparse_gemm | 40.38 | 68.08 | 20.38 | 0.59x | 1.98x | 3.34x |
| q_proj | 32 | fused_dense_sparse | 40.31 | 81.42 | 22.50 | 0.50x | 1.79x | 3.62x |
| q_proj | 128 | activation_quant |  -  | 55.27 | 16.51 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 42.31 | 68.85 | 47.60 | 0.61x | 0.89x | 1.45x |
| q_proj | 128 | sparse_gemm | 42.35 | 68.26 | 20.38 | 0.62x | 2.08x | 3.35x |
| q_proj | 128 | fused_dense_sparse | 42.29 | 81.29 | 34.51 | 0.52x | 1.23x | 2.36x |
| q_proj | 512 | activation_quant |  -  | 55.42 | 16.22 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 114.26 | 145.03 | 80.58 | 0.79x | 1.42x | 1.80x |
| q_proj | 512 | sparse_gemm | 114.67 | 67.94 | 23.97 | 1.69x | 4.78x | 2.83x |
| q_proj | 512 | fused_dense_sparse | 114.48 | 162.37 | 72.53 | 0.71x | 1.58x | 2.24x |


## 4. End-to-end speedup (CUDA over Triton)

Rows: projection. Cells: `triton_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-8B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 4096->4096 | **7.23x** | **3.81x** | **2.91x** | **2.55x** |
| kv_proj | 4096->2048 | **9.87x** | **3.82x** | **3.78x** | **2.57x** |
| o_proj | 4096->4096 | **7.22x** | **3.79x** | **2.90x** | **2.54x** |
| gate_up_proj | 4096->24576 | **1.44x** | **2.49x** | **2.09x** | **2.15x** |
| down_proj | 12288->4096 | **5.13x** | **4.45x** | **3.29x** | **2.09x** |


## 5. CUDA end-to-end bottleneck hint

For each shape, compare CUDA `activation_quant` against CUDA `fused_dense_sparse`. A larger `quant_share` means launch/prologue dominates; a larger `fused_share` means the main CUDA matmul kernel dominates.

| model | proj | T | shape | quant_us | fused_us | quant_share | fused_share | likely_bottleneck |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-8B | q_proj | 1 | 4096->4096 | 16.41 | 16.58 | 49.8% | 50.2% | quant/prologue dominated |
| Qwen3-8B | q_proj | 32 | 4096->4096 | 16.49 | 22.50 | 42.3% | 57.7% | quant/prologue dominated |
| Qwen3-8B | q_proj | 128 | 4096->4096 | 16.51 | 34.51 | 32.4% | 67.6% | mixed |
| Qwen3-8B | q_proj | 512 | 4096->4096 | 16.22 | 72.53 | 18.3% | 81.7% | main fused kernel dominated |
| Qwen3-8B | kv_proj | 1 | 4096->2048 | 16.42 | 14.26 | 53.5% | 46.5% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 32 | 4096->2048 | 16.31 | 23.77 | 40.7% | 59.3% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 128 | 4096->2048 | 16.42 | 22.05 | 42.7% | 57.3% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 512 | 4096->2048 | 16.37 | 48.23 | 25.3% | 74.7% | mixed |
| Qwen3-8B | o_proj | 1 | 4096->4096 | 16.18 | 16.65 | 49.3% | 50.7% | quant/prologue dominated |
| Qwen3-8B | o_proj | 32 | 4096->4096 | 16.30 | 22.53 | 42.0% | 58.0% | quant/prologue dominated |
| Qwen3-8B | o_proj | 128 | 4096->4096 | 16.50 | 34.48 | 32.4% | 67.6% | mixed |
| Qwen3-8B | o_proj | 512 | 4096->4096 | 16.38 | 72.52 | 18.4% | 81.6% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 1 | 4096->24576 | 16.64 | 77.12 | 17.7% | 82.3% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 32 | 4096->24576 | 16.56 | 60.77 | 21.4% | 78.6% | mixed |
| Qwen3-8B | gate_up_proj | 128 | 4096->24576 | 16.60 | 133.68 | 11.0% | 89.0% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 512 | 4096->24576 | 16.58 | 443.17 | 3.6% | 96.4% | main fused kernel dominated |
| Qwen3-8B | down_proj | 1 | 12288->4096 | 24.74 | 45.69 | 35.1% | 64.9% | quant/prologue dominated |
| Qwen3-8B | down_proj | 32 | 12288->4096 | 24.73 | 46.54 | 34.7% | 65.3% | mixed |
| Qwen3-8B | down_proj | 128 | 12288->4096 | 25.21 | 80.25 | 23.9% | 76.1% | mixed |
| Qwen3-8B | down_proj | 512 | 12288->4096 | 51.05 | 260.16 | 16.4% | 83.6% | main fused kernel dominated |
