# Qwen3 multi-scale kernel benchmark

- Timestamp: `20260429_220448`
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
| proj | shape | T=32 | T=128 |
|---|---|---:|---:|
| q_proj | 4096->4096 | **0.46x** | **0.58x** |
| kv_proj | 4096->2048 | **0.38x** | **0.45x** |
| o_proj | 4096->4096 | **0.48x** | **0.59x** |
| gate_up_proj | 4096->24576 | **3.03x** | **1.79x** |
| down_proj | 12288->4096 | **0.79x** | **0.85x** |


## 2. End-to-end raw latencies (us)


### Qwen3-8B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 12288->4096 | 32 | 120.97 | 316.79 | 153.15 | 0.38x | 0.79x | 2.07x |
| down_proj | 12288->4096 | 128 | 134.02 | 347.78 | 157.20 | 0.39x | 0.85x | 2.21x |
| gate_up_proj | 4096->24576 | 32 | 222.21 | 181.56 | 73.24 | 1.22x | 3.03x | 2.48x |
| gate_up_proj | 4096->24576 | 128 | 263.07 | 306.66 | 146.94 | 0.86x | 1.79x | 2.09x |
| kv_proj | 4096->2048 | 32 | 13.17 | 135.26 | 34.40 | 0.10x | 0.38x | 3.93x |
| kv_proj | 4096->2048 | 128 | 19.05 | 134.53 | 42.36 | 0.14x | 0.45x | 3.18x |
| o_proj | 4096->4096 | 32 | 19.85 | 134.43 | 41.39 | 0.15x | 0.48x | 3.25x |
| o_proj | 4096->4096 | 128 | 30.34 | 135.10 | 51.67 | 0.22x | 0.59x | 2.61x |
| q_proj | 4096->4096 | 32 | 23.27 | 200.64 | 50.49 | 0.12x | 0.46x | 3.97x |
| q_proj | 4096->4096 | 128 | 30.20 | 134.12 | 51.65 | 0.23x | 0.58x | 2.60x |


## 3. Sub-kernel breakdown (us)


### Qwen3-8B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 32 | activation_quant |  -  | 164.31 | 24.74 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 120.98 | 141.80 | 117.85 | 0.85x | 1.03x | 1.20x |
| down_proj | 32 | sparse_gemm | 120.95 | 68.22 | 21.17 | 1.77x | 5.71x | 3.22x |
| down_proj | 32 | fused_dense_sparse | 120.99 | 154.12 | 128.34 | 0.78x | 0.94x | 1.20x |
| down_proj | 128 | activation_quant |  -  | 164.86 | 25.32 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 134.01 | 168.66 | 143.97 | 0.79x | 0.93x | 1.17x |
| down_proj | 128 | sparse_gemm | 133.93 | 67.82 | 22.14 | 1.97x | 6.05x | 3.06x |
| down_proj | 128 | fused_dense_sparse | 133.95 | 181.74 | 131.94 | 0.74x | 1.02x | 1.38x |
| gate_up_proj | 32 | activation_quant |  -  | 54.93 | 16.89 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 222.24 | 110.85 | 64.72 | 2.00x | 3.43x | 1.71x |
| gate_up_proj | 32 | sparse_gemm | 222.34 | 67.68 | 21.11 | 3.28x | 10.53x | 3.21x |
| gate_up_proj | 32 | fused_dense_sparse | 222.34 | 124.00 | 60.89 | 1.79x | 3.65x | 2.04x |
| gate_up_proj | 128 | activation_quant |  -  | 55.38 | 17.09 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 262.22 | 220.89 | 146.60 | 1.19x | 1.79x | 1.51x |
| gate_up_proj | 128 | sparse_gemm | 263.28 | 68.28 | 36.95 | 3.86x | 7.13x | 1.85x |
| gate_up_proj | 128 | fused_dense_sparse | 262.47 | 249.92 | 133.58 | 1.05x | 1.96x | 1.87x |
| kv_proj | 32 | activation_quant |  -  | 54.88 | 16.80 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 13.24 | 68.27 | 31.75 | 0.19x | 0.42x | 2.15x |
| kv_proj | 32 | sparse_gemm | 13.21 | 67.74 | 21.42 | 0.20x | 0.62x | 3.16x |
| kv_proj | 32 | fused_dense_sparse | 13.15 | 82.08 | 22.94 | 0.16x | 0.57x | 3.58x |
| kv_proj | 128 | activation_quant |  -  | 55.13 | 16.75 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 19.04 | 69.84 | 46.24 | 0.27x | 0.41x | 1.51x |
| kv_proj | 128 | sparse_gemm | 19.02 | 68.62 | 21.28 | 0.28x | 0.89x | 3.22x |
| kv_proj | 128 | fused_dense_sparse | 19.04 | 81.09 | 30.56 | 0.23x | 0.62x | 2.65x |
| o_proj | 32 | activation_quant |  -  | 54.90 | 16.80 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 19.86 | 69.26 | 37.72 | 0.29x | 0.53x | 1.84x |
| o_proj | 32 | sparse_gemm | 19.84 | 67.93 | 21.08 | 0.29x | 0.94x | 3.22x |
| o_proj | 32 | fused_dense_sparse | 19.83 | 83.17 | 29.95 | 0.24x | 0.66x | 2.78x |
| o_proj | 128 | activation_quant |  -  | 55.14 | 16.64 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 30.11 | 69.67 | 47.33 | 0.43x | 0.64x | 1.47x |
| o_proj | 128 | sparse_gemm | 30.18 | 67.90 | 21.11 | 0.44x | 1.43x | 3.22x |
| o_proj | 128 | fused_dense_sparse | 30.15 | 83.16 | 40.17 | 0.36x | 0.75x | 2.07x |
| q_proj | 32 | activation_quant |  -  | 65.58 | 25.76 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 21.41 | 104.44 | 37.67 | 0.21x | 0.57x | 2.77x |
| q_proj | 32 | sparse_gemm | 23.53 | 104.31 | 37.20 | 0.23x | 0.63x | 2.80x |
| q_proj | 32 | fused_dense_sparse | 23.80 | 122.58 | 30.00 | 0.19x | 0.79x | 4.09x |
| q_proj | 128 | activation_quant |  -  | 66.18 | 26.44 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 30.17 | 69.65 | 47.35 | 0.43x | 0.64x | 1.47x |
| q_proj | 128 | sparse_gemm | 30.13 | 68.81 | 21.33 | 0.44x | 1.41x | 3.23x |
| q_proj | 128 | fused_dense_sparse | 30.17 | 81.82 | 40.13 | 0.37x | 0.75x | 2.04x |


## 4. End-to-end speedup (CUDA over Triton)

Rows: projection. Cells: `triton_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-8B
| proj | shape | T=32 | T=128 |
|---|---|---:|---:|
| q_proj | 4096->4096 | **3.97x** | **2.60x** |
| kv_proj | 4096->2048 | **3.93x** | **3.18x** |
| o_proj | 4096->4096 | **3.25x** | **2.61x** |
| gate_up_proj | 4096->24576 | **2.48x** | **2.09x** |
| down_proj | 12288->4096 | **2.07x** | **2.21x** |


## 5. CUDA end-to-end bottleneck hint

For each shape, compare CUDA `activation_quant` against CUDA `fused_dense_sparse`. A larger `quant_share` means launch/prologue dominates; a larger `fused_share` means the main CUDA matmul kernel dominates.

| model | proj | T | shape | quant_us | fused_us | quant_share | fused_share | likely_bottleneck |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-8B | q_proj | 32 | 4096->4096 | 25.76 | 30.00 | 46.2% | 53.8% | quant/prologue dominated |
| Qwen3-8B | q_proj | 128 | 4096->4096 | 26.44 | 40.13 | 39.7% | 60.3% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 32 | 4096->2048 | 16.80 | 22.94 | 42.3% | 57.7% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 128 | 4096->2048 | 16.75 | 30.56 | 35.4% | 64.6% | quant/prologue dominated |
| Qwen3-8B | o_proj | 32 | 4096->4096 | 16.80 | 29.95 | 35.9% | 64.1% | quant/prologue dominated |
| Qwen3-8B | o_proj | 128 | 4096->4096 | 16.64 | 40.17 | 29.3% | 70.7% | mixed |
| Qwen3-8B | gate_up_proj | 32 | 4096->24576 | 16.89 | 60.89 | 21.7% | 78.3% | mixed |
| Qwen3-8B | gate_up_proj | 128 | 4096->24576 | 17.09 | 133.58 | 11.3% | 88.7% | main fused kernel dominated |
| Qwen3-8B | down_proj | 32 | 12288->4096 | 24.74 | 128.34 | 16.2% | 83.8% | main fused kernel dominated |
| Qwen3-8B | down_proj | 128 | 12288->4096 | 25.32 | 131.94 | 16.1% | 83.9% | main fused kernel dominated |
