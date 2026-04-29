# Qwen3 multi-scale kernel benchmark

- Timestamp: `20260429_192025`
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
| proj | shape | T=128 |
|---|---|---:|
| q_proj | 4096->4096 | **0.57x** |
| kv_proj | 4096->2048 | **0.45x** |
| o_proj | 4096->4096 | **0.58x** |
| gate_up_proj | 4096->24576 | **1.75x** |
| down_proj | 12288->4096 | **0.83x** |


## 2. End-to-end raw latencies (us)


### Qwen3-8B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 12288->4096 | 128 | 134.03 | 353.47 | 160.52 | 0.38x | 0.83x | 2.20x |
| gate_up_proj | 4096->24576 | 128 | 260.91 | 309.76 | 149.22 | 0.84x | 1.75x | 2.08x |
| kv_proj | 4096->2048 | 128 | 19.37 | 134.65 | 42.99 | 0.14x | 0.45x | 3.13x |
| o_proj | 4096->4096 | 128 | 30.44 | 134.84 | 52.83 | 0.23x | 0.58x | 2.55x |
| q_proj | 4096->4096 | 128 | 30.23 | 135.76 | 52.66 | 0.22x | 0.57x | 2.58x |


## 3. Sub-kernel breakdown (us)


### Qwen3-8B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 128 | activation_quant |  -  | 167.64 | 25.78 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 134.05 | 171.20 | 145.69 | 0.78x | 0.92x | 1.18x |
| down_proj | 128 | sparse_gemm | 133.99 | 67.98 | 22.46 | 1.97x | 5.96x | 3.03x |
| down_proj | 128 | fused_dense_sparse | 133.94 | 184.25 | 134.32 | 0.73x | 1.00x | 1.37x |
| gate_up_proj | 128 | activation_quant |  -  | 55.92 | 16.62 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 260.69 | 222.32 | 148.42 | 1.17x | 1.76x | 1.50x |
| gate_up_proj | 128 | sparse_gemm | 260.96 | 68.11 | 37.28 | 3.83x | 7.00x | 1.83x |
| gate_up_proj | 128 | fused_dense_sparse | 260.83 | 252.25 | 135.95 | 1.03x | 1.92x | 1.86x |
| kv_proj | 128 | activation_quant |  -  | 55.86 | 16.74 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 19.42 | 69.88 | 47.12 | 0.28x | 0.41x | 1.48x |
| kv_proj | 128 | sparse_gemm | 19.39 | 69.15 | 19.74 | 0.28x | 0.98x | 3.50x |
| kv_proj | 128 | fused_dense_sparse | 19.39 | 82.59 | 31.27 | 0.23x | 0.62x | 2.64x |
| o_proj | 128 | activation_quant |  -  | 55.88 | 16.56 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 30.33 | 69.35 | 48.13 | 0.44x | 0.63x | 1.44x |
| o_proj | 128 | sparse_gemm | 30.43 | 68.42 | 19.67 | 0.44x | 1.55x | 3.48x |
| o_proj | 128 | fused_dense_sparse | 30.46 | 82.09 | 41.00 | 0.37x | 0.74x | 2.00x |
| q_proj | 128 | activation_quant |  -  | 60.46 | 16.74 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 32.11 | 70.61 | 47.90 | 0.45x | 0.67x | 1.47x |
| q_proj | 128 | sparse_gemm | 30.09 | 68.71 | 19.90 | 0.44x | 1.51x | 3.45x |
| q_proj | 128 | fused_dense_sparse | 30.14 | 83.62 | 41.01 | 0.36x | 0.74x | 2.04x |


## 4. End-to-end speedup (CUDA over Triton)

Rows: projection. Cells: `triton_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-8B
| proj | shape | T=128 |
|---|---|---:|
| q_proj | 4096->4096 | **2.58x** |
| kv_proj | 4096->2048 | **3.13x** |
| o_proj | 4096->4096 | **2.55x** |
| gate_up_proj | 4096->24576 | **2.08x** |
| down_proj | 12288->4096 | **2.20x** |


## 5. CUDA end-to-end bottleneck hint

For each shape, compare CUDA `activation_quant` against CUDA `fused_dense_sparse`. A larger `quant_share` means launch/prologue dominates; a larger `fused_share` means the main CUDA matmul kernel dominates.

| model | proj | T | shape | quant_us | fused_us | quant_share | fused_share | likely_bottleneck |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-8B | q_proj | 128 | 4096->4096 | 16.74 | 41.01 | 29.0% | 71.0% | mixed |
| Qwen3-8B | kv_proj | 128 | 4096->2048 | 16.74 | 31.27 | 34.9% | 65.1% | mixed |
| Qwen3-8B | o_proj | 128 | 4096->4096 | 16.56 | 41.00 | 28.8% | 71.2% | mixed |
| Qwen3-8B | gate_up_proj | 128 | 4096->24576 | 16.62 | 135.95 | 10.9% | 89.1% | main fused kernel dominated |
| Qwen3-8B | down_proj | 128 | 12288->4096 | 25.78 | 134.32 | 16.1% | 83.9% | main fused kernel dominated |
