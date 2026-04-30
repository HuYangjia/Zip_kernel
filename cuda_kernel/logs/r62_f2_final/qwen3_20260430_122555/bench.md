# Qwen3 multi-scale kernel benchmark

- Timestamp: `20260430_122555`
- Device: `NVIDIA GeForce RTX 4090`
- PyTorch: `2.8.0+cu126`  Triton: `3.4.0`
- Baseline: cuBLAS FP16 matmul (`torch.matmul` on `fp16`)
- CUDA path: `activation_quant_cuda` + `fused_dense_sparse_cuda` (T=1 uses `fused_quant_gemv_cuda`, with automatic fallback on unsupported decode-group counts)
- Triton path: `quantize_activation_s4` + `fused_dense_sparse_gemm`
- hp_ratio: `0.05`  (block-sparse density)
- Stats: stable microbenchmark helper = 50 warmup, 100 inner, 3 repeats, min-of-means


## 1. End-to-end speedup vs FP16

Rows: projection. Cells: `fp16_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-0.6B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 1024->2048 | **0.97x** | **0.27x** | **0.24x** | **0.37x** |
| kv_proj | 1024->2048 | **1.02x** | **0.26x** | **0.23x** | **0.36x** |
| o_proj | 2048->1024 | **0.88x** | **0.19x** | **0.23x** | **0.33x** |
| gate_up_proj | 1024->6144 | **1.74x** | **0.51x** | **0.58x** | **0.89x** |
| down_proj | 3072->1024 | **0.95x** | **0.27x** | **0.32x** | **0.45x** |

### Qwen3-1.7B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 2048->2048 | **1.51x** | **0.34x** | **0.38x** | **0.58x** |
| kv_proj | 2048->2048 | **1.54x** | **0.34x** | **0.38x** | **0.59x** |
| o_proj | 2048->2048 | **1.53x** | **0.34x** | **0.38x** | **0.59x** |
| gate_up_proj | 2048->12288 | **2.35x** | **1.69x** | **1.54x** | **1.21x** |
| down_proj | 6144->2048 | **1.53x** | **0.80x** | **0.73x** | **0.87x** |

### Qwen3-4B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 2560->4096 | **1.96x** | **0.79x** | **0.79x** | **1.23x** |
| kv_proj | 2560->2048 | **1.42x** | **0.44x** | **0.45x** | **0.62x** |
| o_proj | 4096->2560 | **1.62x** | **0.76x** | **0.70x** | **0.98x** |
| gate_up_proj | 2560->19456 | **2.26x** | **2.52x** | **1.42x** | **1.45x** |
| down_proj | 9728->2560 | **1.55x** | **1.04x** | **0.85x** | **0.84x** |

### Qwen3-8B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 4096->4096 | **2.19x** | **1.18x** | **0.90x** | **1.34x** |
| kv_proj | 4096->2048 | **1.56x** | **0.59x** | **0.66x** | **0.79x** |
| o_proj | 4096->4096 | **2.19x** | **1.18x** | **0.90x** | **1.34x** |
| gate_up_proj | 4096->24576 | **2.29x** | **3.25x** | **1.85x** | **1.52x** |
| down_proj | 12288->4096 | **2.12x** | **1.73x** | **1.25x** | **1.07x** |


## 2. End-to-end raw latencies (us)


### Qwen3-0.6B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 3072->1024 | 1 | 9.53 | 133.41 | 10.09 | 0.07x | 0.95x | 13.23x |
| down_proj | 3072->1024 | 32 | 9.18 | 133.58 | 33.80 | 0.07x | 0.27x | 3.95x |
| down_proj | 3072->1024 | 128 | 10.81 | 134.58 | 34.04 | 0.08x | 0.32x | 3.95x |
| down_proj | 3072->1024 | 512 | 17.13 | 134.51 | 38.05 | 0.13x | 0.45x | 3.54x |
| gate_up_proj | 1024->6144 | 1 | 16.35 | 133.91 | 9.39 | 0.12x | 1.74x | 14.26x |
| gate_up_proj | 1024->6144 | 32 | 15.55 | 133.52 | 30.42 | 0.12x | 0.51x | 4.39x |
| gate_up_proj | 1024->6144 | 128 | 17.25 | 133.62 | 29.85 | 0.13x | 0.58x | 4.48x |
| gate_up_proj | 1024->6144 | 512 | 45.13 | 133.44 | 50.90 | 0.34x | 0.89x | 2.62x |
| kv_proj | 1024->2048 | 1 | 7.68 | 133.04 | 7.54 | 0.06x | 1.02x | 17.65x |
| kv_proj | 1024->2048 | 32 | 7.89 | 132.18 | 30.38 | 0.06x | 0.26x | 4.35x |
| kv_proj | 1024->2048 | 128 | 7.00 | 132.84 | 30.18 | 0.05x | 0.23x | 4.40x |
| kv_proj | 1024->2048 | 512 | 10.79 | 133.34 | 30.29 | 0.08x | 0.36x | 4.40x |
| o_proj | 2048->1024 | 1 | 6.66 | 132.60 | 7.53 | 0.05x | 0.88x | 17.62x |
| o_proj | 2048->1024 | 32 | 6.51 | 132.81 | 34.11 | 0.05x | 0.19x | 3.89x |
| o_proj | 2048->1024 | 128 | 7.79 | 133.81 | 34.36 | 0.06x | 0.23x | 3.89x |
| o_proj | 2048->1024 | 512 | 11.37 | 133.02 | 34.21 | 0.09x | 0.33x | 3.89x |
| q_proj | 1024->2048 | 1 | 7.38 | 133.69 | 7.61 | 0.06x | 0.97x | 17.58x |
| q_proj | 1024->2048 | 32 | 8.21 | 133.19 | 30.05 | 0.06x | 0.27x | 4.43x |
| q_proj | 1024->2048 | 128 | 7.35 | 133.15 | 30.21 | 0.06x | 0.24x | 4.41x |
| q_proj | 1024->2048 | 512 | 11.07 | 132.39 | 30.27 | 0.08x | 0.37x | 4.37x |

### Qwen3-1.7B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 6144->2048 | 1 | 31.39 | 137.41 | 20.55 | 0.23x | 1.53x | 6.69x |
| down_proj | 6144->2048 | 32 | 30.90 | 160.48 | 38.78 | 0.19x | 0.80x | 4.14x |
| down_proj | 6144->2048 | 128 | 35.08 | 163.23 | 48.38 | 0.21x | 0.73x | 3.37x |
| down_proj | 6144->2048 | 512 | 75.86 | 228.70 | 86.99 | 0.33x | 0.87x | 2.63x |
| gate_up_proj | 2048->12288 | 1 | 58.93 | 131.53 | 25.03 | 0.45x | 2.35x | 5.26x |
| gate_up_proj | 2048->12288 | 32 | 57.78 | 132.44 | 34.20 | 0.44x | 1.69x | 3.87x |
| gate_up_proj | 2048->12288 | 128 | 65.34 | 133.88 | 42.32 | 0.49x | 1.54x | 3.16x |
| gate_up_proj | 2048->12288 | 512 | 146.94 | 266.69 | 121.78 | 0.55x | 1.21x | 2.19x |
| kv_proj | 2048->2048 | 1 | 11.44 | 132.79 | 7.41 | 0.09x | 1.54x | 17.92x |
| kv_proj | 2048->2048 | 32 | 11.52 | 133.54 | 34.05 | 0.09x | 0.34x | 3.92x |
| kv_proj | 2048->2048 | 128 | 13.11 | 133.22 | 34.13 | 0.10x | 0.38x | 3.90x |
| kv_proj | 2048->2048 | 512 | 20.34 | 133.23 | 34.54 | 0.15x | 0.59x | 3.86x |
| o_proj | 2048->2048 | 1 | 11.23 | 133.08 | 7.36 | 0.08x | 1.53x | 18.09x |
| o_proj | 2048->2048 | 32 | 11.51 | 132.64 | 34.05 | 0.09x | 0.34x | 3.90x |
| o_proj | 2048->2048 | 128 | 13.00 | 132.65 | 34.33 | 0.10x | 0.38x | 3.86x |
| o_proj | 2048->2048 | 512 | 20.28 | 133.73 | 34.56 | 0.15x | 0.59x | 3.87x |
| q_proj | 2048->2048 | 1 | 11.21 | 132.75 | 7.44 | 0.08x | 1.51x | 17.85x |
| q_proj | 2048->2048 | 32 | 11.59 | 133.08 | 34.14 | 0.09x | 0.34x | 3.90x |
| q_proj | 2048->2048 | 128 | 13.01 | 133.78 | 34.22 | 0.10x | 0.38x | 3.91x |
| q_proj | 2048->2048 | 512 | 20.20 | 134.06 | 34.55 | 0.15x | 0.58x | 3.88x |

### Qwen3-4B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 9728->2560 | 1 | 59.10 | 220.60 | 38.15 | 0.27x | 1.55x | 5.78x |
| down_proj | 9728->2560 | 32 | 60.16 | 249.58 | 57.79 | 0.24x | 1.04x | 4.32x |
| down_proj | 9728->2560 | 128 | 66.88 | 273.78 | 78.77 | 0.24x | 0.85x | 3.48x |
| down_proj | 9728->2560 | 512 | 161.93 | 486.21 | 193.41 | 0.33x | 0.84x | 2.51x |
| gate_up_proj | 2560->19456 | 1 | 109.99 | 132.67 | 48.65 | 0.83x | 2.26x | 2.73x |
| gate_up_proj | 2560->19456 | 32 | 118.13 | 134.06 | 46.89 | 0.88x | 2.52x | 2.86x |
| gate_up_proj | 2560->19456 | 128 | 129.15 | 187.47 | 91.08 | 0.69x | 1.42x | 2.06x |
| gate_up_proj | 2560->19456 | 512 | 332.86 | 505.29 | 229.94 | 0.66x | 1.45x | 2.20x |
| kv_proj | 2560->2048 | 1 | 14.20 | 133.15 | 9.99 | 0.11x | 1.42x | 13.32x |
| kv_proj | 2560->2048 | 32 | 15.19 | 133.91 | 34.15 | 0.11x | 0.44x | 3.92x |
| kv_proj | 2560->2048 | 128 | 15.46 | 134.55 | 34.19 | 0.11x | 0.45x | 3.94x |
| kv_proj | 2560->2048 | 512 | 26.29 | 133.97 | 42.45 | 0.20x | 0.62x | 3.16x |
| o_proj | 4096->2560 | 1 | 25.77 | 132.94 | 15.91 | 0.19x | 1.62x | 8.35x |
| o_proj | 4096->2560 | 32 | 26.04 | 134.64 | 34.10 | 0.19x | 0.76x | 3.95x |
| o_proj | 4096->2560 | 128 | 29.47 | 135.23 | 41.89 | 0.22x | 0.70x | 3.23x |
| o_proj | 4096->2560 | 512 | 80.44 | 211.26 | 82.45 | 0.38x | 0.98x | 2.56x |
| q_proj | 2560->4096 | 1 | 26.01 | 133.17 | 13.27 | 0.20x | 1.96x | 10.04x |
| q_proj | 2560->4096 | 32 | 26.82 | 134.65 | 34.01 | 0.20x | 0.79x | 3.96x |
| q_proj | 2560->4096 | 128 | 27.73 | 134.52 | 35.13 | 0.21x | 0.79x | 3.83x |
| q_proj | 2560->4096 | 512 | 72.03 | 144.69 | 58.72 | 0.50x | 1.23x | 2.46x |

### Qwen3-8B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 12288->4096 | 1 | 112.23 | 273.10 | 53.02 | 0.41x | 2.12x | 5.15x |
| down_proj | 12288->4096 | 32 | 123.89 | 318.58 | 71.68 | 0.39x | 1.73x | 4.44x |
| down_proj | 12288->4096 | 128 | 134.24 | 349.82 | 107.04 | 0.38x | 1.25x | 3.27x |
| down_proj | 12288->4096 | 512 | 322.13 | 644.75 | 301.03 | 0.50x | 1.07x | 2.14x |
| gate_up_proj | 4096->24576 | 1 | 216.30 | 134.89 | 94.58 | 1.60x | 2.29x | 1.43x |
| gate_up_proj | 4096->24576 | 32 | 238.78 | 182.17 | 73.37 | 1.31x | 3.25x | 2.48x |
| gate_up_proj | 4096->24576 | 128 | 274.33 | 308.59 | 147.98 | 0.89x | 1.85x | 2.09x |
| gate_up_proj | 4096->24576 | 512 | 700.01 | 986.32 | 460.85 | 0.71x | 1.52x | 2.14x |
| kv_proj | 4096->2048 | 1 | 21.28 | 135.34 | 13.68 | 0.16x | 1.56x | 9.89x |
| kv_proj | 4096->2048 | 32 | 21.22 | 135.14 | 35.72 | 0.16x | 0.59x | 3.78x |
| kv_proj | 4096->2048 | 128 | 22.45 | 134.01 | 34.20 | 0.17x | 0.66x | 3.92x |
| kv_proj | 4096->2048 | 512 | 47.64 | 155.36 | 60.65 | 0.31x | 0.79x | 2.56x |
| o_proj | 4096->4096 | 1 | 40.70 | 134.86 | 18.58 | 0.30x | 2.19x | 7.26x |
| o_proj | 4096->4096 | 32 | 40.41 | 135.43 | 34.33 | 0.30x | 1.18x | 3.94x |
| o_proj | 4096->4096 | 128 | 42.32 | 135.40 | 46.85 | 0.31x | 0.90x | 2.89x |
| o_proj | 4096->4096 | 512 | 115.58 | 220.63 | 86.07 | 0.52x | 1.34x | 2.56x |
| q_proj | 4096->4096 | 1 | 40.77 | 134.99 | 18.62 | 0.30x | 2.19x | 7.25x |
| q_proj | 4096->4096 | 32 | 40.40 | 135.86 | 34.23 | 0.30x | 1.18x | 3.97x |
| q_proj | 4096->4096 | 128 | 42.35 | 135.96 | 46.83 | 0.31x | 0.90x | 2.90x |
| q_proj | 4096->4096 | 512 | 115.66 | 221.03 | 86.38 | 0.52x | 1.34x | 2.56x |


## 3. Sub-kernel breakdown (us)


### Qwen3-0.6B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 43.96 | 15.94 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 9.49 | 70.29 | 12.51 | 0.14x | 0.76x | 5.62x |
| down_proj | 1 | sparse_gemm | 9.54 | 68.59 | 19.63 | 0.14x | 0.49x | 3.49x |
| down_proj | 1 | fused_dense_sparse | 9.54 | 81.58 | 13.93 | 0.12x | 0.69x | 5.86x |
| down_proj | 32 | activation_quant |  -  | 44.99 | 16.11 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 9.20 | 69.41 | 24.55 | 0.13x | 0.37x | 2.83x |
| down_proj | 32 | sparse_gemm | 9.24 | 68.72 | 19.68 | 0.13x | 0.47x | 3.49x |
| down_proj | 32 | fused_dense_sparse | 9.23 | 81.66 | 17.56 | 0.11x | 0.53x | 4.65x |
| down_proj | 128 | activation_quant |  -  | 44.67 | 16.02 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 10.81 | 70.66 | 30.38 | 0.15x | 0.36x | 2.33x |
| down_proj | 128 | sparse_gemm | 10.78 | 68.75 | 19.66 | 0.16x | 0.55x | 3.50x |
| down_proj | 128 | fused_dense_sparse | 10.75 | 82.07 | 17.72 | 0.13x | 0.61x | 4.63x |
| down_proj | 512 | activation_quant |  -  | 43.90 | 16.18 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 17.06 | 69.59 | 37.94 | 0.25x | 0.45x | 1.83x |
| down_proj | 512 | sparse_gemm | 17.13 | 68.32 | 19.74 | 0.25x | 0.87x | 3.46x |
| down_proj | 512 | fused_dense_sparse | 17.17 | 81.44 | 28.20 | 0.21x | 0.61x | 2.89x |
| gate_up_proj | 1 | activation_quant |  -  | 43.51 | 15.88 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 16.40 | 69.20 | 12.59 | 0.24x | 1.30x | 5.50x |
| gate_up_proj | 1 | sparse_gemm | 16.23 | 68.30 | 19.42 | 0.24x | 0.84x | 3.52x |
| gate_up_proj | 1 | fused_dense_sparse | 16.30 | 81.03 | 13.85 | 0.20x | 1.18x | 5.85x |
| gate_up_proj | 32 | activation_quant |  -  | 44.61 | 16.07 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 15.65 | 69.54 | 14.06 | 0.23x | 1.11x | 4.95x |
| gate_up_proj | 32 | sparse_gemm | 15.58 | 68.49 | 19.52 | 0.23x | 0.80x | 3.51x |
| gate_up_proj | 32 | fused_dense_sparse | 15.62 | 81.00 | 15.10 | 0.19x | 1.03x | 5.36x |
| gate_up_proj | 128 | activation_quant |  -  | 43.14 | 16.25 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 17.22 | 69.20 | 19.02 | 0.25x | 0.91x | 3.64x |
| gate_up_proj | 128 | sparse_gemm | 17.29 | 67.87 | 19.33 | 0.25x | 0.89x | 3.51x |
| gate_up_proj | 128 | fused_dense_sparse | 17.27 | 81.24 | 17.67 | 0.21x | 0.98x | 4.60x |
| gate_up_proj | 512 | activation_quant |  -  | 44.57 | 16.27 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 44.83 | 69.77 | 51.13 | 0.64x | 0.88x | 1.36x |
| gate_up_proj | 512 | sparse_gemm | 45.10 | 68.69 | 30.11 | 0.66x | 1.50x | 2.28x |
| gate_up_proj | 512 | fused_dense_sparse | 45.03 | 81.78 | 46.06 | 0.55x | 0.98x | 1.78x |
| kv_proj | 1 | activation_quant |  -  | 42.67 | 15.93 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 7.62 | 69.23 | 12.48 | 0.11x | 0.61x | 5.55x |
| kv_proj | 1 | sparse_gemm | 7.66 | 68.61 | 19.50 | 0.11x | 0.39x | 3.52x |
| kv_proj | 1 | fused_dense_sparse | 7.58 | 82.04 | 14.13 | 0.09x | 0.54x | 5.81x |
| kv_proj | 32 | activation_quant |  -  | 42.75 | 16.33 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 7.94 | 69.94 | 12.44 | 0.11x | 0.64x | 5.62x |
| kv_proj | 32 | sparse_gemm | 7.93 | 68.60 | 19.58 | 0.12x | 0.40x | 3.50x |
| kv_proj | 32 | fused_dense_sparse | 7.91 | 80.72 | 13.91 | 0.10x | 0.57x | 5.80x |
| kv_proj | 128 | activation_quant |  -  | 43.88 | 16.07 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 7.02 | 69.21 | 13.93 | 0.10x | 0.50x | 4.97x |
| kv_proj | 128 | sparse_gemm | 7.03 | 68.15 | 19.66 | 0.10x | 0.36x | 3.47x |
| kv_proj | 128 | fused_dense_sparse | 7.01 | 81.08 | 14.75 | 0.09x | 0.48x | 5.50x |
| kv_proj | 512 | activation_quant |  -  | 44.47 | 16.11 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 10.75 | 69.14 | 20.74 | 0.16x | 0.52x | 3.33x |
| kv_proj | 512 | sparse_gemm | 10.81 | 68.88 | 19.77 | 0.16x | 0.55x | 3.48x |
| kv_proj | 512 | fused_dense_sparse | 10.83 | 81.57 | 21.47 | 0.13x | 0.50x | 3.80x |
| o_proj | 1 | activation_quant |  -  | 44.26 | 16.20 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 6.68 | 68.77 | 12.53 | 0.10x | 0.53x | 5.49x |
| o_proj | 1 | sparse_gemm | 6.68 | 68.16 | 19.52 | 0.10x | 0.34x | 3.49x |
| o_proj | 1 | fused_dense_sparse | 6.72 | 80.67 | 13.92 | 0.08x | 0.48x | 5.79x |
| o_proj | 32 | activation_quant |  -  | 42.84 | 16.15 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 6.49 | 68.63 | 16.60 | 0.09x | 0.39x | 4.13x |
| o_proj | 32 | sparse_gemm | 6.49 | 68.38 | 19.60 | 0.09x | 0.33x | 3.49x |
| o_proj | 32 | fused_dense_sparse | 6.50 | 80.24 | 17.66 | 0.08x | 0.37x | 4.54x |
| o_proj | 128 | activation_quant |  -  | 42.66 | 16.16 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 7.86 | 69.19 | 19.87 | 0.11x | 0.40x | 3.48x |
| o_proj | 128 | sparse_gemm | 7.84 | 68.06 | 19.74 | 0.12x | 0.40x | 3.45x |
| o_proj | 128 | fused_dense_sparse | 7.81 | 80.61 | 17.57 | 0.10x | 0.44x | 4.59x |
| o_proj | 512 | activation_quant |  -  | 43.00 | 16.15 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 11.41 | 69.07 | 26.22 | 0.17x | 0.44x | 2.63x |
| o_proj | 512 | sparse_gemm | 11.45 | 68.62 | 19.60 | 0.17x | 0.58x | 3.50x |
| o_proj | 512 | fused_dense_sparse | 11.36 | 80.90 | 22.99 | 0.14x | 0.49x | 3.52x |
| q_proj | 1 | activation_quant |  -  | 43.23 | 16.02 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 7.52 | 68.52 | 12.56 | 0.11x | 0.60x | 5.45x |
| q_proj | 1 | sparse_gemm | 7.40 | 69.47 | 19.92 | 0.11x | 0.37x | 3.49x |
| q_proj | 1 | fused_dense_sparse | 7.28 | 82.63 | 14.04 | 0.09x | 0.52x | 5.88x |
| q_proj | 32 | activation_quant |  -  | 44.34 | 16.05 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 8.17 | 69.87 | 12.41 | 0.12x | 0.66x | 5.63x |
| q_proj | 32 | sparse_gemm | 7.76 | 67.49 | 19.54 | 0.12x | 0.40x | 3.45x |
| q_proj | 32 | fused_dense_sparse | 7.99 | 82.63 | 13.86 | 0.10x | 0.58x | 5.96x |
| q_proj | 128 | activation_quant |  -  | 44.62 | 16.01 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 7.12 | 69.58 | 13.92 | 0.10x | 0.51x | 5.00x |
| q_proj | 128 | sparse_gemm | 7.30 | 68.37 | 19.70 | 0.11x | 0.37x | 3.47x |
| q_proj | 128 | fused_dense_sparse | 7.18 | 81.08 | 14.75 | 0.09x | 0.49x | 5.50x |
| q_proj | 512 | activation_quant |  -  | 42.88 | 16.04 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 11.14 | 69.00 | 20.74 | 0.16x | 0.54x | 3.33x |
| q_proj | 512 | sparse_gemm | 11.08 | 68.36 | 19.59 | 0.16x | 0.57x | 3.49x |
| q_proj | 512 | fused_dense_sparse | 10.97 | 82.20 | 21.28 | 0.13x | 0.52x | 3.86x |

### Qwen3-1.7B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 58.49 | 16.16 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 31.01 | 70.47 | 15.40 | 0.44x | 2.01x | 4.58x |
| down_proj | 1 | sparse_gemm | 31.40 | 67.39 | 19.69 | 0.47x | 1.60x | 3.42x |
| down_proj | 1 | fused_dense_sparse | 31.03 | 81.15 | 16.08 | 0.38x | 1.93x | 5.05x |
| down_proj | 32 | activation_quant |  -  | 82.80 | 16.03 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 30.83 | 70.35 | 48.67 | 0.44x | 0.63x | 1.45x |
| down_proj | 32 | sparse_gemm | 30.74 | 68.44 | 19.58 | 0.45x | 1.57x | 3.50x |
| down_proj | 32 | fused_dense_sparse | 30.82 | 81.17 | 25.03 | 0.38x | 1.23x | 3.24x |
| down_proj | 128 | activation_quant |  -  | 83.55 | 16.35 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 35.10 | 71.54 | 72.95 | 0.49x | 0.48x | 0.98x |
| down_proj | 128 | sparse_gemm | 35.11 | 68.18 | 19.59 | 0.51x | 1.79x | 3.48x |
| down_proj | 128 | fused_dense_sparse | 35.08 | 81.83 | 33.98 | 0.43x | 1.03x | 2.41x |
| down_proj | 512 | activation_quant |  -  | 83.94 | 16.83 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 75.66 | 128.88 | 87.06 | 0.59x | 0.87x | 1.48x |
| down_proj | 512 | sparse_gemm | 75.74 | 67.57 | 19.46 | 1.12x | 3.89x | 3.47x |
| down_proj | 512 | fused_dense_sparse | 75.65 | 142.41 | 70.65 | 0.53x | 1.07x | 2.02x |
| gate_up_proj | 1 | activation_quant |  -  | 44.34 | 15.74 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 59.09 | 68.69 | 20.51 | 0.86x | 2.88x | 3.35x |
| gate_up_proj | 1 | sparse_gemm | 59.04 | 67.91 | 19.36 | 0.87x | 3.05x | 3.51x |
| gate_up_proj | 1 | fused_dense_sparse | 59.02 | 81.01 | 22.05 | 0.73x | 2.68x | 3.67x |
| gate_up_proj | 32 | activation_quant |  -  | 44.57 | 16.13 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 57.89 | 68.57 | 24.99 | 0.84x | 2.32x | 2.74x |
| gate_up_proj | 32 | sparse_gemm | 57.85 | 67.79 | 19.45 | 0.85x | 2.97x | 3.49x |
| gate_up_proj | 32 | fused_dense_sparse | 57.81 | 81.11 | 23.10 | 0.71x | 2.50x | 3.51x |
| gate_up_proj | 128 | activation_quant |  -  | 44.60 | 16.35 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 65.03 | 72.14 | 40.99 | 0.90x | 1.59x | 1.76x |
| gate_up_proj | 128 | sparse_gemm | 65.02 | 68.09 | 20.20 | 0.96x | 3.22x | 3.37x |
| gate_up_proj | 128 | fused_dense_sparse | 65.07 | 84.30 | 35.17 | 0.77x | 1.85x | 2.40x |
| gate_up_proj | 512 | activation_quant |  -  | 43.72 | 15.88 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 146.09 | 214.98 | 131.46 | 0.68x | 1.11x | 1.64x |
| gate_up_proj | 512 | sparse_gemm | 147.05 | 68.67 | 56.39 | 2.14x | 2.61x | 1.22x |
| gate_up_proj | 512 | fused_dense_sparse | 145.84 | 238.83 | 113.42 | 0.61x | 1.29x | 2.11x |
| kv_proj | 1 | activation_quant |  -  | 43.14 | 15.81 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 11.34 | 69.03 | 12.27 | 0.16x | 0.92x | 5.63x |
| kv_proj | 1 | sparse_gemm | 11.41 | 68.20 | 19.36 | 0.17x | 0.59x | 3.52x |
| kv_proj | 1 | fused_dense_sparse | 11.43 | 80.96 | 13.81 | 0.14x | 0.83x | 5.86x |
| kv_proj | 32 | activation_quant |  -  | 43.99 | 16.14 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 11.50 | 69.29 | 16.76 | 0.17x | 0.69x | 4.13x |
| kv_proj | 32 | sparse_gemm | 11.62 | 68.33 | 19.60 | 0.17x | 0.59x | 3.49x |
| kv_proj | 32 | fused_dense_sparse | 11.58 | 80.93 | 17.68 | 0.14x | 0.65x | 4.58x |
| kv_proj | 128 | activation_quant |  -  | 44.47 | 16.15 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 13.07 | 69.02 | 24.58 | 0.19x | 0.53x | 2.81x |
| kv_proj | 128 | sparse_gemm | 13.14 | 68.37 | 19.76 | 0.19x | 0.66x | 3.46x |
| kv_proj | 128 | fused_dense_sparse | 13.13 | 81.17 | 17.71 | 0.16x | 0.74x | 4.58x |
| kv_proj | 512 | activation_quant |  -  | 44.06 | 16.08 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 20.40 | 69.73 | 33.18 | 0.29x | 0.61x | 2.10x |
| kv_proj | 512 | sparse_gemm | 20.39 | 67.95 | 19.66 | 0.30x | 1.04x | 3.46x |
| kv_proj | 512 | fused_dense_sparse | 20.38 | 81.92 | 27.34 | 0.25x | 0.75x | 3.00x |
| o_proj | 1 | activation_quant |  -  | 43.75 | 15.79 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 11.20 | 68.80 | 12.25 | 0.16x | 0.91x | 5.62x |
| o_proj | 1 | sparse_gemm | 11.20 | 67.97 | 19.29 | 0.16x | 0.58x | 3.52x |
| o_proj | 1 | fused_dense_sparse | 11.22 | 81.26 | 13.75 | 0.14x | 0.82x | 5.91x |
| o_proj | 32 | activation_quant |  -  | 44.33 | 16.14 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 11.55 | 67.90 | 16.76 | 0.17x | 0.69x | 4.05x |
| o_proj | 32 | sparse_gemm | 11.56 | 67.09 | 19.59 | 0.17x | 0.59x | 3.42x |
| o_proj | 32 | fused_dense_sparse | 11.58 | 81.12 | 17.71 | 0.14x | 0.65x | 4.58x |
| o_proj | 128 | activation_quant |  -  | 43.58 | 16.19 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 13.08 | 68.63 | 24.56 | 0.19x | 0.53x | 2.79x |
| o_proj | 128 | sparse_gemm | 13.07 | 67.79 | 19.66 | 0.19x | 0.66x | 3.45x |
| o_proj | 128 | fused_dense_sparse | 12.99 | 81.41 | 17.83 | 0.16x | 0.73x | 4.57x |
| o_proj | 512 | activation_quant |  -  | 43.12 | 16.16 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 20.20 | 68.38 | 33.20 | 0.30x | 0.61x | 2.06x |
| o_proj | 512 | sparse_gemm | 20.25 | 66.86 | 19.58 | 0.30x | 1.03x | 3.41x |
| o_proj | 512 | fused_dense_sparse | 20.12 | 82.14 | 27.32 | 0.24x | 0.74x | 3.01x |
| q_proj | 1 | activation_quant |  -  | 43.63 | 15.82 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 11.22 | 68.99 | 12.32 | 0.16x | 0.91x | 5.60x |
| q_proj | 1 | sparse_gemm | 11.31 | 68.08 | 19.27 | 0.17x | 0.59x | 3.53x |
| q_proj | 1 | fused_dense_sparse | 11.24 | 80.90 | 13.66 | 0.14x | 0.82x | 5.92x |
| q_proj | 32 | activation_quant |  -  | 43.69 | 16.05 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 11.56 | 69.13 | 16.77 | 0.17x | 0.69x | 4.12x |
| q_proj | 32 | sparse_gemm | 11.55 | 67.89 | 19.46 | 0.17x | 0.59x | 3.49x |
| q_proj | 32 | fused_dense_sparse | 11.47 | 82.38 | 17.69 | 0.14x | 0.65x | 4.66x |
| q_proj | 128 | activation_quant |  -  | 43.83 | 16.14 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 13.06 | 69.16 | 24.58 | 0.19x | 0.53x | 2.81x |
| q_proj | 128 | sparse_gemm | 13.03 | 68.28 | 19.63 | 0.19x | 0.66x | 3.48x |
| q_proj | 128 | fused_dense_sparse | 13.13 | 81.52 | 17.67 | 0.16x | 0.74x | 4.61x |
| q_proj | 512 | activation_quant |  -  | 44.20 | 16.29 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 20.16 | 68.85 | 33.20 | 0.29x | 0.61x | 2.07x |
| q_proj | 512 | sparse_gemm | 20.15 | 67.93 | 19.71 | 0.30x | 1.02x | 3.45x |
| q_proj | 512 | fused_dense_sparse | 20.26 | 81.10 | 27.32 | 0.25x | 0.74x | 2.97x |

### Qwen3-4B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 92.55 | 19.37 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 58.94 | 113.80 | 29.19 | 0.52x | 2.02x | 3.90x |
| down_proj | 1 | sparse_gemm | 59.00 | 69.06 | 19.71 | 0.85x | 2.99x | 3.50x |
| down_proj | 1 | fused_dense_sparse | 59.14 | 130.45 | 30.40 | 0.45x | 1.95x | 4.29x |
| down_proj | 32 | activation_quant |  -  | 129.40 | 21.28 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 60.13 | 112.01 | 91.15 | 0.54x | 0.66x | 1.23x |
| down_proj | 32 | sparse_gemm | 60.20 | 68.82 | 19.56 | 0.87x | 3.08x | 3.52x |
| down_proj | 32 | fused_dense_sparse | 60.17 | 120.67 | 36.69 | 0.50x | 1.64x | 3.29x |
| down_proj | 128 | activation_quant |  -  | 129.70 | 21.53 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 66.93 | 133.13 | 109.30 | 0.50x | 0.61x | 1.22x |
| down_proj | 128 | sparse_gemm | 66.90 | 68.59 | 19.98 | 0.98x | 3.35x | 3.43x |
| down_proj | 128 | fused_dense_sparse | 66.94 | 143.73 | 56.80 | 0.47x | 1.18x | 2.53x |
| down_proj | 512 | activation_quant |  -  | 129.95 | 24.24 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 164.09 | 327.56 | 247.66 | 0.50x | 0.66x | 1.32x |
| down_proj | 512 | sparse_gemm | 161.65 | 68.21 | 24.69 | 2.37x | 6.55x | 2.76x |
| down_proj | 512 | fused_dense_sparse | 161.37 | 355.54 | 168.93 | 0.45x | 0.96x | 2.10x |
| gate_up_proj | 1 | activation_quant |  -  | 43.25 | 15.87 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 110.03 | 69.72 | 37.71 | 1.58x | 2.92x | 1.85x |
| gate_up_proj | 1 | sparse_gemm | 110.01 | 67.94 | 19.54 | 1.62x | 5.63x | 3.48x |
| gate_up_proj | 1 | fused_dense_sparse | 110.07 | 81.99 | 40.67 | 1.34x | 2.71x | 2.02x |
| gate_up_proj | 32 | activation_quant |  -  | 42.81 | 15.93 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 118.06 | 68.75 | 44.99 | 1.72x | 2.62x | 1.53x |
| gate_up_proj | 32 | sparse_gemm | 118.09 | 67.24 | 19.51 | 1.76x | 6.05x | 3.45x |
| gate_up_proj | 32 | fused_dense_sparse | 118.10 | 81.81 | 38.53 | 1.44x | 3.06x | 2.12x |
| gate_up_proj | 128 | activation_quant |  -  | 44.53 | 16.21 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 129.31 | 142.61 | 102.37 | 0.91x | 1.26x | 1.39x |
| gate_up_proj | 128 | sparse_gemm | 129.21 | 67.24 | 27.16 | 1.92x | 4.76x | 2.48x |
| gate_up_proj | 128 | fused_dense_sparse | 129.49 | 152.98 | 82.44 | 0.85x | 1.57x | 1.86x |
| gate_up_proj | 512 | activation_quant |  -  | 45.15 | 16.47 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 329.04 | 441.22 | 260.90 | 0.75x | 1.26x | 1.69x |
| gate_up_proj | 512 | sparse_gemm | 330.09 | 67.32 | 85.05 | 4.90x | 3.88x | 0.79x |
| gate_up_proj | 512 | fused_dense_sparse | 329.55 | 470.18 | 221.70 | 0.70x | 1.49x | 2.12x |
| kv_proj | 1 | activation_quant |  -  | 42.70 | 15.93 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 14.22 | 69.88 | 12.41 | 0.20x | 1.15x | 5.63x |
| kv_proj | 1 | sparse_gemm | 14.22 | 68.19 | 19.50 | 0.21x | 0.73x | 3.50x |
| kv_proj | 1 | fused_dense_sparse | 14.19 | 81.40 | 13.90 | 0.17x | 1.02x | 5.86x |
| kv_proj | 32 | activation_quant |  -  | 44.53 | 16.02 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 15.22 | 69.19 | 20.85 | 0.22x | 0.73x | 3.32x |
| kv_proj | 32 | sparse_gemm | 15.18 | 68.52 | 19.44 | 0.22x | 0.78x | 3.53x |
| kv_proj | 32 | fused_dense_sparse | 15.17 | 81.21 | 17.74 | 0.19x | 0.86x | 4.58x |
| kv_proj | 128 | activation_quant |  -  | 44.66 | 16.20 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 15.48 | 69.53 | 34.78 | 0.22x | 0.45x | 2.00x |
| kv_proj | 128 | sparse_gemm | 15.52 | 68.07 | 19.56 | 0.23x | 0.79x | 3.48x |
| kv_proj | 128 | fused_dense_sparse | 15.49 | 80.84 | 17.64 | 0.19x | 0.88x | 4.58x |
| kv_proj | 512 | activation_quant |  -  | 44.21 | 16.16 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 26.26 | 70.25 | 42.63 | 0.37x | 0.62x | 1.65x |
| kv_proj | 512 | sparse_gemm | 26.27 | 69.18 | 19.64 | 0.38x | 1.34x | 3.52x |
| kv_proj | 512 | fused_dense_sparse | 26.40 | 82.86 | 33.82 | 0.32x | 0.78x | 2.45x |
| o_proj | 1 | activation_quant |  -  | 43.97 | 15.98 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 25.79 | 68.58 | 13.31 | 0.38x | 1.94x | 5.15x |
| o_proj | 1 | sparse_gemm | 25.84 | 67.47 | 19.39 | 0.38x | 1.33x | 3.48x |
| o_proj | 1 | fused_dense_sparse | 25.83 | 80.83 | 13.92 | 0.32x | 1.86x | 5.81x |
| o_proj | 32 | activation_quant |  -  | 55.16 | 16.04 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 26.01 | 69.87 | 37.70 | 0.37x | 0.69x | 1.85x |
| o_proj | 32 | sparse_gemm | 25.93 | 68.94 | 19.66 | 0.38x | 1.32x | 3.51x |
| o_proj | 32 | fused_dense_sparse | 25.99 | 81.01 | 21.25 | 0.32x | 1.22x | 3.81x |
| o_proj | 128 | activation_quant |  -  | 55.29 | 16.07 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 29.64 | 69.41 | 46.95 | 0.43x | 0.63x | 1.48x |
| o_proj | 128 | sparse_gemm | 29.59 | 67.36 | 19.38 | 0.44x | 1.53x | 3.48x |
| o_proj | 128 | fused_dense_sparse | 29.57 | 82.28 | 29.90 | 0.36x | 0.99x | 2.75x |
| o_proj | 512 | activation_quant |  -  | 55.43 | 16.36 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 80.23 | 141.91 | 72.07 | 0.57x | 1.11x | 1.97x |
| o_proj | 512 | sparse_gemm | 80.39 | 67.03 | 19.39 | 1.20x | 4.15x | 3.46x |
| o_proj | 512 | fused_dense_sparse | 80.24 | 155.17 | 68.31 | 0.52x | 1.17x | 2.27x |
| q_proj | 1 | activation_quant |  -  | 44.19 | 15.94 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 25.96 | 68.59 | 12.79 | 0.38x | 2.03x | 5.36x |
| q_proj | 1 | sparse_gemm | 25.94 | 68.19 | 19.53 | 0.38x | 1.33x | 3.49x |
| q_proj | 1 | fused_dense_sparse | 25.89 | 80.41 | 14.08 | 0.32x | 1.84x | 5.71x |
| q_proj | 32 | activation_quant |  -  | 43.98 | 16.15 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 26.84 | 69.45 | 29.38 | 0.39x | 0.91x | 2.36x |
| q_proj | 32 | sparse_gemm | 26.88 | 69.05 | 19.68 | 0.39x | 1.37x | 3.51x |
| q_proj | 32 | fused_dense_sparse | 26.85 | 81.00 | 17.73 | 0.33x | 1.51x | 4.57x |
| q_proj | 128 | activation_quant |  -  | 44.58 | 16.24 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 27.82 | 69.49 | 36.15 | 0.40x | 0.77x | 1.92x |
| q_proj | 128 | sparse_gemm | 27.88 | 68.49 | 19.74 | 0.41x | 1.41x | 3.47x |
| q_proj | 128 | fused_dense_sparse | 27.90 | 81.83 | 27.26 | 0.34x | 1.02x | 3.00x |
| q_proj | 512 | activation_quant |  -  | 45.16 | 16.35 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 72.50 | 92.82 | 55.14 | 0.78x | 1.31x | 1.68x |
| q_proj | 512 | sparse_gemm | 72.12 | 68.18 | 24.00 | 1.06x | 3.00x | 2.84x |
| q_proj | 512 | fused_dense_sparse | 72.24 | 108.14 | 49.76 | 0.67x | 1.45x | 2.17x |

### Qwen3-8B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 115.74 | 24.77 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 112.23 | 145.16 | 43.90 | 0.77x | 2.56x | 3.31x |
| down_proj | 1 | sparse_gemm | 112.37 | 68.80 | 19.66 | 1.63x | 5.72x | 3.50x |
| down_proj | 1 | fused_dense_sparse | 112.24 | 157.77 | 45.67 | 0.71x | 2.46x | 3.45x |
| down_proj | 32 | activation_quant |  -  | 164.27 | 24.72 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 123.75 | 142.08 | 118.65 | 0.87x | 1.04x | 1.20x |
| down_proj | 32 | sparse_gemm | 123.88 | 69.23 | 19.45 | 1.79x | 6.37x | 3.56x |
| down_proj | 32 | fused_dense_sparse | 123.86 | 154.78 | 46.38 | 0.80x | 2.67x | 3.34x |
| down_proj | 128 | activation_quant |  -  | 165.84 | 25.54 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 134.21 | 170.15 | 143.75 | 0.79x | 0.93x | 1.18x |
| down_proj | 128 | sparse_gemm | 134.32 | 68.25 | 22.37 | 1.97x | 6.00x | 3.05x |
| down_proj | 128 | fused_dense_sparse | 134.37 | 183.11 | 80.47 | 0.73x | 1.67x | 2.28x |
| down_proj | 512 | activation_quant |  -  | 165.92 | 51.27 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 320.90 | 430.47 | 333.71 | 0.75x | 0.96x | 1.29x |
| down_proj | 512 | sparse_gemm | 320.72 | 68.42 | 36.59 | 4.69x | 8.77x | 1.87x |
| down_proj | 512 | fused_dense_sparse | 320.55 | 476.55 | 252.95 | 0.67x | 1.27x | 1.88x |
| gate_up_proj | 1 | activation_quant |  -  | 44.58 | 16.37 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 216.46 | 78.70 | 72.93 | 2.75x | 2.97x | 1.08x |
| gate_up_proj | 1 | sparse_gemm | 216.21 | 68.38 | 19.35 | 3.16x | 11.17x | 3.53x |
| gate_up_proj | 1 | fused_dense_sparse | 216.32 | 87.04 | 77.32 | 2.49x | 2.80x | 1.13x |
| gate_up_proj | 32 | activation_quant |  -  | 56.31 | 16.31 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 238.73 | 111.52 | 64.43 | 2.14x | 3.71x | 1.73x |
| gate_up_proj | 32 | sparse_gemm | 238.69 | 67.24 | 19.68 | 3.55x | 12.13x | 3.42x |
| gate_up_proj | 32 | fused_dense_sparse | 238.77 | 124.85 | 60.71 | 1.91x | 3.93x | 2.06x |
| gate_up_proj | 128 | activation_quant |  -  | 55.60 | 16.43 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 274.26 | 221.14 | 147.18 | 1.24x | 1.86x | 1.50x |
| gate_up_proj | 128 | sparse_gemm | 274.50 | 68.57 | 37.27 | 4.00x | 7.37x | 1.84x |
| gate_up_proj | 128 | fused_dense_sparse | 274.27 | 250.85 | 135.16 | 1.09x | 2.03x | 1.86x |
| gate_up_proj | 512 | activation_quant |  -  | 55.96 | 16.52 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 693.24 | 862.26 | 451.66 | 0.80x | 1.53x | 1.91x |
| gate_up_proj | 512 | sparse_gemm | 712.71 | 68.28 | 107.43 | 10.44x | 6.63x | 0.64x |
| gate_up_proj | 512 | fused_dense_sparse | 693.38 | 945.19 | 445.28 | 0.73x | 1.56x | 2.12x |
| kv_proj | 1 | activation_quant |  -  | 43.71 | 15.96 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 21.35 | 70.00 | 12.46 | 0.30x | 1.71x | 5.62x |
| kv_proj | 1 | sparse_gemm | 21.34 | 68.89 | 19.47 | 0.31x | 1.10x | 3.54x |
| kv_proj | 1 | fused_dense_sparse | 21.33 | 82.68 | 14.08 | 0.26x | 1.52x | 5.87x |
| kv_proj | 32 | activation_quant |  -  | 55.41 | 16.08 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 21.23 | 69.59 | 32.06 | 0.31x | 0.66x | 2.17x |
| kv_proj | 32 | sparse_gemm | 21.29 | 68.26 | 19.59 | 0.31x | 1.09x | 3.48x |
| kv_proj | 32 | fused_dense_sparse | 21.26 | 81.86 | 24.19 | 0.26x | 0.88x | 3.38x |
| kv_proj | 128 | activation_quant |  -  | 55.61 | 15.96 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 22.39 | 68.96 | 47.19 | 0.32x | 0.47x | 1.46x |
| kv_proj | 128 | sparse_gemm | 22.39 | 68.27 | 19.57 | 0.33x | 1.14x | 3.49x |
| kv_proj | 128 | fused_dense_sparse | 22.35 | 81.07 | 22.00 | 0.28x | 1.02x | 3.69x |
| kv_proj | 512 | activation_quant |  -  | 55.67 | 16.38 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 47.81 | 87.09 | 58.03 | 0.55x | 0.82x | 1.50x |
| kv_proj | 512 | sparse_gemm | 47.64 | 68.01 | 19.36 | 0.70x | 2.46x | 3.51x |
| kv_proj | 512 | fused_dense_sparse | 47.48 | 96.48 | 48.19 | 0.49x | 0.99x | 2.00x |
| o_proj | 1 | activation_quant |  -  | 44.53 | 15.90 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 40.68 | 69.29 | 15.95 | 0.59x | 2.55x | 4.34x |
| o_proj | 1 | sparse_gemm | 40.63 | 67.71 | 19.38 | 0.60x | 2.10x | 3.49x |
| o_proj | 1 | fused_dense_sparse | 40.77 | 81.90 | 16.69 | 0.50x | 2.44x | 4.91x |
| o_proj | 32 | activation_quant |  -  | 55.46 | 15.90 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 40.39 | 69.57 | 37.85 | 0.58x | 1.07x | 1.84x |
| o_proj | 32 | sparse_gemm | 40.37 | 68.46 | 19.73 | 0.59x | 2.05x | 3.47x |
| o_proj | 32 | fused_dense_sparse | 40.43 | 82.81 | 22.44 | 0.49x | 1.80x | 3.69x |
| o_proj | 128 | activation_quant |  -  | 55.57 | 16.12 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 42.45 | 70.02 | 47.75 | 0.61x | 0.89x | 1.47x |
| o_proj | 128 | sparse_gemm | 42.41 | 69.04 | 19.71 | 0.61x | 2.15x | 3.50x |
| o_proj | 128 | fused_dense_sparse | 42.41 | 83.09 | 35.00 | 0.51x | 1.21x | 2.37x |
| o_proj | 512 | activation_quant |  -  | 55.63 | 16.17 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 115.23 | 145.17 | 80.68 | 0.79x | 1.43x | 1.80x |
| o_proj | 512 | sparse_gemm | 115.46 | 67.58 | 24.14 | 1.71x | 4.78x | 2.80x |
| o_proj | 512 | fused_dense_sparse | 115.41 | 162.45 | 73.50 | 0.71x | 1.57x | 2.21x |
| q_proj | 1 | activation_quant |  -  | 44.03 | 16.27 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 40.46 | 69.01 | 15.92 | 0.59x | 2.54x | 4.33x |
| q_proj | 1 | sparse_gemm | 40.66 | 67.85 | 19.42 | 0.60x | 2.09x | 3.49x |
| q_proj | 1 | fused_dense_sparse | 40.77 | 81.25 | 16.66 | 0.50x | 2.45x | 4.88x |
| q_proj | 32 | activation_quant |  -  | 55.48 | 16.16 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 40.44 | 69.56 | 37.86 | 0.58x | 1.07x | 1.84x |
| q_proj | 32 | sparse_gemm | 40.41 | 68.03 | 19.74 | 0.59x | 2.05x | 3.45x |
| q_proj | 32 | fused_dense_sparse | 40.42 | 81.57 | 22.47 | 0.50x | 1.80x | 3.63x |
| q_proj | 128 | activation_quant |  -  | 55.58 | 16.01 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 42.30 | 70.14 | 47.73 | 0.60x | 0.89x | 1.47x |
| q_proj | 128 | sparse_gemm | 42.41 | 68.84 | 19.85 | 0.62x | 2.14x | 3.47x |
| q_proj | 128 | fused_dense_sparse | 42.38 | 82.54 | 34.94 | 0.51x | 1.21x | 2.36x |
| q_proj | 512 | activation_quant |  -  | 55.61 | 16.08 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 115.25 | 145.27 | 80.63 | 0.79x | 1.43x | 1.80x |
| q_proj | 512 | sparse_gemm | 115.62 | 68.68 | 24.13 | 1.68x | 4.79x | 2.85x |
| q_proj | 512 | fused_dense_sparse | 115.29 | 162.52 | 73.52 | 0.71x | 1.57x | 2.21x |


## 4. End-to-end speedup (CUDA over Triton)

Rows: projection. Cells: `triton_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-0.6B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 1024->2048 | **17.58x** | **4.43x** | **4.41x** | **4.37x** |
| kv_proj | 1024->2048 | **17.65x** | **4.35x** | **4.40x** | **4.40x** |
| o_proj | 2048->1024 | **17.62x** | **3.89x** | **3.89x** | **3.89x** |
| gate_up_proj | 1024->6144 | **14.26x** | **4.39x** | **4.48x** | **2.62x** |
| down_proj | 3072->1024 | **13.23x** | **3.95x** | **3.95x** | **3.54x** |

### Qwen3-1.7B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 2048->2048 | **17.85x** | **3.90x** | **3.91x** | **3.88x** |
| kv_proj | 2048->2048 | **17.92x** | **3.92x** | **3.90x** | **3.86x** |
| o_proj | 2048->2048 | **18.09x** | **3.90x** | **3.86x** | **3.87x** |
| gate_up_proj | 2048->12288 | **5.26x** | **3.87x** | **3.16x** | **2.19x** |
| down_proj | 6144->2048 | **6.69x** | **4.14x** | **3.37x** | **2.63x** |

### Qwen3-4B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 2560->4096 | **10.04x** | **3.96x** | **3.83x** | **2.46x** |
| kv_proj | 2560->2048 | **13.32x** | **3.92x** | **3.94x** | **3.16x** |
| o_proj | 4096->2560 | **8.35x** | **3.95x** | **3.23x** | **2.56x** |
| gate_up_proj | 2560->19456 | **2.73x** | **2.86x** | **2.06x** | **2.20x** |
| down_proj | 9728->2560 | **5.78x** | **4.32x** | **3.48x** | **2.51x** |

### Qwen3-8B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 4096->4096 | **7.25x** | **3.97x** | **2.90x** | **2.56x** |
| kv_proj | 4096->2048 | **9.89x** | **3.78x** | **3.92x** | **2.56x** |
| o_proj | 4096->4096 | **7.26x** | **3.94x** | **2.89x** | **2.56x** |
| gate_up_proj | 4096->24576 | **1.43x** | **2.48x** | **2.09x** | **2.14x** |
| down_proj | 12288->4096 | **5.15x** | **4.44x** | **3.27x** | **2.14x** |


## 5. CUDA end-to-end bottleneck hint

For each shape, compare CUDA `activation_quant` against CUDA `fused_dense_sparse`. A larger `quant_share` means launch/prologue dominates; a larger `fused_share` means the main CUDA matmul kernel dominates.

| model | proj | T | shape | quant_us | fused_us | quant_share | fused_share | likely_bottleneck |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | q_proj | 1 | 1024->2048 | 16.02 | 14.04 | 53.3% | 46.7% | quant/prologue dominated |
| Qwen3-0.6B | q_proj | 32 | 1024->2048 | 16.05 | 13.86 | 53.6% | 46.4% | quant/prologue dominated |
| Qwen3-0.6B | q_proj | 128 | 1024->2048 | 16.01 | 14.75 | 52.0% | 48.0% | quant/prologue dominated |
| Qwen3-0.6B | q_proj | 512 | 1024->2048 | 16.04 | 21.28 | 43.0% | 57.0% | quant/prologue dominated |
| Qwen3-0.6B | kv_proj | 1 | 1024->2048 | 15.93 | 14.13 | 53.0% | 47.0% | quant/prologue dominated |
| Qwen3-0.6B | kv_proj | 32 | 1024->2048 | 16.33 | 13.91 | 54.0% | 46.0% | quant/prologue dominated |
| Qwen3-0.6B | kv_proj | 128 | 1024->2048 | 16.07 | 14.75 | 52.1% | 47.9% | quant/prologue dominated |
| Qwen3-0.6B | kv_proj | 512 | 1024->2048 | 16.11 | 21.47 | 42.9% | 57.1% | quant/prologue dominated |
| Qwen3-0.6B | o_proj | 1 | 2048->1024 | 16.20 | 13.92 | 53.8% | 46.2% | quant/prologue dominated |
| Qwen3-0.6B | o_proj | 32 | 2048->1024 | 16.15 | 17.66 | 47.8% | 52.2% | quant/prologue dominated |
| Qwen3-0.6B | o_proj | 128 | 2048->1024 | 16.16 | 17.57 | 47.9% | 52.1% | quant/prologue dominated |
| Qwen3-0.6B | o_proj | 512 | 2048->1024 | 16.15 | 22.99 | 41.3% | 58.7% | quant/prologue dominated |
| Qwen3-0.6B | gate_up_proj | 1 | 1024->6144 | 15.88 | 13.85 | 53.4% | 46.6% | quant/prologue dominated |
| Qwen3-0.6B | gate_up_proj | 32 | 1024->6144 | 16.07 | 15.10 | 51.5% | 48.5% | quant/prologue dominated |
| Qwen3-0.6B | gate_up_proj | 128 | 1024->6144 | 16.25 | 17.67 | 47.9% | 52.1% | quant/prologue dominated |
| Qwen3-0.6B | gate_up_proj | 512 | 1024->6144 | 16.27 | 46.06 | 26.1% | 73.9% | mixed |
| Qwen3-0.6B | down_proj | 1 | 3072->1024 | 15.94 | 13.93 | 53.4% | 46.6% | quant/prologue dominated |
| Qwen3-0.6B | down_proj | 32 | 3072->1024 | 16.11 | 17.56 | 47.8% | 52.2% | quant/prologue dominated |
| Qwen3-0.6B | down_proj | 128 | 3072->1024 | 16.02 | 17.72 | 47.5% | 52.5% | quant/prologue dominated |
| Qwen3-0.6B | down_proj | 512 | 3072->1024 | 16.18 | 28.20 | 36.5% | 63.5% | quant/prologue dominated |
| Qwen3-1.7B | q_proj | 1 | 2048->2048 | 15.82 | 13.66 | 53.7% | 46.3% | quant/prologue dominated |
| Qwen3-1.7B | q_proj | 32 | 2048->2048 | 16.05 | 17.69 | 47.6% | 52.4% | quant/prologue dominated |
| Qwen3-1.7B | q_proj | 128 | 2048->2048 | 16.14 | 17.67 | 47.7% | 52.3% | quant/prologue dominated |
| Qwen3-1.7B | q_proj | 512 | 2048->2048 | 16.29 | 27.32 | 37.4% | 62.6% | quant/prologue dominated |
| Qwen3-1.7B | kv_proj | 1 | 2048->2048 | 15.81 | 13.81 | 53.4% | 46.6% | quant/prologue dominated |
| Qwen3-1.7B | kv_proj | 32 | 2048->2048 | 16.14 | 17.68 | 47.7% | 52.3% | quant/prologue dominated |
| Qwen3-1.7B | kv_proj | 128 | 2048->2048 | 16.15 | 17.71 | 47.7% | 52.3% | quant/prologue dominated |
| Qwen3-1.7B | kv_proj | 512 | 2048->2048 | 16.08 | 27.34 | 37.0% | 63.0% | quant/prologue dominated |
| Qwen3-1.7B | o_proj | 1 | 2048->2048 | 15.79 | 13.75 | 53.4% | 46.6% | quant/prologue dominated |
| Qwen3-1.7B | o_proj | 32 | 2048->2048 | 16.14 | 17.71 | 47.7% | 52.3% | quant/prologue dominated |
| Qwen3-1.7B | o_proj | 128 | 2048->2048 | 16.19 | 17.83 | 47.6% | 52.4% | quant/prologue dominated |
| Qwen3-1.7B | o_proj | 512 | 2048->2048 | 16.16 | 27.32 | 37.2% | 62.8% | quant/prologue dominated |
| Qwen3-1.7B | gate_up_proj | 1 | 2048->12288 | 15.74 | 22.05 | 41.7% | 58.3% | quant/prologue dominated |
| Qwen3-1.7B | gate_up_proj | 32 | 2048->12288 | 16.13 | 23.10 | 41.1% | 58.9% | quant/prologue dominated |
| Qwen3-1.7B | gate_up_proj | 128 | 2048->12288 | 16.35 | 35.17 | 31.7% | 68.3% | mixed |
| Qwen3-1.7B | gate_up_proj | 512 | 2048->12288 | 15.88 | 113.42 | 12.3% | 87.7% | main fused kernel dominated |
| Qwen3-1.7B | down_proj | 1 | 6144->2048 | 16.16 | 16.08 | 50.1% | 49.9% | quant/prologue dominated |
| Qwen3-1.7B | down_proj | 32 | 6144->2048 | 16.03 | 25.03 | 39.0% | 61.0% | quant/prologue dominated |
| Qwen3-1.7B | down_proj | 128 | 6144->2048 | 16.35 | 33.98 | 32.5% | 67.5% | mixed |
| Qwen3-1.7B | down_proj | 512 | 6144->2048 | 16.83 | 70.65 | 19.2% | 80.8% | main fused kernel dominated |
| Qwen3-4B | q_proj | 1 | 2560->4096 | 15.94 | 14.08 | 53.1% | 46.9% | quant/prologue dominated |
| Qwen3-4B | q_proj | 32 | 2560->4096 | 16.15 | 17.73 | 47.7% | 52.3% | quant/prologue dominated |
| Qwen3-4B | q_proj | 128 | 2560->4096 | 16.24 | 27.26 | 37.3% | 62.7% | quant/prologue dominated |
| Qwen3-4B | q_proj | 512 | 2560->4096 | 16.35 | 49.76 | 24.7% | 75.3% | mixed |
| Qwen3-4B | kv_proj | 1 | 2560->2048 | 15.93 | 13.90 | 53.4% | 46.6% | quant/prologue dominated |
| Qwen3-4B | kv_proj | 32 | 2560->2048 | 16.02 | 17.74 | 47.5% | 52.5% | quant/prologue dominated |
| Qwen3-4B | kv_proj | 128 | 2560->2048 | 16.20 | 17.64 | 47.9% | 52.1% | quant/prologue dominated |
| Qwen3-4B | kv_proj | 512 | 2560->2048 | 16.16 | 33.82 | 32.3% | 67.7% | mixed |
| Qwen3-4B | o_proj | 1 | 4096->2560 | 15.98 | 13.92 | 53.5% | 46.5% | quant/prologue dominated |
| Qwen3-4B | o_proj | 32 | 4096->2560 | 16.04 | 21.25 | 43.0% | 57.0% | quant/prologue dominated |
| Qwen3-4B | o_proj | 128 | 4096->2560 | 16.07 | 29.90 | 35.0% | 65.0% | mixed |
| Qwen3-4B | o_proj | 512 | 4096->2560 | 16.36 | 68.31 | 19.3% | 80.7% | main fused kernel dominated |
| Qwen3-4B | gate_up_proj | 1 | 2560->19456 | 15.87 | 40.67 | 28.1% | 71.9% | mixed |
| Qwen3-4B | gate_up_proj | 32 | 2560->19456 | 15.93 | 38.53 | 29.2% | 70.8% | mixed |
| Qwen3-4B | gate_up_proj | 128 | 2560->19456 | 16.21 | 82.44 | 16.4% | 83.6% | main fused kernel dominated |
| Qwen3-4B | gate_up_proj | 512 | 2560->19456 | 16.47 | 221.70 | 6.9% | 93.1% | main fused kernel dominated |
| Qwen3-4B | down_proj | 1 | 9728->2560 | 19.37 | 30.40 | 38.9% | 61.1% | quant/prologue dominated |
| Qwen3-4B | down_proj | 32 | 9728->2560 | 21.28 | 36.69 | 36.7% | 63.3% | quant/prologue dominated |
| Qwen3-4B | down_proj | 128 | 9728->2560 | 21.53 | 56.80 | 27.5% | 72.5% | mixed |
| Qwen3-4B | down_proj | 512 | 9728->2560 | 24.24 | 168.93 | 12.5% | 87.5% | main fused kernel dominated |
| Qwen3-8B | q_proj | 1 | 4096->4096 | 16.27 | 16.66 | 49.4% | 50.6% | quant/prologue dominated |
| Qwen3-8B | q_proj | 32 | 4096->4096 | 16.16 | 22.47 | 41.8% | 58.2% | quant/prologue dominated |
| Qwen3-8B | q_proj | 128 | 4096->4096 | 16.01 | 34.94 | 31.4% | 68.6% | mixed |
| Qwen3-8B | q_proj | 512 | 4096->4096 | 16.08 | 73.52 | 17.9% | 82.1% | main fused kernel dominated |
| Qwen3-8B | kv_proj | 1 | 4096->2048 | 15.96 | 14.08 | 53.1% | 46.9% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 32 | 4096->2048 | 16.08 | 24.19 | 39.9% | 60.1% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 128 | 4096->2048 | 15.96 | 22.00 | 42.1% | 57.9% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 512 | 4096->2048 | 16.38 | 48.19 | 25.4% | 74.6% | mixed |
| Qwen3-8B | o_proj | 1 | 4096->4096 | 15.90 | 16.69 | 48.8% | 51.2% | quant/prologue dominated |
| Qwen3-8B | o_proj | 32 | 4096->4096 | 15.90 | 22.44 | 41.5% | 58.5% | quant/prologue dominated |
| Qwen3-8B | o_proj | 128 | 4096->4096 | 16.12 | 35.00 | 31.5% | 68.5% | mixed |
| Qwen3-8B | o_proj | 512 | 4096->4096 | 16.17 | 73.50 | 18.0% | 82.0% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 1 | 4096->24576 | 16.37 | 77.32 | 17.5% | 82.5% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 32 | 4096->24576 | 16.31 | 60.71 | 21.2% | 78.8% | mixed |
| Qwen3-8B | gate_up_proj | 128 | 4096->24576 | 16.43 | 135.16 | 10.8% | 89.2% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 512 | 4096->24576 | 16.52 | 445.28 | 3.6% | 96.4% | main fused kernel dominated |
| Qwen3-8B | down_proj | 1 | 12288->4096 | 24.77 | 45.67 | 35.2% | 64.8% | quant/prologue dominated |
| Qwen3-8B | down_proj | 32 | 12288->4096 | 24.72 | 46.38 | 34.8% | 65.2% | mixed |
| Qwen3-8B | down_proj | 128 | 12288->4096 | 25.54 | 80.47 | 24.1% | 75.9% | mixed |
| Qwen3-8B | down_proj | 512 | 12288->4096 | 51.27 | 252.95 | 16.9% | 83.1% | main fused kernel dominated |
