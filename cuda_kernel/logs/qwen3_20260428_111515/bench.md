# Qwen3 multi-scale kernel benchmark

- Timestamp: `20260428_111515`
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
| proj | shape | T=1 | T=8 | T=128 | T=512 | T=1024 |
|---|---|---:|---:|---:|---:|---:|
| q_proj | 1024->2048 | **1.16x** | **0.35x** | **0.35x** | **0.54x** | **0.76x** |
| kv_proj | 1024->2048 | **1.19x** | **0.34x** | **0.35x** | **0.54x** | **0.76x** |
| o_proj | 2048->1024 | **1.20x** | **0.45x** | **0.40x** | **0.45x** | **0.72x** |
| gate_up_proj | 1024->6144 | **1.59x** | **0.34x** | **0.59x** | **0.72x** | **0.82x** |
| down_proj | 3072->1024 | **0.88x** | **0.33x** | **0.28x** | **0.47x** | **0.69x** |

### Qwen3-1.7B
| proj | shape | T=1 | T=8 | T=128 | T=512 | T=1024 |
|---|---|---:|---:|---:|---:|---:|
| q_proj | 2048->2048 | **1.16x** | **0.44x** | **0.38x** | **0.60x** | **0.90x** |
| kv_proj | 2048->2048 | **1.15x** | **0.44x** | **0.38x** | **0.60x** | **0.90x** |
| o_proj | 2048->2048 | **1.15x** | **0.44x** | **0.38x** | **0.60x** | **0.90x** |
| gate_up_proj | 2048->12288 | **2.17x** | **0.75x** | **0.87x** | **1.02x** | **1.12x** |
| down_proj | 6144->2048 | **1.44x** | **0.18x** | **0.27x** | **0.65x** | **0.98x** |

### Qwen3-4B
| proj | shape | T=1 | T=8 | T=128 | T=512 | T=1024 |
|---|---|---:|---:|---:|---:|---:|
| q_proj | 2560->4096 | **1.87x** | **0.39x** | **0.43x** | **1.01x** | **1.06x** |
| kv_proj | 2560->2048 | **0.99x** | **0.39x** | **0.32x** | **0.58x** | **0.99x** |
| o_proj | 4096->2560 | **1.49x** | **0.26x** | **0.35x** | **0.76x** | **0.79x** |
| gate_up_proj | 2560->19456 | **2.16x** | **2.32x** | **1.14x** | **1.28x** | **1.22x** |
| down_proj | 9728->2560 | **1.45x** | **0.17x** | **0.30x** | **0.90x** | **0.83x** |

### Qwen3-8B
| proj | shape | T=1 | T=8 | T=128 | T=512 | T=1024 |
|---|---|---:|---:|---:|---:|---:|
| q_proj | 4096->4096 | **2.03x** | **0.31x** | **0.49x** | **1.11x** | **1.11x** |
| kv_proj | 4096->2048 | **1.38x** | **0.26x** | **0.31x** | **0.65x** | **1.06x** |
| o_proj | 4096->4096 | **1.24x** | **0.31x** | **0.49x** | **1.10x** | **1.11x** |
| gate_up_proj | 4096->24576 | **2.23x** | **3.32x** | **1.45x** | **1.42x** | **1.33x** |
| down_proj | 12288->4096 | **2.03x** | **0.83x** | **0.69x** | **1.18x** | **1.27x** |


## 2. End-to-end raw latencies (us)


### Qwen3-0.6B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 3072->1024 | 1 | 8.73 | 134.40 | 9.96 | 0.06x | 0.88x | 13.49x |
| down_proj | 3072->1024 | 8 | 11.92 | 135.56 | 35.87 | 0.09x | 0.33x | 3.78x |
| down_proj | 3072->1024 | 128 | 11.50 | 136.46 | 41.68 | 0.08x | 0.28x | 3.27x |
| down_proj | 3072->1024 | 512 | 23.36 | 136.06 | 49.76 | 0.17x | 0.47x | 2.73x |
| down_proj | 3072->1024 | 1024 | 46.95 | 148.41 | 68.01 | 0.32x | 0.69x | 2.18x |
| gate_up_proj | 1024->6144 | 1 | 15.17 | 134.92 | 9.51 | 0.11x | 1.59x | 14.19x |
| gate_up_proj | 1024->6144 | 8 | 9.15 | 134.57 | 26.76 | 0.07x | 0.34x | 5.03x |
| gate_up_proj | 1024->6144 | 128 | 15.82 | 135.47 | 26.78 | 0.12x | 0.59x | 5.06x |
| gate_up_proj | 1024->6144 | 512 | 44.01 | 135.36 | 60.74 | 0.33x | 0.72x | 2.23x |
| gate_up_proj | 1024->6144 | 1024 | 80.00 | 150.09 | 97.41 | 0.53x | 0.82x | 1.54x |
| kv_proj | 1024->2048 | 1 | 8.77 | 134.55 | 7.39 | 0.07x | 1.19x | 18.22x |
| kv_proj | 1024->2048 | 8 | 9.18 | 135.94 | 26.97 | 0.07x | 0.34x | 5.04x |
| kv_proj | 1024->2048 | 128 | 9.44 | 135.57 | 26.90 | 0.07x | 0.35x | 5.04x |
| kv_proj | 1024->2048 | 512 | 15.61 | 135.32 | 28.87 | 0.12x | 0.54x | 4.69x |
| kv_proj | 1024->2048 | 1024 | 29.78 | 151.37 | 39.26 | 0.20x | 0.76x | 3.86x |
| o_proj | 2048->1024 | 1 | 8.91 | 134.67 | 7.43 | 0.07x | 1.20x | 18.11x |
| o_proj | 2048->1024 | 8 | 11.92 | 134.53 | 26.62 | 0.09x | 0.45x | 5.05x |
| o_proj | 2048->1024 | 128 | 11.44 | 135.15 | 28.69 | 0.08x | 0.40x | 4.71x |
| o_proj | 2048->1024 | 512 | 16.68 | 134.42 | 36.69 | 0.12x | 0.45x | 3.66x |
| o_proj | 2048->1024 | 1024 | 34.08 | 148.37 | 47.27 | 0.23x | 0.72x | 3.14x |
| q_proj | 1024->2048 | 1 | 8.78 | 136.36 | 7.55 | 0.06x | 1.16x | 18.07x |
| q_proj | 1024->2048 | 8 | 9.21 | 135.54 | 26.53 | 0.07x | 0.35x | 5.11x |
| q_proj | 1024->2048 | 128 | 9.44 | 135.84 | 26.84 | 0.07x | 0.35x | 5.06x |
| q_proj | 1024->2048 | 512 | 15.53 | 135.14 | 28.79 | 0.11x | 0.54x | 4.69x |
| q_proj | 1024->2048 | 1024 | 29.82 | 149.81 | 39.22 | 0.20x | 0.76x | 3.82x |

### Qwen3-1.7B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 6144->2048 | 1 | 29.94 | 139.95 | 20.81 | 0.21x | 1.44x | 6.73x |
| down_proj | 6144->2048 | 8 | 12.21 | 145.20 | 66.22 | 0.08x | 0.18x | 2.19x |
| down_proj | 6144->2048 | 128 | 24.64 | 164.99 | 89.87 | 0.15x | 0.27x | 1.84x |
| down_proj | 6144->2048 | 512 | 78.04 | 232.28 | 119.74 | 0.34x | 0.65x | 1.94x |
| down_proj | 6144->2048 | 1024 | 158.65 | 331.76 | 162.34 | 0.48x | 0.98x | 2.04x |
| gate_up_proj | 2048->12288 | 1 | 55.23 | 134.44 | 25.48 | 0.41x | 2.17x | 5.28x |
| gate_up_proj | 2048->12288 | 8 | 20.77 | 135.12 | 27.64 | 0.15x | 0.75x | 4.89x |
| gate_up_proj | 2048->12288 | 128 | 46.74 | 136.09 | 53.63 | 0.34x | 0.87x | 2.54x |
| gate_up_proj | 2048->12288 | 512 | 156.94 | 270.15 | 153.68 | 0.58x | 1.02x | 1.76x |
| gate_up_proj | 2048->12288 | 1024 | 327.49 | 494.90 | 293.17 | 0.66x | 1.12x | 1.69x |
| kv_proj | 2048->2048 | 1 | 8.55 | 135.08 | 7.45 | 0.06x | 1.15x | 18.12x |
| kv_proj | 2048->2048 | 8 | 11.88 | 135.07 | 26.91 | 0.09x | 0.44x | 5.02x |
| kv_proj | 2048->2048 | 128 | 12.87 | 134.77 | 33.74 | 0.10x | 0.38x | 3.99x |
| kv_proj | 2048->2048 | 512 | 27.88 | 135.17 | 46.58 | 0.21x | 0.60x | 2.90x |
| kv_proj | 2048->2048 | 1024 | 55.95 | 148.69 | 62.01 | 0.38x | 0.90x | 2.40x |
| o_proj | 2048->2048 | 1 | 8.57 | 133.58 | 7.44 | 0.06x | 1.15x | 17.94x |
| o_proj | 2048->2048 | 8 | 11.99 | 133.89 | 26.99 | 0.09x | 0.44x | 4.96x |
| o_proj | 2048->2048 | 128 | 12.85 | 133.65 | 33.76 | 0.10x | 0.38x | 3.96x |
| o_proj | 2048->2048 | 512 | 27.87 | 135.02 | 46.56 | 0.21x | 0.60x | 2.90x |
| o_proj | 2048->2048 | 1024 | 55.97 | 149.62 | 61.90 | 0.37x | 0.90x | 2.42x |
| q_proj | 2048->2048 | 1 | 8.63 | 135.76 | 7.43 | 0.06x | 1.16x | 18.26x |
| q_proj | 2048->2048 | 8 | 11.96 | 135.80 | 27.18 | 0.09x | 0.44x | 5.00x |
| q_proj | 2048->2048 | 128 | 12.87 | 134.60 | 33.76 | 0.10x | 0.38x | 3.99x |
| q_proj | 2048->2048 | 512 | 27.85 | 133.81 | 46.44 | 0.21x | 0.60x | 2.88x |
| q_proj | 2048->2048 | 1024 | 55.95 | 149.56 | 61.90 | 0.37x | 0.90x | 2.42x |

### Qwen3-4B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 9728->2560 | 1 | 55.65 | 223.35 | 38.36 | 0.25x | 1.45x | 5.82x |
| down_proj | 9728->2560 | 8 | 19.89 | 231.96 | 116.81 | 0.09x | 0.17x | 1.99x |
| down_proj | 9728->2560 | 128 | 45.43 | 276.67 | 152.73 | 0.16x | 0.30x | 1.81x |
| down_proj | 9728->2560 | 512 | 185.21 | 491.87 | 205.54 | 0.38x | 0.90x | 2.39x |
| down_proj | 9728->2560 | 1024 | 331.33 | 693.82 | 400.30 | 0.48x | 0.83x | 1.73x |
| gate_up_proj | 2560->19456 | 1 | 106.20 | 133.74 | 49.12 | 0.79x | 2.16x | 2.72x |
| gate_up_proj | 2560->19456 | 8 | 115.50 | 134.59 | 49.75 | 0.86x | 2.32x | 2.71x |
| gate_up_proj | 2560->19456 | 128 | 136.15 | 189.93 | 119.48 | 0.72x | 1.14x | 1.59x |
| gate_up_proj | 2560->19456 | 512 | 375.29 | 510.50 | 292.83 | 0.74x | 1.28x | 1.74x |
| gate_up_proj | 2560->19456 | 1024 | 680.12 | 943.97 | 557.02 | 0.72x | 1.22x | 1.69x |
| kv_proj | 2560->2048 | 1 | 9.90 | 135.84 | 10.01 | 0.07x | 0.99x | 13.56x |
| kv_proj | 2560->2048 | 8 | 11.99 | 135.44 | 30.65 | 0.09x | 0.39x | 4.42x |
| kv_proj | 2560->2048 | 128 | 14.56 | 135.34 | 45.78 | 0.11x | 0.32x | 2.96x |
| kv_proj | 2560->2048 | 512 | 34.44 | 135.37 | 59.80 | 0.25x | 0.58x | 2.26x |
| kv_proj | 2560->2048 | 1024 | 72.80 | 149.79 | 73.57 | 0.49x | 0.99x | 2.04x |
| o_proj | 4096->2560 | 1 | 24.25 | 134.79 | 16.25 | 0.18x | 1.49x | 8.29x |
| o_proj | 4096->2560 | 8 | 12.09 | 134.07 | 46.45 | 0.09x | 0.26x | 2.89x |
| o_proj | 4096->2560 | 128 | 23.06 | 135.12 | 65.57 | 0.17x | 0.35x | 2.06x |
| o_proj | 4096->2560 | 512 | 74.16 | 213.83 | 97.31 | 0.35x | 0.76x | 2.20x |
| o_proj | 4096->2560 | 1024 | 142.29 | 298.16 | 180.43 | 0.48x | 0.79x | 1.65x |
| q_proj | 2560->4096 | 1 | 24.72 | 134.42 | 13.22 | 0.18x | 1.87x | 10.17x |
| q_proj | 2560->4096 | 8 | 12.03 | 136.01 | 30.76 | 0.09x | 0.39x | 4.42x |
| q_proj | 2560->4096 | 128 | 20.12 | 135.94 | 46.53 | 0.15x | 0.43x | 2.92x |
| q_proj | 2560->4096 | 512 | 72.85 | 147.64 | 72.13 | 0.49x | 1.01x | 2.05x |
| q_proj | 2560->4096 | 1024 | 143.58 | 245.48 | 135.82 | 0.58x | 1.06x | 1.81x |

### Qwen3-8B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 12288->4096 | 1 | 109.70 | 276.67 | 53.98 | 0.40x | 2.03x | 5.13x |
| down_proj | 12288->4096 | 8 | 118.30 | 287.88 | 142.23 | 0.41x | 0.83x | 2.02x |
| down_proj | 12288->4096 | 128 | 133.84 | 353.67 | 194.44 | 0.38x | 0.69x | 1.82x |
| down_proj | 12288->4096 | 512 | 359.49 | 652.95 | 304.11 | 0.55x | 1.18x | 2.15x |
| down_proj | 12288->4096 | 1024 | 715.37 | 1119.07 | 563.74 | 0.64x | 1.27x | 1.99x |
| gate_up_proj | 4096->24576 | 1 | 212.49 | 135.72 | 95.47 | 1.57x | 2.23x | 1.42x |
| gate_up_proj | 4096->24576 | 8 | 221.35 | 136.16 | 66.61 | 1.63x | 3.32x | 2.04x |
| gate_up_proj | 4096->24576 | 128 | 263.71 | 311.50 | 182.17 | 0.85x | 1.45x | 1.71x |
| gate_up_proj | 4096->24576 | 512 | 750.82 | 994.38 | 527.63 | 0.76x | 1.42x | 1.88x |
| gate_up_proj | 4096->24576 | 1024 | 1363.17 | 1920.58 | 1023.32 | 0.71x | 1.33x | 1.88x |
| kv_proj | 4096->2048 | 1 | 19.21 | 135.64 | 13.92 | 0.14x | 1.38x | 9.75x |
| kv_proj | 4096->2048 | 8 | 12.04 | 136.61 | 45.47 | 0.09x | 0.26x | 3.00x |
| kv_proj | 4096->2048 | 128 | 19.43 | 135.33 | 62.47 | 0.14x | 0.31x | 2.17x |
| kv_proj | 4096->2048 | 512 | 53.10 | 157.31 | 81.70 | 0.34x | 0.65x | 1.93x |
| kv_proj | 4096->2048 | 1024 | 114.94 | 225.14 | 108.17 | 0.51x | 1.06x | 2.08x |
| o_proj | 4096->4096 | 1 | 23.50 | 134.55 | 18.91 | 0.17x | 1.24x | 7.11x |
| o_proj | 4096->4096 | 8 | 14.66 | 135.48 | 47.72 | 0.11x | 0.31x | 2.84x |
| o_proj | 4096->4096 | 128 | 30.90 | 135.09 | 63.16 | 0.23x | 0.49x | 2.14x |
| o_proj | 4096->4096 | 512 | 113.99 | 222.59 | 103.40 | 0.51x | 1.10x | 2.15x |
| o_proj | 4096->4096 | 1024 | 212.91 | 379.90 | 191.79 | 0.56x | 1.11x | 1.98x |
| q_proj | 4096->4096 | 1 | 38.37 | 135.15 | 18.88 | 0.28x | 2.03x | 7.16x |
| q_proj | 4096->4096 | 8 | 14.62 | 135.68 | 47.66 | 0.11x | 0.31x | 2.85x |
| q_proj | 4096->4096 | 128 | 30.94 | 136.64 | 63.17 | 0.23x | 0.49x | 2.16x |
| q_proj | 4096->4096 | 512 | 115.03 | 222.65 | 103.43 | 0.52x | 1.11x | 2.15x |
| q_proj | 4096->4096 | 1024 | 212.62 | 379.79 | 191.77 | 0.56x | 1.11x | 1.98x |


## 3. Sub-kernel breakdown (us)


### Qwen3-0.6B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 43.23 | 14.37 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 8.75 | 69.56 | 10.89 | 0.13x | 0.80x | 6.39x |
| down_proj | 1 | sparse_gemm | 8.82 | 68.20 | 18.02 | 0.13x | 0.49x | 3.78x |
| down_proj | 1 | fused_dense_sparse | 8.83 | 81.87 | 12.17 | 0.11x | 0.73x | 6.73x |
| down_proj | 8 | activation_quant |  -  | 44.74 | 14.58 |  -  |  -  |  -  |
| down_proj | 8 | dense_gemm | 11.90 | 69.14 | 30.44 | 0.17x | 0.39x | 2.27x |
| down_proj | 8 | sparse_gemm | 11.98 | 69.43 | 18.00 | 0.17x | 0.67x | 3.86x |
| down_proj | 8 | fused_dense_sparse | 11.93 | 81.27 | 26.70 | 0.15x | 0.45x | 3.04x |
| down_proj | 128 | activation_quant |  -  | 44.62 | 14.49 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 11.47 | 68.91 | 30.71 | 0.17x | 0.37x | 2.24x |
| down_proj | 128 | sparse_gemm | 11.42 | 69.25 | 18.07 | 0.16x | 0.63x | 3.83x |
| down_proj | 128 | fused_dense_sparse | 11.48 | 82.44 | 32.43 | 0.14x | 0.35x | 2.54x |
| down_proj | 512 | activation_quant |  -  | 44.51 | 14.60 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 23.35 | 69.89 | 38.32 | 0.33x | 0.61x | 1.82x |
| down_proj | 512 | sparse_gemm | 23.30 | 69.01 | 18.21 | 0.34x | 1.28x | 3.79x |
| down_proj | 512 | fused_dense_sparse | 23.30 | 82.47 | 39.78 | 0.28x | 0.59x | 2.07x |
| down_proj | 1024 | activation_quant |  -  | 62.97 | 14.79 |  -  |  -  |  -  |
| down_proj | 1024 | dense_gemm | 46.57 | 70.90 | 46.45 | 0.66x | 1.00x | 1.53x |
| down_proj | 1024 | sparse_gemm | 46.81 | 68.74 | 18.18 | 0.68x | 2.57x | 3.78x |
| down_proj | 1024 | fused_dense_sparse | 46.71 | 81.08 | 53.99 | 0.58x | 0.87x | 1.50x |
| gate_up_proj | 1 | activation_quant |  -  | 43.28 | 14.44 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 8.68 | 69.96 | 10.78 | 0.12x | 0.81x | 6.49x |
| gate_up_proj | 1 | sparse_gemm | 15.10 | 68.84 | 17.92 | 0.22x | 0.84x | 3.84x |
| gate_up_proj | 1 | fused_dense_sparse | 15.14 | 82.12 | 12.08 | 0.18x | 1.25x | 6.80x |
| gate_up_proj | 8 | activation_quant |  -  | 44.68 | 14.51 |  -  |  -  |  -  |
| gate_up_proj | 8 | dense_gemm | 9.18 | 70.06 | 10.82 | 0.13x | 0.85x | 6.47x |
| gate_up_proj | 8 | sparse_gemm | 9.25 | 69.35 | 17.86 | 0.13x | 0.52x | 3.88x |
| gate_up_proj | 8 | fused_dense_sparse | 9.16 | 80.65 | 11.99 | 0.11x | 0.76x | 6.73x |
| gate_up_proj | 128 | activation_quant |  -  | 44.39 | 14.53 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 15.83 | 70.44 | 19.27 | 0.22x | 0.82x | 3.66x |
| gate_up_proj | 128 | sparse_gemm | 15.82 | 68.74 | 17.86 | 0.23x | 0.89x | 3.85x |
| gate_up_proj | 128 | fused_dense_sparse | 15.82 | 81.68 | 19.80 | 0.19x | 0.80x | 4.12x |
| gate_up_proj | 512 | activation_quant |  -  | 44.72 | 14.55 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 44.67 | 70.13 | 51.71 | 0.64x | 0.86x | 1.36x |
| gate_up_proj | 512 | sparse_gemm | 43.88 | 68.76 | 30.42 | 0.64x | 1.44x | 2.26x |
| gate_up_proj | 512 | fused_dense_sparse | 43.80 | 82.09 | 55.67 | 0.53x | 0.79x | 1.47x |
| gate_up_proj | 1024 | activation_quant |  -  | 61.51 | 14.79 |  -  |  -  |  -  |
| gate_up_proj | 1024 | dense_gemm | 79.78 | 112.63 | 85.48 | 0.71x | 0.93x | 1.32x |
| gate_up_proj | 1024 | sparse_gemm | 79.73 | 67.57 | 54.73 | 1.18x | 1.46x | 1.23x |
| gate_up_proj | 1024 | fused_dense_sparse | 79.47 | 126.21 | 91.40 | 0.63x | 0.87x | 1.38x |
| kv_proj | 1 | activation_quant |  -  | 43.53 | 14.36 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 8.70 | 68.99 | 10.77 | 0.13x | 0.81x | 6.40x |
| kv_proj | 1 | sparse_gemm | 8.72 | 68.36 | 17.96 | 0.13x | 0.49x | 3.81x |
| kv_proj | 1 | fused_dense_sparse | 8.73 | 80.91 | 12.03 | 0.11x | 0.73x | 6.72x |
| kv_proj | 8 | activation_quant |  -  | 44.87 | 14.68 |  -  |  -  |  -  |
| kv_proj | 8 | dense_gemm | 9.25 | 69.47 | 10.89 | 0.13x | 0.85x | 6.38x |
| kv_proj | 8 | sparse_gemm | 9.20 | 68.83 | 18.09 | 0.13x | 0.51x | 3.80x |
| kv_proj | 8 | fused_dense_sparse | 9.22 | 81.40 | 11.97 | 0.11x | 0.77x | 6.80x |
| kv_proj | 128 | activation_quant |  -  | 44.29 | 14.52 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 9.47 | 70.32 | 14.13 | 0.13x | 0.67x | 4.98x |
| kv_proj | 128 | sparse_gemm | 9.46 | 68.81 | 18.00 | 0.14x | 0.53x | 3.82x |
| kv_proj | 128 | fused_dense_sparse | 9.46 | 81.85 | 15.69 | 0.12x | 0.60x | 5.22x |
| kv_proj | 512 | activation_quant |  -  | 44.56 | 14.64 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 15.62 | 70.00 | 21.03 | 0.22x | 0.74x | 3.33x |
| kv_proj | 512 | sparse_gemm | 15.62 | 68.54 | 18.12 | 0.23x | 0.86x | 3.78x |
| kv_proj | 512 | fused_dense_sparse | 15.63 | 81.23 | 23.92 | 0.19x | 0.65x | 3.40x |
| kv_proj | 1024 | activation_quant |  -  | 62.56 | 14.69 |  -  |  -  |  -  |
| kv_proj | 1024 | dense_gemm | 29.78 | 70.38 | 30.91 | 0.42x | 0.96x | 2.28x |
| kv_proj | 1024 | sparse_gemm | 29.82 | 69.31 | 21.28 | 0.43x | 1.40x | 3.26x |
| kv_proj | 1024 | fused_dense_sparse | 29.77 | 81.93 | 33.42 | 0.36x | 0.89x | 2.45x |
| o_proj | 1 | activation_quant |  -  | 44.28 | 14.62 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 8.82 | 68.04 | 10.84 | 0.13x | 0.81x | 6.27x |
| o_proj | 1 | sparse_gemm | 8.85 | 66.99 | 17.96 | 0.13x | 0.49x | 3.73x |
| o_proj | 1 | fused_dense_sparse | 8.86 | 81.14 | 12.06 | 0.11x | 0.73x | 6.73x |
| o_proj | 8 | activation_quant |  -  | 44.03 | 14.58 |  -  |  -  |  -  |
| o_proj | 8 | dense_gemm | 11.89 | 68.67 | 19.92 | 0.17x | 0.60x | 3.45x |
| o_proj | 8 | sparse_gemm | 12.03 | 69.55 | 17.86 | 0.17x | 0.67x | 3.89x |
| o_proj | 8 | fused_dense_sparse | 11.98 | 82.04 | 18.37 | 0.15x | 0.65x | 4.47x |
| o_proj | 128 | activation_quant |  -  | 43.42 | 14.64 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 11.51 | 67.57 | 20.12 | 0.17x | 0.57x | 3.36x |
| o_proj | 128 | sparse_gemm | 11.43 | 69.16 | 18.11 | 0.17x | 0.63x | 3.82x |
| o_proj | 128 | fused_dense_sparse | 11.48 | 82.17 | 21.91 | 0.14x | 0.52x | 3.75x |
| o_proj | 512 | activation_quant |  -  | 43.21 | 14.42 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 16.71 | 68.51 | 26.53 | 0.24x | 0.63x | 2.58x |
| o_proj | 512 | sparse_gemm | 16.66 | 68.08 | 18.16 | 0.24x | 0.92x | 3.75x |
| o_proj | 512 | fused_dense_sparse | 16.67 | 81.30 | 29.05 | 0.21x | 0.57x | 2.80x |
| o_proj | 1024 | activation_quant |  -  | 61.93 | 14.60 |  -  |  -  |  -  |
| o_proj | 1024 | dense_gemm | 34.02 | 69.43 | 33.27 | 0.49x | 1.02x | 2.09x |
| o_proj | 1024 | sparse_gemm | 34.04 | 68.70 | 17.86 | 0.50x | 1.91x | 3.85x |
| o_proj | 1024 | fused_dense_sparse | 34.10 | 81.83 | 37.92 | 0.42x | 0.90x | 2.16x |
| q_proj | 1 | activation_quant |  -  | 45.26 | 14.59 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 8.91 | 69.68 | 10.91 | 0.13x | 0.82x | 6.39x |
| q_proj | 1 | sparse_gemm | 8.87 | 69.35 | 18.16 | 0.13x | 0.49x | 3.82x |
| q_proj | 1 | fused_dense_sparse | 8.90 | 83.21 | 12.13 | 0.11x | 0.73x | 6.86x |
| q_proj | 8 | activation_quant |  -  | 43.65 | 14.60 |  -  |  -  |  -  |
| q_proj | 8 | dense_gemm | 9.19 | 69.78 | 10.95 | 0.13x | 0.84x | 6.37x |
| q_proj | 8 | sparse_gemm | 9.28 | 69.63 | 17.99 | 0.13x | 0.52x | 3.87x |
| q_proj | 8 | fused_dense_sparse | 9.18 | 81.58 | 12.17 | 0.11x | 0.75x | 6.71x |
| q_proj | 128 | activation_quant |  -  | 43.07 | 14.36 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 9.44 | 69.14 | 14.07 | 0.14x | 0.67x | 4.91x |
| q_proj | 128 | sparse_gemm | 9.45 | 68.94 | 18.10 | 0.14x | 0.52x | 3.81x |
| q_proj | 128 | fused_dense_sparse | 9.45 | 81.26 | 15.69 | 0.12x | 0.60x | 5.18x |
| q_proj | 512 | activation_quant |  -  | 43.03 | 14.58 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 15.48 | 69.09 | 21.02 | 0.22x | 0.74x | 3.29x |
| q_proj | 512 | sparse_gemm | 15.53 | 68.65 | 18.15 | 0.23x | 0.86x | 3.78x |
| q_proj | 512 | fused_dense_sparse | 15.54 | 81.54 | 23.88 | 0.19x | 0.65x | 3.42x |
| q_proj | 1024 | activation_quant |  -  | 62.30 | 14.68 |  -  |  -  |  -  |
| q_proj | 1024 | dense_gemm | 29.77 | 70.18 | 30.92 | 0.42x | 0.96x | 2.27x |
| q_proj | 1024 | sparse_gemm | 29.78 | 68.82 | 21.29 | 0.43x | 1.40x | 3.23x |
| q_proj | 1024 | fused_dense_sparse | 29.77 | 81.28 | 33.43 | 0.37x | 0.89x | 2.43x |

### Qwen3-1.7B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 59.57 | 14.54 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 13.88 | 71.80 | 15.64 | 0.19x | 0.89x | 4.59x |
| down_proj | 1 | sparse_gemm | 29.93 | 67.01 | 17.94 | 0.45x | 1.67x | 3.74x |
| down_proj | 1 | fused_dense_sparse | 29.94 | 82.74 | 16.29 | 0.36x | 1.84x | 5.08x |
| down_proj | 8 | activation_quant |  -  | 66.57 | 14.49 |  -  |  -  |  -  |
| down_proj | 8 | dense_gemm | 12.24 | 71.10 | 60.49 | 0.17x | 0.20x | 1.18x |
| down_proj | 8 | sparse_gemm | 12.20 | 69.41 | 18.14 | 0.18x | 0.67x | 3.83x |
| down_proj | 8 | fused_dense_sparse | 12.24 | 81.98 | 52.07 | 0.15x | 0.23x | 1.57x |
| down_proj | 128 | activation_quant |  -  | 85.01 | 14.72 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 24.63 | 72.66 | 74.05 | 0.34x | 0.33x | 0.98x |
| down_proj | 128 | sparse_gemm | 24.67 | 68.85 | 18.15 | 0.36x | 1.36x | 3.79x |
| down_proj | 128 | fused_dense_sparse | 24.65 | 82.19 | 75.86 | 0.30x | 0.32x | 1.08x |
| down_proj | 512 | activation_quant |  -  | 84.91 | 16.41 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 77.50 | 130.41 | 88.66 | 0.59x | 0.87x | 1.47x |
| down_proj | 512 | sparse_gemm | 77.78 | 68.12 | 19.16 | 1.14x | 4.06x | 3.56x |
| down_proj | 512 | fused_dense_sparse | 77.50 | 144.09 | 103.29 | 0.54x | 0.75x | 1.39x |
| down_proj | 1024 | activation_quant |  -  | 88.50 | 31.69 |  -  |  -  |  -  |
| down_proj | 1024 | dense_gemm | 156.86 | 217.85 | 173.34 | 0.72x | 0.90x | 1.26x |
| down_proj | 1024 | sparse_gemm | 158.17 | 67.48 | 26.90 | 2.34x | 5.88x | 2.51x |
| down_proj | 1024 | fused_dense_sparse | 157.35 | 242.40 | 130.80 | 0.65x | 1.20x | 1.85x |
| gate_up_proj | 1 | activation_quant |  -  | 43.94 | 14.18 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 42.11 | 68.63 | 20.80 | 0.61x | 2.02x | 3.30x |
| gate_up_proj | 1 | sparse_gemm | 55.20 | 67.81 | 18.16 | 0.81x | 3.04x | 3.73x |
| gate_up_proj | 1 | fused_dense_sparse | 55.24 | 81.17 | 22.33 | 0.68x | 2.47x | 3.63x |
| gate_up_proj | 8 | activation_quant |  -  | 45.33 | 14.39 |  -  |  -  |  -  |
| gate_up_proj | 8 | dense_gemm | 20.78 | 70.12 | 19.95 | 0.30x | 1.04x | 3.52x |
| gate_up_proj | 8 | sparse_gemm | 20.76 | 69.22 | 17.99 | 0.30x | 1.15x | 3.85x |
| gate_up_proj | 8 | fused_dense_sparse | 20.76 | 82.01 | 21.04 | 0.25x | 0.99x | 3.90x |
| gate_up_proj | 128 | activation_quant |  -  | 44.35 | 14.71 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 46.38 | 73.30 | 41.65 | 0.63x | 1.11x | 1.76x |
| gate_up_proj | 128 | sparse_gemm | 46.85 | 68.66 | 20.41 | 0.68x | 2.30x | 3.36x |
| gate_up_proj | 128 | fused_dense_sparse | 46.56 | 85.30 | 46.94 | 0.55x | 0.99x | 1.82x |
| gate_up_proj | 512 | activation_quant |  -  | 43.84 | 14.52 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 154.77 | 217.85 | 132.51 | 0.71x | 1.17x | 1.64x |
| gate_up_proj | 512 | sparse_gemm | 156.70 | 68.50 | 57.14 | 2.29x | 2.74x | 1.20x |
| gate_up_proj | 512 | fused_dense_sparse | 155.35 | 242.39 | 146.09 | 0.64x | 1.06x | 1.66x |
| gate_up_proj | 1024 | activation_quant |  -  | 61.19 | 14.41 |  -  |  -  |  -  |
| gate_up_proj | 1024 | dense_gemm | 326.15 | 431.79 | 257.57 | 0.76x | 1.27x | 1.68x |
| gate_up_proj | 1024 | sparse_gemm | 327.66 | 67.87 | 106.87 | 4.83x | 3.07x | 0.64x |
| gate_up_proj | 1024 | fused_dense_sparse | 324.21 | 467.27 | 283.37 | 0.69x | 1.14x | 1.65x |
| kv_proj | 1 | activation_quant |  -  | 44.74 | 14.47 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 8.65 | 68.49 | 10.57 | 0.13x | 0.82x | 6.48x |
| kv_proj | 1 | sparse_gemm | 8.71 | 67.30 | 17.87 | 0.13x | 0.49x | 3.77x |
| kv_proj | 1 | fused_dense_sparse | 8.70 | 81.00 | 11.93 | 0.11x | 0.73x | 6.79x |
| kv_proj | 8 | activation_quant |  -  | 44.56 | 14.57 |  -  |  -  |  -  |
| kv_proj | 8 | dense_gemm | 11.95 | 70.03 | 19.96 | 0.17x | 0.60x | 3.51x |
| kv_proj | 8 | sparse_gemm | 11.88 | 69.10 | 18.10 | 0.17x | 0.66x | 3.82x |
| kv_proj | 8 | fused_dense_sparse | 12.00 | 80.96 | 18.93 | 0.15x | 0.63x | 4.28x |
| kv_proj | 128 | activation_quant |  -  | 43.91 | 14.52 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 12.84 | 69.51 | 24.83 | 0.18x | 0.52x | 2.80x |
| kv_proj | 128 | sparse_gemm | 12.85 | 68.23 | 18.05 | 0.19x | 0.71x | 3.78x |
| kv_proj | 128 | fused_dense_sparse | 12.85 | 81.90 | 27.06 | 0.16x | 0.47x | 3.03x |
| kv_proj | 512 | activation_quant |  -  | 44.93 | 14.81 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 27.81 | 68.63 | 33.63 | 0.41x | 0.83x | 2.04x |
| kv_proj | 512 | sparse_gemm | 27.79 | 67.38 | 17.97 | 0.41x | 1.55x | 3.75x |
| kv_proj | 512 | fused_dense_sparse | 27.89 | 81.19 | 39.16 | 0.34x | 0.71x | 2.07x |
| kv_proj | 1024 | activation_quant |  -  | 61.77 | 14.57 |  -  |  -  |  -  |
| kv_proj | 1024 | dense_gemm | 55.60 | 75.41 | 47.65 | 0.74x | 1.17x | 1.58x |
| kv_proj | 1024 | sparse_gemm | 55.83 | 68.02 | 22.40 | 0.82x | 2.49x | 3.04x |
| kv_proj | 1024 | fused_dense_sparse | 55.86 | 83.79 | 52.52 | 0.67x | 1.06x | 1.60x |
| o_proj | 1 | activation_quant |  -  | 43.13 | 14.35 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 8.65 | 67.86 | 10.50 | 0.13x | 0.82x | 6.47x |
| o_proj | 1 | sparse_gemm | 8.61 | 67.52 | 17.75 | 0.13x | 0.49x | 3.80x |
| o_proj | 1 | fused_dense_sparse | 8.63 | 80.86 | 11.85 | 0.11x | 0.73x | 6.83x |
| o_proj | 8 | activation_quant |  -  | 43.45 | 14.57 |  -  |  -  |  -  |
| o_proj | 8 | dense_gemm | 11.99 | 69.51 | 19.94 | 0.17x | 0.60x | 3.49x |
| o_proj | 8 | sparse_gemm | 12.39 | 68.46 | 18.00 | 0.18x | 0.69x | 3.80x |
| o_proj | 8 | fused_dense_sparse | 11.92 | 81.09 | 18.92 | 0.15x | 0.63x | 4.29x |
| o_proj | 128 | activation_quant |  -  | 44.71 | 14.52 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 12.83 | 69.34 | 24.84 | 0.19x | 0.52x | 2.79x |
| o_proj | 128 | sparse_gemm | 12.86 | 67.99 | 17.95 | 0.19x | 0.72x | 3.79x |
| o_proj | 128 | fused_dense_sparse | 12.84 | 81.30 | 27.06 | 0.16x | 0.47x | 3.00x |
| o_proj | 512 | activation_quant |  -  | 44.43 | 14.63 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 27.78 | 69.48 | 33.60 | 0.40x | 0.83x | 2.07x |
| o_proj | 512 | sparse_gemm | 27.79 | 68.10 | 18.16 | 0.41x | 1.53x | 3.75x |
| o_proj | 512 | fused_dense_sparse | 27.89 | 81.59 | 39.16 | 0.34x | 0.71x | 2.08x |
| o_proj | 1024 | activation_quant |  -  | 61.64 | 14.82 |  -  |  -  |  -  |
| o_proj | 1024 | dense_gemm | 55.65 | 75.76 | 47.94 | 0.73x | 1.16x | 1.58x |
| o_proj | 1024 | sparse_gemm | 55.69 | 67.66 | 22.41 | 0.82x | 2.49x | 3.02x |
| o_proj | 1024 | fused_dense_sparse | 55.72 | 83.80 | 52.55 | 0.66x | 1.06x | 1.59x |
| q_proj | 1 | activation_quant |  -  | 43.06 | 14.42 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 8.53 | 69.05 | 10.61 | 0.12x | 0.80x | 6.51x |
| q_proj | 1 | sparse_gemm | 8.68 | 67.84 | 17.83 | 0.13x | 0.49x | 3.81x |
| q_proj | 1 | fused_dense_sparse | 8.64 | 81.43 | 11.93 | 0.11x | 0.72x | 6.83x |
| q_proj | 8 | activation_quant |  -  | 44.76 | 14.60 |  -  |  -  |  -  |
| q_proj | 8 | dense_gemm | 11.92 | 71.27 | 19.93 | 0.17x | 0.60x | 3.58x |
| q_proj | 8 | sparse_gemm | 11.92 | 69.70 | 18.13 | 0.17x | 0.66x | 3.84x |
| q_proj | 8 | fused_dense_sparse | 11.88 | 82.01 | 18.89 | 0.14x | 0.63x | 4.34x |
| q_proj | 128 | activation_quant |  -  | 45.14 | 14.22 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 12.88 | 69.45 | 24.83 | 0.19x | 0.52x | 2.80x |
| q_proj | 128 | sparse_gemm | 12.86 | 68.97 | 18.08 | 0.19x | 0.71x | 3.81x |
| q_proj | 128 | fused_dense_sparse | 12.88 | 81.55 | 27.04 | 0.16x | 0.48x | 3.02x |
| q_proj | 512 | activation_quant |  -  | 44.60 | 14.42 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 27.77 | 68.90 | 33.69 | 0.40x | 0.82x | 2.05x |
| q_proj | 512 | sparse_gemm | 27.84 | 67.64 | 18.26 | 0.41x | 1.52x | 3.70x |
| q_proj | 512 | fused_dense_sparse | 27.82 | 81.93 | 39.08 | 0.34x | 0.71x | 2.10x |
| q_proj | 1024 | activation_quant |  -  | 61.36 | 14.81 |  -  |  -  |  -  |
| q_proj | 1024 | dense_gemm | 55.61 | 75.29 | 47.51 | 0.74x | 1.17x | 1.58x |
| q_proj | 1024 | sparse_gemm | 55.88 | 67.92 | 22.34 | 0.82x | 2.50x | 3.04x |
| q_proj | 1024 | fused_dense_sparse | 55.67 | 83.72 | 52.41 | 0.66x | 1.06x | 1.60x |

### Qwen3-4B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 93.29 | 19.54 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 39.76 | 115.26 | 29.50 | 0.34x | 1.35x | 3.91x |
| down_proj | 1 | sparse_gemm | 55.67 | 68.90 | 18.01 | 0.81x | 3.09x | 3.83x |
| down_proj | 1 | fused_dense_sparse | 55.64 | 131.94 | 30.69 | 0.42x | 1.81x | 4.30x |
| down_proj | 8 | activation_quant |  -  | 104.73 | 21.46 |  -  |  -  |  -  |
| down_proj | 8 | dense_gemm | 19.87 | 112.61 | 91.62 | 0.18x | 0.22x | 1.23x |
| down_proj | 8 | sparse_gemm | 19.92 | 70.45 | 18.24 | 0.28x | 1.09x | 3.86x |
| down_proj | 8 | fused_dense_sparse | 19.87 | 127.97 | 95.86 | 0.16x | 0.21x | 1.34x |
| down_proj | 128 | activation_quant |  -  | 131.09 | 21.70 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 45.42 | 134.72 | 110.39 | 0.34x | 0.41x | 1.22x |
| down_proj | 128 | sparse_gemm | 45.41 | 69.18 | 20.10 | 0.66x | 2.26x | 3.44x |
| down_proj | 128 | fused_dense_sparse | 45.42 | 145.44 | 131.01 | 0.31x | 0.35x | 1.11x |
| down_proj | 512 | activation_quant |  -  | 131.30 | 24.87 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 184.50 | 331.48 | 250.45 | 0.56x | 0.74x | 1.32x |
| down_proj | 512 | sparse_gemm | 185.80 | 68.82 | 25.10 | 2.70x | 7.40x | 2.74x |
| down_proj | 512 | fused_dense_sparse | 185.53 | 359.78 | 181.61 | 0.52x | 1.02x | 1.98x |
| down_proj | 1024 | activation_quant |  -  | 136.98 | 47.24 |  -  |  -  |  -  |
| down_proj | 1024 | dense_gemm | 334.11 | 522.42 | 377.75 | 0.64x | 0.88x | 1.38x |
| down_proj | 1024 | sparse_gemm | 335.27 | 67.51 | 39.38 | 4.97x | 8.51x | 1.71x |
| down_proj | 1024 | fused_dense_sparse | 334.76 | 556.19 | 354.70 | 0.60x | 0.94x | 1.57x |
| gate_up_proj | 1 | activation_quant |  -  | 43.67 | 14.23 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 105.36 | 69.55 | 38.15 | 1.51x | 2.76x | 1.82x |
| gate_up_proj | 1 | sparse_gemm | 106.21 | 68.18 | 18.13 | 1.56x | 5.86x | 3.76x |
| gate_up_proj | 1 | fused_dense_sparse | 106.17 | 81.99 | 41.13 | 1.29x | 2.58x | 1.99x |
| gate_up_proj | 8 | activation_quant |  -  | 44.62 | 14.72 |  -  |  -  |  -  |
| gate_up_proj | 8 | dense_gemm | 115.47 | 70.23 | 38.29 | 1.64x | 3.02x | 1.83x |
| gate_up_proj | 8 | sparse_gemm | 115.49 | 69.45 | 18.18 | 1.66x | 6.35x | 3.82x |
| gate_up_proj | 8 | fused_dense_sparse | 115.46 | 81.99 | 41.90 | 1.41x | 2.76x | 1.96x |
| gate_up_proj | 128 | activation_quant |  -  | 44.53 | 14.72 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 135.70 | 144.25 | 103.45 | 0.94x | 1.31x | 1.39x |
| gate_up_proj | 128 | sparse_gemm | 136.31 | 68.43 | 27.62 | 1.99x | 4.94x | 2.48x |
| gate_up_proj | 128 | fused_dense_sparse | 136.02 | 154.86 | 111.69 | 0.88x | 1.22x | 1.39x |
| gate_up_proj | 512 | activation_quant |  -  | 43.70 | 14.88 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 377.82 | 445.45 | 263.63 | 0.85x | 1.43x | 1.69x |
| gate_up_proj | 512 | sparse_gemm | 376.18 | 67.71 | 86.05 | 5.56x | 4.37x | 0.79x |
| gate_up_proj | 512 | fused_dense_sparse | 377.62 | 474.87 | 284.25 | 0.80x | 1.33x | 1.67x |
| gate_up_proj | 1024 | activation_quant |  -  | 62.76 | 14.70 |  -  |  -  |  -  |
| gate_up_proj | 1024 | dense_gemm | 678.21 | 851.74 | 504.35 | 0.80x | 1.34x | 1.69x |
| gate_up_proj | 1024 | sparse_gemm | 683.28 | 78.45 | 167.75 | 8.71x | 4.07x | 0.47x |
| gate_up_proj | 1024 | fused_dense_sparse | 681.05 | 914.74 | 544.16 | 0.74x | 1.25x | 1.68x |
| kv_proj | 1 | activation_quant |  -  | 44.94 | 14.46 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 8.74 | 69.26 | 10.83 | 0.13x | 0.81x | 6.39x |
| kv_proj | 1 | sparse_gemm | 9.97 | 68.57 | 17.92 | 0.15x | 0.56x | 3.83x |
| kv_proj | 1 | fused_dense_sparse | 9.94 | 81.95 | 12.06 | 0.12x | 0.82x | 6.79x |
| kv_proj | 8 | activation_quant |  -  | 44.97 | 14.54 |  -  |  -  |  -  |
| kv_proj | 8 | dense_gemm | 12.01 | 70.66 | 29.26 | 0.17x | 0.41x | 2.42x |
| kv_proj | 8 | sparse_gemm | 11.98 | 69.52 | 18.04 | 0.17x | 0.66x | 3.85x |
| kv_proj | 8 | fused_dense_sparse | 12.07 | 81.88 | 22.93 | 0.15x | 0.53x | 3.57x |
| kv_proj | 128 | activation_quant |  -  | 45.35 | 14.66 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 14.53 | 70.06 | 35.31 | 0.21x | 0.41x | 1.98x |
| kv_proj | 128 | sparse_gemm | 14.53 | 69.12 | 18.20 | 0.21x | 0.80x | 3.80x |
| kv_proj | 128 | fused_dense_sparse | 14.55 | 81.92 | 37.61 | 0.18x | 0.39x | 2.18x |
| kv_proj | 512 | activation_quant |  -  | 44.90 | 14.66 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 34.34 | 70.37 | 43.31 | 0.49x | 0.79x | 1.62x |
| kv_proj | 512 | sparse_gemm | 34.35 | 68.72 | 17.98 | 0.50x | 1.91x | 3.82x |
| kv_proj | 512 | fused_dense_sparse | 34.29 | 82.21 | 51.19 | 0.42x | 0.67x | 1.61x |
| kv_proj | 1024 | activation_quant |  -  | 61.99 | 14.89 |  -  |  -  |  -  |
| kv_proj | 1024 | dense_gemm | 72.67 | 93.82 | 55.81 | 0.77x | 1.30x | 1.68x |
| kv_proj | 1024 | sparse_gemm | 72.65 | 68.21 | 22.49 | 1.07x | 3.23x | 3.03x |
| kv_proj | 1024 | fused_dense_sparse | 72.47 | 103.20 | 62.29 | 0.70x | 1.16x | 1.66x |
| o_proj | 1 | activation_quant |  -  | 44.36 | 14.61 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 11.40 | 68.93 | 13.46 | 0.17x | 0.85x | 5.12x |
| o_proj | 1 | sparse_gemm | 24.27 | 68.11 | 18.23 | 0.36x | 1.33x | 3.74x |
| o_proj | 1 | fused_dense_sparse | 24.24 | 82.16 | 13.66 | 0.30x | 1.77x | 6.01x |
| o_proj | 8 | activation_quant |  -  | 44.44 | 14.47 |  -  |  -  |  -  |
| o_proj | 8 | dense_gemm | 12.04 | 70.59 | 38.27 | 0.17x | 0.31x | 1.84x |
| o_proj | 8 | sparse_gemm | 11.99 | 69.35 | 18.06 | 0.17x | 0.66x | 3.84x |
| o_proj | 8 | fused_dense_sparse | 12.05 | 80.83 | 35.46 | 0.15x | 0.34x | 2.28x |
| o_proj | 128 | activation_quant |  -  | 56.19 | 14.63 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 23.05 | 68.70 | 47.80 | 0.34x | 0.48x | 1.44x |
| o_proj | 128 | sparse_gemm | 23.10 | 68.32 | 19.49 | 0.34x | 1.19x | 3.51x |
| o_proj | 128 | fused_dense_sparse | 23.07 | 82.10 | 54.09 | 0.28x | 0.43x | 1.52x |
| o_proj | 512 | activation_quant |  -  | 56.26 | 14.66 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 73.55 | 143.39 | 72.94 | 0.51x | 1.01x | 1.97x |
| o_proj | 512 | sparse_gemm | 73.91 | 68.57 | 19.14 | 1.08x | 3.86x | 3.58x |
| o_proj | 512 | fused_dense_sparse | 73.76 | 156.90 | 85.01 | 0.47x | 0.87x | 1.85x |
| o_proj | 1024 | activation_quant |  -  | 61.83 | 16.64 |  -  |  -  |  -  |
| o_proj | 1024 | dense_gemm | 140.60 | 223.48 | 147.82 | 0.63x | 0.95x | 1.51x |
| o_proj | 1024 | sparse_gemm | 142.10 | 69.01 | 29.32 | 2.06x | 4.85x | 2.35x |
| o_proj | 1024 | fused_dense_sparse | 141.24 | 238.59 | 164.65 | 0.59x | 0.86x | 1.45x |
| q_proj | 1 | activation_quant |  -  | 43.53 | 14.26 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 12.13 | 69.47 | 10.81 | 0.17x | 1.12x | 6.42x |
| q_proj | 1 | sparse_gemm | 24.74 | 69.07 | 18.10 | 0.36x | 1.37x | 3.82x |
| q_proj | 1 | fused_dense_sparse | 24.73 | 82.32 | 12.00 | 0.30x | 2.06x | 6.86x |
| q_proj | 8 | activation_quant |  -  | 44.06 | 14.58 |  -  |  -  |  -  |
| q_proj | 8 | dense_gemm | 11.95 | 70.97 | 29.30 | 0.17x | 0.41x | 2.42x |
| q_proj | 8 | sparse_gemm | 12.02 | 69.29 | 18.24 | 0.17x | 0.66x | 3.80x |
| q_proj | 8 | fused_dense_sparse | 11.97 | 82.00 | 22.98 | 0.15x | 0.52x | 3.57x |
| q_proj | 128 | activation_quant |  -  | 45.40 | 14.64 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 20.11 | 69.77 | 36.72 | 0.29x | 0.55x | 1.90x |
| q_proj | 128 | sparse_gemm | 20.11 | 68.76 | 18.15 | 0.29x | 1.11x | 3.79x |
| q_proj | 128 | fused_dense_sparse | 20.10 | 83.40 | 38.33 | 0.24x | 0.52x | 2.18x |
| q_proj | 512 | activation_quant |  -  | 45.06 | 14.74 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 72.50 | 93.85 | 55.83 | 0.77x | 1.30x | 1.68x |
| q_proj | 512 | sparse_gemm | 72.78 | 68.87 | 24.38 | 1.06x | 2.99x | 2.82x |
| q_proj | 512 | fused_dense_sparse | 72.41 | 109.43 | 63.21 | 0.66x | 1.15x | 1.73x |
| q_proj | 1024 | activation_quant |  -  | 61.91 | 14.88 |  -  |  -  |  -  |
| q_proj | 1024 | dense_gemm | 140.55 | 184.08 | 113.98 | 0.76x | 1.23x | 1.62x |
| q_proj | 1024 | sparse_gemm | 140.74 | 68.61 | 40.73 | 2.05x | 3.45x | 1.68x |
| q_proj | 1024 | fused_dense_sparse | 142.05 | 206.75 | 124.69 | 0.69x | 1.14x | 1.66x |

### Qwen3-8B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 117.28 | 25.09 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 108.66 | 146.42 | 44.49 | 0.74x | 2.44x | 3.29x |
| down_proj | 1 | sparse_gemm | 109.71 | 68.16 | 18.07 | 1.61x | 6.07x | 3.77x |
| down_proj | 1 | fused_dense_sparse | 109.59 | 159.92 | 46.28 | 0.69x | 2.37x | 3.46x |
| down_proj | 8 | activation_quant |  -  | 132.64 | 24.97 |  -  |  -  |  -  |
| down_proj | 8 | dense_gemm | 118.26 | 143.37 | 118.86 | 0.82x | 0.99x | 1.21x |
| down_proj | 8 | sparse_gemm | 118.28 | 69.56 | 18.10 | 1.70x | 6.53x | 3.84x |
| down_proj | 8 | fused_dense_sparse | 118.30 | 155.35 | 116.34 | 0.76x | 1.02x | 1.34x |
| down_proj | 128 | activation_quant |  -  | 167.74 | 25.52 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 133.70 | 172.08 | 146.04 | 0.78x | 0.92x | 1.18x |
| down_proj | 128 | sparse_gemm | 133.80 | 68.70 | 22.58 | 1.95x | 5.93x | 3.04x |
| down_proj | 128 | fused_dense_sparse | 133.87 | 185.21 | 168.78 | 0.72x | 0.79x | 1.10x |
| down_proj | 512 | activation_quant |  -  | 167.83 | 51.65 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 361.05 | 436.45 | 336.59 | 0.83x | 1.07x | 1.30x |
| down_proj | 512 | sparse_gemm | 359.00 | 69.50 | 36.99 | 5.17x | 9.71x | 1.88x |
| down_proj | 512 | fused_dense_sparse | 357.53 | 482.55 | 252.69 | 0.74x | 1.41x | 1.91x |
| down_proj | 1024 | activation_quant |  -  | 175.16 | 77.03 |  -  |  -  |  -  |
| down_proj | 1024 | dense_gemm | 703.29 | 871.10 | 670.72 | 0.81x | 1.05x | 1.30x |
| down_proj | 1024 | sparse_gemm | 705.85 | 73.65 | 61.80 | 9.58x | 11.42x | 1.19x |
| down_proj | 1024 | fused_dense_sparse | 706.88 | 935.44 | 482.41 | 0.76x | 1.47x | 1.94x |
| gate_up_proj | 1 | activation_quant |  -  | 45.17 | 14.86 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 212.43 | 79.69 | 73.77 | 2.67x | 2.88x | 1.08x |
| gate_up_proj | 1 | sparse_gemm | 212.42 | 68.78 | 17.94 | 3.09x | 11.84x | 3.83x |
| gate_up_proj | 1 | fused_dense_sparse | 212.39 | 88.91 | 78.28 | 2.39x | 2.71x | 1.14x |
| gate_up_proj | 8 | activation_quant |  -  | 45.14 | 14.90 |  -  |  -  |  -  |
| gate_up_proj | 8 | dense_gemm | 221.35 | 77.10 | 49.88 | 2.87x | 4.44x | 1.55x |
| gate_up_proj | 8 | sparse_gemm | 221.34 | 69.92 | 18.01 | 3.17x | 12.29x | 3.88x |
| gate_up_proj | 8 | fused_dense_sparse | 221.35 | 89.66 | 54.82 | 2.47x | 4.04x | 1.64x |
| gate_up_proj | 128 | activation_quant |  -  | 56.22 | 14.81 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 263.08 | 223.73 | 148.65 | 1.18x | 1.77x | 1.51x |
| gate_up_proj | 128 | sparse_gemm | 263.16 | 69.18 | 37.63 | 3.80x | 6.99x | 1.84x |
| gate_up_proj | 128 | fused_dense_sparse | 263.31 | 253.68 | 169.83 | 1.04x | 1.55x | 1.49x |
| gate_up_proj | 512 | activation_quant |  -  | 56.59 | 14.79 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 748.30 | 869.55 | 459.51 | 0.86x | 1.63x | 1.89x |
| gate_up_proj | 512 | sparse_gemm | 773.17 | 68.62 | 108.73 | 11.27x | 7.11x | 0.63x |
| gate_up_proj | 512 | fused_dense_sparse | 741.91 | 955.67 | 514.76 | 0.78x | 1.44x | 1.86x |
| gate_up_proj | 1024 | activation_quant |  -  | 62.62 | 16.51 |  -  |  -  |  -  |
| gate_up_proj | 1024 | dense_gemm | 1333.90 | 1748.27 | 894.82 | 0.76x | 1.49x | 1.95x |
| gate_up_proj | 1024 | sparse_gemm | 1342.45 | 128.66 | 213.87 | 10.43x | 6.28x | 0.60x |
| gate_up_proj | 1024 | fused_dense_sparse | 1331.31 | 1850.61 | 1001.10 | 0.72x | 1.33x | 1.85x |
| kv_proj | 1 | activation_quant |  -  | 43.21 | 14.38 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 10.27 | 68.74 | 10.98 | 0.15x | 0.94x | 6.26x |
| kv_proj | 1 | sparse_gemm | 19.16 | 67.02 | 18.10 | 0.29x | 1.06x | 3.70x |
| kv_proj | 1 | fused_dense_sparse | 19.24 | 82.08 | 12.02 | 0.23x | 1.60x | 6.83x |
| kv_proj | 8 | activation_quant |  -  | 43.97 | 14.63 |  -  |  -  |  -  |
| kv_proj | 8 | dense_gemm | 12.06 | 69.10 | 38.23 | 0.17x | 0.32x | 1.81x |
| kv_proj | 8 | sparse_gemm | 11.97 | 68.13 | 18.04 | 0.18x | 0.66x | 3.78x |
| kv_proj | 8 | fused_dense_sparse | 12.00 | 82.44 | 34.00 | 0.15x | 0.35x | 2.42x |
| kv_proj | 128 | activation_quant |  -  | 56.18 | 14.70 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 19.40 | 69.90 | 47.33 | 0.28x | 0.41x | 1.48x |
| kv_proj | 128 | sparse_gemm | 19.42 | 67.49 | 18.25 | 0.29x | 1.06x | 3.70x |
| kv_proj | 128 | fused_dense_sparse | 19.40 | 82.53 | 50.98 | 0.24x | 0.38x | 1.62x |
| kv_proj | 512 | activation_quant |  -  | 56.24 | 14.48 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 52.99 | 88.13 | 58.79 | 0.60x | 0.90x | 1.50x |
| kv_proj | 512 | sparse_gemm | 52.97 | 68.46 | 17.79 | 0.77x | 2.98x | 3.85x |
| kv_proj | 512 | fused_dense_sparse | 52.87 | 97.60 | 69.53 | 0.54x | 0.76x | 1.40x |
| kv_proj | 1024 | activation_quant |  -  | 63.07 | 16.63 |  -  |  -  |  -  |
| kv_proj | 1024 | dense_gemm | 114.42 | 146.84 | 81.84 | 0.78x | 1.40x | 1.79x |
| kv_proj | 1024 | sparse_gemm | 114.95 | 68.85 | 24.82 | 1.67x | 4.63x | 2.77x |
| kv_proj | 1024 | fused_dense_sparse | 114.04 | 164.38 | 91.83 | 0.69x | 1.24x | 1.79x |
| o_proj | 1 | activation_quant |  -  | 44.04 | 14.32 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 16.74 | 69.28 | 16.16 | 0.24x | 1.04x | 4.29x |
| o_proj | 1 | sparse_gemm | 23.27 | 67.92 | 17.89 | 0.34x | 1.30x | 3.80x |
| o_proj | 1 | fused_dense_sparse | 23.58 | 81.19 | 16.89 | 0.29x | 1.40x | 4.81x |
| o_proj | 8 | activation_quant |  -  | 43.21 | 14.69 |  -  |  -  |  -  |
| o_proj | 8 | dense_gemm | 14.67 | 70.29 | 38.10 | 0.21x | 0.38x | 1.84x |
| o_proj | 8 | sparse_gemm | 14.67 | 69.23 | 18.07 | 0.21x | 0.81x | 3.83x |
| o_proj | 8 | fused_dense_sparse | 14.67 | 81.27 | 36.13 | 0.18x | 0.41x | 2.25x |
| o_proj | 128 | activation_quant |  -  | 56.22 | 14.61 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 30.78 | 69.60 | 48.27 | 0.44x | 0.64x | 1.44x |
| o_proj | 128 | sparse_gemm | 30.78 | 68.54 | 18.05 | 0.45x | 1.71x | 3.80x |
| o_proj | 128 | fused_dense_sparse | 30.88 | 81.76 | 51.43 | 0.38x | 0.60x | 1.59x |
| o_proj | 512 | activation_quant |  -  | 56.27 | 14.38 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 113.82 | 156.05 | 81.64 | 0.73x | 1.39x | 1.91x |
| o_proj | 512 | sparse_gemm | 114.40 | 68.22 | 26.06 | 1.68x | 4.39x | 2.62x |
| o_proj | 512 | fused_dense_sparse | 116.08 | 171.93 | 90.92 | 0.68x | 1.28x | 1.89x |
| o_proj | 1024 | activation_quant |  -  | 63.83 | 16.41 |  -  |  -  |  -  |
| o_proj | 1024 | dense_gemm | 211.90 | 292.32 | 154.89 | 0.72x | 1.37x | 1.89x |
| o_proj | 1024 | sparse_gemm | 212.81 | 68.71 | 42.93 | 3.10x | 4.96x | 1.60x |
| o_proj | 1024 | fused_dense_sparse | 212.60 | 320.28 | 175.18 | 0.66x | 1.21x | 1.83x |
| q_proj | 1 | activation_quant |  -  | 43.00 | 14.26 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 16.74 | 68.31 | 16.08 | 0.25x | 1.04x | 4.25x |
| q_proj | 1 | sparse_gemm | 38.37 | 67.30 | 17.79 | 0.57x | 2.16x | 3.78x |
| q_proj | 1 | fused_dense_sparse | 38.34 | 81.05 | 16.82 | 0.47x | 2.28x | 4.82x |
| q_proj | 8 | activation_quant |  -  | 43.65 | 14.62 |  -  |  -  |  -  |
| q_proj | 8 | dense_gemm | 14.64 | 70.65 | 38.05 | 0.21x | 0.38x | 1.86x |
| q_proj | 8 | sparse_gemm | 14.66 | 68.37 | 17.95 | 0.21x | 0.82x | 3.81x |
| q_proj | 8 | fused_dense_sparse | 14.61 | 82.39 | 36.15 | 0.18x | 0.40x | 2.28x |
| q_proj | 128 | activation_quant |  -  | 56.18 | 14.58 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 30.71 | 68.51 | 48.12 | 0.45x | 0.64x | 1.42x |
| q_proj | 128 | sparse_gemm | 30.83 | 67.34 | 18.15 | 0.46x | 1.70x | 3.71x |
| q_proj | 128 | fused_dense_sparse | 30.80 | 83.13 | 51.33 | 0.37x | 0.60x | 1.62x |
| q_proj | 512 | activation_quant |  -  | 56.27 | 14.33 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 114.03 | 146.91 | 81.62 | 0.78x | 1.40x | 1.80x |
| q_proj | 512 | sparse_gemm | 113.79 | 67.30 | 24.40 | 1.69x | 4.66x | 2.76x |
| q_proj | 512 | fused_dense_sparse | 113.94 | 164.45 | 91.06 | 0.69x | 1.25x | 1.81x |
| q_proj | 1024 | activation_quant |  -  | 62.06 | 16.41 |  -  |  -  |  -  |
| q_proj | 1024 | dense_gemm | 213.33 | 291.43 | 154.93 | 0.73x | 1.38x | 1.88x |
| q_proj | 1024 | sparse_gemm | 214.46 | 67.84 | 42.87 | 3.16x | 5.00x | 1.58x |
| q_proj | 1024 | fused_dense_sparse | 211.08 | 320.13 | 175.36 | 0.66x | 1.20x | 1.83x |


## 4. End-to-end speedup (CUDA over Triton)

Rows: projection. Cells: `triton_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-0.6B
| proj | shape | T=1 | T=8 | T=128 | T=512 | T=1024 |
|---|---|---:|---:|---:|---:|---:|
| q_proj | 1024->2048 | **18.07x** | **5.11x** | **5.06x** | **4.69x** | **3.82x** |
| kv_proj | 1024->2048 | **18.22x** | **5.04x** | **5.04x** | **4.69x** | **3.86x** |
| o_proj | 2048->1024 | **18.11x** | **5.05x** | **4.71x** | **3.66x** | **3.14x** |
| gate_up_proj | 1024->6144 | **14.19x** | **5.03x** | **5.06x** | **2.23x** | **1.54x** |
| down_proj | 3072->1024 | **13.49x** | **3.78x** | **3.27x** | **2.73x** | **2.18x** |

### Qwen3-1.7B
| proj | shape | T=1 | T=8 | T=128 | T=512 | T=1024 |
|---|---|---:|---:|---:|---:|---:|
| q_proj | 2048->2048 | **18.26x** | **5.00x** | **3.99x** | **2.88x** | **2.42x** |
| kv_proj | 2048->2048 | **18.12x** | **5.02x** | **3.99x** | **2.90x** | **2.40x** |
| o_proj | 2048->2048 | **17.94x** | **4.96x** | **3.96x** | **2.90x** | **2.42x** |
| gate_up_proj | 2048->12288 | **5.28x** | **4.89x** | **2.54x** | **1.76x** | **1.69x** |
| down_proj | 6144->2048 | **6.73x** | **2.19x** | **1.84x** | **1.94x** | **2.04x** |

### Qwen3-4B
| proj | shape | T=1 | T=8 | T=128 | T=512 | T=1024 |
|---|---|---:|---:|---:|---:|---:|
| q_proj | 2560->4096 | **10.17x** | **4.42x** | **2.92x** | **2.05x** | **1.81x** |
| kv_proj | 2560->2048 | **13.56x** | **4.42x** | **2.96x** | **2.26x** | **2.04x** |
| o_proj | 4096->2560 | **8.29x** | **2.89x** | **2.06x** | **2.20x** | **1.65x** |
| gate_up_proj | 2560->19456 | **2.72x** | **2.71x** | **1.59x** | **1.74x** | **1.69x** |
| down_proj | 9728->2560 | **5.82x** | **1.99x** | **1.81x** | **2.39x** | **1.73x** |

### Qwen3-8B
| proj | shape | T=1 | T=8 | T=128 | T=512 | T=1024 |
|---|---|---:|---:|---:|---:|---:|
| q_proj | 4096->4096 | **7.16x** | **2.85x** | **2.16x** | **2.15x** | **1.98x** |
| kv_proj | 4096->2048 | **9.75x** | **3.00x** | **2.17x** | **1.93x** | **2.08x** |
| o_proj | 4096->4096 | **7.11x** | **2.84x** | **2.14x** | **2.15x** | **1.98x** |
| gate_up_proj | 4096->24576 | **1.42x** | **2.04x** | **1.71x** | **1.88x** | **1.88x** |
| down_proj | 12288->4096 | **5.13x** | **2.02x** | **1.82x** | **2.15x** | **1.99x** |


## 5. CUDA end-to-end bottleneck hint

For each shape, compare CUDA `activation_quant` against CUDA `fused_dense_sparse`. A larger `quant_share` means launch/prologue dominates; a larger `fused_share` means the main CUDA matmul kernel dominates.

| model | proj | T | shape | quant_us | fused_us | quant_share | fused_share | likely_bottleneck |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-0.6B | q_proj | 1 | 1024->2048 | 14.59 | 12.13 | 54.6% | 45.4% | quant/prologue dominated |
| Qwen3-0.6B | q_proj | 8 | 1024->2048 | 14.60 | 12.17 | 54.5% | 45.5% | quant/prologue dominated |
| Qwen3-0.6B | q_proj | 128 | 1024->2048 | 14.36 | 15.69 | 47.8% | 52.2% | quant/prologue dominated |
| Qwen3-0.6B | q_proj | 512 | 1024->2048 | 14.58 | 23.88 | 37.9% | 62.1% | quant/prologue dominated |
| Qwen3-0.6B | q_proj | 1024 | 1024->2048 | 14.68 | 33.43 | 30.5% | 69.5% | mixed |
| Qwen3-0.6B | kv_proj | 1 | 1024->2048 | 14.36 | 12.03 | 54.4% | 45.6% | quant/prologue dominated |
| Qwen3-0.6B | kv_proj | 8 | 1024->2048 | 14.68 | 11.97 | 55.1% | 44.9% | quant/prologue dominated |
| Qwen3-0.6B | kv_proj | 128 | 1024->2048 | 14.52 | 15.69 | 48.1% | 51.9% | quant/prologue dominated |
| Qwen3-0.6B | kv_proj | 512 | 1024->2048 | 14.64 | 23.92 | 38.0% | 62.0% | quant/prologue dominated |
| Qwen3-0.6B | kv_proj | 1024 | 1024->2048 | 14.69 | 33.42 | 30.5% | 69.5% | mixed |
| Qwen3-0.6B | o_proj | 1 | 2048->1024 | 14.62 | 12.06 | 54.8% | 45.2% | quant/prologue dominated |
| Qwen3-0.6B | o_proj | 8 | 2048->1024 | 14.58 | 18.37 | 44.3% | 55.7% | quant/prologue dominated |
| Qwen3-0.6B | o_proj | 128 | 2048->1024 | 14.64 | 21.91 | 40.1% | 59.9% | quant/prologue dominated |
| Qwen3-0.6B | o_proj | 512 | 2048->1024 | 14.42 | 29.05 | 33.2% | 66.8% | mixed |
| Qwen3-0.6B | o_proj | 1024 | 2048->1024 | 14.60 | 37.92 | 27.8% | 72.2% | mixed |
| Qwen3-0.6B | gate_up_proj | 1 | 1024->6144 | 14.44 | 12.08 | 54.4% | 45.6% | quant/prologue dominated |
| Qwen3-0.6B | gate_up_proj | 8 | 1024->6144 | 14.51 | 11.99 | 54.8% | 45.2% | quant/prologue dominated |
| Qwen3-0.6B | gate_up_proj | 128 | 1024->6144 | 14.53 | 19.80 | 42.3% | 57.7% | quant/prologue dominated |
| Qwen3-0.6B | gate_up_proj | 512 | 1024->6144 | 14.55 | 55.67 | 20.7% | 79.3% | mixed |
| Qwen3-0.6B | gate_up_proj | 1024 | 1024->6144 | 14.79 | 91.40 | 13.9% | 86.1% | main fused kernel dominated |
| Qwen3-0.6B | down_proj | 1 | 3072->1024 | 14.37 | 12.17 | 54.2% | 45.8% | quant/prologue dominated |
| Qwen3-0.6B | down_proj | 8 | 3072->1024 | 14.58 | 26.70 | 35.3% | 64.7% | quant/prologue dominated |
| Qwen3-0.6B | down_proj | 128 | 3072->1024 | 14.49 | 32.43 | 30.9% | 69.1% | mixed |
| Qwen3-0.6B | down_proj | 512 | 3072->1024 | 14.60 | 39.78 | 26.8% | 73.2% | mixed |
| Qwen3-0.6B | down_proj | 1024 | 3072->1024 | 14.79 | 53.99 | 21.5% | 78.5% | mixed |
| Qwen3-1.7B | q_proj | 1 | 2048->2048 | 14.42 | 11.93 | 54.7% | 45.3% | quant/prologue dominated |
| Qwen3-1.7B | q_proj | 8 | 2048->2048 | 14.60 | 18.89 | 43.6% | 56.4% | quant/prologue dominated |
| Qwen3-1.7B | q_proj | 128 | 2048->2048 | 14.22 | 27.04 | 34.5% | 65.5% | mixed |
| Qwen3-1.7B | q_proj | 512 | 2048->2048 | 14.42 | 39.08 | 26.9% | 73.1% | mixed |
| Qwen3-1.7B | q_proj | 1024 | 2048->2048 | 14.81 | 52.41 | 22.0% | 78.0% | mixed |
| Qwen3-1.7B | kv_proj | 1 | 2048->2048 | 14.47 | 11.93 | 54.8% | 45.2% | quant/prologue dominated |
| Qwen3-1.7B | kv_proj | 8 | 2048->2048 | 14.57 | 18.93 | 43.5% | 56.5% | quant/prologue dominated |
| Qwen3-1.7B | kv_proj | 128 | 2048->2048 | 14.52 | 27.06 | 34.9% | 65.1% | mixed |
| Qwen3-1.7B | kv_proj | 512 | 2048->2048 | 14.81 | 39.16 | 27.4% | 72.6% | mixed |
| Qwen3-1.7B | kv_proj | 1024 | 2048->2048 | 14.57 | 52.52 | 21.7% | 78.3% | mixed |
| Qwen3-1.7B | o_proj | 1 | 2048->2048 | 14.35 | 11.85 | 54.8% | 45.2% | quant/prologue dominated |
| Qwen3-1.7B | o_proj | 8 | 2048->2048 | 14.57 | 18.92 | 43.5% | 56.5% | quant/prologue dominated |
| Qwen3-1.7B | o_proj | 128 | 2048->2048 | 14.52 | 27.06 | 34.9% | 65.1% | mixed |
| Qwen3-1.7B | o_proj | 512 | 2048->2048 | 14.63 | 39.16 | 27.2% | 72.8% | mixed |
| Qwen3-1.7B | o_proj | 1024 | 2048->2048 | 14.82 | 52.55 | 22.0% | 78.0% | mixed |
| Qwen3-1.7B | gate_up_proj | 1 | 2048->12288 | 14.18 | 22.33 | 38.8% | 61.2% | quant/prologue dominated |
| Qwen3-1.7B | gate_up_proj | 8 | 2048->12288 | 14.39 | 21.04 | 40.6% | 59.4% | quant/prologue dominated |
| Qwen3-1.7B | gate_up_proj | 128 | 2048->12288 | 14.71 | 46.94 | 23.9% | 76.1% | mixed |
| Qwen3-1.7B | gate_up_proj | 512 | 2048->12288 | 14.52 | 146.09 | 9.0% | 91.0% | main fused kernel dominated |
| Qwen3-1.7B | gate_up_proj | 1024 | 2048->12288 | 14.41 | 283.37 | 4.8% | 95.2% | main fused kernel dominated |
| Qwen3-1.7B | down_proj | 1 | 6144->2048 | 14.54 | 16.29 | 47.2% | 52.8% | quant/prologue dominated |
| Qwen3-1.7B | down_proj | 8 | 6144->2048 | 14.49 | 52.07 | 21.8% | 78.2% | mixed |
| Qwen3-1.7B | down_proj | 128 | 6144->2048 | 14.72 | 75.86 | 16.2% | 83.8% | main fused kernel dominated |
| Qwen3-1.7B | down_proj | 512 | 6144->2048 | 16.41 | 103.29 | 13.7% | 86.3% | main fused kernel dominated |
| Qwen3-1.7B | down_proj | 1024 | 6144->2048 | 31.69 | 130.80 | 19.5% | 80.5% | main fused kernel dominated |
| Qwen3-4B | q_proj | 1 | 2560->4096 | 14.26 | 12.00 | 54.3% | 45.7% | quant/prologue dominated |
| Qwen3-4B | q_proj | 8 | 2560->4096 | 14.58 | 22.98 | 38.8% | 61.2% | quant/prologue dominated |
| Qwen3-4B | q_proj | 128 | 2560->4096 | 14.64 | 38.33 | 27.6% | 72.4% | mixed |
| Qwen3-4B | q_proj | 512 | 2560->4096 | 14.74 | 63.21 | 18.9% | 81.1% | main fused kernel dominated |
| Qwen3-4B | q_proj | 1024 | 2560->4096 | 14.88 | 124.69 | 10.7% | 89.3% | main fused kernel dominated |
| Qwen3-4B | kv_proj | 1 | 2560->2048 | 14.46 | 12.06 | 54.5% | 45.5% | quant/prologue dominated |
| Qwen3-4B | kv_proj | 8 | 2560->2048 | 14.54 | 22.93 | 38.8% | 61.2% | quant/prologue dominated |
| Qwen3-4B | kv_proj | 128 | 2560->2048 | 14.66 | 37.61 | 28.1% | 71.9% | mixed |
| Qwen3-4B | kv_proj | 512 | 2560->2048 | 14.66 | 51.19 | 22.3% | 77.7% | mixed |
| Qwen3-4B | kv_proj | 1024 | 2560->2048 | 14.89 | 62.29 | 19.3% | 80.7% | main fused kernel dominated |
| Qwen3-4B | o_proj | 1 | 4096->2560 | 14.61 | 13.66 | 51.7% | 48.3% | quant/prologue dominated |
| Qwen3-4B | o_proj | 8 | 4096->2560 | 14.47 | 35.46 | 29.0% | 71.0% | mixed |
| Qwen3-4B | o_proj | 128 | 4096->2560 | 14.63 | 54.09 | 21.3% | 78.7% | mixed |
| Qwen3-4B | o_proj | 512 | 4096->2560 | 14.66 | 85.01 | 14.7% | 85.3% | main fused kernel dominated |
| Qwen3-4B | o_proj | 1024 | 4096->2560 | 16.64 | 164.65 | 9.2% | 90.8% | main fused kernel dominated |
| Qwen3-4B | gate_up_proj | 1 | 2560->19456 | 14.23 | 41.13 | 25.7% | 74.3% | mixed |
| Qwen3-4B | gate_up_proj | 8 | 2560->19456 | 14.72 | 41.90 | 26.0% | 74.0% | mixed |
| Qwen3-4B | gate_up_proj | 128 | 2560->19456 | 14.72 | 111.69 | 11.6% | 88.4% | main fused kernel dominated |
| Qwen3-4B | gate_up_proj | 512 | 2560->19456 | 14.88 | 284.25 | 5.0% | 95.0% | main fused kernel dominated |
| Qwen3-4B | gate_up_proj | 1024 | 2560->19456 | 14.70 | 544.16 | 2.6% | 97.4% | main fused kernel dominated |
| Qwen3-4B | down_proj | 1 | 9728->2560 | 19.54 | 30.69 | 38.9% | 61.1% | quant/prologue dominated |
| Qwen3-4B | down_proj | 8 | 9728->2560 | 21.46 | 95.86 | 18.3% | 81.7% | main fused kernel dominated |
| Qwen3-4B | down_proj | 128 | 9728->2560 | 21.70 | 131.01 | 14.2% | 85.8% | main fused kernel dominated |
| Qwen3-4B | down_proj | 512 | 9728->2560 | 24.87 | 181.61 | 12.0% | 88.0% | main fused kernel dominated |
| Qwen3-4B | down_proj | 1024 | 9728->2560 | 47.24 | 354.70 | 11.8% | 88.2% | main fused kernel dominated |
| Qwen3-8B | q_proj | 1 | 4096->4096 | 14.26 | 16.82 | 45.9% | 54.1% | quant/prologue dominated |
| Qwen3-8B | q_proj | 8 | 4096->4096 | 14.62 | 36.15 | 28.8% | 71.2% | mixed |
| Qwen3-8B | q_proj | 128 | 4096->4096 | 14.58 | 51.33 | 22.1% | 77.9% | mixed |
| Qwen3-8B | q_proj | 512 | 4096->4096 | 14.33 | 91.06 | 13.6% | 86.4% | main fused kernel dominated |
| Qwen3-8B | q_proj | 1024 | 4096->4096 | 16.41 | 175.36 | 8.6% | 91.4% | main fused kernel dominated |
| Qwen3-8B | kv_proj | 1 | 4096->2048 | 14.38 | 12.02 | 54.5% | 45.5% | quant/prologue dominated |
| Qwen3-8B | kv_proj | 8 | 4096->2048 | 14.63 | 34.00 | 30.1% | 69.9% | mixed |
| Qwen3-8B | kv_proj | 128 | 4096->2048 | 14.70 | 50.98 | 22.4% | 77.6% | mixed |
| Qwen3-8B | kv_proj | 512 | 4096->2048 | 14.48 | 69.53 | 17.2% | 82.8% | main fused kernel dominated |
| Qwen3-8B | kv_proj | 1024 | 4096->2048 | 16.63 | 91.83 | 15.3% | 84.7% | main fused kernel dominated |
| Qwen3-8B | o_proj | 1 | 4096->4096 | 14.32 | 16.89 | 45.9% | 54.1% | quant/prologue dominated |
| Qwen3-8B | o_proj | 8 | 4096->4096 | 14.69 | 36.13 | 28.9% | 71.1% | mixed |
| Qwen3-8B | o_proj | 128 | 4096->4096 | 14.61 | 51.43 | 22.1% | 77.9% | mixed |
| Qwen3-8B | o_proj | 512 | 4096->4096 | 14.38 | 90.92 | 13.7% | 86.3% | main fused kernel dominated |
| Qwen3-8B | o_proj | 1024 | 4096->4096 | 16.41 | 175.18 | 8.6% | 91.4% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 1 | 4096->24576 | 14.86 | 78.28 | 16.0% | 84.0% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 8 | 4096->24576 | 14.90 | 54.82 | 21.4% | 78.6% | mixed |
| Qwen3-8B | gate_up_proj | 128 | 4096->24576 | 14.81 | 169.83 | 8.0% | 92.0% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 512 | 4096->24576 | 14.79 | 514.76 | 2.8% | 97.2% | main fused kernel dominated |
| Qwen3-8B | gate_up_proj | 1024 | 4096->24576 | 16.51 | 1001.10 | 1.6% | 98.4% | main fused kernel dominated |
| Qwen3-8B | down_proj | 1 | 12288->4096 | 25.09 | 46.28 | 35.2% | 64.8% | quant/prologue dominated |
| Qwen3-8B | down_proj | 8 | 12288->4096 | 24.97 | 116.34 | 17.7% | 82.3% | main fused kernel dominated |
| Qwen3-8B | down_proj | 128 | 12288->4096 | 25.52 | 168.78 | 13.1% | 86.9% | main fused kernel dominated |
| Qwen3-8B | down_proj | 512 | 12288->4096 | 51.65 | 252.69 | 17.0% | 83.0% | main fused kernel dominated |
| Qwen3-8B | down_proj | 1024 | 12288->4096 | 77.03 | 482.41 | 13.8% | 86.2% | main fused kernel dominated |
