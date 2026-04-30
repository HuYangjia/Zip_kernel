# Qwen3 multi-scale kernel benchmark

- Timestamp: `20260430_124225`
- Device: `NVIDIA GeForce RTX 4090`
- PyTorch: `2.8.0+cu126`  Triton: `3.4.0`
- Baseline: cuBLAS FP16 matmul (`torch.matmul` on `fp16`)
- CUDA path: `activation_quant_cuda` + `fused_dense_sparse_cuda` (T=1 uses `fused_quant_gemv_cuda`, with automatic fallback on unsupported decode-group counts)
- Triton path: `quantize_activation_s4` + `fused_dense_sparse_gemm`
- hp_ratio: `0.05`  (block-sparse density)
- Stats: stable microbenchmark helper = 50 warmup, 100 inner, 3 repeats, min-of-means


## 1. End-to-end speedup vs FP16

Rows: projection. Cells: `fp16_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-14B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 5120->5120 | **2.30x** | **1.57x** | **1.03x** | **0.98x** |
| kv_proj | 5120->2048 | **1.48x** | **0.78x** | **0.74x** | **0.83x** |
| o_proj | 5120->5120 | **2.30x** | **1.56x** | **1.02x** | **0.98x** |
| gate_up_proj | 5120->34816 | **2.07x** | **1.45x** | **0.96x** | **0.77x** |
| down_proj | 17408->5120 | **2.27x** | **1.83x** | **1.27x** | **0.80x** |

### Qwen2.5-32B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 5120->5120 | **2.29x** | **1.55x** | **1.02x** | **0.99x** |
| kv_proj | 5120->2048 | **1.47x** | **0.78x** | **0.73x** | **0.83x** |
| o_proj | 5120->5120 | **2.30x** | **1.56x** | **1.03x** | **0.99x** |
| gate_up_proj | 5120->55296 | **2.00x** | **1.76x** | **0.96x** | **0.71x** |
| down_proj | 27648->5120 | **0.83x** | **1.08x** | **0.99x** | **0.70x** |

### LLaMA3-70B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 8192->8192 | **2.23x** | **2.31x** | **1.70x** | **1.06x** |
| kv_proj | 8192->2048 | **1.50x** | **0.83x** | **0.77x** | **0.89x** |
| o_proj | 8192->8192 | **2.19x** | **2.31x** | **1.70x** | **1.06x** |
| gate_up_proj | 8192->57344 | **2.03x** | **1.85x** | **1.00x** | **0.69x** |
| down_proj | 28672->8192 | **1.16x** | **1.20x** | **1.16x** | **1.01x** |


## 2. End-to-end raw latencies (us)


### Qwen3-14B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 17408->5120 | 1 | 193.22 | 425.15 | 85.24 | 0.45x | 2.27x | 4.99x |
| down_proj | 17408->5120 | 32 | 226.15 | 452.33 | 123.64 | 0.50x | 1.83x | 3.66x |
| down_proj | 17408->5120 | 128 | 244.41 | 624.86 | 193.05 | 0.39x | 1.27x | 3.24x |
| down_proj | 17408->5120 | 512 | 586.00 | 1243.81 | 734.52 | 0.47x | 0.80x | 1.69x |
| gate_up_proj | 5120->34816 | 1 | 378.26 | 260.64 | 183.08 | 1.45x | 2.07x | 1.42x |
| gate_up_proj | 5120->34816 | 32 | 434.49 | 310.77 | 299.79 | 1.40x | 1.45x | 1.04x |
| gate_up_proj | 5120->34816 | 128 | 456.43 | 555.85 | 477.28 | 0.82x | 0.96x | 1.16x |
| gate_up_proj | 5120->34816 | 512 | 1174.76 | 1708.75 | 1517.26 | 0.69x | 0.77x | 1.13x |
| kv_proj | 5120->2048 | 1 | 25.59 | 133.43 | 17.33 | 0.19x | 1.48x | 7.70x |
| kv_proj | 5120->2048 | 32 | 27.22 | 134.77 | 34.93 | 0.20x | 0.78x | 3.86x |
| kv_proj | 5120->2048 | 128 | 30.23 | 141.09 | 41.06 | 0.21x | 0.74x | 3.44x |
| kv_proj | 5120->2048 | 512 | 62.73 | 197.81 | 75.88 | 0.32x | 0.83x | 2.61x |
| o_proj | 5120->5120 | 1 | 60.68 | 132.39 | 26.42 | 0.46x | 2.30x | 5.01x |
| o_proj | 5120->5120 | 32 | 64.26 | 136.70 | 41.22 | 0.47x | 1.56x | 3.32x |
| o_proj | 5120->5120 | 128 | 69.12 | 195.80 | 67.53 | 0.35x | 1.02x | 2.90x |
| o_proj | 5120->5120 | 512 | 172.33 | 370.45 | 175.33 | 0.47x | 0.98x | 2.11x |
| q_proj | 5120->5120 | 1 | 61.04 | 133.55 | 26.56 | 0.46x | 2.30x | 5.03x |
| q_proj | 5120->5120 | 32 | 64.33 | 135.75 | 41.02 | 0.47x | 1.57x | 3.31x |
| q_proj | 5120->5120 | 128 | 69.12 | 195.62 | 67.27 | 0.35x | 1.03x | 2.91x |
| q_proj | 5120->5120 | 512 | 171.66 | 370.49 | 175.34 | 0.46x | 0.98x | 2.11x |

### Qwen2.5-32B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 27648->5120 | 1 | 302.80 | 705.43 | 364.46 | 0.43x | 0.83x | 1.94x |
| down_proj | 27648->5120 | 32 | 326.34 | 735.80 | 300.95 | 0.44x | 1.08x | 2.44x |
| down_proj | 27648->5120 | 128 | 378.47 | 1021.26 | 382.66 | 0.37x | 0.99x | 2.67x |
| down_proj | 27648->5120 | 512 | 918.62 | 1955.23 | 1310.84 | 0.47x | 0.70x | 1.49x |
| gate_up_proj | 5120->55296 | 1 | 596.74 | 361.80 | 298.44 | 1.65x | 2.00x | 1.21x |
| gate_up_proj | 5120->55296 | 32 | 665.11 | 386.23 | 378.74 | 1.72x | 1.76x | 1.02x |
| gate_up_proj | 5120->55296 | 128 | 674.05 | 756.35 | 703.64 | 0.89x | 0.96x | 1.07x |
| gate_up_proj | 5120->55296 | 512 | 1854.10 | 2647.04 | 2595.49 | 0.70x | 0.71x | 1.02x |
| kv_proj | 5120->2048 | 1 | 25.66 | 132.01 | 17.41 | 0.19x | 1.47x | 7.58x |
| kv_proj | 5120->2048 | 32 | 27.22 | 135.49 | 35.11 | 0.20x | 0.78x | 3.86x |
| kv_proj | 5120->2048 | 128 | 30.30 | 141.71 | 41.28 | 0.21x | 0.73x | 3.43x |
| kv_proj | 5120->2048 | 512 | 62.86 | 197.60 | 75.70 | 0.32x | 0.83x | 2.61x |
| o_proj | 5120->5120 | 1 | 61.00 | 132.94 | 26.56 | 0.46x | 2.30x | 5.00x |
| o_proj | 5120->5120 | 32 | 64.54 | 137.22 | 41.47 | 0.47x | 1.56x | 3.31x |
| o_proj | 5120->5120 | 128 | 69.52 | 196.59 | 67.78 | 0.35x | 1.03x | 2.90x |
| o_proj | 5120->5120 | 512 | 174.21 | 372.46 | 176.52 | 0.47x | 0.99x | 2.11x |
| q_proj | 5120->5120 | 1 | 60.96 | 132.00 | 26.61 | 0.46x | 2.29x | 4.96x |
| q_proj | 5120->5120 | 32 | 64.26 | 137.22 | 41.40 | 0.47x | 1.55x | 3.31x |
| q_proj | 5120->5120 | 128 | 69.38 | 196.58 | 67.85 | 0.35x | 1.02x | 2.90x |
| q_proj | 5120->5120 | 512 | 174.67 | 372.05 | 176.40 | 0.47x | 0.99x | 2.11x |

### LLaMA3-70B - end-to-end (us)
| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| down_proj | 28672->8192 | 1 | 497.52 | 722.23 | 428.21 | 0.69x | 1.16x | 1.69x |
| down_proj | 28672->8192 | 32 | 528.76 | 775.35 | 441.09 | 0.68x | 1.20x | 1.76x |
| down_proj | 28672->8192 | 128 | 603.52 | 1070.73 | 519.31 | 0.56x | 1.16x | 2.06x |
| down_proj | 28672->8192 | 512 | 1575.34 | 2556.98 | 1562.63 | 0.62x | 1.01x | 1.64x |
| gate_up_proj | 8192->57344 | 1 | 985.90 | 624.84 | 485.25 | 1.58x | 2.03x | 1.29x |
| gate_up_proj | 8192->57344 | 32 | 1099.90 | 660.19 | 593.42 | 1.67x | 1.85x | 1.11x |
| gate_up_proj | 8192->57344 | 128 | 1142.68 | 1228.37 | 1147.19 | 0.93x | 1.00x | 1.07x |
| gate_up_proj | 8192->57344 | 512 | 3030.39 | 4420.33 | 4380.58 | 0.69x | 0.69x | 1.01x |
| kv_proj | 8192->2048 | 1 | 39.17 | 184.07 | 26.16 | 0.21x | 1.50x | 7.04x |
| kv_proj | 8192->2048 | 32 | 40.58 | 213.17 | 48.80 | 0.19x | 0.83x | 4.37x |
| kv_proj | 8192->2048 | 128 | 46.80 | 216.00 | 60.71 | 0.22x | 0.77x | 3.56x |
| kv_proj | 8192->2048 | 512 | 100.82 | 305.29 | 112.86 | 0.33x | 0.89x | 2.71x |
| o_proj | 8192->8192 | 1 | 146.65 | 203.16 | 66.86 | 0.72x | 2.19x | 3.04x |
| o_proj | 8192->8192 | 32 | 172.13 | 216.52 | 74.41 | 0.79x | 2.31x | 2.91x |
| o_proj | 8192->8192 | 128 | 191.03 | 310.15 | 112.48 | 0.62x | 1.70x | 2.76x |
| o_proj | 8192->8192 | 512 | 400.36 | 737.13 | 377.12 | 0.54x | 1.06x | 1.95x |
| q_proj | 8192->8192 | 1 | 146.69 | 202.17 | 65.78 | 0.73x | 2.23x | 3.07x |
| q_proj | 8192->8192 | 32 | 172.14 | 216.72 | 74.53 | 0.79x | 2.31x | 2.91x |
| q_proj | 8192->8192 | 128 | 191.06 | 310.24 | 112.56 | 0.62x | 1.70x | 2.76x |
| q_proj | 8192->8192 | 512 | 398.73 | 737.06 | 377.00 | 0.54x | 1.06x | 1.96x |


## 3. Sub-kernel breakdown (us)


### Qwen3-14B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 173.13 | 34.36 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 193.16 | 201.46 | 72.69 | 0.96x | 2.66x | 2.77x |
| down_proj | 1 | sparse_gemm | 193.30 | 66.90 | 19.76 | 2.89x | 9.78x | 3.39x |
| down_proj | 1 | fused_dense_sparse | 193.26 | 253.28 | 76.11 | 0.76x | 2.54x | 3.33x |
| down_proj | 32 | activation_quant |  -  | 232.08 | 34.29 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 226.26 | 201.42 | 341.58 | 1.12x | 0.66x | 0.59x |
| down_proj | 32 | sparse_gemm | 226.18 | 67.74 | 20.44 | 3.34x | 11.07x | 3.31x |
| down_proj | 32 | fused_dense_sparse | 226.21 | 219.92 | 89.44 | 1.03x | 2.53x | 2.46x |
| down_proj | 128 | activation_quant |  -  | 233.38 | 34.84 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 244.12 | 356.67 | 474.03 | 0.68x | 0.51x | 0.75x |
| down_proj | 128 | sparse_gemm | 244.32 | 68.04 | 29.88 | 3.59x | 8.18x | 2.28x |
| down_proj | 128 | fused_dense_sparse | 244.14 | 389.29 | 158.45 | 0.63x | 1.54x | 2.46x |
| down_proj | 512 | activation_quant |  -  | 233.14 | 69.98 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 584.12 | 921.87 | 845.57 | 0.63x | 0.69x | 1.09x |
| down_proj | 512 | sparse_gemm | 584.88 | 75.92 | 55.43 | 7.70x | 10.55x | 1.37x |
| down_proj | 512 | fused_dense_sparse | 584.21 | 992.27 | 653.72 | 0.59x | 0.89x | 1.52x |
| gate_up_proj | 1 | activation_quant |  -  | 48.99 | 16.43 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 378.26 | 197.27 | 178.18 | 1.92x | 2.12x | 1.11x |
| gate_up_proj | 1 | sparse_gemm | 378.19 | 69.18 | 19.77 | 5.47x | 19.13x | 3.50x |
| gate_up_proj | 1 | fused_dense_sparse | 378.27 | 206.45 | 184.27 | 1.83x | 2.05x | 1.12x |
| gate_up_proj | 32 | activation_quant |  -  | 72.77 | 16.56 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 434.08 | 225.57 | 387.62 | 1.92x | 1.12x | 0.58x |
| gate_up_proj | 32 | sparse_gemm | 434.29 | 67.94 | 19.71 | 6.39x | 22.03x | 3.45x |
| gate_up_proj | 32 | fused_dense_sparse | 434.16 | 238.25 | 282.76 | 1.82x | 1.54x | 0.84x |
| gate_up_proj | 128 | activation_quant |  -  | 70.68 | 16.77 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 456.08 | 460.72 | 673.79 | 0.99x | 0.68x | 0.68x |
| gate_up_proj | 128 | sparse_gemm | 456.16 | 68.55 | 46.33 | 6.65x | 9.85x | 1.48x |
| gate_up_proj | 128 | fused_dense_sparse | 456.12 | 478.43 | 451.90 | 0.95x | 1.01x | 1.06x |
| gate_up_proj | 512 | activation_quant |  -  | 70.79 | 16.70 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 1162.04 | 1534.12 | 2572.53 | 0.76x | 0.45x | 0.60x |
| gate_up_proj | 512 | sparse_gemm | 1167.96 | 108.50 | 153.82 | 10.76x | 7.59x | 0.71x |
| gate_up_proj | 512 | fused_dense_sparse | 1174.64 | 1631.72 | 1471.33 | 0.72x | 0.80x | 1.11x |
| kv_proj | 1 | activation_quant |  -  | 49.00 | 16.21 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 25.54 | 68.87 | 13.22 | 0.37x | 1.93x | 5.21x |
| kv_proj | 1 | sparse_gemm | 25.61 | 67.71 | 19.68 | 0.38x | 1.30x | 3.44x |
| kv_proj | 1 | fused_dense_sparse | 25.65 | 80.82 | 13.98 | 0.32x | 1.84x | 5.78x |
| kv_proj | 32 | activation_quant |  -  | 70.15 | 16.31 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 27.15 | 70.22 | 41.15 | 0.39x | 0.66x | 1.71x |
| kv_proj | 32 | sparse_gemm | 27.16 | 68.07 | 19.69 | 0.40x | 1.38x | 3.46x |
| kv_proj | 32 | fused_dense_sparse | 27.15 | 80.97 | 21.04 | 0.34x | 1.29x | 3.85x |
| kv_proj | 128 | activation_quant |  -  | 70.17 | 16.51 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 30.24 | 69.14 | 69.10 | 0.44x | 0.44x | 1.00x |
| kv_proj | 128 | sparse_gemm | 30.28 | 68.50 | 21.38 | 0.44x | 1.42x | 3.20x |
| kv_proj | 128 | fused_dense_sparse | 30.34 | 81.19 | 26.32 | 0.37x | 1.15x | 3.09x |
| kv_proj | 512 | activation_quant |  -  | 70.29 | 16.33 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 62.69 | 107.47 | 80.16 | 0.58x | 0.78x | 1.34x |
| kv_proj | 512 | sparse_gemm | 62.68 | 68.12 | 20.87 | 0.92x | 3.00x | 3.26x |
| kv_proj | 512 | fused_dense_sparse | 62.66 | 124.78 | 61.29 | 0.50x | 1.02x | 2.04x |
| o_proj | 1 | activation_quant |  -  | 48.99 | 16.17 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 60.97 | 69.13 | 22.75 | 0.88x | 2.68x | 3.04x |
| o_proj | 1 | sparse_gemm | 61.02 | 67.86 | 19.55 | 0.90x | 3.12x | 3.47x |
| o_proj | 1 | fused_dense_sparse | 60.95 | 81.01 | 24.23 | 0.75x | 2.52x | 3.34x |
| o_proj | 32 | activation_quant |  -  | 69.65 | 16.49 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 64.27 | 68.81 | 107.70 | 0.93x | 0.60x | 0.64x |
| o_proj | 32 | sparse_gemm | 64.24 | 67.43 | 19.88 | 0.95x | 3.23x | 3.39x |
| o_proj | 32 | fused_dense_sparse | 64.30 | 81.15 | 26.79 | 0.79x | 2.40x | 3.03x |
| o_proj | 128 | activation_quant |  -  | 70.33 | 16.63 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 69.12 | 106.89 | 128.62 | 0.65x | 0.54x | 0.83x |
| o_proj | 128 | sparse_gemm | 69.05 | 67.79 | 19.54 | 1.02x | 3.53x | 3.47x |
| o_proj | 128 | fused_dense_sparse | 69.09 | 123.78 | 53.23 | 0.56x | 1.30x | 2.33x |
| o_proj | 512 | activation_quant |  -  | 70.25 | 16.41 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 172.12 | 274.46 | 224.50 | 0.63x | 0.77x | 1.22x |
| o_proj | 512 | sparse_gemm | 172.35 | 67.38 | 31.16 | 2.56x | 5.53x | 2.16x |
| o_proj | 512 | fused_dense_sparse | 172.25 | 297.74 | 161.93 | 0.58x | 1.06x | 1.84x |
| q_proj | 1 | activation_quant |  -  | 53.31 | 16.27 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 61.49 | 69.63 | 22.58 | 0.88x | 2.72x | 3.08x |
| q_proj | 1 | sparse_gemm | 61.03 | 69.25 | 19.78 | 0.88x | 3.08x | 3.50x |
| q_proj | 1 | fused_dense_sparse | 61.03 | 81.18 | 24.08 | 0.75x | 2.53x | 3.37x |
| q_proj | 32 | activation_quant |  -  | 69.31 | 16.30 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 64.36 | 67.73 | 107.28 | 0.95x | 0.60x | 0.63x |
| q_proj | 32 | sparse_gemm | 64.34 | 68.21 | 20.08 | 0.94x | 3.20x | 3.40x |
| q_proj | 32 | fused_dense_sparse | 64.30 | 80.85 | 26.71 | 0.80x | 2.41x | 3.03x |
| q_proj | 128 | activation_quant |  -  | 69.98 | 16.47 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 69.08 | 106.83 | 128.57 | 0.65x | 0.54x | 0.83x |
| q_proj | 128 | sparse_gemm | 69.07 | 67.88 | 19.58 | 1.02x | 3.53x | 3.47x |
| q_proj | 128 | fused_dense_sparse | 69.29 | 123.69 | 53.25 | 0.56x | 1.30x | 2.32x |
| q_proj | 512 | activation_quant |  -  | 70.25 | 16.29 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 171.27 | 272.92 | 223.44 | 0.63x | 0.77x | 1.22x |
| q_proj | 512 | sparse_gemm | 171.13 | 67.90 | 30.83 | 2.52x | 5.55x | 2.20x |
| q_proj | 512 | fused_dense_sparse | 171.43 | 297.43 | 161.70 | 0.58x | 1.06x | 1.84x |

### Qwen2.5-32B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 274.22 | 66.68 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 302.96 | 317.58 | 296.33 | 0.95x | 1.02x | 1.07x |
| down_proj | 1 | sparse_gemm | 302.73 | 68.24 | 19.69 | 4.44x | 15.38x | 3.47x |
| down_proj | 1 | fused_dense_sparse | 302.82 | 399.80 | 287.82 | 0.76x | 1.05x | 1.39x |
| down_proj | 32 | activation_quant |  -  | 363.89 | 119.45 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 326.40 | 317.70 | 554.57 | 1.03x | 0.59x | 0.57x |
| down_proj | 32 | sparse_gemm | 326.48 | 67.68 | 25.11 | 4.82x | 13.00x | 2.70x |
| down_proj | 32 | fused_dense_sparse | 326.48 | 342.07 | 156.00 | 0.95x | 2.09x | 2.19x |
| down_proj | 128 | activation_quant |  -  | 365.97 | 119.77 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 378.28 | 564.20 | 778.06 | 0.67x | 0.49x | 0.73x |
| down_proj | 128 | sparse_gemm | 378.18 | 67.81 | 41.54 | 5.58x | 9.10x | 1.63x |
| down_proj | 128 | fused_dense_sparse | 378.18 | 616.89 | 247.77 | 0.61x | 1.53x | 2.49x |
| down_proj | 512 | activation_quant |  -  | 364.99 | 120.36 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 909.52 | 1465.90 | 1729.16 | 0.62x | 0.53x | 0.85x |
| down_proj | 512 | sparse_gemm | 911.42 | 93.23 | 67.92 | 9.78x | 13.42x | 1.37x |
| down_proj | 512 | fused_dense_sparse | 911.79 | 1553.53 | 1120.72 | 0.59x | 0.81x | 1.39x |
| gate_up_proj | 1 | activation_quant |  -  | 49.27 | 16.36 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 596.86 | 296.81 | 280.51 | 2.01x | 2.13x | 1.06x |
| gate_up_proj | 1 | sparse_gemm | 596.90 | 69.25 | 20.01 | 8.62x | 29.83x | 3.46x |
| gate_up_proj | 1 | fused_dense_sparse | 596.82 | 306.39 | 288.70 | 1.95x | 2.07x | 1.06x |
| gate_up_proj | 32 | activation_quant |  -  | 78.56 | 16.72 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 664.88 | 299.80 | 545.40 | 2.22x | 1.22x | 0.55x |
| gate_up_proj | 32 | sparse_gemm | 664.92 | 69.03 | 20.43 | 9.63x | 32.55x | 3.38x |
| gate_up_proj | 32 | fused_dense_sparse | 665.20 | 310.32 | 360.33 | 2.14x | 1.85x | 0.86x |
| gate_up_proj | 128 | activation_quant |  -  | 70.93 | 16.61 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 673.63 | 653.41 | 1052.52 | 1.03x | 0.64x | 0.62x |
| gate_up_proj | 128 | sparse_gemm | 673.83 | 68.78 | 67.75 | 9.80x | 9.95x | 1.02x |
| gate_up_proj | 128 | fused_dense_sparse | 673.28 | 680.74 | 679.06 | 0.99x | 0.99x | 1.00x |
| gate_up_proj | 512 | activation_quant |  -  | 71.15 | 16.65 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 1844.08 | 2448.95 | 4060.81 | 0.75x | 0.45x | 0.60x |
| gate_up_proj | 512 | sparse_gemm | 1837.22 | 167.54 | 244.82 | 10.97x | 7.50x | 0.68x |
| gate_up_proj | 512 | fused_dense_sparse | 1842.45 | 2578.31 | 2564.94 | 0.71x | 0.72x | 1.01x |
| kv_proj | 1 | activation_quant |  -  | 49.27 | 16.13 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 25.57 | 68.59 | 13.29 | 0.37x | 1.92x | 5.16x |
| kv_proj | 1 | sparse_gemm | 25.61 | 67.71 | 19.90 | 0.38x | 1.29x | 3.40x |
| kv_proj | 1 | fused_dense_sparse | 25.65 | 81.22 | 13.96 | 0.32x | 1.84x | 5.82x |
| kv_proj | 32 | activation_quant |  -  | 70.53 | 16.64 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 27.19 | 68.97 | 41.38 | 0.39x | 0.66x | 1.67x |
| kv_proj | 32 | sparse_gemm | 27.25 | 68.26 | 19.82 | 0.40x | 1.37x | 3.44x |
| kv_proj | 32 | fused_dense_sparse | 27.18 | 81.91 | 21.12 | 0.33x | 1.29x | 3.88x |
| kv_proj | 128 | activation_quant |  -  | 70.53 | 16.52 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 30.33 | 68.91 | 69.49 | 0.44x | 0.44x | 0.99x |
| kv_proj | 128 | sparse_gemm | 30.29 | 67.57 | 21.49 | 0.45x | 1.41x | 3.14x |
| kv_proj | 128 | fused_dense_sparse | 30.23 | 80.38 | 26.46 | 0.38x | 1.14x | 3.04x |
| kv_proj | 512 | activation_quant |  -  | 70.65 | 16.41 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 63.12 | 107.93 | 80.45 | 0.58x | 0.78x | 1.34x |
| kv_proj | 512 | sparse_gemm | 62.95 | 67.89 | 21.01 | 0.93x | 3.00x | 3.23x |
| kv_proj | 512 | fused_dense_sparse | 63.13 | 124.54 | 61.19 | 0.51x | 1.03x | 2.04x |
| o_proj | 1 | activation_quant |  -  | 48.99 | 16.10 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 60.98 | 69.26 | 22.76 | 0.88x | 2.68x | 3.04x |
| o_proj | 1 | sparse_gemm | 61.00 | 67.92 | 19.76 | 0.90x | 3.09x | 3.44x |
| o_proj | 1 | fused_dense_sparse | 60.95 | 81.06 | 24.31 | 0.75x | 2.51x | 3.33x |
| o_proj | 32 | activation_quant |  -  | 70.03 | 16.45 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 64.41 | 68.86 | 108.23 | 0.94x | 0.60x | 0.64x |
| o_proj | 32 | sparse_gemm | 64.35 | 67.39 | 19.74 | 0.95x | 3.26x | 3.41x |
| o_proj | 32 | fused_dense_sparse | 64.29 | 80.92 | 27.36 | 0.79x | 2.35x | 2.96x |
| o_proj | 128 | activation_quant |  -  | 70.74 | 16.38 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 69.36 | 107.37 | 129.35 | 0.65x | 0.54x | 0.83x |
| o_proj | 128 | sparse_gemm | 69.42 | 67.73 | 19.53 | 1.02x | 3.55x | 3.47x |
| o_proj | 128 | fused_dense_sparse | 69.50 | 124.32 | 53.38 | 0.56x | 1.30x | 2.33x |
| o_proj | 512 | activation_quant |  -  | 70.67 | 16.42 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 173.82 | 275.27 | 225.77 | 0.63x | 0.77x | 1.22x |
| o_proj | 512 | sparse_gemm | 174.34 | 67.62 | 31.25 | 2.58x | 5.58x | 2.16x |
| o_proj | 512 | fused_dense_sparse | 174.46 | 298.83 | 162.37 | 0.58x | 1.07x | 1.84x |
| q_proj | 1 | activation_quant |  -  | 49.22 | 16.08 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 60.99 | 69.39 | 22.76 | 0.88x | 2.68x | 3.05x |
| q_proj | 1 | sparse_gemm | 60.99 | 68.28 | 19.87 | 0.89x | 3.07x | 3.44x |
| q_proj | 1 | fused_dense_sparse | 60.96 | 81.44 | 24.32 | 0.75x | 2.51x | 3.35x |
| q_proj | 32 | activation_quant |  -  | 70.03 | 16.26 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 64.42 | 68.72 | 108.14 | 0.94x | 0.60x | 0.64x |
| q_proj | 32 | sparse_gemm | 64.39 | 68.27 | 19.92 | 0.94x | 3.23x | 3.43x |
| q_proj | 32 | fused_dense_sparse | 64.45 | 80.62 | 27.29 | 0.80x | 2.36x | 2.95x |
| q_proj | 128 | activation_quant |  -  | 70.75 | 16.41 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 69.42 | 107.37 | 129.22 | 0.65x | 0.54x | 0.83x |
| q_proj | 128 | sparse_gemm | 69.46 | 67.60 | 19.62 | 1.03x | 3.54x | 3.45x |
| q_proj | 128 | fused_dense_sparse | 69.36 | 124.26 | 53.44 | 0.56x | 1.30x | 2.33x |
| q_proj | 512 | activation_quant |  -  | 70.65 | 16.52 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 173.79 | 274.68 | 225.84 | 0.63x | 0.77x | 1.22x |
| q_proj | 512 | sparse_gemm | 174.05 | 66.74 | 31.16 | 2.61x | 5.59x | 2.14x |
| q_proj | 512 | fused_dense_sparse | 174.49 | 297.36 | 162.53 | 0.59x | 1.07x | 1.83x |

### LLaMA3-70B - sub-kernels
| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| down_proj | 1 | activation_quant |  -  | 284.32 | 69.67 |  -  |  -  |  -  |
| down_proj | 1 | dense_gemm | 497.91 | 360.97 | 457.01 | 1.38x | 1.09x | 0.79x |
| down_proj | 1 | sparse_gemm | 497.78 | 96.67 | 27.03 | 5.15x | 18.41x | 3.58x |
| down_proj | 1 | fused_dense_sparse | 497.75 | 413.42 | 346.54 | 1.20x | 1.44x | 1.19x |
| down_proj | 32 | activation_quant |  -  | 375.67 | 123.86 |  -  |  -  |  -  |
| down_proj | 32 | dense_gemm | 528.73 | 345.55 | 657.22 | 1.53x | 0.80x | 0.53x |
| down_proj | 32 | sparse_gemm | 528.88 | 68.27 | 26.02 | 7.75x | 20.33x | 2.62x |
| down_proj | 32 | fused_dense_sparse | 528.71 | 367.38 | 308.24 | 1.44x | 1.72x | 1.19x |
| down_proj | 128 | activation_quant |  -  | 376.28 | 124.03 |  -  |  -  |  -  |
| down_proj | 128 | dense_gemm | 602.56 | 600.83 | 830.41 | 1.00x | 0.73x | 0.72x |
| down_proj | 128 | sparse_gemm | 603.26 | 97.25 | 46.90 | 6.20x | 12.86x | 2.07x |
| down_proj | 128 | fused_dense_sparse | 602.82 | 653.92 | 377.71 | 0.92x | 1.60x | 1.73x |
| down_proj | 512 | activation_quant |  -  | 377.21 | 124.77 |  -  |  -  |  -  |
| down_proj | 512 | dense_gemm | 1523.37 | 1997.25 | 3414.96 | 0.76x | 0.45x | 0.58x |
| down_proj | 512 | sparse_gemm | 1542.05 | 134.87 | 93.51 | 11.43x | 16.49x | 1.44x |
| down_proj | 512 | fused_dense_sparse | 1546.68 | 2132.32 | 1430.00 | 0.73x | 1.08x | 1.49x |
| gate_up_proj | 1 | activation_quant |  -  | 77.78 | 16.46 |  -  |  -  |  -  |
| gate_up_proj | 1 | dense_gemm | 985.87 | 520.67 | 501.20 | 1.89x | 1.97x | 1.04x |
| gate_up_proj | 1 | sparse_gemm | 985.98 | 69.98 | 19.59 | 14.09x | 50.33x | 3.57x |
| gate_up_proj | 1 | fused_dense_sparse | 985.89 | 538.18 | 512.17 | 1.83x | 1.92x | 1.05x |
| gate_up_proj | 32 | activation_quant |  -  | 111.62 | 18.17 |  -  |  -  |  -  |
| gate_up_proj | 32 | dense_gemm | 1099.72 | 524.28 | 752.45 | 2.10x | 1.46x | 0.70x |
| gate_up_proj | 32 | sparse_gemm | 1099.91 | 69.96 | 25.38 | 15.72x | 43.33x | 2.76x |
| gate_up_proj | 32 | fused_dense_sparse | 1099.78 | 541.87 | 569.34 | 2.03x | 1.93x | 0.95x |
| gate_up_proj | 128 | activation_quant |  -  | 109.88 | 18.41 |  -  |  -  |  -  |
| gate_up_proj | 128 | dense_gemm | 1143.20 | 1060.23 | 2176.51 | 1.08x | 0.53x | 0.49x |
| gate_up_proj | 128 | sparse_gemm | 1142.39 | 76.08 | 76.81 | 15.02x | 14.87x | 0.99x |
| gate_up_proj | 128 | fused_dense_sparse | 1142.90 | 1143.63 | 1121.43 | 1.00x | 1.02x | 1.02x |
| gate_up_proj | 512 | activation_quant |  -  | 109.96 | 20.87 |  -  |  -  |  -  |
| gate_up_proj | 512 | dense_gemm | 2988.44 | 4128.89 | 6899.54 | 0.72x | 0.43x | 0.60x |
| gate_up_proj | 512 | sparse_gemm | 2998.54 | 249.06 | 280.20 | 12.04x | 10.70x | 0.89x |
| gate_up_proj | 512 | fused_dense_sparse | 3006.28 | 4338.41 | 4346.17 | 0.69x | 0.69x | 1.00x |
| kv_proj | 1 | activation_quant |  -  | 77.75 | 16.45 |  -  |  -  |  -  |
| kv_proj | 1 | dense_gemm | 39.13 | 95.55 | 20.04 | 0.41x | 1.95x | 4.77x |
| kv_proj | 1 | sparse_gemm | 39.11 | 69.40 | 19.28 | 0.56x | 2.03x | 3.60x |
| kv_proj | 1 | fused_dense_sparse | 39.13 | 107.44 | 20.69 | 0.36x | 1.89x | 5.19x |
| kv_proj | 32 | activation_quant |  -  | 109.04 | 18.16 |  -  |  -  |  -  |
| kv_proj | 32 | dense_gemm | 40.57 | 96.28 | 63.85 | 0.42x | 0.64x | 1.51x |
| kv_proj | 32 | sparse_gemm | 40.50 | 67.16 | 19.52 | 0.60x | 2.08x | 3.44x |
| kv_proj | 32 | fused_dense_sparse | 40.54 | 105.55 | 30.90 | 0.38x | 1.31x | 3.42x |
| kv_proj | 128 | activation_quant |  -  | 108.83 | 18.31 |  -  |  -  |  -  |
| kv_proj | 128 | dense_gemm | 46.83 | 97.13 | 92.93 | 0.48x | 0.50x | 1.05x |
| kv_proj | 128 | sparse_gemm | 46.79 | 67.71 | 19.39 | 0.69x | 2.41x | 3.49x |
| kv_proj | 128 | fused_dense_sparse | 46.94 | 105.39 | 42.09 | 0.45x | 1.12x | 2.50x |
| kv_proj | 512 | activation_quant |  -  | 109.45 | 21.06 |  -  |  -  |  -  |
| kv_proj | 512 | dense_gemm | 100.13 | 173.78 | 110.49 | 0.58x | 0.91x | 1.57x |
| kv_proj | 512 | sparse_gemm | 100.57 | 67.78 | 20.58 | 1.48x | 4.89x | 3.29x |
| kv_proj | 512 | fused_dense_sparse | 100.33 | 193.54 | 91.91 | 0.52x | 1.09x | 2.11x |
| o_proj | 1 | activation_quant |  -  | 77.36 | 16.37 |  -  |  -  |  -  |
| o_proj | 1 | dense_gemm | 146.66 | 103.79 | 52.89 | 1.41x | 2.77x | 1.96x |
| o_proj | 1 | sparse_gemm | 146.69 | 67.70 | 19.38 | 2.17x | 7.57x | 3.49x |
| o_proj | 1 | fused_dense_sparse | 146.51 | 125.50 | 55.30 | 1.17x | 2.65x | 2.27x |
| o_proj | 32 | activation_quant |  -  | 109.13 | 18.19 |  -  |  -  |  -  |
| o_proj | 32 | dense_gemm | 172.08 | 97.48 | 93.35 | 1.77x | 1.84x | 1.04x |
| o_proj | 32 | sparse_gemm | 172.19 | 68.22 | 19.56 | 2.52x | 8.80x | 3.49x |
| o_proj | 32 | fused_dense_sparse | 172.14 | 106.89 | 55.51 | 1.61x | 3.10x | 1.93x |
| o_proj | 128 | activation_quant |  -  | 109.48 | 18.41 |  -  |  -  |  -  |
| o_proj | 128 | dense_gemm | 191.26 | 173.49 | 110.62 | 1.10x | 1.73x | 1.57x |
| o_proj | 128 | sparse_gemm | 190.98 | 67.93 | 23.85 | 2.81x | 8.01x | 2.85x |
| o_proj | 128 | fused_dense_sparse | 190.83 | 199.25 | 93.52 | 0.96x | 2.04x | 2.13x |
| o_proj | 512 | activation_quant |  -  | 109.47 | 20.81 |  -  |  -  |  -  |
| o_proj | 512 | dense_gemm | 398.14 | 583.12 | 432.96 | 0.68x | 0.92x | 1.35x |
| o_proj | 512 | sparse_gemm | 398.78 | 67.78 | 49.40 | 5.88x | 8.07x | 1.37x |
| o_proj | 512 | fused_dense_sparse | 399.38 | 627.76 | 356.17 | 0.64x | 1.12x | 1.76x |
| q_proj | 1 | activation_quant |  -  | 77.36 | 16.37 |  -  |  -  |  -  |
| q_proj | 1 | dense_gemm | 146.65 | 103.62 | 52.78 | 1.42x | 2.78x | 1.96x |
| q_proj | 1 | sparse_gemm | 146.66 | 67.80 | 19.63 | 2.16x | 7.47x | 3.45x |
| q_proj | 1 | fused_dense_sparse | 146.73 | 124.75 | 54.79 | 1.18x | 2.68x | 2.28x |
| q_proj | 32 | activation_quant |  -  | 108.55 | 18.07 |  -  |  -  |  -  |
| q_proj | 32 | dense_gemm | 171.93 | 97.09 | 92.82 | 1.77x | 1.85x | 1.05x |
| q_proj | 32 | sparse_gemm | 171.86 | 67.94 | 19.59 | 2.53x | 8.77x | 3.47x |
| q_proj | 32 | fused_dense_sparse | 171.84 | 106.35 | 55.21 | 1.62x | 3.11x | 1.93x |
| q_proj | 128 | activation_quant |  -  | 109.43 | 18.40 |  -  |  -  |  -  |
| q_proj | 128 | dense_gemm | 190.89 | 173.58 | 110.64 | 1.10x | 1.73x | 1.57x |
| q_proj | 128 | sparse_gemm | 190.84 | 68.42 | 23.84 | 2.79x | 8.01x | 2.87x |
| q_proj | 128 | fused_dense_sparse | 190.95 | 199.16 | 93.50 | 0.96x | 2.04x | 2.13x |
| q_proj | 512 | activation_quant |  -  | 109.46 | 20.82 |  -  |  -  |  -  |
| q_proj | 512 | dense_gemm | 394.72 | 581.47 | 432.88 | 0.68x | 0.91x | 1.34x |
| q_proj | 512 | sparse_gemm | 397.62 | 69.77 | 48.98 | 5.70x | 8.12x | 1.42x |
| q_proj | 512 | fused_dense_sparse | 395.91 | 627.82 | 356.12 | 0.63x | 1.11x | 1.76x |


## 4. End-to-end speedup (CUDA over Triton)

Rows: projection. Cells: `triton_us / cuda_us` (>1.0x means CUDA wins).


### Qwen3-14B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 5120->5120 | **5.03x** | **3.31x** | **2.91x** | **2.11x** |
| kv_proj | 5120->2048 | **7.70x** | **3.86x** | **3.44x** | **2.61x** |
| o_proj | 5120->5120 | **5.01x** | **3.32x** | **2.90x** | **2.11x** |
| gate_up_proj | 5120->34816 | **1.42x** | **1.04x** | **1.16x** | **1.13x** |
| down_proj | 17408->5120 | **4.99x** | **3.66x** | **3.24x** | **1.69x** |

### Qwen2.5-32B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 5120->5120 | **4.96x** | **3.31x** | **2.90x** | **2.11x** |
| kv_proj | 5120->2048 | **7.58x** | **3.86x** | **3.43x** | **2.61x** |
| o_proj | 5120->5120 | **5.00x** | **3.31x** | **2.90x** | **2.11x** |
| gate_up_proj | 5120->55296 | **1.21x** | **1.02x** | **1.07x** | **1.02x** |
| down_proj | 27648->5120 | **1.94x** | **2.44x** | **2.67x** | **1.49x** |

### LLaMA3-70B
| proj | shape | T=1 | T=32 | T=128 | T=512 |
|---|---|---:|---:|---:|---:|
| q_proj | 8192->8192 | **3.07x** | **2.91x** | **2.76x** | **1.96x** |
| kv_proj | 8192->2048 | **7.04x** | **4.37x** | **3.56x** | **2.71x** |
| o_proj | 8192->8192 | **3.04x** | **2.91x** | **2.76x** | **1.95x** |
| gate_up_proj | 8192->57344 | **1.29x** | **1.11x** | **1.07x** | **1.01x** |
| down_proj | 28672->8192 | **1.69x** | **1.76x** | **2.06x** | **1.64x** |


## 5. CUDA end-to-end bottleneck hint

For each shape, compare CUDA `activation_quant` against CUDA `fused_dense_sparse`. A larger `quant_share` means launch/prologue dominates; a larger `fused_share` means the main CUDA matmul kernel dominates.

| model | proj | T | shape | quant_us | fused_us | quant_share | fused_share | likely_bottleneck |
|---|---|---:|---|---:|---:|---:|---:|---|
| Qwen3-14B | q_proj | 1 | 5120->5120 | 16.27 | 24.08 | 40.3% | 59.7% | quant/prologue dominated |
| Qwen3-14B | q_proj | 32 | 5120->5120 | 16.30 | 26.71 | 37.9% | 62.1% | quant/prologue dominated |
| Qwen3-14B | q_proj | 128 | 5120->5120 | 16.47 | 53.25 | 23.6% | 76.4% | mixed |
| Qwen3-14B | q_proj | 512 | 5120->5120 | 16.29 | 161.70 | 9.2% | 90.8% | main fused kernel dominated |
| Qwen3-14B | kv_proj | 1 | 5120->2048 | 16.21 | 13.98 | 53.7% | 46.3% | quant/prologue dominated |
| Qwen3-14B | kv_proj | 32 | 5120->2048 | 16.31 | 21.04 | 43.7% | 56.3% | quant/prologue dominated |
| Qwen3-14B | kv_proj | 128 | 5120->2048 | 16.51 | 26.32 | 38.5% | 61.5% | quant/prologue dominated |
| Qwen3-14B | kv_proj | 512 | 5120->2048 | 16.33 | 61.29 | 21.0% | 79.0% | mixed |
| Qwen3-14B | o_proj | 1 | 5120->5120 | 16.17 | 24.23 | 40.0% | 60.0% | quant/prologue dominated |
| Qwen3-14B | o_proj | 32 | 5120->5120 | 16.49 | 26.79 | 38.1% | 61.9% | quant/prologue dominated |
| Qwen3-14B | o_proj | 128 | 5120->5120 | 16.63 | 53.23 | 23.8% | 76.2% | mixed |
| Qwen3-14B | o_proj | 512 | 5120->5120 | 16.41 | 161.93 | 9.2% | 90.8% | main fused kernel dominated |
| Qwen3-14B | gate_up_proj | 1 | 5120->34816 | 16.43 | 184.27 | 8.2% | 91.8% | main fused kernel dominated |
| Qwen3-14B | gate_up_proj | 32 | 5120->34816 | 16.56 | 282.76 | 5.5% | 94.5% | main fused kernel dominated |
| Qwen3-14B | gate_up_proj | 128 | 5120->34816 | 16.77 | 451.90 | 3.6% | 96.4% | main fused kernel dominated |
| Qwen3-14B | gate_up_proj | 512 | 5120->34816 | 16.70 | 1471.33 | 1.1% | 98.9% | main fused kernel dominated |
| Qwen3-14B | down_proj | 1 | 17408->5120 | 34.36 | 76.11 | 31.1% | 68.9% | mixed |
| Qwen3-14B | down_proj | 32 | 17408->5120 | 34.29 | 89.44 | 27.7% | 72.3% | mixed |
| Qwen3-14B | down_proj | 128 | 17408->5120 | 34.84 | 158.45 | 18.0% | 82.0% | main fused kernel dominated |
| Qwen3-14B | down_proj | 512 | 17408->5120 | 69.98 | 653.72 | 9.7% | 90.3% | main fused kernel dominated |
| Qwen2.5-32B | q_proj | 1 | 5120->5120 | 16.08 | 24.32 | 39.8% | 60.2% | quant/prologue dominated |
| Qwen2.5-32B | q_proj | 32 | 5120->5120 | 16.26 | 27.29 | 37.3% | 62.7% | quant/prologue dominated |
| Qwen2.5-32B | q_proj | 128 | 5120->5120 | 16.41 | 53.44 | 23.5% | 76.5% | mixed |
| Qwen2.5-32B | q_proj | 512 | 5120->5120 | 16.52 | 162.53 | 9.2% | 90.8% | main fused kernel dominated |
| Qwen2.5-32B | kv_proj | 1 | 5120->2048 | 16.13 | 13.96 | 53.6% | 46.4% | quant/prologue dominated |
| Qwen2.5-32B | kv_proj | 32 | 5120->2048 | 16.64 | 21.12 | 44.1% | 55.9% | quant/prologue dominated |
| Qwen2.5-32B | kv_proj | 128 | 5120->2048 | 16.52 | 26.46 | 38.4% | 61.6% | quant/prologue dominated |
| Qwen2.5-32B | kv_proj | 512 | 5120->2048 | 16.41 | 61.19 | 21.1% | 78.9% | mixed |
| Qwen2.5-32B | o_proj | 1 | 5120->5120 | 16.10 | 24.31 | 39.8% | 60.2% | quant/prologue dominated |
| Qwen2.5-32B | o_proj | 32 | 5120->5120 | 16.45 | 27.36 | 37.5% | 62.5% | quant/prologue dominated |
| Qwen2.5-32B | o_proj | 128 | 5120->5120 | 16.38 | 53.38 | 23.5% | 76.5% | mixed |
| Qwen2.5-32B | o_proj | 512 | 5120->5120 | 16.42 | 162.37 | 9.2% | 90.8% | main fused kernel dominated |
| Qwen2.5-32B | gate_up_proj | 1 | 5120->55296 | 16.36 | 288.70 | 5.4% | 94.6% | main fused kernel dominated |
| Qwen2.5-32B | gate_up_proj | 32 | 5120->55296 | 16.72 | 360.33 | 4.4% | 95.6% | main fused kernel dominated |
| Qwen2.5-32B | gate_up_proj | 128 | 5120->55296 | 16.61 | 679.06 | 2.4% | 97.6% | main fused kernel dominated |
| Qwen2.5-32B | gate_up_proj | 512 | 5120->55296 | 16.65 | 2564.94 | 0.6% | 99.4% | main fused kernel dominated |
| Qwen2.5-32B | down_proj | 1 | 27648->5120 | 66.68 | 287.82 | 18.8% | 81.2% | main fused kernel dominated |
| Qwen2.5-32B | down_proj | 32 | 27648->5120 | 119.45 | 156.00 | 43.4% | 56.6% | quant/prologue dominated |
| Qwen2.5-32B | down_proj | 128 | 27648->5120 | 119.77 | 247.77 | 32.6% | 67.4% | mixed |
| Qwen2.5-32B | down_proj | 512 | 27648->5120 | 120.36 | 1120.72 | 9.7% | 90.3% | main fused kernel dominated |
| LLaMA3-70B | q_proj | 1 | 8192->8192 | 16.37 | 54.79 | 23.0% | 77.0% | mixed |
| LLaMA3-70B | q_proj | 32 | 8192->8192 | 18.07 | 55.21 | 24.7% | 75.3% | mixed |
| LLaMA3-70B | q_proj | 128 | 8192->8192 | 18.40 | 93.50 | 16.4% | 83.6% | main fused kernel dominated |
| LLaMA3-70B | q_proj | 512 | 8192->8192 | 20.82 | 356.12 | 5.5% | 94.5% | main fused kernel dominated |
| LLaMA3-70B | kv_proj | 1 | 8192->2048 | 16.45 | 20.69 | 44.3% | 55.7% | quant/prologue dominated |
| LLaMA3-70B | kv_proj | 32 | 8192->2048 | 18.16 | 30.90 | 37.0% | 63.0% | quant/prologue dominated |
| LLaMA3-70B | kv_proj | 128 | 8192->2048 | 18.31 | 42.09 | 30.3% | 69.7% | mixed |
| LLaMA3-70B | kv_proj | 512 | 8192->2048 | 21.06 | 91.91 | 18.6% | 81.4% | main fused kernel dominated |
| LLaMA3-70B | o_proj | 1 | 8192->8192 | 16.37 | 55.30 | 22.8% | 77.2% | mixed |
| LLaMA3-70B | o_proj | 32 | 8192->8192 | 18.19 | 55.51 | 24.7% | 75.3% | mixed |
| LLaMA3-70B | o_proj | 128 | 8192->8192 | 18.41 | 93.52 | 16.4% | 83.6% | main fused kernel dominated |
| LLaMA3-70B | o_proj | 512 | 8192->8192 | 20.81 | 356.17 | 5.5% | 94.5% | main fused kernel dominated |
| LLaMA3-70B | gate_up_proj | 1 | 8192->57344 | 16.46 | 512.17 | 3.1% | 96.9% | main fused kernel dominated |
| LLaMA3-70B | gate_up_proj | 32 | 8192->57344 | 18.17 | 569.34 | 3.1% | 96.9% | main fused kernel dominated |
| LLaMA3-70B | gate_up_proj | 128 | 8192->57344 | 18.41 | 1121.43 | 1.6% | 98.4% | main fused kernel dominated |
| LLaMA3-70B | gate_up_proj | 512 | 8192->57344 | 20.87 | 4346.17 | 0.5% | 99.5% | main fused kernel dominated |
| LLaMA3-70B | down_proj | 1 | 28672->8192 | 69.67 | 346.54 | 16.7% | 83.3% | main fused kernel dominated |
| LLaMA3-70B | down_proj | 32 | 28672->8192 | 123.86 | 308.24 | 28.7% | 71.3% | mixed |
| LLaMA3-70B | down_proj | 128 | 28672->8192 | 124.03 | 377.71 | 24.7% | 75.3% | mixed |
| LLaMA3-70B | down_proj | 512 | 28672->8192 | 124.77 | 1430.00 | 8.0% | 92.0% | main fused kernel dominated |
