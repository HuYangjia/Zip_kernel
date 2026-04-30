# r62 F2 Final Qwen3 roofline report (cold-cache, 4 models x 4 T)

Source: `/root/Zip_kernel/cuda_kernel/logs/r62_f2_final/qwen3_20260430_122555/bench.json`

GPU model: RTX 4090 (vendor spec, ACHIEVABLE_FRACTION=0.85)

## §1 Hardware constants and formulas

| Parameter | Value | Note |
| --- | --- | --- |
| HBM bandwidth | 1008 GB/s | RTX 4090 vendor spec |
| FP16/BF16 TC peak | 165.2 TFLOPS | boost clock, no sparsity |
| INT4 TC peak | 660.6 TOPS | boost clock |
| ACHIEVABLE_FRACTION | 0.85 | engineering derating |

**Formulas** (all time in us, eff_* = peak × ACHIEVABLE_FRACTION):

- **FP16 roofline**: `t = max(2·T·d_in·d_out / eff_fp16, (2·d_in·d_out + 2·T·d_in + 2·T·d_out) / eff_hbm)`
- **CUDA quant (T>=2)**: mem-bound, `bytes = 2·T·d_in + 0.5·T·d_in + 2·T + 4·T·n_groups`
- **CUDA GEMM (T>=2)**: `t = max(2·T·d_in·d_out / eff_int4, bytes_gemm / eff_hbm)` with `bytes_gemm = 0.5·d_in·d_out + 0.5·T·d_in + 4·d_out·n_g + 4·T·n_g + 2·T + 2·T·d_out`
- **CUDA T=1 fused**: `bytes = 0.5·d_in·d_out + 2·d_in + 4·d_out·n_g + 2·d_out`, compute = `2·d_in·d_out / eff_int4`
- **CUDA e2e (T>=2)** = `t_quant + t_gemm` (serial)

### Systematic biases
1. Kernel launch overhead (5-10us/launch) not in the roofline — T<=8 rows will under-estimate achievable time.
2. L2 cache reuse — in the cold-cache bench we explicitly L2-flush the FP16 side before each sample, so FP16 efficiency numbers here are honest (no warm-cache cheating).
3. Epilogue FMA cost (dequant) folded into the INT4 TC peak — slightly optimistic.

## §2 FP16 efficiency distribution (by T)

`fp16_eff = fp16_roof_us / fp16_us` — how close cuBLAS (cold-cache) gets to the RTX 4090 physical limit.

| label | n | min | median | max |
|---|---:|---:|---:|---:|
| 1 | 20 | 64% | 93% | 109% |
| 32 | 20 | 62% | 95% | 104% |
| 128 | 20 | 75% | 90% | 98% |
| 512 | 20 | 95% | 123% | 151% |

## §3 CUDA efficiency distribution

`cuda_eff = cuda_roof_us / cuda_us` — how close our W4A4 kernel gets to its own roofline.

### §3.1 By T

| label | n | min | median | max |
|---|---:|---:|---:|---:|
| 1 | 20 | 17% | 39% | 66% |
| 32 | 20 | 5% | 19% | 88% |
| 128 | 20 | 7% | 22% | 48% |
| 512 | 20 | 19% | 31% | 43% |

### §3.2 By proj

| label | n | min | median | max |
|---|---:|---:|---:|---:|
| q_proj | 16 | 5% | 27% | 56% |
| kv_proj | 16 | 5% | 18% | 38% |
| o_proj | 16 | 5% | 25% | 56% |
| gate_up_proj | 16 | 15% | 45% | 88% |
| down_proj | 16 | 7% | 30% | 59% |

## §4 Per-shape detail

Per-row: measured time, roofline time, efficiency, and actual-vs-roof speedup against FP16.

### Qwen3-0.6B

| proj | T | shape | fp16_us | fp16_roof | fp16_eff | cuda_us | cuda_roof | cuda_eff | speedup (actual / roof) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| q_proj | 1 | 1024→2048 | 7.38 | 4.90 | 66% | 7.61 | 1.31 | 17% | 0.97× (3.75×) |
| q_proj | 32 | 1024→2048 | 8.21 | 5.12 | 62% | 30.05 | 1.57 | 5% | 0.27× (3.26×) |
| q_proj | 128 | 1024→2048 | 7.35 | 5.81 | 79% | 30.21 | 2.38 | 8% | 0.24× (2.44×) |
| q_proj | 512 | 1024→2048 | 11.07 | 15.29 | 138% | 30.27 | 5.62 | 19% | 0.37× (2.72×) |
| kv_proj | 1 | 1024→2048 | 7.68 | 4.90 | 64% | 7.54 | 1.31 | 17% | 1.02× (3.75×) |
| kv_proj | 32 | 1024→2048 | 7.89 | 5.12 | 65% | 30.38 | 1.57 | 5% | 0.26× (3.26×) |
| kv_proj | 128 | 1024→2048 | 7.00 | 5.81 | 83% | 30.18 | 2.38 | 8% | 0.23× (2.44×) |
| kv_proj | 512 | 1024→2048 | 10.79 | 15.29 | 142% | 30.29 | 5.62 | 19% | 0.36× (2.72×) |
| o_proj | 1 | 2048→1024 | 6.66 | 4.90 | 74% | 7.53 | 1.31 | 17% | 0.88× (3.75×) |
| o_proj | 32 | 2048→1024 | 6.51 | 5.12 | 79% | 34.11 | 1.61 | 5% | 0.19× (3.18×) |
| o_proj | 128 | 2048→1024 | 7.79 | 5.81 | 75% | 34.36 | 2.54 | 7% | 0.23× (2.29×) |
| o_proj | 512 | 2048→1024 | 11.37 | 15.29 | 135% | 34.21 | 6.92 | 20% | 0.33× (2.21×) |
| gate_up_proj | 1 | 1024→6144 | 16.35 | 14.70 | 90% | 9.39 | 3.92 | 42% | 1.74× (3.75×) |
| gate_up_proj | 32 | 1024→6144 | 15.55 | 15.22 | 98% | 30.42 | 4.48 | 15% | 0.51× (3.40×) |
| gate_up_proj | 128 | 1024→6144 | 17.25 | 16.83 | 98% | 29.85 | 6.21 | 21% | 0.58× (2.71×) |
| gate_up_proj | 512 | 1024→6144 | 45.13 | 45.88 | 102% | 50.90 | 13.12 | 26% | 0.89× (3.50×) |
| down_proj | 1 | 3072→1024 | 9.53 | 7.35 | 77% | 10.09 | 1.96 | 19% | 0.95× (3.75×) |
| down_proj | 32 | 3072→1024 | 9.18 | 7.65 | 83% | 33.80 | 2.38 | 7% | 0.27× (3.22×) |
| down_proj | 128 | 3072→1024 | 10.81 | 8.57 | 79% | 34.04 | 3.66 | 11% | 0.32× (2.34×) |
| down_proj | 512 | 3072→1024 | 17.13 | 22.94 | 134% | 38.05 | 10.38 | 27% | 0.45× (2.21×) |

### Qwen3-1.7B

| proj | T | shape | fp16_us | fp16_roof | fp16_eff | cuda_us | cuda_roof | cuda_eff | speedup (actual / roof) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| q_proj | 1 | 2048→2048 | 11.21 | 9.80 | 87% | 7.44 | 2.61 | 35% | 1.51× (3.75×) |
| q_proj | 32 | 2048→2048 | 11.59 | 10.10 | 87% | 34.14 | 2.99 | 9% | 0.34× (3.38×) |
| q_proj | 128 | 2048→2048 | 13.01 | 11.01 | 85% | 34.22 | 4.15 | 12% | 0.38× (2.65×) |
| q_proj | 512 | 2048→2048 | 20.20 | 30.59 | 151% | 34.55 | 10.75 | 31% | 0.58× (2.85×) |
| kv_proj | 1 | 2048→2048 | 11.44 | 9.80 | 86% | 7.41 | 2.61 | 35% | 1.54× (3.75×) |
| kv_proj | 32 | 2048→2048 | 11.52 | 10.10 | 88% | 34.05 | 2.99 | 9% | 0.34× (3.38×) |
| kv_proj | 128 | 2048→2048 | 13.11 | 11.01 | 84% | 34.13 | 4.15 | 12% | 0.38× (2.65×) |
| kv_proj | 512 | 2048→2048 | 20.34 | 30.59 | 150% | 34.54 | 10.75 | 31% | 0.59× (2.85×) |
| o_proj | 1 | 2048→2048 | 11.23 | 9.80 | 87% | 7.36 | 2.61 | 35% | 1.53× (3.75×) |
| o_proj | 32 | 2048→2048 | 11.51 | 10.10 | 88% | 34.05 | 2.99 | 9% | 0.34× (3.38×) |
| o_proj | 128 | 2048→2048 | 13.00 | 11.01 | 85% | 34.33 | 4.15 | 12% | 0.38× (2.65×) |
| o_proj | 512 | 2048→2048 | 20.28 | 30.59 | 151% | 34.56 | 10.75 | 31% | 0.59× (2.85×) |
| gate_up_proj | 1 | 2048→12288 | 58.93 | 58.78 | 100% | 25.03 | 15.64 | 62% | 2.35× (3.76×) |
| gate_up_proj | 32 | 2048→12288 | 57.78 | 59.81 | 104% | 34.20 | 16.76 | 49% | 1.69× (3.57×) |
| gate_up_proj | 128 | 2048→12288 | 65.34 | 63.03 | 96% | 42.32 | 20.21 | 48% | 1.54× (3.12×) |
| gate_up_proj | 512 | 2048→12288 | 146.94 | 183.52 | 125% | 121.78 | 48.99 | 40% | 1.21× (3.75×) |
| down_proj | 1 | 6144→2048 | 31.39 | 29.39 | 94% | 20.55 | 7.82 | 38% | 1.53× (3.76×) |
| down_proj | 32 | 6144→2048 | 30.90 | 29.98 | 97% | 38.78 | 8.66 | 22% | 0.80× (3.46×) |
| down_proj | 128 | 6144→2048 | 35.08 | 31.82 | 91% | 48.38 | 11.23 | 23% | 0.73× (2.83×) |
| down_proj | 512 | 6144→2048 | 75.86 | 91.76 | 121% | 86.99 | 32.24 | 37% | 0.87× (2.85×) |

### Qwen3-4B

| proj | T | shape | fp16_us | fp16_roof | fp16_eff | cuda_us | cuda_roof | cuda_eff | speedup (actual / roof) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| q_proj | 1 | 2560→4096 | 26.01 | 24.49 | 94% | 13.27 | 6.52 | 49% | 1.96× (3.76×) |
| q_proj | 32 | 2560→4096 | 26.82 | 24.97 | 93% | 34.01 | 7.10 | 21% | 0.79× (3.52×) |
| q_proj | 128 | 2560→4096 | 27.73 | 26.47 | 95% | 35.13 | 8.90 | 25% | 0.79× (2.97×) |
| q_proj | 512 | 2560→4096 | 72.03 | 76.47 | 106% | 58.72 | 23.00 | 39% | 1.23× (3.33×) |
| kv_proj | 1 | 2560→2048 | 14.20 | 12.25 | 86% | 9.99 | 3.26 | 33% | 1.42× (3.76×) |
| kv_proj | 32 | 2560→2048 | 15.19 | 12.58 | 83% | 34.15 | 3.70 | 11% | 0.44× (3.40×) |
| kv_proj | 128 | 2560→2048 | 15.46 | 13.62 | 88% | 34.19 | 5.03 | 15% | 0.45× (2.70×) |
| kv_proj | 512 | 2560→2048 | 26.29 | 38.23 | 145% | 42.45 | 13.43 | 32% | 0.62× (2.85×) |
| o_proj | 1 | 4096→2560 | 25.77 | 24.49 | 95% | 15.91 | 6.52 | 41% | 1.62× (3.76×) |
| o_proj | 32 | 4096→2560 | 26.04 | 24.97 | 96% | 34.10 | 7.16 | 21% | 0.76× (3.49×) |
| o_proj | 128 | 4096→2560 | 29.47 | 26.47 | 90% | 41.89 | 9.14 | 22% | 0.70× (2.90×) |
| o_proj | 512 | 4096→2560 | 80.44 | 76.47 | 95% | 82.45 | 25.32 | 31% | 0.98× (3.02×) |
| gate_up_proj | 1 | 2560→19456 | 109.99 | 116.32 | 106% | 48.65 | 30.93 | 64% | 2.26× (3.76×) |
| gate_up_proj | 32 | 2560→19456 | 118.13 | 117.91 | 100% | 46.89 | 32.63 | 70% | 2.52× (3.61×) |
| gate_up_proj | 128 | 2560→19456 | 129.15 | 122.84 | 95% | 91.08 | 37.87 | 42% | 1.42× (3.24×) |
| gate_up_proj | 512 | 2560→19456 | 332.86 | 363.22 | 109% | 229.94 | 94.70 | 41% | 1.45× (3.84×) |
| down_proj | 1 | 9728→2560 | 59.10 | 58.16 | 98% | 38.15 | 15.47 | 41% | 1.55× (3.76×) |
| down_proj | 32 | 9728→2560 | 60.16 | 59.05 | 98% | 57.79 | 16.75 | 29% | 1.04× (3.53×) |
| down_proj | 128 | 9728→2560 | 66.88 | 61.80 | 92% | 78.77 | 20.66 | 26% | 0.85× (2.99×) |
| down_proj | 512 | 9728→2560 | 161.93 | 181.61 | 112% | 193.41 | 60.13 | 31% | 0.84× (3.02×) |

### Qwen3-8B

| proj | T | shape | fp16_us | fp16_roof | fp16_eff | cuda_us | cuda_roof | cuda_eff | speedup (actual / roof) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| q_proj | 1 | 4096→4096 | 40.77 | 39.18 | 96% | 18.62 | 10.42 | 56% | 2.19× (3.76×) |
| q_proj | 32 | 4096→4096 | 40.40 | 39.77 | 98% | 34.23 | 11.18 | 33% | 1.18× (3.56×) |
| q_proj | 128 | 4096→4096 | 42.35 | 41.61 | 98% | 46.83 | 13.50 | 29% | 0.90× (3.08×) |
| q_proj | 512 | 4096→4096 | 115.66 | 122.35 | 106% | 86.38 | 36.79 | 43% | 1.34× (3.33×) |
| kv_proj | 1 | 4096→2048 | 21.28 | 19.60 | 92% | 13.68 | 5.22 | 38% | 1.56× (3.76×) |
| kv_proj | 32 | 4096→2048 | 21.22 | 20.04 | 94% | 35.72 | 5.82 | 16% | 0.59× (3.44×) |
| kv_proj | 128 | 4096→2048 | 22.45 | 21.42 | 95% | 34.20 | 7.69 | 22% | 0.66× (2.79×) |
| kv_proj | 512 | 4096→2048 | 47.64 | 61.17 | 128% | 60.65 | 21.49 | 35% | 0.79× (2.85×) |
| o_proj | 1 | 4096→4096 | 40.70 | 39.18 | 96% | 18.58 | 10.42 | 56% | 2.19× (3.76×) |
| o_proj | 32 | 4096→4096 | 40.41 | 39.77 | 98% | 34.33 | 11.18 | 33% | 1.18× (3.56×) |
| o_proj | 128 | 4096→4096 | 42.32 | 41.61 | 98% | 46.85 | 13.50 | 29% | 0.90× (3.08×) |
| o_proj | 512 | 4096→4096 | 115.58 | 122.35 | 106% | 86.07 | 36.79 | 43% | 1.34× (3.33×) |
| gate_up_proj | 1 | 4096→24576 | 216.30 | 235.04 | 109% | 94.58 | 62.48 | 66% | 2.29× (3.76×) |
| gate_up_proj | 32 | 4096→24576 | 238.78 | 237.12 | 99% | 73.37 | 64.72 | 88% | 3.25× (3.66×) |
| gate_up_proj | 128 | 4096→24576 | 274.33 | 243.54 | 89% | 147.98 | 71.63 | 48% | 1.85× (3.40×) |
| gate_up_proj | 512 | 4096→24576 | 700.01 | 734.08 | 105% | 460.85 | 189.77 | 41% | 1.52× (3.87×) |
| down_proj | 1 | 12288→4096 | 112.23 | 117.53 | 105% | 53.02 | 31.25 | 59% | 2.12× (3.76×) |
| down_proj | 32 | 12288→4096 | 123.89 | 118.71 | 96% | 71.68 | 32.92 | 46% | 1.73× (3.61×) |
| down_proj | 128 | 12288→4096 | 134.24 | 122.38 | 91% | 107.04 | 38.05 | 36% | 1.25× (3.22×) |
| down_proj | 512 | 12288→4096 | 322.13 | 367.04 | 114% | 301.03 | 110.38 | 37% | 1.07× (3.33×) |

## §5 CUDA implementation-gap TOP-15 (worst cuda_efficiency)

| rank | model | proj | T | shape | cuda_us | cuda_roof | cuda_eff | gemm_bound |
|---:|:---|:---|---:|:---:|---:|---:|---:|:---:|
| 1 | Qwen3-0.6B | o_proj | 32 | 2048→1024 | 34.11 | 1.61 | 5% | mem |
| 2 | Qwen3-0.6B | kv_proj | 32 | 1024→2048 | 30.38 | 1.57 | 5% | mem |
| 3 | Qwen3-0.6B | q_proj | 32 | 1024→2048 | 30.05 | 1.57 | 5% | mem |
| 4 | Qwen3-0.6B | down_proj | 32 | 3072→1024 | 33.80 | 2.38 | 7% | mem |
| 5 | Qwen3-0.6B | o_proj | 128 | 2048→1024 | 34.36 | 2.54 | 7% | mem |
| 6 | Qwen3-0.6B | q_proj | 128 | 1024→2048 | 30.21 | 2.38 | 8% | mem |
| 7 | Qwen3-0.6B | kv_proj | 128 | 1024→2048 | 30.18 | 2.38 | 8% | mem |
| 8 | Qwen3-1.7B | q_proj | 32 | 2048→2048 | 34.14 | 2.99 | 9% | mem |
| 9 | Qwen3-1.7B | o_proj | 32 | 2048→2048 | 34.05 | 2.99 | 9% | mem |
| 10 | Qwen3-1.7B | kv_proj | 32 | 2048→2048 | 34.05 | 2.99 | 9% | mem |
| 11 | Qwen3-0.6B | down_proj | 128 | 3072→1024 | 34.04 | 3.66 | 11% | mem |
| 12 | Qwen3-4B | kv_proj | 32 | 2560→2048 | 34.15 | 3.70 | 11% | mem |
| 13 | Qwen3-1.7B | o_proj | 128 | 2048→2048 | 34.33 | 4.15 | 12% | mem |
| 14 | Qwen3-1.7B | q_proj | 128 | 2048→2048 | 34.22 | 4.15 | 12% | mem |
| 15 | Qwen3-1.7B | kv_proj | 128 | 2048→2048 | 34.13 | 4.15 | 12% | mem |

## §6 Physics gap — `cuda_roof / fp16_roof`

Rows where `cuda_roof / fp16_roof > 1.0` are shapes where **W4A4 cannot beat FP16 even at the physical limit** — these should route to FP16 via policy, no kernel work can rescue them.

- Rows with cuda_roof faster than fp16_roof (W4A4 can win at ceiling): **80 / 80**
- Rows where fp16_roof is faster (W4A4 loses at ceiling): **0 / 80**

## §7 Conclusions

- Out of **80** shapes, **1** reach `cuda_eff >= 0.80` — already near the physical limit.
- **72** shapes sit at `cuda_eff < 0.50` — real implementation slack remains (§5 top offenders).
- **0** shapes have `cuda_roof >= fp16_roof` — W4A4 loses at the ceiling; these must route to FP16 via policy, no kernel work can rescue them.
- Measured today: **35 / 80** shapes actually beat FP16; **45** lose.  Of the losing shapes, **45** are *implementation-gap* losses (fixable) and **0** are *physics* losses (unfixable).

### Aggregate stats (measured)

| metric | value |
|---|---:|
| shapes | 80 |
| median speedup vs FP16 | 0.895× |
| mean speedup vs FP16 | 1.048× |
| wins (≥ 1.00×) | 35 / 80 |
| clear wins (≥ 1.10×) | 32 / 80 |
| median INT4 efficiency | 30.9% |
| max INT4 efficiency | 88.2% |
