# r63 Larger models (Qwen3-14B + Qwen2.5-32B + LLaMA3-70B) bench

Source: `/root/Zip_kernel/cuda_kernel/logs/r63_large_models/qwen3_20260430_124225/bench.json`

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
| 1 | 15 | 95% | 107% | 111% |
| 32 | 15 | 92% | 96% | 104% |
| 128 | 15 | 85% | 92% | 101% |
| 512 | 15 | 109% | 111% | 123% |

## §3 CUDA efficiency distribution

`cuda_eff = cuda_roof_us / cuda_us` — how close our W4A4 kernel gets to its own roofline.

### §3.1 By T

| label | n | min | median | max |
|---|---:|---:|---:|---:|
| 1 | 15 | 24% | 60% | 65% |
| 32 | 15 | 21% | 42% | 58% |
| 128 | 15 | 23% | 30% | 43% |
| 512 | 15 | 20% | 31% | 38% |

### §3.2 By proj

| label | n | min | median | max |
|---|---:|---:|---:|---:|
| q_proj | 12 | 30% | 42% | 63% |
| kv_proj | 12 | 21% | 30% | 40% |
| o_proj | 12 | 30% | 42% | 62% |
| gate_up_proj | 12 | 20% | 33% | 60% |
| down_proj | 12 | 23% | 31% | 65% |

## §4 Per-shape detail

Per-row: measured time, roofline time, efficiency, and actual-vs-roof speedup against FP16.

### LLaMA3-70B

| proj | T | shape | fp16_us | fp16_roof | fp16_eff | cuda_us | cuda_roof | cuda_eff | speedup (actual / roof) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| q_proj | 1 | 8192→8192 | 146.69 | 156.69 | 107% | 65.78 | 41.65 | 63% | 2.23× (3.76×) |
| q_proj | 32 | 8192→8192 | 172.14 | 157.87 | 92% | 74.53 | 43.16 | 58% | 2.31× (3.66×) |
| q_proj | 128 | 8192→8192 | 191.06 | 161.55 | 85% | 112.56 | 47.81 | 42% | 1.70× (3.38×) |
| q_proj | 512 | 8192→8192 | 398.73 | 489.39 | 123% | 377.00 | 134.78 | 36% | 1.06× (3.63×) |
| kv_proj | 1 | 8192→2048 | 39.17 | 39.19 | 100% | 26.16 | 10.43 | 40% | 1.50× (3.76×) |
| kv_proj | 32 | 8192→2048 | 40.58 | 39.93 | 98% | 48.80 | 11.49 | 24% | 0.83× (3.47×) |
| kv_proj | 128 | 8192→2048 | 46.80 | 42.22 | 90% | 60.71 | 14.76 | 24% | 0.77× (2.86×) |
| kv_proj | 512 | 8192→2048 | 100.82 | 122.35 | 121% | 112.86 | 42.99 | 38% | 0.89× (2.85×) |
| o_proj | 1 | 8192→8192 | 146.65 | 156.69 | 107% | 66.86 | 41.65 | 62% | 2.19× (3.76×) |
| o_proj | 32 | 8192→8192 | 172.13 | 157.87 | 92% | 74.41 | 43.16 | 58% | 2.31× (3.66×) |
| o_proj | 128 | 8192→8192 | 191.03 | 161.55 | 85% | 112.48 | 47.81 | 43% | 1.70× (3.38×) |
| o_proj | 512 | 8192→8192 | 400.36 | 489.39 | 122% | 377.12 | 134.78 | 36% | 1.06× (3.63×) |
| gate_up_proj | 1 | 8192→57344 | 985.90 | 1096.70 | 111% | 485.25 | 291.42 | 60% | 2.03× (3.76×) |
| gate_up_proj | 32 | 8192→57344 | 1099.90 | 1101.45 | 100% | 593.42 | 296.49 | 50% | 1.85× (3.71×) |
| gate_up_proj | 128 | 8192→57344 | 1142.68 | 1116.13 | 98% | 1147.19 | 312.15 | 27% | 1.00× (3.58×) |
| gate_up_proj | 512 | 8192→57344 | 3030.39 | 3425.70 | 113% | 4380.58 | 869.08 | 20% | 0.69× (3.94×) |
| down_proj | 1 | 28672→8192 | 497.52 | 548.36 | 110% | 428.21 | 145.72 | 34% | 1.16× (3.76×) |
| down_proj | 32 | 28672→8192 | 528.76 | 551.03 | 104% | 441.09 | 149.53 | 34% | 1.20× (3.69×) |
| down_proj | 128 | 28672→8192 | 603.52 | 559.29 | 93% | 519.31 | 161.20 | 31% | 1.16× (3.47×) |
| down_proj | 512 | 28672→8192 | 1575.34 | 1712.85 | 109% | 1562.63 | 471.71 | 30% | 1.01× (3.63×) |

### Qwen2.5-32B

| proj | T | shape | fp16_us | fp16_roof | fp16_eff | cuda_us | cuda_roof | cuda_eff | speedup (actual / roof) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| q_proj | 1 | 5120→5120 | 60.96 | 61.22 | 100% | 26.61 | 16.28 | 61% | 2.29× (3.76×) |
| q_proj | 32 | 5120→5120 | 64.26 | 61.96 | 96% | 41.40 | 17.22 | 42% | 1.55× (3.60×) |
| q_proj | 128 | 5120→5120 | 69.38 | 64.25 | 93% | 67.85 | 20.13 | 30% | 1.02× (3.19×) |
| q_proj | 512 | 5120→5120 | 174.67 | 191.17 | 109% | 176.40 | 55.55 | 31% | 0.99× (3.44×) |
| kv_proj | 1 | 5120→2048 | 25.66 | 24.49 | 95% | 17.41 | 6.52 | 37% | 1.47× (3.76×) |
| kv_proj | 32 | 5120→2048 | 27.22 | 25.01 | 92% | 35.11 | 7.24 | 21% | 0.78× (3.45×) |
| kv_proj | 128 | 5120→2048 | 30.30 | 26.62 | 88% | 41.28 | 9.46 | 23% | 0.73× (2.81×) |
| kv_proj | 512 | 5120→2048 | 62.86 | 76.47 | 122% | 75.70 | 26.87 | 35% | 0.83× (2.85×) |
| o_proj | 1 | 5120→5120 | 61.00 | 61.22 | 100% | 26.56 | 16.28 | 61% | 2.30× (3.76×) |
| o_proj | 32 | 5120→5120 | 64.54 | 61.96 | 96% | 41.47 | 17.22 | 42% | 1.56× (3.60×) |
| o_proj | 128 | 5120→5120 | 69.52 | 64.25 | 92% | 67.78 | 20.13 | 30% | 1.03× (3.19×) |
| o_proj | 512 | 5120→5120 | 174.21 | 191.17 | 110% | 176.52 | 55.55 | 31% | 0.99× (3.44×) |
| gate_up_proj | 1 | 5120→55296 | 596.74 | 661.01 | 111% | 298.44 | 175.68 | 59% | 2.00× (3.76×) |
| gate_up_proj | 32 | 5120→55296 | 665.11 | 665.38 | 100% | 378.74 | 180.26 | 48% | 1.76× (3.69×) |
| gate_up_proj | 128 | 5120→55296 | 674.05 | 678.92 | 101% | 703.64 | 194.41 | 28% | 0.96× (3.49×) |
| gate_up_proj | 512 | 5120→55296 | 1854.10 | 2064.59 | 111% | 2595.49 | 524.05 | 20% | 0.71× (3.94×) |
| down_proj | 1 | 27648→5120 | 302.80 | 330.51 | 109% | 364.46 | 87.85 | 24% | 0.83× (3.76×) |
| down_proj | 32 | 27648→5120 | 326.34 | 332.88 | 102% | 300.95 | 91.32 | 30% | 1.08× (3.65×) |
| down_proj | 128 | 27648→5120 | 378.47 | 340.22 | 90% | 382.66 | 101.95 | 27% | 0.99× (3.34×) |
| down_proj | 512 | 27648→5120 | 918.62 | 1032.30 | 112% | 1310.84 | 299.97 | 23% | 0.70× (3.44×) |

### Qwen3-14B

| proj | T | shape | fp16_us | fp16_roof | fp16_eff | cuda_us | cuda_roof | cuda_eff | speedup (actual / roof) |
|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|
| q_proj | 1 | 5120→5120 | 61.04 | 61.22 | 100% | 26.56 | 16.28 | 61% | 2.30× (3.76×) |
| q_proj | 32 | 5120→5120 | 64.33 | 61.96 | 96% | 41.02 | 17.22 | 42% | 1.57× (3.60×) |
| q_proj | 128 | 5120→5120 | 69.12 | 64.25 | 93% | 67.27 | 20.13 | 30% | 1.03× (3.19×) |
| q_proj | 512 | 5120→5120 | 171.66 | 191.17 | 111% | 175.34 | 55.55 | 32% | 0.98× (3.44×) |
| kv_proj | 1 | 5120→2048 | 25.59 | 24.49 | 96% | 17.33 | 6.52 | 38% | 1.48× (3.76×) |
| kv_proj | 32 | 5120→2048 | 27.22 | 25.01 | 92% | 34.93 | 7.24 | 21% | 0.78× (3.45×) |
| kv_proj | 128 | 5120→2048 | 30.23 | 26.62 | 88% | 41.06 | 9.46 | 23% | 0.74× (2.81×) |
| kv_proj | 512 | 5120→2048 | 62.73 | 76.47 | 122% | 75.88 | 26.87 | 35% | 0.83× (2.85×) |
| o_proj | 1 | 5120→5120 | 60.68 | 61.22 | 101% | 26.42 | 16.28 | 62% | 2.30× (3.76×) |
| o_proj | 32 | 5120→5120 | 64.26 | 61.96 | 96% | 41.22 | 17.22 | 42% | 1.56× (3.60×) |
| o_proj | 128 | 5120→5120 | 69.12 | 64.25 | 93% | 67.53 | 20.13 | 30% | 1.02× (3.19×) |
| o_proj | 512 | 5120→5120 | 172.33 | 191.17 | 111% | 175.33 | 55.55 | 32% | 0.98× (3.44×) |
| gate_up_proj | 1 | 5120→34816 | 378.26 | 416.19 | 110% | 183.08 | 110.62 | 60% | 2.07× (3.76×) |
| gate_up_proj | 32 | 5120→34816 | 434.49 | 419.08 | 96% | 299.79 | 113.71 | 38% | 1.45× (3.69×) |
| gate_up_proj | 128 | 5120→34816 | 456.43 | 428.03 | 94% | 477.28 | 123.27 | 26% | 0.96× (3.47×) |
| gate_up_proj | 512 | 5120→34816 | 1174.76 | 1299.93 | 111% | 1517.26 | 332.83 | 22% | 0.77× (3.91×) |
| down_proj | 1 | 17408→5120 | 193.22 | 208.10 | 108% | 85.24 | 55.32 | 65% | 2.27× (3.76×) |
| down_proj | 32 | 17408→5120 | 226.15 | 209.73 | 93% | 123.64 | 57.64 | 47% | 1.83× (3.64×) |
| down_proj | 128 | 17408→5120 | 244.41 | 214.78 | 88% | 193.05 | 64.76 | 34% | 1.27× (3.32×) |
| down_proj | 512 | 17408→5120 | 586.00 | 649.96 | 111% | 734.52 | 188.87 | 26% | 0.80× (3.44×) |

## §5 CUDA implementation-gap TOP-15 (worst cuda_efficiency)

| rank | model | proj | T | shape | cuda_us | cuda_roof | cuda_eff | gemm_bound |
|---:|:---|:---|---:|:---:|---:|---:|---:|:---:|
| 1 | LLaMA3-70B | gate_up_proj | 512 | 8192→57344 | 4380.58 | 869.08 | 20% | compute |
| 2 | Qwen2.5-32B | gate_up_proj | 512 | 5120→55296 | 2595.49 | 524.05 | 20% | compute |
| 3 | Qwen2.5-32B | kv_proj | 32 | 5120→2048 | 35.11 | 7.24 | 21% | mem |
| 4 | Qwen3-14B | kv_proj | 32 | 5120→2048 | 34.93 | 7.24 | 21% | mem |
| 5 | Qwen3-14B | gate_up_proj | 512 | 5120→34816 | 1517.26 | 332.83 | 22% | compute |
| 6 | Qwen2.5-32B | down_proj | 512 | 27648→5120 | 1310.84 | 299.97 | 23% | compute |
| 7 | Qwen2.5-32B | kv_proj | 128 | 5120→2048 | 41.28 | 9.46 | 23% | mem |
| 8 | Qwen3-14B | kv_proj | 128 | 5120→2048 | 41.06 | 9.46 | 23% | mem |
| 9 | LLaMA3-70B | kv_proj | 32 | 8192→2048 | 48.80 | 11.49 | 24% | mem |
| 10 | Qwen2.5-32B | down_proj | 1 | 27648→5120 | 364.46 | 87.85 | 24% | mem |
| 11 | LLaMA3-70B | kv_proj | 128 | 8192→2048 | 60.71 | 14.76 | 24% | mem |
| 12 | Qwen3-14B | down_proj | 512 | 17408→5120 | 734.52 | 188.87 | 26% | compute |
| 13 | Qwen3-14B | gate_up_proj | 128 | 5120→34816 | 477.28 | 123.27 | 26% | mem |
| 14 | Qwen2.5-32B | down_proj | 128 | 27648→5120 | 382.66 | 101.95 | 27% | mem |
| 15 | LLaMA3-70B | gate_up_proj | 128 | 8192→57344 | 1147.19 | 312.15 | 27% | mem |

## §6 Physics gap — `cuda_roof / fp16_roof`

Rows where `cuda_roof / fp16_roof > 1.0` are shapes where **W4A4 cannot beat FP16 even at the physical limit** — these should route to FP16 via policy, no kernel work can rescue them.

- Rows with cuda_roof faster than fp16_roof (W4A4 can win at ceiling): **60 / 60**
- Rows where fp16_roof is faster (W4A4 loses at ceiling): **0 / 60**

## §7 Conclusions

- Out of **60** shapes, **0** reach `cuda_eff >= 0.80` — already near the physical limit.
- **48** shapes sit at `cuda_eff < 0.50` — real implementation slack remains (§5 top offenders).
- **0** shapes have `cuda_roof >= fp16_roof` — W4A4 loses at the ceiling; these must route to FP16 via policy, no kernel work can rescue them.
- Measured today: **37 / 60** shapes actually beat FP16; **23** lose.  Of the losing shapes, **23** are *implementation-gap* losses (fixable) and **0** are *physics* losses (unfixable).

### Aggregate stats (measured)

| metric | value |
|---|---:|
| shapes | 60 |
| median speedup vs FP16 | 1.073× |
| mean speedup vs FP16 | 1.336× |
| wins (≥ 1.00×) | 37 / 60 |
| clear wins (≥ 1.10×) | 29 / 60 |
| median INT4 efficiency | 34.7% |
| max INT4 efficiency | 64.9% |
