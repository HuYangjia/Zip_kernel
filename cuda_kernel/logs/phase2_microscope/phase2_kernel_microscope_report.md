# Phase 2 Kernel Microscope Report

Joined sources:
- Phase 1 timeline attribution (`phase1_timeline/phase1_attribution.json`)
- Phase 1 launch tax (`phase1_timeline/<tag>/launch_tax.json`)
- SASS static profile (`phase2_microscope/_sass/sass_profile.json`)
- Microbench bisection (`phase2_microscope/<tag>/bisection.json`)

## Global SASS finding

Static analysis of **all 42 compiled kernel instantiations** in `hkust_v9_cuda.so` gives:
- `tc_underutil`: **42** / 42 kernels
- `high_reg_pressure`: **18** / 42 kernels

→ **The decisive finding is `tc_underutil` firing on 42/42 kernels**: every `*_mma_int4_kernel` has HMMA/IMMA fraction **< 2 %** of SASS, with CUDA-core FMA (FFMA+IMAD) at 24-36 %.  This means the roofline denominator (660 TOPS INT4 Tensor Core peak) is effectively **unreachable with the current dequant-into-FMA chain**; real throughput is CUDA-core-bound at roughly 1/8 of TC peak, which matches the 13-39 % `cuda_eff` observed in the Roofline report.

## Per-shape bottleneck attribution

| shape | T | plain_us | body_us | launch_tax_us | launch% | Δ_l2% | Δ_xzero% | Δ_scale1% | primary_bottleneck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode_T1_kv_2560_2048 | 1 | 0.00 | 0.00 | 0.00 | 0% | +0.9 | +1.9 | +0.9 | `tc_underutil` |
| decode_T1_q_2048_2048 | 1 | 46.77 | 15.28 | 33.60 | 72% | +2.2 | +2.2 | +2.1 | `launch_sparse` |
| large_T1024_gu_4096_24576 | 1024 | 1127.07 | 41274.02 | 2.93 | 0% | -0.2 | +0.1 | -0.1 | `tc_underutil` |
| mid_T128_kv_2560_2048 | 128 | 80.56 | 66.99 | 16.41 | 20% | +0.8 | -27.9 | +0.7 | `x_zero_anomaly` |
| prefill_T1024_down_3072_1024 | 1024 | 0.00 | 0.00 | 0.00 | 0% | +1.3 | +0.8 | +3.0 | `epilogue_fma_bound` |
| prefill_T512_gu_2048_12288 | 512 | 0.00 | 0.00 | 0.00 | 0% | -0.4 | +0.0 | -0.4 | `tc_underutil` |
| worst_T8_kv_1024_2048 | 8 | 55.01 | 18.56 | 38.06 | 69% | +2.2 | +2.5 | +3.3 | `launch_sparse` |
| worst_T8_q_4096_4096 | 8 | 0.00 | 0.00 | 0.00 | 0% | +2.0 | +3.3 | +3.0 | `epilogue_fma_bound` |

## Per-shape evidence (one block per shape)

### `decode_T1_kv_2560_2048`

- shape: T=1, d_in=2560, d_out=2048, Qwen3-4B kv_proj
- bisection (base=68.26us): Δ_l2=+0.9%, Δ_xzero=+1.9%, Δ_scale1=+0.9%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 1.9%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

### `decode_T1_q_2048_2048`

- shape: T=1, d_in=2048, d_out=2048, Qwen3-1.7B q_proj
- Phase 1: forward=111.13us (path=fused, quant=36.99us, gemm=28.14us, python_glue=33.46us)
- launch tax: 33.60 us (71.8% of plain); graph_replay=13.17us, plain=46.77us
- bisection (base=46.54us): Δ_l2=+2.2%, Δ_xzero=+2.2%, Δ_scale1=+2.1%
- **Primary bottleneck: `launch_sparse`** — launch_tax ≈ 33.6us = 72% of plain

### `large_T1024_gu_4096_24576`

- shape: T=1024, d_in=4096, d_out=24576, Qwen3-8B gate_up_proj
- Phase 1: forward=382.68us (path=fused, quant=81.35us, gemm=45.29us, python_glue=230.09us)
- launch tax: 2.93 us (0.3% of plain); graph_replay=1124.14us, plain=1127.07us
- bisection (base=1126.72us): Δ_l2=-0.2%, Δ_xzero=+0.1%, Δ_scale1=-0.1%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 0.2%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

### `mid_T128_kv_2560_2048`

- shape: T=128, d_in=2560, d_out=2048, Qwen3-4B kv_proj
- Phase 1: forward=139.47us (path=split, quant=25.59us, gemm=48.30us, python_glue=52.47us)
- launch tax: 16.41 us (20.4% of plain); graph_replay=64.15us, plain=80.56us
- bisection (base=86.49us): Δ_l2=+0.8%, Δ_xzero=-27.9%, Δ_scale1=+0.7%
- **Primary bottleneck: `x_zero_anomaly`** — X=0 is 27.9% SLOWER than random input — data-dependent kernel path; investigate separately

### `prefill_T1024_down_3072_1024`

- shape: T=1024, d_in=3072, d_out=1024, Qwen3-0.6B down_proj
- bisection (base=82.44us): Δ_l2=+1.3%, Δ_xzero=+0.8%, Δ_scale1=+3.0%
- **Primary bottleneck: `epilogue_fma_bound`** — scale=1 speeds up by 3.0% -> epilogue FMA is a real tail consumer

### `prefill_T512_gu_2048_12288`

- shape: T=512, d_in=2048, d_out=12288, Qwen3-1.7B gate_up_proj
- bisection (base=160.72us): Δ_l2=-0.4%, Δ_xzero=+0.0%, Δ_scale1=-0.4%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 0.4%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

### `worst_T8_kv_1024_2048`

- shape: T=8, d_in=1024, d_out=2048, Qwen3-0.6B kv_proj
- Phase 1: forward=89.56us (path=fused, quant=24.39us, gemm=19.01us, python_glue=37.28us)
- launch tax: 38.06 us (69.2% of plain); graph_replay=16.95us, plain=55.01us
- bisection (base=82.94us): Δ_l2=+2.2%, Δ_xzero=+2.5%, Δ_scale1=+3.3%
- **Primary bottleneck: `launch_sparse`** — launch_tax ≈ 38.1us = 69% of plain

### `worst_T8_q_4096_4096`

- shape: T=8, d_in=4096, d_out=4096, Qwen3-8B q_proj
- bisection (base=82.74us): Δ_l2=+2.0%, Δ_xzero=+3.3%, Δ_scale1=+3.0%
- **Primary bottleneck: `epilogue_fma_bound`** — scale=1 speeds up by 3.0% -> epilogue FMA is a real tail consumer

## Bottleneck category coverage

- `epilogue_fma_bound`: 2 shape(s)
- `launch_sparse`: 2 shape(s)
- `tc_underutil`: 3 shape(s)
- `x_zero_anomaly`: 1 shape(s)

