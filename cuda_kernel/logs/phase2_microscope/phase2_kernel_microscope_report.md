# Phase 2 Kernel Microscope Report

> ⚠️  **REDIAGNOSED 2026-04-28** — the "Global SASS finding" section
> below used issue-slot-count TC% to conclude "Tensor Core not
> emitted".  That conclusion was *wrong*: re-analysis with MAC-weighted
> share shows `mac_tc_share ≥ 99 %` on every `*_mma_int4_kernel`.  IMMA
> is active and carries virtually all MAC work; the 13–39 % cuda_eff
> comes from **MMA pipeline starvation** (epilogue / IMAD / async-copy
> serialisation), not from CUDA-core-only execution.  The `tc_underutil`
> label is retained as a stable taxonomy key but its meaning is changed
> accordingly.  See
> [`phase2_tc_rediagnosis.md`](phase2_tc_rediagnosis.md) for the
> evidence chain and the sub-bottleneck decomposition driving Step 2
> expected-gain recalibration.

Joined sources:
- Phase 1 timeline attribution (`phase1_timeline/phase1_attribution.json`)
- Phase 1 launch tax (`phase1_timeline/<tag>/launch_tax.json`)
- SASS static profile (`phase2_microscope/_sass/sass_profile.json`)
- Microbench bisection (`phase2_microscope/<tag>/bisection.json`)

## Global SASS finding

Static analysis of **all 42 compiled kernel instantiations** in `hkust_v9_cuda.so` gives:
- `tc_underutil`: **42** / 42 kernels (now meaning *MMA pipeline
  starvation*; see banner above)
- `high_reg_pressure`: **18** / 42 kernels

→ **Rediagnosed reading**: every `*_mma_int4_kernel` has IMMA
issue-slot share of 0.8–1.7 % and MAC-weighted TC share ≥ 99 %.  One
`mma.m16n8k64.s4` does 8192 MACs, so 32–64 IMMAs are already carrying
the full compute budget; the tensor pipeline is busy only ~24 % of the
time because the warp scheduler spends the rest on (i) in-kernel HFMA2
dequant epilogue, (ii) shared-memory swizzle IMAD address math, and
(iii) only 2-stage `cp.async` double-buffering.  This is a *pipeline
organisation* problem, not a *TC-not-emitted* problem.  The roofline
denominator (660 TOPS INT4 Tensor Core peak) is therefore reachable in
principle; the 13–39 % observed `cuda_eff` reflects the 76 % tensor
pipeline idle share.

## Per-shape bottleneck attribution

| shape | T | plain_us | body_us | launch_tax_us | launch% | Δ_l2% | Δ_xzero% | Δ_scale1% | primary_bottleneck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| decode_T1_kv_2560_2048 | 1 | 0.00 | 0.00 | 0.00 | 0% | +0.8 | +1.3 | +1.1 | `tc_underutil` |
| decode_T1_q_2048_2048 | 1 | 46.77 | 15.28 | 33.60 | 72% | +1.0 | +1.1 | +0.9 | `launch_sparse` |
| large_T1024_gu_4096_24576 | 1024 | 1127.07 | 41274.02 | 2.93 | 0% | -0.1 | +0.1 | -0.1 | `tc_underutil` |
| mid_T128_kv_2560_2048 | 128 | 80.56 | 66.99 | 16.41 | 20% | +0.7 | +1.5 | +2.2 | `tc_underutil` |
| prefill_T1024_down_3072_1024 | 1024 | 0.00 | 0.00 | 0.00 | 0% | -0.1 | +0.8 | -0.2 | `tc_underutil` |
| prefill_T512_gu_2048_12288 | 512 | 0.00 | 0.00 | 0.00 | 0% | -0.5 | -0.1 | -0.5 | `tc_underutil` |
| worst_T8_kv_1024_2048 | 8 | 55.01 | 18.56 | 38.06 | 69% | +1.0 | +1.1 | +0.8 | `launch_sparse` |
| worst_T8_q_4096_4096 | 8 | 0.00 | 0.00 | 0.00 | 0% | +1.3 | +1.1 | +2.0 | `tc_underutil` |

## Per-shape evidence (one block per shape)

### `decode_T1_kv_2560_2048`

- shape: T=1, d_in=2560, d_out=2048, Qwen3-4B kv_proj
- bisection (base=43.10us): Δ_l2=+0.8%, Δ_xzero=+1.3%, Δ_scale1=+1.1%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 1.3%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

### `decode_T1_q_2048_2048`

- shape: T=1, d_in=2048, d_out=2048, Qwen3-1.7B q_proj
- Phase 1: forward=111.13us (path=fused, quant=36.99us, gemm=28.14us, python_glue=33.46us)
- launch tax: 33.60 us (71.8% of plain); graph_replay=13.17us, plain=46.77us
- bisection (base=42.02us): Δ_l2=+1.0%, Δ_xzero=+1.1%, Δ_scale1=+0.9%
- **Primary bottleneck: `launch_sparse`** — launch_tax ≈ 33.6us = 72% of plain

### `large_T1024_gu_4096_24576`

- shape: T=1024, d_in=4096, d_out=24576, Qwen3-8B gate_up_proj
- Phase 1: forward=382.68us (path=fused, quant=81.35us, gemm=45.29us, python_glue=230.09us)
- launch tax: 2.93 us (0.3% of plain); graph_replay=1124.14us, plain=1127.07us
- bisection (base=1127.25us): Δ_l2=-0.1%, Δ_xzero=+0.1%, Δ_scale1=-0.1%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 0.1%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

### `mid_T128_kv_2560_2048`

- shape: T=128, d_in=2560, d_out=2048, Qwen3-4B kv_proj
- Phase 1: forward=139.47us (path=split, quant=25.59us, gemm=48.30us, python_glue=52.47us)
- launch tax: 16.41 us (20.4% of plain); graph_replay=64.15us, plain=80.56us
- bisection (base=78.86us): Δ_l2=+0.7%, Δ_xzero=+1.5%, Δ_scale1=+2.2%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 2.2%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

### `prefill_T1024_down_3072_1024`

- shape: T=1024, d_in=3072, d_out=1024, Qwen3-0.6B down_proj
- bisection (base=79.17us): Δ_l2=-0.1%, Δ_xzero=+0.8%, Δ_scale1=-0.2%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 0.8%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

### `prefill_T512_gu_2048_12288`

- shape: T=512, d_in=2048, d_out=12288, Qwen3-1.7B gate_up_proj
- bisection (base=161.16us): Δ_l2=-0.5%, Δ_xzero=-0.1%, Δ_scale1=-0.5%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 0.5%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

### `worst_T8_kv_1024_2048`

- shape: T=8, d_in=1024, d_out=2048, Qwen3-0.6B kv_proj
- Phase 1: forward=89.56us (path=fused, quant=24.39us, gemm=19.01us, python_glue=37.28us)
- launch tax: 38.06 us (69.2% of plain); graph_replay=16.95us, plain=55.01us
- bisection (base=52.29us): Δ_l2=+1.0%, Δ_xzero=+1.1%, Δ_scale1=+0.8%
- **Primary bottleneck: `launch_sparse`** — launch_tax ≈ 38.1us = 69% of plain

### `worst_T8_q_4096_4096`

- shape: T=8, d_in=4096, d_out=4096, Qwen3-8B q_proj
- bisection (base=75.80us): Δ_l2=+1.3%, Δ_xzero=+1.1%, Δ_scale1=+2.0%
- **Primary bottleneck: `tc_underutil`** — all bisection deltas < 3% (max 2.0%); SASS shows TC% < 2% on all MMA kernels -> CUDA-core FMA bound

## Bottleneck category coverage

- `launch_sparse`: 2 shape(s)
- `tc_underutil`: 6 shape(s)

