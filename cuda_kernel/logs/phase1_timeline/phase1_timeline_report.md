# Phase 1 — Timeline Attribution Report

_Generated 2026-04-28T09:29:40.083367Z_
_Run meta: git_sha=2db4d21 @ 2026-04-28 (autodl RTX 4090, nsys 2025.1.1.0)_

Every representative shape was run under `nsys profile -t cuda,nvtx,osrt` with the inner driver emitting one `phase1.iter_<i>` NVTX range per profiled forward, plus four fine-grained sub-ranges instrumented via `HKUST_V9_PROFILE=1`:  
* `ops.linear_forward` — whole forward, outermost range.  
* `dispatcher.select_impl` — backend resolution (per-kernel).  
* `cuda.activation_quant` — SINT4 activation quantisation body.  
* `cuda.fused_dense_sparse` — fused dense+sparse MMA/GEMV body.  
Each row below is the mean over all profiled iterations (n=10 unless otherwise noted).

## 1. Attribution Table (per forward, microseconds)

| shape | T | path | forward | disp | quant | fused | dense | sparse | kernel_body | python_glue | inter_kernel_gap |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| decode_T1_q_2048_2048 | 1 | fused | 111.13 | 12.54 | 36.99 | 28.14 | 0.00 | 0.00 | 12.93 | 33.46 | 64.75 |
| worst_T8_kv_1024_2048 | 8 | fused | 89.56 | 8.88 | 24.39 | 19.01 | 0.00 | 0.00 | 16.76 | 37.28 | 35.53 |
| mid_T128_kv_2560_2048 | 128 | split | 139.47 | 13.12 | 25.59 | 0.00 | 17.87 | 30.44 | 64.44 | 52.47 | 22.57 |
| large_T1024_gu_4096_24576 | 1024 | fused | 382.68 | 25.94 | 81.35 | 45.29 | 0.00 | 0.00 | 1056.44 | 230.09 | 0.00 |

## 2. Fraction of forward (percentages)

Each row sums to ≥100%: `quant + fused` is the on-device kernel work, `python_glue` is host-side time inside the forward, and the `forward - kernel_body` gap accounts for CUDA API + dispatcher overhead.

| shape | quant% | fused% | kernel_body% | python_glue% | inter_kernel_gap% |
|---|---:|---:|---:|---:|---:|
| decode_T1_q_2048_2048 | 33.3% | 25.3% | 11.6% | 30.1% | 58.3% |
| worst_T8_kv_1024_2048 | 27.2% | 21.2% | 18.7% | 41.6% | 39.7% |
| mid_T128_kv_2560_2048 | 18.3% | 0.0% | 46.2% | 37.6% | 16.2% |
| large_T1024_gu_4096_24576 | 21.3% | 11.8% | 276.1% | 60.1% | 0.0% |

## 3. Cross-check against CUDA-Graph launch-tax measurement

The Graph-replay driver amortises all kernel launches into a single `cudaGraphLaunch`.  Difference = aggregate kernel-launch API overhead.

| shape | plain_us | graph_us | launch_tax_us | tax %% | nvtx_kernel_body_us |
|---|---:|---:|---:|---:|---:|
| decode_T1_q_2048_2048 | 46.766 | 13.169 | 33.597 | 71.84% | 12.925 |
| worst_T8_kv_1024_2048 | 55.009 | 16.947 | 38.062 | 69.19% | 16.758 |
| mid_T128_kv_2560_2048 | 80.558 | 64.152 | 16.406 | 20.37% | 64.442 |
| large_T1024_gu_4096_24576 | 1127.066 | 1124.136 | 2.93 | 0.26% | 1056.439 |

## 4. Verdict — bottleneck class per shape

Threshold per requirements.md §2.6: if `launch_tax / plain > 30%` the shape is marked `launch_bound` and CUDA Graph is the recommended first optimisation.  Otherwise the shape is `kernel_bound` and the Phase 2 microscope should drive the next step.

| shape | launch_tax % | class | recommended next step |
|---|---:|---|---|
| decode_T1_q_2048_2048 | 71.84% | **launch_bound** | CUDA Graph capture (Phase 3 cluster A) |
| worst_T8_kv_1024_2048 | 69.19% | **launch_bound** | CUDA Graph capture (Phase 3 cluster A) |
| mid_T128_kv_2560_2048 | 20.37% | kernel_bound | Phase 2 microscope (SASS + microbench bisection) |
| large_T1024_gu_4096_24576 | 0.26% | kernel_bound | Phase 2 microscope (SASS + microbench bisection) |

## 5. Known measurement caveats

* `inter_kernel_gap_us` can be slightly negative in theory because of NVTX push/pop timestamps not being perfectly aligned with CUDA stream timestamps; we clamp at 0.  
* `nsys` has ~1% CUPTI sampling overhead; raw `t_plain_us` from the launch-tax driver is the authoritative wall-clock.  
* Kernel body time in this table comes from CUPTI kernel events *inside* each `phase1.iter_<i>` window — warmup kernels are excluded automatically by the attribution logic, so this number is cleaner than what `launch_tax.md`'s `t_body` column reports.  
* GPU-metric sampling was **not** enabled (container PMU blocked, ERR_NVGPUCTRPERM).  Phase 2 will substitute with microbench bisection + SASS static analysis.
