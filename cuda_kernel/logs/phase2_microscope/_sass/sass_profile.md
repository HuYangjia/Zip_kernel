# Phase 2 SASS Static Profile

One row per kernel instantiation compiled into `hkust_v9_cuda.so`.
`TC%` = issue-slot share of HMMA+IMMA (useful as a scheduler
indicator but **not** a compute-share proxy); `MAC_TC%` =
MAC-weighted compute share of TC under sm_89 s4 IMMA throughput
(8192 MAC/IMMA vs 1 MAC/FFMA, 4 MAC/DP4A) — this is the field
that now drives the ``tc_underutil`` verdict; `FMA%` counts
FFMA+IMAD; `LD%` counts LDG+LDS.  Verdicts follow task-item.md
§9 as amended by ``phase2_tc_rediagnosis.md`` (2026-04-28).

| # | family | regs | shared | stack | insts | TC% | MAC_TC% | FMA% | LD% | verdicts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `activation_quant21act_quant_phase_a_maxEPK` | 12 | 16 | 0 | 80 | 0.0 | 0.0 | 17.5 | 3.8 | tc_underutil |
| 2 | `activation_quant22act_quant_phase_b_packEPK6__halfPKiS5_PaPS` | 17 | 16 | 0 | 208 | 0.0 | 0.0 | 30.3 | 1.9 | tc_underutil |
| 3 | `activation_quant23activation_quant_kernelILi1EEEvPK6__halfPKiPaPS` | 32 | 48 | 0 | 584 | 0.0 | 0.0 | 22.3 | 2.7 | tc_underutil |
| 4 | `activation_quant23activation_quant_kernelILi4EEEvPK6__halfPKiPaPS` | 32 | 160 | 0 | 592 | 0.0 | 0.0 | 22.3 | 3.7 | tc_underutil |
| 5 | `activation_quant23activation_quant_kernelILi8EEEvPK6__halfPKiPaPS` | 32 | 304 | 0 | 584 | 0.0 | 0.0 | 22.6 | 2.7 | tc_underutil |
| 6 | `activation_quant26activation_quant_kernel_spILi1EEEvPK6__halfPKiPaPS` | 30 | 0 | 0 | 544 | 0.0 | 0.0 | 23.0 | 4.8 | tc_underutil |
| 7 | `activation_quant26activation_quant_kernel_spILi2EEEvPK6__halfPKiPaPS` | 30 | 0 | 0 | 544 | 0.0 | 0.0 | 23.0 | 4.8 | tc_underutil |
| 8 | `activation_quant26activation_quant_kernel_spILi4EEEvPK6__halfPKiPaPS` | 30 | 0 | 0 | 536 | 0.0 | 0.0 | 23.3 | 3.4 | tc_underutil |
| 9 | `fused_quant_gemv23fused_quant_gemv_kernelILi8EEEvPK6__halfPKilPKhllS4_S4_llllS8_S6_S6_lllPS` | 40 | 16042 | 0 | 1760 | 0.0 | 0.0 | 32.8 | 6.1 | tc_underutil |
| 10 | `dense_gemv_decode24dense_gemv_decode_kernelILi8EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 34 | 706 | 0 | 176 | 0.0 | 0.0 | 27.8 | 1.7 | tc_underutil |
| 11 | `fused_gemv_decode24fused_gemv_decode_kernelILi8EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 39 | 706 | 0 | 504 | 0.0 | 0.0 | 28.0 | 2.8 | tc_underutil |
| 12 | `fused_gemv_smallT24fused_gemv_smallT_kernelILi8ELi16EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 56 | 9248 | 0 | 2184 | 0.0 | 0.0 | 33.4 | 3.4 | tc_underutil |
| 13 | `fused_gemv_smallT24fused_gemv_smallT_kernelILi8ELi2EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 48 | 1156 | 0 | 952 | 0.0 | 0.0 | 36.6 | 3.4 | tc_underutil |
| 14 | `fused_gemv_smallT24fused_gemv_smallT_kernelILi8ELi4EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 44 | 2312 | 0 | 1144 | 0.0 | 0.0 | 35.2 | 3.3 | tc_underutil |
| 15 | `fused_gemv_smallT24fused_gemv_smallT_kernelILi8ELi8EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 48 | 4624 | 0 | 1480 | 0.0 | 0.0 | 34.8 | 3.4 | tc_underutil |
| 16 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi32ELi128ELi128EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 141 | 20800 | 0 | 2672 | 0.6 | 99.4 | 27.4 | 1.8 | high_reg_pressure |
| 17 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi32ELi128ELi64EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 141 | 12608 | 0 | 2672 | 0.6 | 99.4 | 27.4 | 1.8 | high_reg_pressure |
| 18 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi32ELi32ELi128EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 141 | 20800 | 0 | 2672 | 0.6 | 99.4 | 27.4 | 1.8 | high_reg_pressure |
| 19 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi32ELi32ELi64EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 141 | 12608 | 0 | 2672 | 0.6 | 99.4 | 27.4 | 1.8 | high_reg_pressure |
| 20 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi64ELi128ELi128EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 194 | 25216 | 0 | 3616 | 0.9 | 99.6 | 29.2 | 2.0 | high_reg_pressure |
| 21 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi64ELi128ELi64EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 194 | 17024 | 0 | 3616 | 0.9 | 99.6 | 29.3 | 2.0 | high_reg_pressure |
| 22 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi64ELi32ELi128EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 194 | 25216 | 0 | 3616 | 0.9 | 99.6 | 29.2 | 2.0 | high_reg_pressure |
| 23 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi64ELi32ELi64EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 194 | 17024 | 0 | 3616 | 0.9 | 99.6 | 29.3 | 2.0 | high_reg_pressure |
| 24 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi8ELi128ELi128EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 72 | 17488 | 0 | 1432 | 0.3 | 98.9 | 24.9 | 1.3 | - |
| 25 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi8ELi128ELi64EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 72 | 9296 | 0 | 1432 | 0.3 | 98.9 | 24.9 | 1.3 | - |
| 26 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi8ELi32ELi128EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 72 | 17488 | 0 | 1432 | 0.3 | 98.9 | 24.9 | 1.3 | - |
| 27 | `dense_gemm_mma_int426dense_gemm_mma_int4_kernelILi8ELi32ELi64EEEvPKhS3_PK6__halfS6_PKiS6_PS` | 72 | 9296 | 0 | 1432 | 0.3 | 98.9 | 24.9 | 1.3 | - |
| 28 | `sparse_gemm_mma_int427sparse_gemm_mma_int4_kernelILi32EEEvPKhPKiS5_S3_PK6__halfS8_PS` | 130 | 20800 | 0 | 1584 | 1.0 | 99.7 | 23.7 | 1.9 | high_reg_pressure |
| 29 | `sparse_gemm_mma_int427sparse_gemm_mma_int4_kernelILi64EEEvPKhPKiS5_S3_PK6__halfS8_PS` | 170 | 24960 | 0 | 2488 | 1.3 | 99.8 | 24.3 | 1.8 | high_reg_pressure |
| 30 | `sparse_gemm_mma_int427sparse_gemm_mma_int4_kernelILi8EEEvPKhPKiS5_S3_PK6__halfS8_PS` | 60 | 17680 | 0 | 736 | 0.5 | 99.5 | 23.6 | 2.4 | - |
| 31 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi32ELb0ELi128EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 162 | 21056 | 0 | 2880 | 1.1 | 99.7 | 24.7 | 2.2 | high_reg_pressure |
| 32 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi32ELb0ELi64EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 162 | 12736 | 0 | 2880 | 1.1 | 99.7 | 24.5 | 2.2 | high_reg_pressure |
| 33 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi32ELb1ELi128EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 144 | 37440 | 0 | 3864 | 0.8 | 99.6 | 28.0 | 2.4 | high_reg_pressure |
| 34 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi32ELb1ELi64EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 146 | 20928 | 0 | 3872 | 0.8 | 99.6 | 28.1 | 2.4 | high_reg_pressure |
| 35 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi64ELb0ELi128EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 192 | 25472 | 0 | 3840 | 1.7 | 99.8 | 24.1 | 2.8 | high_reg_pressure |
| 36 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi64ELb0ELi64EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 204 | 17152 | 0 | 3856 | 1.7 | 99.8 | 24.2 | 2.8 | high_reg_pressure |
| 37 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi64ELb1ELi128EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 202 | 41856 | 0 | 5000 | 1.3 | 99.7 | 28.2 | 2.6 | high_reg_pressure |
| 38 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi64ELb1ELi64EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 202 | 25344 | 0 | 5000 | 1.3 | 99.7 | 28.3 | 2.6 | high_reg_pressure |
| 39 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi8ELb0ELi128EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 86 | 17744 | 0 | 1552 | 0.5 | 99.5 | 20.8 | 1.7 | - |
| 40 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi8ELb0ELi64EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 86 | 9424 | 0 | 1560 | 0.5 | 99.5 | 20.8 | 1.7 | - |
| 41 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi8ELb1ELi128EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 87 | 34128 | 0 | 2424 | 0.3 | 99.0 | 26.1 | 2.1 | - |
| 42 | `fused_dense_sparse_mma_int434fused_dense_sparse_mma_int4_kernelILi8ELb1ELi64EEEvPKhS3_PK6__halfS6_PKiS6_S3_S8_S8_PS` | 87 | 17616 | 0 | 2424 | 0.3 | 99.0 | 26.3 | 2.1 | - |

## Aggregate verdict counts

- `high_reg_pressure`: **18** / 42 kernels
- `tc_underutil`: **15** / 42 kernels
