# Capability Probe Report

- Timestamp (UTC): 20260428_084505
- Host: autodl-container-d4f6428c8a-e20470d8 (linux)
- Git SHA: n/a
- Python: 3.10.20

## GPU
- Name: NVIDIA GeForce RTX 4090
- Driver: 580.105.08
- Memory: 49140 MiB
- Compute capability: 8.9

## Tooling
- nsys: OK  `NVIDIA Nsight Systems version 2025.1.1.0`
- ncu: OK  `NVIDIA (R) Nsight Compute Command Line Profiler`
- cuobjdump: OK  `cuobjdump: NVIDIA (R) fat binary listing tool`
- nvcc: OK  `nvcc: NVIDIA (R) Cuda compiler driver`
- ptxas: OK  `ptxas: NVIDIA (R) Ptx optimizing assembler`

## Torch
- torch 2.8.0+cu126 (CUDA 12.6)
- device: NVIDIA GeForce RTX 4090 (SM (8, 9))
- SM count: 128

## CUDA Graph capture + replay
- ok: True

## nsys GPU Metrics Sampling
- overall ok: True
- report file exists: True
- rc: 0
- permission_denied: False

## cuda_kernel import
- ok: True
- module file: /root/kernel/cuda_kernel/ops.py
- activation_quant_cuda present: True
- fused_dense_sparse_cuda present: True

## Decision matrix
- phase1_timeline_available: True
- phase2_gpu_metrics_available: False
- phase2_microbench_bisection_available: True
- sass_static_analysis_available: True
- sass_available_with_cuda_path: True
- cuda_graph_replay_available: True
- ncu_available: True
- ncu_sm_counters_available: False
- cuda_kernel_importable: True
