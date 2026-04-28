# Capability Probe Report

- Timestamp (UTC): 20260428_084128
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
- cuobjdump: MISSING  `executable not found`
- nvcc: MISSING  `executable not found`
- ptxas: MISSING  `executable not found`

## Torch
- torch 2.8.0+cu126 (CUDA 12.6)
- device: NVIDIA GeForce RTX 4090 (SM (8, 9))
- SM count: 128

## CUDA Graph capture + replay
- ok: True

## nsys GPU Metrics Sampling
- overall ok: False
- report file exists: False
- rc: 1
- permission_denied: False
- stderr tail:
```
Illegal --gpu-metrics-devices arguments.
None of the installed GPUs are supported:
See the user guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics

usage: nsys profile [<args>] [application] [<application args>]
Try 'nsys profile --help' for more information.
```

## cuda_kernel import
- ok: True
- module file: /root/kernel/cuda_kernel/ops.py
- activation_quant_cuda present: True
- fused_dense_sparse_cuda present: True

## Decision matrix
- phase1_timeline_available: False
- phase2_gpu_metrics_available: False
- sass_static_analysis_available: False
- cuda_graph_replay_available: True
- ncu_available: True
- cuda_kernel_importable: True
