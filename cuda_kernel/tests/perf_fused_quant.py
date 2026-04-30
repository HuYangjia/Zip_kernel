"""P0.2 performance check — fused path vs legacy composition.

Measures the kernel-time speedup on a representative mid-shape set.
Expected after P0.2 (correctness-first): modest gains (~5-15us off
kernel time) because we saved only 1 launch + 1 HBM round-trip for
X_s4/sum_X; no cp.async / group-cache overlap tuning yet.
"""
import kernel.cuda_kernel.ops as ops
import torch
import time

dev = torch.device("cuda:0")


def bench_us(fn, warmup=200, outer=10, inner=100):
    # Minimal version of the project's standard timer skeleton
    # (per memory bmmiahpl).  Same min-over-outer of mean-over-inner.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(outer):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        us = start.elapsed_time(end) * 1000.0 / inner
        best = min(best, us)
    return best


SHAPES = [
    # (T, d_in, d_out, label)
    (32,  4096,  4096,  "Qwen3-8B q_proj T=32"),
    (128, 4096,  4096,  "Qwen3-8B q_proj T=128"),
    (32,  4096,  2048,  "Qwen3-8B kv_proj T=32"),
    (128, 4096,  2048,  "Qwen3-8B kv_proj T=128"),
    (32,  4096, 24576,  "Qwen3-8B gate_up T=32"),
    (128, 4096, 24576,  "Qwen3-8B gate_up T=128"),
    (32,  14336, 4096,  "Qwen3-8B down_proj T=32"),
    (128, 14336, 4096,  "Qwen3-8B down_proj T=128"),
    (32,  1024,  2048,  "Qwen3-0.6B q_proj T=32"),
    (128, 1024,  2048,  "Qwen3-0.6B q_proj T=128"),
    (32,  2048,  2048,  "Qwen3-1.7B q_proj T=32"),
    (128, 2048,  2048,  "Qwen3-1.7B q_proj T=128"),
    (32,  5120, 34816,  "Qwen3-14B gate_up T=32"),
    (128, 5120, 34816,  "Qwen3-14B gate_up T=128"),
]

print(f"{'shape':<32} {'legacy_us':>10} {'fused_us':>10} {'Δus':>8} {'speedup':>8}")
print("-" * 72)

for T, d_in, d_out, label in SHAPES:
    torch.manual_seed(42)
    X = torch.randn(T, d_in, dtype=torch.float16, device=dev) * 0.1
    perm = torch.randperm(d_in, device=dev).to(torch.int32)
    W_low = torch.randint(0, 16, (d_out, d_in // 2), dtype=torch.int8, device=dev)
    n_g = d_in // 128
    scale_u4 = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 0.01 + 0.001).contiguous()
    zero_u4  = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 14.0).contiguous()

    empty_hpb = torch.zeros((0, 128, 64), dtype=torch.int8, device=dev)
    hp_ro = torch.zeros((d_out // 128) + 1, dtype=torch.int32, device=dev)
    hp_ci = torch.zeros(0, dtype=torch.int32, device=dev)

    # Legacy: activation_quant + fused_dense_sparse
    def legacy():
        X_s4, scale_x, sum_X = ops.activation_quant_cuda(X, perm)
        ops.fused_dense_sparse_cuda_int4(
            W_low, empty_hpb, hp_ro, hp_ci,
            X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
        )

    def fused():
        ops.fused_quant_dense_sparse_cuda_int4(
            X, perm, W_low, empty_hpb, hp_ro, hp_ci,
            scale_u4, zero_u4, d_out, d_in,
        )

    us_leg = bench_us(legacy)
    us_fu  = bench_us(fused)
    delta  = us_fu - us_leg
    speedup = us_leg / us_fu if us_fu > 0 else 0.0
    print(f"{label:<32} {us_leg:>9.2f} {us_fu:>9.2f} {delta:>+7.2f} {speedup:>7.2f}x")
