"""R34 Split-K micro-benchmark: target kv_proj / down_proj mid-batch shapes.

A/B comparison: CUDA R34 (auto) vs CUDA R34 with Split-K force-disabled
(HKUST_V9_DISABLE_SPLITK=1).  Also shows FP16 baseline.
"""
import os
import sys
sys.path.insert(0, '/root')

import torch

from kernel.cuda_kernel import ops
from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.pack_utils import BCOL, pack_s4_le


def make_inputs(T, d_out, d_in, seed=0xBEEF, device="cuda"):
    torch.manual_seed(seed)
    X = torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device=device)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)

    n_groups = d_in // BCOL
    W_low_s4 = torch.randint(-8, 8, (d_out, d_in), dtype=torch.int8, device=device)
    W_low_packed = pack_s4_le(W_low_s4)
    scale_u4 = (torch.rand(d_out, n_groups, device=device) * 0.05 + 0.001).to(torch.float16)
    zero_u4 = (torch.randn(d_out, n_groups, device=device) * 0.2).to(torch.float16)
    return W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, X


def bench(fn, n_warm=10, n_outer=10, n_inner=50):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()

    means = []
    for _ in range(n_outer):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        means.append(start.elapsed_time(end) * 1000.0 / n_inner)
    return min(means)


SHAPES = [
    # (T, d_out, d_in, label) -- Split-K should trigger when base_grid < 64.
    (32,  1024, 4096,  "kv_T32_1k_4k"),
    (64,  1024, 4096,  "kv_T64_1k_4k"),
    (128, 1024, 4096,  "kv_T128_1k_4k"),
    (256, 1024, 4096,  "kv_T256_1k_4k"),
    (32,  2048, 4096,  "kv_T32_2k_4k"),
    (64,  2048, 4096,  "kv_T64_2k_4k"),
    (32,  4096, 4096,  "q_T32_4k_4k"),
    (64,  4096, 4096,  "q_T64_4k_4k"),
    (128, 4096, 4096,  "q_T128_4k_4k"),
    (32,  4096, 12288, "down_T32_4k_12k"),
    (64,  4096, 12288, "down_T64_4k_12k"),
]


def run_cuda(args):
    return ops.dense_gemm_cuda_int4(*args)


def run_fp16(W_deq_fp16, X_fp16):
    return torch.matmul(W_deq_fp16, X_fp16.t())


print("\n| Shape | T | d_out | d_in | base | FP16 | R34(off) | R34(on) | "
      "off/FP16 | on/FP16 | on/off |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

results = []
for T, d_out, d_in, label in SHAPES:
    W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, X_fp16 = make_inputs(T, d_out, d_in)
    cuda_args = (W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x)

    # FP16 baseline: dequantise W first.
    Wg = W_low_packed.view(d_out, d_in // 2, 1)
    low  = (Wg & 0x0F).to(torch.int8)
    high = ((Wg >> 4) & 0x0F).to(torch.int8)
    q_W = torch.cat([low, high], dim=-1).view(d_out, d_in // BCOL, BCOL).float()
    W_deq = ((q_W - zero_u4.unsqueeze(-1).float()) * scale_u4.unsqueeze(-1).float()).view(d_out, d_in).to(torch.float16)

    t_fp = bench(lambda: run_fp16(W_deq, X_fp16))

    # Force split-k OFF
    os.environ["HKUST_V9_DISABLE_SPLITK"] = "1"
    t_off = bench(lambda: run_cuda(cuda_args))

    # Split-K ON (default dispatch decides)
    os.environ["HKUST_V9_DISABLE_SPLITK"] = "0"
    t_on = bench(lambda: run_cuda(cuda_args))

    base_grid = -(-d_out // 128) * -(-T // 32)
    print(f"| {label} | {T} | {d_out} | {d_in} | {base_grid} | "
          f"{t_fp:6.2f} | {t_off:6.2f} | {t_on:6.2f} | "
          f"{t_fp/t_off:.2f}x | {t_fp/t_on:.2f}x | {t_off/t_on:.2f}x |")
    results.append((label, t_fp, t_off, t_on))

# Summary
print("\n=== Summary: Split-K wins ===")
wins, losses, neutral = 0, 0, 0
for label, fp, off, on in results:
    delta = (off - on) / off * 100
    if delta > 3:
        mark = "WIN"
        wins += 1
    elif delta < -3:
        mark = "LOSS"
        losses += 1
    else:
        mark = "==="
        neutral += 1
    print(f"  [{mark:4s}] {label:20s}  off={off:6.2f}us on={on:6.2f}us  delta={delta:+.1f}%")
print(f"\n  wins={wins}  losses={losses}  neutral={neutral}")
