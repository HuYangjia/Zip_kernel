"""Minimal validation: does HKUST_V9_FUSED_FORCE_SPLITK env var take effect?

Launch the kernel under 3 values (1, 4, 8) and print the per-call latency.
If the env var is respected, timings should differ substantially for a
K-long shape like 32B gu T=1024.
"""
import os
import sys
import time
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
    return W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x


T, d_out, d_in = 1024, 55296, 5120  # 32B gu T=1024
args = make_inputs(T, d_out, d_in)


def bench(n=50):
    # warmup
    for _ in range(20):
        ops.dense_gemm_cuda_int4(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        ops.dense_gemm_cuda_int4(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / n


for sk in [1, 2, 4, 8]:
    os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(sk)
    t = bench()
    print(f"force_sk={sk}: {t:.2f} us")

# Also test without env var (auto)
os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)
print(f"auto        : {bench():.2f} us")
