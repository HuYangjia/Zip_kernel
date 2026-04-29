"""Single-shot launcher so NCU can pin exactly one kernel invocation.

Usage (inside NCU wrapper):
    ncu --set full --launch-skip 200 --launch-count 1 \
        python ncu_bottleneck_probe.py <d_out> <d_in> <T>
"""
from __future__ import annotations
import os, sys, torch
sys.path.insert(0, "/root")
from kernel.cuda_kernel import ops  # noqa: E402

def main():
    d_out, d_in, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ng = d_in // 128
    torch.manual_seed(0)
    W = torch.randint(-8, 7, (d_out, d_in // 2), dtype=torch.int8, device="cuda")
    X = torch.randint(-8, 7, (T, d_in // 2), dtype=torch.int8, device="cuda")
    s = torch.rand(d_out, ng, dtype=torch.float16, device="cuda") * 0.1 + 0.01
    z = torch.randint(-4, 4, (d_out, ng), device="cuda").to(torch.float16)
    sx = torch.rand(T, dtype=torch.float16, device="cuda") * 0.1 + 0.01
    sumX = torch.randint(-100, 100, (T, ng), dtype=torch.int32, device="cuda")
    Y = torch.empty((d_out, T), dtype=torch.float16, device="cuda")
    Wh = torch.zeros(0, 128, 64, dtype=torch.int8, device="cuda")
    ro = torch.zeros(d_out // 128 + 1, dtype=torch.int32, device="cuda")
    ci = torch.zeros(0, dtype=torch.int32, device="cuda")

    def call():
        ops._ext.fused_dense_sparse_mma_int4_launch(
            W, Wh, ro, ci, X, s, z, sumX, sx, Y, d_out, d_in
        )

    # 200 warmup so JIT/caches settle before NCU samples.
    for _ in range(400):
        call()
    torch.cuda.synchronize()
    # One launch for NCU to capture.
    call()
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()
