"""Revisit R16's rejection of fused_gemv_smallT under the current
r62-F2 dispatcher (kBm=64 gate, split-K, improved kBn selection).

At R16 the smallT kernel (dp4a, 1-warp-per-row, loops T cols) was
slower than main MMA.  After r62 F2 the main kernel's small-T
branch has different grid-deficit handling, so the trade-off may
have shifted.

Output: per-shape speedup smallT / main_MMA at T in {2, 4, 8, 16}.
"""
import kernel.cuda_kernel.ops as ops
import torch

dev = torch.device("cuda:0")


def bench_us(fn, warmup=200, outer=10, inner=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000.0 / inner
        best = min(best, us)
    return best


SHAPES = [
    # (d_in, d_out, label)
    (1024,  2048, "Qwen3-0.6B q"),
    (2048,  2048, "Qwen3-1.7B q"),
    (2048,  1024, "Qwen3-0.6B o"),
    (4096,  4096, "Qwen3-8B q"),
    (4096,  2048, "Qwen3-8B kv"),
    (4096, 12288, "Qwen3-8B gu"),
    (14336, 4096, "Qwen3-8B down"),
]
TS = [2, 4, 8, 16]

print(f"{'shape':<18} {'T':>3} {'mma us':>7} {'smT us':>7} {'smT/mma':>8} {'winner':>8}")
for d_in, d_out, lbl in SHAPES:
    for T in TS:
        torch.manual_seed(0)
        X = torch.randn(T, d_in, dtype=torch.float16, device=dev) * 0.1
        perm = torch.randperm(d_in, device=dev).to(torch.int32)

        W_low = torch.randint(0, 16, (d_out, d_in // 2), dtype=torch.int8, device=dev)
        n_g = d_in // 128
        scale_u4 = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 0.01 + 0.001).contiguous()
        zero_u4  = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 14.0).contiguous()
        empty_hpb = torch.zeros((0, 128, 64), dtype=torch.int8, device=dev)
        hp_ro = torch.zeros((d_out // 128) + 1, dtype=torch.int32, device=dev)
        hp_ci = torch.zeros(0, dtype=torch.int32, device=dev)

        X_s4, scale_x, sum_X = ops.activation_quant_cuda(X, perm)

        def run_mma():
            ops.fused_dense_sparse_cuda_int4(
                W_low, empty_hpb, hp_ro, hp_ci,
                X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
            )

        def run_smt():
            ops.fused_gemv_cuda_smallT(
                W_low, empty_hpb, hp_ro, hp_ci,
                X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
            )

        us_mma = bench_us(run_mma)
        try:
            us_smt = bench_us(run_smt)
            winner = "smT" if us_smt < us_mma * 0.95 else ("mma" if us_mma < us_smt * 0.95 else "tie")
            print(f"{lbl:<18} {T:>3} {us_mma:>7.2f} {us_smt:>7.2f} {us_smt/us_mma:>7.2f}x {winner:>8}")
        except Exception as ex:
            print(f"{lbl:<18} {T:>3} {us_mma:>7.2f}  SKIP: {ex!s:.40}")
