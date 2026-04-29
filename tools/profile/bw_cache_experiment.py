"""Stage F (r61) — occupancy hypothesis experiment.

Drives HKUST_V9_FUSED_FORCE_CACHE={'', '0', '1'} across the canonical
11-shape sweep and reports effective HBM bandwidth for each mode, plus a
parity check of the cache=0 path against the default path.

If cache=0 yields materially higher effective HBM bandwidth without
parity regressions, the "smem-induced low occupancy caps HBM BW"
hypothesis is confirmed and we can proceed with the permanent smem
reduction in F.2.
"""
from __future__ import annotations

import os
import sys
import statistics

import torch

sys.path.insert(0, "/root/Zip_kernel")
from kernel.cuda_kernel import ops  # noqa: E402


def run_once(d_out, d_in, T, cache_env, iters=1500):
    if cache_env == "":
        os.environ.pop("HKUST_V9_FUSED_FORCE_CACHE", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_CACHE"] = cache_env
    ng = d_in // 128
    device = "cuda"
    torch.manual_seed(0)
    W = torch.randint(-8, 7, (d_out, d_in // 2), dtype=torch.int8, device=device)
    X = torch.randint(-8, 7, (T, d_in // 2), dtype=torch.int8, device=device)
    s = torch.rand(d_out, ng, dtype=torch.float16, device=device) * 0.1 + 0.01
    z = torch.randint(-4, 4, (d_out, ng), device=device).to(torch.float16)
    sx = torch.rand(T, dtype=torch.float16, device=device) * 0.1 + 0.01
    sumX = torch.randint(-100, 100, (T, ng), dtype=torch.int32, device=device)
    Y = torch.empty((d_out, T), dtype=torch.float16, device=device)
    Wh = torch.zeros(0, 128, 64, dtype=torch.int8, device=device)
    ro = torch.zeros(d_out // 128 + 1, dtype=torch.int32, device=device)
    ci = torch.zeros(0, dtype=torch.int32, device=device)

    def call():
        ops._ext.fused_dense_sparse_mma_int4_launch(
            W, Wh, ro, ci, X, s, z, sumX, sx, Y, d_out, d_in
        )

    for _ in range(500):
        call()
    torch.cuda.synchronize()

    # Median of 5 windows, each iters launches, for lower noise.
    samples = []
    for _ in range(5):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            call()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1) / iters * 1000)
    t_us = statistics.median(samples)

    B_gemm = 0.5 * d_in * d_out + 0.5 * T * d_in + 4 * d_out * ng + 2 * T * d_out
    B_quant = 2 * T * d_in + 0.5 * T * d_in + 2 * T + 4 * T * ng
    bw = (B_gemm + B_quant) / (t_us * 1e3)  # GB/s
    return t_us, bw, Y.clone()


def main():
    shapes = [
        (1024, 1024, 128),
        (2048, 2048, 128),
        (4096, 4096, 128),
        (1024, 4096, 128),
        (4096, 1024, 128),
        (2048, 4096, 128),
        (4096, 2048, 128),
        (4096, 4096, 32),
        (4096, 4096, 1),
        (4096, 14336, 128),
        (14336, 4096, 128),
    ]
    hdr = (
        f"{'shape':>22}  {'t_def':>7}  {'t_off':>7}  {'t_on':>7}  "
        f"{'bw_def':>6}  {'bw_off':>6}  {'bw_on':>6}  {'off/def':>7}  {'par':>8}"
    )
    print(hdr)
    for shape in shapes:
        t_def, bw_def, y_def = run_once(*shape, cache_env="")
        t_off, bw_off, y_off = run_once(*shape, cache_env="0")
        t_on, bw_on, y_on = run_once(*shape, cache_env="1")
        denom = y_def.abs().max().item() + 1e-9
        err_off = (y_def - y_off).abs().max().item() / denom
        speedup = t_def / t_off
        print(
            f"{str(shape):>22}  {t_def:7.2f}  {t_off:7.2f}  {t_on:7.2f}  "
            f"{bw_def:6.0f}  {bw_off:6.0f}  {bw_on:6.0f}  {speedup:6.2f}x  {err_off:8.2e}"
        )


if __name__ == "__main__":
    main()
