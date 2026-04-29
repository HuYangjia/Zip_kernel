"""Stage F.3 (r61) — kBm occupancy probe.

After F.2 tightened the group-cache gate, we check whether kBm=64
(halves sW) is now a win for more shapes than R44/R52 allowed.
"""
from __future__ import annotations

import os
import sys
import statistics

import torch

sys.path.insert(0, "/root")
from kernel.cuda_kernel import ops  # noqa: E402


def run_once(d_out, d_in, T, kbm_env, iters=1500):
    if kbm_env == "":
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = kbm_env
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
    bw = (B_gemm + B_quant) / (t_us * 1e3)
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
        f"{'shape':>22}  {'t_def':>7}  {'t_64':>7}  {'t_128':>7}  "
        f"{'bw_def':>6}  {'bw_64':>6}  {'bw_128':>6}  {'64/def':>7}  {'par':>8}"
    )
    print(hdr)
    for shape in shapes:
        t_def, bw_def, y_def = run_once(*shape, kbm_env="")
        t_64, bw_64, y_64 = run_once(*shape, kbm_env="64")
        t_128, bw_128, _ = run_once(*shape, kbm_env="128")
        err_64 = (y_def - y_64).abs().max().item() / (y_def.abs().max().item() + 1e-9)
        print(
            f"{str(shape):>22}  {t_def:7.2f}  {t_64:7.2f}  {t_128:7.2f}  "
            f"{bw_def:6.0f}  {bw_64:6.0f}  {bw_128:6.0f}  {t_def/t_64:6.2f}x  {err_64:8.2e}"
        )


if __name__ == "__main__":
    main()
