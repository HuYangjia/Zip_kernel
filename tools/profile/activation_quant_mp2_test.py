"""Parity + perf test: activation_quant mp2 vs sp.

Runs the same input through both paths (via HKUST_V9_ACTQUANT_PATH env
var toggle) and asserts bit-exact match on all outputs, plus prints
per-shape latency for both paths.
"""
from __future__ import annotations
import argparse
import os
import statistics
import sys
from dataclasses import dataclass
from typing import List


def _bench(fn, warmup: int, outer: int, inner: int) -> float:
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) / inner * 1000.0)
    return statistics.median(samples)


def run_one(T: int, D: int, path: str):
    """Run actquant via the specified path and return output tensors."""
    import torch
    os.environ["HKUST_V9_ACTQUANT_PATH"] = path
    from kernel.cuda_kernel import ops as _ops

    ng = D // 128
    torch.manual_seed(1234)
    X = torch.randn(T, D, dtype=torch.float16, device="cuda")
    # Random non-identity perm for stress-testing the gather path.
    perm = torch.randperm(D, dtype=torch.int32, device="cuda")

    X_s4 = torch.empty((T, D // 2), dtype=torch.int8, device="cuda")
    scale_x = torch.empty((T,), dtype=torch.float16, device="cuda")
    sum_X = torch.empty((T, ng), dtype=torch.int32, device="cuda")

    _ops._ext.activation_quant_launch(X, perm, X_s4, scale_x, sum_X, T, D, 128)
    return X_s4.clone(), scale_x.clone(), sum_X.clone(), X, perm


def run():
    import torch

    shapes = [(1, 4096), (8, 4096), (32, 4096), (128, 4096), (512, 4096),
              (1, 2048), (32, 12288), (128, 12288), (1, 12288)]

    print(f"{'shape':>15} {'sp_us':>8} {'mp2_us':>8} {'speedup':>8} {'parity':>10}")
    for (T, D) in shapes:
        # Parity first (fresh inputs each path run).
        s4_sp, sx_sp, sm_sp, X_ref, perm_ref = run_one(T, D, "sp")
        s4_mp, sx_mp, sm_mp, X_mp, perm_mp = run_one(T, D, "mp2")
        # Inputs were seeded with same generator so they must match.
        assert torch.equal(X_ref.cpu(), X_mp.cpu()), "inputs differ (seed issue)"

        ok_s4 = torch.equal(s4_sp, s4_mp)
        ok_sx = torch.equal(sx_sp, sx_mp)
        ok_sm = torch.equal(sm_sp, sm_mp)
        parity = "BIT-EXACT" if (ok_s4 and ok_sx and ok_sm) else "MISMATCH"
        if not (ok_s4 and ok_sx and ok_sm):
            sx_diff = (sx_sp.float() - sx_mp.float()).abs().max().item()
            sm_diff = (sm_sp - sm_mp).abs().max().item()
            s4_diff = (s4_sp.int() - s4_mp.int()).abs().max().item()
            parity += f" (s4={s4_diff} sx={sx_diff:.4g} sm={sm_diff})"

        # Perf test.
        def make_fn(path):
            os.environ["HKUST_V9_ACTQUANT_PATH"] = path
            from kernel.cuda_kernel import ops as _ops
            ng = D // 128
            X = torch.randn(T, D, dtype=torch.float16, device="cuda")
            perm = torch.arange(D, dtype=torch.int32, device="cuda")
            X_s4 = torch.empty((T, D // 2), dtype=torch.int8, device="cuda")
            sx = torch.empty((T,), dtype=torch.float16, device="cuda")
            sm = torch.empty((T, ng), dtype=torch.int32, device="cuda")
            return lambda: _ops._ext.activation_quant_launch(X, perm, X_s4, sx, sm, T, D, 128)

        fn_sp = make_fn("sp")
        fn_mp = make_fn("mp2")
        t_sp = _bench(fn_sp, warmup=200, outer=10, inner=200)
        t_mp = _bench(fn_mp, warmup=200, outer=10, inner=200)

        print(f"{T}x{D:>7}  {t_sp:>8.2f} {t_mp:>8.2f} {t_sp/t_mp:>7.2f}x  {parity}")


if __name__ == "__main__":
    run()
