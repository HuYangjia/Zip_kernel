"""Sanity check: verify the r65 "regression" shapes are GPU drift, not
real dispatcher regression.  Run each suspect + each target in the
same process with trial-randomised median.
"""
import os
import random
import statistics
from pathlib import Path

import torch
import kernel.cuda_kernel.ops as ops


def bench_us(fn, warmup=300, outer=6, inner=150):
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


SUSPECTS = [
    # r65 "regressions" (should be GPU drift only)
    dict(label="0.6B q T=32",   T=32,  d_in=1024, d_out=2048),
    dict(label="0.6B kv T=32",  T=32,  d_in=1024, d_out=2048),
    dict(label="0.6B kv T=128", T=128, d_in=1024, d_out=2048),
    dict(label="1.7B dn T=32",  T=32,  d_in=6144, d_out=2048),
    # Confirmed wins (should still show improvement)
    dict(label="14B gu T=128",  T=128, d_in=5120, d_out=34816),
    dict(label="14B gu T=32",   T=32,  d_in=5120, d_out=34816),
    dict(label="8B gu T=32",    T=32,  d_in=4096, d_out=24576),
]


def prep(sh):
    dev = torch.device("cuda:0")
    T, d_in, d_out = sh["T"], sh["d_in"], sh["d_out"]
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
    def run():
        ops.fused_dense_sparse_cuda_int4(
            W_low, empty_hpb, hp_ro, hp_ci,
            X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
        )
    return run


def main():
    # 5-trial median for each shape
    rng = random.Random(20260430)
    shapes = list(SUSPECTS)
    plan = [(i, t) for i in range(len(shapes)) for t in range(5)]
    rng.shuffle(plan)
    acc = {i: [] for i in range(len(shapes))}
    runs = [prep(sh) for sh in shapes]
    # First warmup all (eliminate compilation effect)
    for i, _ in plan[:3]:  # just a few warm trials in random order first
        bench_us(runs[i], warmup=200)
    # Real measurements
    for i, _ in plan:
        acc[i].append(bench_us(runs[i]))
    print(f"{'shape':<18}  {'med':>7}  {'min':>7}  {'max':>7}  {'spread%':>8}")
    for i, sh in enumerate(shapes):
        m = statistics.median(acc[i])
        mn, mx = min(acc[i]), max(acc[i])
        sp = (mx - mn) / m * 100
        print(f"  {sh['label']:<16}  {m:>7.2f}  {mn:>7.2f}  {mx:>7.2f}  {sp:>+7.1f}%")


if __name__ == "__main__":
    main()
