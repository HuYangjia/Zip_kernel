"""C.4 validation: interleaved-trial median for T=48/64 q/o shapes.

Verify whether q/o T=48/64 really regressed after C.4 fixes, or it
was noise in the 3-trial sweep.
"""
import os
import random
import statistics

import torch
import kernel.cuda_kernel.ops as ops


def bench_us(fn, warmup=500, outer=10, inner=200):
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
        best = min(best, s.elapsed_time(e) * 1000.0 / inner)
    return best


def prep(T, d_in, d_out):
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    X = torch.randn(T, d_in, dtype=torch.float16, device=dev) * 0.1
    perm = torch.randperm(d_in, device=dev).to(torch.int32)
    W = torch.randint(0, 16, (d_out, d_in // 2), dtype=torch.int8, device=dev)
    n_g = d_in // 128
    su = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 0.01 + 0.001).contiguous()
    zu = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 14.0).contiguous()
    hpb = torch.zeros((0, 128, 64), dtype=torch.int8, device=dev)
    hpro = torch.zeros((d_out // 128) + 1, dtype=torch.int32, device=dev)
    hpci = torch.zeros(0, dtype=torch.int32, device=dev)
    X_s4, sx, sum_X = ops.activation_quant_cuda(X, perm)

    def run():
        ops.fused_dense_sparse_cuda_int4(
            W, hpb, hpro, hpci, X_s4, su, zu, sum_X, sx, d_out, d_in,
        )
    return run


def set_mode(m):
    for k in ("HKUST_V9_FUSED_FORCE_KBM", "HKUST_V9_FUSED_FORCE_KBN"):
        os.environ.pop(k, None)
    if m == "kBm=64":
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = "64"
    elif m == "k64":
        os.environ["HKUST_V9_FUSED_FORCE_KBN"] = "64"


CASES = [
    (48, 4096, 4096, "T=48 (q/o)"),
    (64, 4096, 4096, "T=64 (q/o)"),
    (48, 4096, 2048, "T=48 kv"),
    (48, 14336, 4096, "T=48 dn"),
]

N_TRIALS = 7
MODES = ["auto", "kBm=64", "k64"]

rng = random.Random(1)
for T, d_in, d_out, lbl in CASES:
    run = prep(T, d_in, d_out)
    plan = [(m, i) for m in MODES for i in range(N_TRIALS)]
    rng.shuffle(plan)
    per = {m: [] for m in MODES}
    for m, _ in plan:
        set_mode(m)
        per[m].append(bench_us(run))
    auto_m = statistics.median(per["auto"])
    km64_m = statistics.median(per["kBm=64"])
    k64_m = statistics.median(per["k64"])
    print("%-12s  auto=%.2f  kBm=64=%.2f  k64=%.2f  gain vs best: %.1f%%" %
          (lbl, auto_m, km64_m, k64_m,
           (auto_m - min(km64_m, k64_m)) / auto_m * 100))
