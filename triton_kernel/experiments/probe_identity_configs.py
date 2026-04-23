"""Probe the best fixed config for identity-perm large-T shapes.

Motivation: sweep_v9 uses identity perm and reports quant regressed
+37% at T=8192 d_in=14336. Microbench with random perm showed
autotune picked BT=16 BD=256 stages=2 because that was best for
L2-thrashing random-perm access.  But identity perm likely wants
a *large* BD config for max memcpy throughput.

This script measures a few candidate configs on identity-perm
large-T shapes to see which is actually best.
"""
import torch
import triton
import triton.language as tl
from kernel.triton_kernel.activation_quant import (
    quantize_activation_kernel_fast,
    quantize_activation_s4,
)
from kernel.triton_kernel.pack_utils import BCOL
from kernel.triton_kernel.benchmarks._bench_util import time_ms


CANDIDATES = [
    # (BT, BD, num_warps, num_stages)
    (16, 256, 2, 2),    # what autotune currently picks for identity T=8192
    (64, 1024, 4, 3),
    (64, 2048, 8, 3),
    (128, 1024, 8, 2),
    (128, 2048, 8, 3),  # original "large-BD" config (deleted earlier)
    (256, 512, 8, 2),
]


def bench_one(T, D, BT, BD, nw, ns, perm_fn):
    n_groups = D // BCOL
    X = torch.randn(T, D, dtype=torch.float16, device='cuda') * 0.5
    perm = perm_fn(D)
    X_s4 = torch.empty((T, D // 2), dtype=torch.int8, device='cuda')
    scale_x = torch.empty((T,), dtype=torch.float16, device='cuda')
    sum_X = torch.empty((T, n_groups), dtype=torch.int32, device='cuda')
    grid = (triton.cdiv(T, BT),)

    def run():
        quantize_activation_kernel_fast[grid](
            X, perm, X_s4, scale_x, sum_X, T, D,
            X.stride(0), X.stride(1),
            X_s4.stride(0), X_s4.stride(1),
            sum_X.stride(0), sum_X.stride(1),
            N_GROUPS=n_groups, BCOL_K=BCOL,
            BT=BT, BD=BD, num_warps=nw, num_stages=ns,
        )

    # warmup
    for _ in range(50):
        run()
    torch.cuda.synchronize()
    try:
        t = time_ms(run, n_warmup=50, n_iter=100, n_repeat=3) * 1000
    except Exception as e:
        return None
    return t


def main():
    shapes = [
        (8192, 11008),
        (8192, 14336),
        (2048, 14336),
    ]
    print("=== IDENTITY perm ===")
    print(f"{'T':>5} {'D':>6} " + "  ".join(f"{f'{bt},{bd}/w{nw}s{ns}':>16}" for (bt, bd, nw, ns) in CANDIDATES))
    id_fn = lambda D: torch.arange(D, dtype=torch.int32, device='cuda')
    for T, D in shapes:
        row = []
        for (BT, BD, nw, ns) in CANDIDATES:
            # Skip absurd configs early: BT > T is OK (mask), BD > D is OK (mask),
            # but if BT*BD is too big we'll OOM shared mem -> catch and skip.
            t = bench_one(T, D, BT, BD, nw, ns, id_fn)
            row.append(f"{t:>14.1f}us" if t else f"{'FAIL':>16}")
        print(f"{T:>5} {D:>6} " + "  ".join(row))

    print()
    print("=== RANDOM perm ===")
    rnd_fn = lambda D: torch.randperm(D, dtype=torch.int32, device='cuda')
    print(f"{'T':>5} {'D':>6} " + "  ".join(f"{f'{bt},{bd}/w{nw}s{ns}':>16}" for (bt, bd, nw, ns) in CANDIDATES))
    for T, D in shapes:
        row = []
        for (BT, BD, nw, ns) in CANDIDATES:
            t = bench_one(T, D, BT, BD, nw, ns, rnd_fn)
            row.append(f"{t:>14.1f}us" if t else f"{'FAIL':>16}")
        print(f"{T:>5} {D:>6} " + "  ".join(row))


if __name__ == '__main__':
    main()
