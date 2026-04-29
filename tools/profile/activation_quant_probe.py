"""P0 probe — measure activation_quant CUDA kernel vs HBM roofline.

Reads the gather path (X[t, perm[.]]) so the theoretical lower bound is
    t_roof = (T*D*sizeof(fp16) + T*D/2 + T*ng*4 + T*2) / 1TB/s

We compare measured us vs roof to spot the gap, and bucket by T to see
the per-SM-utilisation story.
"""
from __future__ import annotations
import argparse
import os
import statistics
from dataclasses import dataclass
from typing import List


@dataclass
class Row:
    T: int
    D: int
    ng: int
    t_us: float
    roof_us: float

    @property
    def eff(self) -> float:
        return self.roof_us / self.t_us * 100 if self.t_us > 0 else 0.0


def _roof_us(T: int, D: int, peak_bw_gbps: float = 1008.0, eff: float = 0.85) -> float:
    """HBM-bound lower bound for activation_quant.

    Traffic per call:
      - read X (T, D) fp16 via gather on perm   -> T*D*2 bytes (random)
      - read perm (D) int32                     -> D*4 bytes (coalesced)
      - write X_s4 (T, D/2) int8                -> T*(D/2) bytes
      - write scale_x (T,) fp16                 -> T*2 bytes
      - write sum_X (T, D/128) int32            -> T*(D/128)*4 bytes
    """
    ng = D // 128
    bytes_read_X    = T * D * 2
    bytes_read_perm = D * 4
    bytes_write_s4  = T * (D // 2)
    bytes_write_sc  = T * 2
    bytes_write_sm  = T * ng * 4
    total = bytes_read_X + bytes_read_perm + bytes_write_s4 + bytes_write_sc + bytes_write_sm
    bw = peak_bw_gbps * 1e9 * eff
    return total / bw * 1e6


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


def run():
    import torch
    from kernel.cuda_kernel import ops as _ops  # noqa

    shapes = []
    for D in (2048, 4096, 12288):
        for T in (1, 2, 4, 8, 16, 32, 64, 128, 512):
            shapes.append((T, D))

    rows: List[Row] = []
    for (T, D) in shapes:
        ng = D // 128
        torch.manual_seed(0)
        X   = torch.randn(T, D, dtype=torch.float16, device="cuda")
        perm = torch.arange(D, dtype=torch.int32, device="cuda")
        X_s4    = torch.empty((T, D // 2), dtype=torch.int8, device="cuda")
        scale_x = torch.empty((T,), dtype=torch.float16, device="cuda")
        sum_X   = torch.empty((T, ng), dtype=torch.int32, device="cuda")

        def fn():
            _ops._ext.activation_quant_launch(
                X, perm, X_s4, scale_x, sum_X, T, D, 128
            )

        t_us = _bench(fn, warmup=200, outer=10, inner=200)
        rows.append(Row(T=T, D=D, ng=ng, t_us=t_us, roof_us=_roof_us(T, D)))

    # Print table.
    print()
    print(f"{'T':>4} {'D':>6} {'ng':>4} {'meas_us':>9} {'roof_us':>9} {'eff':>6} {'ratio':>7}")
    for r in rows:
        print(f"{r.T:>4} {r.D:>6} {r.ng:>4} {r.t_us:>9.2f} {r.roof_us:>9.2f} {r.eff:>5.1f}% {r.t_us/r.roof_us:>6.1f}x")

    # Bucket by T.
    print()
    print("== Grouped by T (median us) ==")
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[r.T].append(r)
    for T in sorted(g):
        rs = g[T]
        print(f"T={T:>3}: n={len(rs)}  median_us={statistics.median(r.t_us for r in rs):.2f}  "
              f"median_eff={statistics.median(r.eff for r in rs):.1f}%")


if __name__ == "__main__":
    run()
