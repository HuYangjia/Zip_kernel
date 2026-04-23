"""Measure per-stage CUDA launch overhead in the V9 decode path.

Goal: verify that the 3 sequential kernel launches (quant + dense + sparse)
are the dominant cost when T<=16, which would validate CUDA Graph capture
as the primary decode-speedup direction.

Approach:
  - Build a decode-sized input (T=1, d_out=4096, d_in=4096, hp=0).
  - Measure 3 variants back-to-back under the min-of-means protocol:
      (1) baseline  : v9_linear_forward_decode, warm + 3 windows of 300 iters
      (2) graphed   : same call but wrapped in torch.cuda.CUDAGraph.replay
      (3) per-stage : call each of quant / dense / combine separately with
                       the same protocol, to size their individual overhead.
  - Delta (1) - (2) gives the launch-overhead amortised by CUDA Graph.
  - (3) components should sum to approximately (1) minus small bookkeeping.

Run on the server only:
    python -m kernel.triton_kernel.benchmarks.bench_decode_launch_overhead
"""
from __future__ import annotations

import time
import torch

from kernel.triton_kernel.benchmarks.sweep_v9 import _build_pack
from kernel.triton_kernel.v9_linear import V9WeightContainer, v9_linear_forward_decode


def _min_of_means_ms(fn, warmup=80, windows=3, iters=300):
    """Standard decode-regime micro-timer (see team memo)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(windows):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        means.append((t1 - t0) / iters * 1000.0)  # ms
    return min(means), means


def bench(T, d_out, d_in, hp_ratio, device="cuda", verbose=True):
    torch.manual_seed(0)
    W = _build_pack(d_out, d_in, hp_ratio)
    X_fp16 = torch.randn(T, d_in, device=device, dtype=torch.float16)

    def f_plain():
        return v9_linear_forward_decode(X_fp16, W)

    # Warm everything first so autotune settles.
    for _ in range(50):
        f_plain()
    torch.cuda.synchronize()

    plain_ms, _ = _min_of_means_ms(f_plain, warmup=50, windows=3, iters=300)

    # ---- CUDA Graph capture ----
    # Capture a single forward, then replay N times in the timer loop.
    static_X = X_fp16.clone()
    # Prime via one plain call.
    _ = v9_linear_forward_decode(static_X, W)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    # The output tensor must be allocated OUTSIDE the capture region so
    # that replay writes into a stable address.
    with torch.cuda.graph(g):
        static_Y = v9_linear_forward_decode(static_X, W)

    def f_graph():
        g.replay()
        return static_Y

    graph_ms, _ = _min_of_means_ms(f_graph, warmup=50, windows=3, iters=300)

    if verbose:
        print(f"T={T:>4}  d_out={d_out:>5}  d_in={d_in:>5}  hp={hp_ratio:.2f}")
        print(f"  plain   : {plain_ms*1000:7.2f} us")
        print(f"  graphed : {graph_ms*1000:7.2f} us")
        print(f"  saved   : {(plain_ms-graph_ms)*1000:7.2f} us  ({100*(plain_ms-graph_ms)/plain_ms:+.1f}%)")
        print()
    return plain_ms, graph_ms


def main():
    print("=" * 72)
    print("V9 Decode path: CUDA Graph amortisation probe")
    print(f"Torch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    cases = [
        (1,   4096,  4096, 0.0),
        (1,   4096,  4096, 0.05),
        (1,  11008,  4096, 0.0),
        (1,  11008,  4096, 0.05),
        (1,  14336,  4096, 0.0),
        (1,  14336,  4096, 0.05),
        (1,  28672,  4096, 0.0),
        (1,  28672,  4096, 0.05),
        (16,  4096,  4096, 0.0),
        (16,  4096,  4096, 0.05),
        (16, 14336,  4096, 0.05),
        (16, 28672,  4096, 0.05),
        (64,  4096,  4096, 0.0),
        (64, 14336,  4096, 0.05),
    ]
    print(f"\n{'T':>4}{'d_out':>7}{'d_in':>7}{'hp':>6}{'plain_us':>12}{'graph_us':>12}{'saved_us':>11}{'ratio':>9}")
    print("-" * 72)
    for T, d_out, d_in, hp in cases:
        p, g = bench(T, d_out, d_in, hp, verbose=False)
        p_us = p * 1000
        g_us = g * 1000
        saved = p_us - g_us
        ratio = p / g
        print(f"{T:>4}{d_out:>7}{d_in:>7}{hp:>6.2f}{p_us:>11.1f}us{g_us:>11.1f}us{saved:>9.1f}us  {ratio:>5.2f}x")


if __name__ == "__main__":
    main()
