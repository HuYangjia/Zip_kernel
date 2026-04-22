"""Micro-benchmark the runtime cost of the prefill/decode dispatcher.

Following the project's GPU micro-bench protocol (see
kernel/triton_kernel/benchmarks/_bench_util.py):

- >=50 iterations of warm-up so the GPU is on boost clocks and Triton
  autotune cache is hot
- each measurement window is >=100 iterations
- >=3 windows, report min-of-means (rejects CUDA runtime jitter)

We compare three entry points on the exact same pipeline body:
  v9_linear_forward          -- dispatcher (branch on T)
  v9_linear_forward_decode   -- explicit decode entry
  v9_linear_forward_prefill  -- explicit prefill entry

On decode shapes (T <= 128), the "prefill entry" still routes through
the decode body because its implementation falls back on the correct
regime; this is a correctness guarantee, not a perf claim.

The cost of the dispatcher is just one integer compare + one Python
call, so it should be <<1 us relative to pipeline work of 100-500 us.
Any measurable regression here would be a red flag.
"""

import argparse
import torch

from kernel.triton_kernel.v9_linear import (
    DECODE_T_THRESHOLD,
    v9_linear_forward,
    v9_linear_forward_decode,
    v9_linear_forward_prefill,
)
from kernel.triton_kernel.tests.test_end2end import _synthesize_pack
from kernel.triton_kernel.benchmarks._bench_util import time_ms


CASES = [
    # (label, bs, d_in, d_out, hp_ratio)
    ("decode-bs1-hp0",      1,    4096, 4096,  0.0),
    ("decode-bs1-hp10",     1,    4096, 4096,  0.1),
    ("decode-bs64-hp5",     64,   4096, 11008, 0.05),
    ("prefill-bs512-hp0",   512,  4096, 4096,  0.0),
    ("prefill-bs2048-hp10", 2048, 4096, 4096,  0.1),
]


def bench_one(label, bs, d_in, d_out, hp_ratio):
    torch.manual_seed(0)
    W = _synthesize_pack(d_out=d_out, d_in=d_in, hp_ratio=hp_ratio, seed=0)
    X = torch.randn(bs, d_in, dtype=torch.float16, device="cuda")

    def f_dispatch():
        return v9_linear_forward(X, W)

    def f_decode():
        return v9_linear_forward_decode(X, W)

    def f_prefill():
        return v9_linear_forward_prefill(X, W)

    # Extra warm-up pass over all three entries so each has hit its
    # autotune cache once (shared underlying kernels, so this is fast).
    for _ in range(5):
        f_dispatch()
        f_decode()
        f_prefill()
    torch.cuda.synchronize()

    t_disp = time_ms(f_dispatch)
    t_dec = time_ms(f_decode)
    t_pre = time_ms(f_prefill)

    overhead_dec = (t_disp - t_dec) * 1000  # us, signed
    overhead_pre = (t_disp - t_pre) * 1000

    regime = "decode" if bs <= DECODE_T_THRESHOLD else "prefill"
    print(
        "{:<25} T={:<5} {:<7} | dispatch={:7.4f}ms  decode={:7.4f}ms  prefill={:7.4f}ms"
        " | overhead_vs_decode={:+5.1f}us  vs_prefill={:+5.1f}us".format(
            label, bs, regime, t_disp, t_dec, t_pre, overhead_dec, overhead_pre
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=DECODE_T_THRESHOLD)
    args = parser.parse_args()
    print(f"DECODE_T_THRESHOLD = {args.threshold}")
    print(f"Benchmark entries (min-of-means over 3 windows of 100 iters)\n")
    for case in CASES:
        bench_one(*case)


if __name__ == "__main__":
    main()
