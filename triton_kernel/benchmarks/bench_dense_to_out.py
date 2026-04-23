"""Micro-benchmark: dense+transpose fused kernel vs plain dense+combine (P3 Step 1).

Measures three configurations at identical shapes:

  * plain_dense   : dense_gemm_u4_s4(...)
                    -> .transpose(0, 1).contiguous()
  * fused_dense   : dense_gemm_u4_s4_to_out(...)                     [new, P3 step 1]
  * plain_combined: dense_gemm_u4_s4 + the existing combine-transpose Triton
                    kernel used inside v9_linear when T*d_out >= 4M

Only ``quantize_activation_s4`` is NOT measured here (we're isolating the
dense-output path); it is amortised across all three configs.  The focus
is on how much HBM traffic / launch overhead the new fused kernel removes
by eliminating the ``Y_low (d_out, T)`` intermediate.

Timer: min-of-means per CUDA microbench protocol:
    * 50 warm-up iterations  (GPU to boost clock, autotune to converge)
    * 3 windows x 200 iterations each -> record min-of-means
Reports median/mean/p95 in microseconds and fused-vs-plain speedup.

Run:
    python -m kernel.triton_kernel.benchmarks.bench_dense_to_out
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_gemm_to_out import dense_gemm_u4_s4_to_out  # noqa: E402
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton_kernel.v9_linear import _combine_transpose  # noqa: E402

# ---------------------------------------------------------------------------
# V9 container (hp=0) -- copy of the helper in tests/test_dense.py, verbatim.
# ---------------------------------------------------------------------------

def _build_zero_hp_pack(d_out: int, d_in: int, seed: int = 0):
    torch.manual_seed(seed)
    n_groups = d_in // BCOL
    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device="cuda")
    scale_u4 = (torch.rand(d_out, n_groups, device="cuda") * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, n_groups), device="cuda").to(torch.float16)
    Q_s8 = torch.zeros(0, BROW, BCOL, dtype=torch.int8, device="cuda")
    scale_s8 = torch.zeros(0, BROW, dtype=torch.float16, device="cuda")
    hp_idx = torch.zeros((0, 2), dtype=torch.int32, device="cuda")
    perm = torch.arange(d_in, dtype=torch.int32, device="cuda")
    return pack_v9_weights({
        "Q_u4_permuted": Q_u4, "scale_u4_raw": scale_u4, "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8, "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_idx, "perm": perm,
    })


# ---------------------------------------------------------------------------
# Microbench timer (min-of-means)
# ---------------------------------------------------------------------------

def _bench(fn: Callable[[], None], warmup: int = 50, windows: int = 3, iters: int = 200) -> float:
    """Return min-of-means latency in microseconds."""
    # Warm-up: let GPU reach boost clock, Triton autotune converge.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    means_us: List[float] = []
    for _w in range(windows):
        # Use CUDA events for sub-microsecond resolution.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        # elapsed_time is in milliseconds.
        mean_us = start.elapsed_time(end) * 1000.0 / iters
        means_us.append(mean_us)
    return min(means_us)


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

# Decode-focused shapes: covers the subset where Step 1 is expected to shine.
# (T, d_out, d_in) triples.
DECODE_SHAPES: List[Tuple[int, int, int]] = [
    # Small square: where the fused kernel should win most (launch/HBM dominant).
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    (4, 4096, 4096),
    (16, 4096, 4096),
    # Large d_out: near HBM roof, smaller relative wins expected.
    (1, 14336, 4096),
    (1, 28672, 4096),
    (4, 14336, 4096),
    (16, 14336, 4096),
    (16, 28672, 4096),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--skip-combined", action="store_true",
                        help="Skip the plain+combine_transpose row (saves time).")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.cuda.set_device(0)

    print(f"Bench protocol: warmup={args.warmup}, windows={args.windows}, iters={args.iters}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    print(
        f"{'T':>3s} {'d_out':>6s} {'d_in':>6s} "
        f"{'plain_us':>10s} {'combined_us':>12s} {'fused_us':>10s} "
        f"{'vs_plain':>10s} {'vs_combined':>12s}"
    )
    print("-" * 86)

    total_plain = 0.0
    total_fused = 0.0
    wins = 0
    for (T, d_out, d_in) in DECODE_SHAPES:
        W = _build_zero_hp_pack(d_out=d_out, d_in=d_in, seed=T)
        X = torch.randn(T, d_in, device="cuda", dtype=torch.float16) * 0.3
        # Pre-quantise once: we are only benchmarking the dense+output path.
        X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)

        def plain() -> None:
            Y_low = dense_gemm_u4_s4(
                W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x
            )
            # Must return the contiguous (T, d_out) result so HBM traffic is real.
            Y_low.transpose(0, 1).contiguous()

        def combined() -> None:
            Y_low = dense_gemm_u4_s4(
                W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x
            )
            _combine_transpose(Y_low, None, d_out=d_out, T=T)

        def fused() -> None:
            dense_gemm_u4_s4_to_out(
                W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x
            )

        t_plain = _bench(plain, warmup=args.warmup, windows=args.windows, iters=args.iters)
        t_comb = (
            _bench(combined, warmup=args.warmup, windows=args.windows, iters=args.iters)
            if not args.skip_combined
            else float("nan")
        )
        t_fused = _bench(fused, warmup=args.warmup, windows=args.windows, iters=args.iters)

        total_plain += t_plain
        total_fused += t_fused
        if t_fused < t_plain:
            wins += 1

        vs_plain = t_plain / t_fused
        vs_comb = (t_comb / t_fused) if t_comb == t_comb else float("nan")
        print(
            f"{T:>3d} {d_out:>6d} {d_in:>6d} "
            f"{t_plain:>10.2f} {t_comb:>12.2f} {t_fused:>10.2f} "
            f"{vs_plain:>9.2f}x {vs_comb:>11.2f}x"
        )

    print("-" * 86)
    print(
        f"Summary: {wins}/{len(DECODE_SHAPES)} shapes improved, "
        f"total plain={total_plain:.1f}us total fused={total_fused:.1f}us  "
        f"avg speedup={total_plain / total_fused:.2f}x"
    )


if __name__ == "__main__":
    main()
