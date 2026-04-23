"""Micro-benchmark: Split-K dense kernel vs fused-to-out vs plain vs FP16 cuBLAS (P4 Step 4.2).

Four configurations measured at identical shapes:

  * plain_dense   : dense_gemm_u4_s4(...).transpose(0,1).contiguous()
  * fused_dense   : dense_gemm_u4_s4_to_out(...)                 [P3 Step 1 baseline]
  * splitk_auto   : dense_gemm_u4_s4_splitk(...) with wrapper policy [P4 Step 4.2 target]
  * fp16_cublas   : torch.matmul(X_fp16, W_fp16.T)                [FP16 roof ref]

The activation quantisation pass ``quantize_activation_s4`` is NOT part of
any row (consistent with bench_dense_to_out), so the comparison isolates
the weight-load + GEMM + output path.  ``scale_x``, ``sum_X`` are
pre-computed once per shape.

Timer: min-of-means per shared CUDA microbench protocol.
    50 warmup + 3 windows x 200 iters

Reports latency in us and speedups vs plain and vs FP16 cuBLAS.

Run:
    python -m kernel.triton_kernel.benchmarks.bench_dense_splitk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_gemm_splitk import (  # noqa: E402
    _choose_split_k,
    dense_gemm_u4_s4_splitk,
)
from kernel.triton_kernel.dense_gemm_to_out import dense_gemm_u4_s4_to_out  # noqa: E402
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402


# ---------------------------------------------------------------------------
# V9 zero-hp pack (identical to bench_dense_to_out)
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
# Microbench (min-of-means, CUDA events)
# ---------------------------------------------------------------------------

def _bench(fn: Callable[[], None], warmup: int = 50, windows: int = 3, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means_us: List[float] = []
    for _w in range(windows):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        means_us.append(start.elapsed_time(end) * 1000.0 / iters)
    return min(means_us)


# ---------------------------------------------------------------------------
# Shapes (same list as bench_dense_to_out so we can cross-reference)
# ---------------------------------------------------------------------------

DECODE_SHAPES: List[Tuple[int, int, int]] = [
    (1, 4096, 4096),
    (1, 4096, 11008),
    (1, 11008, 4096),
    (4, 4096, 4096),
    (16, 4096, 4096),
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
    parser.add_argument("--skip-fp16", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.cuda.set_device(0)

    print(f"Bench protocol: warmup={args.warmup}, windows={args.windows}, iters={args.iters}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    hdr = (
        f"{'T':>3s} {'d_out':>6s} {'d_in':>6s} {'sk':>3s} "
        f"{'plain_us':>10s} {'fused_us':>10s} {'splitk_us':>10s} {'fp16_us':>10s} "
        f"{'vs_plain':>9s} {'vs_fused':>9s} {'vs_fp16':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))

    total_plain = 0.0
    total_fused = 0.0
    total_splitk = 0.0
    total_fp16 = 0.0
    wins_vs_fused = 0
    rows: List[str] = []

    for (T, d_out, d_in) in DECODE_SHAPES:
        W = _build_zero_hp_pack(d_out=d_out, d_in=d_in, seed=T)
        X = torch.randn(T, d_in, device="cuda", dtype=torch.float16) * 0.3
        X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)

        # FP16 reference weight for cuBLAS comparison.
        # Build a fake FP16 weight of shape (d_out, d_in) -- value doesn't matter
        # for timing as long as shape is right; HBM traffic is what we care about.
        W_fp16 = torch.randn(d_out, d_in, device="cuda", dtype=torch.float16) * 0.01

        sk = _choose_split_k(d_out, T, d_in)

        def plain() -> None:
            Y_low = dense_gemm_u4_s4(
                W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
            )
            Y_low.transpose(0, 1).contiguous()

        def fused() -> None:
            dense_gemm_u4_s4_to_out(
                W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
            )

        def splitk() -> None:
            dense_gemm_u4_s4_splitk(
                W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
                split_k=sk,
            )

        def fp16() -> None:
            # (T, d_in) @ (d_in, d_out) -> (T, d_out) via cuBLAS.
            torch.matmul(X, W_fp16.t())

        t_plain = _bench(plain, warmup=args.warmup, windows=args.windows, iters=args.iters)
        t_fused = _bench(fused, warmup=args.warmup, windows=args.windows, iters=args.iters)
        t_splitk = _bench(splitk, warmup=args.warmup, windows=args.windows, iters=args.iters)
        t_fp16 = (
            _bench(fp16, warmup=args.warmup, windows=args.windows, iters=args.iters)
            if not args.skip_fp16
            else float("nan")
        )

        total_plain += t_plain
        total_fused += t_fused
        total_splitk += t_splitk
        total_fp16 += t_fp16 if t_fp16 == t_fp16 else 0.0
        if t_splitk < t_fused:
            wins_vs_fused += 1

        row = (
            f"{T:>3d} {d_out:>6d} {d_in:>6d} {sk:>3d} "
            f"{t_plain:>10.2f} {t_fused:>10.2f} {t_splitk:>10.2f} {t_fp16:>10.2f} "
            f"{t_plain / t_splitk:>8.2f}x {t_fused / t_splitk:>8.2f}x "
            f"{t_fp16 / t_splitk:>8.2f}x"
        )
        print(row)
        rows.append(row)

    print("-" * len(hdr))
    print(
        f"Summary: splitk improved vs fused on {wins_vs_fused}/{len(DECODE_SHAPES)} shapes. "
        f"Totals (us): plain={total_plain:.1f}  fused={total_fused:.1f}  "
        f"splitk={total_splitk:.1f}  fp16={total_fp16:.1f}. "
        f"splitk vs fused avg={total_fused / total_splitk:.2f}x, "
        f"splitk vs fp16 avg={total_fp16 / total_splitk:.2f}x "
        f"(>1 means splitk is faster than that baseline)."
    )


if __name__ == "__main__":
    main()
