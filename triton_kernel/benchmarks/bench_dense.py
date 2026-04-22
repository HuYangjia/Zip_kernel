"""Benchmark: Kernel (1) dense GEMM vs cuBLAS FP16 baseline.

Reports latency on a Qwen3-typical layer shape (11008, 4096) x batch*seq=256.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton_kernel.benchmarks._bench_util import time_ms as _time_ms  # noqa: E402


def _build_pack(d_out: int, d_in: int):
    n_groups = d_in // BCOL
    torch.manual_seed(0)
    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device="cuda")
    scale_u4 = (torch.rand(d_out, n_groups, device="cuda") * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, n_groups), device="cuda").to(torch.float16)
    perm = torch.arange(d_in, dtype=torch.int32, device="cuda")
    return pack_v9_weights({
        "Q_u4_permuted": Q_u4,
        "scale_u4_raw": scale_u4, "zero_u4_raw": zero_u4,
        "Q_s8_blocks": torch.zeros(0, BROW, BCOL, dtype=torch.int8, device="cuda"),
        "scale_s8_per_block": torch.zeros(0, BROW, dtype=torch.float16, device="cuda"),
        "hp_block_indices": torch.zeros((0, 2), dtype=torch.int32, device="cuda"),
        "perm": perm,
    })


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return

    shapes = [(11008, 4096), (4096, 4096), (4096, 11008)]
    T = 256

    print(f"{'d_out':>8} {'d_in':>8} {'T':>6} {'V9(ms)':>10} {'FP16(ms)':>10} {'speedup':>10}")
    for d_out, d_in in shapes:
        W = _build_pack(d_out, d_in)
        X = torch.randn(T, d_in, device="cuda", dtype=torch.float16)

        # pre-quantize activation to isolate GEMM latency
        X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)

        def v9_fn():
            dense_gemm_u4_s4(W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x)

        W_fp = torch.randn(d_out, d_in, device="cuda", dtype=torch.float16)

        def fp16_fn():
            torch.matmul(X, W_fp.t())

        t_v9 = _time_ms(v9_fn)
        t_fp16 = _time_ms(fp16_fn)
        print(f"{d_out:>8} {d_in:>8} {T:>6} {t_v9:>10.3f} {t_fp16:>10.3f} {t_fp16 / t_v9:>10.2f}x")


if __name__ == "__main__":
    main()
