"""Benchmark: Kernel (2) sparse GEMM at 5% HP-block sparsity vs Kernel (1)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from kernel.triton.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton.sparse_s4s4_gemm import sparse_gemm_s4_s4  # noqa: E402


def _build_pack(d_out: int, d_in: int, hp_ratio: float = 0.05):
    nrow = d_out // BROW
    ncol = d_in // BCOL
    torch.manual_seed(0)
    device = "cuda"

    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device=device)
    scale_u4 = (torch.rand(d_out, ncol, device=device) * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, ncol), device=device).to(torch.float16)

    n_hp = max(1, int(nrow * ncol * hp_ratio))
    brs = torch.randint(0, nrow, (n_hp,), device=device)
    bcs = torch.randint(0, ncol, (n_hp,), device=device)
    # Dedup
    combined = brs * ncol + bcs
    combined = torch.unique(combined)[:n_hp]
    brs = (combined // ncol).to(torch.int32)
    bcs = (combined % ncol).to(torch.int32)
    hp_indices = torch.stack([brs, bcs], dim=-1)

    Q_s8_blocks = torch.randint(-64, 64, (len(brs), BROW, BCOL), dtype=torch.int8, device=device)
    scale_s8 = (torch.rand(len(brs), BROW, device=device) * 0.005 + 0.001).to(torch.float16)
    perm = torch.arange(d_in, dtype=torch.int32, device=device)

    return pack_v9_weights({
        "Q_u4_permuted": Q_u4, "scale_u4_raw": scale_u4, "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8_blocks, "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices, "perm": perm,
    })


def _time_ms(fn, n_warmup=10, n_iter=50) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return

    shapes = [(11008, 4096), (4096, 4096), (4096, 11008)]
    T = 256

    print(f"{'d_out':>8} {'d_in':>8} {'T':>6} {'K1(ms)':>10} {'K2(ms)':>10} {'K2/K1':>10}")
    for d_out, d_in in shapes:
        W = _build_pack(d_out, d_in, hp_ratio=0.05)
        X = torch.randn(T, d_in, device="cuda", dtype=torch.float16)
        X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)

        def k1():
            dense_gemm_u4_s4(W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x)

        def k2():
            sparse_gemm_s4_s4(
                W.W_high_blocks_packed, W.hp_row_offsets, W.hp_col_indices,
                X_s4, W.scale_u4, scale_x, d_out=d_out, d_in=d_in,
            )

        t1 = _time_ms(k1)
        t2 = _time_ms(k2)
        print(f"{d_out:>8} {d_in:>8} {T:>6} {t1:>10.3f} {t2:>10.3f} {t2 / t1 * 100:>9.2f}%")


if __name__ == "__main__":
    main()
