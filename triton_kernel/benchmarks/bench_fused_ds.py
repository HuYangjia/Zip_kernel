"""Microbenchmark: fused_dense_sparse_gemm vs (dense + sparse + fp16-sum).

Compares the fused kernel against running dense_gemm_u4_s4 and
sparse_gemm_s4_s4 back to back, for realistic prefill shapes with
non-zero hp_ratio.

Run from repo root:
    PYTHONPATH=. python -m kernel.triton_kernel.benchmarks.bench_fused_ds
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton_kernel.sparse_s4s4_gemm import sparse_gemm_s4_s4  # noqa: E402
from kernel.triton_kernel.fused_dense_sparse_gemm import fused_dense_sparse_gemm  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton_kernel.benchmarks._bench_util import time_ms  # noqa: E402


def _synthesize_pack(d_out: int, d_in: int, hp_ratio: float = 0.05, seed: int = 0):
    nrow = d_out // BROW
    ncol = d_in // BCOL
    torch.manual_seed(seed)
    device = "cuda"

    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device=device)
    scale_u4 = (torch.rand(d_out, ncol, device=device) * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, ncol), device=device).to(torch.float16)

    n_hp = max(1, int(nrow * ncol * hp_ratio))
    combined = torch.unique(torch.randint(0, nrow * ncol, (n_hp * 2,), device=device))[:n_hp]
    brs = (combined // ncol).to(torch.int32)
    bcs = (combined % ncol).to(torch.int32)
    hp_indices = torch.stack([brs, bcs], dim=-1)
    Q_s8_blocks = torch.randint(-64, 64, (len(brs), BROW, BCOL), dtype=torch.int8, device=device)
    scale_s8 = (torch.rand(len(brs), BROW, device=device) * 0.005 + 0.001).to(torch.float16)
    perm = torch.arange(d_in, dtype=torch.int32, device=device)

    return pack_v9_weights({
        "Q_u4_permuted": Q_u4,
        "scale_u4_raw": scale_u4,
        "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8_blocks,
        "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices,
        "perm": perm,
    })


SHAPES = [
    # (d_out, d_in, T, hp_ratio)
    # Prefill-heavy shapes where hp>0, where fusion saves the most.
    (4096, 4096, 2048, 0.05),
    (4096, 4096, 8192, 0.05),
    (11008, 4096, 2048, 0.05),
    (11008, 4096, 8192, 0.05),
    (14336, 4096, 2048, 0.05),
    (14336, 4096, 8192, 0.05),
    (28672, 4096, 2048, 0.05),
    (28672, 4096, 8192, 0.05),
    # Mid batch
    (4096, 4096, 512, 0.05),
    (11008, 4096, 512, 0.05),
    # Lighter hp_ratio
    (11008, 4096, 2048, 0.02),
    (14336, 4096, 8192, 0.02),
    # Higher hp_ratio
    (11008, 4096, 2048, 0.10),
    (14336, 4096, 8192, 0.10),
]


def run_one(d_out, d_in, T, hp_ratio):
    W = _synthesize_pack(d_out, d_in, hp_ratio=hp_ratio)
    torch.manual_seed(42)
    X_fp16 = (torch.randn(T, d_in, dtype=torch.float32, device="cuda") * 0.5).to(torch.float16)
    X_s4, scale_x, sum_X = quantize_activation_s4(X_fp16, W.perm, bcol=BCOL)

    # --- baseline: dense + sparse back to back ---
    def run_baseline():
        Y_low = dense_gemm_u4_s4(
            W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
        )
        Y_high = sparse_gemm_s4_s4(
            W.W_high_blocks_packed,
            W.hp_row_offsets,
            W.hp_col_indices,
            X_s4,
            W.scale_u4, scale_x,
            d_out=d_out, d_in=d_in,
        )
        return Y_low, Y_high

    # --- fused ---
    def run_fused():
        return fused_dense_sparse_gemm(
            W.W_low_packed,
            W.W_high_blocks_packed,
            W.hp_row_offsets,
            W.hp_col_indices,
            X_s4,
            W.scale_u4,
            W.zero_u4,
            sum_X,
            scale_x,
            d_out=d_out,
            d_in=d_in,
        )

    # Warm up & time
    base_ms = time_ms(run_baseline) * 1000.0   # us
    fused_ms = time_ms(run_fused) * 1000.0     # us
    delta_pct = 100.0 * (fused_ms - base_ms) / base_ms if base_ms > 0 else 0.0
    return base_ms, fused_ms, delta_pct


def main():
    print(f"{'d_out':>6} {'d_in':>6} {'T':>5} {'hp':>5} | {'base_us':>10} {'fused_us':>10} {'Δ%':>8}")
    print("-" * 70)
    results = []
    for shape in SHAPES:
        base_us, fused_us, d_pct = run_one(*shape)
        results.append((shape, base_us, fused_us, d_pct))
        print(f"{shape[0]:>6} {shape[1]:>6} {shape[2]:>5} {shape[3]:>5.2f} |"
              f" {base_us:>10.2f} {fused_us:>10.2f} {d_pct:>+7.1f}%")

    # summary
    improved = sum(1 for _, _, _, d in results if d < -2.0)
    regressed = sum(1 for _, _, _, d in results if d > +2.0)
    total = len(results)
    avg = sum(d for _, _, _, d in results) / total
    print("-" * 70)
    print(f"improved: {improved}/{total}   regressed: {regressed}/{total}   avg Δ: {avg:+.1f}%")


if __name__ == "__main__":
    main()
