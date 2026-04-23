"""Correctness tests for the fused dense+sparse GEMM kernel.

Verifies that ``fused_dense_sparse_gemm`` returns exactly
``Y_low + 16 * Y_high`` for every (d_out, d_in, T, hp_ratio) combination,
where Y_low and Y_high come from the existing independently-tested
dense_gemm_u4_s4 and sparse_gemm_s4_s4 kernels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton_kernel.sparse_s4s4_gemm import sparse_gemm_s4_s4  # noqa: E402
from kernel.triton_kernel.fused_dense_sparse_gemm import fused_dense_sparse_gemm  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402


def _synthesize_pack(d_out: int, d_in: int, hp_ratio: float = 0.05, seed: int = 0):
    """Same fixture as test_end2end._synthesize_pack, duplicated here to
    avoid cross-test import coupling."""
    nrow = d_out // BROW
    ncol = d_in // BCOL
    torch.manual_seed(seed)
    device = "cuda"

    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device=device)
    scale_u4 = (torch.rand(d_out, ncol, device=device) * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, ncol), device=device).to(torch.float16)

    if hp_ratio > 0.0:
        n_hp = max(1, int(nrow * ncol * hp_ratio))
        combined = torch.unique(
            torch.randint(0, nrow * ncol, (n_hp * 2,), device=device)
        )[:n_hp]
        brs = (combined // ncol).to(torch.int32)
        bcs = (combined % ncol).to(torch.int32)
        hp_indices = torch.stack([brs, bcs], dim=-1)
        Q_s8_blocks = torch.randint(
            -64, 64, (len(brs), BROW, BCOL), dtype=torch.int8, device=device
        )
        scale_s8 = (torch.rand(len(brs), BROW, device=device) * 0.005 + 0.001).to(
            torch.float16
        )
    else:
        hp_indices = torch.zeros((0, 2), dtype=torch.int32, device=device)
        Q_s8_blocks = torch.zeros((0, BROW, BCOL), dtype=torch.int8, device=device)
        scale_s8 = torch.zeros((0, BROW), dtype=torch.float16, device=device)

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


@pytest.mark.parametrize(
    "d_out,d_in,T,hp_ratio",
    [
        (128, 256, 16, 0.10),      # tiny sanity with at least 1 hp block
        (256, 512, 64, 0.05),
        (256, 256, 32, 0.10),
        (512, 1024, 128, 0.02),    # realistic-ish
    ],
)
def test_fused_matches_separate(d_out, d_in, T, hp_ratio):
    W = _synthesize_pack(d_out, d_in, hp_ratio=hp_ratio)
    # Random activations on device.
    torch.manual_seed(1234)
    X_fp16 = (torch.randn(T, d_in, dtype=torch.float32, device="cuda") * 0.5).to(torch.float16)
    X_s4, scale_x, sum_X = quantize_activation_s4(X_fp16, W.perm, bcol=BCOL)

    # Reference: run dense and sparse separately, combine in fp32.
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
    Y_ref = Y_low.to(torch.float32) + 16.0 * Y_high.to(torch.float32)

    # Fused kernel.
    Y_fused = fused_dense_sparse_gemm(
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
    assert Y_fused.shape == (d_out, T)
    assert Y_fused.dtype == torch.float16

    abs_err = (Y_fused.to(torch.float32) - Y_ref).abs()
    ref_norm = Y_ref.abs().max().clamp(min=1e-4)
    rel_err = (abs_err / ref_norm).max().item()
    # 5e-3 is tight enough to catch any algorithmic bug while allowing
    # for the fused kernel carrying the dense+sparse sum in fp32 (fewer
    # rounding errors than the reference which casts Y_low/Y_high to fp16
    # before summing).
    assert rel_err < 5e-3, (
        f"max rel err {rel_err:.3e} for d_out={d_out} d_in={d_in} T={T} hp={hp_ratio}"
    )


@pytest.mark.parametrize("T", [1, 16, 128])
def test_fused_zero_hp_equals_dense(T):
    """When hp_ratio=0, the sparse branch must iterate zero blocks and
    the output must exactly equal the dense-only path."""
    d_out, d_in = 256, 256
    W = _synthesize_pack(d_out, d_in, hp_ratio=0.0)
    torch.manual_seed(7)
    X_fp16 = (torch.randn(T, d_in, dtype=torch.float32, device="cuda") * 0.3).to(torch.float16)
    X_s4, scale_x, sum_X = quantize_activation_s4(X_fp16, W.perm, bcol=BCOL)

    Y_dense = dense_gemm_u4_s4(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
    )
    Y_fused = fused_dense_sparse_gemm(
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
    # Must be exactly bit-equal since sparse contributes 0.
    torch.testing.assert_close(Y_fused, Y_dense, rtol=0, atol=0)
