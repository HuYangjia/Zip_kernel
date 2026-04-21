"""Correctness test for Kernel (2) block-sparse SINT4 x SINT4 GEMM.

Uses 100% high-precision blocks (all SINT8) so Kernel (1) + Kernel (2) must
reproduce the SINT8 fakequant reference.
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
    pytest.skip("CUDA required for sparse GEMM test", allow_module_level=True)

from kernel.triton.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton.sparse_s4s4_gemm import sparse_gemm_s4_s4  # noqa: E402
from kernel.triton.v9_linear import reconstruct_w_fakequant_fp16  # noqa: E402


def test_sparse_full_hp_matches_fakequant():
    """100% high-precision blocks: Kernel(1) + Kernel(2) ==~ fakequant."""
    d_out, d_in, T = 256, 256, 16
    nrow = d_out // BROW
    ncol = d_in // BCOL
    n_hp = nrow * ncol

    torch.manual_seed(0)
    device = "cuda"
    # Build SINT8 weights for every block.
    W_s8 = torch.randint(-64, 64, (d_out, d_in), dtype=torch.int8, device=device)

    # Per-block scales (per-row within the block).
    Q_s8_blocks = torch.empty((n_hp, BROW, BCOL), dtype=torch.int8, device=device)
    scale_s8 = torch.empty((n_hp, BROW), dtype=torch.float16, device=device)
    hp_indices = torch.empty((n_hp, 2), dtype=torch.int32, device=device)
    idx = 0
    for br in range(nrow):
        for bc in range(ncol):
            Q_s8_blocks[idx] = W_s8[br * BROW:(br + 1) * BROW, bc * BCOL:(bc + 1) * BCOL]
            scale_s8[idx] = (torch.rand(BROW, device=device) * 0.005 + 0.001).to(torch.float16)
            hp_indices[idx, 0] = br
            hp_indices[idx, 1] = bc
            idx += 1

    # UINT4 baseline: doesn't matter since SINT8 blocks will override scale/zero.
    # Still must have a well-defined value in W_low: use q_s8 & 0x0F directly.
    Q_u4_baseline = (W_s8.to(torch.int32) & 0x0F).to(torch.int8)
    scale_u4 = torch.ones(d_out, ncol, dtype=torch.float16, device=device) * 0.01
    zero_u4 = torch.zeros(d_out, ncol, dtype=torch.float16, device=device)

    perm = torch.arange(d_in, dtype=torch.int32, device=device)

    W = pack_v9_weights({
        "Q_u4_permuted": Q_u4_baseline,
        "scale_u4_raw": scale_u4, "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8_blocks, "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices, "perm": perm,
    })

    # Activation
    X = torch.randn(T, d_in, device=device, dtype=torch.float16)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)

    # Kernel(1) + Kernel(2)
    Y_low = dense_gemm_u4_s4(W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x)
    Y_high = sparse_gemm_s4_s4(
        W.W_high_blocks_packed, W.hp_row_offsets, W.hp_col_indices,
        X_s4, W.scale_u4, scale_x, d_out=d_out, d_in=d_in,
    )
    Y_kernel = (Y_low + 16.0 * Y_high).transpose(0, 1).contiguous()

    # Reference: reconstruct fakequant weight, fakequant activation, matmul
    W_fp = reconstruct_w_fakequant_fp16(W)
    max_abs = X.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    sref = (max_abs / 7.0).to(torch.float16).to(torch.float32)
    q = torch.clamp(torch.round(X.to(torch.float32) / sref), -8.0, 7.0)
    X_dq = (q * sref).to(torch.float16)
    Y_ref = X_dq @ W_fp.t()

    num = (Y_kernel.to(torch.float32) - Y_ref.to(torch.float32)).abs().max()
    den = Y_ref.to(torch.float32).abs().max().clamp(min=1e-6)
    rel_err = (num / den).item()
    assert rel_err <= 5e-3, f"sparse + dense rel_err={rel_err:.4e}"
