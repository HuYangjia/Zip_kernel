"""Correctness test for Kernel (1) dense UINT4 x SINT4 GEMM.

Uses a degenerate synthetic scenario with 0 high-precision blocks so that
Kernel (1) alone should reproduce the fakequant reference.
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
    pytest.skip("CUDA required for dense GEMM test", allow_module_level=True)

from kernel.triton.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton.v9_linear import reconstruct_w_fakequant_fp16  # noqa: E402


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


def test_dense_kernel_matches_fakequant_zero_hp():
    d_out, d_in, T = 256, 256, 32
    W = _build_zero_hp_pack(d_out, d_in)
    assert W.n_hp_blocks == 0

    # Random fp16 activations
    torch.manual_seed(42)
    X = torch.randn(T, d_in, device="cuda", dtype=torch.float16)

    # Kernel path
    X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)
    Y_low = dense_gemm_u4_s4(W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x)
    Y_kernel = Y_low.transpose(0, 1).contiguous()   # (T, d_out)

    # Reference: fakequant path (reconstruct fp16 weight, fakequant activation, matmul)
    W_fp = reconstruct_w_fakequant_fp16(W)
    max_abs = X.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale_ref = (max_abs / 7.0).to(torch.float16).to(torch.float32)
    q = torch.clamp(torch.round(X.to(torch.float32) / scale_ref), -8.0, 7.0)
    X_dequant = (q * scale_ref).to(torch.float16)
    Y_ref = X_dequant @ W_fp.t()

    rel_err = (Y_kernel.to(torch.float32) - Y_ref.to(torch.float32)).abs().max() / Y_ref.to(torch.float32).abs().max().clamp(min=1e-6)
    assert rel_err.item() <= 5e-3, f"dense kernel rel_err={rel_err.item():.4e}"
