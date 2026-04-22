"""Unit tests for the fused activation quantization Triton kernel.

Requires a CUDA device.  When CUDA is unavailable the tests are skipped.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for activation kernel tests", allow_module_level=True)

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, pack_s4_le, unpack_s4_le  # noqa: E402


def _torch_reference(X: torch.Tensor, perm: torch.Tensor, bcol: int):
    """PyTorch reference implementation of the fused activation kernel."""
    X_perm = X[:, perm.to(torch.long)]
    max_abs = X_perm.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale_x = (max_abs / 7.0).to(torch.float16).to(torch.float32)
    q = torch.clamp(torch.round(X_perm.to(torch.float32) / scale_x), -8.0, 7.0).to(torch.int32)
    # Group sum
    T, D = q.shape
    n_groups = D // bcol
    q_grouped = q.reshape(T, n_groups, bcol)
    sum_X = q_grouped.sum(dim=-1).to(torch.int32)
    # Pack
    X_s4 = pack_s4_le(q.cpu()).to(X.device)
    return X_s4, scale_x.squeeze(-1).to(torch.float16), sum_X


@pytest.mark.parametrize("T,D", [(8, 128), (32, 256), (4, 512)])
def test_activation_quant_matches_reference(T, D):
    torch.manual_seed(0)
    device = "cuda"
    X = (torch.randn(T, D, device=device) * 2.0).to(torch.float16)
    perm = torch.randperm(D, device=device).to(torch.int32)

    X_s4_k, scale_x_k, sum_X_k = quantize_activation_s4(X, perm, bcol=BCOL)
    X_s4_r, scale_x_r, sum_X_r = _torch_reference(X, perm, BCOL)

    # X_s4 must be bitwise equal
    assert torch.equal(X_s4_k.cpu(), X_s4_r.cpu()), "X_s4 mismatch between kernel and reference"

    # sum_X must be bitwise equal
    assert torch.equal(sum_X_k.cpu(), sum_X_r.cpu()), "sum_X mismatch between kernel and reference"

    # scale_x relative error <= 1e-6 (fp16 precision)
    sk = scale_x_k.cpu().to(torch.float32)
    sr = scale_x_r.cpu().to(torch.float32)
    rel_err = (sk - sr).abs() / (sr.abs() + 1e-12)
    assert rel_err.max().item() <= 1e-3, f"scale_x rel error too large: {rel_err.max().item()}"


def test_activation_quant_unpacked_range():
    """After kernel output, unpacked SINT4 must be in [-8, 7]."""
    torch.manual_seed(1)
    T, D = 16, 256
    X = torch.randn(T, D, device="cuda", dtype=torch.float16) * 5.0
    perm = torch.arange(D, device="cuda", dtype=torch.int32)
    X_s4, _, _ = quantize_activation_s4(X, perm, bcol=BCOL)
    unpacked = unpack_s4_le(X_s4.cpu(), signed=True).to(torch.int32)
    assert unpacked.min() >= -8
    assert unpacked.max() <= 7
