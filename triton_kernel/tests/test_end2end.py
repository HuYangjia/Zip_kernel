"""End-to-end correctness test for `v9_linear_forward`.

Since the `gptq_submatrix_mixed.py` True-Quant 7-tuple export is owned by a
separate feature (`v9_quantization_speedup`), this test synthesizes a realistic
5% high-precision block scenario and checks that `v9_linear_forward`
matches `v9_linear_fakequant` within the required relative error.
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
    pytest.skip("CUDA required for end-to-end test", allow_module_level=True)

from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton_kernel.v9_linear import v9_linear_fakequant, v9_linear_forward  # noqa: E402


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
    perm = torch.randperm(d_in, device=device).to(torch.int32)

    return pack_v9_weights({
        "Q_u4_permuted": Q_u4,
        "scale_u4_raw": scale_u4, "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8_blocks, "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices, "perm": perm,
    })


@pytest.mark.parametrize("batch,seq,d_in,d_out", [
    (1, 16, 256, 256),
    (2, 8, 512, 256),
])
def test_end2end_matches_fakequant(batch, seq, d_in, d_out):
    W = _synthesize_pack(d_out, d_in, hp_ratio=0.05)
    X = torch.randn(batch, seq, d_in, device="cuda", dtype=torch.float16)

    Y_kernel = v9_linear_forward(X, W)
    Y_ref = v9_linear_fakequant(X, W)

    assert Y_kernel.shape == Y_ref.shape == (batch, seq, d_out)

    diff = (Y_kernel.to(torch.float32) - Y_ref.to(torch.float32)).abs()
    max_abs_err = diff.max().item()
    max_ref = Y_ref.to(torch.float32).abs().max().clamp(min=1e-6).item()
    rel_err = max_abs_err / max_ref

    if rel_err > 1e-2:
        # Provide diagnostic info
        flat_idx = diff.flatten().argmax().item()
        shape = diff.shape
        idx = []
        stride = 1
        rem = flat_idx
        for dim in shape[::-1]:
            idx.append(rem % dim)
            rem //= dim
        idx = idx[::-1]
        print(f"[end2end] max abs err={max_abs_err:.4e} at idx={idx}")
        print(f"[end2end] Y_kernel value={Y_kernel[tuple(idx)].item():.4f}")
        print(f"[end2end] Y_ref    value={Y_ref[tuple(idx)].item():.4f}")

    assert rel_err <= 1e-2, (
        f"end-to-end rel_err={rel_err:.4e} (max_abs={max_abs_err:.4e}, max_ref={max_ref:.4e})"
    )
