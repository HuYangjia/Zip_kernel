"""Correctness tests for dense_gemm_u4_s4_to_out (P3 Step 1).

Checks that fusing dense+transpose into a single kernel matches the
existing ``dense_gemm_u4_s4 -> .transpose(0,1).contiguous()`` path
bit-for-bit on hp=0 weights across a decode-focused shape sweep.

Run:
    pytest kernel/triton_kernel/tests/test_dense_gemm_to_out.py -v
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
    pytest.skip("CUDA required for dense_to_out test", allow_module_level=True)

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_gemm_to_out import dense_gemm_u4_s4_to_out  # noqa: E402
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: zero-hp V9 pack, identical to the helper in test_dense.py
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
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 2, 4, 8, 16])
@pytest.mark.parametrize(
    "d_out,d_in",
    [
        (4096, 4096),
        (14336, 4096),
        (28672, 4096),
        (4096, 11008),
        (11008, 4096),
    ],
)
def test_bitexact_vs_dense_plus_transpose(T, d_out, d_in):
    """dense_gemm_u4_s4_to_out must produce the same FP16 bits as
    ``dense_gemm_u4_s4(...).transpose(0,1).contiguous()`` on the hp=0 path.
    """
    W = _build_zero_hp_pack(d_out=d_out, d_in=d_in, seed=T)
    assert W.n_hp_blocks == 0
    X = (torch.randn(T, d_in, device="cuda", dtype=torch.float16) * 0.3).contiguous()

    X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)
    Y_low = dense_gemm_u4_s4(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x
    )
    Y_ref = Y_low.transpose(0, 1).contiguous()   # (T, d_out), bit-exact

    # Path B: the fused kernel.  Feed *identical* quant outputs so the MMA
    # input is bit-identical; only the store layout differs.
    Y_new = dense_gemm_u4_s4_to_out(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
    )
    assert Y_new.shape == (T, d_out)
    assert Y_new.dtype == torch.float16
    assert Y_new.is_contiguous()
    assert Y_new.stride() == (d_out, 1)

    if not torch.equal(Y_ref, Y_new):
        # Diagnose: report max absolute delta so the CI log is informative.
        diff = (Y_ref.float() - Y_new.float()).abs()
        n_bad = int((diff > 0).sum().item())
        pytest.fail(
            f"dense_to_out mismatched at (T={T}, d_out={d_out}, d_in={d_in}); "
            f"max|delta|={diff.max().item():.3e}, "
            f"n_bad={n_bad} / {Y_ref.numel()}"
        )


@pytest.mark.parametrize(
    "T,d_out,d_in",
    [
        (1, 4096, 4096),
        (16, 14336, 4096),
        (4, 28672, 4096),
    ],
)
def test_nonlocal_T_still_correct(T, d_out, d_in):
    """Light smoke test on a few (T, d_out, d_in) triples outside the
    dominant decode grid, in case autotune picks a different config.
    """
    W = _build_zero_hp_pack(d_out=d_out, d_in=d_in, seed=T + 1000)
    X = torch.randn(T, d_in, device="cuda", dtype=torch.float16) * 0.3
    X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)
    Y_ref = dense_gemm_u4_s4(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x
    ).transpose(0, 1).contiguous()
    Y_new = dense_gemm_u4_s4_to_out(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x
    )
    assert torch.equal(Y_ref, Y_new), \
        f"bit-exact fail on (T={T}, d_out={d_out}, d_in={d_in})"
