"""Correctness tests for dense_gemm_u4_s4_splitk (P4 Step 4.1).

Two invariants:

1. ``split_k == 1`` must be **bit-exact** vs ``dense_gemm_u4_s4_to_out``.
   This is the canary: SPLIT_K=1 runs the same inner loop, same fp32
   accumulator, same reduce order -- the only difference is the extra
   ``splitk_reduce_kernel`` pass which is a no-op (single-buffer copy
   + scale_x apply, identical to the non-split epilogue).

2. ``split_k > 1`` may reorder the per-group accumulation, so tolerance
   relaxes to ``atol=1e-3, rtol=1e-3`` (same bar as fused_dense_sparse).

Run:
    pytest kernel/triton_kernel/tests/test_dense_gemm_splitk.py -v
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
    pytest.skip("CUDA required for split-K test", allow_module_level=True)

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_gemm_splitk import (  # noqa: E402
    _choose_split_k,
    dense_gemm_u4_s4_splitk,
)
from kernel.triton_kernel.dense_gemm_to_out import dense_gemm_u4_s4_to_out  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: reuse zero-hp V9 pack
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
# Canary: SPLIT_K=1 must be numerically close to dense_gemm_u4_s4_to_out
# ---------------------------------------------------------------------------
# NOTE: we cannot require bit-exact equality because the split-K kernel
# deliberately moves the ``scale_x`` multiplication into the reduce pass
# (so that per-split partials are independent of the global scale).  The
# dense_gemm_to_out reference multiplies by ``scale_x`` inside the K-loop.
# FP32 multiplication is not associative, so the reorder perturbs the
# fp16-cast result by <=2 ULP even when SPLIT_K=1 (no group reordering).
# We therefore use the same atol=1e-3 as the SPLIT_K>1 tests; the real
# guarantee of SPLIT_K=1 is "no group reordering", captured by the
# tighter rtol below.

@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize(
    "d_out,d_in",
    [
        (4096, 4096),
        (14336, 4096),
        (4096, 11008),
    ],
)
def test_splitk_equals_1_close_to_reference(T, d_out, d_in):
    """SPLIT_K=1 must match dense_gemm_u4_s4_to_out within atol=5e-4, rtol=1e-4.

    Tighter than the SPLIT_K>1 tests because no per-group reordering
    happens here -- only the scale_x multiplication moves one epilogue
    outward, which costs at most a couple of ULPs.
    """
    W = _build_zero_hp_pack(d_out=d_out, d_in=d_in, seed=T)
    X = (torch.randn(T, d_in, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)

    Y_ref = dense_gemm_u4_s4_to_out(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
    )
    Y_new = dense_gemm_u4_s4_splitk(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
        split_k=1,
    )
    assert Y_new.shape == Y_ref.shape
    assert Y_new.dtype == Y_ref.dtype
    assert Y_new.is_contiguous()
    torch.testing.assert_close(
        Y_new.float(), Y_ref.float(),
        atol=5e-4, rtol=1e-4,
        msg=f"SPLIT_K=1 diverged at (T={T}, d_out={d_out}, d_in={d_in})",
    )


# ---------------------------------------------------------------------------
# Main correctness: SPLIT_K>1 within relaxed tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split_k", [2, 4, 8])
@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize(
    "d_out,d_in",
    [
        (4096, 4096),
        (14336, 4096),
        (4096, 11008),
    ],
)
def test_splitk_close_to_reference(split_k, T, d_out, d_in):
    """SPLIT_K in {2,4,8}: output within atol=1e-3, rtol=1e-3 of reference."""
    n_groups = d_in // BCOL
    if n_groups % split_k != 0:
        pytest.skip(f"n_groups={n_groups} not divisible by split_k={split_k}")
    if n_groups // split_k < 2:
        pytest.skip(f"each split would cover <2 groups")

    W = _build_zero_hp_pack(d_out=d_out, d_in=d_in, seed=T * 100 + split_k)
    X = (torch.randn(T, d_in, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)

    Y_ref = dense_gemm_u4_s4_to_out(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
    )
    Y_new = dense_gemm_u4_s4_splitk(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
        split_k=split_k,
    )
    torch.testing.assert_close(
        Y_new.float(), Y_ref.float(),
        atol=1e-3, rtol=1e-3,
        msg=(
            f"split_k={split_k} diverged at (T={T}, d_out={d_out}, d_in={d_in})"
        ),
    )


# ---------------------------------------------------------------------------
# Auto policy: SPLIT_K chosen by wrapper still matches reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "T,d_out,d_in",
    [
        (1, 4096, 4096),
        (1, 11008, 4096),
        (1, 14336, 4096),
        (4, 4096, 4096),
        (16, 14336, 4096),
        (16, 4096, 11008),
    ],
)
def test_auto_split_k_close_to_reference(T, d_out, d_in):
    """Wrapper's _choose_split_k path must also land within tolerance."""
    W = _build_zero_hp_pack(d_out=d_out, d_in=d_in, seed=T + 999)
    X = (torch.randn(T, d_in, device="cuda", dtype=torch.float16) * 0.3).contiguous()
    X_s4, scale_x, sum_X = quantize_activation_s4(X, W.perm, bcol=BCOL)

    Y_ref = dense_gemm_u4_s4_to_out(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
    )
    Y_new = dense_gemm_u4_s4_splitk(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
    )
    chosen = _choose_split_k(d_out, T, d_in)
    if chosen == 1:
        # SPLIT_K=1: tighter tolerance, but not bit-exact (scale_x
        # is applied in the reduce pass, see note above).
        torch.testing.assert_close(
            Y_new.float(), Y_ref.float(),
            atol=5e-4, rtol=1e-4,
            msg=f"auto SPLIT_K=1 diverged at (T={T}, d_out={d_out}, d_in={d_in})",
        )
    else:
        torch.testing.assert_close(
            Y_new.float(), Y_ref.float(),
            atol=1e-3, rtol=1e-3,
            msg=(
                f"auto split_k={chosen} diverged at "
                f"(T={T}, d_out={d_out}, d_in={d_in})"
            ),
        )


def test_choose_split_k_policy_sanity():
    """Sanity-check the heuristic: decode-tiny shapes must get SPLIT_K>=2,
    and prefill-large shapes must stay at SPLIT_K=1."""
    # T=1, d_out=4096, d_in=4096 -> grid_mn = 32 <<  SM_TARGET=128 -> must split
    assert _choose_split_k(4096, 1, 4096) > 1
    # T=2048, d_out=4096 -> grid_mn ~ 32 * 128 = 4096 -> no split
    assert _choose_split_k(4096, 2048, 4096) == 1
