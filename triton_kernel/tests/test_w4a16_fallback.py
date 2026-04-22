"""Correctness test for the W4A16 fallback path in _v9_forward_prefill.

The prefill forward switches between two dense paths:
  - int4 online GEMM (default)
  - W4A16 fallback: dequant W to FP16 first, then cuBLAS FP16 GEMM

Both must produce numerically equivalent outputs for the same weight.
The fallback only activates when hp_ratio == 0 and T crosses a size
threshold; we construct tests that intentionally land in the fallback
and compare against the explicit int4 entry as ground truth.
"""

import pytest
import torch

from kernel.triton_kernel.v9_linear import (
    v9_linear_forward,
    v9_linear_forward_decode,
    v9_linear_forward_prefill,
    reconstruct_w_fakequant_fp16,
)
from kernel.triton_kernel.tests.test_end2end import _synthesize_pack


def _build_hp_zero_pack(d_out, d_in, seed=0):
    """Synthesize a V9 pack with hp_ratio=0 (no sparse blocks at all)."""
    # Re-use test_end2end helper with hp_ratio=0.0; it already handles the
    # degenerate branch (empty hp_indices etc.)
    return _synthesize_pack(d_out, d_in, hp_ratio=0.0, seed=seed)


@pytest.mark.parametrize(
    "bs,d_in,d_out",
    [
        # Right at the T >= 512 / small-shape branch
        (512, 4096, 4096),
        # Inside the T >= 1024 universal-win branch
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        # Non-square shape
        (1024, 4096, 11008),
    ],
)
def test_w4a16_matches_int4_prefill(bs, d_in, d_out):
    """For hp_ratio=0 the W4A16 fallback must match the int4 path within
    the FP16 numerical tolerance.

    The tolerance here (2e-2) is slightly looser than the main end2end
    tolerance (1e-2) because the W4A16 path accumulates in FP16 inside
    cuBLAS while the int4 path accumulates in FP32 inside our Triton
    kernel; the FP16-accumulator extra error in a d_in=4096 dot product
    is on the order of a few tenths of a percent, which is exactly what
    we observe empirically.

    We compare against ``v9_linear_fakequant`` (pure pytorch reference that
    handles the activation-order permutation internally) rather than
    calling ``torch.nn.functional.linear(X, reconstruct_w_fakequant_fp16)``
    directly -- the latter forgets the perm and gives nonsense."""
    torch.manual_seed(0)
    W = _build_hp_zero_pack(d_out, d_in, seed=0)
    X = torch.randn(bs, d_in, device="cuda", dtype=torch.float16)

    from kernel.triton_kernel.v9_linear import v9_linear_fakequant
    Y_ref = v9_linear_fakequant(X, W)

    # System under test: the dispatcher-picked prefill path (will use
    # W4A16 fallback for these shapes).
    Y_kernel = v9_linear_forward(X, W)

    diff = (Y_kernel.to(torch.float32) - Y_ref.to(torch.float32)).abs()
    max_abs = diff.max().item()
    max_ref = Y_ref.to(torch.float32).abs().max().clamp(min=1e-6).item()
    rel = max_abs / max_ref
    assert rel <= 2e-2, (
        f"W4A16 vs fakequant rel_err={rel:.4e} "
        f"(max_abs={max_abs:.4e}, max_ref={max_ref:.4e}) "
        f"shape=({d_out},{d_in}) bs={bs}"
    )


def test_w4a16_fallback_disabled_when_hp_positive():
    """With hp_ratio>0 we must NOT use the W4A16 fallback (it would drop
    the sparse contribution). We verify by comparing against the
    decode entry (which never uses the fallback)."""
    torch.manual_seed(0)
    d_in, d_out, bs = 4096, 4096, 2048
    W = _synthesize_pack(d_out, d_in, hp_ratio=0.1, seed=0)
    X = torch.randn(bs, d_in, device="cuda", dtype=torch.float16)

    # Prefill path (should take the int4 branch since hp>0)
    Y_prefill = v9_linear_forward_prefill(X, W)
    # Decode path (always int4)
    Y_decode = v9_linear_forward_decode(X, W)
    # They must be bit-identical -- no fp16 cuBLAS call happens, so both
    # go through the int4 pipeline.
    assert torch.equal(Y_prefill, Y_decode), (
        "Prefill path unexpectedly diverged from decode path when hp>0; "
        "this usually means the W4A16 fallback was taken and dropped the "
        "sparse contribution."
    )
