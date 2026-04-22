"""Smoke-test the prefill/decode dispatcher split.

Verifies:
1. The dispatcher ``v9_linear_forward`` and the two explicit entries
   ``v9_linear_forward_decode`` / ``v9_linear_forward_prefill`` all return
   bit-identical results (they share the same underlying pipeline today).
2. Both decode-regime shapes (T <= DECODE_T_THRESHOLD) and
   prefill-regime shapes (T > DECODE_T_THRESHOLD) round-trip correctly.
3. 3D input reshape still works under the dispatcher.
"""

import pytest
import torch

from kernel.triton_kernel.v9_linear import (
    DECODE_T_THRESHOLD,
    v9_linear_forward,
    v9_linear_forward_decode,
    v9_linear_forward_prefill,
)
from kernel.triton_kernel.tests.test_end2end import _synthesize_pack


@pytest.mark.parametrize(
    "bs,d_in,d_out,hp_ratio",
    [
        # Decode regime (T <= DECODE_T_THRESHOLD)
        (1, 4096, 4096, 0.0),
        (1, 4096, 4096, 0.1),
        (16, 4096, 4096, 0.05),
        (64, 4096, 11008, 0.1),
        (DECODE_T_THRESHOLD, 4096, 4096, 0.05),
        # Prefill regime (T > DECODE_T_THRESHOLD)
        (DECODE_T_THRESHOLD + 1, 4096, 4096, 0.05),
        (512, 4096, 11008, 0.0),
        (2048, 4096, 4096, 0.1),
    ],
)
def test_dispatcher_entries_agree(bs, d_in, d_out, hp_ratio):
    torch.manual_seed(0)
    W = _synthesize_pack(d_out=d_out, d_in=d_in, hp_ratio=hp_ratio, seed=0)
    X = torch.randn(bs, d_in, dtype=torch.float16, device="cuda")

    y_dispatch = v9_linear_forward(X, W)
    y_decode = v9_linear_forward_decode(X, W)
    y_prefill = v9_linear_forward_prefill(X, W)

    assert y_dispatch.shape == (bs, d_out)
    # Today all three entries wrap the exact same kernel sequence, so
    # their outputs must be bit-identical (no FP non-determinism is
    # injected by the dispatcher).
    assert torch.equal(y_dispatch, y_decode), (
        f"decode entry diverged (T={bs}, hp={hp_ratio})"
    )
    assert torch.equal(y_dispatch, y_prefill), (
        f"prefill entry diverged (T={bs}, hp={hp_ratio})"
    )


def test_dispatcher_handles_3d_input():
    torch.manual_seed(0)
    d_in, d_out = 4096, 4096
    W = _synthesize_pack(d_out=d_out, d_in=d_in, hp_ratio=0.1, seed=0)

    # Decode-ish 3D
    X = torch.randn(4, 8, d_in, dtype=torch.float16, device="cuda")
    y_3d = v9_linear_forward(X, W)
    assert y_3d.shape == (4, 8, d_out)

    # Prefill-ish 3D (T = 4 * 512 > 128)
    X2 = torch.randn(4, 512, d_in, dtype=torch.float16, device="cuda")
    y_3d_pre = v9_linear_forward(X2, W)
    assert y_3d_pre.shape == (4, 512, d_out)

    # Consistency with the 2D path
    y_flat = v9_linear_forward(X.reshape(-1, d_in), W).reshape(4, 8, d_out)
    assert torch.equal(y_3d, y_flat)