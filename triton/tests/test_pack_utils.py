"""Unit tests for `pack_utils.pack_v9_weights`.

These tests do NOT require a GPU.  They construct a small synthetic GPTQ
output (d_out=256, d_in=256 = 2 block-rows x 2 block-cols; 2 high-precision
blocks) and check the layout invariants listed in requirements.md sec 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Allow running via `pytest kernel/triton/tests/` without an installed package.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # add workspace root

from kernel.triton.pack_utils import (  # noqa: E402
    BCOL,
    BROW,
    pack_s4_le,
    pack_v9_weights,
    unpack_s4_le,
)


def _build_synthetic_gptq_outputs(
    d_out: int = 256,
    d_in: int = 256,
    hp_pairs=((0, 1), (1, 0)),
    seed: int = 0,
):
    """Build a minimal synthetic GPTQ output dict."""
    g = torch.Generator().manual_seed(seed)
    n_groups = d_in // BCOL

    # UINT4 baseline weights and params.
    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int32, generator=g)
    scale_u4_raw = torch.rand((d_out, n_groups), generator=g).to(torch.float16) * 0.01 + 0.001
    zero_u4_raw = torch.randint(0, 16, (d_out, n_groups), dtype=torch.int32, generator=g).to(torch.float16)

    # High-precision SINT8 blocks
    n_hp = len(hp_pairs)
    Q_s8 = torch.randint(-128, 128, (n_hp, BROW, BCOL), dtype=torch.int32, generator=g).to(torch.int8)
    scale_s8 = (torch.rand((n_hp, BROW), generator=g).to(torch.float16) * 0.005 + 0.001)
    hp_indices = torch.tensor(hp_pairs, dtype=torch.int32)

    perm = torch.randperm(d_in, generator=g).to(torch.int32)

    return {
        "Q_u4_permuted": Q_u4.to(torch.int8),
        "scale_u4_raw": scale_u4_raw,
        "zero_u4_raw": zero_u4_raw,
        "Q_s8_blocks": Q_s8,
        "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices,
        "perm": perm,
    }


def test_pack_s4_le_roundtrip_signed():
    # Values in SINT4 range
    x = torch.tensor([[-8, -1, 0, 7, 3, -4, -2, 1]], dtype=torch.int32)
    packed = pack_s4_le(x)
    assert packed.shape == (1, 4)
    unpacked = unpack_s4_le(packed, signed=True)
    assert torch.equal(unpacked.to(torch.int32), x)


def test_pack_s4_le_roundtrip_unsigned():
    x = torch.tensor([[0, 15, 1, 2, 8, 9, 14, 3]], dtype=torch.int32)
    packed = pack_s4_le(x)
    unpacked = unpack_s4_le(packed, signed=False)
    assert torch.equal(unpacked.to(torch.int32), x)


def test_pack_v9_weights_shapes_and_invariants():
    gptq = _build_synthetic_gptq_outputs()
    W = pack_v9_weights(gptq)

    d_out, d_in = 256, 256
    n_groups = d_in // BCOL
    nrow = d_out // BROW

    # Basic shapes
    assert W.W_low_packed.shape == (d_out, d_in // 2)
    assert W.W_low_packed.dtype == torch.int8
    assert W.scale_u4.shape == (d_out, n_groups)
    assert W.zero_u4.shape == (d_out, n_groups)
    assert W.hp_row_offsets.shape == (nrow + 1,)
    assert W.hp_col_indices.shape == (W.n_hp_blocks,)
    assert W.W_high_blocks_packed.shape == (W.n_hp_blocks, BROW, BCOL // 2)
    assert W.perm.shape == (d_in,)
    assert W.block_shape == (BROW, BCOL)
    assert W.d_out == d_out
    assert W.d_in == d_in


def test_pack_v9_weights_bsr_monotone():
    gptq = _build_synthetic_gptq_outputs(hp_pairs=((1, 0), (0, 1), (1, 1), (0, 0)))
    W = pack_v9_weights(gptq)
    nrow = 2
    # hp_row_offsets monotone non-decreasing and last == n_hp
    ro = W.hp_row_offsets.tolist()
    assert ro[-1] == W.n_hp_blocks
    for i in range(len(ro) - 1):
        assert ro[i] <= ro[i + 1]
    # Inside each row, hp_col_indices ascending
    ci = W.hp_col_indices.tolist()
    for br in range(nrow):
        s, e = ro[br], ro[br + 1]
        segment = ci[s:e]
        assert segment == sorted(segment)


def test_pack_v9_weights_scale_override_for_sint8_blocks():
    hp_pairs = ((0, 1), (1, 0))
    gptq = _build_synthetic_gptq_outputs(hp_pairs=hp_pairs)
    W = pack_v9_weights(gptq)

    scale_s8 = gptq["scale_s8_per_block"].to(torch.float16)
    # For each HP block, scale_u4 on that (rows, bc) must equal scale_s8
    # and zero_u4 must equal 0 - 8 = -8 after pre-subtract of 8.
    # Map from sorted BSR order back to original idx:
    # sorted by (br, bc) asc
    hp_sorted = sorted(range(len(hp_pairs)), key=lambda i: (hp_pairs[i][0], hp_pairs[i][1]))
    for pos, orig_idx in enumerate(hp_sorted):
        br, bc = hp_pairs[orig_idx]
        r0, r1 = br * BROW, (br + 1) * BROW
        # scale_u4[r0:r1, bc] should match scale_s8[orig_idx, :r1-r0]
        got_scale = W.scale_u4[r0:r1, bc].cpu().to(torch.float32)
        expect_scale = scale_s8[orig_idx, : r1 - r0].cpu().to(torch.float32)
        assert torch.allclose(got_scale, expect_scale, atol=1e-5)
        # zero_u4 should be -8 (pre-subtracted) since original override was 0
        got_zero = W.zero_u4[r0:r1, bc].cpu().to(torch.float32)
        assert torch.all(got_zero == -8.0), f"zero_u4 should be -8 for SINT8 groups, got {got_zero[:4]}"


def test_pack_v9_weights_bitsplit_exactness():
    """Reconstruct q_s8 from W_low_packed + W_high_blocks_packed and check equality."""
    hp_pairs = ((0, 0), (1, 1))
    gptq = _build_synthetic_gptq_outputs(hp_pairs=hp_pairs)
    W = pack_v9_weights(gptq)

    # Unpack low-bit (currently pre-subtracted 8, so add 8 back to get UINT4 [0,15]).
    w_low_s4 = unpack_s4_le(W.W_low_packed, signed=True).to(torch.int32)
    w_low_u4 = w_low_s4 + 8   # in [0, 15]

    # Unpack high blocks
    w_high = unpack_s4_le(W.W_high_blocks_packed, signed=True).to(torch.int32)

    hp_sorted = sorted(range(len(hp_pairs)), key=lambda i: (hp_pairs[i][0], hp_pairs[i][1]))
    for pos, orig_idx in enumerate(hp_sorted):
        br, bc = hp_pairs[orig_idx]
        r0, r1 = br * BROW, (br + 1) * BROW
        c0, c1 = bc * BCOL, (bc + 1) * BCOL
        q_low_block = w_low_u4[r0:r1, c0:c1]
        q_high_block = w_high[pos, : r1 - r0, : c1 - c0]
        recon = q_high_block * 16 + q_low_block                     # should equal original q_s8
        expected = gptq["Q_s8_blocks"][orig_idx, : r1 - r0, : c1 - c0].to(torch.int32)
        assert torch.equal(recon, expected)


def test_pack_v9_weights_rejects_wrong_block_shape():
    gptq = _build_synthetic_gptq_outputs()
    with pytest.raises(ValueError):
        pack_v9_weights(gptq, brow=64, bcol=128)
    with pytest.raises(ValueError):
        pack_v9_weights(gptq, brow=128, bcol=64)
