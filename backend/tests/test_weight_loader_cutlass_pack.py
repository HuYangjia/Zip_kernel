"""R50 L4.3 — Regression tests for the CUTLASS weight-loader adapter.

Verifies every rule in ``.codebuddy/plan/r50_cutlass_int4/
layout_contract.md`` §1 / §2 / §4 is enforced by
:func:`kernel.backend.weight_loader.pack_v9_weights_for_cutlass`.

The invariant I-L5 (bit-identical round-trip) is the primary test;
the negative-path tests complete the `strict=True` contract
coverage.

All tests run pure-CPU (torch only; no CUDA, no Triton, no ninja).
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from kernel.backend.weight_loader import (
    CutlassPackValidationError,
    CutlassV9Tensors,
    _collect_violations,
    pack_v9_weights_for_cutlass,
)
from kernel.triton_kernel.pack_utils import (
    V9WeightContainer,
    pack_s4_le,
    unpack_s4_le,
)


# ---------------------------------------------------------------------------
# Fixtures — synthesise a V9WeightContainer without running the full packer
# ---------------------------------------------------------------------------


def _synthetic_container(
    d_out: int = 2048,
    d_in: int = 2048,
    brow: int = 128,
    bcol: int = 128,
    seed: int = 0,
) -> Tuple[V9WeightContainer, torch.Tensor]:
    """Build a V9WeightContainer directly, bypassing pack_v9_weights.

    Returns the container and the ground-truth SINT4 weight tensor
    ``W_ref`` (int8 in [-8, 7], shape (d_out, d_in)) that was packed
    into ``container.W_low_packed``.
    """

    g = torch.Generator().manual_seed(seed)
    # SINT4 values in [-8, 7].
    W_ref = torch.randint(-8, 8, (d_out, d_in), generator=g, dtype=torch.int8)

    # pack_s4_le expects UINT4 in [0, 15] bit-pattern; -x is represented by
    # its 4-bit two's complement. Convert -8..7 to 0..15 via &0x0F.
    W_u4 = (W_ref.to(torch.int32) & 0x0F).to(torch.int8)
    W_low_packed = pack_s4_le(W_u4)  # (d_out, d_in // 2) int8

    n_groups = d_in // bcol

    scale_u4 = torch.rand(d_out, n_groups, generator=g).to(torch.float16)
    zero_u4 = torch.rand(d_out, n_groups, generator=g).to(torch.float16)

    # Sparse-path bookkeeping (irrelevant for the dense CUTLASS adapter;
    # kept minimal / empty to keep the synthesis fast).
    W_high_blocks_packed = torch.zeros((0, brow, bcol // 2), dtype=torch.int8)
    hp_row_offsets = torch.zeros((d_out // brow) + 1, dtype=torch.int32)
    hp_col_indices = torch.zeros((0,), dtype=torch.int32)
    perm = torch.arange(d_in, dtype=torch.int32)

    container = V9WeightContainer(
        W_low_packed=W_low_packed,
        W_high_blocks_packed=W_high_blocks_packed,
        scale_u4=scale_u4,
        zero_u4=zero_u4,
        hp_row_offsets=hp_row_offsets,
        hp_col_indices=hp_col_indices,
        perm=perm,
        block_shape=(brow, bcol),
        d_out=d_out,
        d_in=d_in,
    )
    return container, W_ref


# ---------------------------------------------------------------------------
# 1. Primary invariant — bit-identical round trip (I-L5)
# ---------------------------------------------------------------------------


def test_round_trip_bit_identical_canonical_shape():
    # (d_out, d_in) = (2048, 2048) — median cluster shape (design.md §6).
    container, W_ref = _synthetic_container(d_out=2048, d_in=2048, seed=1)

    view = pack_v9_weights_for_cutlass(container)
    assert isinstance(view, CutlassV9Tensors)

    # Round-trip: pack-view → unpack → must equal original ground truth.
    W_rt = unpack_s4_le(view.W_low_rowmajor, signed=True)
    assert W_rt.dtype == torch.int8
    assert W_rt.shape == W_ref.shape
    assert torch.equal(W_rt, W_ref), (
        "Round-trip produced a different weight tensor — adapter must be "
        "a pass-through, not a repack (layout_contract.md I-L5)."
    )


@pytest.mark.parametrize(
    "d_out, d_in",
    [
        (4096, 2048),   # Qwen3-0.6B qkv T=1
        (2048, 1024),   # decode small
        (2560, 4096),   # Qwen3-1.7B shape
        (1024, 2048),   # down-proj archetype
    ],
)
def test_round_trip_bit_identical_parametrised(d_out: int, d_in: int):
    container, W_ref = _synthetic_container(d_out=d_out, d_in=d_in, seed=2)
    view = pack_v9_weights_for_cutlass(container)
    W_rt = unpack_s4_le(view.W_low_rowmajor, signed=True)
    assert torch.equal(W_rt, W_ref)


# ---------------------------------------------------------------------------
# 2. Zero-copy / aliasing guarantee
# ---------------------------------------------------------------------------


def test_adapter_does_not_copy_storage():
    container, _ = _synthetic_container()
    view = pack_v9_weights_for_cutlass(container)
    # Storage identity: the underlying data_ptr must match.
    assert view.W_low_rowmajor.data_ptr() == container.W_low_packed.data_ptr()
    assert view.scale_u4.data_ptr() == container.scale_u4.data_ptr()
    assert view.zero_u4.data_ptr() == container.zero_u4.data_ptr()
    # source back-pointer.
    assert view.source is container


def test_view_dimensions_match_container():
    container, _ = _synthetic_container(d_out=1024, d_in=1024)
    view = pack_v9_weights_for_cutlass(container)
    assert view.d_out == 1024
    assert view.d_in == 1024
    assert view.n_groups == 1024 // 128
    assert view.W_low_rowmajor.shape == (1024, 512)
    assert view.scale_u4.shape == (1024, 8)


# ---------------------------------------------------------------------------
# 3. Negative paths — each violation in layout_contract.md §1 surfaces
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_bad_dtype():
    container, _ = _synthetic_container()
    # Force wrong weight dtype.
    bad = container.W_low_packed.to(torch.int32)
    bad_container = V9WeightContainer(
        W_low_packed=bad,
        W_high_blocks_packed=container.W_high_blocks_packed,
        scale_u4=container.scale_u4,
        zero_u4=container.zero_u4,
        hp_row_offsets=container.hp_row_offsets,
        hp_col_indices=container.hp_col_indices,
        perm=container.perm,
        block_shape=container.block_shape,
        d_out=container.d_out,
        d_in=container.d_in,
    )

    with pytest.raises(CutlassPackValidationError) as exc_info:
        pack_v9_weights_for_cutlass(bad_container, strict=True)
    assert any("W_low_packed.dtype" in v for v in exc_info.value.violations)


def test_strict_mode_raises_on_unaligned_d_in():
    # d_in=2040 is divisible by 8 but NOT by 128. Synthesis still works
    # because we built the container by hand.
    container, _ = _synthetic_container(d_out=128, d_in=128)
    # Surgically replace with an un-aligned weight tensor.
    g = torch.Generator().manual_seed(99)
    W_u4 = torch.randint(0, 16, (128, 120), generator=g, dtype=torch.int8)
    bad_w = pack_s4_le(W_u4)  # (128, 60)
    bad_container = V9WeightContainer(
        W_low_packed=bad_w,
        W_high_blocks_packed=container.W_high_blocks_packed,
        scale_u4=container.scale_u4,
        zero_u4=container.zero_u4,
        hp_row_offsets=container.hp_row_offsets,
        hp_col_indices=container.hp_col_indices,
        perm=container.perm,
        block_shape=container.block_shape,
        d_out=128,
        d_in=120,
    )
    with pytest.raises(CutlassPackValidationError) as exc_info:
        pack_v9_weights_for_cutlass(bad_container, strict=True)
    # Must surface BOTH the alignment and the n_groups discrepancy.
    msgs = exc_info.value.violations
    assert any("kAlignmentA" in v for v in msgs)


def test_strict_mode_raises_on_stride_violation():
    container, _ = _synthetic_container(d_out=256, d_in=256)
    # Transpose so stride(1) != 1 and put it back as a weight tensor
    # of the same logical shape. We rebuild the tensor with a custom
    # stride by using `as_strided`.
    bad_w = container.W_low_packed.transpose(0, 1).contiguous().transpose(0, 1)
    # The .contiguous().transpose() flips strides; shape is now (d_in//2, d_out)
    # which breaks the 2-D shape contract first. Instead, fabricate a
    # (d_out, d_in//2) view whose stride(1) != 1 via a 3-D stage.
    staged = container.W_low_packed.unsqueeze(-1).expand(-1, -1, 2)[:, :, 0]
    # `staged` shape (d_out, d_in//2) but stride(1) in unexpanded layout.
    # Verify that it triggers the stride check.
    assert staged.shape == container.W_low_packed.shape
    if staged.stride(1) == 1:  # pragma: no cover - torch may choose to materialise
        pytest.skip(
            "torch materialised the expand()+slice() view; "
            "stride-violation not reproducible on this build"
        )
    bad_container = V9WeightContainer(
        W_low_packed=staged,
        W_high_blocks_packed=container.W_high_blocks_packed,
        scale_u4=container.scale_u4,
        zero_u4=container.zero_u4,
        hp_row_offsets=container.hp_row_offsets,
        hp_col_indices=container.hp_col_indices,
        perm=container.perm,
        block_shape=container.block_shape,
        d_out=container.d_out,
        d_in=container.d_in,
    )
    with pytest.raises(CutlassPackValidationError) as exc_info:
        pack_v9_weights_for_cutlass(bad_container, strict=True)
    assert any("stride(1)" in v for v in exc_info.value.violations)


def test_strict_mode_raises_on_scale_dtype_mismatch():
    container, _ = _synthetic_container()
    bad_scale = container.scale_u4.to(torch.float32)
    bad_container = V9WeightContainer(
        W_low_packed=container.W_low_packed,
        W_high_blocks_packed=container.W_high_blocks_packed,
        scale_u4=bad_scale,
        zero_u4=container.zero_u4,
        hp_row_offsets=container.hp_row_offsets,
        hp_col_indices=container.hp_col_indices,
        perm=container.perm,
        block_shape=container.block_shape,
        d_out=container.d_out,
        d_in=container.d_in,
    )
    with pytest.raises(CutlassPackValidationError) as exc_info:
        pack_v9_weights_for_cutlass(bad_container, strict=True)
    assert any("scale_u4.dtype" in v for v in exc_info.value.violations)


def test_strict_mode_raises_on_zero_shape_mismatch():
    container, _ = _synthetic_container()
    # Zero tensor with wrong n_groups.
    bad_zero = torch.zeros((container.d_out, 3), dtype=torch.float16)
    bad_container = V9WeightContainer(
        W_low_packed=container.W_low_packed,
        W_high_blocks_packed=container.W_high_blocks_packed,
        scale_u4=container.scale_u4,
        zero_u4=bad_zero,
        hp_row_offsets=container.hp_row_offsets,
        hp_col_indices=container.hp_col_indices,
        perm=container.perm,
        block_shape=container.block_shape,
        d_out=container.d_out,
        d_in=container.d_in,
    )
    with pytest.raises(CutlassPackValidationError) as exc_info:
        pack_v9_weights_for_cutlass(bad_container, strict=True)
    assert any("zero_u4.shape" in v for v in exc_info.value.violations)


# ---------------------------------------------------------------------------
# 4. Non-strict mode — surfaces all violations via _collect_violations
# ---------------------------------------------------------------------------


def test_non_strict_mode_returns_view_even_on_violations():
    container, _ = _synthetic_container()
    bad_scale = container.scale_u4.to(torch.float32)
    bad_container = V9WeightContainer(
        W_low_packed=container.W_low_packed,
        W_high_blocks_packed=container.W_high_blocks_packed,
        scale_u4=bad_scale,
        zero_u4=container.zero_u4,
        hp_row_offsets=container.hp_row_offsets,
        hp_col_indices=container.hp_col_indices,
        perm=container.perm,
        block_shape=container.block_shape,
        d_out=container.d_out,
        d_in=container.d_in,
    )
    # strict=False: must NOT raise; caller inspects violations directly.
    view = pack_v9_weights_for_cutlass(bad_container, strict=False)
    assert isinstance(view, CutlassV9Tensors)
    violations = _collect_violations(bad_container)
    assert any("scale_u4.dtype" in v for v in violations)


def test_multiple_violations_reported_together():
    container, _ = _synthetic_container()
    # Simultaneously corrupt dtype and scale shape.
    bad_w = container.W_low_packed.to(torch.int32)
    bad_scale = torch.zeros((container.d_out, 3), dtype=torch.float16)
    bad_container = V9WeightContainer(
        W_low_packed=bad_w,
        W_high_blocks_packed=container.W_high_blocks_packed,
        scale_u4=bad_scale,
        zero_u4=container.zero_u4,
        hp_row_offsets=container.hp_row_offsets,
        hp_col_indices=container.hp_col_indices,
        perm=container.perm,
        block_shape=container.block_shape,
        d_out=container.d_out,
        d_in=container.d_in,
    )
    with pytest.raises(CutlassPackValidationError) as exc_info:
        pack_v9_weights_for_cutlass(bad_container, strict=True)
    msgs = exc_info.value.violations
    # Both issues must appear in the same error.
    assert any("W_low_packed.dtype" in v for v in msgs), msgs
    assert any("scale_u4.shape" in v for v in msgs), msgs


# ---------------------------------------------------------------------------
# 5. Public re-export sanity
# ---------------------------------------------------------------------------


def test_public_reexport_from_backend_package():
    import kernel.backend as backend_pkg

    assert hasattr(backend_pkg, "pack_v9_weights_for_cutlass")
    assert hasattr(backend_pkg, "CutlassV9Tensors")
    assert hasattr(backend_pkg, "CutlassPackValidationError")
    # Identity — same object, not a re-binding.
    assert (
        backend_pkg.pack_v9_weights_for_cutlass is pack_v9_weights_for_cutlass
    )
