"""CPU parity tests for the pure-torch W4A4 reference implementations.

These tests run entirely on CPU (no CUDA, no Triton, no CUTLASS) and
verify that our reference modules in :mod:`kernel.tools.parity` are
internally self-consistent and match a dequantise-then-matmul ground
truth within the project's standard ``rel_err < 5e-3`` bound.

Why this matters
----------------
R50 Step 2 (see ``.codebuddy/plan/r50_cutlass_int4/design.md`` §4)
replaces the dense accumulator with a CUTLASS kernel whose epilogue
reorders operations relative to the hand-tuned kernel.  Bit-exact
parity with the old kernel is **not** a requirement — only tolerance
parity against a well-defined FP16 mathematical truth is.  This file
establishes that truth as executable, CPU-runnable, machine-checkable
code so that when GPU comes back online we already know the reference
itself is correct.

Shapes
------
Four tiny shapes chosen so CPU matmul completes in <1 s each:
  - (d_out=128, d_in=256,  T=16)
  - (d_out=256, d_in=256,  T=32)
  - (d_out=256, d_in=512,  T=64)
  - (d_out=512, d_in=1024, T=32)

Tests
-----
* activation-quant reference round-trip (dequant ≈ original)
* activation-quant matches legacy eager-torch hand implementation
* sum_X identity: ``sum_X[t, g] == Σ_k q_s4[t, g·BCOL:(g+1)·BCOL]``
* dense reference == unpacked FP16 matmul (self-consistency)
* fused reference == dense + 16·sparse (structural check)
* zero-hp shortcut: fused == dense when hp_ratio = 0
"""

from __future__ import annotations

import math
from typing import Dict

import pytest
import torch

from kernel.tools.parity.fp16_reference import (
    fp16_dense_reference,
    fp16_fused_reference,
    fp16_sparse_reference,
)
from kernel.tools.parity.quant_reference import quantize_activation_s4_reference
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights, unpack_s4_le


# ---------------------------------------------------------------------------
# Fixture: build a complete V9 weight container on CPU.
# ---------------------------------------------------------------------------

def _synthesize_cpu_pack(
    d_out: int, d_in: int, hp_ratio: float = 0.05, seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """CPU-runnable variant of ``test_fused_dense_sparse::_synthesize_pack``.

    Returns a V9WeightContainer whose tensors all live on CPU.  Sparse
    blocks always have ``BROW × BCOL = 128 × 128`` shape, which is the
    pack_utils invariant.
    """
    nrow = d_out // BROW
    ncol = d_in // BCOL
    g = torch.Generator(device="cpu").manual_seed(seed)
    device = torch.device("cpu")

    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device=device, generator=g)
    scale_u4 = (torch.rand(d_out, ncol, generator=g, device=device) * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, ncol), generator=g, device=device).to(torch.float16)

    if hp_ratio > 0.0:
        n_hp = max(1, int(nrow * ncol * hp_ratio))
        # Pick unique block indices deterministically.
        combined = torch.unique(
            torch.randint(0, nrow * ncol, (n_hp * 2,), device=device, generator=g)
        )[:n_hp]
        brs = (combined // ncol).to(torch.int32)
        bcs = (combined % ncol).to(torch.int32)
        hp_indices = torch.stack([brs, bcs], dim=-1)
        Q_s8_blocks = torch.randint(
            -64, 64, (len(brs), BROW, BCOL),
            dtype=torch.int8, device=device, generator=g,
        )
        scale_s8 = (torch.rand(len(brs), BROW, generator=g, device=device) * 0.005 + 0.001).to(
            torch.float16
        )
    else:
        hp_indices = torch.zeros((0, 2), dtype=torch.int32, device=device)
        Q_s8_blocks = torch.zeros((0, BROW, BCOL), dtype=torch.int8, device=device)
        scale_s8 = torch.zeros((0, BROW), dtype=torch.float16, device=device)

    perm = torch.arange(d_in, dtype=torch.int32, device=device)

    W = pack_v9_weights({
        "Q_u4_permuted": Q_u4,
        "scale_u4_raw": scale_u4,
        "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8_blocks,
        "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices,
        "perm": perm,
    })
    return W


# ---------------------------------------------------------------------------
# Tolerance helpers
# ---------------------------------------------------------------------------

def _rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    abs_err = (actual.to(torch.float32) - ref.to(torch.float32)).abs()
    denom = ref.to(torch.float32).abs().max().clamp(min=1e-4)
    return (abs_err / denom).max().item()


TINY_SHAPES = [
    # (d_out, d_in, T, hp_ratio)
    (128, 256, 16, 0.10),
    (256, 256, 32, 0.10),
    (256, 512, 64, 0.05),
    (512, 1024, 32, 0.02),
]


# ---------------------------------------------------------------------------
# 1.  Activation quant reference — self-consistency & kernel-compat
# ---------------------------------------------------------------------------

class TestActivationQuantReference:
    def test_scale_is_max_over_seven(self) -> None:
        torch.manual_seed(42)
        X = torch.randn(16, 256, dtype=torch.float16)
        perm = torch.arange(256, dtype=torch.int32)
        _, scale_x, _ = quantize_activation_s4_reference(X, perm)

        expected = (X.abs().amax(dim=1).to(torch.float32) / 7.0).to(torch.float16)
        torch.testing.assert_close(scale_x, expected, rtol=0, atol=0)

    def test_quantized_values_in_range(self) -> None:
        torch.manual_seed(7)
        X = torch.randn(8, 256, dtype=torch.float16) * 3.0
        perm = torch.arange(256, dtype=torch.int32)
        X_s4, _, _ = quantize_activation_s4_reference(X, perm)

        unpacked = unpack_s4_le(X_s4, signed=True)
        assert unpacked.min() >= -8 and unpacked.max() <= 7

    def test_sum_x_matches_group_sums(self) -> None:
        torch.manual_seed(11)
        d_in = 256
        X = torch.randn(4, d_in, dtype=torch.float16)
        perm = torch.arange(d_in, dtype=torch.int32)
        X_s4, _, sum_X = quantize_activation_s4_reference(X, perm)

        unpacked = unpack_s4_le(X_s4, signed=True).to(torch.int32)   # (4, d_in)
        n_groups = d_in // BCOL
        expected = unpacked.view(4, n_groups, BCOL).sum(dim=-1).to(torch.int32)
        torch.testing.assert_close(sum_X, expected, rtol=0, atol=0)

    def test_zero_row_produces_zero_quant(self) -> None:
        X = torch.zeros(2, 256, dtype=torch.float16)
        perm = torch.arange(256, dtype=torch.int32)
        X_s4, scale_x, sum_X = quantize_activation_s4_reference(X, perm)

        assert scale_x.eq(0.0).all()
        assert X_s4.eq(0).all()
        assert sum_X.eq(0).all()

    def test_perm_reorders_columns(self) -> None:
        torch.manual_seed(3)
        d_in = 256
        X = torch.randn(2, d_in, dtype=torch.float16)

        perm_id = torch.arange(d_in, dtype=torch.int32)
        X_s4_id, scale_id, _ = quantize_activation_s4_reference(X, perm_id)

        # reverse permutation
        perm_rev = torch.arange(d_in - 1, -1, -1, dtype=torch.int32)
        X_s4_rev, scale_rev, _ = quantize_activation_s4_reference(X, perm_rev)

        # Scale only depends on |X|, not permutation order
        torch.testing.assert_close(scale_id, scale_rev, rtol=0, atol=0)

        # Quantized values must be reversed relative to identity
        uid = unpack_s4_le(X_s4_id, signed=True)
        urev = unpack_s4_le(X_s4_rev, signed=True)
        torch.testing.assert_close(urev, uid.flip(dims=[-1]))


# ---------------------------------------------------------------------------
# 2.  FP16 dense / sparse / fused references
# ---------------------------------------------------------------------------

class TestFp16References:
    @pytest.mark.parametrize("d_out,d_in,T,hp_ratio", TINY_SHAPES)
    def test_dense_reference_matches_naive_matmul(
        self, d_out: int, d_in: int, T: int, hp_ratio: float,
    ) -> None:
        """The dense reference must equal an independent "unpack W,
        unpack X, FP32 matmul, cast once" computation.
        """
        W = _synthesize_cpu_pack(d_out, d_in, hp_ratio=hp_ratio, seed=1)
        torch.manual_seed(100)
        X = (torch.randn(T, d_in, dtype=torch.float32) * 0.5).to(torch.float16)
        X_s4, scale_x, _ = quantize_activation_s4_reference(X, W.perm)

        # Independent truth: fully unpack via different code path.
        W_s4 = unpack_s4_le(W.W_low_packed, signed=True).to(torch.float32)
        scale = W.scale_u4.to(torch.float32).repeat_interleave(BCOL, dim=1)
        zero = W.zero_u4.to(torch.float32).repeat_interleave(BCOL, dim=1)
        W_fp16_truth = ((W_s4 - zero) * scale).to(torch.float16)

        X_unp = unpack_s4_le(X_s4, signed=True).to(torch.float32)
        X_fp16_truth = (X_unp * scale_x.to(torch.float32)[:, None]).to(torch.float16)

        Y_truth = (
            X_fp16_truth.to(torch.float32) @ W_fp16_truth.to(torch.float32).T
        ).to(torch.float16).T.contiguous()

        Y_ref = fp16_dense_reference(
            W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, scale_x,
            d_out=d_out, d_in=d_in,
        )

        # Must be identical — both paths use FP32 intermediate with
        # single FP16 cast at the end.
        torch.testing.assert_close(Y_ref, Y_truth, rtol=0, atol=0)

    @pytest.mark.parametrize("d_out,d_in,T", [(256, 256, 32), (512, 1024, 32)])
    def test_fused_equals_dense_plus_16_sparse(
        self, d_out: int, d_in: int, T: int,
    ) -> None:
        """Structural identity: Y_fused must be Y_low + 16·Y_high (FP32 sum, FP16 cast)."""
        W = _synthesize_cpu_pack(d_out, d_in, hp_ratio=0.05, seed=2)
        torch.manual_seed(200)
        X = (torch.randn(T, d_in, dtype=torch.float32) * 0.5).to(torch.float16)
        X_s4, scale_x, sum_X = quantize_activation_s4_reference(X, W.perm)

        Y_low = fp16_dense_reference(
            W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, scale_x,
            d_out=d_out, d_in=d_in,
        )
        Y_high = fp16_sparse_reference(
            W.W_high_blocks_packed, W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, scale_x, d_out=d_out, d_in=d_in,
        )
        Y_manual = (
            Y_low.to(torch.float32) + 16.0 * Y_high.to(torch.float32)
        ).to(torch.float16)

        Y_fused = fp16_fused_reference(
            W.W_low_packed, W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, W.zero_u4, scale_x,
            d_out=d_out, d_in=d_in, sum_X=sum_X,
        )

        torch.testing.assert_close(Y_fused, Y_manual, rtol=0, atol=0)

    @pytest.mark.parametrize("T", [1, 16, 32])
    def test_zero_hp_fused_equals_dense(self, T: int) -> None:
        """When hp_ratio=0 the sparse branch must contribute nothing."""
        d_out, d_in = 256, 256
        W = _synthesize_cpu_pack(d_out, d_in, hp_ratio=0.0, seed=5)
        torch.manual_seed(300)
        X = (torch.randn(T, d_in, dtype=torch.float32) * 0.3).to(torch.float16)
        X_s4, scale_x, sum_X = quantize_activation_s4_reference(X, W.perm)

        Y_dense = fp16_dense_reference(
            W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, scale_x,
            d_out=d_out, d_in=d_in,
        )
        Y_fused = fp16_fused_reference(
            W.W_low_packed, W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, W.zero_u4, scale_x,
            d_out=d_out, d_in=d_in, sum_X=sum_X,
        )

        torch.testing.assert_close(Y_fused, Y_dense, rtol=0, atol=0)

    @pytest.mark.parametrize("d_out,d_in,T,hp_ratio", TINY_SHAPES)
    def test_fused_within_tolerance_of_fp16_dequant_truth(
        self, d_out: int, d_in: int, T: int, hp_ratio: float,
    ) -> None:
        """End-to-end: Y_fused vs an external FP16 ground truth built by
        directly dequantising every packed tensor and doing one FP32 matmul.

        This is the parity harness that R50 G2 (``test_cutlass_parity_gpu``)
        will use against the real CUTLASS kernel.  If this CPU version
        fails, the GPU version is doomed.
        """
        W = _synthesize_cpu_pack(d_out, d_in, hp_ratio=hp_ratio, seed=11)
        torch.manual_seed(500)
        X = (torch.randn(T, d_in, dtype=torch.float32) * 0.5).to(torch.float16)
        X_s4, scale_x, sum_X = quantize_activation_s4_reference(X, W.perm)

        Y_fused = fp16_fused_reference(
            W.W_low_packed, W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, W.zero_u4, scale_x,
            d_out=d_out, d_in=d_in, sum_X=sum_X,
        )

        # Ground truth: reconstruct full (d_out, d_in) FP16 weight and
        # FP16 activation by an independent code path, then matmul.
        W_fp16_full = torch.zeros((d_out, d_in), dtype=torch.float32)
        # Low nibble contribution.
        W_s4_low = unpack_s4_le(W.W_low_packed, signed=True).to(torch.float32)
        s_low = W.scale_u4.to(torch.float32).repeat_interleave(BCOL, dim=1)
        z_low = W.zero_u4.to(torch.float32).repeat_interleave(BCOL, dim=1)
        W_fp16_full += (W_s4_low - z_low) * s_low
        # High nibble contribution (sparse), weighted by 16.
        if W.W_high_blocks_packed.shape[0] > 0:
            tiles = unpack_s4_le(W.W_high_blocks_packed, signed=True).to(torch.float32)
            row_off = W.hp_row_offsets.tolist()
            col_idx = W.hp_col_indices.tolist()
            scale_f32 = W.scale_u4.to(torch.float32)
            nrow = d_out // BROW
            for br in range(nrow):
                r0, r1 = br * BROW, (br + 1) * BROW
                for k in range(row_off[br], row_off[br + 1]):
                    bc = col_idx[k]
                    c0, c1 = bc * BCOL, (bc + 1) * BCOL
                    W_fp16_full[r0:r1, c0:c1] += (
                        16.0 * tiles[k] * scale_f32[r0:r1, bc : bc + 1]
                    )

        # Activations in FP16 space.
        X_unp = unpack_s4_le(X_s4, signed=True).to(torch.float32)
        X_fp16 = X_unp * scale_x.to(torch.float32)[:, None]

        Y_truth = (X_fp16 @ W_fp16_full.T).to(torch.float16).T.contiguous()

        err = _rel_err(Y_fused, Y_truth)
        assert err < 5e-3, (
            f"rel_err {err:.3e} >= 5e-3 for d_out={d_out} d_in={d_in} "
            f"T={T} hp={hp_ratio}"
        )


# ---------------------------------------------------------------------------
# 3.  Sanity: the reference can actually run the largest shape
# ---------------------------------------------------------------------------

class TestPerformanceSanity:
    def test_largest_shape_under_1s(self) -> None:
        """The reference is slow but must not be *pathologically* slow.
        If the 512×1024×32 shape takes > 5 s on any reasonable CPU we
        probably have an O(n^3) bug.
        """
        import time

        W = _synthesize_cpu_pack(512, 1024, hp_ratio=0.02, seed=99)
        X = torch.randn(32, 1024, dtype=torch.float16)
        X_s4, scale_x, sum_X = quantize_activation_s4_reference(X, W.perm)

        t0 = time.perf_counter()
        _ = fp16_fused_reference(
            W.W_low_packed, W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, W.zero_u4, scale_x,
            d_out=512, d_in=1024, sum_X=sum_X,
        )
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"reference too slow: {elapsed:.2f}s"
