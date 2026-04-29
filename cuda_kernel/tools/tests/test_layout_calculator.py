"""R50 L3.0 tests for :mod:`kernel.cuda_kernel.tools.layout_calculator`.

Six tests exactly matching the acceptance criteria in
``.codebuddy/plan/r50_cutlass_int4/layout_contract.md`` §8:

1. ``test_contract_summary_frozen_constants``  — the 14 frozen
   constants match layout_contract.md §2 exactly.
2. ``test_problem_shape_canonical_cluster_median`` — the median
   tc_underutil cluster shape (d_out=2048, d_in=2048, T=8) maps to
   the expected GemmCoord / tile grid.
3. ``test_problem_shape_tall_skinny_T1`` — decode-regime T=1 shape
   produces cta_grid_n == 1 (tail predication territory).
4. ``test_problem_shape_large_cluster`` — the large_T1024_gu_4096_24576
   shape fills >1 wave on a 128-SM GPU.
5. ``test_bytes_per_cta_fits_ada_smem`` — the default tile at 3
   stages fits Ada's 100 KB budget (positive case) and raises on
   absurd tile+stages (negative case).
6. ``test_verify_alignment_negative_paths`` — every alignment
   invariant raises on purpose-broken inputs.

All run on Mac / CPU; no torch, no CUDA, no triton.
"""

from __future__ import annotations

import pytest

from kernel.cuda_kernel.tools import layout_calculator as lc


# ---------------------------------------------------------------------------
# 1. frozen constants
# ---------------------------------------------------------------------------


def test_contract_summary_frozen_constants():
    summary = lc.contract_summary()

    # Atom (non-negotiable, must match SM80_16x8x64_S32S4S4S32_TN).
    assert summary["INSTR_M"] == 16
    assert summary["INSTR_N"] == 8
    assert summary["INSTR_K"] == 64

    # Threadblock tile (layout_contract.md §2.2).
    assert summary["TILE_M"] == 128
    assert summary["TILE_N"] == 128
    assert summary["TILE_K"] == 128

    # Warp shape (layout_contract.md §2.2).
    assert summary["WARP_M"] == 64
    assert summary["WARP_N"] == 64
    assert summary["WARP_K"] == 128
    assert summary["WARPS_PER_CTA"] == 4

    # Pipeline and alignment.
    assert summary["STAGES"] == 3
    assert summary["ALIGN_A"] == 128
    assert summary["ALIGN_B"] == 128

    # Group-K == BCOL from pack_utils (invariant I-L3).
    assert summary["GROUP_K"] == 128

    # Ada SMEM soft limit = 100 KB.
    assert summary["ADA_SMEM_SOFT_LIMIT_BYTES"] == 100 * 1024


# ---------------------------------------------------------------------------
# 2 / 3 / 4. problem shape across cluster archetypes
# ---------------------------------------------------------------------------


def test_problem_shape_canonical_cluster_median():
    # Median shape across the 84 tc_underutil cluster (design.md §3.1 / §6).
    # d_out=2048, d_in=2048, T=8 should map to:
    #   M=2048 N=8 K=2048
    #   cta_grid = (2048/128=16) * (ceil(8/128)=1) = 16
    #   k_tiles = 2048 / 128 = 16
    #   n_groups = 2048 / 128 = 16
    shape = lc.cutlass_problem_shape(d_out=2048, d_in=2048, T=8)

    assert shape.M == 2048
    assert shape.N == 8
    assert shape.K == 2048
    assert shape.cta_grid_m == 16
    assert shape.cta_grid_n == 1
    assert shape.cta_grid_total == 16
    assert shape.k_tiles == 16
    assert shape.n_groups_in_problem == 16
    assert shape.m_padded == 2048
    # tail-predication territory: T=8 padded to TILE_N=128
    assert shape.n_padded == 128
    assert shape.k_padded == 2048


def test_problem_shape_tall_skinny_T1():
    # Decode regime T=1: one CTA in N, M grid fills alone.
    shape = lc.cutlass_problem_shape(d_out=4096, d_in=2048, T=1)
    assert shape.cta_grid_m == 32
    assert shape.cta_grid_n == 1
    assert shape.cta_grid_total == 32
    # padded 1 → TILE_N=128 (heavy tail waste; motivates a potential
    # small-T SplitK variant in L3.4, see layout_contract.md §7)
    assert shape.n_padded == 128


def test_problem_shape_large_cluster():
    # large_T1024_gu_4096_24576 (audit re-classified shape; design.md §9).
    # Expect cta_grid_total to exceed one wave on a 128-SM GPU.
    shape = lc.cutlass_problem_shape(d_out=24576, d_in=4096, T=1024)
    # 24576/128 = 192; 1024/128 = 8; total = 1536 CTAs
    assert shape.cta_grid_m == 192
    assert shape.cta_grid_n == 8
    assert shape.cta_grid_total == 1536
    # On 4090 (128 SMs) this is 12 waves — definitely compute-bound,
    # launch-tax negligible, good Step 2 target.
    assert shape.cta_grid_total // 128 >= 10


# ---------------------------------------------------------------------------
# 5. shared-memory footprint
# ---------------------------------------------------------------------------


def test_bytes_per_cta_fits_ada_smem_positive():
    # Default: 3 stages × (128×128 + 128×128) × 4 bit / 8
    #        = 3 × 32768 × 4 / 8
    #        = 49152 bytes = 48 KB (well under 100 KB).
    bytes_ = lc.bytes_per_cta_k_slice()
    assert bytes_ == 48 * 1024
    assert bytes_ < lc.ADA_SMEM_SOFT_LIMIT_BYTES


def test_bytes_per_cta_rejects_oversize():
    # Absurd tile: (512, 512, 512) × 6 stages = 1.5 MB → must raise.
    with pytest.raises(ValueError, match="exceeds Ada"):
        lc.bytes_per_cta_k_slice(tile_mnk=(512, 512, 512), stages=6)

    # Negative / zero dims → must raise.
    with pytest.raises(ValueError, match="positive"):
        lc.bytes_per_cta_k_slice(tile_mnk=(0, 128, 128), stages=3)
    with pytest.raises(ValueError, match="positive"):
        lc.bytes_per_cta_k_slice(tile_mnk=(128, 128, 128), stages=0)


# ---------------------------------------------------------------------------
# 6. alignment invariants (negative paths)
# ---------------------------------------------------------------------------


def test_verify_alignment_positive():
    # Canonical cluster shapes must all pass.
    # (d_out, d_in, T) triples drawn from design.md §3.1 + §6 shape list.
    for d_out, d_in, T in [
        (2048, 2048, 8),
        (4096, 2048, 1),
        (24576, 4096, 1024),
        (1024, 2048, 8),
        (2048, 1024, 128),
        (2560, 4096, 1),
    ]:
        lc.verify_alignment(d_out, d_in, T)  # does not raise


def test_verify_alignment_negative_paths():
    # d_in not divisible by 128 → fails kAlignmentA.
    with pytest.raises(ValueError, match="ALIGN_A"):
        lc.verify_alignment(d_out=2048, d_in=2047, T=8)

    # d_in divisible by 128 but 64 (hypothetical) — still must pass 128
    # (we don't have a 64-but-not-128 value to exercise cleanly, so
    # exercise the GROUP_K path explicitly).
    # Here d_in=64 fails ALIGN_A=128 first — matches the order of checks.
    with pytest.raises(ValueError, match="ALIGN_A"):
        lc.verify_alignment(d_out=2048, d_in=64, T=8)

    # Zero / negative → must raise.
    with pytest.raises(ValueError, match="d_out"):
        lc.verify_alignment(d_out=0, d_in=2048, T=8)
    with pytest.raises(ValueError, match="d_in"):
        lc.verify_alignment(d_out=2048, d_in=-128, T=8)
    with pytest.raises(ValueError, match=r"^verify_alignment: T"):
        lc.verify_alignment(d_out=2048, d_in=2048, T=0)


# ---------------------------------------------------------------------------
# 7. problem shape edge cases (bonus — round out coverage)
# ---------------------------------------------------------------------------


def test_problem_shape_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        lc.cutlass_problem_shape(d_out=0, d_in=2048, T=8)
    with pytest.raises(ValueError, match="positive"):
        lc.cutlass_problem_shape(d_out=2048, d_in=0, T=8)
    with pytest.raises(ValueError, match="positive"):
        lc.cutlass_problem_shape(d_out=2048, d_in=2048, T=0)
