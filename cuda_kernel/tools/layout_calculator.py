"""R50 L3.0 CUTLASS INT4 layout calculator (pure Python, CPU-only).

This module is the single source of truth for translating a *logical*
W4A4 GEMM problem shape `(d_out, d_in, T)` into the *concrete* CUTLASS
template parameters frozen by
`.codebuddy/plan/r50_cutlass_int4/layout_contract.md` (L3.0).

It never imports torch or CUDA; it is safe to run on any host
(Mac dev, CI, Linux) at import time.

Three responsibilities:

1. `cutlass_problem_shape(d_out, d_in, T)`
   → a dict describing the CUTLASS `GemmCoord` (M, N, K), the tile
   grid, the number of K-iterations, and group-K bookkeeping.

2. `bytes_per_cta_k_slice(tile_mnk, stages)`
   → static shared-memory footprint per CTA; asserts it fits in
   Ada's 100 KB / SM limit.

3. `verify_alignment(d_out, d_in, T)`
   → raises `ValueError` if the shape breaks any contract invariant
   (kAlignmentA/B=128, tile_k=BCOL=128, etc.).

Unit-tested in
`kernel/cuda_kernel/tests/test_layout_calculator.py` (6 tests, all
CPU).

References:
- layout_contract.md §2.1 (problem shape), §2.2 (tile), §2.4 (layouts)
- pack_utils.py BCOL=128 (group-K constant; equals CUTLASS tile_k)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Frozen constants (layout_contract.md §2)
# ---------------------------------------------------------------------------

# Instruction shape of `SM80_16x8x64_S32S4S4S32_TN` (atom-mandated, non-negotiable).
INSTR_M: int = 16
INSTR_N: int = 8
INSTR_K: int = 64

# Threadblock shape (layout_contract.md §2.2).
TILE_M: int = 128
TILE_N: int = 128
TILE_K: int = 128

# Warp shape (layout_contract.md §2.2).
WARP_M: int = 64
WARP_N: int = 64
WARP_K: int = 128
WARPS_PER_CTA: int = (TILE_M // WARP_M) * (TILE_N // WARP_N) * (TILE_K // WARP_K)  # = 4

# CUTLASS `cp.async` pipeline stages (addresses sub-bottleneck B3).
STAGES: int = 3

# kAlignmentA/B in elements (INT4). See layout_contract.md §2.2.
ALIGN_A: int = 128
ALIGN_B: int = 128

# Group-K constant from V9 pack format (BCOL in pack_utils.py).
# Contract invariant I-L3: tile_k == group_k.
GROUP_K: int = 128

# Ada SM shared-memory soft limit (RTX 4090: 100 KB opt-in per block).
ADA_SMEM_SOFT_LIMIT_BYTES: int = 100 * 1024


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CutlassProblemShape:
    """Output of :func:`cutlass_problem_shape`.

    All integer fields. `d_out / d_in / T` are the *logical* problem
    dimensions; the remaining fields derive from the frozen tile
    schedule.
    """

    # logical problem (matches `launch(...)` arguments):
    d_out: int
    d_in: int
    T: int

    # CUTLASS `GemmCoord` (layout_contract.md §2.1):
    # D (d_out, T)  =  A (d_out, d_in) @ B (d_in, T)  ==> M=d_out, N=T, K=d_in
    M: int
    N: int
    K: int

    # tile grid:
    cta_grid_m: int
    cta_grid_n: int
    cta_grid_total: int

    # iteration counts:
    k_tiles: int          # number of CTA-K tiles (= K / TILE_K, ceiled)
    n_groups_in_problem: int  # number of quant groups across K (= d_in / GROUP_K)

    # bookkeeping:
    m_padded: int         # d_out rounded up to TILE_M
    n_padded: int         # T rounded up to TILE_N
    k_padded: int         # d_in rounded up to TILE_K


# ---------------------------------------------------------------------------
# Primary helpers
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def cutlass_problem_shape(d_out: int, d_in: int, T: int) -> CutlassProblemShape:
    """Compute CUTLASS problem-shape bookkeeping.

    Call site: L3.4 kernel launcher (to build the `GemmCoord`) and
    tests (to assert the tile grid agrees with bench-time measurement).

    No alignment check here; use :func:`verify_alignment` before this
    if your caller has not yet validated.
    """

    if d_out <= 0 or d_in <= 0 or T <= 0:
        raise ValueError(
            f"cutlass_problem_shape requires strictly positive dims, got "
            f"d_out={d_out}, d_in={d_in}, T={T}"
        )

    M, N, K = d_out, T, d_in
    m_padded = _ceil_div(M, TILE_M) * TILE_M
    n_padded = _ceil_div(N, TILE_N) * TILE_N
    k_padded = _ceil_div(K, TILE_K) * TILE_K

    cta_grid_m = _ceil_div(M, TILE_M)
    cta_grid_n = _ceil_div(N, TILE_N)

    return CutlassProblemShape(
        d_out=d_out,
        d_in=d_in,
        T=T,
        M=M,
        N=N,
        K=K,
        cta_grid_m=cta_grid_m,
        cta_grid_n=cta_grid_n,
        cta_grid_total=cta_grid_m * cta_grid_n,
        k_tiles=_ceil_div(K, TILE_K),
        n_groups_in_problem=_ceil_div(K, GROUP_K),
        m_padded=m_padded,
        n_padded=n_padded,
        k_padded=k_padded,
    )


def bytes_per_cta_k_slice(
    tile_mnk: Tuple[int, int, int] = (TILE_M, TILE_N, TILE_K),
    stages: int = STAGES,
) -> int:
    """Static shared-memory footprint per CTA.

    Accounts only for the A+B operand double-buffer. Does NOT account
    for epilogue scratch (which on Ada/Sm80 is typically much smaller
    than 10 KB and rides in unused SMEM tail).

    Formula: ``stages * (tile_m * tile_k + tile_n * tile_k) * (4 bits) / 8``

    Raises ``ValueError`` if the computed footprint exceeds
    :data:`ADA_SMEM_SOFT_LIMIT_BYTES`.
    """

    tile_m, tile_n, tile_k = tile_mnk
    if tile_m <= 0 or tile_n <= 0 or tile_k <= 0 or stages <= 0:
        raise ValueError(
            f"bytes_per_cta_k_slice requires positive args, got "
            f"tile_mnk={tile_mnk}, stages={stages}"
        )

    bits = stages * (tile_m * tile_k + tile_n * tile_k) * 4
    # Round up to bytes (always divisible by 8 for the nominal tile).
    bytes_ = (bits + 7) // 8

    if bytes_ > ADA_SMEM_SOFT_LIMIT_BYTES:
        raise ValueError(
            f"CTA shared-memory footprint {bytes_} bytes exceeds Ada's "
            f"{ADA_SMEM_SOFT_LIMIT_BYTES}-byte soft limit for "
            f"tile_mnk={tile_mnk}, stages={stages}. Reduce stages or "
            f"tile dims."
        )

    return bytes_


def verify_alignment(d_out: int, d_in: int, T: int) -> None:
    """Raise ``ValueError`` if problem shape violates any
    layout_contract.md invariant.

    Enforces:
      - ``d_in % ALIGN_A == 0``  (weight K-alignment for `cp.async.128b`)
      - ``d_in % ALIGN_B == 0``  (activation K-alignment)
      - ``d_in % GROUP_K == 0``  (quant-group alignment with CTA-K)
      - ``d_in % TILE_K == 0``   (CTA-K alignment; follows group=tile=128)
      - ``d_out > 0``, ``T > 0``

    Note: ``d_out`` and ``T`` are *not* required to be multiples of
    TILE_M / TILE_N — CUTLASS supports CTA-level predication on the
    tail, and the existing kernel already exercises tails like d_out=9216.
    """

    if d_out <= 0:
        raise ValueError(f"verify_alignment: d_out must be > 0, got {d_out}")
    if d_in <= 0:
        raise ValueError(f"verify_alignment: d_in must be > 0, got {d_in}")
    if T <= 0:
        raise ValueError(f"verify_alignment: T must be > 0, got {T}")

    # K-dim alignment gates (hard; `cp.async.128b` requires this).
    if d_in % ALIGN_A != 0:
        raise ValueError(
            f"verify_alignment: d_in={d_in} not divisible by "
            f"ALIGN_A={ALIGN_A} (CUTLASS cp.async.128b for INT4 requires "
            f"128-element alignment)"
        )
    if d_in % ALIGN_B != 0:
        raise ValueError(
            f"verify_alignment: d_in={d_in} not divisible by "
            f"ALIGN_B={ALIGN_B}"
        )
    if d_in % GROUP_K != 0:
        raise ValueError(
            f"verify_alignment: d_in={d_in} not divisible by GROUP_K="
            f"{GROUP_K}; V9 pack format requires groups aligned to BCOL"
        )
    if d_in % TILE_K != 0:
        raise ValueError(
            f"verify_alignment: d_in={d_in} not divisible by TILE_K="
            f"{TILE_K}; invariant I-L3 violated"
        )


# ---------------------------------------------------------------------------
# Public helpers re-exported (makes the module self-contained)
# ---------------------------------------------------------------------------


def contract_summary() -> Dict[str, int]:
    """Return every frozen constant as a dict, for reporting / logging.

    Consumers: L3.4 build-time header emission, test suite assertions.
    """
    return {
        "INSTR_M": INSTR_M,
        "INSTR_N": INSTR_N,
        "INSTR_K": INSTR_K,
        "TILE_M": TILE_M,
        "TILE_N": TILE_N,
        "TILE_K": TILE_K,
        "WARP_M": WARP_M,
        "WARP_N": WARP_N,
        "WARP_K": WARP_K,
        "WARPS_PER_CTA": WARPS_PER_CTA,
        "STAGES": STAGES,
        "ALIGN_A": ALIGN_A,
        "ALIGN_B": ALIGN_B,
        "GROUP_K": GROUP_K,
        "ADA_SMEM_SOFT_LIMIT_BYTES": ADA_SMEM_SOFT_LIMIT_BYTES,
    }


__all__ = [
    # constants
    "INSTR_M",
    "INSTR_N",
    "INSTR_K",
    "TILE_M",
    "TILE_N",
    "TILE_K",
    "WARP_M",
    "WARP_N",
    "WARP_K",
    "WARPS_PER_CTA",
    "STAGES",
    "ALIGN_A",
    "ALIGN_B",
    "GROUP_K",
    "ADA_SMEM_SOFT_LIMIT_BYTES",
    # data classes
    "CutlassProblemShape",
    # helpers
    "cutlass_problem_shape",
    "bytes_per_cta_k_slice",
    "verify_alignment",
    "contract_summary",
]
