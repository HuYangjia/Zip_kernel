"""Qwen3 → Atom W4A4 GEMM shape enumeration (4-fused convention).

The Atom kernel exposes a single ``DenseLayerGEMM_i4_o16(M, N, K)`` op.
For a Qwen3 layer in the **4-fused** decomposition (see MODEL_SELECTION.md
§3) we need to time the following four shapes per (model, phase, batch):

  * ``qkv_fused``     :  d_in = hidden,         d_out = q_out + 2 * kv_out
  * ``o_proj``        :  d_in = q_out,          d_out = hidden
  * ``gate_up_fused`` :  d_in = hidden,         d_out = 2 * intermediate
  * ``down_proj``     :  d_in = intermediate,   d_out = hidden

For each shape the Atom GEMM mapping is:

  * ``M`` = batch * seqlen     (rows of the activation matrix)
  * ``K`` = d_in               (the contracted dimension)
  * ``N`` = d_out              (the output feature count)

With keeper_size = 128 fixed, the Atom kernel will internally split
``K`` into ``(K - 128)`` INT4 channels + 128 INT8 outlier channels.
We therefore require ``K > 128 + group_size = 256`` for every shape;
this is satisfied by all 3 selected models (smallest hidden = 2560).

We deliberately re-export the *fused* ProjShape list rather than
re-deriving it here so there is exactly one source of truth for the
shape numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernel.bench.configs.qwen3_shapes import (
    PhaseConfig,
    Qwen3Config,
    enumerate_fused_projs,
)


# Atom kernel constants (mirror src/GEMM/Dense_layer_gemm_i4_o16.cuh).
# These are the *only* numbers the bench driver may depend on; do not
# duplicate them inside individual scripts.
ATOM_GROUP_SIZE: int = 128
ATOM_KEEPER_SIZE: int = 128
ATOM_MIN_K: int = ATOM_GROUP_SIZE + ATOM_KEEPER_SIZE  # 256, smallest valid K


@dataclass(frozen=True)
class AtomGemmShape:
    """One GEMM as the Atom kernel sees it.

    Fields
    ------
    proj : str
        One of ``qkv_fused`` / ``o`` / ``gate_up_fused`` / ``down``.
        Echoed in JSON output so consumers can join with the BF16 bench.
    M, N, K : int
        The Atom kernel's (rows, output_cols, contracted) dims.
    d_in, d_out : int
        The original Linear's in/out features (== K, N).  Kept separate
        for readability of the JSON output.
    """
    proj: str
    M: int
    N: int
    K: int
    d_in: int
    d_out: int

    def assert_valid_for_atom(self) -> None:
        """Raise ValueError if the shape is unsupported by Atom kernel.

        Constraints come from the kernel's tile sizes (16 in M/N) and
        the keeper/group split (K must be ≥ 256 and (K - 128) divisible
        by group_size = 128).
        """
        if self.K < ATOM_MIN_K:
            raise ValueError(
                f"{self.proj}: K={self.K} < {ATOM_MIN_K} "
                f"(group_size + keeper_size); Atom kernel cannot run."
            )
        if self.K % ATOM_GROUP_SIZE != 0:
            # Required for the rmsnorm_fp16_i4 / reorder_fp16_i4 / activate_fp16_i4
            # quant kernels — they hardcode group_size=128 along K.
            raise ValueError(
                f"{self.proj}: K={self.K} not divisible by "
                f"group_size={ATOM_GROUP_SIZE}; quant kernels cannot run."
            )
        if (self.K - ATOM_KEEPER_SIZE) % ATOM_GROUP_SIZE != 0:
            raise ValueError(
                f"{self.proj}: (K - keeper)={self.K - ATOM_KEEPER_SIZE} "
                f"not divisible by group_size={ATOM_GROUP_SIZE}."
            )
        if self.M % 16 != 0:
            raise ValueError(
                f"{self.proj}: M={self.M} not divisible by 16; "
                "Atom mma tile is 16 along M."
            )
        if self.N % 16 != 0:
            raise ValueError(
                f"{self.proj}: N={self.N} not divisible by 16; "
                "Atom mma tile is 16 along N."
            )


def enumerate_atom_shapes(
    cfg: Qwen3Config,
    phase: PhaseConfig,
    batch: int,
) -> list[AtomGemmShape]:
    """Return the 4 fused GEMM shapes for one (model, phase, batch) point.

    ``M = batch * seqlen``.  Caller is responsible for asserting validity
    (use ``shape.assert_valid_for_atom()``); the enumeration step itself
    does not raise so that the driver can collect *all* invalid shapes
    in one pass and report them together in VALIDATION_LOG.
    """
    M = batch * phase.seqlen
    fused = enumerate_fused_projs(cfg)
    out: list[AtomGemmShape] = []
    for p in fused:
        out.append(
            AtomGemmShape(
                proj=p.proj,
                M=M,
                N=p.d_out,
                K=p.d_in,
                d_in=p.d_in,
                d_out=p.d_out,
            )
        )
    return out


__all__ = [
    "ATOM_GROUP_SIZE",
    "ATOM_KEEPER_SIZE",
    "ATOM_MIN_K",
    "AtomGemmShape",
    "enumerate_atom_shapes",
]
