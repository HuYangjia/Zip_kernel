"""V9 offline weight packing utilities.

Provides:
- Shared block-shape constants BROW / BCOL (both 128, aligned with group_size)
- 4-bit little-endian pack / unpack helpers
- `pack_v9_weights(...)` : convert GPTQ submatrix-mixed outputs into an
  inference-ready container (dense low-bit layer + 2D block-sparse high-bit
  layer in BSR format).

All runtime-visible strings are English per project convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Shared block-shape constants.  Hard coded to 128 per requirements 1.7 / 7.3.
# ---------------------------------------------------------------------------
BROW: int = 128
BCOL: int = 128


# ---------------------------------------------------------------------------
# 4-bit little-endian pack / unpack helpers (Python-side, torch tensors)
# ---------------------------------------------------------------------------

def pack_s4_le(tensor: torch.Tensor) -> torch.Tensor:
    """Pack a 4-bit integer tensor along the last dim with little-endian layout.

    Layout: byte[i] = (x[2i+1] << 4) | (x[2i] & 0x0F)

    Accepts both UINT4 values in [0, 15] and SINT4 values in [-8, 7]; the
    two's-complement pattern is preserved by masking with 0x0F.

    Args:
        tensor: int8 / int16 / int32 tensor whose last dim is even-sized.

    Returns:
        int8 tensor with last dim halved (packed bytes).
    """
    if tensor.shape[-1] % 2 != 0:
        raise ValueError(
            f"pack_s4_le requires last dim to be even, got {tensor.shape[-1]}"
        )
    x = tensor.to(torch.int32)
    low = x[..., 0::2] & 0x0F
    high = x[..., 1::2] & 0x0F
    packed = (high << 4) | low                         # int32 in [0, 255]
    # Cast to int8 relies on PyTorch's natural wrap-around (0..255 -> [-128, 127])
    # which is cheaper than an explicit torch.where.
    return packed.to(torch.int8)


def unpack_s4_le(packed: torch.Tensor, signed: bool = True) -> torch.Tensor:
    """Inverse of `pack_s4_le`.

    Args:
        packed: int8 tensor produced by `pack_s4_le`.
        signed: if True return SINT4 values in [-8, 7]; else UINT4 in [0, 15].

    Returns:
        int8 tensor with last dim doubled.
    """
    p = packed.to(torch.int32) & 0xFF
    low = p & 0x0F
    high = (p >> 4) & 0x0F
    out = torch.empty(*p.shape[:-1], p.shape[-1] * 2, dtype=torch.int32,
                      device=p.device)
    out[..., 0::2] = low
    out[..., 1::2] = high
    if signed:
        out = torch.where(out >= 8, out - 16, out)
    return out.to(torch.int8)


# ---------------------------------------------------------------------------
# W container
# ---------------------------------------------------------------------------

@dataclass
class V9WeightContainer:
    """Inference-ready V9 weight container."""

    W_low_packed: torch.Tensor          # (d_out, d_in // 2) int8
    W_high_blocks_packed: torch.Tensor  # (n_hp_blocks, brow, bcol // 2) int8
    scale_u4: torch.Tensor              # (d_out, n_groups) fp16
    zero_u4: torch.Tensor               # (d_out, n_groups) fp16  (pre-subtracted 8)
    hp_row_offsets: torch.Tensor        # (nrow + 1,) int32  (BSR indptr)
    hp_col_indices: torch.Tensor        # (n_hp_blocks,) int32
    perm: torch.Tensor                  # (d_in,) int32
    block_shape: Tuple[int, int]        # (brow, bcol)
    d_out: int
    d_in: int

    @property
    def n_hp_blocks(self) -> int:
        return int(self.W_high_blocks_packed.shape[0])

    @property
    def n_groups(self) -> int:
        return int(self.scale_u4.shape[1])


# ---------------------------------------------------------------------------
# Core packing routine
# ---------------------------------------------------------------------------

def pack_v9_weights(
    gptq_outputs: Dict[str, torch.Tensor],
    brow: int = BROW,
    bcol: int = BCOL,
) -> V9WeightContainer:
    """Pack GPTQ submatrix-mixed outputs into a True-Quant inference container.

    Expected keys in `gptq_outputs` (see kernel_algorithm.md sec 6.1):
        - Q_u4_permuted       : (d_out, d_in) int8/int32, values in [0, 15]
        - scale_u4_raw        : (d_out, n_groups) fp16/fp32
        - zero_u4_raw         : (d_out, n_groups) fp16/fp32 (values in [0, 15])
        - Q_s8_blocks         : (n_hp_blocks, brow, bcol) int8, values in [-128, 127]
        - scale_s8_per_block  : (n_hp_blocks, brow) fp16/fp32
        - hp_block_indices    : (n_hp_blocks, 2) int32  (br, bc) pairs
        - perm                : (d_in,) int32

    The container produced is consumed by the three Triton kernels
    (activation_quant, dense_u4s4_gemm, sparse_s4s4_gemm).

    Design notes
    ------------
    * UINT4 weights and zero are pre-subtracted by 8 so that the online GEMM
      runs a uniform SINT4 x SINT4 MMA (see `triton_kernel_prompt.md` sec 1.4).
      The dequant epilogue is algebraically unchanged because the offset is
      absorbed into `zero_u4` once and for all.
    * High-precision blocks are reordered into BSR layout: sorted by `br`
      ascending, ties broken by `bc` ascending. `hp_row_offsets[br]` is the
      CSR-style indptr into `hp_col_indices` / `W_high_blocks_packed`.
    """

    if brow != BROW or bcol != BCOL:
        raise ValueError(
            f"V9 pack requires brow == bcol == 128, got brow={brow}, bcol={bcol}"
        )

    # ---- extract & sanity-check -------------------------------------------------
    Q_u4_permuted: torch.Tensor = gptq_outputs["Q_u4_permuted"]
    scale_u4_raw: torch.Tensor = gptq_outputs["scale_u4_raw"]
    zero_u4_raw: torch.Tensor = gptq_outputs["zero_u4_raw"]
    Q_s8_blocks: torch.Tensor = gptq_outputs["Q_s8_blocks"]
    scale_s8_per_block: torch.Tensor = gptq_outputs["scale_s8_per_block"]
    hp_block_indices: torch.Tensor = gptq_outputs["hp_block_indices"]
    perm: torch.Tensor = gptq_outputs["perm"]

    d_out, d_in = int(Q_u4_permuted.shape[0]), int(Q_u4_permuted.shape[1])
    n_groups = (d_in + bcol - 1) // bcol
    nrow = (d_out + brow - 1) // brow

    if d_in % bcol != 0:
        # group_size == bcol enforced upstream; tail padding not supported by packer.
        raise ValueError(
            f"V9 pack requires d_in ({d_in}) divisible by bcol ({bcol})"
        )
    if scale_u4_raw.shape != (d_out, n_groups):
        raise ValueError(
            f"scale_u4_raw expected shape ({d_out}, {n_groups}), got {tuple(scale_u4_raw.shape)}"
        )
    if zero_u4_raw.shape != (d_out, n_groups):
        raise ValueError(
            f"zero_u4_raw expected shape ({d_out}, {n_groups}), got {tuple(zero_u4_raw.shape)}"
        )

    n_hp = int(hp_block_indices.shape[0]) if hp_block_indices.numel() > 0 else 0

    device = Q_u4_permuted.device

    # ---- initialize W_low / scale / zero ---------------------------------------
    W_low = Q_u4_permuted.clone().to(torch.int32)    # values in [0, 15]
    scale_u4 = scale_u4_raw.clone().to(torch.float16)
    zero_u4 = zero_u4_raw.clone().to(torch.float16)

    # ---- bit-split SINT8 blocks into W_low (low) + W_high (high) ---------------
    # Collect (br, bc, Q_high_block) tuples so we can reorder into BSR afterwards.
    high_blocks: List[Tuple[int, int, torch.Tensor]] = []

    hp_idx_cpu = hp_block_indices.to(torch.int64).cpu().numpy() if n_hp > 0 else np.zeros((0, 2), dtype=np.int64)

    for idx in range(n_hp):
        br = int(hp_idx_cpu[idx, 0])
        bc = int(hp_idx_cpu[idx, 1])
        r0, r1 = br * brow, min((br + 1) * brow, d_out)
        c0, c1 = bc * bcol, min((bc + 1) * bcol, d_in)

        q_s8 = Q_s8_blocks[idx, : r1 - r0, : c1 - c0].to(torch.int32)
        # Arithmetic right shift for SINT8 values
        q_high = q_s8 >> 4                  # in [-8, 7]
        q_low = q_s8 & 0x0F                 # in [0, 15]

        # Self-check: bit-split is lossless.
        assert torch.equal(q_high * 16 + q_low, q_s8), (
            f"bit-split mismatch at block idx={idx} (br={br}, bc={bc})"
        )

        # Write low bits into W_low
        W_low[r0:r1, c0:c1] = q_low

        # Prepare a (brow, bcol) tile (pad with 0 for tail blocks) and stash.
        tile_high = torch.zeros((brow, bcol), dtype=torch.int32, device=device)
        tile_high[: r1 - r0, : c1 - c0] = q_high
        high_blocks.append((br, bc, tile_high))

        # Overwrite scale_u4 / zero_u4 for this SINT8 group.
        # scale_s8_per_block[idx] has shape (brow,) (row-wise s8 scale for this block).
        s_s8 = scale_s8_per_block[idx, : r1 - r0].to(torch.float16)
        scale_u4[r0:r1, bc] = s_s8
        zero_u4[r0:r1, bc] = 0.0

    # ---- pre-subtract 8 (UINT4 -> SINT4 for MMA) -------------------------------
    # W_low stays in [0, 15] logically; subtract 8 so online MMA sees SINT4.
    # zero_u4 is subtracted by the same 8 so the epilogue dequant is unchanged:
    #   (q_u4 - zero_u4) * scale == ((q_u4 - 8) - (zero_u4 - 8)) * scale
    W_low_s4 = (W_low - 8).to(torch.int8)                   # in [-8, 7]
    zero_u4 = (zero_u4 - 8.0).to(torch.float16)             # fp16

    # ---- BSR reorder of high-precision blocks ---------------------------------
    if n_hp > 0:
        high_blocks.sort(key=lambda t: (t[0], t[1]))
        br_sorted = torch.tensor([b[0] for b in high_blocks], dtype=torch.int32, device=device)
        bc_sorted = torch.tensor([b[1] for b in high_blocks], dtype=torch.int32, device=device)
        W_high_tiles = torch.stack([b[2].to(torch.int8) for b in high_blocks], dim=0)

        # Build indptr of length nrow + 1.
        hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
        counts = torch.bincount(br_sorted.to(torch.int64), minlength=nrow)
        hp_row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
        assert int(hp_row_offsets[-1].item()) == n_hp, (
            f"hp_row_offsets[-1] ({int(hp_row_offsets[-1].item())}) != n_hp ({n_hp})"
        )

        hp_col_indices = bc_sorted
    else:
        W_high_tiles = torch.zeros((0, brow, bcol), dtype=torch.int8, device=device)
        hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
        hp_col_indices = torch.zeros((0,), dtype=torch.int32, device=device)

    # ---- 4-bit little-endian packing -------------------------------------------
    # Pack directly on the source device to avoid HBM <-> CPU round-trips.
    W_low_packed = pack_s4_le(W_low_s4)                     # (d_out, d_in//2) int8
    W_high_blocks_packed = pack_s4_le(W_high_tiles)         # (n_hp, brow, bcol//2)

    return V9WeightContainer(
        W_low_packed=W_low_packed.to(device),
        W_high_blocks_packed=W_high_blocks_packed.to(device),
        scale_u4=scale_u4.to(device),
        zero_u4=zero_u4.to(device),
        hp_row_offsets=hp_row_offsets,
        hp_col_indices=hp_col_indices,
        perm=perm.to(torch.int32).to(device),
        block_shape=(brow, bcol),
        d_out=d_out,
        d_in=d_in,
    )


__all__ = [
    "BROW",
    "BCOL",
    "pack_s4_le",
    "unpack_s4_le",
    "V9WeightContainer",
    "pack_v9_weights",
]
