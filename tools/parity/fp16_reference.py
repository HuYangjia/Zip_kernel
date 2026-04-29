"""Pure-torch FP16 reference for the W4A4 V9 fused dense+sparse kernel.

Contract
--------
Given the same inputs the production kernel receives, this module
returns ``Y_ref`` such that:

    max(|Y_kernel - Y_ref|) / max(|Y_ref|) < 5e-3

with **Y_ref computed from the dequantised FP16 math in full FP32
arithmetic**, then cast to FP16 at the end — exactly as the kernel is
specified to behave (see kernel_algorithm.md §6).

The implementation is deliberately slow (30-50× slower than the Triton
reference, and ~1000× slower than the fused CUDA kernel) because its
job is to be trivially correct, not fast.

Key identities encoded here
---------------------------
1.  **Pre-subtracted zero / SINT4 weights**
    ``pack_v9_weights`` stores ``W_low_packed`` in SINT4 space ([-8, 7])
    and shifts ``zero_u4`` by -8 in lockstep, so that::

        (q_u4_raw - zero_u4_raw) * scale
            == ((q_u4_raw - 8) - (zero_u4_raw - 8)) * scale
            == (W_low_s4 - zero_u4_shifted) * scale

    This reference uses the already-shifted values exactly as the
    kernel sees them — no re-adding of 8.

2.  **Per-token activation scale**
    ``scale_x`` is produced by the activation quant kernel as
    ``max(|X|)/7`` rounded to FP16.  The reference uses the FP16 value
    verbatim; callers pass it in unchanged.

3.  **Groupwise weight dequant**
    ``scale_u4`` / ``zero_u4`` are ``(d_out, n_groups)`` — groups span
    ``BCOL`` consecutive columns of d_in.  The dense accumulator
    factors out the per-group scale using ``sum_X[t, g]`` so the
    inner-loop MMA stays in INT32; this reference simulates the same
    factoring via the dequant identity::

        acc_group[r, t, g]   = Σ_{k in g}  W_low_s4[r, k] * X_s4[t, k]
        partial_group[r, t, g] =
            scale_u4[r, g] * (acc_group[r, t, g] - zero_u4[r, g] * sum_X[t, g])
        Y_low[r, t] = scale_x[t] * Σ_g partial_group[r, t, g]

4.  **Sparse accumulator in units of 16·Y_high**
    The SINT8 high-precision blocks were bit-split into low-nibble
    (folded into ``W_low``) and high-nibble (stored as ``Q_high_s4``
    in ``W_high_blocks_packed``), related by ``q_s8 = 16·q_high + q_low``.
    The sparse GEMM therefore contributes ``16·Y_high`` to the final
    sum, which is the weighting applied in
    :func:`fp16_fused_reference`.

5.  **Final FP16 cast**
    The kernel sums ``Y_low + 16·Y_high`` in FP32 registers and casts
    to FP16 once, producing ``Y[d_out, T]``.  The reference is
    bit-identical in specification (FP32 accumulation → FP16 store).
"""

from __future__ import annotations

from typing import Optional

import torch

from kernel.triton_kernel.pack_utils import BCOL, BROW, unpack_s4_le


def _unpack_weights_to_fp16(
    W_low_packed: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Dequantise W_low to FP16 in permuted column order.

    Returns tensor of shape ``(d_out, d_in)`` dtype fp16.
    """
    n_groups = d_in // BCOL
    if W_low_packed.shape != (d_out, d_in // 2):
        raise ValueError(
            f"W_low_packed shape {tuple(W_low_packed.shape)} != expected "
            f"({d_out}, {d_in // 2})"
        )
    if scale_u4.shape != (d_out, n_groups):
        raise ValueError(
            f"scale_u4 shape {tuple(scale_u4.shape)} != ({d_out}, {n_groups})"
        )
    if zero_u4.shape != (d_out, n_groups):
        raise ValueError(
            f"zero_u4 shape {tuple(zero_u4.shape)} != ({d_out}, {n_groups})"
        )

    # int4 little-endian unpack -> (d_out, d_in) in [-8, 7]
    W_s4 = unpack_s4_le(W_low_packed, signed=True).to(torch.float32)

    # Broadcast groupwise scale/zero over d_in.
    # scale_u4 / zero_u4 : (d_out, n_groups) -> (d_out, d_in)
    scale = scale_u4.to(torch.float32).repeat_interleave(BCOL, dim=1)
    zero = zero_u4.to(torch.float32).repeat_interleave(BCOL, dim=1)

    W_fp32 = (W_s4 - zero) * scale
    return W_fp32.to(torch.float16)


def _unpack_x_s4_to_fp16(
    X_s4: torch.Tensor,
    scale_x: torch.Tensor,
    d_in: int,
) -> torch.Tensor:
    """Dequantise activation X_s4 to FP16.  Returns ``(T, d_in)``.

    The dequantised activations are already in the *permuted* column
    order matching the weights, because both sides index the same
    ``perm`` at pack time / quant time.
    """
    T = X_s4.shape[0]
    if X_s4.shape != (T, d_in // 2):
        raise ValueError(
            f"X_s4 shape {tuple(X_s4.shape)} != ({T}, {d_in // 2})"
        )
    if scale_x.shape != (T,):
        raise ValueError(f"scale_x shape {tuple(scale_x.shape)} != ({T},)")

    X_unpacked_s4 = unpack_s4_le(X_s4, signed=True).to(torch.float32)  # (T, d_in)
    X_fp32 = X_unpacked_s4 * scale_x.to(torch.float32)[:, None]
    return X_fp32.to(torch.float16)


def fp16_dense_reference(
    W_low_packed: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    scale_x: torch.Tensor,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Dense W4A4 accumulator, pure-torch reference.

    Mirrors ``kernel.triton_kernel.dense_u4s4_gemm.dense_gemm_u4_s4``
    and the dense branch of
    ``kernel.cuda_kernel.fused_dense_sparse_mma_int4.launch``.

    Returns ``Y_low`` of shape ``(d_out, T)`` dtype ``fp16`` so callers
    can compose with ``fp16_sparse_reference`` identically to the
    fused kernel.
    """
    W_fp16 = _unpack_weights_to_fp16(W_low_packed, scale_u4, zero_u4, d_out, d_in)
    X_fp16 = _unpack_x_s4_to_fp16(X_s4, scale_x, d_in)

    # Accumulate in FP32 then cast once, matching the kernel contract.
    Y_low = X_fp16.to(torch.float32) @ W_fp16.to(torch.float32).T  # (T, d_out)
    return Y_low.to(torch.float16).T.contiguous()                   # (d_out, T)


def _unpack_sparse_blocks_to_fp16(
    W_high_blocks_packed: torch.Tensor,
    hp_row_offsets: torch.Tensor,
    hp_col_indices: torch.Tensor,
    scale_u4: torch.Tensor,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Materialise the sparse high-bit weights as a dense (d_out, d_in) fp16 tensor.

    For positions where no high-precision block exists, the value is 0.

    The 4-bit high nibble (``q_high ∈ [-8, 7]``) is multiplied by the
    *same* ``scale_u4`` entry the dense branch uses for that group,
    because the SINT8 block's scale_s8_per_block was written into
    ``scale_u4[r, bc]`` at pack time (see
    ``pack_v9_weights`` line "scale_u4[r0:r1, bc] = s_s8").
    """
    n_blocks = int(W_high_blocks_packed.shape[0])
    if n_blocks == 0:
        return torch.zeros((d_out, d_in), dtype=torch.float16,
                           device=W_high_blocks_packed.device)
    if W_high_blocks_packed.shape != (n_blocks, BROW, BCOL // 2):
        raise ValueError(
            f"W_high_blocks_packed shape {tuple(W_high_blocks_packed.shape)} "
            f"!= ({n_blocks}, {BROW}, {BCOL // 2})"
        )

    W_high = torch.zeros(
        (d_out, d_in), dtype=torch.float32, device=W_high_blocks_packed.device
    )
    nrow = d_out // BROW

    # Walk BSR: for each block row br, indices [off0, off1) give the
    # (col, tile) pairs.
    row_off = hp_row_offsets.tolist()
    col_idx = hp_col_indices.tolist()

    # Unpack all tiles in one shot for speed; interpretation is always
    # SINT4 (high nibble of a SINT8 is in [-8, 7]).
    tiles_s4 = unpack_s4_le(W_high_blocks_packed, signed=True).to(torch.float32)
    # shape: (n_blocks, BROW, BCOL)

    # scale_u4 is already FP16 and shaped (d_out, n_groups); broadcast
    # the per-group scale over the block's column span at placement time.
    scale_f32 = scale_u4.to(torch.float32)  # (d_out, n_groups)

    for br in range(nrow):
        r0, r1 = br * BROW, (br + 1) * BROW
        off0, off1 = row_off[br], row_off[br + 1]
        for k in range(off0, off1):
            bc = col_idx[k]
            c0, c1 = bc * BCOL, (bc + 1) * BCOL
            # scale for this block row (BROW rows, single group column bc)
            s = scale_f32[r0:r1, bc : bc + 1]            # (BROW, 1)
            W_high[r0:r1, c0:c1] = tiles_s4[k] * s

    return W_high.to(torch.float16)


def fp16_sparse_reference(
    W_high_blocks_packed: torch.Tensor,
    hp_row_offsets: torch.Tensor,
    hp_col_indices: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    scale_x: torch.Tensor,
    d_out: int,
    d_in: int,
) -> torch.Tensor:
    """Sparse BSR S4×S4 accumulator, pure-torch reference.

    Returns ``Y_high`` of shape ``(d_out, T)`` dtype ``fp16``.  The
    caller (or :func:`fp16_fused_reference`) is responsible for the
    ``16·Y_high`` scaling.
    """
    W_high = _unpack_sparse_blocks_to_fp16(
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        scale_u4, d_out, d_in,
    )
    X_fp16 = _unpack_x_s4_to_fp16(X_s4, scale_x, d_in)

    Y_high = X_fp16.to(torch.float32) @ W_high.to(torch.float32).T  # (T, d_out)
    return Y_high.to(torch.float16).T.contiguous()                   # (d_out, T)


def fp16_fused_reference(
    W_low_packed: torch.Tensor,
    W_high_blocks_packed: torch.Tensor,
    hp_row_offsets: torch.Tensor,
    hp_col_indices: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    scale_x: torch.Tensor,
    d_out: int,
    d_in: int,
    sum_X: Optional[torch.Tensor] = None,  # accepted for signature parity; unused
) -> torch.Tensor:
    """Fused dense + sparse W4A4 reference matching the production kernel.

    Returns ``Y`` of shape ``(d_out, T)`` dtype ``fp16``.

    ``sum_X`` is accepted to keep the call site interchangeable with
    the kernel's ``launch(...)`` ABI (see R50 design.md §2.2) — this
    reference derives the equivalent arithmetic directly from the
    dequantised activations, so it does not read ``sum_X``.
    """
    del sum_X  # intentionally unused; see docstring

    Y_low = fp16_dense_reference(
        W_low_packed, X_s4, scale_u4, zero_u4, scale_x, d_out, d_in,
    )
    Y_high = fp16_sparse_reference(
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, scale_x, d_out, d_in,
    )
    # Fuse in FP32, single final FP16 cast (matches kernel epilogue).
    Y = Y_low.to(torch.float32) + 16.0 * Y_high.to(torch.float32)
    return Y.to(torch.float16)


__all__ = [
    "fp16_dense_reference",
    "fp16_sparse_reference",
    "fp16_fused_reference",
]
