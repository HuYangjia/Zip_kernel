"""W4A4 four-op callable factory for the *naive* CUDA backend.

Sibling of ``qwen3_w4a4_ops.py`` but uses ``kernel.cuda_kernel.ops_naive``
which exposes the textbook reference kernels (activation_quant_naive,
dense_gemm_naive, sparse_gemm_naive, reduce_sum_naive) instead of the
optimised Tensor-Core pipeline.

Key differences vs. the optimised factory:

  * The callable for each fused-projection group launches FOUR kernels
    in sequence (quant → dense → sparse → reduce_sum), because the
    naive backend does not fuse.  The per-op timing this factory
    reports is therefore the wall time of all four naive kernels for
    ONE projection call — directly comparable to the optimised
    ``legacy_mma`` two-kernel path on the same (T, d_in, d_out) shape.

  * Sparse path: we build a ~5% block-sparse BSR for every shape (not
    empty as in the optimised bench).  Density is fixed at 5% so the
    sparse kernel actually launches with real work and the naive
    vs. optimised comparison covers the full 4-step pipeline.
    (Shapes stay shape-legal: d_out % 128 == 0, d_in % 128 == 0.)

  * All tensor layouts (W_low uint4 pack, W_high_blocks BSR,
    scale_u4/zero_u4, hp_row_offsets/hp_col_indices, perm) are
    byte-identical to the optimised factory, so a parity test can
    feed the same inputs to both backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from kernel.cuda_kernel import ops_naive as naive_ops
from kernel.triton_kernel.pack_utils import BCOL
from kernel.bench.configs.qwen3_shapes import (
    Qwen3Config, enumerate_fused_projs,
)

# -----------------------------------------------------------------------------
# Fixed 5% block sparsity target.  "~5%" rounded to nearest whole block.
# -----------------------------------------------------------------------------
_SPARSITY_PCT = 0.05


@dataclass
class NaiveOpBundle:
    """All tensors needed to launch one naive 4-kernel projection."""
    name: str
    d_in: int
    d_out: int
    T: int

    X_fp16: torch.Tensor       # [T, d_in] fp16 — input activation
    perm: torch.Tensor         # [d_in] int32 — permutation for quant

    W_low: torch.Tensor        # [d_out, d_in//2] int8 UINT4 packed
    scale_u4: torch.Tensor     # [d_out, n_groups] fp16
    zero_u4: torch.Tensor      # [d_out, n_groups] fp16

    W_high_blocks: torch.Tensor   # [n_blocks, 128, BCOL/2] int8 SINT4
    hp_row_offsets: torch.Tensor  # [d_out//128 + 1] int32
    hp_col_indices: torch.Tensor  # [n_blocks] int32


def _build_sparse_bsr(
    d_out: int, d_in: int, *, density: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a ~`density` block-sparse BSR with shape-legal indices.

    Returns (W_high_blocks, hp_row_offsets, hp_col_indices).

    The BSR grid is (d_out/128) row-blocks × (d_in/BCOL) col-groups; we
    sample each cell independently with probability ``density`` (no
    duplicates per row).  For Qwen3 shapes with n_groups in [20, 136],
    this yields 1–7 blocks per row-block, giving 4-kernel pipelines
    that exercise sparse while staying well inside budget.
    """
    n_row_blocks = d_out // 128
    n_groups     = d_in  // BCOL

    # Generate a boolean mask on CPU for determinism; small enough.
    gen = torch.Generator(device="cpu").manual_seed(
        (d_out * 2654435761 ^ d_in) & 0x7FFFFFFF
    )
    mask = torch.rand(
        (n_row_blocks, n_groups), generator=gen, dtype=torch.float32
    ) < density

    # Ensure at least 1 block per row-block so every sparse CTA has
    # real work (matches the "exercise the kernel" goal).
    for r in range(n_row_blocks):
        if not mask[r].any():
            c = int(torch.randint(
                0, n_groups, (1,), generator=gen, dtype=torch.int64
            ).item())
            mask[r, c] = True

    row_offsets = torch.zeros(n_row_blocks + 1, dtype=torch.int32)
    col_indices_list: list[int] = []
    for r in range(n_row_blocks):
        cs = torch.nonzero(mask[r], as_tuple=False).flatten().to(torch.int32)
        col_indices_list.append(cs)
        row_offsets[r + 1] = row_offsets[r] + len(cs)
    col_indices = torch.cat(col_indices_list) if col_indices_list else \
                  torch.zeros(0, dtype=torch.int32)
    n_blocks = int(row_offsets[-1].item())

    # Random SINT4-packed block payloads (range -8..7).  We store as
    # uint8 bytes with two nibbles per byte.
    W_high = torch.randint(
        0, 256, (n_blocks, 128, BCOL // 2), dtype=torch.int64,
        device="cpu", generator=gen,
    ).to(torch.int8)

    return (
        W_high.to(device).contiguous(),
        row_offsets.to(device).contiguous(),
        col_indices.to(device).contiguous(),
    )


def _build_bundle(
    d_in: int, d_out: int, T: int, *,
    device: torch.device, name: str,
) -> NaiveOpBundle:
    if d_in % BCOL != 0:
        raise ValueError(f"{name}: d_in {d_in} not divisible by BCOL {BCOL}")
    if d_out % 128 != 0:
        raise ValueError(f"{name}: d_out {d_out} not divisible by 128")

    X_fp16 = (torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
              ).contiguous()
    perm = torch.randperm(d_in, device=device).to(torch.int32).contiguous()

    # UINT4-packed weights
    W_low = torch.randint(
        0, 16, (d_out, d_in // 2), dtype=torch.int8, device=device
    ).contiguous()

    n_groups = d_in // BCOL
    scale_u4 = (torch.rand(d_out, n_groups, dtype=torch.float16, device=device)
                * 0.01 + 0.001).contiguous()
    zero_u4 = (torch.rand(d_out, n_groups, dtype=torch.float16, device=device)
               * 14.0).contiguous()

    W_high_blocks, hp_row_offsets, hp_col_indices = _build_sparse_bsr(
        d_out, d_in, density=_SPARSITY_PCT, device=device,
    )

    return NaiveOpBundle(
        name=name, d_in=d_in, d_out=d_out, T=T,
        X_fp16=X_fp16, perm=perm,
        W_low=W_low, scale_u4=scale_u4, zero_u4=zero_u4,
        W_high_blocks=W_high_blocks,
        hp_row_offsets=hp_row_offsets,
        hp_col_indices=hp_col_indices,
    )


def _make_callable(b: NaiveOpBundle) -> Callable[[], torch.Tensor]:
    """Zero-arg lambda that launches the full naive 4-kernel pipeline."""
    d_in, d_out = b.d_in, b.d_out

    def fn():
        # K1: quant
        X_s4, scale_x, sum_X = naive_ops.activation_quant_naive(
            b.X_fp16, b.perm
        )
        # K2: dense GEMM
        Y_low = naive_ops.dense_gemm_naive(
            b.W_low, X_s4, b.scale_u4, b.zero_u4, sum_X, scale_x
        )
        # K3: sparse GEMM
        Y_high = naive_ops.sparse_gemm_naive(
            b.W_high_blocks, b.hp_row_offsets, b.hp_col_indices,
            X_s4, b.scale_u4, scale_x, d_out, d_in,
        )
        # K4: reduce sum
        return naive_ops.reduce_sum_naive(Y_low, Y_high)

    return fn


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
@dataclass
class NaiveFourOp:
    qkv_fused:     tuple[NaiveOpBundle, Callable[[], torch.Tensor]]
    o_proj:        tuple[NaiveOpBundle, Callable[[], torch.Tensor]]
    gate_up_fused: tuple[NaiveOpBundle, Callable[[], torch.Tensor]]
    down_proj:     tuple[NaiveOpBundle, Callable[[], torch.Tensor]]

    def as_list(self) -> list[tuple[str, NaiveOpBundle,
                                    Callable[[], torch.Tensor]]]:
        return [
            ("qkv_fused",     *self.qkv_fused),
            ("o_proj",        *self.o_proj),
            ("gate_up_fused", *self.gate_up_fused),
            ("down_proj",     *self.down_proj),
        ]


def build_four_op_callables_naive(
    cfg: Qwen3Config, *, batch: int, seqlen: int,
    device: torch.device | str = "cuda",
) -> NaiveFourOp:
    """Build the naive 4-kernel pipeline for each of the 4 projection groups."""
    dev = torch.device(device)
    T = batch * seqlen
    bundles: dict[str, NaiveOpBundle] = {}
    for proj in enumerate_fused_projs(cfg):
        bundles[proj.proj] = _build_bundle(
            d_in=proj.d_in, d_out=proj.d_out, T=T,
            device=dev, name=proj.proj,
        )
    return NaiveFourOp(
        qkv_fused     = (bundles["qkv_fused"],     _make_callable(bundles["qkv_fused"])),
        o_proj        = (bundles["o"],             _make_callable(bundles["o"])),
        gate_up_fused = (bundles["gate_up_fused"], _make_callable(bundles["gate_up_fused"])),
        down_proj     = (bundles["down"],          _make_callable(bundles["down"])),
    )


__all__ = [
    "NaiveOpBundle",
    "NaiveFourOp",
    "build_four_op_callables_naive",
]
