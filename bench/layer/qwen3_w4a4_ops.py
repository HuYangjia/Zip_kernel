"""W4A4 fused-op callable factory for the r79 replacement bench.

Covers the 4 fused projection groups that the W4A4 CUDA kernel replaces
in production inference:

  * ``qkv_fused``      : d_in=hidden,       d_out=q_out + 2*kv_out
  * ``o_proj``         : d_in=q_out,        d_out=hidden
  * ``gate_up_fused``  : d_in=hidden,       d_out=2*intermediate
  * ``down_proj``      : d_in=intermediate, d_out=hidden

Dispatch policy (mirrors production ``fused_dense_sparse_e2e_cuda`` with
``HKUST_V9_P0_MODE=0``; see ``cuda_kernel/ops.py``):

  * T == 1                     → ``fused_quant_gemv_cuda``        (path=gemv)
  * T >= 2, pure dense         → ``activation_quant_cuda`` +
                                 ``fused_dense_sparse_cuda_int4``  (path=legacy_mma)

Weights / scale / zero / perm are random but *shape-legal* so the kernel
launches successfully; we never verify logits — this is a timing harness
only. Weight pack format follows
``cuda_kernel/tests/perf_fused_quant.py`` verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from kernel.cuda_kernel import ops as w4a4_ops
from kernel.triton_kernel.pack_utils import BCOL
from kernel.bench.configs.qwen3_shapes import Qwen3Config, enumerate_fused_projs


# -----------------------------------------------------------------------------
# Per-shape weight bundle (fully pre-allocated; reused by every timing iter)
# -----------------------------------------------------------------------------
@dataclass
class W4A4OpBundle:
    """All tensors needed to launch one fused W4A4 projection kernel."""
    name: str               # "qkv_fused" / "o" / "gate_up_fused" / "down"
    d_in: int
    d_out: int
    T: int                  # batch * seqlen for this (phase, bs) combo
    path: str               # "legacy_mma" | "gemv"

    X_fp16: torch.Tensor    # [T, d_in] fp16 — only used by gemv / P0 paths
    perm: torch.Tensor      # [d_in] int32
    W_low: torch.Tensor     # [d_out, d_in//2] int8, 4-bit packed
    W_high_blocks: torch.Tensor     # [0, 128, BCOL//2] int8 (sparse disabled)
    hp_row_offsets: torch.Tensor    # [d_out//128 + 1] int32 (all zeros)
    hp_col_indices: torch.Tensor    # [0] int32
    scale_u4: torch.Tensor  # [d_out, n_groups] fp16
    zero_u4: torch.Tensor   # [d_out, n_groups] fp16

    # For legacy_mma path we pre-quantise X once so the timing loop only
    # includes (activation_quant + MMA). activation_quant is part of the
    # generation cost we *want* to count — leave it inside the closure,
    # NOT in the pre-allocated block.


def _build_bundle(
    d_in: int,
    d_out: int,
    T: int,
    *,
    device: torch.device,
    name: str,
) -> W4A4OpBundle:
    """Allocate one W4A4 kernel's worth of inputs on ``device``.

    Picks the kernel path (gemv / legacy_mma) from T; we always return a
    bundle that is ready for launch — the caller just wraps it in a
    zero-arg lambda.
    """
    if d_in % BCOL != 0:
        raise ValueError(f"{name}: d_in ({d_in}) not divisible by BCOL ({BCOL})")
    if d_out % 128 != 0:
        raise ValueError(f"{name}: d_out ({d_out}) not divisible by 128")

    path = "gemv" if T == 1 else "legacy_mma"

    # Activation: realistic scale (~0.4) for the max-abs quant path.
    X_fp16 = (torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4).contiguous()
    perm = torch.randperm(d_in, device=device).to(torch.int32).contiguous()

    # 4-bit weight, packed as 2 nibbles per int8 (d_out × d_in//2).
    W_low = torch.randint(
        0, 16, (d_out, d_in // 2), dtype=torch.int8, device=device
    ).contiguous()

    # Per-group scale/zero, layout: [d_out, n_groups] fp16.
    n_groups = d_in // BCOL
    scale_u4 = (torch.rand(d_out, n_groups, dtype=torch.float16, device=device)
                * 0.01 + 0.001).contiguous()
    zero_u4 = (torch.rand(d_out, n_groups, dtype=torch.float16, device=device)
               * 14.0).contiguous()

    # Sparse path disabled: empty hp tensors.
    W_high_blocks = torch.zeros((0, 128, BCOL // 2), dtype=torch.int8, device=device)
    hp_row_offsets = torch.zeros((d_out // 128) + 1, dtype=torch.int32, device=device)
    hp_col_indices = torch.zeros(0, dtype=torch.int32, device=device)

    return W4A4OpBundle(
        name=name,
        d_in=d_in,
        d_out=d_out,
        T=T,
        path=path,
        X_fp16=X_fp16,
        perm=perm,
        W_low=W_low,
        W_high_blocks=W_high_blocks,
        hp_row_offsets=hp_row_offsets,
        hp_col_indices=hp_col_indices,
        scale_u4=scale_u4,
        zero_u4=zero_u4,
    )


def _make_callable(bundle: W4A4OpBundle) -> Callable[[], torch.Tensor]:
    """Wrap a pre-built bundle in a zero-arg lambda that launches the kernel.

    The closure is reentrant (no in-place accumulation on inputs).  The
    returned Y tensor is fresh every call (kernel allocates), matching
    what a real layer would do.
    """
    d_in, d_out, T = bundle.d_in, bundle.d_out, bundle.T

    if bundle.path == "gemv":
        # T == 1 — single-kernel fused quant + GEMV.
        def fn():
            return w4a4_ops.fused_quant_gemv_cuda(
                bundle.X_fp16, bundle.perm,
                bundle.W_low, bundle.W_high_blocks,
                bundle.hp_row_offsets, bundle.hp_col_indices,
                bundle.scale_u4, bundle.zero_u4,
                d_out, d_in,
            )
        return fn

    # Legacy two-step MMA (T >= 2, production default per ops.py gate).
    # Both activation_quant AND fused_dense_sparse MUST be inside the
    # closure because production inference pays both launches per op.
    def fn():
        X_s4, scale_x, sum_X = w4a4_ops.activation_quant_cuda(
            bundle.X_fp16, bundle.perm
        )
        return w4a4_ops.fused_dense_sparse_cuda_int4(
            bundle.W_low, bundle.W_high_blocks,
            bundle.hp_row_offsets, bundle.hp_col_indices,
            X_s4, bundle.scale_u4, bundle.zero_u4,
            sum_X, scale_x, d_out, d_in,
        )
    return fn


# -----------------------------------------------------------------------------
# Public API: build all 4 fused-op callables for one (model, phase, bs).
# -----------------------------------------------------------------------------
@dataclass
class W4A4FourOp:
    """Holds bundle + zero-arg callable for each of the 4 fused groups."""
    qkv_fused: tuple[W4A4OpBundle, Callable[[], torch.Tensor]]
    o_proj:        tuple[W4A4OpBundle, Callable[[], torch.Tensor]]
    gate_up_fused: tuple[W4A4OpBundle, Callable[[], torch.Tensor]]
    down_proj:     tuple[W4A4OpBundle, Callable[[], torch.Tensor]]

    def as_list(self) -> list[tuple[str, W4A4OpBundle, Callable[[], torch.Tensor]]]:
        return [
            ("qkv_fused",     *self.qkv_fused),
            ("o_proj",        *self.o_proj),
            ("gate_up_fused", *self.gate_up_fused),
            ("down_proj",     *self.down_proj),
        ]


def build_four_op_callables(
    cfg: Qwen3Config,
    *,
    batch: int,
    seqlen: int,
    device: torch.device | str = "cuda",
) -> W4A4FourOp:
    """Build the 4 fused-op callables for one (model, phase, bs) triple.

    T = batch * seqlen.  Shape legality (d_in/d_out % 128 == 0) is
    verified against Qwen3 config — all three selected models (4B / 8B
    / 14B) pass by construction; see qwen3_shapes.py.
    """
    dev = torch.device(device)
    T = batch * seqlen

    bundles: dict[str, W4A4OpBundle] = {}
    for proj in enumerate_fused_projs(cfg):
        bundles[proj.proj] = _build_bundle(
            d_in=proj.d_in, d_out=proj.d_out, T=T,
            device=dev, name=proj.proj,
        )

    return W4A4FourOp(
        qkv_fused     = (bundles["qkv_fused"],     _make_callable(bundles["qkv_fused"])),
        o_proj        = (bundles["o"],             _make_callable(bundles["o"])),
        gate_up_fused = (bundles["gate_up_fused"], _make_callable(bundles["gate_up_fused"])),
        down_proj     = (bundles["down"],          _make_callable(bundles["down"])),
    )


__all__ = [
    "W4A4OpBundle",
    "W4A4FourOp",
    "build_four_op_callables",
]
