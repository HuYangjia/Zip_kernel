"""Shared Phase 1 shape table + V9 input constructors.

Rationale
---------
Multiple Phase 1/2 driver scripts (``phase1_collect_timeline.py``,
``phase1_measure_launch_tax.py``, ``phase2_microbench_bisection.py``)
all need the same fixed-seed V9 inputs.  Keeping the construction code
in a single module removes drift risk: the 4 representative shapes
defined in requirements.md §2.1 are authoritative *here*.

The input builder mirrors ``bench_cuda_vs_triton._make_sparse_inputs``
but is packaged as a function returning both ``X`` and a populated
``V9WeightContainer`` so callers can just write::

    X, W, meta = build_shape_inputs("decode_T1")
    y = v9_linear_forward(X, W)

Note the module does NOT import anything from the R38 bench file so we
keep the coupling unidirectional (profile tooling -> production code,
never the reverse).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch

from kernel.triton_kernel.pack_utils import (
    BCOL,
    BROW,
    V9WeightContainer,
    pack_s4_le,
)


# ---------------------------------------------------------------------------
# Representative shape table  (requirements.md §2.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseShape:
    tag: str              # filesystem-safe short identifier
    T: int
    d_in: int
    d_out: int
    hp_ratio: float       # sparse ratio for the high-precision residual
    model: str            # for report annotation only
    proj: str
    note: str = ""


# These four shapes correspond directly to requirements.md §2.1:
#   - decode representative      (T=1,   Qwen3-1.7B  q_proj,       2048 -> 2048)
#   - worst-efficiency shape     (T=8,   Qwen3-0.6B  kv_proj,      1024 -> 2048)
#   - mid-T representative       (T=128, Qwen3-4B    kv_proj,      2560 -> 2048)
#   - large shape representative (T=1024,Qwen3-8B    gate_up_proj, 4096 -> 24576)
PHASE1_SHAPES: Tuple[PhaseShape, ...] = (
    PhaseShape(
        tag="decode_T1_q_2048_2048",
        T=1, d_in=2048, d_out=2048, hp_ratio=0.05,
        model="Qwen3-1.7B", proj="q_proj",
        note="decode path representative (T=1 fused GEMV)",
    ),
    PhaseShape(
        tag="worst_T8_kv_1024_2048",
        T=8, d_in=1024, d_out=2048, hp_ratio=0.05,
        model="Qwen3-0.6B", proj="kv_proj",
        note="current worst cuda_eff (~5%) per roofline_report §5",
    ),
    PhaseShape(
        tag="mid_T128_kv_2560_2048",
        T=128, d_in=2560, d_out=2048, hp_ratio=0.05,
        model="Qwen3-4B", proj="kv_proj",
        note="mid-T regime, representative of 13% median eff bucket",
    ),
    PhaseShape(
        tag="large_T1024_gu_4096_24576",
        T=1024, d_in=4096, d_out=24576, hp_ratio=0.05,
        model="Qwen3-8B", proj="gate_up_proj",
        note="large-prefill shape, representative of 34% median eff bucket",
    ),
)

PHASE1_SHAPES_BY_TAG: Dict[str, PhaseShape] = {s.tag: s for s in PHASE1_SHAPES}


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------

@dataclass
class BuiltInputs:
    """Everything a driver script might need for one shape."""

    shape: PhaseShape
    X: torch.Tensor
    W: V9WeightContainer
    # Raw tensors, kept around in case a kernel-level microbench wants to call
    # ``activation_quant_cuda`` / ``fused_dense_sparse_cuda`` directly.
    X_s4: torch.Tensor
    scale_x: torch.Tensor
    sum_X: torch.Tensor
    perm: torch.Tensor
    meta: Dict[str, int] = field(default_factory=dict)


def _assert_divisible(name: str, n: int, div: int) -> None:
    if n % div != 0:
        raise ValueError(f"{name}={n} must be divisible by {div}")


def build_shape_inputs(
    tag: str,
    *,
    device: str | torch.device = "cuda",
    seed: int | None = None,
) -> BuiltInputs:
    """Construct a V9 forward input bundle for ``tag``.

    Mirrors ``_make_sparse_inputs`` from bench_cuda_vs_triton.py 1:1,
    but returns a populated :class:`V9WeightContainer` ready to feed
    ``v9_linear_forward``.  Deterministic for a given tag (seeded on
    the shape's structural hash).
    """
    from kernel.triton_kernel.activation_quant import quantize_activation_s4

    if tag not in PHASE1_SHAPES_BY_TAG:
        raise KeyError(
            f"unknown Phase 1 shape tag {tag!r}; "
            f"valid: {sorted(PHASE1_SHAPES_BY_TAG)}"
        )
    shape = PHASE1_SHAPES_BY_TAG[tag]
    T, d_in, d_out = shape.T, shape.d_in, shape.d_out
    _assert_divisible("d_in", d_in, BCOL)
    _assert_divisible("d_out", d_out, BROW)

    dev = torch.device(device)
    # A stable but distinct seed per shape — same formula as the R38 bench.
    s = 0xBEEF + T + d_in + d_out if seed is None else seed
    torch.manual_seed(s)

    # ---- activation side --------------------------------------------------
    X = torch.randn(T, d_in, dtype=torch.float16, device=dev) * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device=dev)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)

    # ---- dense low-bits weight -------------------------------------------
    n_groups = d_in // BCOL
    W_s4 = torch.randint(-8, 8, (d_out, d_in), dtype=torch.int8, device=dev)
    W_low_packed = pack_s4_le(W_s4)
    scale_u4 = (torch.rand(d_out, n_groups, device=dev) * 0.05 + 0.001).to(
        torch.float16
    )
    zero_u4 = (torch.randn(d_out, n_groups, device=dev) * 0.2).to(torch.float16)

    # ---- sparse high-precision residual -----------------------------------
    nrow = d_out // BROW
    ncol = d_in // BCOL
    total_blocks = nrow * ncol
    n_hp = max(1, int(total_blocks * shape.hp_ratio))

    torch.manual_seed((T * d_in * d_out) ^ 0xA5A5)
    flat = torch.randperm(total_blocks, device=dev)[:n_hp]
    br = (flat // ncol).to(torch.int32)
    bc = (flat % ncol).to(torch.int32)
    order = torch.argsort(br.to(torch.int64) * 1_000_000 + bc.to(torch.int64))
    br_sorted = br[order]
    bc_sorted = bc[order]

    W_high_s4 = torch.randint(
        -8, 8, (n_hp, BROW, BCOL), dtype=torch.int8, device=dev
    )
    W_high_blocks_packed = pack_s4_le(W_high_s4)

    hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=dev)
    counts = torch.bincount(br_sorted.to(torch.int64), minlength=nrow)
    hp_row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    W = V9WeightContainer(
        W_low_packed=W_low_packed,
        W_high_blocks_packed=W_high_blocks_packed,
        scale_u4=scale_u4,
        zero_u4=zero_u4,
        hp_row_offsets=hp_row_offsets,
        hp_col_indices=bc_sorted,
        perm=perm,
        block_shape=(BROW, BCOL),
        d_out=d_out,
        d_in=d_in,
    )

    meta = {
        "T": T,
        "d_in": d_in,
        "d_out": d_out,
        "n_groups": n_groups,
        "n_hp_blocks": int(n_hp),
        "total_blocks": int(total_blocks),
    }
    return BuiltInputs(
        shape=shape,
        X=X,
        W=W,
        X_s4=X_s4,
        scale_x=scale_x,
        sum_X=sum_X,
        perm=perm,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Minimal benchmarking helper — wraps the R38 three-piece estimator
# ---------------------------------------------------------------------------

def time_forward_us(
    X: torch.Tensor,
    W: V9WeightContainer,
    *,
    warmup: int = 50,
    outer: int = 3,
    inner: int = 100,
) -> float:
    """min-of-means in microseconds for ``v9_linear_forward(X, W)``.

    Matches the three-piece microbench contract documented in
    ``_bench_util.py``: ``warmup >= 50``, ``inner >= 100``,
    ``outer >= 3``.  Each outer batch uses a single event pair to
    amortise CUDA API overhead.
    """
    from kernel.backend import v9_linear_forward

    device = X.device
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        v9_linear_forward(X, W)
    torch.cuda.synchronize(device)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    means_us: list[float] = []
    for _ in range(outer):
        start_ev.record()
        for _ in range(inner):
            v9_linear_forward(X, W)
        end_ev.record()
        torch.cuda.synchronize(device)
        means_us.append(start_ev.elapsed_time(end_ev) * 1000.0 / inner)
    return min(means_us)


__all__ = [
    "PhaseShape",
    "PHASE1_SHAPES",
    "PHASE1_SHAPES_BY_TAG",
    "BuiltInputs",
    "build_shape_inputs",
    "time_forward_us",
]
