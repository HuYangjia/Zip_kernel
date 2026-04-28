"""Top-level V9 Linear dispatcher.

This module owns the Python-side pipeline (quant → dense → [sparse] →
combine+transpose) that was previously implemented inside
``kernel.triton_kernel.v9_linear``.  The pipeline structure is
identical; the only change is that each sub-kernel call goes through
:func:`select_impl`, which picks the Triton or CUDA implementation
based on the active :mod:`kernel.backend.policy`.

Why lift the pipeline here instead of patching ``triton_kernel/v9_linear.py``?
-----------------------------------------------------------------------------
- Keeps ``triton_kernel`` pure (no knowledge of CUDA; can be imported on
  machines without an NVCC toolchain).
- Keeps ``cuda_kernel`` pure (drop-in replacements at kernel granularity;
  no duplication of the quant/dense/sparse composition logic).
- Makes the backend-selection seam a single file that is easy to audit.

The W4A16 fallback and the fused-combine-transpose kernel remain
Triton-only for now (the fp16 transpose is already near-optimal on
RTX 4090 and the W4A16 path uses cuBLAS directly).  These are reused
verbatim from ``kernel.triton_kernel.v9_linear``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch

from kernel.triton_kernel.pack_utils import BCOL, V9WeightContainer
from kernel.triton_kernel.dequant_w4_to_fp16 import dequant_u4_to_fp16
from kernel.triton_kernel.v9_linear import (
    DECODE_T_THRESHOLD,
    _combine_transpose,
    reconstruct_w_fakequant_fp16,
    v9_linear_fakequant,
)

from .policy import ShapeContext, current_policy
from .registry import (
    KERNEL_ACTIVATION_QUANT,
    KERNEL_DENSE_GEMM,
    KERNEL_FUSED_DENSE_SPARSE,
    KERNEL_SPARSE_GEMM,
    BackendKernel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-kernel impl selection
# ---------------------------------------------------------------------------


def select_impl(kernel_name: str, args: tuple, kwargs: dict) -> Callable[..., Any]:
    """Return the concrete implementation to call for this invocation.

    Order of resolution::

        policy(kernel_name, ctx) == 'cuda'  and  cuda impl exists  -> CUDA
        policy(kernel_name, ctx) == 'cuda'  and  cuda impl missing -> Triton (silent fallback)
        policy(kernel_name, ctx) == 'triton'                       -> Triton

    The ``args/kwargs`` arguments are reserved for future learned
    policies that inspect tensor shapes; the hand-rolled policy only
    needs the :class:`ShapeContext` built here.

    NB: this function is on the hot path for decode (one call per
    kernel, 2-4 kernels per forward).  Keep it allocation-free apart
    from the :class:`ShapeContext` dataclass.
    """
    ctx = _build_shape_context(kernel_name, args, kwargs)
    choice = current_policy()(kernel_name, ctx)

    if choice == "cuda":
        fn = BackendKernel.cuda_impl(kernel_name)
        if fn is not None:
            return fn
        # Silent fallback: we promised in set_backend_policy('cuda')'s
        # docstring that missing CUDA impls transparently degrade.
        return BackendKernel.triton_impl(kernel_name)

    if choice == "triton":
        return BackendKernel.triton_impl(kernel_name)

    raise ValueError(
        f"Policy returned unexpected backend {choice!r} for kernel {kernel_name!r}"
    )


def _build_shape_context(
    kernel_name: str, args: tuple, kwargs: dict
) -> ShapeContext:
    """Best-effort ShapeContext extraction from call-site arguments.

    We know the Python signatures of each kernel from the Triton
    implementations (they are our reference contract), so we can peek
    at the right positional arg to recover T / d_out / d_in.  Missing
    info stays at the ShapeContext default of -1, which only the
    conservative ``auto`` policy consults.
    """
    if kernel_name == KERNEL_ACTIVATION_QUANT:
        # quantize_activation_s4(X_fp16, perm, bcol=BCOL)
        x = args[0] if args else kwargs.get("X_fp16")
        if isinstance(x, torch.Tensor):
            if x.dim() == 2:
                T, D = int(x.shape[0]), int(x.shape[1])
            elif x.dim() == 3:
                T, D = int(x.shape[0] * x.shape[1]), int(x.shape[2])
            else:
                T, D = -1, -1
            return ShapeContext(T=T, d_in=D, n_groups=(D // BCOL) if D > 0 else -1)
        return ShapeContext()

    if kernel_name in (KERNEL_DENSE_GEMM, KERNEL_FUSED_DENSE_SPARSE):
        # dense_gemm_u4_s4(W_low, X_s4, scale, zero, sum_X, scale_x)
        # fused_dense_sparse_gemm(W_low, W_high_blocks, row_offsets,
        #                         col_indices, X_s4, scale, zero,
        #                         sum_X, scale_x, d_out, d_in)
        x_s4 = (
            args[1] if kernel_name == KERNEL_DENSE_GEMM and len(args) >= 2
            else args[4] if kernel_name == KERNEL_FUSED_DENSE_SPARSE and len(args) >= 5
            else kwargs.get("X_s4")
        )
        w_low = args[0] if args else kwargs.get("W_low_packed")
        T = int(x_s4.shape[0]) if isinstance(x_s4, torch.Tensor) else -1
        d_out = int(w_low.shape[0]) if isinstance(w_low, torch.Tensor) else -1
        d_in = int(w_low.shape[1] * 2) if isinstance(w_low, torch.Tensor) else -1
        return ShapeContext(T=T, d_out=d_out, d_in=d_in,
                            n_groups=(d_in // BCOL) if d_in > 0 else -1)

    if kernel_name == KERNEL_SPARSE_GEMM:
        # sparse_gemm_s4_s4(W_high_blocks, row_offsets, col_indices,
        #                   X_s4, scale, scale_x, d_out, d_in)
        d_out = kwargs.get("d_out", -1)
        d_in = kwargs.get("d_in", -1)
        x_s4 = args[3] if len(args) >= 4 else kwargs.get("X_s4")
        T = int(x_s4.shape[0]) if isinstance(x_s4, torch.Tensor) else -1
        n_hp = 0
        if args and isinstance(args[0], torch.Tensor):
            n_hp = int(args[0].shape[0])
        return ShapeContext(T=T, d_out=d_out, d_in=d_in, n_hp_blocks=n_hp)

    return ShapeContext()


# ---------------------------------------------------------------------------
# Pipeline implementation (mirror of triton_kernel.v9_linear, but each
# sub-kernel call is routed through select_impl)
# ---------------------------------------------------------------------------


def _forward_decode(
    X_2d: torch.Tensor, W: V9WeightContainer, T: int, d_out: int, d_in: int
) -> torch.Tensor:
    # (1) activation quant
    quant_fn = select_impl(KERNEL_ACTIVATION_QUANT, (X_2d, W.perm), {"bcol": BCOL})
    X_s4, scale_x, sum_X = quant_fn(X_2d, W.perm, bcol=BCOL)

    # (2+3) dense [+ sparse].
    # Round 46: prefer the fused_dense_sparse single-kernel path when
    # hp_blocks>0, mirroring _forward_prefill.  Measured on RTX 4090
    # (bench_r46_fused_vs_split_20260427_222239.json, hp=0.05):
    #
    #   shape (T, d_out, d_in)   split   fused   fused save
    #   (1,   4096,  4096)       49.05   37.01   +24.5% ✓
    #   (1,   4096, 11008)      107.61   99.29    +7.7% ✓
    #   (1,  11008,  4096)       47.21   38.33   +18.8% ✓
    #   (8,   4096,  4096)       46.85   34.71   +25.9% ✓
    #   (16,  4096,  4096)       47.43   39.23   +17.3% ✓
    #   (32,  4096,  4096)       48.34   39.48   +18.3% ✓
    #   (64,  4096,  4096)       57.49   48.69   +15.3% ✓
    #   (128, 4096,  4096)       63.29   50.81   +19.7% ✓
    #   (16,  1024,  5120)       52.09   42.76   +17.9% ✓
    #   (64,  1024,  5120)       52.55   43.43   +17.4% ✓
    #   (16,  4096, 11008)      117.02  115.28    +1.5% ·  neutral
    #   (64,  4096, 11008)      137.81  150.73    -9.4% ×  loss (down_proj)
    #
    # Rule of thumb: fused wins whenever d_in <= d_out OR T <= 16
    # (i.e. the dense branch is NOT MMA-bound at the level where fused
    # prologue/epilogue overhead dominates).  It LOSES on down_proj-
    # like shapes with large d_in and mid-T.  Conservative gate:
    # enable fused when hp_blocks>0 AND (d_in <= d_out OR T <= 16).
    use_fused_decode = (
        W.n_hp_blocks > 0
        and (d_in <= d_out or T <= 16)
    )

    if use_fused_decode:
        fused_args = (
            W.W_low_packed,
            W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4,
            W.scale_u4, W.zero_u4, sum_X, scale_x,
        )
        fused_kwargs = {"d_out": d_out, "d_in": d_in}
        fused_fn = select_impl(KERNEL_FUSED_DENSE_SPARSE, fused_args, fused_kwargs)
        Y_low = fused_fn(*fused_args, **fused_kwargs)
        Y_high = None
    else:
        # Legacy split path: dense (+ sparse).
        dense_args = (W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x)
        dense_fn = select_impl(KERNEL_DENSE_GEMM, dense_args, {})
        Y_low = dense_fn(*dense_args)
        Y_high = None
        if W.n_hp_blocks > 0:
            sparse_args = (
                W.W_high_blocks_packed,
                W.hp_row_offsets, W.hp_col_indices,
                X_s4, W.scale_u4, scale_x,
            )
            sparse_kwargs = {"d_out": d_out, "d_in": d_in}
            sparse_fn = select_impl(KERNEL_SPARSE_GEMM, sparse_args, sparse_kwargs)
            Y_high = sparse_fn(*sparse_args, **sparse_kwargs)

    # (4) combine+transpose (Triton-only; falls back to torch native on small T)
    return _combine_transpose(Y_low, Y_high, d_out=d_out, T=T)


def _forward_prefill(
    X_2d: torch.Tensor, W: V9WeightContainer, T: int, d_out: int, d_in: int
) -> torch.Tensor:
    # W4A16 fallback eligibility (dense-only, hp==0).  Reused verbatim
    # from triton_kernel.v9_linear -- cuBLAS FP16 GEMM is already
    # optimal on RTX 4090 for this regime; no CUDA override planned.
    use_w4a16 = (
        W.n_hp_blocks == 0
        and (T >= 1024 or (T >= 512 and (d_out * d_in) <= (4096 * 4096)))
    )
    if use_w4a16:
        W_fp16 = dequant_u4_to_fp16(W)
        X_perm = X_2d.index_select(1, W.perm.to(torch.long))
        return torch.nn.functional.linear(X_perm, W_fp16)

    # (1) activation quant
    quant_fn = select_impl(KERNEL_ACTIVATION_QUANT, (X_2d, W.perm), {"bcol": BCOL})
    X_s4, scale_x, sum_X = quant_fn(X_2d, W.perm, bcol=BCOL)

    # (2+3) dense [+ sparse, fused]
    if W.n_hp_blocks > 0:
        fused_args = (
            W.W_low_packed,
            W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4,
            W.scale_u4, W.zero_u4, sum_X, scale_x,
        )
        fused_kwargs = {"d_out": d_out, "d_in": d_in}
        fused_fn = select_impl(KERNEL_FUSED_DENSE_SPARSE, fused_args, fused_kwargs)
        Y_low = fused_fn(*fused_args, **fused_kwargs)
        Y_high = None
    else:
        dense_args = (W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x)
        dense_fn = select_impl(KERNEL_DENSE_GEMM, dense_args, {})
        Y_low = dense_fn(*dense_args)
        Y_high = None

    return _combine_transpose(Y_low, Y_high, d_out=d_out, T=T)


# ---------------------------------------------------------------------------
# Public entry points (same signatures as triton_kernel.v9_linear)
# ---------------------------------------------------------------------------


def v9_linear_forward(X_fp16: torch.Tensor, W: V9WeightContainer) -> torch.Tensor:
    """Backend-aware V9 Linear forward.

    Same contract as :func:`kernel.triton_kernel.v9_linear.v9_linear_forward`:
    accepts ``X_fp16`` with last dim equal to ``W.d_in`` and returns a
    tensor with last dim replaced by ``W.d_out``.  Per-kernel backend
    selection happens inside the pipeline via the active policy.
    """
    assert X_fp16.is_cuda and X_fp16.dtype == torch.float16
    original_shape = X_fp16.shape
    d_in = W.d_in
    d_out = W.d_out
    if X_fp16.shape[-1] != d_in:
        raise ValueError(
            f"X last dim ({X_fp16.shape[-1]}) must match d_in ({d_in})"
        )
    X_2d = X_fp16.reshape(-1, d_in)
    T = X_2d.shape[0]
    if T <= DECODE_T_THRESHOLD:
        Y_out = _forward_decode(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    else:
        Y_out = _forward_prefill(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    out_shape = original_shape[:-1] + (d_out,)
    return Y_out.reshape(out_shape)


def v9_linear_forward_decode(
    X_fp16: torch.Tensor, W: V9WeightContainer
) -> torch.Tensor:
    """Explicit decode-path entry (for callers pinning T <= DECODE_T_THRESHOLD)."""
    assert X_fp16.is_cuda and X_fp16.dtype == torch.float16
    original_shape = X_fp16.shape
    d_in, d_out = W.d_in, W.d_out
    if X_fp16.shape[-1] != d_in:
        raise ValueError(
            f"X last dim ({X_fp16.shape[-1]}) must match d_in ({d_in})"
        )
    X_2d = X_fp16.reshape(-1, d_in)
    T = X_2d.shape[0]
    if T <= DECODE_T_THRESHOLD:
        Y_out = _forward_decode(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    else:
        # Safety fallback so this entry stays correctness-safe on misuse.
        Y_out = _forward_prefill(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    out_shape = original_shape[:-1] + (d_out,)
    return Y_out.reshape(out_shape)


def v9_linear_forward_prefill(
    X_fp16: torch.Tensor, W: V9WeightContainer
) -> torch.Tensor:
    """Explicit prefill-path entry (for callers pinning T > DECODE_T_THRESHOLD)."""
    assert X_fp16.is_cuda and X_fp16.dtype == torch.float16
    original_shape = X_fp16.shape
    d_in, d_out = W.d_in, W.d_out
    if X_fp16.shape[-1] != d_in:
        raise ValueError(
            f"X last dim ({X_fp16.shape[-1]}) must match d_in ({d_in})"
        )
    X_2d = X_fp16.reshape(-1, d_in)
    T = X_2d.shape[0]
    if T > DECODE_T_THRESHOLD:
        Y_out = _forward_prefill(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    else:
        Y_out = _forward_decode(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    out_shape = original_shape[:-1] + (d_out,)
    return Y_out.reshape(out_shape)


__all__ = [
    "v9_linear_forward",
    "v9_linear_forward_decode",
    "v9_linear_forward_prefill",
    "v9_linear_fakequant",
    "reconstruct_w_fakequant_fp16",
    "select_impl",
]
