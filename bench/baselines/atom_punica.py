"""Thin PyTorch wrappers around the Atom punica-atom kernel.

This module is the **only** place that imports ``punica.ops``.  Everything
else in ``kernel.bench.baselines`` operates on the dataclasses defined
here, so the rest of the bench tree can be type-checked / read on a
machine without the Atom CUDA kernel installed.

What is wrapped
---------------
We expose three callables, each returning a zero-argument lambda suitable
for ``kernel.bench.layer.timing.measure``:

  1. ``build_quantize_callable(M, K)``
     The cost of producing the (outlier_int8, norms_int4, outlier_scale,
     norm_scale) quad from a fp16 hidden_states tensor of shape (M, K).
     This is what the Atom paper calls "online activation quant"; in
     punica-atom it is fused into ``rmsnorm_fp16_i4`` / ``activate_fp16_i4``
     so we call those directly with weight=ones / b=ones, and report the
     result as ``T_quant_for_<op>``.

  2. ``build_gemm_callable(M, N, K)``
     The cost of one ``dense_layer_gemm_i4_fp16(M, N, K)`` call with
     pre-quantized inputs and randomly-initialised INT4 weights.  This
     is the "pure GEMM" timing — no quant, no dequant, just the kernel
     work.  Output is fp16 of shape (M, N).

  3. ``build_e2e_callable(M, N, K)``
     End-to-end-of-replaced-region: rmsnorm_fp16_i4(input) → gemm → out.
     This is what we *actually* sum into the baseline layer timing.
     We deliberately use rmsnorm rather than reorder, because the Atom
     decoder layer always feeds GEMMs from rmsnorm/activate (the only
     exception is o_proj, which uses reorder; we expose that explicitly
     as ``build_e2e_o_callable``).

Failure modes
-------------
If ``import punica.ops._kernels`` raises (e.g. running on macOS, or on a
non-RTX-4090 GPU), every ``build_*`` function raises
``BaselineUnavailable`` with the original exception chained in.  The
bench driver catches that and skips the point with a clear note.

Versioning
----------
Tested against punica-atom commit included in the public Atom MLSys'24
repository (no tag — it's a snapshot inside ``e2e/punica-atom/``).
If the upstream API changes, the *only* place to update is this file.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import torch

from kernel.bench.baselines import BaselineUnavailable


# ---------------------------------------------------------------------------
# Lazy import of punica.ops — keeps the source tree portable.
# ---------------------------------------------------------------------------
_PUNICA_OPS: Any = None  # filled by _ensure_punica()


def _ensure_punica() -> Any:
    """Import ``punica.ops`` (compiled C++ extension) or raise.

    We import lazily so the bench tree itself is importable on machines
    where the Atom kernel is not installed.  The result is cached.
    """
    global _PUNICA_OPS
    if _PUNICA_OPS is not None:
        return _PUNICA_OPS
    try:
        ops = importlib.import_module("punica.ops")
        # Touch the C++ module so we get a clear error here, not later.
        importlib.import_module("punica.ops._kernels")
    except Exception as e:  # ImportError, OSError (cuda mismatch), ...
        raise BaselineUnavailable(
            "punica.ops is not importable on this host. "
            "See kernel/bench/baselines/BASELINE_SETUP.md for the autodl "
            "build instructions.  Original error: "
            f"{type(e).__name__}: {e}"
        ) from e
    _PUNICA_OPS = ops
    return ops


# ---------------------------------------------------------------------------
# Helpers — the punica-atom data layout (mirrored from punica/models/llama.py)
# ---------------------------------------------------------------------------
_GROUP_SIZE: int = 128
_KEEPER_SIZE: int = 128


def _alloc_quantized_tuple(
    M: int,
    K: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the (outlier, norms, outlier_scale, norm_scale) tuple
    that ``dense_layer_gemm_i4_fp16`` consumes as activation A.

    Shapes copied verbatim from ``punica.ops.activate_fp16_i4``:
      * outlier      : (M, 128)              int8
      * norms        : (M, (K - 128) // 2)   int8  (two int4s packed per byte)
      * outlier_sc   : (scale_size(M),)      fp16
      * norm_sc      : (K // 128 - 1, scale_size(M))  fp16
    """
    ops = _ensure_punica()
    outlier = torch.empty((M, _KEEPER_SIZE), dtype=torch.int8, device=device)
    norms = torch.empty((M, (K - _KEEPER_SIZE) // 2), dtype=torch.int8, device=device)
    outlier_sc = torch.empty((ops.scale_size(M),), dtype=torch.float16, device=device)
    norm_sc = torch.empty(
        (K // _GROUP_SIZE - 1, ops.scale_size(M)),
        dtype=torch.float16,
        device=device,
    )
    return outlier, norms, outlier_sc, norm_sc


def _alloc_weight(
    N: int,
    K: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the (weight_int4, weight_int8, scale_int4, scale_int8) tuple
    that ``dense_layer_gemm_i4_fp16`` consumes as weight B.

    Shapes mirror ``LinearInt4`` in punica/models/llama.py.
    """
    ops = _ensure_punica()
    weight_int4 = torch.empty(
        (N, (K - _KEEPER_SIZE) // 2), dtype=torch.uint8, device=device
    )
    weight_int8 = torch.empty((N, _KEEPER_SIZE), dtype=torch.int8, device=device)
    scale_int4 = torch.empty(
        (K // _GROUP_SIZE - 1, ops.scale_size(N)),
        dtype=torch.float16,
        device=device,
    )
    scale_int8 = torch.empty((ops.scale_size(N),), dtype=torch.float16, device=device)
    return weight_int4, weight_int8, scale_int4, scale_int8


# ---------------------------------------------------------------------------
# Public callable factories
# ---------------------------------------------------------------------------
def build_gemm_callable(
    M: int, N: int, K: int, *, device: torch.device | str = "cuda",
) -> Callable[[], torch.Tensor]:
    """Pure GEMM: pre-quantized A + pre-quantized B → fp16 output.

    ``M``, ``N``, ``K`` follow the standard convention:
      * M = batch * seqlen (rows of activation)
      * K = d_in           (input features, must be divisible by 128 and ≥ 256)
      * N = d_out          (output features, divisible by 16)

    The returned lambda allocates *no* new tensors per call (output tensor
    is allocated by the kernel itself).  Safe for ``timing.measure``.
    """
    ops = _ensure_punica()
    dev = torch.device(device)

    a_outlier, a_norms, a_out_sc, a_norm_sc = _alloc_quantized_tuple(M, K, device=dev)
    w_int4, w_int8, w_sc4, w_sc8 = _alloc_weight(N, K, device=dev)

    # Touch with random data once — the kernel does not validate magnitudes,
    # but real values give realistic memory traffic.
    # NOTE: torch.Tensor.random_(from, to) requires from >= 0; for signed
    # int8 we have to go through randint() and copy_().
    a_outlier.copy_(torch.randint(-8, 9, a_outlier.shape, device=dev, dtype=torch.int8))
    a_norms.copy_(torch.randint(-128, 128, a_norms.shape, device=dev, dtype=torch.int8))
    a_out_sc.uniform_(-0.1, 0.1)
    a_norm_sc.uniform_(-0.1, 0.1)
    # uint8 weight: full byte range is fine via random_ (from=0).
    w_int4.random_(0, 256)
    w_int8.copy_(torch.randint(-128, 128, w_int8.shape, device=dev, dtype=torch.int8))
    w_sc4.uniform_(-0.1, 0.1)
    w_sc8.uniform_(-0.1, 0.1)

    def _gemm() -> torch.Tensor:
        return ops.dense_layer_gemm_i4_fp16(
            a_norms, w_int4, a_norm_sc, w_sc4,
            a_outlier, w_int8, a_out_sc, w_sc8,
        )

    return _gemm


def build_quantize_via_rmsnorm_callable(
    M: int, K: int, *, device: torch.device | str = "cuda",
) -> Callable[[], tuple]:
    """Cost of producing the activation tuple via Atom's fused RMSNorm+quant.

    This is the cost a Qwen3 layer pays *before* qkv_fused or gate_up_fused.
    Per Atom design, the rmsnorm op writes the four quantized buffers
    directly, so timing this gives us the "online quant for the next GEMM"
    figure.

    Note we do NOT separately time torch RMSNorm — Atom replaces that op
    entirely; what we are measuring is exactly the *delta* the baseline
    pays in place of (BF16 RMSNorm + free-fp16-input).
    """
    ops = _ensure_punica()
    dev = torch.device(device)

    hidden = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.4
    weight = torch.ones(K, dtype=torch.float16, device=dev)
    reorder = torch.randperm(K, dtype=torch.int16, device=dev)

    def _rmsnorm_quant():
        return ops.rmsnorm_fp16_i4(hidden, weight, reorder, 1e-6)

    return _rmsnorm_quant


def build_quantize_via_reorder_callable(
    M: int, K: int, *, device: torch.device | str = "cuda",
) -> Callable[[], tuple]:
    """Cost of quantizing fp16 input via Atom's reorder kernel.

    This is the path taken right before ``o_proj`` in the Atom decoder
    (attention output is already in fp16, so reorder + quant fuses the
    channel permutation with the fp16 → INT4/INT8 cast).
    """
    ops = _ensure_punica()
    dev = torch.device(device)

    hidden = torch.randn(M, K, dtype=torch.float16, device=dev) * 0.4
    reorder = torch.randperm(K, dtype=torch.int16, device=dev)

    def _reorder():
        return ops.reorder_fp16_i4(hidden, reorder)

    return _reorder


def build_quantize_via_activate_callable(
    M: int, K_intermediate: int, *, device: torch.device | str = "cuda",
) -> Callable[[], tuple]:
    """Cost of computing silu(gate)*up + quantization, fused.

    This is the path taken right before ``down_proj`` in the Atom MLP.
    Inputs are two fp16 tensors of shape (M, intermediate); output is the
    quantized tuple consumed by down_proj.
    """
    ops = _ensure_punica()
    dev = torch.device(device)

    a = torch.randn(M, K_intermediate, dtype=torch.float16, device=dev) * 0.4
    b = torch.randn(M, K_intermediate, dtype=torch.float16, device=dev) * 0.4

    def _activate():
        return ops.activate_fp16_i4(a, b)

    return _activate


__all__ = [
    "build_gemm_callable",
    "build_quantize_via_rmsnorm_callable",
    "build_quantize_via_reorder_callable",
    "build_quantize_via_activate_callable",
]
