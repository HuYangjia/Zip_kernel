"""R50 L4.1 — V9 → CUTLASS INT4 weight-loader adapter.

This module exposes the single public entry point
:func:`pack_v9_weights_for_cutlass` that L3.4's host-side launcher
(and the test suite) uses to obtain CUTLASS-ready tensor views of an
existing :class:`kernel.triton_kernel.pack_utils.V9WeightContainer`.

**This is NOT a repack.**  Per ``.codebuddy/plan/r50_cutlass_int4/
layout_contract.md`` Decision Log §D.2, the CUTLASS 2.x
``DefaultMma`` kernel for ``int4b_t`` applies its own shared-memory
swizzle internally; the global-memory input format it expects is
plain ``TensorRef<int4b_t, RowMajor>`` — which is **exactly** the
``(d_out, d_in//2) int8`` little-endian half-byte layout that V9
already produces.  The adapter therefore performs only:

1. dtype / shape / stride / alignment validation (frozen contract §1),
2. a pass-through wrap into :class:`CutlassV9Tensors`,

and returns in O(1) Python time, without touching tensor storage.
The round-trip through ``unpack_s4_le`` is bit-identical
(invariant I-L5 in ``layout_contract.md``) and is enforced by the
regression test at
``kernel/backend/tests/test_weight_loader_cutlass_pack.py``.

Public surface (re-exported from ``kernel.backend``):

* :class:`CutlassV9Tensors` — dataclass view
* :func:`pack_v9_weights_for_cutlass` — adapter
* :class:`CutlassPackValidationError` — raised on strict contract violations

Zero GPU dependency at import time.  Safe on any host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from kernel.triton_kernel.pack_utils import V9WeightContainer

# Source-of-truth constants — mirror layout_contract.md §2 without
# importing kernel.cuda_kernel.tools (which would be a reverse
# layering: backend must not depend on cuda_kernel internals).
_ALIGN_A: int = 128  # kAlignmentA, INT4 elements (== 64 bytes)
_GROUP_K: int = 128  # == BCOL in V9 pack format


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class CutlassPackValidationError(ValueError):
    """Raised when a :class:`V9WeightContainer` violates the contract
    required by :func:`pack_v9_weights_for_cutlass` in strict mode.

    The ``violations`` attribute lists every contract rule that was
    broken, so callers can surface a complete diagnostic rather than
    just the first failure.
    """

    def __init__(self, violations: List[str]):
        self.violations = list(violations)
        message = (
            "V9WeightContainer does not satisfy the CUTLASS INT4 layout "
            f"contract. {len(violations)} violation(s):\n  - "
            + "\n  - ".join(violations)
        )
        super().__init__(message)


@dataclass(frozen=True)
class CutlassV9Tensors:
    """CUTLASS-facing view of a :class:`V9WeightContainer`.

    All tensor fields **share storage** with the source container.
    Modifying the underlying memory through this view is observable
    in the source, and vice versa; the adapter never copies.

    Fields map directly to the T1/T5/T6 rows of
    ``layout_contract.md`` §1.
    """

    # T1: weight, `TensorRef<int4b_t, RowMajor>{ptr, d_in}` at the C++ side.
    W_low_rowmajor: torch.Tensor  # (d_out, d_in // 2) int8, stride(1) == 1

    # T5 / T6: per-(M, group) scale / zero (zero is pre-subtracted 8 upstream).
    scale_u4: torch.Tensor        # (d_out, n_groups) fp16
    zero_u4: torch.Tensor         # (d_out, n_groups) fp16

    # logical dimensions (redundant with tensor shapes, kept for C++ launcher convenience)
    d_out: int
    d_in: int
    n_groups: int

    # provenance: keep a weak link back to the source container so callers
    # can resurrect sparse metadata (hp_*, perm, block_shape) when they
    # need the full V9 graph; for the dense-CUTLASS path those fields are
    # irrelevant and intentionally elided from this view.
    source: V9WeightContainer

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"CutlassV9Tensors(d_out={self.d_out}, d_in={self.d_in}, "
            f"n_groups={self.n_groups}, device={self.W_low_rowmajor.device})"
        )


def pack_v9_weights_for_cutlass(
    container: V9WeightContainer,
    *,
    strict: bool = True,
) -> CutlassV9Tensors:
    """Adapt a V9 weight container to the CUTLASS INT4 GEMM tensor view.

    Parameters
    ----------
    container : V9WeightContainer
        The V9 weight container produced by
        :func:`kernel.triton_kernel.pack_utils.pack_v9_weights`.
    strict : bool, default True
        When ``True`` (production default), raise
        :class:`CutlassPackValidationError` listing **every** contract
        violation if the container is malformed.  When ``False``,
        return the view anyway and leave it to the caller (typically
        the regression test) to inspect ``strict=True`` as a separate
        diagnostic pass.

    Returns
    -------
    CutlassV9Tensors
        Zero-copy view of ``container``'s weight/scale/zero tensors,
        pre-validated against the T1/T5/T6 contract rows of
        ``layout_contract.md`` §1.

    Raises
    ------
    CutlassPackValidationError
        (strict mode only) if the container violates at least one
        rule from ``layout_contract.md`` §1 / §2.
    """

    violations = _collect_violations(container)
    if strict and violations:
        raise CutlassPackValidationError(violations)

    # Pass-through wrap. The V9WeightContainer already stores
    # W_low_packed as (d_out, d_in//2) int8 row-major with stride(1)==1,
    # which is the exact memory layout the CUTLASS `int4b_t` RowMajor
    # TensorRef reads (see layout_contract.md §1, §2.4).
    return CutlassV9Tensors(
        W_low_rowmajor=container.W_low_packed,
        scale_u4=container.scale_u4,
        zero_u4=container.zero_u4,
        d_out=int(container.d_out),
        d_in=int(container.d_in),
        n_groups=int(container.n_groups),
        source=container,
    )


# ---------------------------------------------------------------------------
# Validation helpers (private)
# ---------------------------------------------------------------------------


def _collect_violations(container: V9WeightContainer) -> List[str]:
    """Enumerate every ``layout_contract.md`` §1 / §2 rule violation.

    Separated from :func:`pack_v9_weights_for_cutlass` so the regression
    test can call it directly and assert a *complete* list.
    """

    violations: List[str] = []

    # ---- T1: W_low_packed --------------------------------------------------
    W = container.W_low_packed
    d_out_decl, d_in_decl = int(container.d_out), int(container.d_in)

    if not isinstance(W, torch.Tensor):
        violations.append(f"T1: W_low_packed is not a torch.Tensor (got {type(W)!r})")
    else:
        if W.dtype != torch.int8:
            violations.append(
                f"T1: W_low_packed.dtype must be int8, got {W.dtype}"
            )
        if W.dim() != 2:
            violations.append(
                f"T1: W_low_packed must be 2-D, got shape {tuple(W.shape)}"
            )
        else:
            d_out_obs, d_in_half_obs = int(W.shape[0]), int(W.shape[1])
            if d_out_obs != d_out_decl:
                violations.append(
                    f"T1: W_low_packed.shape[0]={d_out_obs} disagrees with "
                    f"container.d_out={d_out_decl}"
                )
            if d_in_half_obs * 2 != d_in_decl:
                violations.append(
                    f"T1: W_low_packed.shape[1]={d_in_half_obs} must equal "
                    f"d_in/2={d_in_decl // 2} (d_in={d_in_decl})"
                )
            if W.stride(1) != 1:
                violations.append(
                    f"T1: W_low_packed.stride(1)={W.stride(1)} must be 1 "
                    "(CUTLASS TensorRef requires the K-dim contiguous)"
                )

    # K-alignment (cp.async.128b needs 128-INT4-element alignment).
    if d_in_decl % _ALIGN_A != 0:
        violations.append(
            f"Alignment: d_in={d_in_decl} not divisible by kAlignmentA={_ALIGN_A} "
            "(CUTLASS cp.async.128b for int4b_t requires 128-element alignment)"
        )
    if d_in_decl % _GROUP_K != 0:
        violations.append(
            f"Alignment: d_in={d_in_decl} not divisible by GROUP_K={_GROUP_K} "
            "(V9 pack format requires K-groups aligned to BCOL)"
        )

    n_groups_expected = d_in_decl // _GROUP_K if d_in_decl > 0 else 0

    # ---- T5: scale_u4 ------------------------------------------------------
    S = container.scale_u4
    violations.extend(
        _validate_per_group_tensor(
            S, "T5", "scale_u4", d_out_decl, n_groups_expected
        )
    )

    # ---- T6: zero_u4 -------------------------------------------------------
    Z = container.zero_u4
    violations.extend(
        _validate_per_group_tensor(
            Z, "T6", "zero_u4", d_out_decl, n_groups_expected
        )
    )

    # ---- device / cross-tensor consistency --------------------------------
    devs = []
    for name, t in [("W_low_packed", W), ("scale_u4", S), ("zero_u4", Z)]:
        if isinstance(t, torch.Tensor):
            devs.append((name, t.device))
    if devs and len(set(d for _, d in devs)) > 1:
        violations.append(
            "Device: W_low_packed / scale_u4 / zero_u4 straddle multiple "
            f"devices: {devs}"
        )

    # n_groups agreement between container property and derived expectation.
    if isinstance(S, torch.Tensor) and S.dim() == 2:
        n_groups_obs = int(S.shape[1])
        if n_groups_expected > 0 and n_groups_obs != n_groups_expected:
            violations.append(
                f"n_groups: scale_u4.shape[1]={n_groups_obs} disagrees with "
                f"d_in/GROUP_K={n_groups_expected}"
            )

    return violations


def _validate_per_group_tensor(
    t: Optional[torch.Tensor],
    tag: str,
    name: str,
    d_out_decl: int,
    n_groups_expected: int,
) -> List[str]:
    """Check a `(d_out, n_groups) fp16 stride(1)==1` scale/zero tensor.

    Returns a list of violations (empty if all good).  Never raises.
    """

    out: List[str] = []
    if not isinstance(t, torch.Tensor):
        out.append(f"{tag}: {name} is not a torch.Tensor (got {type(t)!r})")
        return out

    if t.dtype != torch.float16:
        out.append(f"{tag}: {name}.dtype must be float16, got {t.dtype}")
    if t.dim() != 2:
        out.append(f"{tag}: {name} must be 2-D, got shape {tuple(t.shape)}")
        return out

    d_out_obs, n_groups_obs = int(t.shape[0]), int(t.shape[1])
    if d_out_obs != d_out_decl:
        out.append(
            f"{tag}: {name}.shape[0]={d_out_obs} disagrees with "
            f"container.d_out={d_out_decl}"
        )
    if n_groups_expected > 0 and n_groups_obs != n_groups_expected:
        out.append(
            f"{tag}: {name}.shape[1]={n_groups_obs} must equal "
            f"d_in/GROUP_K={n_groups_expected}"
        )
    if t.stride(1) != 1:
        out.append(
            f"{tag}: {name}.stride(1)={t.stride(1)} must be 1 "
            "(epilogue broadcast expects group-contiguous layout)"
        )
    return out


__all__ = [
    "CutlassV9Tensors",
    "CutlassPackValidationError",
    "pack_v9_weights_for_cutlass",
]
