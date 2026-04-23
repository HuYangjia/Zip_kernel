"""Fused per-token SINT4 activation quantization Triton kernel.

Produces three outputs from a FP16 activation tensor in one kernel launch:
  - X_s4    : (batch*seq, d_in // 2) int8, 4-bit little-endian packed SINT4
  - scale_x : (batch*seq,)            fp16, per-token symmetric scale
  - sum_X   : (batch*seq, n_groups)   int32, per-group sum of SINT4 activations

See requirements.md section 2 and triton_kernel_prompt.md section 4.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as tl_libdevice

from .pack_utils import BCOL


# ---------------------------------------------------------------------------
# Per-device identity-perm cache
# ---------------------------------------------------------------------------
# When the L2-thrash workaround triggers we replace a random permutation
# with the identity.  Allocating torch.arange(D) per call would show up
# as ~3-5us of host+launch overhead on decode shapes, which is significant
# relative to the kernel itself.  We cache one (device, D) -> tensor
# mapping so the hot path becomes a dict lookup.
#
# In addition we keep a small set of "known identity" data_ptrs so that
# a caller passing their own torch.arange(D) (e.g. benchmarks, unit
# tests, or sweep harnesses that pre-build W.perm=arange) also gets
# detected as identity.  The first encounter pays one GPU sync
# (torch.equal against our cached arange), subsequent encounters are
# a pure dict lookup.
_IDENTITY_PERM_CACHE: dict = {}
_KNOWN_IDENTITY_PTRS: set = set()


def _identity_perm(D: int, device: torch.device) -> torch.Tensor:
    key = (str(device), D)
    t = _IDENTITY_PERM_CACHE.get(key)
    if t is None:
        t = torch.arange(D, dtype=torch.int32, device=device)
        _IDENTITY_PERM_CACHE[key] = t
        _KNOWN_IDENTITY_PTRS.add(t.data_ptr())
    return t


def _is_identity_perm(perm: torch.Tensor) -> bool:
    """Return True if `perm` is semantically the identity permutation.

    Fast path: memoize by data_ptr.  First time we see a new pointer we
    do a *single* torch.equal check against our cached identity tensor,
    which costs one GPU sync (~10us) but is then cached forever.
    """
    ptr = perm.data_ptr()
    if ptr in _KNOWN_IDENTITY_PTRS:
        return True
    # Only worth the sync if it *might* be identity: require int32/int64
    # and shape-compatible with a cached arange.  If the user never
    # constructs torch.arange, this path is never entered.
    D = perm.numel()
    key = (str(perm.device), D)
    cached = _IDENTITY_PERM_CACHE.get(key)
    if cached is None:
        # No cached identity for this (device, D) yet -- build it lazily.
        cached = torch.arange(D, dtype=torch.int32, device=perm.device)
        _IDENTITY_PERM_CACHE[key] = cached
        _KNOWN_IDENTITY_PTRS.add(cached.data_ptr())
    if perm.dtype != cached.dtype:
        perm_cmp = perm.to(cached.dtype)
    else:
        perm_cmp = perm
    if torch.equal(perm_cmp, cached):
        _KNOWN_IDENTITY_PTRS.add(ptr)
        return True
    return False




# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Autotune config selection rationale (2026-04-23 rework)
# ---------------------------------------------------------------------------
# The original config list (2026-04-22 version, kept as `# old` comments
# below) biased toward BD=2048 for "large-T prefill".  That turned out to
# be a trap: at D=11008/14336 the Pass-2 loop unrolls N_GROUPS=86..112
# groups, and each iteration keeps 4 live (BT, BCOL/2)=(BT, 64) SINT4
# tiles in registers (x_lo, x_hi, q_lo, q_hi), plus perm_lo/hi for the
# permuted gather.  With BD=2048 and BT=128 the compiler runs out of
# registers and spills to local memory; autotune then has no choice but
# to fall back to the tiniest config it knows (BT=16, BD=512), which
# turns T=8192 prefill into 512 CTAs launching serially -> 17 ms.
#
# The new grid is built around two principles:
#   (1) Cap BD at 1024.  Past that, register pressure outweighs loop
#       latency hiding for this kernel (verified by microbench).
#   (2) Add BT in {64, 128, 256} paired with BD in {256, 512, 1024} so
#       that when T is large we get wide CTAs with few iterations rather
#       than narrow CTAs with many iterations.
#
# A `prune_configs_by.early_config_prune` hook filters out obviously bad
# configs *before* Triton compiles and benchmarks them, cutting autotune
# dispatch cost by ~4x on first call.

def _prune_quant_configs(configs, named_args, **kwargs):
    """Remove configs that are provably dominated for the given shape.

    Run before compilation, so filtering here saves both compile time and
    benchmark time.  Keeps at least one config so autotune never fails.
    """
    T = named_args["T"]
    D = named_args["D"]
    kept = []
    for c in configs:
        bt = c.kwargs["BT"]
        bd = c.kwargs["BD"]
        # Rule 1: BT must not exceed T by more than 4x (else wasted lanes)
        if bt > 4 * T and bt > 16:
            continue
        # Rule 2: BD must not exceed D (partial tiles only hurt)
        if bd > D:
            continue
        # Rule 3: at very small T (<= 16) we *never* want BT > 32
        # (bulk of BT lanes sit idle, just wastes SM cycles)
        if T <= 16 and bt > 32:
            continue
        kept.append(c)
    # Defensive: always keep at least one
    return kept if kept else [configs[0]]


@triton.autotune(
    configs=[
        # ----- Decode family (BT <= 32) ---------------------------------
        triton.Config({"BT": 16,  "BD": 256},  num_warps=2, num_stages=2),
        triton.Config({"BT": 16,  "BD": 512},  num_warps=2, num_stages=3),
        triton.Config({"BT": 32,  "BD": 256},  num_warps=2, num_stages=2),
        triton.Config({"BT": 32,  "BD": 512},  num_warps=4, num_stages=3),
        triton.Config({"BT": 32,  "BD": 1024}, num_warps=4, num_stages=2),
        # ----- Medium family (BT = 64) ----------------------------------
        triton.Config({"BT": 64,  "BD": 256},  num_warps=4, num_stages=3),
        triton.Config({"BT": 64,  "BD": 512},  num_warps=4, num_stages=2),
        triton.Config({"BT": 64,  "BD": 512},  num_warps=4, num_stages=3),
        triton.Config({"BT": 64,  "BD": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BT": 64,  "BD": 1024}, num_warps=8, num_stages=3),
        # ----- Large-T family (BT = 128) --------------------------------
        triton.Config({"BT": 128, "BD": 256},  num_warps=4, num_stages=3),
        triton.Config({"BT": 128, "BD": 512},  num_warps=4, num_stages=2),
        triton.Config({"BT": 128, "BD": 512},  num_warps=8, num_stages=3),
        triton.Config({"BT": 128, "BD": 1024}, num_warps=8, num_stages=2),
        # Restored: baseline frequently picked this for T=2048 prefill.
        triton.Config({"BT": 128, "BD": 1024}, num_warps=8, num_stages=3),
        # ----- Very-large-T family (BT = 256) ---------------------------
        triton.Config({"BT": 256, "BD": 256},  num_warps=8, num_stages=3),
        triton.Config({"BT": 256, "BD": 512},  num_warps=8, num_stages=2),
    ],
    key=["T", "D", "N_GROUPS"],
    prune_configs_by={"early_config_prune": _prune_quant_configs},
)
@triton.jit
def quantize_activation_kernel(
    X_ptr, perm_ptr,
    X_s4_ptr, scale_x_ptr, sum_X_ptr,
    T, D,                              # batch*seq, d_in
    stride_xt, stride_xd,              # X strides
    stride_qt, stride_qd,              # X_s4 strides (packed, last dim = D // 2)
    stride_st,                         # sum_X strides (T, n_groups)
    stride_sg,
    N_GROUPS: tl.constexpr,
    BCOL_K: tl.constexpr,              # group size along d_in
    BT: tl.constexpr,                  # tokens per program
    BD: tl.constexpr,                  # tile along d_in for streaming pass
):
    pid_t = tl.program_id(0)
    t_start = pid_t * BT
    offs_t = t_start + tl.arange(0, BT)
    mask_t = offs_t < T

    # ------------------------------------------------------------------
    # Pass 1: compute per-token max(|X|) in permuted order.
    # Streaming along d_in with tile size BD so we never cache the whole row.
    # ------------------------------------------------------------------
    max_abs = tl.zeros((BT,), dtype=tl.float32)
    for d_start in range(0, D, BD):
        offs_d = d_start + tl.arange(0, BD)
        mask_d = offs_d < D
        # Gather permuted column indices (int32)
        perm_idx = tl.load(perm_ptr + offs_d, mask=mask_d, other=0).to(tl.int32)
        # Load X[t, perm_idx]
        x_ptrs = X_ptr + offs_t[:, None] * stride_xt + perm_idx[None, :] * stride_xd
        x_tile = tl.load(
            x_ptrs,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        tile_max = tl.max(tl.abs(x_tile), axis=1)
        max_abs = tl.maximum(max_abs, tile_max)

    # scale_x = max / 7   (symmetric SINT4; clamp denom away from zero)
    # Important: both the stored scale and the value used for quantization
    # must go through the same fp16 rounding, otherwise the kernel output
    # differs bitwise from a reference that rounds scale to fp16 first.
    # Also: use x / scale (single rounding) rather than x * (1/scale) (two
    # roundings) to match numpy/torch's `torch.round(x / s)` reference.
    scale_fp32 = max_abs / 7.0
    scale_fp16 = scale_fp32.to(tl.float16)           # round-to-fp16
    scale = scale_fp16.to(tl.float32)                # back to fp32 for math
    # For zero rows: use 1.0 in the divide so we don't get NaN, but zero out
    # the result via a mask below.
    scale_safe = tl.where(scale > 0.0, scale, 1.0)
    scale_is_zero = scale <= 0.0

    tl.store(scale_x_ptr + offs_t, scale_fp16, mask=mask_t)

    # ------------------------------------------------------------------
    # Pass 2: quantize, pack (little-endian 4-bit), and accumulate sum_X
    # per group.  We walk d_in in tiles of BCOL_K (== group size) so the
    # per-group reduction is trivial.
    #
    # Implementation note: Triton 3.x forbids Python-style slicing on
    # constexpr dims (e.g. `q[:, :, 0]` or `q[:, 0::2]`).  To obtain the
    # even/odd columns needed for 4-bit packing we issue two separate loads
    # using strided offsets (2*i for low nibble, 2*i+1 for high nibble).
    # This yields two (BT, BCOL_K//2) tiles directly, so no reshape/slice
    # is required.
    # ------------------------------------------------------------------
    offs_h = tl.arange(0, BCOL_K // 2)
    for g in range(0, N_GROUPS):
        d_start = g * BCOL_K
        # Even (low-nibble) column indices within this group: 2*h
        offs_d_lo = d_start + 2 * offs_h
        # Odd  (high-nibble) column indices within this group: 2*h + 1
        offs_d_hi = d_start + 2 * offs_h + 1
        mask_d_lo = offs_d_lo < D
        mask_d_hi = offs_d_hi < D

        # Gather permuted column indices separately for even/odd cols.
        perm_lo = tl.load(perm_ptr + offs_d_lo, mask=mask_d_lo, other=0).to(tl.int32)
        perm_hi = tl.load(perm_ptr + offs_d_hi, mask=mask_d_hi, other=0).to(tl.int32)

        # Load FP16 activations for even / odd columns.
        x_lo = tl.load(
            X_ptr + offs_t[:, None] * stride_xt + perm_lo[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d_lo[None, :],
            other=0.0,
        ).to(tl.float32)
        x_hi = tl.load(
            X_ptr + offs_t[:, None] * stride_xt + perm_hi[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d_hi[None, :],
            other=0.0,
        ).to(tl.float32)

        # q = clamp(round(x / scale), -8, 7)  -- inlined for lo/hi.
        # Use division (single fp32 rounding) to match torch.round(x / s),
        # and libdevice.rint for IEEE round-half-to-even (matches torch.round).
        q_lo = x_lo / scale_safe[:, None]
        q_lo = tl_libdevice.rint(q_lo)
        q_lo = tl.minimum(tl.maximum(q_lo, -8.0), 7.0)
        q_lo_i32 = q_lo.to(tl.int32)
        # Zero rows must produce q=0, not garbage from dividing by the 1.0 fallback.
        q_lo_i32 = tl.where(scale_is_zero[:, None], 0, q_lo_i32)
        q_lo_i32 = tl.where(mask_d_lo[None, :], q_lo_i32, 0)

        q_hi = x_hi / scale_safe[:, None]
        q_hi = tl_libdevice.rint(q_hi)
        q_hi = tl.minimum(tl.maximum(q_hi, -8.0), 7.0)
        q_hi_i32 = q_hi.to(tl.int32)
        q_hi_i32 = tl.where(scale_is_zero[:, None], 0, q_hi_i32)
        q_hi_i32 = tl.where(mask_d_hi[None, :], q_hi_i32, 0)

        # sum_X[t, g] = sum_k q_i32[t, k] over the full group (low+high).
        g_sum = tl.sum(q_lo_i32, axis=1) + tl.sum(q_hi_i32, axis=1)
        tl.store(sum_X_ptr + offs_t * stride_st + g * stride_sg, g_sum, mask=mask_t)

        # Pack two consecutive SINT4 values into one int8 byte (little-endian).
        # After q & 0x0F we have the 4-bit two's-complement pattern.
        low = q_lo_i32 & 0x0F
        high = q_hi_i32 & 0x0F
        packed = ((high << 4) | low) & 0xFF
        # Convert 0..255 -> signed int8
        packed_i8 = tl.where(packed >= 128, packed - 256, packed).to(tl.int8)

        # Store packed bytes.  X_s4 layout: (T, D // 2).  Column offset is
        # (d_start // 2) + [0 .. BCOL_K/2).
        byte_offs = (d_start // 2) + offs_h
        byte_mask = byte_offs < (D // 2)
        qs_ptrs = X_s4_ptr + offs_t[:, None] * stride_qt + byte_offs[None, :] * stride_qd
        tl.store(
            qs_ptrs,
            packed_i8,
            mask=mask_t[:, None] & byte_mask[None, :],
        )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def quantize_activation_s4(
    X_fp16: torch.Tensor,
    perm: torch.Tensor,
    bcol: int = BCOL,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused activation quantization wrapper.

    BT / BD / num_warps are picked by Triton autotune (keyed on (T, D, N_GROUPS)),
    so the first call at a new shape pays a short auto-tuning cost and then
    caches the best config.

    Args:
        X_fp16: (batch, seq_len, d_in) or (T, d_in) fp16 tensor.
        perm:   (d_in,) int32 permutation (act-order).
        bcol:   group size (default 128).

    Returns:
        X_s4    : (T, d_in // 2) int8 packed SINT4
        scale_x : (T,) fp16
        sum_X   : (T, n_groups) int32
    """
    assert X_fp16.is_cuda, "quantize_activation_s4 requires a CUDA tensor"
    assert X_fp16.dtype == torch.float16, "X must be fp16"
    assert perm.dtype in (torch.int32, torch.int64), "perm must be int"

    original_shape = X_fp16.shape
    if X_fp16.dim() == 3:
        T = original_shape[0] * original_shape[1]
        D = original_shape[2]
    elif X_fp16.dim() == 2:
        T, D = original_shape
    else:
        raise ValueError(f"X must be 2D or 3D, got shape {original_shape}")

    if D % bcol != 0:
        raise ValueError(f"d_in ({D}) must be divisible by bcol ({bcol})")
    if D % 2 != 0:
        raise ValueError(f"d_in ({D}) must be even for 4-bit packing")

    X_2d = X_fp16.reshape(T, D).contiguous()
    perm = perm.to(torch.int32).contiguous()

    n_groups = D // bcol
    device = X_2d.device

    # ---------------------------------------------------------------
    # Large-T L2-thrash workaround.
    #
    # When T * D * 2 (bytes of X) exceeds ~L2 capacity (72 MiB on 4090),
    # the in-kernel permuted gather thrashes L2 because each warp's 32
    # lanes touch 32 different random columns per iteration.  Measured
    # on RTX 4090 (D=11008):
    #   T=2048 perm=rand :   537us   (perm=id:  388us)  ratio 1.38x
    #   T=8192 perm=rand : 18566us   (perm=id: 1449us)  ratio 12.81x !!
    #
    # Above ~32 MiB (T*D*2B) the ratio jumps from ~1.4x to >10x.  At
    # that point materialising X_perm = X[:, perm] via torch native
    # index_select (coalesced 1D gather along the contiguous T axis) is
    # cheaper than paying the L2-miss penalty every kernel iteration.
    # For small T the extra launch + HBM traffic hurts more than it
    # helps, so we only take this path above the threshold.
    #
    # Threshold calibrated empirically on RTX 4090 (72 MiB L2):
    #   T=2048, D=11008 (22M elems)   : kernel=537us  pre-perm=628us   KEEP KERNEL
    #   T=8192, D=4096  (33M elems)   : kernel=732us  pre-perm=464us   SWITCH
    #   T=8192, D=11008 (90M elems)   : kernel=18ms   pre-perm=1.9ms   SWITCH!
    # The crossover is ~32M elements (= 64 MiB fp16).
    # ---------------------------------------------------------------
    _L2_THRASH_THRESHOLD_ELEMS = 32 * 1024 * 1024  # == 64 MiB at fp16
    if T * D > _L2_THRASH_THRESHOLD_ELEMS and not _is_identity_perm(perm):
        # Pre-permute X along the feature dim.  After this the kernel
        # walks X in contiguous order -> 100% coalesced loads.
        X_2d = X_2d.index_select(1, perm.to(torch.long)).contiguous()
        perm = _identity_perm(D, device)

    X_s4 = torch.empty((T, D // 2), dtype=torch.int8, device=device)
    scale_x = torch.empty((T,), dtype=torch.float16, device=device)
    sum_X = torch.empty((T, n_groups), dtype=torch.int32, device=device)

    # ---------------------------------------------------------------
    # Fast-path dispatch
    # ---------------------------------------------------------------
    # For T <= 512 autotune always converges on
    #   (BT=16, BD=512, num_warps=2, num_stages=3)
    # across every d_in we tested (4096 / 11008 / 14336).  Keeping the
    # @triton.autotune wrapper on that regime just adds 15-45us of
    # Python-side dispatcher overhead per launch (measured with
    # probe_quant_dispatch.py).  We short-circuit to the fixed-config
    # kernel and save that overhead entirely.
    #
    # At T >= 1024 autotune may select a different BD / num_stages
    # (e.g. T=2048 picks BD=256, stages=2), so we leave those on the
    # autotune path -- there the dispatcher cost is a tiny fraction
    # of the total anyway.
    # ---------------------------------------------------------------
    if T <= 512:
        _FAST_BT = 16
        _FAST_BD = 512
        grid_fast = (triton.cdiv(T, _FAST_BT),)
        quantize_activation_kernel_fast[grid_fast](
            X_2d, perm,
            X_s4, scale_x, sum_X,
            T, D,
            X_2d.stride(0), X_2d.stride(1),
            X_s4.stride(0), X_s4.stride(1),
            sum_X.stride(0), sum_X.stride(1),
            N_GROUPS=n_groups,
            BCOL_K=bcol,
            BT=_FAST_BT,
            BD=_FAST_BD,
            num_warps=2,
            num_stages=3,
        )
        return X_s4, scale_x, sum_X

    # autotune picks BT/BD/num_warps; grid depends on BT so pass a callable.
    grid = lambda META: (triton.cdiv(T, META["BT"]),)
    quantize_activation_kernel[grid](
        X_2d, perm,
        X_s4, scale_x, sum_X,
        T, D,
        X_2d.stride(0), X_2d.stride(1),
        X_s4.stride(0), X_s4.stride(1),
        sum_X.stride(0), sum_X.stride(1),
        N_GROUPS=n_groups,
        BCOL_K=bcol,
    )

    return X_s4, scale_x, sum_X


__all__ = ["quantize_activation_kernel", "quantize_activation_s4"]


# ---------------------------------------------------------------------------
# Fixed-config "fast path" kernel for decode / small-T shapes
# ---------------------------------------------------------------------------
# Why a second kernel?
# --------------------
# For T <= 64 and D <= 8192 the autotuned wrapper consistently picks
# (BT=16, BD=512, warps=2, stages=3).  But the autotune dispatcher itself
# costs 15-45us per launch (measured with probe_quant_dispatch.py):
#
#     T=1,  D=4096  :  autotune=87.6us  fixed=43.1us   (-50.8%)
#     T=16, D=4096  :  autotune=89.6us  fixed=73.9us   (-17.5%)
#     T=64, D=4096  :  autotune=89.3us  fixed=73.7us   (-17.5%)
#
# For larger D (D>=11008) the real work dwarfs the dispatcher cost, so
# we keep the autotune path for them.
#
# Kernel body is IDENTICAL to quantize_activation_kernel -- do not
# diverge them.  If you fix a bug in one, mirror it to the other.
# ---------------------------------------------------------------------------

@triton.jit
def quantize_activation_kernel_fast(
    X_ptr, perm_ptr,
    X_s4_ptr, scale_x_ptr, sum_X_ptr,
    T, D,
    stride_xt, stride_xd,
    stride_qt, stride_qd,
    stride_st, stride_sg,
    N_GROUPS: tl.constexpr,
    BCOL_K: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Fast-path variant: fixed (BT, BD, num_warps, num_stages).

    MUST be kept bitwise-equivalent to `quantize_activation_kernel`;
    the only difference is the absence of the @triton.autotune wrapper
    (and therefore no dispatcher overhead on the Python side).
    """
    pid_t = tl.program_id(0)
    t_start = pid_t * BT
    offs_t = t_start + tl.arange(0, BT)
    mask_t = offs_t < T

    max_abs = tl.zeros((BT,), dtype=tl.float32)
    for d_start in range(0, D, BD):
        offs_d = d_start + tl.arange(0, BD)
        mask_d = offs_d < D
        perm_idx = tl.load(perm_ptr + offs_d, mask=mask_d, other=0).to(tl.int32)
        x_ptrs = X_ptr + offs_t[:, None] * stride_xt + perm_idx[None, :] * stride_xd
        x_tile = tl.load(
            x_ptrs,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        tile_max = tl.max(tl.abs(x_tile), axis=1)
        max_abs = tl.maximum(max_abs, tile_max)

    scale_fp32 = max_abs / 7.0
    scale_fp16 = scale_fp32.to(tl.float16)
    scale = scale_fp16.to(tl.float32)
    scale_safe = tl.where(scale > 0.0, scale, 1.0)
    scale_is_zero = scale <= 0.0

    tl.store(scale_x_ptr + offs_t, scale_fp16, mask=mask_t)

    offs_h = tl.arange(0, BCOL_K // 2)
    for g in range(0, N_GROUPS):
        d_start = g * BCOL_K
        offs_d_lo = d_start + 2 * offs_h
        offs_d_hi = d_start + 2 * offs_h + 1
        mask_d_lo = offs_d_lo < D
        mask_d_hi = offs_d_hi < D

        perm_lo = tl.load(perm_ptr + offs_d_lo, mask=mask_d_lo, other=0).to(tl.int32)
        perm_hi = tl.load(perm_ptr + offs_d_hi, mask=mask_d_hi, other=0).to(tl.int32)

        x_lo = tl.load(
            X_ptr + offs_t[:, None] * stride_xt + perm_lo[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d_lo[None, :],
            other=0.0,
        ).to(tl.float32)
        x_hi = tl.load(
            X_ptr + offs_t[:, None] * stride_xt + perm_hi[None, :] * stride_xd,
            mask=mask_t[:, None] & mask_d_hi[None, :],
            other=0.0,
        ).to(tl.float32)

        q_lo = x_lo / scale_safe[:, None]
        q_lo = tl_libdevice.rint(q_lo)
        q_lo = tl.minimum(tl.maximum(q_lo, -8.0), 7.0)
        q_lo_i32 = q_lo.to(tl.int32)
        q_lo_i32 = tl.where(scale_is_zero[:, None], 0, q_lo_i32)
        q_lo_i32 = tl.where(mask_d_lo[None, :], q_lo_i32, 0)

        q_hi = x_hi / scale_safe[:, None]
        q_hi = tl_libdevice.rint(q_hi)
        q_hi = tl.minimum(tl.maximum(q_hi, -8.0), 7.0)
        q_hi_i32 = q_hi.to(tl.int32)
        q_hi_i32 = tl.where(scale_is_zero[:, None], 0, q_hi_i32)
        q_hi_i32 = tl.where(mask_d_hi[None, :], q_hi_i32, 0)

        g_sum = tl.sum(q_lo_i32, axis=1) + tl.sum(q_hi_i32, axis=1)
        tl.store(sum_X_ptr + offs_t * stride_st + g * stride_sg, g_sum, mask=mask_t)

        low = q_lo_i32 & 0x0F
        high = q_hi_i32 & 0x0F
        packed = ((high << 4) | low) & 0xFF
        packed_i8 = tl.where(packed >= 128, packed - 256, packed).to(tl.int8)

        byte_offs = (d_start // 2) + offs_h
        byte_mask = byte_offs < (D // 2)
        qs_ptrs = X_s4_ptr + offs_t[:, None] * stride_qt + byte_offs[None, :] * stride_qd
        tl.store(
            qs_ptrs,
            packed_i8,
            mask=mask_t[:, None] & byte_mask[None, :],
        )


__all__ = [
    "quantize_activation_kernel",
    "quantize_activation_kernel_fast",
    "quantize_activation_s4",
]
