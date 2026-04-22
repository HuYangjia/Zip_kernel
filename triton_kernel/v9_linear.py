"""V9 end-to-end Linear forward wrapper.

Combines activation quantization + Kernel (1) + Kernel (2) into a single
Python entry point.  Also provides a pure-PyTorch FakeQuant reference for
correctness testing.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .activation_quant import quantize_activation_s4
from .dense_u4s4_gemm import dense_gemm_u4_s4
from .dequant_w4_to_fp16 import dequant_u4_to_fp16
from .pack_utils import BCOL, V9WeightContainer, unpack_s4_le
from .sparse_s4s4_gemm import sparse_gemm_s4_s4


# ---------------------------------------------------------------------------
# Fused combine + transpose kernel.
#
# Replaces the two separate traversals
#     Y_low.add_(Y_high, alpha=16.0)
#     Y_out = Y_low.transpose(0, 1).contiguous()
# which together touch the full (d_out, T) fp16 surface **twice**
# (one load+store for the add, one load+store for the contiguous copy),
# with a single pass that reads Y_low and Y_high row-by-row from the
# (d_out, T) layout and stores directly into the (T, d_out) output layout.
#
# Kernel grid: (cdiv(T, BT), cdiv(d_out, BD))   -- one program per output tile.
# Each program:
#     * loads a (BT, BD) tile from Y_low.T   (reading Y_low with stride (T,1))
#     * optionally loads the same tile from Y_high.T
#     * writes to Y_out[t0:t1, d0:d1] with stride (d_out, 1)   -- COALESCED
#
# This layout choice is critical: the coalesced writes are on the output
# dimension (d_out) which is contiguous in Y_out, so BD consecutive threads
# emit a single sector, while the (d_out, T) input reads are non-coalesced
# but cache-friendly because each warp reads BT rows of BD contiguous cols.
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BT": 32,  "BD": 256}, num_warps=4),
        triton.Config({"BT": 64,  "BD": 128}, num_warps=4),
        triton.Config({"BT": 32,  "BD": 512}, num_warps=8),
        triton.Config({"BT": 64,  "BD": 256}, num_warps=8),
        triton.Config({"BT": 128, "BD": 128}, num_warps=8),
    ],
    key=["T", "d_out", "HAS_HIGH"],
)
@triton.jit
def _combine_transpose_kernel(
    Y_low_ptr,          # (d_out, T) fp16
    Y_high_ptr,         # (d_out, T) fp16 -- may alias Y_low if HAS_HIGH=False
    Y_out_ptr,          # (T, d_out) fp16
    T, d_out,
    stride_low_d, stride_low_t,
    stride_high_d, stride_high_t,
    stride_out_t, stride_out_d,
    BT: tl.constexpr, BD: tl.constexpr,
    HAS_HIGH: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_d = pid_d * BD + tl.arange(0, BD)
    mask_t = offs_t < T
    mask_d = offs_d < d_out

    # Load Y_low[d, t] -> tile shape (BD, BT).
    low_ptrs = Y_low_ptr + offs_d[:, None] * stride_low_d + offs_t[None, :] * stride_low_t
    low_val = tl.load(low_ptrs, mask=mask_d[:, None] & mask_t[None, :], other=0.0)

    if HAS_HIGH:
        high_ptrs = (
            Y_high_ptr
            + offs_d[:, None] * stride_high_d
            + offs_t[None, :] * stride_high_t
        )
        high_val = tl.load(
            high_ptrs, mask=mask_d[:, None] & mask_t[None, :], other=0.0
        )
        # fp16 add in fp32 to avoid subnormal rounding, then back to fp16.
        out_val = (low_val.to(tl.float32) + 16.0 * high_val.to(tl.float32)).to(tl.float16)
    else:
        out_val = low_val

    # Transpose on write: (BD, BT) tile -> Y_out[t, d] stride (d_out, 1)
    # so that BD consecutive threads along the last axis hit contiguous memory.
    out_tile = tl.trans(out_val)                          # (BT, BD)
    out_ptrs = (
        Y_out_ptr
        + offs_t[:, None] * stride_out_t
        + offs_d[None, :] * stride_out_d
    )
    tl.store(out_ptrs, out_tile, mask=mask_t[:, None] & mask_d[None, :])


def _combine_transpose(
    Y_low: torch.Tensor,
    Y_high: torch.Tensor | None,
    d_out: int,
    T: int,
) -> torch.Tensor:
    """Fused combine + transpose: returns (T, d_out) fp16.

    If ``Y_high`` is None, this degenerates to a pure transpose, replacing
    the previous ``Y_low.transpose(0,1).contiguous()`` call with a single
    pass that is slightly faster (no intermediate add) and keeps the kernel
    launch count identical.

    Small-T fast path
    -----------------
    The Triton kernel pays a fixed ~55-65us launch + autotune-dispatch
    overhead.  PyTorch's native ``.t().contiguous()`` is a highly tuned
    memcpy kernel that beats our fused kernel on surfaces below ~4M
    elements (8 MiB fp16).  Measured on RTX 4090 with HAS_HIGH=False:

        surf        torch    triton   winner
        262K elem   11.6us   62.0us   torch (5.3x)
        2M elem     27.2us   52.6us   torch (1.9x)
        8M elem     104us    62.1us   triton (1.7x)

    With HAS_HIGH=True the crossover is similar: torch's ``add_`` + native
    transpose sequence stays ahead of our fused kernel until ~4M elements.

    So we fall back to torch when ``T * d_out <= 4M``.
    """
    assert Y_low.is_cuda and Y_low.dtype == torch.float16
    # Threshold tuned empirically; see docstring for the microbench table.
    # Above 4M elements the fused Triton kernel amortises its launch cost
    # and wins by eliminating one full pass over the surface; below 4M the
    # launch overhead dominates.
    SMALL_SURFACE = 4 * 1024 * 1024  # elements (= 8 MiB fp16)
    if T * d_out <= SMALL_SURFACE:
        if Y_high is None:
            return Y_low.transpose(0, 1).contiguous()
        # Accumulate in-place into Y_low (saves one temp alloc), then transpose.
        # NB: Y_low is a fresh buffer returned by dense_gemm_u4_s4, so mutating
        # it is safe within v9_linear_forward.
        Y_low.add_(Y_high, alpha=16.0)
        return Y_low.transpose(0, 1).contiguous()

    Y_out = torch.empty((T, d_out), dtype=torch.float16, device=Y_low.device)
    if Y_high is None:
        y_high_ptr = Y_low          # harmless alias; kernel ignores it when HAS_HIGH=False
        stride_h_d, stride_h_t = Y_low.stride(0), Y_low.stride(1)
        has_high = False
    else:
        assert Y_high.shape == Y_low.shape and Y_high.dtype == torch.float16
        y_high_ptr = Y_high
        stride_h_d, stride_h_t = Y_high.stride(0), Y_high.stride(1)
        has_high = True
    grid = lambda META: (triton.cdiv(T, META["BT"]), triton.cdiv(d_out, META["BD"]))
    _combine_transpose_kernel[grid](
        Y_low, y_high_ptr, Y_out,
        T, d_out,
        Y_low.stride(0), Y_low.stride(1),
        stride_h_d, stride_h_t,
        Y_out.stride(0), Y_out.stride(1),
        HAS_HIGH=has_high,
    )
    return Y_out


# ---------------------------------------------------------------------------
# Prefill / Decode dispatcher
# ---------------------------------------------------------------------------
#
# Sweep data (sweep_20260422_154306.csv, 168 shapes on RTX 4090) shows
# two regimes with opposite bottlenecks -- see
# research/analysis_20260422_next_steps.md for the full table:
#
#   Regime        Stage breakdown                        Current speedup
#   ---------     ------------------------------------   ---------------
#   decode        quant 33-44% + sparse 25-27% dominate;  0.47-0.69x
#    (T <= 128)   dense already hits 73% HBM peak at
#                 bs=1, d_out=28672 (memory-bound);
#                 enemy = kernel launch overhead.
#   prefill       dense 83-91% of v9_total; median
#    (T >  128)   dense_ms / fp16_ms = 1.27x;              0.66-0.73x
#                 HBM bw util 1.6-7% (TC underused);
#                 enemy = TC occupancy + dequant overhead.
#
# Sharing one forward path forces us to share one set of autotune configs
# and one pipeline structure, which is sub-optimal for both.  We split
# the entry point into two regime-specific forwards and a cheap runtime
# dispatcher, so future kernel specialisation (Phase B / C in the
# analysis doc) can proceed independently on each side.
#
# This commit only splits the Python-level dispatch; the underlying
# Triton kernels are still shared.  Subsequent commits will customise
# the autotune grids, add a prefill-specific Split-K, and make the
# decode path eligible for CUDA-Graph capture.

# T at which decode transitions to prefill.  Chosen from the stage-share
# table: bs<=128 still has dense < 70% of v9_total (decode-like), bs>=128
# flips to dense-dominated (prefill-like).  Revisit after Phase B/C.
DECODE_T_THRESHOLD = 128


def _v9_forward_decode(
    X_2d: torch.Tensor, W: V9WeightContainer, T: int, d_out: int, d_in: int
) -> torch.Tensor:
    """Decode-regime forward (T <= DECODE_T_THRESHOLD).

    Bottleneck profile (from sweep data):
      - quant kernel: 33-44% of v9_total (launch overhead heavy)
      - sparse kernel (if hp>0): 25-27% of v9_total (same reason)
      - dense: 37-52% of v9_total but already near HBM roof
      - combine: 3-6% (small surface, falls back to torch native)

    For now this body is identical to the monolithic path; it will
    diverge from ``_v9_forward_prefill`` in Phase C as we specialise
    autotune tiles for small T and ultimately capture the whole
    pipeline in a CUDA Graph.
    """
    # (1) Activation quantization
    X_s4, scale_x, sum_X = quantize_activation_s4(X_2d, W.perm, bcol=BCOL)

    # (2) Dense low-bit GEMM -- produces Y_low in (d_out, T) layout.
    Y_low = dense_gemm_u4_s4(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
    )

    # (3) Sparse high-bit GEMM -- skipped entirely when hp=0.
    Y_high: torch.Tensor | None = None
    if W.n_hp_blocks > 0:
        Y_high = sparse_gemm_s4_s4(
            W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, scale_x,
            d_out=d_out, d_in=d_in,
        )

    # (4) Fused combine + transpose.  Decode T is tiny, so this always
    #     falls through to the torch-native fast path inside
    #     _combine_transpose (surf = T * d_out well below the 4M element
    #     threshold), which is what we want here.
    return _combine_transpose(Y_low, Y_high, d_out=d_out, T=T)


def _v9_forward_prefill(
    X_2d: torch.Tensor, W: V9WeightContainer, T: int, d_out: int, d_in: int
) -> torch.Tensor:
    """Prefill-regime forward (T > DECODE_T_THRESHOLD).

    Bottleneck profile (from sweep data):
      - dense: 83-91% of v9_total, median dense_ms / fp16_ms = 1.27x
        -> this is where Phase B (expanded autotune, Split-K, inline
        PTX dequant, explicit K-loop pipelining) will land.
      - sparse (if hp>0): 7-13% of v9_total; real work, not launch bound.
      - quant: 5-16% (prefill N is large, quant kernel already amortises).
      - combine: 4-7%; the fused Triton kernel wins here because
        T * d_out crosses the 4M element threshold.

    W4A16 fallback (Phase B-2, 2026-04-22)
    --------------------------------------
    When the batch is large enough for the GEMM to amortise a one-shot
    weight dequantisation, we switch to: ``W_fp16 = dequant(W); Y = X @
    W_fp16^T`` via cuBLAS. The dedicated Triton dequant kernel
    (``dequant_u4_to_fp16``) runs at ~40x the speed of the torch-native
    ``reconstruct_w_fakequant_fp16`` helper, taking 0.05-0.35 ms for
    common shapes, which is dwarfed by the GEMM work at T >= 512-1024.

    Measured wins (RTX 4090, hp_ratio=0, fp16 dtype):
      shape         bs     int4 GEMM   DQ+FP16    delta
      4096x4096    2048    0.568 ms    0.446 ms   +21%
      4096x4096    8192    2.268 ms    1.763 ms   +22%
      28672x4096   8192    16.36 ms    12.51 ms   +24%
      8192x8192    8192    9.31  ms    7.84  ms   +16%

    Decision rule (conservative, only switch when we have high confidence):
      - hp_ratio > 0           -> always stay on int4 (sparse add-back path
                                  is not yet wired into the fp16 fallback)
      - T >= 1024              -> W4A16 fallback (winner on every shape)
      - 512 <= T < 1024 and
        d_out * d_in <= 4096*4096 -> W4A16 fallback
      - otherwise              -> int4 GEMM (current path)

    The decision only affects dense; quant and combine kernels are
    unchanged because with hp_ratio == 0 the combine stage degenerates
    to ``Y = Y_low.transpose().contiguous()`` (handled below inline).
    """
    # W4A16 fallback eligibility (dense-only). If the weight carries any
    # high-precision sparse blocks we stay on the int4 path -- adding
    # sparse contribution back onto a cuBLAS fp16 GEMM result would
    # require materialising Y in (d_out, T) layout again and we gain
    # nothing over the existing fused combine+transpose pipeline.
    use_w4a16 = (
        W.n_hp_blocks == 0
        and (
            T >= 1024
            or (T >= 512 and (d_out * d_in) <= (4096 * 4096))
        )
    )
    if use_w4a16:
        # One-shot dequant to a dense FP16 weight, then cuBLAS FP16 GEMM.
        #
        # IMPORTANT: V9 stores W_low in *permuted* column order (GPTQ
        # act-order). The int4 path compensates by permuting X inside
        # ``quantize_activation_s4``. In the W4A16 fallback we do not
        # go through that quant kernel, so we must re-permute X here
        # to keep the column alignment consistent.
        W_fp16 = dequant_u4_to_fp16(W)        # (d_out, d_in) in permuted col order
        X_perm = X_2d.index_select(1, W.perm.to(torch.long))  # (T, d_in) permuted
        return torch.nn.functional.linear(X_perm, W_fp16)

    # --- default int4 path -----------------------------------------------------
    # (1) Activation quantization
    X_s4, scale_x, sum_X = quantize_activation_s4(X_2d, W.perm, bcol=BCOL)

    # (2) Dense low-bit GEMM
    Y_low = dense_gemm_u4_s4(
        W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x,
    )

    # (3) Sparse high-bit GEMM
    Y_high: torch.Tensor | None = None
    if W.n_hp_blocks > 0:
        Y_high = sparse_gemm_s4_s4(
            W.W_high_blocks_packed,
            W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, scale_x,
            d_out=d_out, d_in=d_in,
        )

    # (4) Fused combine + transpose: on prefill surfaces (T * d_out >= 4M)
    #     the Triton fused kernel wins against torch .t().contiguous().
    return _combine_transpose(Y_low, Y_high, d_out=d_out, T=T)


def v9_linear_forward(X_fp16: torch.Tensor, W: V9WeightContainer) -> torch.Tensor:
    """V9 Linear forward.  Returns Y_fp16 with shape matching X on all-but-last dim.

    Runtime dispatcher: picks ``_v9_forward_decode`` when the flattened
    batch ``T = numel(X) / d_in`` is ``<= DECODE_T_THRESHOLD`` and
    ``_v9_forward_prefill`` otherwise.  The two regime-specific forwards
    share the same 4-stage pipeline today but will diverge in subsequent
    commits as their autotune grids and kernel choices specialise
    (see research/analysis_20260422_next_steps.md).

    Pipeline (shared, for now):
      (1) per-token SINT4 activation quantization (fused kernel)
      (2) dense UINT4 x SINT4 GEMM   -> Y_low  (d_out, T)
      (3) block-sparse SINT4 x SINT4 GEMM -> Y_high (d_out, T)   [only if hp>0]
      (4) **fused**: Y_out[t, d] = Y_low[d, t] + 16 * Y_high[d, t]
          (single-pass combine + transpose, see ``_combine_transpose_kernel``)

    Rationale for the prior Stage-4 fusion (still valid)
    ----------------------------------------------------
    The former epilogue did two *independent* traversals of the whole
    ``(d_out, T)`` fp16 surface::

        Y_low.add_(Y_high, alpha=16.0)               # 1 load + 1 store
        Y_out = Y_low.transpose(0, 1).contiguous()   # 1 load + 1 store

    ``(d_out * T)`` fp16 is not small: at ``d_out = d_in = 4096, bs = 2048``
    that is 16 MiB touched **four times** end-to-end.  The fused path keeps
    the dense kernel's output layout (critical for store coalescing -- a
    prior attempt to make dense write directly into a ``(T, d_out)`` view
    regressed bs=2048 shapes by ~120% because it spread consecutive N-tile
    stores across ``2 * d_out``-byte strides), and performs *one* pass
    that reads ``Y_low`` (and optionally ``Y_high``), combines, and stores
    directly into the ``(T, d_out)`` final layout.
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
        Y_out = _v9_forward_decode(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    else:
        Y_out = _v9_forward_prefill(X_2d, W, T=T, d_out=d_out, d_in=d_in)

    out_shape = original_shape[:-1] + (d_out,)
    return Y_out.reshape(out_shape)


def v9_linear_forward_decode(
    X_fp16: torch.Tensor, W: V9WeightContainer
) -> torch.Tensor:
    """Explicit decode-path entry for callers that already know they are
    in the decode regime (e.g. serving loops holding T fixed = 1).

    Skips the dispatch branch and any T-threshold overhead, and will be
    the attachment point for future CUDA-Graph capture.  Falls back to
    the dispatcher if called with T > DECODE_T_THRESHOLD (emits a
    warning-free correctness path, no perf promise).
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
        Y_out = _v9_forward_decode(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    else:
        Y_out = _v9_forward_prefill(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    out_shape = original_shape[:-1] + (d_out,)
    return Y_out.reshape(out_shape)


def v9_linear_forward_prefill(
    X_fp16: torch.Tensor, W: V9WeightContainer
) -> torch.Tensor:
    """Explicit prefill-path entry for callers that already know they are
    in the prefill regime (e.g. first forward pass over a long prompt).

    Same shape-handling contract as ``v9_linear_forward``; falls back to
    the decode path if T <= DECODE_T_THRESHOLD so it is always
    correctness-safe.
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
    if T > DECODE_T_THRESHOLD:
        Y_out = _v9_forward_prefill(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    else:
        Y_out = _v9_forward_decode(X_2d, W, T=T, d_out=d_out, d_in=d_in)
    out_shape = original_shape[:-1] + (d_out,)
    return Y_out.reshape(out_shape)


# ---------------------------------------------------------------------------
# Reference: fakequant Linear reconstructed from the packed container
# ---------------------------------------------------------------------------

def reconstruct_w_fakequant_fp16(W: V9WeightContainer) -> torch.Tensor:
    """Rebuild the fp16 dequantized weight (permuted column order) from a V9 pack.

    Useful for cross-checking kernel outputs.  Returns (d_out, d_in) fp16.
    """
    d_out, d_in = W.d_out, W.d_in
    device = W.scale_u4.device

    # Unpack low-bit SINT4 weights -> integer [-8, 7]
    w_low_s4 = unpack_s4_le(W.W_low_packed, signed=True).to(torch.float32)
    zero_fp = W.zero_u4.to(torch.float32)                # already pre-subtracted 8
    scale_fp = W.scale_u4.to(torch.float32)

    bcol = BCOL
    n_groups = d_in // bcol

    # Y_low_contrib per group: (w_low - zero) * scale
    # Broadcast over the bcol columns.
    scale_expand = scale_fp.repeat_interleave(bcol, dim=1)       # (d_out, d_in)
    zero_expand = zero_fp.repeat_interleave(bcol, dim=1)         # (d_out, d_in)
    w_fp_low = (w_low_s4 - zero_expand) * scale_expand

    # Add 16 * W_high contributions.
    w_fp_high = torch.zeros_like(w_fp_low)
    if W.n_hp_blocks > 0:
        w_high_s4 = unpack_s4_le(W.W_high_blocks_packed, signed=True).to(torch.float32)
        # Iterate blocks (Python loop is OK for reference path)
        hp_row_offsets = W.hp_row_offsets.cpu().tolist()
        hp_col_indices = W.hp_col_indices.cpu().tolist()
        nrow = (d_out + W.block_shape[0] - 1) // W.block_shape[0]
        for br in range(nrow):
            s, e = hp_row_offsets[br], hp_row_offsets[br + 1]
            for idx in range(s, e):
                bc = hp_col_indices[idx]
                r0, r1 = br * W.block_shape[0], min((br + 1) * W.block_shape[0], d_out)
                c0, c1 = bc * W.block_shape[1], min((bc + 1) * W.block_shape[1], d_in)
                tile = w_high_s4[idx, : r1 - r0, : c1 - c0]
                # Scale for this block is scale_u4[r0:r1, bc] (bc == group index).
                sc = scale_fp[r0:r1, bc: bc + 1]
                w_fp_high[r0:r1, c0:c1] += 16.0 * tile * sc

    w_fp = (w_fp_low + w_fp_high).to(torch.float16)
    return w_fp


def v9_linear_fakequant(X_fp16: torch.Tensor, W: V9WeightContainer) -> torch.Tensor:
    """Reference Linear forward using the dequantized fp16 weight.

    NB: this consumes the same V9 container so cross-checks are apples-to-apples.
    Reconstruction is expensive; for benchmarking against stock FP16 Linear,
    pass a pre-built `W_fakequant_fp16` instead.
    """
    d_in = W.d_in
    original_shape = X_fp16.shape
    X_2d = X_fp16.reshape(-1, d_in)

    # Permute input columns
    X_perm = X_2d[:, W.perm.to(torch.long)]

    # Reconstruct fakequant weight and quantize activation in fp16 for apples-to-apples.
    W_fp = reconstruct_w_fakequant_fp16(W)          # (d_out, d_in)

    # Quantize activation just like the kernel does (per-token symmetric SINT4),
    # so the reference reflects the same algorithm, not the FP16 upper bound.
    max_abs = X_perm.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale_x = (max_abs / 7.0).to(torch.float16).to(torch.float32)
    q = torch.clamp(torch.round(X_perm.to(torch.float32) / scale_x), -8.0, 7.0)
    X_dequant = (q * scale_x).to(torch.float16)

    Y_2d = X_dequant @ W_fp.t()
    out_shape = original_shape[:-1] + (W.d_out,)
    return Y_2d.reshape(out_shape)


__all__ = [
    "v9_linear_forward",
    "v9_linear_forward_decode",
    "v9_linear_forward_prefill",
    "v9_linear_fakequant",
    "reconstruct_w_fakequant_fp16",
    "DECODE_T_THRESHOLD",
]
