"""Kernel (1b) - Dense UINT4 x SINT4 GEMM with fused transpose-to-output.

Drop-in replacement for the pair

    Y_low = dense_gemm_u4_s4(...)                            # (d_out, T)
    Y_out = Y_low.transpose(0, 1).contiguous()               # (T, d_out)

when sparse high-bit blocks are absent (``n_hp_blocks == 0``).  Saves:
  * one kernel launch (+ its autotune dispatch cost)
  * one HBM write + read of the ``(d_out, T)`` FP16 ``Y_low`` surface
  * the ``_combine_transpose`` / ``.t().contiguous()`` pass over that surface

The GEMM main loop is identical to ``dense_gemm_kernel``; only the epilogue
differs.  We transpose the FP32 accumulator in-place (tl.trans) and store
into ``Y_out[offs_n, offs_m]`` with stride ``(d_out, 1)`` so that the last
axis of each store tile walks contiguous memory.  This mirrors the
``_combine_transpose_kernel`` layout choice but merges it into the GEMM
kernel itself.

Shape conventions (identical to dense_u4s4_gemm):
  - W_low_packed : (d_out, d_in // 2) int8   SINT4 packed, little-endian
  - X_s4         : (T,     d_in // 2) int8   SINT4 packed, little-endian
  - scale_u4     : (d_out, n_groups)  fp16
  - zero_u4      : (d_out, n_groups)  fp16   (pre-subtracted 8)
  - sum_X        : (T,     n_groups)  int32
  - scale_x      : (T,)                fp16
  - Y_out        : (T,     d_out)      fp16   <-- NEW layout vs dense_gemm_kernel

Why a separate file (not a flag on the existing kernel)?
--------------------------------------------------------
``tl.constexpr`` dispatch on the store layout works in principle but
Triton's autotune cache is keyed on the source AST hash; toggling a
constexpr at call-sites splits the cache and caused 10--20 us of extra
compilation in prior attempts.  A dedicated kernel keeps both cache
lines hot independently.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .dense_u4s4_gemm import _unpack_packed_s4_rowmajor
from .pack_utils import BCOL


# ---------------------------------------------------------------------------
# Triton kernel - dense + fused transpose, hp=0 path
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # --- decode regime (small N/T): BN tiny, M medium.  Same tile
        #     shapes as dense_gemm_kernel's decode configs -- the only
        #     difference vs dense_gemm_kernel is the epilogue, which has
        #     no effect on optimal tile shape in the K-loop.
        triton.Config({"BM": 64,  "BN": 16,  "BK": 128, "GROUP_SIZE_M": 1}, num_warps=2, num_stages=3),
        triton.Config({"BM": 128, "BN": 16,  "BK": 128, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 32,  "BK": 128, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=3),
        # --- mid regime (T in 32..128): square-ish tiles + L2 swizzle.
        triton.Config({"BM": 64,  "BN": 64,  "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 64,  "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64,  "BN": 128, "BK": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3),
    ],
    key=["d_out", "d_in", "T"],
)
@triton.jit
def dense_gemm_to_out_kernel(
    W_low_ptr,          # (d_out, d_in // 2) int8 packed SINT4
    X_s4_ptr,           # (T,     d_in // 2) int8 packed SINT4
    scale_u4_ptr,       # (d_out, n_groups) fp16
    zero_u4_ptr,        # (d_out, n_groups) fp16
    sum_X_ptr,          # (T,     n_groups) int32
    scale_x_ptr,        # (T,)    fp16
    Y_out_ptr,          # (T, d_out) fp16   <-- NEW layout
    d_out, d_in, T,
    stride_w_m, stride_w_k,
    stride_x_n, stride_x_k,
    stride_su_m, stride_su_g,
    stride_zu_m, stride_zu_g,
    stride_sx_n, stride_sx_g,
    stride_yo_t, stride_yo_d,   # Y_out strides (stride_yo_d is 1 for contiguous)
    N_GROUPS: tl.constexpr,
    BCOL_K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ------------------------------------------------------------------
    # Grouped program-ID swizzle (L2 locality) -- same as dense_gemm_kernel.
    # Re-derived to keep this kernel self-contained; diverges from the
    # parent only in the epilogue below.
    # ------------------------------------------------------------------
    pid_m_raw = tl.program_id(0)
    pid_n_raw = tl.program_id(1)
    num_pid_m = tl.cdiv(d_out, BM)
    num_pid_n = tl.cdiv(T,     BN)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    pid = pid_m_raw * num_pid_n + pid_n_raw
    first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < d_out
    mask_n = offs_n < T

    # scale_x[j] is constant across the whole K loop.
    sx = tl.load(scale_x_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    # Final accumulator in FP32 for dequant output.
    y_acc = tl.zeros((BM, BN), dtype=tl.float32)

    BK_HALF: tl.constexpr = BK // 2
    tl.static_assert(BK == BCOL_K, "BK must equal BCOL_K (group size) for per-group epilogue")

    offs_k_half = tl.arange(0, BK_HALF)

    n_k_iters = tl.cdiv(d_in, BK)
    for k_block in range(0, n_k_iters):
        k_byte_start = k_block * BK_HALF
        k_bytes = k_byte_start + offs_k_half
        k_bytes_mask = k_bytes < (d_in // 2)

        # Load packed W tile (BM, BK_HALF)
        w_ptrs = W_low_ptr + offs_m[:, None] * stride_w_m + k_bytes[None, :] * stride_w_k
        w_packed = tl.load(
            w_ptrs,
            mask=mask_m[:, None] & k_bytes_mask[None, :],
            other=0,
        )
        # Load packed X tile (BN, BK_HALF)
        x_ptrs = X_s4_ptr + offs_n[:, None] * stride_x_n + k_bytes[None, :] * stride_x_k
        x_packed = tl.load(
            x_ptrs,
            mask=mask_n[:, None] & k_bytes_mask[None, :],
            other=0,
        )

        # Unpack to SINT4 int8 tiles (BM, BK) and (BN, BK)
        w_tile = _unpack_packed_s4_rowmajor(w_packed, BM, BK)
        x_tile = _unpack_packed_s4_rowmajor(x_packed, BN, BK)

        x_tile_t = tl.trans(x_tile)
        acc_g = tl.dot(w_tile, x_tile_t, out_dtype=tl.int32)

        # Per-group dequant.
        g = k_block
        scale_g = tl.load(
            scale_u4_ptr + offs_m * stride_su_m + g * stride_su_g,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        zero_g = tl.load(
            zero_u4_ptr + offs_m * stride_zu_m + g * stride_zu_g,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        sum_X_g = tl.load(
            sum_X_ptr + offs_n * stride_sx_n + g * stride_sx_g,
            mask=mask_n,
            other=0,
        ).to(tl.float32)

        corrected = acc_g.to(tl.float32) - zero_g[:, None] * sum_X_g[None, :]
        y_acc += corrected * scale_g[:, None] * sx[None, :]

    # ------------------------------------------------------------------
    # Epilogue: transpose and write Y_out[offs_n, offs_m].
    #
    # We transpose the (BM, BN) accumulator to (BN, BM) *in registers* via
    # tl.trans, then compute the store pointer using Y_out strides:
    #
    #   Y_out[t, d] = y_out_tile[t - n0, d - m0]
    #
    # With stride_yo_d == 1 (Y_out contiguous on the d_out axis) the
    # right-most axis of the store pointer walks contiguous memory, so
    # a warp worth of threads emit a single coalesced HBM sector per
    # instruction.  This is the same layout trick used in
    # ``_combine_transpose_kernel``, just fused into the GEMM store.
    # ------------------------------------------------------------------
    y_tile = y_acc.to(tl.float16)                       # (BM, BN)
    y_tile_t = tl.trans(y_tile)                         # (BN, BM)
    y_out_ptrs = (
        Y_out_ptr
        + offs_n[:, None] * stride_yo_t
        + offs_m[None, :] * stride_yo_d
    )
    tl.store(
        y_out_ptrs,
        y_tile_t,
        mask=mask_n[:, None] & mask_m[None, :],
    )


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def dense_gemm_u4_s4_to_out(
    W_low_packed: torch.Tensor,
    X_s4: torch.Tensor,
    scale_u4: torch.Tensor,
    zero_u4: torch.Tensor,
    sum_X: torch.Tensor,
    scale_x: torch.Tensor,
    T: int | None = None,
    d_out: int | None = None,
) -> torch.Tensor:
    """Dense UINT4 x SINT4 GEMM with fused transpose-to-output (hp=0 path).

    Returns ``Y_out`` with shape ``(T, d_out)`` fp16, already in the layout
    expected by downstream code -- no further transpose / contiguous pass
    needed.

    For callers that already know ``T`` / ``d_out`` (e.g. v9_linear), they
    can be passed explicitly to skip the shape-sniff overhead; otherwise
    they are inferred from the input tensors.
    """
    assert W_low_packed.is_cuda and X_s4.is_cuda
    assert W_low_packed.dtype == torch.int8 and X_s4.dtype == torch.int8
    _d_out, d_in_half = W_low_packed.shape
    _T, d_in_half_x = X_s4.shape
    assert d_in_half == d_in_half_x, "d_in mismatch between W and X"
    d_in = d_in_half * 2
    if T is None:
        T = _T
    else:
        assert T == _T
    if d_out is None:
        d_out = _d_out
    else:
        assert d_out == _d_out

    bcol = BCOL
    n_groups = d_in // bcol
    assert scale_u4.shape == (d_out, n_groups)
    assert zero_u4.shape == (d_out, n_groups)
    assert sum_X.shape == (T, n_groups)
    assert scale_x.shape == (T,)

    W_low_packed = W_low_packed.contiguous()
    X_s4 = X_s4.contiguous()
    scale_u4 = scale_u4.contiguous().to(torch.float16)
    zero_u4 = zero_u4.contiguous().to(torch.float16)
    sum_X = sum_X.contiguous().to(torch.int32)
    scale_x = scale_x.contiguous().to(torch.float16)

    # Allocate (T, d_out) fp16, row-major contiguous -> stride_yo_d == 1.
    Y_out = torch.empty((T, d_out), dtype=torch.float16, device=W_low_packed.device)

    grid = lambda META: (triton.cdiv(d_out, META["BM"]), triton.cdiv(T, META["BN"]))
    dense_gemm_to_out_kernel[grid](
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, Y_out,
        d_out, d_in, T,
        W_low_packed.stride(0), W_low_packed.stride(1),
        X_s4.stride(0), X_s4.stride(1),
        scale_u4.stride(0), scale_u4.stride(1),
        zero_u4.stride(0), zero_u4.stride(1),
        sum_X.stride(0), sum_X.stride(1),
        Y_out.stride(0), Y_out.stride(1),
        N_GROUPS=n_groups,
        BCOL_K=bcol,
    )
    return Y_out


__all__ = ["dense_gemm_to_out_kernel", "dense_gemm_u4_s4_to_out"]
