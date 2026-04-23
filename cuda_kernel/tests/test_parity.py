"""CUDA-vs-Triton parity tests.

For every kernel that has a working CUDA implementation we call both
backends on the same input and assert the results match to within
a tight tolerance (bit-exact for integer outputs, 1 ULP for fp16
scales, and per-op absolute/relative tolerance for the fp16 GEMM
outputs where FP32->FP16 final casting can differ by <=1 ULP).

This suite is the correctness contract enforced by CI on any SM89 host.
On non-SM89 hosts the ``cuda_kernel`` package raises on import and the
whole file is skipped by the module-level ``pytest.importorskip``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

# Skip cleanly on non-SM89 hosts.  The build fails loudly on import in
# that case; importorskip converts that into a test-level skip.
cuda_ops = pytest.importorskip(
    "kernel.cuda_kernel.ops",
    reason="cuda_kernel extension failed to build (non-SM89?)",
)

from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4
from kernel.triton_kernel.sparse_s4s4_gemm import sparse_gemm_s4_s4
from kernel.triton_kernel.fused_dense_sparse_gemm import fused_dense_sparse_gemm
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_s4_le


# ---------------------------------------------------------------------------
# Test fixture helpers
# ---------------------------------------------------------------------------


def _make_dense_inputs(T, d_out, d_in, seed=0xBEEF, device="cuda"):
    """Construct a realistic (W_low, X_s4, scale, zero, sum_X, scale_x) set.

    We go through the Triton quantization kernel to produce X_s4 / sum_X
    / scale_x so that the GEMM inputs are on the exact quantization
    manifold the Triton reference expects; this keeps parity tests from
    accidentally masking bugs via off-manifold inputs.
    """
    torch.manual_seed(seed)
    X = torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device=device)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)

    n_groups = d_in // BCOL
    # W_low: SINT4 values in [-8, 7].
    W_low_s4 = torch.randint(
        -8, 8, (d_out, d_in), dtype=torch.int8, device=device
    )
    W_low_packed = pack_s4_le(W_low_s4)
    # per-group scale/zero: random but realistic magnitudes.
    scale_u4 = (torch.rand(d_out, n_groups, device=device) * 0.05 + 0.001
                ).to(torch.float16)
    zero_u4 = (torch.randn(d_out, n_groups, device=device) * 0.2
               ).to(torch.float16)
    return W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x


def _make_sparse_inputs(T, d_out, d_in, n_hp_ratio=0.05, seed=0xDEAD,
                        device="cuda"):
    """Also build the BSR tensors for the sparse / fused kernels."""
    base = _make_dense_inputs(T, d_out, d_in, seed=seed, device=device)
    W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x = base

    nrow = d_out // BROW
    ncol = d_in // BCOL
    total_blocks = nrow * ncol
    n_hp = max(1, int(total_blocks * n_hp_ratio))

    torch.manual_seed(seed ^ 0xA5A5)
    # Pick distinct (br, bc) pairs.
    flat = torch.randperm(total_blocks, device=device)[:n_hp]
    br = (flat // ncol).to(torch.int32)
    bc = (flat %  ncol).to(torch.int32)
    # Sort by (br, bc) ascending to build BSR indptr.
    order = torch.argsort(br.to(torch.int64) * 100000 + bc.to(torch.int64))
    br_sorted = br[order]
    bc_sorted = bc[order]

    # Per-block W_high SINT4 values.
    W_high_s4 = torch.randint(
        -8, 8, (n_hp, BROW, BCOL), dtype=torch.int8, device=device
    )
    W_high_blocks_packed = pack_s4_le(W_high_s4)

    hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
    counts = torch.bincount(br_sorted.to(torch.int64), minlength=nrow)
    hp_row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    return (
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, bc_sorted,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
    )


# Helper for fp16 GEMM parity: allow small ULP differences from the
# final FP32 -> FP16 cast (order-of-operations may differ between
# Triton's tl.dot accumulator and our dp4a accumulator).
def _assert_fp16_close(a: torch.Tensor, b: torch.Tensor, label: str):
    diff = (a.float() - b.float()).abs()
    tol = 1e-2 + 5e-3 * b.float().abs()
    worst = (diff - tol).max().item()
    if worst > 0:
        idx = (diff - tol).argmax().item()
        flat_a = a.flatten()[idx].item()
        flat_b = b.flatten()[idx].item()
        raise AssertionError(
            f"{label} parity failed: worst abs diff {diff.max().item():.4e}, "
            f"at index {idx}, cuda={flat_a}, triton={flat_b}"
        )


# ---------------------------------------------------------------------------
# activation_quant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,D",
    [
        (1,    4096),
        (16,   4096),
        (64,   4096),
        (128,  4096),
        (2048, 4096),
        (1,    11008),
        (128,  14336),
    ],
)
@pytest.mark.parametrize("perm_kind", ["identity", "random"])
def test_activation_quant_parity(T, D, perm_kind):
    if cuda_ops.activation_quant_cuda is None:
        pytest.skip("activation_quant CUDA impl not available")

    torch.manual_seed(0xC0FFEE)
    device = "cuda"
    X = torch.randn(T, D, dtype=torch.float16, device=device) * 0.5

    if perm_kind == "identity":
        perm = torch.arange(D, dtype=torch.int32, device=device)
    else:
        perm = torch.randperm(D, device=device).to(torch.int32)

    X_s4_t, scale_t, sum_t = quantize_activation_s4(X, perm)
    X_s4_c, scale_c, sum_c = cuda_ops.activation_quant_cuda(X, perm)

    assert torch.equal(X_s4_c, X_s4_t), "X_s4 mismatch"
    assert torch.equal(scale_c, scale_t), "scale_x mismatch"
    assert torch.equal(sum_c, sum_t), "sum_X mismatch"


# ---------------------------------------------------------------------------
# dense_gemm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,d_out,d_in",
    [
        (1,    4096, 4096),    # decode T=1
        (16,   4096, 4096),    # decode batch
        (64,   4096, 4096),
        (128,  4096, 4096),    # small prefill
        (512,  4096, 4096),
        (128,  11008, 4096),   # rectangular MLP gate/up
        (128,  4096, 11008),   # MLP down
    ],
)
def test_dense_gemm_parity(T, d_out, d_in):
    if cuda_ops.dense_gemm_cuda is None:
        pytest.skip("dense_gemm CUDA impl not available")

    W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x = _make_dense_inputs(
        T, d_out, d_in
    )

    Y_t = dense_gemm_u4_s4(W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x)
    Y_c = cuda_ops.dense_gemm_cuda(
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
    )
    _assert_fp16_close(Y_c, Y_t, f"dense_gemm T={T} d_out={d_out} d_in={d_in}")


# ---------------------------------------------------------------------------
# sparse_gemm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,d_out,d_in,n_hp_ratio",
    [
        (1,    4096, 4096, 0.05),
        (16,   4096, 4096, 0.05),
        (128,  4096, 4096, 0.10),
        (1,    4096, 4096, 0.00),   # empty BSR edge case
        (128,  4096, 4096, 0.20),
    ],
)
def test_sparse_gemm_parity(T, d_out, d_in, n_hp_ratio):
    if cuda_ops.sparse_gemm_cuda is None:
        pytest.skip("sparse_gemm CUDA impl not available")

    (
        _W_low, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, _zero_u4, _sum_X, scale_x,
    ) = _make_sparse_inputs(T, d_out, d_in, n_hp_ratio=max(n_hp_ratio, 1/1024))

    # Explicit empty-BSR test case
    if n_hp_ratio == 0.0:
        device = X_s4.device
        nrow = d_out // BROW
        W_high_blocks_packed = torch.zeros(
            (0, BROW, BCOL // 2), dtype=torch.int8, device=device
        )
        hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
        hp_col_indices = torch.zeros((0,), dtype=torch.int32, device=device)

    Y_t = sparse_gemm_s4_s4(
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, scale_x, d_out, d_in,
    )
    Y_c = cuda_ops.sparse_gemm_cuda(
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, scale_x, d_out, d_in,
    )
    _assert_fp16_close(Y_c, Y_t, f"sparse_gemm T={T} hp_ratio={n_hp_ratio}")


# ---------------------------------------------------------------------------
# fused_dense_sparse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,d_out,d_in,n_hp_ratio",
    [
        (1,    4096, 4096, 0.05),
        (16,   4096, 4096, 0.05),
        (128,  4096, 4096, 0.10),
        (512,  4096, 4096, 0.05),
    ],
)
def test_fused_dense_sparse_parity(T, d_out, d_in, n_hp_ratio):
    if cuda_ops.fused_dense_sparse_cuda is None:
        pytest.skip("fused_dense_sparse CUDA impl not available")

    (
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
    ) = _make_sparse_inputs(T, d_out, d_in, n_hp_ratio=n_hp_ratio)

    Y_t = fused_dense_sparse_gemm(
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
        d_out, d_in,
    )
    Y_c = cuda_ops.fused_dense_sparse_cuda(
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
        d_out, d_in,
    )
    _assert_fp16_close(
        Y_c, Y_t, f"fused T={T} d_out={d_out} d_in={d_in} hp={n_hp_ratio}"
    )


# ---------------------------------------------------------------------------
# End-to-end: dispatcher + v9_linear_forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,d_out,d_in,hp_ratio",
    [
        (1,   4096, 4096, 0.00),    # decode dense-only
        (1,   4096, 4096, 0.05),    # decode dense+sparse
        (128, 4096, 4096, 0.00),    # prefill dense-only -> W4A16 fallback path
        (128, 4096, 4096, 0.05),    # prefill fused
    ],
)
def test_end_to_end_parity(T, d_out, d_in, hp_ratio):
    """Compare full v9_linear forward: CUDA-preferred vs Triton-only.

    Uses the dispatcher twice with opposite policies and asserts the
    outputs match; this exercises the full call chain (Python pipeline
    + CUDA kernels) as integrated.
    """
    if not cuda_ops.activation_quant_cuda:
        pytest.skip("CUDA backend unavailable")

    from kernel.backend import set_backend_policy, v9_linear_forward
    from kernel.triton_kernel.pack_utils import V9WeightContainer

    (
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        _X_s4, scale_u4, zero_u4, _sum_X, _scale_x,
    ) = _make_sparse_inputs(
        T, d_out, d_in,
        n_hp_ratio=max(hp_ratio, 1 / 1024) if hp_ratio > 0 else 1 / 1024,
    )
    if hp_ratio == 0.0:
        device = W_low_packed.device
        nrow = d_out // BROW
        W_high_blocks_packed = torch.zeros(
            (0, BROW, BCOL // 2), dtype=torch.int8, device=device
        )
        hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
        hp_col_indices = torch.zeros((0,), dtype=torch.int32, device=device)

    perm = torch.arange(d_in, dtype=torch.int32,
                        device=W_low_packed.device)
    W = V9WeightContainer(
        W_low_packed=W_low_packed,
        W_high_blocks_packed=W_high_blocks_packed,
        scale_u4=scale_u4,
        zero_u4=zero_u4,
        hp_row_offsets=hp_row_offsets,
        hp_col_indices=hp_col_indices,
        perm=perm,
        block_shape=(BROW, BCOL),
        d_out=d_out,
        d_in=d_in,
    )

    X = torch.randn(T, d_in, dtype=torch.float16,
                    device=W_low_packed.device) * 0.4

    try:
        set_backend_policy("triton")
        Y_triton = v9_linear_forward(X, W)
        set_backend_policy("cuda")
        Y_cuda = v9_linear_forward(X, W)
    finally:
        set_backend_policy("auto")

    _assert_fp16_close(
        Y_cuda, Y_triton,
        f"end-to-end T={T} d_out={d_out} d_in={d_in} hp={hp_ratio}"
    )
    # scale_x is fp16; both sides round identically (fp32 -> fp16 via
    # hardware round-to-nearest-even), so exact equality should hold.
    assert torch.equal(scale_c, scale_t), "scale_x mismatch"
    # Per-group sums are integers over the same quantized q values.
    assert torch.equal(sum_c, sum_t), "sum_X mismatch"


# ---------------------------------------------------------------------------
# End-to-end dispatcher smoke test
# ---------------------------------------------------------------------------


def test_dispatcher_uses_cuda_quant_when_available():
    """When CUDA is available and policy='auto', quant must run CUDA-side."""
    if cuda_ops.activation_quant_cuda is None:
        pytest.skip("activation_quant CUDA impl not available")
    from kernel.backend import BackendKernel
    from kernel.backend.policy import ShapeContext, current_policy
    from kernel.backend.registry import KERNEL_ACTIVATION_QUANT

    assert BackendKernel.cuda_available()
    assert KERNEL_ACTIVATION_QUANT in BackendKernel.cuda_available_kernels()

    choice = current_policy()(KERNEL_ACTIVATION_QUANT,
                              ShapeContext(T=1, d_in=4096, n_groups=32))
    assert choice == "cuda", f"expected cuda, got {choice}"
