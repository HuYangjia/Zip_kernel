"""CUDA MMA vs Triton parity tests.

For every GEMM kernel we run both MMA variants (INT8 and INT4) against
the Triton reference and assert output equivalence to within
FP32->FP16 rounding tolerance.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

cuda_ops = pytest.importorskip(
    "kernel.cuda_kernel.ops",
    reason="cuda_kernel extension failed to build (non-SM89?)",
)

from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4
from kernel.triton_kernel.sparse_s4s4_gemm import sparse_gemm_s4_s4
from kernel.triton_kernel.fused_dense_sparse_gemm import fused_dense_sparse_gemm
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_s4_le


def _make_dense_inputs(T, d_out, d_in, seed=0xBEEF, device="cuda"):
    torch.manual_seed(seed)
    X = torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device=device)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)

    n_groups = d_in // BCOL
    W_low_s4 = torch.randint(
        -8, 8, (d_out, d_in), dtype=torch.int8, device=device
    )
    W_low_packed = pack_s4_le(W_low_s4)
    scale_u4 = (torch.rand(d_out, n_groups, device=device) * 0.05 + 0.001
                ).to(torch.float16)
    zero_u4 = (torch.randn(d_out, n_groups, device=device) * 0.2
               ).to(torch.float16)
    return W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x


def _make_sparse_inputs(T, d_out, d_in, n_hp_ratio=0.05, seed=0xDEAD,
                        device="cuda"):
    base = _make_dense_inputs(T, d_out, d_in, seed=seed, device=device)
    W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x = base

    nrow = d_out // BROW
    ncol = d_in // BCOL
    total_blocks = nrow * ncol
    n_hp = max(1, int(total_blocks * n_hp_ratio))

    torch.manual_seed(seed ^ 0xA5A5)
    flat = torch.randperm(total_blocks, device=device)[:n_hp]
    br = (flat // ncol).to(torch.int32)
    bc = (flat %  ncol).to(torch.int32)
    order = torch.argsort(br.to(torch.int64) * 100000 + bc.to(torch.int64))
    br_sorted = br[order]
    bc_sorted = bc[order]

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
            f"at index {idx}, cuda={flat_a}, ref={flat_b}"
        )


# ---------------------------------------------------------------------------
# activation_quant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "T,D",
    [
        (1,    4096),
        (16,   4096),
        (128,  4096),
        (2048, 4096),
        (1,    11008),
        (128,  14336),
    ],
)
@pytest.mark.parametrize("perm_kind", ["identity", "random"])
def test_activation_quant_parity(T, D, perm_kind):
    torch.manual_seed(0xC0FFEE)
    X = torch.randn(T, D, dtype=torch.float16, device="cuda") * 0.5
    if perm_kind == "identity":
        perm = torch.arange(D, dtype=torch.int32, device="cuda")
    else:
        perm = torch.randperm(D, device="cuda").to(torch.int32)

    X_s4_t, scale_t, sum_t = quantize_activation_s4(X, perm)
    X_s4_c, scale_c, sum_c = cuda_ops.activation_quant_cuda(X, perm)

    assert torch.equal(X_s4_c, X_s4_t)
    assert torch.equal(scale_c, scale_t)
    assert torch.equal(sum_c, sum_t)


# ---------------------------------------------------------------------------
# dense_gemm (INT8 MMA, INT4 MMA)
# ---------------------------------------------------------------------------


DENSE_SHAPES = [
    (1,    4096, 4096),
    (16,   4096, 4096),
    (128,  4096, 4096),
    (512,  4096, 4096),
    (128,  11008, 4096),
    (128,  4096, 11008),
]


@pytest.mark.parametrize("T,d_out,d_in", DENSE_SHAPES)
@pytest.mark.parametrize(
    "variant",
    ["int8", "int4"],
)
def test_dense_gemm_parity(T, d_out, d_in, variant):
    W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x = _make_dense_inputs(
        T, d_out, d_in
    )
    Y_ref = dense_gemm_u4_s4(W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x)
    if variant == "int8":
        Y_cuda = cuda_ops.dense_gemm_cuda_int8(
            W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
        )
    else:
        Y_cuda = cuda_ops.dense_gemm_cuda_int4(
            W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
        )
    _assert_fp16_close(
        Y_cuda, Y_ref, f"dense_gemm_{variant} T={T} d_out={d_out} d_in={d_in}"
    )


# ---------------------------------------------------------------------------
# sparse_gemm
# ---------------------------------------------------------------------------


SPARSE_CASES = [
    (1,    4096, 4096, 0.05),
    (16,   4096, 4096, 0.05),
    (128,  4096, 4096, 0.10),
    (1,    4096, 4096, 0.00),
    (128,  4096, 4096, 0.20),
]


@pytest.mark.parametrize("T,d_out,d_in,n_hp_ratio", SPARSE_CASES)
@pytest.mark.parametrize("variant", ["int8", "int4"])
def test_sparse_gemm_parity(T, d_out, d_in, n_hp_ratio, variant):
    (
        _W_low, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, _zero_u4, _sum_X, scale_x,
    ) = _make_sparse_inputs(T, d_out, d_in, n_hp_ratio=max(n_hp_ratio, 1/1024))

    if n_hp_ratio == 0.0:
        device = X_s4.device
        nrow = d_out // BROW
        W_high_blocks_packed = torch.zeros(
            (0, BROW, BCOL // 2), dtype=torch.int8, device=device
        )
        hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
        hp_col_indices = torch.zeros((0,), dtype=torch.int32, device=device)

    Y_ref = sparse_gemm_s4_s4(
        W_high_blocks_packed, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, scale_x, d_out, d_in,
    )
    if variant == "int8":
        Y_cuda = cuda_ops.sparse_gemm_cuda_int8(
            W_high_blocks_packed, hp_row_offsets, hp_col_indices,
            X_s4, scale_u4, scale_x, d_out, d_in,
        )
    else:
        Y_cuda = cuda_ops.sparse_gemm_cuda_int4(
            W_high_blocks_packed, hp_row_offsets, hp_col_indices,
            X_s4, scale_u4, scale_x, d_out, d_in,
        )
    _assert_fp16_close(
        Y_cuda, Y_ref, f"sparse_gemm_{variant} T={T} hp={n_hp_ratio}"
    )


# ---------------------------------------------------------------------------
# fused_dense_sparse
# ---------------------------------------------------------------------------


FUSED_CASES = [
    (1,    4096, 4096, 0.05),
    (16,   4096, 4096, 0.05),
    (128,  4096, 4096, 0.10),
    (512,  4096, 4096, 0.05),
]


@pytest.mark.parametrize("T,d_out,d_in,n_hp_ratio", FUSED_CASES)
@pytest.mark.parametrize("variant", ["int8", "int4"])
def test_fused_dense_sparse_parity(T, d_out, d_in, n_hp_ratio, variant):
    (
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
    ) = _make_sparse_inputs(T, d_out, d_in, n_hp_ratio=n_hp_ratio)

    Y_ref = fused_dense_sparse_gemm(
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
        d_out, d_in,
    )
    if variant == "int8":
        Y_cuda = cuda_ops.fused_dense_sparse_cuda_int8(
            W_low_packed, W_high_blocks_packed,
            hp_row_offsets, hp_col_indices,
            X_s4, scale_u4, zero_u4, sum_X, scale_x,
            d_out, d_in,
        )
    else:
        Y_cuda = cuda_ops.fused_dense_sparse_cuda_int4(
            W_low_packed, W_high_blocks_packed,
            hp_row_offsets, hp_col_indices,
            X_s4, scale_u4, zero_u4, sum_X, scale_x,
            d_out, d_in,
        )
    _assert_fp16_close(
        Y_cuda, Y_ref,
        f"fused_{variant} T={T} d_out={d_out} d_in={d_in} hp={n_hp_ratio}"
    )
