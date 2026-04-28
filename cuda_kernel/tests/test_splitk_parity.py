"""R34 Split-K path parity test.

Uses the same Triton reference as test_parity.py, but specifically
targets shapes that trigger the Split-K dispatch path (base_grid < 128
at kBn=32 and n_groups >= 16).
"""
import sys
sys.path.insert(0, '/root')

import torch

from kernel.cuda_kernel import ops
from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4
from kernel.triton_kernel.pack_utils import BCOL, pack_s4_le


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


SHAPES = [
    # (T, d_out, d_in, label) -- all chosen to hit Split-K path.
    (32,  1024, 4096,  "kv_T32"),    # base=8*1=8 CTAs  -> split=16 cap 4 -> 4
    (64,  1024, 4096,  "kv_T64"),    # base=8*2=16 CTAs -> split=8
    (128, 1024, 4096,  "kv_T128"),   # base=8*4=32 CTAs -> split=4
    (32,  4096, 4096,  "q_T32"),     # base=32*1=32 CTAs -> split=4
    (64,  4096, 4096,  "q_T64"),     # base=32*2=64 CTAs -> split=2
    (128, 4096, 4096,  "q_T128"),    # base=32*4=128 -> NO SPLIT (sanity check)
    (32,  4096, 12288, "down_T32"),  # base=32*1=32 CTAs, n_groups=96 -> split=4
    (128, 4096, 12288, "down_T128"), # base=32*4=128 CTAs -> NO SPLIT (kBn=64 preferred)
]


ok = True
for T, d_out, d_in, label in SHAPES:
    W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x = _make_dense_inputs(
        T, d_out, d_in
    )
    Y_ref = dense_gemm_u4_s4(
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
    )
    Y_cu = ops.dense_gemm_cuda_int4(
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
    )
    diff = (Y_cu.float() - Y_ref.float()).abs()
    tol = 1e-2 + 5e-3 * Y_ref.float().abs()
    n_bad = (diff > tol).sum().item()
    tot = diff.numel()
    status = "OK " if n_bad == 0 else "FAIL"
    max_abs = diff.max().item()
    if n_bad:
        ok = False
    print(f"  [{status}] {label:12s} T={T:4d} d_out={d_out:5d} d_in={d_in:5d}"
          f"  max_abs={max_abs:.4f}  bad={n_bad}/{tot}")

print("\nAll OK" if ok else "\nSome shapes FAILED")
