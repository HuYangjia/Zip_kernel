"""Minimal diagnosis script: one shape, dump Y_low-only (dense) and Y_high-only
(sparse) from L1 naive vs reference (via optimised fused path), print the
first non-zero mismatches.  Helps pinpoint whether the bug is in dense
kernel, sparse kernel, or both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from kernel.cuda_kernel import ops as opt_ops
from kernel.cuda_kernel import ops_naive as naive_ops


def main() -> int:
    device = torch.device("cuda:0")
    # Pick one small shape that failed with sparse.
    T, d_in, d_out, density = 16, 4096, 2048, 0.05

    torch.manual_seed(42)
    X = (torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4).contiguous()
    perm = torch.randperm(d_in, device=device).to(torch.int32).contiguous()
    W_low = torch.randint(0, 16, (d_out, d_in // 2),
                          dtype=torch.int8, device=device).contiguous()
    n_g = d_in // 128
    scale_u4 = (torch.rand(d_out, n_g, dtype=torch.float16, device=device)
                * 0.01 + 0.001).contiguous()
    zero_u4  = (torch.rand(d_out, n_g, dtype=torch.float16, device=device)
                * 14.0).contiguous()

    # --- Build BSR the same way parity does (seed by d_out ^ d_in) ---
    n_row_blocks = d_out // 128
    n_groups     = d_in  // 128
    gen = torch.Generator(device="cpu").manual_seed((d_out * 2654435761 ^ d_in) & 0x7FFFFFFF)
    mask = torch.rand((n_row_blocks, n_groups), generator=gen, dtype=torch.float32) < density
    for r in range(n_row_blocks):
        if not mask[r].any():
            c = int(torch.randint(0, n_groups, (1,), generator=gen, dtype=torch.int64).item())
            mask[r, c] = True
    row_off = torch.zeros(n_row_blocks + 1, dtype=torch.int32)
    col_idx_list = []
    for r in range(n_row_blocks):
        cs = torch.nonzero(mask[r], as_tuple=False).flatten().to(torch.int32)
        col_idx_list.append(cs)
        row_off[r + 1] = row_off[r] + len(cs)
    col_idx = torch.cat(col_idx_list) if col_idx_list else torch.zeros(0, dtype=torch.int32)
    n_blocks = int(row_off[-1].item())
    W_high = torch.randint(0, 256, (n_blocks, 128, 64),
                           dtype=torch.int64, device="cpu", generator=gen).to(torch.int8)
    W_high_blocks  = W_high.to(device).contiguous()
    hp_row_offsets = row_off.to(device).contiguous()
    hp_col_indices = col_idx.to(device).contiguous()

    # --- quant ---
    X_s4, scale_x, sum_X = naive_ops.activation_quant_naive(X, perm)

    # --- PATH 1: reference, using optimised kernels individually ---
    opt_ext = opt_ops._ext
    Y_low_ref  = torch.empty((d_out, T), dtype=torch.float16, device=device)
    opt_ext.dense_gemm_mma_int4_launch(W_low, X_s4, scale_u4, zero_u4,
                                        sum_X, scale_x, Y_low_ref)
    Y_high_ref = torch.empty((d_out, T), dtype=torch.float16, device=device)
    opt_ext.sparse_gemm_mma_int4_launch(W_high_blocks, hp_row_offsets, hp_col_indices,
                                         X_s4, scale_u4, scale_x, Y_high_ref,
                                         d_out, d_in)

    # --- PATH 1b: fused optimised kernel (the one parity uses as reference) ---
    Y_fused = opt_ops.fused_dense_sparse_cuda_int4(
        W_low, W_high_blocks, hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
    )

    # --- PATH 2: naive L1 ---
    Y_low_nai  = naive_ops.dense_gemm_naive(W_low, X_s4, scale_u4, zero_u4,
                                             sum_X, scale_x)
    Y_high_nai = naive_ops.sparse_gemm_naive(W_high_blocks, hp_row_offsets, hp_col_indices,
                                              X_s4, scale_u4, scale_x, d_out, d_in)

    def stat(name, A, B):
        diff = (A.float() - B.float()).abs()
        print(f"{name:10s}: max_abs={diff.max().item():.4g}  "
              f"mean_abs={diff.mean().item():.4g}  "
              f"max_ref={A.abs().max().item():.4g}")
        # find first 5 worst mismatches
        flat = diff.flatten()
        top = torch.topk(flat, 5)
        print(f"  top-5 diff @ flat idx {top.indices.tolist()}")
        for idx in top.indices[:5].tolist():
            r, c = divmod(idx, A.shape[1])
            print(f"    [{r},{c}] ref={A[r,c].item():+.4f} "
                  f"naive={B[r,c].item():+.4f} diff={diff[r,c].item():.4g}")

    print("=" * 60)
    print(f"shape T={T} d_in={d_in} d_out={d_out} density={density}")
    print(f"n_blocks={n_blocks}")
    print("-" * 60)
    print(">>> Y_low (dense-only) comparison:")
    stat("Y_low", Y_low_ref, Y_low_nai)
    print(">>> Y_high (sparse-only) comparison:")
    stat("Y_high", Y_high_ref, Y_high_nai)
    print()
    # Also: what fraction of Y_high is zero (rows not in any block)?
    zero_rows = (Y_high_ref.abs().sum(dim=1) == 0).sum().item()
    print(f"Y_high rows that are exactly 0 (ref): {zero_rows} / {d_out}")

    # Now: is fused == dense+sparse (optimised independent kernels)?
    Y_sum_ref = (Y_low_ref.float() + Y_high_ref.float()).to(torch.float16)
    print(">>> fused vs (dense+sparse) (both using OPTIMISED kernels):")
    stat("fused-sep", Y_fused, Y_sum_ref)

    # And: naive dense+sparse vs fused?
    Y_total_nai = naive_ops.reduce_sum_naive(Y_low_nai, Y_high_nai)
    print(">>> fused vs naive Y_total (parity's actual comparison):")
    stat("fused-nai", Y_fused, Y_total_nai)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
