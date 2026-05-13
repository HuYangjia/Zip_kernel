"""Smoke benchmark: one shape (Qwen3-8B O-proj, T=4 / decode bz=4).

Compares the latency of the 4 L1-naive kernels (quant / dense GEMM /
sparse GEMM / add) against the fused optimised CUDA path, using the
strict GPU micro-benchmarking protocol (warmup + outer-of-min + inner
mean).  This is a sanity check, not the full Qwen3 sweep.
"""
from __future__ import annotations

import sys
import statistics
from pathlib import Path

import torch

_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from kernel.cuda_kernel import ops as opt_ops
from kernel.cuda_kernel import ops_naive as naive_ops


# -------------------------------------------------------------------
# timing helper (matches kernel/tools/profile/_phase1_shapes.py protocol)
# -------------------------------------------------------------------
def time_fn_us(fn, warmup: int = 200, outer: int = 10, inner: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    round_means = []
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        round_means.append(s.elapsed_time(e) * 1e3 / inner)   # us
    return min(round_means)


def main() -> int:
    device = torch.device("cuda:0")

    # Qwen3-8B O-proj:  X(T, d_in) -> Y(d_out, T)
    # d_model = 4096, so O-proj is 4096x4096.
    T, d_in, d_out = 4, 4096, 4096
    density = 0.05   # ~5% BSR blocks

    torch.manual_seed(0)

    # ---- input activation (fp16) ----
    X = (torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4).contiguous()
    perm = torch.randperm(d_in, device=device).to(torch.int32).contiguous()

    # ---- W_low (UINT4 packed) + per-group scale/zero ----
    W_low = torch.randint(0, 16, (d_out, d_in // 2),
                          dtype=torch.int8, device=device).contiguous()
    n_g = d_in // 128
    scale_u4 = (torch.rand(d_out, n_g, dtype=torch.float16, device=device)
                * 0.01 + 0.001).contiguous()
    zero_u4  = (torch.rand(d_out, n_g, dtype=torch.float16, device=device)
                * 14.0).contiguous()

    # ---- BSR W_high ----
    n_row_blocks = d_out // 128
    n_groups     = d_in  // 128
    gen = torch.Generator(device="cpu").manual_seed(0xC0DE)
    mask = torch.rand((n_row_blocks, n_groups), generator=gen,
                      dtype=torch.float32) < density
    for r in range(n_row_blocks):
        if not mask[r].any():
            c = int(torch.randint(0, n_groups, (1,), generator=gen,
                                  dtype=torch.int64).item())
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

    print("=" * 70)
    print(f"Shape: Qwen3-8B O-proj   T={T}  d_in={d_in}  d_out={d_out}  "
          f"density={density}  n_blocks={n_blocks}")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Timing: L1 naive 4 kernels
    # ---------------------------------------------------------------
    def run_quant():
        return naive_ops.activation_quant_naive(X, perm)

    X_s4_nai, scale_x_nai, sum_X_nai = run_quant()

    def run_dense():
        return naive_ops.dense_gemm_naive(
            W_low, X_s4_nai, scale_u4, zero_u4, sum_X_nai, scale_x_nai)

    def run_sparse():
        return naive_ops.sparse_gemm_naive(
            W_high_blocks, hp_row_offsets, hp_col_indices,
            X_s4_nai, scale_u4, scale_x_nai, d_out, d_in)

    Y_low_nai  = run_dense()
    Y_high_nai = run_sparse()

    def run_add():
        return naive_ops.reduce_sum_naive(Y_low_nai, Y_high_nai)

    t_quant  = time_fn_us(run_quant)
    t_dense  = time_fn_us(run_dense)
    t_sparse = time_fn_us(run_sparse)
    t_add    = time_fn_us(run_add)
    t_naive_total = t_quant + t_dense + t_sparse + t_add

    # ---------------------------------------------------------------
    # Timing: optimised fused
    # ---------------------------------------------------------------
    X_s4_opt, scale_x_opt, sum_X_opt = opt_ops.activation_quant_cuda(X, perm)

    def run_opt_quant():
        return opt_ops.activation_quant_cuda(X, perm)

    def run_opt_fused():
        return opt_ops.fused_dense_sparse_cuda_int4(
            W_low, W_high_blocks, hp_row_offsets, hp_col_indices,
            X_s4_opt, scale_u4, zero_u4, sum_X_opt, scale_x_opt,
            d_out, d_in,
        )

    t_opt_quant = time_fn_us(run_opt_quant)
    t_opt_fused = time_fn_us(run_opt_fused)
    t_opt_total = t_opt_quant + t_opt_fused

    # ---------------------------------------------------------------
    # FP16 cuBLAS baseline (theoretical ceiling reference)
    # ---------------------------------------------------------------
    W_fp16 = torch.randn(d_out, d_in, dtype=torch.float16, device=device)

    def run_fp16():
        return torch.matmul(W_fp16, X.t())   # (d_out, T)

    t_fp16 = time_fn_us(run_fp16)

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print()
    print(f"{'kernel':25s} {'time (us)':>10s}    {'vs fp16':>8s}   {'vs opt':>8s}")
    print("-" * 70)
    print(f"{'[naive] quant':25s} {t_quant:>10.2f}")
    print(f"{'[naive] dense GEMM':25s} {t_dense:>10.2f}")
    print(f"{'[naive] sparse GEMM':25s} {t_sparse:>10.2f}")
    print(f"{'[naive] reduce add':25s} {t_add:>10.2f}")
    print(f"{'[naive] TOTAL':25s} {t_naive_total:>10.2f}    "
          f"{t_naive_total/t_fp16:>7.2f}x  {t_naive_total/t_opt_total:>7.2f}x")
    print("-" * 70)
    print(f"{'[opt ] quant':25s} {t_opt_quant:>10.2f}")
    print(f"{'[opt ] fused (dense+sp)':25s} {t_opt_fused:>10.2f}")
    print(f"{'[opt ] TOTAL':25s} {t_opt_total:>10.2f}    "
          f"{t_opt_total/t_fp16:>7.2f}x  {'1.00x':>7s}")
    print("-" * 70)
    print(f"{'[fp16] cuBLAS baseline':25s} {t_fp16:>10.2f}    {'1.00x':>7s}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
