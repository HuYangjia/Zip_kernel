"""C.8 A/B Verification: 5 loser shapes end-to-end timing.

Usage (on autodl):
    cd /root
    python Zip_kernel/kernel/cuda_kernel/tests/c8_ab_verify.py

Measures FP16 (cuBLAS) vs CUDA (W4A4) end-to-end for 5 target shapes.
Uses the project-standard timing methodology:
  warmup=500, outer=20, inner=200, 5 trials median (sensitive A/B config).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from dataclasses import dataclass

import torch

# Path anchoring
_THIS = Path(__file__).resolve()
_IMPORT_ROOT = _THIS.parents[3]  # /root/Zip_kernel
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from kernel.cuda_kernel import ops as cuda_ops
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_s4_le

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP = 500
OUTER = 20
INNER = 200
N_TRIALS = 5
HP_RATIO = 0.05
FLUSH_L2_FP16 = True


@dataclass
class TargetShape:
    name: str
    d_in: int
    d_out: int
    T: int


LOSER_SHAPES = [
    TargetShape("Qwen2.5-32B_gu",  d_in=5120,  d_out=55296, T=2048),
    TargetShape("LLaMA3-70B_gu",   d_in=8192,  d_out=57344, T=2048),
    TargetShape("LLaMA3-70B_kv",   d_in=8192,  d_out=2048,  T=1024),
    TargetShape("Qwen3-1.7B_dn",   d_in=6144,  d_out=2048,  T=1024),
    TargetShape("Qwen3-4B_dn",     d_in=9728,  d_out=2560,  T=1024),
]


# ---------------------------------------------------------------------------
# Timer (matches project standard: min-of-outer of mean-of-inner)
# ---------------------------------------------------------------------------
def time_us(fn, warmup=WARMUP, outer=OUTER, inner=INNER) -> float:
    """min-over-outer of (mean-over-inner of per-iter us)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(outer):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mean_us = (t1 - t0) / inner * 1e6
        best = min(best, mean_us)
    return best


def flush_l2():
    """Flush L2 cache by writing a large buffer."""
    buf = torch.empty(48 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    buf.fill_(0.0)
    del buf


def time_us_flush(fn, warmup=WARMUP, outer=OUTER, inner=INNER) -> float:
    """Like time_us but flushes L2 before each inner iteration (for FP16)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(outer):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(inner):
            flush_l2()
            fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mean_us = (t1 - t0) / inner * 1e6
        best = min(best, mean_us)
    return best


# ---------------------------------------------------------------------------
# Input factory
# ---------------------------------------------------------------------------
def make_inputs(T: int, d_out: int, d_in: int):
    torch.manual_seed(T + d_out + d_in)
    device = "cuda"

    X = torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
    X_fp_t = X.transpose(0, 1).contiguous()
    perm = torch.arange(d_in, dtype=torch.int32, device=device)

    n_groups = d_in // BCOL
    W_s4 = torch.randint(-8, 8, (d_out, d_in), dtype=torch.int8, device=device)
    W_low_packed = pack_s4_le(W_s4)
    scale_u4 = (torch.rand(d_out, n_groups, device=device) * 0.05 + 0.001).half()
    zero_u4 = (torch.randn(d_out, n_groups, device=device) * 0.2).half()
    W_fp = torch.randn(d_out, d_in, dtype=torch.float16, device=device) * 0.02

    # Sparse side
    nrow = d_out // BROW
    ncol = d_in // BCOL
    total_blocks = nrow * ncol
    n_hp = max(1, int(total_blocks * HP_RATIO))

    torch.manual_seed((T * d_in * d_out) ^ 0xA5A5)
    flat = torch.randperm(total_blocks, device=device)[:n_hp]
    br = (flat // ncol).to(torch.int32)
    bc = (flat % ncol).to(torch.int32)
    order = torch.argsort(br.to(torch.int64) * 1_000_000 + bc.to(torch.int64))
    br_sorted = br[order]
    bc_sorted = bc[order]

    W_high_s4 = torch.randint(-8, 8, (n_hp, BROW, BCOL), dtype=torch.int8, device=device)
    W_high_packed = pack_s4_le(W_high_s4)

    hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
    counts = torch.bincount(br_sorted.to(torch.int64), minlength=nrow)
    hp_row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    return dict(
        X=X, X_fp_t=X_fp_t, perm=perm,
        W_low_packed=W_low_packed,
        W_high_packed=W_high_packed,
        hp_row_offsets=hp_row_offsets,
        hp_col_indices=bc_sorted,
        scale_u4=scale_u4, zero_u4=zero_u4,
        W_fp=W_fp,
    )


# ---------------------------------------------------------------------------
# Benchmark one shape
# ---------------------------------------------------------------------------
def bench_shape(shape: TargetShape) -> dict:
    T, d_in, d_out = shape.T, shape.d_in, shape.d_out
    inp = make_inputs(T, d_out, d_in)

    def run_fp16():
        return torch.matmul(inp["W_fp"], inp["X_fp_t"])

    def run_cuda():
        X_s4, sx, sX = cuda_ops.activation_quant_cuda(inp["X"], inp["perm"])
        return cuda_ops.fused_dense_sparse_cuda(
            inp["W_low_packed"], inp["W_high_packed"],
            inp["hp_row_offsets"], inp["hp_col_indices"],
            X_s4, inp["scale_u4"], inp["zero_u4"],
            sX, sx,
            d_out, d_in,
        )

    # Median-of-N-trials
    fp16_trials = []
    cuda_trials = []
    for trial in range(N_TRIALS):
        if FLUSH_L2_FP16:
            t_fp16 = time_us_flush(run_fp16)
        else:
            t_fp16 = time_us(run_fp16)
        t_cuda = time_us(run_cuda)
        fp16_trials.append(t_fp16)
        cuda_trials.append(t_cuda)

    fp16_trials.sort()
    cuda_trials.sort()
    t_fp16_med = fp16_trials[N_TRIALS // 2]
    t_cuda_med = cuda_trials[N_TRIALS // 2]
    speedup = t_fp16_med / t_cuda_med

    return {
        "name": shape.name,
        "T": T,
        "d_in": d_in,
        "d_out": d_out,
        "fp16_us": t_fp16_med,
        "cuda_us": t_cuda_med,
        "speedup": speedup,
        "fp16_all": fp16_trials,
        "cuda_all": cuda_trials,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("C.8 A/B Verification — 5 Loser Shapes")
    print(f"Config: warmup={WARMUP}, outer={OUTER}, inner={INNER}, trials={N_TRIALS}")
    print(f"FP16 L2 flush: {FLUSH_L2_FP16}")
    print("=" * 72)

    # Warm up JIT compilation
    print("\n[JIT] Compiling CUDA extension (first call)...")
    tiny_inp = make_inputs(8, 128, 128)
    X_s4, sx, sX = cuda_ops.activation_quant_cuda(tiny_inp["X"], tiny_inp["perm"])
    cuda_ops.fused_dense_sparse_cuda(
        tiny_inp["W_low_packed"], tiny_inp["W_high_packed"],
        tiny_inp["hp_row_offsets"], tiny_inp["hp_col_indices"],
        X_s4, tiny_inp["scale_u4"], tiny_inp["zero_u4"],
        sX, sx, 128, 128,
    )
    print("[JIT] Done.\n")

    results = []
    for i, shape in enumerate(LOSER_SHAPES):
        print(f"[{i+1}/{len(LOSER_SHAPES)}] {shape.name}  "
              f"T={shape.T} d_in={shape.d_in} d_out={shape.d_out} ...")
        r = bench_shape(shape)
        results.append(r)
        sp_str = f"{r['speedup']:.4f}×"
        win = "WIN" if r['speedup'] >= 1.0 else "LOSE"
        print(f"        FP16={r['fp16_us']:.1f}us  CUDA={r['cuda_us']:.1f}us  "
              f"speedup={sp_str}  [{win}]")
        print(f"        FP16 trials: {[f'{x:.1f}' for x in r['fp16_all']]}")
        print(f"        CUDA trials: {[f'{x:.1f}' for x in r['cuda_all']]}")
        print()

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("-" * 72)
    print(f"{'Shape':<22} {'T':>5} {'d_in':>6} {'d_out':>6} "
          f"{'FP16(us)':>9} {'CUDA(us)':>9} {'Speedup':>8} {'Result':>6}")
    print("-" * 72)
    wins = 0
    for r in results:
        win = "WIN" if r['speedup'] >= 1.0 else "LOSE"
        if r['speedup'] >= 1.0:
            wins += 1
        print(f"{r['name']:<22} {r['T']:>5} {r['d_in']:>6} {r['d_out']:>6} "
              f"{r['fp16_us']:>9.1f} {r['cuda_us']:>9.1f} "
              f"{r['speedup']:>7.4f}× {win:>6}")
    print("-" * 72)
    speedups = [r['speedup'] for r in results]
    print(f"Median speedup: {sorted(speedups)[len(speedups)//2]:.4f}×")
    print(f"Mean speedup:   {sum(speedups)/len(speedups):.4f}×")
    print(f"Wins: {wins}/{len(results)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
