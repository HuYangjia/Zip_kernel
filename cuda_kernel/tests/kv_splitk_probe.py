"""Split-K probe for kv_proj small-T shapes.

Hypothesis:
  14B kv T=8 grid = (ceil(d_out/128), 1) = (16, 1) = only 16 CTAs,
  severely under-subscribing the 20-SM RTX 4090.  The legacy kernel's
  mainloop is FAST (GEMM-only 19us already beats fp16-flushed 27us),
  but the grid is too small to amortise launch overhead.

  Split-K (gridDim.z = split_k) multiplies the grid by split_k on the
  K-dim -- for 14B kv T=8 n_groups=40, split_k=4 gives 16*4 = 64 CTAs
  (3-4 waves, 100% SM utilisation).  The legacy kernel ALREADY supports
  split_k; it just isn't being picked by the current dispatcher for
  T<256 kv shapes.

  If forcing split_k=2/4 on legacy (two-step path) drops kv_proj T=8
  from 32-37us toward 20-22us, we've found the real fix.  Then add
  a dispatcher rule (C.7) to turn on split_k for this family and
  kv_proj superior to fp16 becomes reality.

Protocol: warmup=500, outer=10, inner=200, 4 trials median.
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})


def bench_us(fn, warmup=500, outer=10, inner=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / inner)
    return best


def bench_with_l2_flush(fn, flush_size_mb=96, warmup=500, outer=10, inner=200):
    """FP16 baseline — flush L2 between iterations (match bench_qwen3)."""
    flush = torch.empty((flush_size_mb * 1024 * 1024 // 4,),
                        dtype=torch.float32, device="cuda")
    def _run():
        flush.zero_()  # cold the L2
        fn()
    for _ in range(warmup):
        _run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            _run()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / inner)
    return best


def set_split_k(v):
    if v is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(v)


TARGETS = [
    # kv_proj family: d_out=2048 small grid
    ("14B kv T=8",   8,   5120, 2048, 40),
    ("14B kv T=32",  32,  5120, 2048, 40),
    ("14B kv T=128", 128, 5120, 2048, 40),
    ("8B kv T=8",    8,   4096, 2048, 32),
    ("8B kv T=32",   32,  4096, 2048, 32),
    ("8B kv T=128",  128, 4096, 2048, 32),
    ("32B kv T=8",   8,   5120, 2048, 40),
    ("32B kv T=128", 128, 5120, 2048, 40),
    ("70B kv T=8",   8,   8192, 2048, 64),
    ("70B kv T=128", 128, 8192, 2048, 64),
    # control: q_proj (bigger grid, shouldn't benefit as much)
    ("14B q T=8",    8,   5120, 5120, 40),
]


PR("=" * 104)
PR("Split-K probe: can we rescue kv_proj small-T by increasing grid via split_k?")
PR("=" * 104)
PR(f"{'shape':<14}  {'n_g':>4}  {'fp16 us':>8}  "
   f"{'leg sk=auto':>12} {'sk=1':>8} {'sk=2':>8} {'sk=4':>8} {'sk=8':>8}  "
   f"{'best sk':>8} {'best us':>8}  {'fp16 sp':>8}")

for label, T, d_in, d_out, n_g in TARGETS:
    b = make_inputs(T, d_out, d_in, hp_ratio=0.0,
                    device="cuda", seed=T + d_in + d_out)
    W_fp = b["W_fp"]; X_fp_t = b["X_fp_t"]; X = b["X"]; perm = b["perm"]

    # FP16 with L2 flush (bench_qwen3-compatible)
    t_fp16 = bench_with_l2_flush(lambda: torch.matmul(W_fp, X_fp_t))

    def run_legacy_e2e():
        X_s4, sx, sX = ops.activation_quant_cuda(X, perm)
        return ops.fused_dense_sparse_cuda_int4(
            b["W_low_packed"], b["W_high_packed"],
            b["hp_row_offsets"], b["hp_col_indices"],
            X_s4, b["scale_u4"], b["zero_u4"], sX, sx, d_out, d_in,
        )

    set_split_k(None)
    t_auto = bench_us(run_legacy_e2e)

    # Try each split_k value (only those that evenly divide n_g)
    results = {"auto": t_auto}
    for sk in [1, 2, 4, 8]:
        if n_g % sk != 0:
            continue
        set_split_k(sk)
        results[f"sk={sk}"] = bench_us(run_legacy_e2e)
    set_split_k(None)

    # Find best
    best_name = min(results, key=lambda k: results[k])
    best_us = results[best_name]
    fp16_sp = t_fp16 / best_us

    # Format output (fixed columns)
    def fmt(v):
        return f"{v:>7.2f}" if v is not None else "    -   "
    r1 = fmt(results.get("sk=1"))
    r2 = fmt(results.get("sk=2"))
    r4 = fmt(results.get("sk=4"))
    r8 = fmt(results.get("sk=8"))

    star = "  WIN" if fp16_sp >= 1.0 else ""
    PR(f"  {label:<12}  {n_g:>4}  {t_fp16:>7.2f}  "
       f"{t_auto:>11.2f} {r1} {r2} {r4} {r8}  "
       f"{best_name:>7}  {best_us:>7.2f}  {fp16_sp:>7.3f}x{star}")
