"""C.8 Quick A/B Verification — same 5 loser shapes, reduced timing budget.

Differences vs c8_ab_verify.py:
  * warmup=200, outer=10, inner=100, trials=3  (vs 500/20/200/5)
  * Incremental JSON dump after every shape (observable progress)
  * Line-buffered stdout for real-time log tailing

Budget motivation: c8_ab_verify.py's (500/20/200/5) config is designed for
A/B bisection of <5% diff cases; C.8 expected diffs are 10-30% so we can
safely drop one notch per [[memory:bmmiahpl]] kernel timing spec.

Total estimated runtime: 5-8 minutes for all 5 loser shapes.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import torch

# Path anchoring — same as c8_ab_verify.py
_THIS = Path(__file__).resolve()
_IMPORT_ROOT = _THIS.parents[3]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from kernel.cuda_kernel import ops as cuda_ops
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_s4_le

# ---------------------------------------------------------------------------
# Config: reduced budget for quick sanity check
# ---------------------------------------------------------------------------
WARMUP = 200
OUTER = 10
INNER = 100
N_TRIALS = 3
HP_RATIO = 0.05
FLUSH_L2_FP16 = True

OUT_DIR = _IMPORT_ROOT / "kernel" / "cuda_kernel" / "logs" / "r70_c8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "quick_ab_verify.json"

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
# Timer (same methodology as c8_ab_verify.py, reduced counts)
# ---------------------------------------------------------------------------
def time_us(fn, warmup=WARMUP, outer=OUTER, inner=INNER) -> float:
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
        best = min(best, (t1 - t0) / inner * 1e6)
    return best

def flush_l2():
    buf = torch.empty(48 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    buf.fill_(0.0)
    del buf

def time_us_flush(fn, warmup=WARMUP, outer=OUTER, inner=INNER) -> float:
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
        best = min(best, (t1 - t0) / inner * 1e6)
    return best

# ---------------------------------------------------------------------------
# Input factory — verbatim copy from c8_ab_verify.py
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
# Bench one shape
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

    fp16_trials, cuda_trials = [], []
    for _ in range(N_TRIALS):
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
        "name": shape.name, "T": T, "d_in": d_in, "d_out": d_out,
        "fp16_us": t_fp16_med, "cuda_us": t_cuda_med, "speedup": speedup,
        "fp16_all": fp16_trials, "cuda_all": cuda_trials,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72, flush=True)
    print("C.8 Quick A/B — 5 Loser Shapes  (reduced budget)", flush=True)
    print(f"Config: warmup={WARMUP}, outer={OUTER}, inner={INNER}, "
          f"trials={N_TRIALS}", flush=True)
    print(f"FP16 L2 flush: {FLUSH_L2_FP16}", flush=True)
    print(f"Output: {OUT_JSON}", flush=True)
    print("=" * 72, flush=True)

    print("\n[JIT] Warming up CUDA extension ...", flush=True)
    tiny = make_inputs(8, 128, 128)
    X_s4, sx, sX = cuda_ops.activation_quant_cuda(tiny["X"], tiny["perm"])
    cuda_ops.fused_dense_sparse_cuda(
        tiny["W_low_packed"], tiny["W_high_packed"],
        tiny["hp_row_offsets"], tiny["hp_col_indices"],
        X_s4, tiny["scale_u4"], tiny["zero_u4"],
        sX, sx, 128, 128,
    )
    torch.cuda.synchronize()
    print("[JIT] Done.\n", flush=True)

    results = []
    t_start = time.time()
    for i, shape in enumerate(LOSER_SHAPES):
        print(f"[{i+1}/{len(LOSER_SHAPES)}] {shape.name}  "
              f"T={shape.T} d_in={shape.d_in} d_out={shape.d_out} ...",
              flush=True)
        ts = time.time()
        r = bench_shape(shape)
        elapsed = time.time() - ts
        results.append(r)

        sp = r['speedup']
        win = "WIN " if sp >= 1.0 else "LOSE"
        print(f"    {win}  FP16={r['fp16_us']:.1f}us  CUDA={r['cuda_us']:.1f}us  "
              f"speedup={sp:.4f}x  ({elapsed:.1f}s)", flush=True)
        print(f"    fp16_trials={[f'{x:.1f}' for x in r['fp16_all']]}  "
              f"cuda_trials={[f'{x:.1f}' for x in r['cuda_all']]}",
              flush=True)

        # Flush partial results after every shape
        with open(OUT_JSON, "w") as f:
            json.dump({
                "config": {"warmup": WARMUP, "outer": OUTER, "inner": INNER,
                           "trials": N_TRIALS, "flush_l2_fp16": FLUSH_L2_FP16},
                "completed": i + 1,
                "total": len(LOSER_SHAPES),
                "elapsed_s": time.time() - t_start,
                "results": results,
            }, f, indent=2)
        print(f"    [saved partial -> {OUT_JSON.name}]\n", flush=True)

    # Summary
    print("=" * 72, flush=True)
    print("SUMMARY", flush=True)
    print("-" * 72, flush=True)
    print(f"{'Shape':<22} {'T':>5} {'d_in':>6} {'d_out':>6} "
          f"{'FP16us':>8} {'CUDAus':>8} {'Speedup':>8} {'Res':>5}", flush=True)
    print("-" * 72, flush=True)
    wins = 0
    for r in results:
        res = "WIN" if r['speedup'] >= 1.0 else "LOSE"
        wins += int(r['speedup'] >= 1.0)
        print(f"{r['name']:<22} {r['T']:>5} {r['d_in']:>6} {r['d_out']:>6} "
              f"{r['fp16_us']:>8.1f} {r['cuda_us']:>8.1f} "
              f"{r['speedup']:>7.4f}x {res:>5}", flush=True)
    print("-" * 72, flush=True)
    sps = sorted([r['speedup'] for r in results])
    print(f"Median speedup: {sps[len(sps)//2]:.4f}x", flush=True)
    print(f"Mean speedup:   {sum(sps)/len(sps):.4f}x", flush=True)
    print(f"Wins: {wins}/{len(results)}", flush=True)
    print(f"Total time: {time.time() - t_start:.1f}s", flush=True)
    print("=" * 72, flush=True)

if __name__ == "__main__":
    main()
