"""Q.0-lite — upper-bound probe for 3-stage cp.async.

Idea:
  The current mainloop is 2-stage cp.async with wait<0> (wait-all)
  after each iteration.  A 3-stage (or deeper) pipeline lets the
  next-next group's HBM fetch fly concurrently with the current
  MMA, which should SAVE the wait time -- but only IF the wait
  is actually on the critical path.

  We can cheaply estimate the saving by compiling a DELIBERATELY
  INCORRECT kernel where the wait_group<0> is compile-time replaced
  with a no-op (via -DHKUST_V9_PROBE_SKIP_WAIT).  The kernel will
  produce wrong numbers, but the TIMING is the physical lower bound
  that any deeper pipeline could possibly achieve.

  If (wait=on us) - (wait=off us) < 3%, 3-stage pipeline cannot save
  more than 3%.  Abandon Q-b in favour of Q-a or Q-collapse.

This script must be invoked twice with different build cache dirs:
  Run A: HKUST_V9_PROBE_SKIP_WAIT=0 HKUST_V9_CUDA_BUILD_DIR=/tmp/buildA
  Run B: HKUST_V9_PROBE_SKIP_WAIT=1 HKUST_V9_CUDA_BUILD_DIR=/tmp/buildB

Timing protocol: warmup=300 outer=5 inner=150, single-trial-per-shape
(the signal should be big enough at T=512 that median isn't needed).
"""
import os
import statistics
import sys

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})


def run(b, d_out, d_in):
    return ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        b["X_s4"], b["scale_u4"], b["zero_u4"],
        b["sum_X"], b["scale_x"], d_out, d_in,
    )


def bench_us(fn, warmup=300, outer=5, inner=150):
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


# Focus on the T=512 shapes that would most benefit from a deeper
# pipeline: mid-to-big d_in and any n_groups >= 16 (where cp.async
# is actually used in the kernel).  Also include one sanity
# check shape where wait should not matter (tiny, non-cp.async).
TARGETS = [
    # (label, T, d_in, d_out)
    ("14B gu T=512 (big)",  512, 5120, 34816),
    ("32B dn T=512 (big)",  512, 27648, 5120),
    ("70B gu T=512 (huge)", 512, 8192, 57344),
    ("8B gu T=512 (mid)",   512, 4096, 24576),
    ("14B q T=512 (small)", 512, 5120, 5120),
    ("14B gu T=128",        128, 5120, 34816),
    ("0.6B gu T=512 (ng=8)", 512, 1024, 6144),  # n_g=8 < 16, cp.async OFF
]


probe_mode = os.environ.get("HKUST_V9_PROBE_SKIP_WAIT", "0")
PR("=" * 80)
PR(f"Q.0-lite upper-bound probe — HKUST_V9_PROBE_SKIP_WAIT = {probe_mode}")
PR(f"                             (0=wait on, 1=wait DISABLED, kernel INCORRECT)")
PR("=" * 80)
PR(f"{'shape':<28}  {'us':>10}")

results = []
for label, T, d_in, d_out in TARGETS:
    b = make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda",
                    seed=T + d_in + d_out)
    def _run():
        run(b, d_out, d_in)
    us = bench_us(_run)
    PR(f"  {label:<26}  {us:>9.2f}")
    results.append({"label": label, "T": T, "d_in": d_in, "d_out": d_out,
                    "us": us})

# Dump JSON so the compare script can diff the two runs.
import json
out_path = f"/tmp/q0_lite_wait{probe_mode}.json"
with open(out_path, "w") as f:
    json.dump({"probe_mode": probe_mode, "results": results}, f, indent=2)
PR(f"\nDumped results to {out_path}")
