"""R46 probe: is fused_dense_sparse faster than dense+sparse split for decode?

Current state: dispatcher._forward_decode calls dense_gemm_cuda +
sparse_gemm_cuda separately when hp>0.  In contrast _forward_prefill
calls fused_dense_sparse.  R46 tests whether decode should follow
the prefill pattern (single fused kernel = one kernel launch, one
HBM read of X_s4, less scheduling overhead).

For every (T, d_out, d_in, hp_ratio) in the production shape zoo:
  - time dense_fn + sparse_fn (current decode path)
  - time fused_fn (proposed decode path)
  - report delta

Methodology [[memory:bmmiahpl]]: 50 warmup + 3x100-iter windows +
min-of-means.

Usage:
    python kernel/cuda_kernel/benchmarks/bench_r46_fused_vs_split.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
_IMPORT_ROOT = _THIS.parents[3]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.benchmarks._bench_util import time_ms
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_s4_le
from kernel.cuda_kernel import ops as cuda_ops


def _setup_logging(log_file: Path) -> logging.Logger:
    log = logging.getLogger("bench_r46_fused_vs_split")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def _make_inputs(T, d_out, d_in, hp_ratio, seed=0xBEEF):
    torch.manual_seed(seed)
    device = "cuda"
    X = torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device=device)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)

    n_groups = d_in // BCOL
    W_low_s4 = torch.randint(-8, 8, (d_out, d_in),
                             dtype=torch.int8, device=device)
    W_low_packed = pack_s4_le(W_low_s4)
    scale_u4 = (torch.rand(d_out, n_groups, device=device) * 0.05
                + 0.001).to(torch.float16)
    zero_u4 = (torch.randn(d_out, n_groups, device=device) * 0.2
               ).to(torch.float16)

    nrow = d_out // BROW
    ncol = d_in // BCOL
    total_blocks = nrow * ncol

    if hp_ratio <= 0.0:
        hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32,
                                     device=device)
        hp_col_indices = torch.zeros(0, dtype=torch.int32, device=device)
        W_high_blocks = torch.zeros((0, BROW, BCOL // 2),
                                    dtype=torch.int8, device=device)
    else:
        nnz = max(1, int(round(total_blocks * hp_ratio)))
        row_ids = torch.randint(0, nrow, (nnz,), device=device)
        col_ids = torch.randint(0, ncol, (nnz,), device=device)
        order = torch.argsort(row_ids * (ncol + 1) + col_ids)
        row_ids, col_ids = row_ids[order], col_ids[order]
        hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32,
                                     device=device)
        counts = torch.bincount(row_ids, minlength=nrow)
        hp_row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
        hp_col_indices = col_ids.to(torch.int32)
        W_high_blocks = torch.randint(-128, 127, (nnz, BROW, BCOL // 2),
                                      dtype=torch.int8, device=device)

    return (W_low_packed, W_high_blocks, hp_row_offsets, hp_col_indices,
            X_s4, scale_u4, zero_u4, sum_X, scale_x)


# (T, d_out, d_in, hp_ratio) covering the production zoo + decode band.
PROBES = [
    # decode band (T <= 16), matches bench_cuda_vs_triton shapes:
    (1,    4096, 4096,  0.05),
    (1,    4096, 11008, 0.05),
    (1,   11008, 4096,  0.05),
    (8,    4096, 4096,  0.05),
    (16,   4096, 4096,  0.05),
    # batch band (16 < T <= 128) = decode path in current dispatcher:
    (32,   4096, 4096,  0.05),
    (64,   4096, 4096,  0.05),
    (128,  4096, 4096,  0.05),
    # kv_proj shape (narrow d_out):
    (16,   1024, 5120,  0.05),
    (32,   1024, 5120,  0.05),
    (64,   1024, 5120,  0.05),
    # down_proj shape (narrow d_out big d_in):
    (16,   4096, 11008, 0.05),
    (64,   4096, 11008, 0.05),
]


def main():
    torch.cuda.init()
    torch.backends.cudnn.benchmark = False
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_root = _THIS.parents[1] / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"bench_r46_fused_vs_split_{ts}.log"
    log = _setup_logging(log_file)

    dev = torch.cuda.get_device_name(0)
    log.info(f"Device: {dev}")

    rows = []
    for T, d_out, d_in, hp in PROBES:
        inputs = _make_inputs(T, d_out, d_in, hp)
        (W_low, W_hi, rowoff, colind, X_s4, su4, zu4, sX, sx) = inputs

        def dense_fn():
            return cuda_ops.dense_gemm_cuda_int4(
                W_low, X_s4, su4, zu4, sX, sx
            )

        def sparse_fn():
            return cuda_ops.sparse_gemm_cuda_int4(
                W_hi, rowoff, colind, X_s4, su4, sx, d_out, d_in
            )

        def split_fn():
            Y1 = dense_fn()
            Y2 = sparse_fn()
            return Y1 + Y2

        def fused_fn():
            return cuda_ops.fused_dense_sparse_cuda_int4(
                W_low, W_hi, rowoff, colind,
                X_s4, su4, zu4, sX, sx, d_out, d_in
            )

        us_dense = time_ms(dense_fn, n_warmup=50, n_iter=100, n_repeat=3) * 1000
        us_sparse = time_ms(sparse_fn, n_warmup=50, n_iter=100, n_repeat=3) * 1000
        us_split = time_ms(split_fn, n_warmup=50, n_iter=100, n_repeat=3) * 1000
        us_fused = time_ms(fused_fn, n_warmup=50, n_iter=100, n_repeat=3) * 1000

        win = us_split - us_fused
        win_pct = 100.0 * win / us_split if us_split > 0 else 0
        tag = "✓" if us_fused < us_split * 0.97 else ("×" if us_fused > us_split * 1.03 else "·")
        log.info(
            f"T={T:4d} d_out={d_out:5d} d_in={d_in:5d} hp={hp}  "
            f"dense={us_dense:6.2f}us sparse={us_sparse:6.2f}us "
            f"split={us_split:6.2f}us fused={us_fused:6.2f}us  "
            f"save={win:+6.2f}us ({win_pct:+.1f}%) {tag}"
        )
        rows.append({
            "T": T, "d_out": d_out, "d_in": d_in, "hp_ratio": hp,
            "us_dense": us_dense, "us_sparse": us_sparse,
            "us_split": us_split, "us_fused": us_fused,
            "save_us": win, "save_pct": win_pct,
        })

    out_json = log_root / f"bench_r46_fused_vs_split_{ts}.json"
    with out_json.open("w") as f:
        json.dump({"device": dev, "rows": rows}, f, indent=2)
    log.info(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
