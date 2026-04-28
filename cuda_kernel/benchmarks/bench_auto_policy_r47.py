"""R47 bench: add a real 'auto' policy column to the E2E table.

bench_cuda_vs_triton.py forces 'triton' / 'cuda' for the two
columns; we want to verify that the updated _auto_policy actually
routes a real forward pass into the CUDA fast path (not to Triton).

Outputs a small markdown under logs/cuda_kernel/auto_policy_r47_*.md
and prints to stdout.

Usage (on autodl, from /root with PYTHONPATH=/root):
    python kernel/cuda_kernel/benchmarks/bench_auto_policy_r47.py
"""
from __future__ import annotations

import logging
import pathlib
import sys
import time
from dataclasses import dataclass

import torch

# Reuse the benchmarking helpers / shape list / input builders from
# the canonical script so we stay aligned with the R46 methodology
# (10 warmup x 10 outer x 50 inner, min-of-means).
_HERE = pathlib.Path(__file__).resolve()
_KROOT = _HERE.parents[3]  # kernel's parent dir (contains 'kernel/')
if str(_KROOT) not in sys.path:
    sys.path.insert(0, str(_KROOT))

from kernel.cuda_kernel.benchmarks.bench_cuda_vs_triton import (  # noqa: E402
    SHAPES,
    _bench_fn,
    _make_sparse_inputs,
    BROW,
    BCOL,
)
from kernel.backend import v9_linear_forward  # noqa: E402
from kernel.backend.policy import set_backend_policy  # noqa: E402
from kernel.triton_kernel.pack_utils import V9WeightContainer  # noqa: E402


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("bench_auto_policy_r47")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    return logger


def main():
    log = _setup_logging()
    log.info("R47 auto-policy E2E bench starting")
    rows = []
    for shape in SHAPES:
        data = _make_sparse_inputs(shape.T, shape.d_out, shape.d_in, shape.hp_ratio)
        W = V9WeightContainer(
            W_low_packed=data["W_low_packed"],
            W_high_blocks_packed=data["W_high_blocks_packed"],
            scale_u4=data["scale_u4"],
            zero_u4=data["zero_u4"],
            hp_row_offsets=data["hp_row_offsets"],
            hp_col_indices=data["hp_col_indices"],
            perm=data["perm"],
            block_shape=(BROW, BCOL),
            d_out=shape.d_out,
            d_in=shape.d_in,
        )
        X = data["X"]

        try:
            set_backend_policy("triton")
            t_tri = _bench_fn(lambda: v9_linear_forward(X, W))
            set_backend_policy("cuda")
            t_cud = _bench_fn(lambda: v9_linear_forward(X, W))
            set_backend_policy("auto")
            t_auto = _bench_fn(lambda: v9_linear_forward(X, W))
        finally:
            set_backend_policy("auto")
        rows.append({
            "shape": shape.name, "T": shape.T, "d_in": shape.d_in, "d_out": shape.d_out,
            "triton_us": t_tri, "cuda_us": t_cud, "auto_us": t_auto,
            "auto_vs_tri": t_tri / t_auto,
            "auto_vs_cuda": t_cud / t_auto,
        })
        log.info("%-20s T=%5d triton=%7.2fus cuda=%7.2fus auto=%7.2fus  auto/cuda=%.3fx",
                 shape.name, shape.T, t_tri, t_cud, t_auto, t_cud / t_auto)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = _KROOT / "logs" / "cuda_kernel"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"bench_auto_policy_r47_{ts}.md"
    with md.open("w") as f:
        f.write("# R47 auto-policy E2E benchmark\n\n")
        f.write("Host: `autodl` / RTX 4090 (SM89)\n\n")
        f.write("Stats: min-of-means, 10 outer x 50 inner, 10 warmup.\n\n")
        f.write("Columns: triton = force triton, cuda = force cuda, auto = updated _auto_policy.\n")
        f.write("auto/cuda close to 1.0 means the policy correctly routes to CUDA.\n\n")
        f.write("| shape | T | d_in | d_out | triton (us) | cuda (us) | auto (us) | auto_vs_tri | auto/cuda |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['shape']} | {r['T']} | {r['d_in']} | {r['d_out']} | "
                    f"{r['triton_us']:.2f} | {r['cuda_us']:.2f} | {r['auto_us']:.2f} | "
                    f"{r['auto_vs_tri']:.2f}x | {r['auto_vs_cuda']:.3f}x |\n")
    log.info("wrote %s", md)


if __name__ == "__main__":
    main()
