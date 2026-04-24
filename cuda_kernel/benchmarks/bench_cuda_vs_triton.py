"""CUDA-vs-Triton latency benchmark for the V9 kernel suite.

Usage (on the SM89 host):

    python kernel/cuda_kernel/benchmarks/bench_cuda_vs_triton.py

Outputs:
  - JSON:     logs/cuda_kernel/bench_{TS}.json
  - Markdown: logs/cuda_kernel/bench_{TS}.md

Design rules (per project preferences):
  - Python ``logging`` (INFO to terminal, DEBUG to file) -- not print.
  - Paths anchored to __file__ so the script is idempotent regardless
    of cwd.
  - Statistics via min-of-means (10 outer repeats of 50 inner iters)
    with explicit CUDA stream sync each sample.
  - Warmup separately for CUDA and Triton paths (Triton JITs on the
    first call and its autotuner also caches on the first shape).

Kernels measured:
  - activation_quant      (decode + prefill)
  - dense_gemm            (decode + prefill)
  - sparse_gemm           (decode + prefill)
  - fused_dense_sparse    (the full V9 linear forward)

Backends compared per kernel:
  CUDA: forced via backend policy
  Triton: forced via backend policy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import torch

# -- Path anchoring ---------------------------------------------------------
_THIS = Path(__file__).resolve()
# layout: <root>/kernel/cuda_kernel/benchmarks/<this>.py
# parents[0] = benchmarks, [1] = cuda_kernel, [2] = kernel, [3] = <root>
# On autodl, <root> == /root (and /root/kernel is a symlink to
#                              /root/Zip_kernel/kernel... actually the
# symlink is /root/kernel -> /root/Zip_kernel, so after resolve() the
# path becomes /root/Zip_kernel/cuda_kernel/... and parents[3] == /root).
# Locally, <root> == /Users/.../HKUST.  In both cases importing
# ``kernel.xxx`` works once <root> is on sys.path.
_IMPORT_ROOT = _THIS.parents[3]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4
from kernel.triton_kernel.sparse_s4s4_gemm import sparse_gemm_s4_s4
from kernel.triton_kernel.fused_dense_sparse_gemm import fused_dense_sparse_gemm
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_s4_le, V9WeightContainer
from kernel.cuda_kernel import ops as cuda_ops
from kernel.backend import (
    set_backend_policy,
    v9_linear_forward,
    get_backend_status,
)


# -- Logging setup ----------------------------------------------------------
def _setup_logging(log_file: Path) -> logging.Logger:
    log = logging.getLogger("bench_cuda_vs_triton")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    term = logging.StreamHandler(sys.stdout)
    term.setLevel(logging.INFO)
    term.setFormatter(fmt)
    log.addHandler(term)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fileh = logging.FileHandler(log_file, mode="a")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(fmt)
    log.addHandler(fileh)

    return log


# -- Benchmark primitive ----------------------------------------------------
def _bench_fn(
    fn: Callable[[], None],
    *,
    warmup: int = 10,
    outer: int = 10,
    inner: int = 50,
    device: torch.device | None = None,
) -> float:
    """min-of-means in microseconds.

    - ``warmup`` calls to eat JIT / autotuner / allocator cold paths.
    - ``outer`` batches of ``inner`` calls each; each batch uses a
      single ``cudaEventRecord`` pair to amortise API overhead.
    - Returns the minimum batch mean, which is the robust estimator
      against stray OS/CUDA scheduler noise.
    """
    if device is None:
        device = torch.device("cuda")
    torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    means_us = []
    for _ in range(outer):
        start_ev.record()
        for _ in range(inner):
            fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        # elapsed_time returns ms
        batch_us = start_ev.elapsed_time(end_ev) * 1000.0 / inner
        means_us.append(batch_us)
    return min(means_us)


# -- Inputs -----------------------------------------------------------------
@dataclass
class Shape:
    name: str
    T: int
    d_in: int
    d_out: int
    hp_ratio: float = 0.05   # only used for sparse/fused/end2end


# Representative sizes for 4090-class decode / prefill workloads.
SHAPES: list[Shape] = [
    # decode
    Shape("dec_T1_4k_4k",    T=1,    d_in=4096,  d_out=4096),
    Shape("dec_T1_4k_11k",   T=1,    d_in=4096,  d_out=11008),
    Shape("dec_T1_11k_4k",   T=1,    d_in=11008, d_out=4096),
    Shape("dec_T8_4k_4k",    T=8,    d_in=4096,  d_out=4096),
    Shape("dec_T16_4k_4k",   T=16,   d_in=4096,  d_out=4096),
    # moderate batch
    Shape("bat_T64_4k_4k",   T=64,   d_in=4096,  d_out=4096),
    Shape("bat_T128_4k_4k",  T=128,  d_in=4096,  d_out=4096),
    # prefill (where Triton is expected to dominate)
    Shape("pre_T512_4k_4k",  T=512,  d_in=4096,  d_out=4096),
    Shape("pre_T1024_4k_4k", T=1024, d_in=4096,  d_out=4096),
]


def _make_dense_inputs(T: int, d_out: int, d_in: int, device="cuda"):
    torch.manual_seed(0xBEEF + T + d_out + d_in)
    X = (torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4)
    perm = torch.arange(d_in, dtype=torch.int32, device=device)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)

    n_groups = d_in // BCOL
    W_s4 = torch.randint(-8, 8, (d_out, d_in), dtype=torch.int8, device=device)
    W_low_packed = pack_s4_le(W_s4)
    scale_u4 = (torch.rand(d_out, n_groups, device=device) * 0.05 + 0.001).to(torch.float16)
    zero_u4 = (torch.randn(d_out, n_groups, device=device) * 0.2).to(torch.float16)
    return X, perm, W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x


def _make_sparse_inputs(T, d_out, d_in, hp_ratio, device="cuda"):
    X, perm, W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x = (
        _make_dense_inputs(T, d_out, d_in, device=device)
    )
    nrow = d_out // BROW
    ncol = d_in // BCOL
    total_blocks = nrow * ncol
    n_hp = max(1, int(total_blocks * hp_ratio))

    torch.manual_seed((T * d_in * d_out) ^ 0xA5A5)
    flat = torch.randperm(total_blocks, device=device)[:n_hp]
    br = (flat // ncol).to(torch.int32)
    bc = (flat %  ncol).to(torch.int32)
    order = torch.argsort(br.to(torch.int64) * 1000000 + bc.to(torch.int64))
    br_sorted = br[order]
    bc_sorted = bc[order]

    W_high_s4 = torch.randint(-8, 8, (n_hp, BROW, BCOL), dtype=torch.int8, device=device)
    W_high_blocks_packed = pack_s4_le(W_high_s4)

    hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
    counts = torch.bincount(br_sorted.to(torch.int64), minlength=nrow)
    hp_row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    return dict(
        X=X, perm=perm,
        W_low_packed=W_low_packed,
        W_high_blocks_packed=W_high_blocks_packed,
        hp_row_offsets=hp_row_offsets, hp_col_indices=bc_sorted,
        X_s4=X_s4, scale_u4=scale_u4, zero_u4=zero_u4,
        sum_X=sum_X, scale_x=scale_x,
    )


# -- Individual kernel benchmarks ------------------------------------------
def bench_activation_quant(shape: Shape, log: logging.Logger):
    X = torch.randn(shape.T, shape.d_in, dtype=torch.float16, device="cuda") * 0.4
    perm = torch.arange(shape.d_in, dtype=torch.int32, device="cuda")

    t_tri = _bench_fn(lambda: quantize_activation_s4(X, perm))
    t_cud = _bench_fn(lambda: cuda_ops.activation_quant_cuda(X, perm))
    return {"triton_us": t_tri, "cuda_us": t_cud,
            "speedup": t_tri / t_cud}


def bench_dense_gemm(shape: Shape, log: logging.Logger):
    _, _, W, X_s4, scale_u4, zero_u4, sum_X, scale_x = _make_dense_inputs(
        shape.T, shape.d_out, shape.d_in
    )
    t_tri = _bench_fn(lambda: dense_gemm_u4_s4(W, X_s4, scale_u4, zero_u4, sum_X, scale_x))
    t_cud = _bench_fn(lambda: cuda_ops.dense_gemm_cuda(
        W, X_s4, scale_u4, zero_u4, sum_X, scale_x
    ))
    return {"triton_us": t_tri, "cuda_us": t_cud, "speedup": t_tri / t_cud}


def bench_sparse_gemm(shape: Shape, log: logging.Logger):
    data = _make_sparse_inputs(shape.T, shape.d_out, shape.d_in, shape.hp_ratio)
    args = (data["W_high_blocks_packed"], data["hp_row_offsets"],
            data["hp_col_indices"], data["X_s4"], data["scale_u4"],
            data["scale_x"], shape.d_out, shape.d_in)
    t_tri = _bench_fn(lambda: sparse_gemm_s4_s4(*args))
    t_cud = _bench_fn(lambda: cuda_ops.sparse_gemm_cuda(*args))
    return {"triton_us": t_tri, "cuda_us": t_cud, "speedup": t_tri / t_cud}


def bench_fused(shape: Shape, log: logging.Logger):
    data = _make_sparse_inputs(shape.T, shape.d_out, shape.d_in, shape.hp_ratio)
    args = (data["W_low_packed"], data["W_high_blocks_packed"],
            data["hp_row_offsets"], data["hp_col_indices"],
            data["X_s4"], data["scale_u4"], data["zero_u4"],
            data["sum_X"], data["scale_x"], shape.d_out, shape.d_in)
    t_tri = _bench_fn(lambda: fused_dense_sparse_gemm(*args))
    t_cud = _bench_fn(lambda: cuda_ops.fused_dense_sparse_cuda(*args))
    return {"triton_us": t_tri, "cuda_us": t_cud, "speedup": t_tri / t_cud}


def bench_end_to_end(shape: Shape, log: logging.Logger):
    """Time the full v9_linear_forward under explicit triton/cuda policies."""
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
    finally:
        set_backend_policy("auto")
    return {"triton_us": t_tri, "cuda_us": t_cud, "speedup": t_tri / t_cud}


# -- Driver -----------------------------------------------------------------
KERNELS = {
    "activation_quant": bench_activation_quant,
    "dense_gemm": bench_dense_gemm,
    "sparse_gemm": bench_sparse_gemm,
    "fused_dense_sparse": bench_fused,
    "end_to_end_v9_linear": bench_end_to_end,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", nargs="+", default=list(KERNELS.keys()),
                        help="which kernels to bench")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: repo-root/logs/cuda_kernel/)")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (_IMPORT_ROOT / "logs" / "cuda_kernel")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"bench_{ts}.log"
    json_file = out_dir / f"bench_{ts}.json"
    md_file = out_dir / f"bench_{ts}.md"

    log = _setup_logging(log_file)
    log.info("output dir: %s", out_dir)
    log.info("backend status: %s", get_backend_status())

    # Row order: (kernel, shape_name)
    results: list[dict] = []
    for kname in args.kernels:
        if kname not in KERNELS:
            log.warning("unknown kernel %s, skipping", kname)
            continue
        fn = KERNELS[kname]
        for shape in SHAPES:
            try:
                r = fn(shape, log)
                rec = {
                    "kernel": kname,
                    "shape": shape.name,
                    "T": shape.T, "d_in": shape.d_in, "d_out": shape.d_out,
                    "hp_ratio": shape.hp_ratio,
                    **r,
                }
                results.append(rec)
                log.info(
                    "%-22s %-18s triton=%8.2fus  cuda=%8.2fus  speedup=%.2fx",
                    kname, shape.name, r["triton_us"], r["cuda_us"], r["speedup"],
                )
            except Exception as e:  # noqa: BLE001
                log.exception("%s %s FAILED: %s", kname, shape.name, e)
                results.append({
                    "kernel": kname, "shape": shape.name,
                    "T": shape.T, "d_in": shape.d_in, "d_out": shape.d_out,
                    "error": str(e),
                })

    # Persist JSON
    json_file.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", json_file)

    # Markdown table per kernel
    lines: list[str] = []
    lines.append(f"# CUDA vs Triton benchmark ({ts})\n")
    lines.append("Host: `autodl` / RTX 4090 (SM89)\n")
    lines.append("Stats: min-of-means, 10 outer x 50 inner, after 10 warmup calls.\n")
    for kname in args.kernels:
        rows = [r for r in results if r["kernel"] == kname]
        if not rows:
            continue
        lines.append(f"\n## `{kname}`\n")
        lines.append("| shape | T | d_in | d_out | triton (us) | cuda (us) | speedup |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            if "error" in r:
                lines.append(
                    f"| {r['shape']} | {r['T']} | {r['d_in']} | {r['d_out']} "
                    f"| FAILED | FAILED | - |"
                )
            else:
                lines.append(
                    f"| {r['shape']} | {r['T']} | {r['d_in']} | {r['d_out']} "
                    f"| {r['triton_us']:.2f} | {r['cuda_us']:.2f} "
                    f"| {r['speedup']:.2f}x |"
                )
    md_file.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", md_file)


if __name__ == "__main__":
    main()
