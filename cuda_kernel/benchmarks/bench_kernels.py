"""CUDA / Triton / cuBLAS-FP16 latency benchmark for the V9 kernel suite.

Runs on the SM89 host (RTX 4090).  For every kernel + shape this
measures three latencies in microseconds::

    fp16   : cuBLAS FP16 matmul reference (the product-level baseline).
    triton : current Triton kernel under test.
    cuda   : our hand-written SM89 CUDA kernel.

The primary ``speedup`` column reported is ``fp16_us / cuda_us`` --
i.e. *how many times faster the CUDA path is than the cuBLAS FP16
reference*.  We additionally log ``fp16_us / triton_us`` so one can
read off Triton-vs-FP16 directly as well.

Usage (on the SM89 host / autodl)::

    python kernel/cuda_kernel/benchmarks/bench_kernels.py

Outputs (auto-timestamped):
  - log:      logs/cuda_kernel/bench_{TS}.log
  - json:     logs/cuda_kernel/bench_{TS}.json
  - markdown: logs/cuda_kernel/bench_{TS}.md

Design rules (per project preferences):
  - ``logging`` module with terminal INFO + file DEBUG (append mode).
  - Paths anchored via ``__file__``; the script is cwd-independent.
  - min-of-means statistics (10 outer batches of 50 inner calls) with
    explicit stream sync per sample.
  - Warmup phase separate from timing (Triton JITs the first call and
    its autotuner caches on first-encountered shape).

FP16 baseline definitions (applied *consistently* per kernel):

  activation_quant       -- baseline is ``X.clone()`` cost: a plain
                            fp16 memcpy of the input tensor.  This
                            represents the "unavoidable" bandwidth if
                            one had to write X once.  Informative only
                            (activation_quant is an *extra* step that
                            cuBLAS FP16 doesn't need).
  dense_gemm             -- ``torch.matmul(W_fp16, X_fp16.T) -> (d_out, T)``
                            i.e. the same logical matmul our W4A8 kernel
                            replaces.
  sparse_gemm            -- same shape, but scaled by ``hp_ratio``: this
                            represents the HP contribution that a naive
                            FP16 path would have to materialise
                            regardless of sparsity.
  fused_dense_sparse     -- ``torch.matmul(W_fp16, X_fp16.T)`` (a single
                            full matmul approximates the combined
                            dense+sparse result space).
  end_to_end_v9_linear   -- ``torch.matmul(X_fp16, W_fp16.T)`` (the same
                            (T, d_out) fp16 path the V9 linear is
                            replacing end to end -- the product-level
                            comparator).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

# ---------------------------------------------------------------------------
# Path anchoring
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
# layout: <root>/kernel/cuda_kernel/benchmarks/<this>.py
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


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def _setup_logging(log_file: Path) -> logging.Logger:
    log = logging.getLogger("bench_kernels")
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


# ---------------------------------------------------------------------------
# min-of-means timing primitive
# ---------------------------------------------------------------------------
def _bench_fn(
    fn: Callable[[], None],
    *,
    warmup: int = 10,
    outer: int = 10,
    inner: int = 50,
    device: torch.device | None = None,
) -> float:
    """Return min-of-means latency in microseconds."""
    if device is None:
        device = torch.device("cuda")
    torch.cuda.synchronize(device)
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
        batch_us = start_ev.elapsed_time(end_ev) * 1000.0 / inner
        means_us.append(batch_us)
    return min(means_us)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------
@dataclass
class Shape:
    name: str
    T: int
    d_in: int
    d_out: int
    hp_ratio: float = 0.05   # only used for sparse/fused/end2end


SHAPES: list[Shape] = [
    # decode (T=1, the main product scenario for CUDA wins)
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


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------
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
    # An FP16 reference weight matching (d_out, d_in) for baseline matmul.
    W_fp = torch.randn(d_out, d_in, dtype=torch.float16, device=device) * 0.02
    return X, perm, W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, W_fp


def _make_sparse_inputs(T, d_out, d_in, hp_ratio, device="cuda"):
    (X, perm, W_low_packed, X_s4, scale_u4, zero_u4,
     sum_X, scale_x, W_fp) = _make_dense_inputs(T, d_out, d_in, device=device)
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
        W_fp=W_fp,
    )


# ---------------------------------------------------------------------------
# Per-kernel benches.  Each returns a dict with fp16_us/triton_us/cuda_us.
# ---------------------------------------------------------------------------
def bench_activation_quant(shape: Shape, log: logging.Logger):
    X = torch.randn(shape.T, shape.d_in, dtype=torch.float16, device="cuda") * 0.4
    perm = torch.arange(shape.d_in, dtype=torch.int32, device="cuda")

    # FP16 reference: plain memcpy of the activation tensor.  This is a
    # BW-bound lower bound for "anything that has to read+write X once".
    X_buf = torch.empty_like(X)
    t_fp = _bench_fn(lambda: X_buf.copy_(X))
    t_tri = _bench_fn(lambda: quantize_activation_s4(X, perm))
    t_cud = _bench_fn(lambda: cuda_ops.activation_quant_cuda(X, perm))
    return {"fp16_us": t_fp, "triton_us": t_tri, "cuda_us": t_cud}


def bench_dense_gemm(shape: Shape, log: logging.Logger):
    (_, _, W, X_s4, scale_u4, zero_u4,
     sum_X, scale_x, W_fp) = _make_dense_inputs(shape.T, shape.d_out, shape.d_in)
    X_fp = torch.randn(shape.T, shape.d_in, dtype=torch.float16, device="cuda") * 0.4
    # Baseline: produce (d_out, T) via cuBLAS FP16.
    t_fp = _bench_fn(lambda: torch.matmul(W_fp, X_fp.t()))
    t_tri = _bench_fn(lambda: dense_gemm_u4_s4(W, X_s4, scale_u4, zero_u4, sum_X, scale_x))
    t_cud = _bench_fn(lambda: cuda_ops.dense_gemm_cuda(
        W, X_s4, scale_u4, zero_u4, sum_X, scale_x
    ))
    return {"fp16_us": t_fp, "triton_us": t_tri, "cuda_us": t_cud}


def bench_sparse_gemm(shape: Shape, log: logging.Logger):
    data = _make_sparse_inputs(shape.T, shape.d_out, shape.d_in, shape.hp_ratio)
    args = (data["W_high_blocks_packed"], data["hp_row_offsets"],
            data["hp_col_indices"], data["X_s4"], data["scale_u4"],
            data["scale_x"], shape.d_out, shape.d_in)
    # FP16 baseline: a full (d_out, T) matmul.  A naive FP16 path that
    # reconstructs the sparse contribution densely pays the full matmul
    # cost; this is the honest comparator.
    X_fp = torch.randn(shape.T, shape.d_in, dtype=torch.float16, device="cuda") * 0.4
    W_fp = data["W_fp"]
    t_fp = _bench_fn(lambda: torch.matmul(W_fp, X_fp.t()))
    t_tri = _bench_fn(lambda: sparse_gemm_s4_s4(*args))
    t_cud = _bench_fn(lambda: cuda_ops.sparse_gemm_cuda(*args))
    return {"fp16_us": t_fp, "triton_us": t_tri, "cuda_us": t_cud}


def bench_fused(shape: Shape, log: logging.Logger):
    data = _make_sparse_inputs(shape.T, shape.d_out, shape.d_in, shape.hp_ratio)
    args = (data["W_low_packed"], data["W_high_blocks_packed"],
            data["hp_row_offsets"], data["hp_col_indices"],
            data["X_s4"], data["scale_u4"], data["zero_u4"],
            data["sum_X"], data["scale_x"], shape.d_out, shape.d_in)
    X_fp = torch.randn(shape.T, shape.d_in, dtype=torch.float16, device="cuda") * 0.4
    W_fp = data["W_fp"]
    t_fp = _bench_fn(lambda: torch.matmul(W_fp, X_fp.t()))
    t_tri = _bench_fn(lambda: fused_dense_sparse_gemm(*args))
    t_cud = _bench_fn(lambda: cuda_ops.fused_dense_sparse_cuda(*args))
    return {"fp16_us": t_fp, "triton_us": t_tri, "cuda_us": t_cud}


def bench_end_to_end(shape: Shape, log: logging.Logger):
    """Time the full v9_linear_forward under explicit triton/cuda policies,
    plus the cuBLAS FP16 equivalent as the product-level baseline."""
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
    # v9_linear_forward returns (T, d_out); compare to X @ W_fp.T.
    W_fp = data["W_fp"]
    t_fp = _bench_fn(lambda: torch.matmul(X, W_fp.t()))

    try:
        set_backend_policy("triton")
        t_tri = _bench_fn(lambda: v9_linear_forward(X, W))
        set_backend_policy("cuda")
        t_cud = _bench_fn(lambda: v9_linear_forward(X, W))
    finally:
        set_backend_policy("auto")
    return {"fp16_us": t_fp, "triton_us": t_tri, "cuda_us": t_cud}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
KERNELS = {
    "activation_quant": bench_activation_quant,
    "dense_gemm": bench_dense_gemm,
    "sparse_gemm": bench_sparse_gemm,
    "fused_dense_sparse": bench_fused,
    "end_to_end_v9_linear": bench_end_to_end,
}


def _enrich(rec: dict) -> dict:
    """Add derived speedup columns.  Primary speedup is vs fp16."""
    fp16 = rec.get("fp16_us")
    tri = rec.get("triton_us")
    cud = rec.get("cuda_us")
    if fp16 and cud:
        rec["speedup_cuda_vs_fp16"] = fp16 / cud
    if fp16 and tri:
        rec["speedup_triton_vs_fp16"] = fp16 / tri
    if tri and cud:
        rec["speedup_cuda_vs_triton"] = tri / cud
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", nargs="+", default=list(KERNELS.keys()),
                        help="which kernels to bench")
    parser.add_argument("--out", type=Path, default=None,
                        help="output dir (default: <repo-root>/logs/cuda_kernel/)")
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
    log.info("baseline for speedup is cuBLAS FP16 matmul (torch.matmul)")

    results: list[dict] = []
    for kname in args.kernels:
        if kname not in KERNELS:
            log.warning("unknown kernel %s, skipping", kname)
            continue
        fn = KERNELS[kname]
        for shape in SHAPES:
            try:
                r = _enrich(fn(shape, log))
                rec = {
                    "kernel": kname,
                    "shape": shape.name,
                    "T": shape.T, "d_in": shape.d_in, "d_out": shape.d_out,
                    "hp_ratio": shape.hp_ratio,
                    **r,
                }
                results.append(rec)
                log.info(
                    "%-22s %-18s fp16=%7.2fus  triton=%7.2fus  cuda=%7.2fus  "
                    "cuda/fp16=%.2fx  triton/fp16=%.2fx",
                    kname, shape.name,
                    r["fp16_us"], r["triton_us"], r["cuda_us"],
                    r.get("speedup_cuda_vs_fp16", 0.0),
                    r.get("speedup_triton_vs_fp16", 0.0),
                )
            except Exception as e:  # noqa: BLE001
                log.exception("%s %s FAILED: %s", kname, shape.name, e)
                results.append({
                    "kernel": kname, "shape": shape.name,
                    "T": shape.T, "d_in": shape.d_in, "d_out": shape.d_out,
                    "error": str(e),
                })

    json_file.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", json_file)

    # Markdown table per kernel
    lines: list[str] = []
    lines.append(f"# CUDA / Triton / cuBLAS-FP16 benchmark ({ts})\n")
    lines.append("Host: `autodl` / RTX 4090 (SM89)\n")
    lines.append("Stats: min-of-means, 10 outer x 50 inner, after 10 warmup calls.\n")
    lines.append("**Baseline** for the `speedup` column is cuBLAS FP16 "
                 "matmul (`torch.matmul` on `torch.float16` tensors).\n")

    for kname in args.kernels:
        rows = [r for r in results if r["kernel"] == kname]
        if not rows:
            continue
        lines.append(f"\n## `{kname}`\n")
        lines.append("| shape | T | d_in | d_out "
                     "| fp16 (us) | triton (us) | cuda (us) "
                     "| **cuda/fp16** | triton/fp16 | cuda/triton |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            if "error" in r:
                lines.append(
                    f"| {r['shape']} | {r['T']} | {r['d_in']} | {r['d_out']} "
                    f"| FAILED | FAILED | FAILED | - | - | - |"
                )
            else:
                lines.append(
                    f"| {r['shape']} | {r['T']} | {r['d_in']} | {r['d_out']} "
                    f"| {r['fp16_us']:.2f} | {r['triton_us']:.2f} | {r['cuda_us']:.2f} "
                    f"| **{r.get('speedup_cuda_vs_fp16', 0):.2f}x** "
                    f"| {r.get('speedup_triton_vs_fp16', 0):.2f}x "
                    f"| {r.get('speedup_cuda_vs_triton', 0):.2f}x |"
                )
    md_file.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", md_file)


if __name__ == "__main__":
    main()
