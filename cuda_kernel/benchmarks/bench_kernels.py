"""CUDA INT4 MMA vs cuBLAS-FP16 benchmark for the V9 kernel suite.

Runs on the SM89 host (RTX 4090).  For every kernel + shape this
measures two latencies in microseconds::

    fp16       : cuBLAS FP16 matmul reference (product baseline).
    cuda_int4  : our SM89 Tensor Core kernel using
                 mma.m16n8k64.s4.s4.s32 (INT4 Tensor Core).

The Triton path is intentionally not benchmarked here; for a
Triton-vs-CUDA correctness check see
``kernel.cuda_kernel.tests.test_parity``.  The INT8 MMA variant was
archived in Round 12 after Round 11 showed INT4 MMA was 1.7-1.9x
faster on every shape.
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
_IMPORT_ROOT = _THIS.parents[3]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_s4_le
from kernel.cuda_kernel import ops as cuda_ops


# ---------------------------------------------------------------------------
# Logging
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
# Timing (min-of-means)
# ---------------------------------------------------------------------------
def _bench_fn(
    fn: Callable[[], None],
    *,
    warmup: int = 10,
    outer: int = 10,
    inner: int = 50,
    device: torch.device | None = None,
) -> float:
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
    hp_ratio: float = 0.05


SHAPES: list[Shape] = [
    # ---- decode (T=1) ----
    Shape("dec_T1_4k_4k",    T=1,    d_in=4096,  d_out=4096),
    Shape("dec_T1_4k_11k",   T=1,    d_in=4096,  d_out=11008),
    Shape("dec_T1_11k_4k",   T=1,    d_in=11008, d_out=4096),
    # ---- small-batch decode (speculative / tree / T<=16) ----
    Shape("dec_T8_4k_4k",    T=8,    d_in=4096,  d_out=4096),
    Shape("dec_T16_4k_4k",   T=16,   d_in=4096,  d_out=4096),
    # ---- mid-batch (continuous batching, T in [64, 128]) ----
    Shape("bat_T64_4k_4k",   T=64,   d_in=4096,  d_out=4096),
    Shape("bat_T128_4k_4k",  T=128,  d_in=4096,  d_out=4096),
    Shape("bat_T128_4k_11k", T=128,  d_in=4096,  d_out=11008),
    Shape("bat_T128_11k_4k", T=128,  d_in=11008, d_out=4096),
    # ---- prefill (T >= 512) ----
    Shape("pre_T512_4k_4k",  T=512,  d_in=4096,  d_out=4096),
    Shape("pre_T1024_4k_4k", T=1024, d_in=4096,  d_out=4096),
    Shape("pre_T2048_4k_4k", T=2048, d_in=4096,  d_out=4096),
    Shape("pre_T1024_4k_11k",T=1024, d_in=4096,  d_out=11008),
    Shape("pre_T1024_11k_4k",T=1024, d_in=11008, d_out=4096),
]


# ---------------------------------------------------------------------------
# Inputs
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
    bc = (flat % ncol).to(torch.int32)
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
# Per-kernel benches.  Each returns {fp16_us, int4_us}.
# ---------------------------------------------------------------------------
def bench_activation_quant(shape: Shape, log: logging.Logger):
    X = torch.randn(shape.T, shape.d_in, dtype=torch.float16, device="cuda") * 0.4
    perm = torch.arange(shape.d_in, dtype=torch.int32, device="cuda")
    X_buf = torch.empty_like(X)
    t_fp = _bench_fn(lambda: X_buf.copy_(X))
    t_c = _bench_fn(lambda: cuda_ops.activation_quant_cuda(X, perm))
    return {"fp16_us": t_fp, "int4_us": t_c}


def bench_dense_gemm(shape: Shape, log: logging.Logger):
    (X_fp, _, W, X_s4, scale_u4, zero_u4,
     sum_X, scale_x, W_fp) = _make_dense_inputs(shape.T, shape.d_out, shape.d_in)
    # Baseline: W_fp (d_out, d_in) @ X_fp.t() -> (d_out, T), matches int4 output layout.
    t_fp = _bench_fn(lambda: torch.matmul(W_fp, X_fp.t()))
    # Auto-dispatch: T=1 -> dp4a GEMV kernel (Round 13); T>1 -> INT4 MMA (Round 12).
    t_i4 = _bench_fn(lambda: cuda_ops.dense_gemm_cuda(
        W, X_s4, scale_u4, zero_u4, sum_X, scale_x
    ))
    return {"fp16_us": t_fp, "int4_us": t_i4}


def bench_sparse_gemm(shape: Shape, log: logging.Logger):
    data = _make_sparse_inputs(shape.T, shape.d_out, shape.d_in, shape.hp_ratio)
    args = (data["W_high_blocks_packed"], data["hp_row_offsets"],
            data["hp_col_indices"], data["X_s4"], data["scale_u4"],
            data["scale_x"], shape.d_out, shape.d_in)
    X_fp = data["X"]
    W_fp = data["W_fp"]
    # Baseline: W_fp @ X_fp.t() -> (d_out, T), matches int4 output layout.
    t_fp = _bench_fn(lambda: torch.matmul(W_fp, X_fp.t()))
    t_i4 = _bench_fn(lambda: cuda_ops.sparse_gemm_cuda_int4(*args))
    return {"fp16_us": t_fp, "int4_us": t_i4}


def bench_fused(shape: Shape, log: logging.Logger):
    data = _make_sparse_inputs(shape.T, shape.d_out, shape.d_in, shape.hp_ratio)
    args = (data["W_low_packed"], data["W_high_blocks_packed"],
            data["hp_row_offsets"], data["hp_col_indices"],
            data["X_s4"], data["scale_u4"], data["zero_u4"],
            data["sum_X"], data["scale_x"], shape.d_out, shape.d_in)
    X_fp = data["X"]
    W_fp = data["W_fp"]
    # Baseline: W_fp @ X_fp.t() -> (d_out, T), matches int4 output layout.
    t_fp = _bench_fn(lambda: torch.matmul(W_fp, X_fp.t()))
    # Auto-dispatch: T=1 -> dp4a GEMV (Round 14); T>1 -> INT4 MMA (Round 12).
    t_i4 = _bench_fn(lambda: cuda_ops.fused_dense_sparse_cuda(*args))
    return {"fp16_us": t_fp, "int4_us": t_i4}


def bench_end_to_end(shape: Shape, log: logging.Logger):
    """Time the full CUDA-backed pipeline (quant + fused GEMM) with INT4 MMA."""
    data = _make_sparse_inputs(shape.T, shape.d_out, shape.d_in, shape.hp_ratio)

    X_fp = data["X"]
    W_fp = data["W_fp"]

    perm = data["perm"]
    W_low_packed = data["W_low_packed"]
    W_high_blocks_packed = data["W_high_blocks_packed"]
    hp_row_offsets = data["hp_row_offsets"]
    hp_col_indices = data["hp_col_indices"]
    scale_u4 = data["scale_u4"]
    zero_u4 = data["zero_u4"]

    # Baseline: W_fp @ X_fp.t() -> (d_out, T), matches int4 output layout.
    # (Was X_fp @ W_fp.t() -> (T, d_out); aligned with other benches now.)
    t_fp = _bench_fn(lambda: torch.matmul(W_fp, X_fp.t()))

    def run_pipeline():
        if shape.T == 1:
            # Round 15c: single fused kernel (quant + GEMV, no intermediate HBM).
            return cuda_ops.fused_quant_gemv_cuda(
                X_fp, perm,
                W_low_packed, W_high_blocks_packed,
                hp_row_offsets, hp_col_indices,
                scale_u4, zero_u4,
                shape.d_out, shape.d_in,
            )
        X_s4, scale_x, sum_X = cuda_ops.activation_quant_cuda(X_fp, perm)
        return cuda_ops.fused_dense_sparse_cuda(
            W_low_packed, W_high_blocks_packed,
            hp_row_offsets, hp_col_indices,
            X_s4, scale_u4, zero_u4, sum_X, scale_x,
            shape.d_out, shape.d_in,
        )

    t_i4 = _bench_fn(run_pipeline)
    return {"fp16_us": t_fp, "int4_us": t_i4}


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
    fp16 = rec.get("fp16_us")
    i4 = rec.get("int4_us")
    if fp16 and i4:
        rec["speedup_int4_vs_fp16"] = fp16 / i4
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernels", nargs="+", default=list(KERNELS.keys()))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (_IMPORT_ROOT / "logs" / "cuda_kernel")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"bench_{ts}.log"
    json_file = out_dir / f"bench_{ts}.json"
    md_file = out_dir / f"bench_{ts}.md"

    log = _setup_logging(log_file)
    log.info("output dir: %s", out_dir)
    log.info("baseline: cuBLAS FP16 matmul (torch.matmul, fp16)")
    log.info("comparing: cuda_int4 (mma.m16n8k64.s4)")

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
                    "%-22s %-18s fp16=%7.2fus  int4=%7.2fus  "
                    "int4/fp16=%.2fx",
                    kname, shape.name,
                    r["fp16_us"], r["int4_us"],
                    r.get("speedup_int4_vs_fp16", 0.0),
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

    lines: list[str] = []
    lines.append(f"# CUDA INT4 MMA benchmark ({ts})\n")
    lines.append("Host: `autodl` / RTX 4090 (SM89)\n")
    lines.append("Stats: min-of-means, 10 outer x 50 inner, after 10 warmup calls.\n")
    lines.append("**Baseline**: cuBLAS FP16 matmul (`torch.matmul` on `torch.float16`).\n")
    lines.append(
        "**CUDA path**: `mma.m16n8k64.s4.s4.s32` (native INT4 Tensor Core MMA).\n"
    )

    for kname in args.kernels:
        rows = [r for r in results if r["kernel"] == kname]
        if not rows:
            continue
        lines.append(f"\n## `{kname}`\n")
        lines.append("| shape | T | d_in | d_out "
                     "| fp16 (us) | int4 (us) | **int4/fp16** |")
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
                    f"| {r['fp16_us']:.2f} | {r['int4_us']:.2f} "
                    f"| **{r.get('speedup_int4_vs_fp16', 0):.2f}x** |"
                )
    md_file.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", md_file)


if __name__ == "__main__":
    main()
