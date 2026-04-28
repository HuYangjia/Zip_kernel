"""R42-P1 benchmark: fused dense+sparse kernel, hp_ratio=0.05, kBm gate A/B.

Purpose
-------
R42-P1 extends the R41-P1 kBm=64 opt-in to hp>0 shapes by remapping
sparse-branch CTA rows (br -> bsr_br, half_row_off).  The production
workload uses hp_ratio=0.05, so this bench measures whether kBm=64
retains its R41-P1 hp=0 speedup when the sparse branch is active.

A: HKUST_V9_FUSED_FORCE_KBM=128  -> baseline (pre-R42)
B: HKUST_V9_FUSED_FORCE_KBM=64   -> new R42 path
C: unset                         -> R42 default gate

Methodology [[memory:bmmiahpl]]:
    50 warm-up + 3 x 100-iter windows + min-of-means.

Usage:
    python kernel/cuda_kernel/benchmarks/bench_r42_fused_hp05.py

Output:
    logs/cuda_kernel/bench_r42_fused_hp05_{TS}.{json,md,log}
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

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
    log = logging.getLogger("bench_r42_fused_hp05")
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


def _make_fused_hp05_inputs(T: int, d_out: int, d_in: int,
                            hp_ratio: float = 0.05, seed: int = 0xBEEF):
    """Build a fused hp>0 input with bsr_nnz ~= hp_ratio * (nrow * ncol)."""
    torch.manual_seed(seed)
    device = "cuda"
    X = torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device=device)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)

    n_groups = d_in // BCOL
    W_low_s4 = torch.randint(
        -8, 8, (d_out, d_in), dtype=torch.int8, device=device
    )
    W_low_packed = pack_s4_le(W_low_s4)
    scale_u4 = (
        torch.rand(d_out, n_groups, device=device) * 0.05 + 0.001
    ).to(torch.float16)
    zero_u4 = (
        torch.randn(d_out, n_groups, device=device) * 0.2
    ).to(torch.float16)

    # Build BSR sparse W_high: (d_out / BROW) x (d_in / BCOL) block grid.
    nrow = d_out // BROW
    ncol = d_in // BCOL
    total_blocks = nrow * ncol
    nnz = max(1, int(round(total_blocks * hp_ratio)))

    # Uniformly random block positions, one per row on average.
    row_ids = torch.randint(0, nrow, (nnz,), device=device)
    col_ids = torch.randint(0, ncol, (nnz,), device=device)
    # Sort by row for BSR row_offsets compatibility.
    order = torch.argsort(row_ids * (ncol + 1) + col_ids)
    row_ids = row_ids[order]
    col_ids = col_ids[order]

    # Build row_offsets.
    hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
    counts = torch.bincount(row_ids, minlength=nrow)
    hp_row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    hp_col_indices = col_ids.to(torch.int32)

    # Block values (random int8 4-bit packed).  Shape (nnz, BROW, BCOL//2).
    W_high_blocks_packed = torch.randint(
        -128, 127, (nnz, BROW, BCOL // 2), dtype=torch.int8, device=device
    )

    return (
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
    )


def _run_fused_cuda(inputs, d_out: int, d_in: int):
    (
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
    ) = inputs
    return cuda_ops.fused_dense_sparse_cuda_int4(
        W_low_packed, W_high_blocks_packed,
        hp_row_offsets, hp_col_indices,
        X_s4, scale_u4, zero_u4, sum_X, scale_x,
        d_out, d_in,
    )


# (label, T, d_out, d_in, gate_expected)
# Focus on d_out <= 2048 shapes where R41/R42 gate can fire.
SHAPES: List[Tuple[str, int, int, int, bool]] = [
    # Qwen3-1.7B (d_out=2048 -> gate hits in T=16..32)
    ("1.7B_q_proj",    16, 2048, 2048, True),
    ("1.7B_q_proj",    32, 2048, 2048, True),
    ("1.7B_q_proj",    64, 2048, 2048, False),
    ("1.7B_o_proj",    16, 2048, 2048, True),
    ("1.7B_o_proj",    32, 2048, 2048, True),
    ("1.7B_down_proj", 16, 2048, 6144, True),
    # Qwen3-14B (d_out=1024, d_in=5120)
    ("14B_kv_proj",    16, 1024, 5120, True),
    ("14B_kv_proj",    32, 1024, 5120, True),
    # Control: d_out=4096 (gate should miss)
    ("8B_q_proj",      16, 4096, 4096, False),
]


def _run_one(force_kbm, label, T, d_out, d_in, log, hp_ratio):
    if force_kbm is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = force_kbm
    inputs = _make_fused_hp05_inputs(T, d_out, d_in, hp_ratio=hp_ratio)
    fn = lambda: _run_fused_cuda(inputs, d_out, d_in)
    ms = time_ms(fn, n_warmup=50, n_iter=100, n_repeat=3)
    log.info(
        f"  [kBm={force_kbm or 'gate'}] {label} T={T} d_out={d_out} "
        f"d_in={d_in}: {ms*1000:.2f} us"
    )
    return ms


def main():
    torch.cuda.init()
    torch.backends.cudnn.benchmark = False
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_root = _THIS.parents[1] / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"bench_r42_fused_hp05_{ts}.log"
    log = _setup_logging(log_file)

    hp_ratio = 0.05
    dev = torch.cuda.get_device_name(0)
    log.info(f"Device: {dev}")
    log.info(f"hp_ratio = {hp_ratio}  |  shapes: {len(SHAPES)}")

    rows = []
    for label, T, d_out, d_in, gate_expected in SHAPES:
        log.info(f"=== {label} T={T} d_out={d_out} d_in={d_in} "
                 f"(gate_expected={gate_expected}) ===")
        ms_128  = _run_one("128", label, T, d_out, d_in, log, hp_ratio)
        ms_64   = _run_one("64",  label, T, d_out, d_in, log, hp_ratio)
        ms_auto = _run_one(None,  label, T, d_out, d_in, log, hp_ratio)
        speedup = ms_128 / ms_64 if ms_64 > 0 else float("nan")
        rows.append({
            "label": label, "T": T, "d_out": d_out, "d_in": d_in,
            "gate_expected": gate_expected,
            "us_kbm128": ms_128 * 1000,
            "us_kbm64":  ms_64  * 1000,
            "us_auto":   ms_auto * 1000,
            "speedup_64_over_128": speedup,
        })
        log.info(f"  -> speedup(kBm=64 / kBm=128) = {speedup:.3f}x\n")

    os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)

    out_json = log_root / f"bench_r42_fused_hp05_{ts}.json"
    out_md   = log_root / f"bench_r42_fused_hp05_{ts}.md"
    with out_json.open("w") as f:
        json.dump({"device": dev, "hp_ratio": hp_ratio, "rows": rows}, f, indent=2)
    lines = [
        "# R42-P1 fused hp_ratio=0.05 bench",
        f"- device: {dev}",
        f"- hp_ratio: {hp_ratio}",
        f"- timestamp: {ts}",
        "",
        "| label | T | d_out | d_in | gate | us kBm=128 | us kBm=64 | "
        "us auto | speedup(64/128) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['T']} | {r['d_out']} | {r['d_in']} | "
            f"{'✓' if r['gate_expected'] else '✗'} | "
            f"{r['us_kbm128']:.2f} | {r['us_kbm64']:.2f} | "
            f"{r['us_auto']:.2f} | {r['speedup_64_over_128']:.3f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    log.info(f"Wrote {out_json}")
    log.info(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
