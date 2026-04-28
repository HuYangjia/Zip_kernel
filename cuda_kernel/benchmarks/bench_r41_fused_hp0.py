"""R41-P1 benchmark: fused dense+sparse kernel, hp=0 regime, kBm gate A/B.

Purpose
-------
Round 41-P1 adds a kBm=64 opt-in path in fused_dense_sparse_mma_int4.cu.
The default gate only fires when:
    hp_col_indices.numel() == 0  AND  T in [16,64]  AND  d_out <= 2048
i.e. decode-like small-batch requests where the kBm=128 grid under-fills
a wave.  Production workloads (hp_ratio=0.05) never fire the gate, so
this bench uses hp=0 (empty BSR) to exercise the new path and compare:

    A: HKUST_V9_FUSED_FORCE_KBM=128  -> baseline (pre-R41)
    B: HKUST_V9_FUSED_FORCE_KBM=64   -> new kBm=64 path
    C: unset                         -> R41 default gate (= A or B)

Methodology (per project microbench contract [[memory:bmmiahpl]]):
    50 warm-up + 3 x 100-iter windows + min-of-means.  CUDA events,
    no nsys.  All heuristics locked via env var during each window.

Usage:
    python kernel/cuda_kernel/benchmarks/bench_r41_fused_hp0.py

Output:
    logs/cuda_kernel/bench_r41_fused_hp0_{TS}.json
    logs/cuda_kernel/bench_r41_fused_hp0_{TS}.md
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


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_file: Path) -> logging.Logger:
    log = logging.getLogger("bench_r41_fused_hp0")
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


# ---------------------------------------------------------------------------
# Input generation (hp=0)
# ---------------------------------------------------------------------------

def _make_fused_hp0_inputs(T: int, d_out: int, d_in: int, seed: int = 0xBEEF):
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

    nrow = d_out // BROW
    W_high_blocks_packed = torch.zeros(
        (0, BROW, BCOL // 2), dtype=torch.int8, device=device
    )
    hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
    hp_col_indices = torch.zeros((0,), dtype=torch.int32, device=device)

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


# ---------------------------------------------------------------------------
# Shapes (Qwen3-like attention/MLP, decode regime)
# ---------------------------------------------------------------------------

# (label, T, d_out, d_in, gate_expected_to_fire)
SHAPES: List[Tuple[str, int, int, int, bool]] = [
    # Qwen3-1.7B style (d_out=2048 -> gate HITS for T in [16,64])
    ("1.7B_q_proj",    16, 2048, 2048, True),
    ("1.7B_q_proj",    32, 2048, 2048, True),
    ("1.7B_q_proj",    64, 2048, 2048, True),
    ("1.7B_q_proj",   128, 2048, 2048, False),  # control: T>64, gate misses
    ("1.7B_o_proj",    16, 2048, 2048, True),
    ("1.7B_down_proj", 16, 2048, 6144, True),
    # Qwen3-8B style (d_out=4096 -> gate misses: d_out>2048)
    ("8B_q_proj",      16, 4096, 4096, False),
    ("8B_q_proj",      32, 4096, 4096, False),
    # Qwen3-14B style
    ("14B_kv_proj",    16, 1024, 5120, True),   # small d_out, gate hits
    # NOTE: 14B_q_proj d_out=5120 + d_in=5120 (n_groups=40) would, if
    #   forced to kBm=64, lose the n_groups windowed cache (n_cta_m>64
    #   disables use_group_cache).  The default gate correctly skips
    #   this (d_out>2048) so we do NOT force-test it here; use the
    #   existing bench_cuda_vs_triton.py for that shape.
]


# ---------------------------------------------------------------------------
# Bench core
# ---------------------------------------------------------------------------

def _run_one(force_kbm: str | None, label: str, T, d_out, d_in, log):
    """Run one (shape, gate-setting) configuration and return mean latency (ms)."""
    # Set the env var BEFORE building inputs to also cover the first
    # CUDA launch's driver warm-up.
    if force_kbm is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = force_kbm

    inputs = _make_fused_hp0_inputs(T, d_out, d_in)
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
    log_file = log_root / f"bench_r41_fused_hp0_{ts}.log"
    log = _setup_logging(log_file)

    dev = torch.cuda.get_device_name(0)
    log.info(f"Device: {dev}")
    log.info(f"Shapes: {len(SHAPES)}")

    rows = []
    for label, T, d_out, d_in, gate_expected in SHAPES:
        log.info(f"=== {label} T={T} d_out={d_out} d_in={d_in} "
                 f"(gate_expected={gate_expected}) ===")
        ms_128  = _run_one("128",  label, T, d_out, d_in, log)
        ms_64   = _run_one("64",   label, T, d_out, d_in, log)
        ms_auto = _run_one(None,   label, T, d_out, d_in, log)

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

    # Persist
    out_json = log_root / f"bench_r41_fused_hp0_{ts}.json"
    out_md   = log_root / f"bench_r41_fused_hp0_{ts}.md"
    with out_json.open("w") as f:
        json.dump({"device": dev, "rows": rows}, f, indent=2)
    lines = ["# R41-P1 fused hp=0 bench",
             f"- device: {dev}",
             f"- timestamp: {ts}",
             "",
             "| label | T | d_out | d_in | gate | us kBm=128 | us kBm=64 | "
             "us auto | speedup(64/128) |",
             "|---|---|---|---|---|---|---|---|---|"]
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
