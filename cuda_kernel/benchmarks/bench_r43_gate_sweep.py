"""R43 gate-sweep benchmark: map the full kBm=64 opportunity surface.

Explores two R43 hypotheses:

  R43-A: re-evaluate T=64 under hp>0 sparse path (R42-P1 changed the
         sparse CTA mapping; the T=64 verdict from R41 hp=0 may not hold).

  R43-B: explore d_out=4096 — gate currently blocks it, but sparse
         branch now supports kBm=64.  Is there a (T, d_out=4096) regime
         where kBm=64 wins?

Methodology [[memory:bmmiahpl]]:
    50 warm-up + 3 x 100-iter windows + min-of-means.

Sweeps T in {8, 16, 32, 48, 64, 96, 128} x d_out in {1024, 2048, 3072,
4096} x hp_ratio in {0.0, 0.05}.  Each (T, d_out, hp) tuple tested with
force_kBm=128 and force_kBm=64.  Output heat-map shows speedup.

Usage:
    python kernel/cuda_kernel/benchmarks/bench_r43_gate_sweep.py
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
    log = logging.getLogger("bench_r43_gate_sweep")
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


def _run(inputs, d_out, d_in):
    (W_low, W_hi, rowoff, colind, X_s4, su4, zu4, sX, sx) = inputs
    return cuda_ops.fused_dense_sparse_cuda_int4(
        W_low, W_hi, rowoff, colind, X_s4, su4, zu4, sX, sx, d_out, d_in,
    )


def _time_case(force_kbm, T, d_out, d_in, hp_ratio):
    if force_kbm is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = force_kbm
    inputs = _make_inputs(T, d_out, d_in, hp_ratio)
    fn = lambda: _run(inputs, d_out, d_in)
    return time_ms(fn, n_warmup=50, n_iter=100, n_repeat=3) * 1000


# Sweep grid.
TS      = [8, 16, 32, 48, 64, 96, 128]
D_OUTS  = [1024, 2048, 3072, 4096]
HP_RATIOS = [0.0, 0.05]
D_IN    = 4096  # fixed; change if needed


def main():
    torch.cuda.init()
    torch.backends.cudnn.benchmark = False
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_root = _THIS.parents[1] / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"bench_r43_gate_sweep_{ts}.log"
    log = _setup_logging(log_file)

    dev = torch.cuda.get_device_name(0)
    log.info(f"Device: {dev}, d_in={D_IN}")
    log.info(f"Grid: T={TS}  d_out={D_OUTS}  hp={HP_RATIOS}")

    rows = []
    for hp in HP_RATIOS:
        for d_out in D_OUTS:
            # Skip if d_out not BROW-aligned.
            if d_out % BROW != 0:
                continue
            for T in TS:
                log.info(f"--- hp={hp} d_out={d_out} T={T} ---")
                us_128 = _time_case("128", T, d_out, D_IN, hp)
                us_64  = _time_case("64",  T, d_out, D_IN, hp)
                us_auto = _time_case(None, T, d_out, D_IN, hp)
                speedup = us_128 / us_64 if us_64 > 0 else float("nan")
                # auto_picks: which forced run is auto closest to?
                #   Closer-to-64 => auto chose kBm=64.
                auto_picks = "64" if abs(us_auto - us_64) < abs(us_auto - us_128) else "128"
                row = {
                    "hp_ratio": hp, "T": T, "d_out": d_out, "d_in": D_IN,
                    "us_kbm128": us_128, "us_kbm64": us_64, "us_auto": us_auto,
                    "auto_picks": auto_picks,
                    "speedup_64_over_128": speedup,
                }
                rows.append(row)
                tag = "✓" if speedup >= 1.05 else ("×" if speedup <= 0.95 else "·")
                log.info(f"  kBm=128 {us_128:6.2f}us  kBm=64 {us_64:6.2f}us  "
                         f"auto {us_auto:6.2f}us (picks={auto_picks})  "
                         f"64/128={speedup:.3f}x {tag}")

    os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)

    out_json = log_root / f"bench_r43_gate_sweep_{ts}.json"
    out_md   = log_root / f"bench_r43_gate_sweep_{ts}.md"
    with out_json.open("w") as f:
        json.dump({"device": dev, "d_in": D_IN, "rows": rows}, f, indent=2)

    # Build heat-map MD.
    lines = [
        "# R43 gate sweep — kBm=64 opportunity surface",
        f"- device: {dev}",
        f"- d_in: {D_IN}",
        f"- timestamp: {ts}",
        "- cell: speedup(kBm=64 / kBm=128).  ✓=>1.05, ×=<0.95, ·=neutral",
        "",
    ]
    for hp in HP_RATIOS:
        lines.append(f"## hp_ratio = {hp}")
        lines.append("")
        lines.append("| T \\ d_out | " + " | ".join(str(d) for d in D_OUTS) + " |")
        lines.append("|" + "---|" * (len(D_OUTS) + 1))
        for T in TS:
            cells = []
            for d in D_OUTS:
                r = next((x for x in rows
                          if x["hp_ratio"] == hp and x["T"] == T
                          and x["d_out"] == d), None)
                if r is None:
                    cells.append("-")
                    continue
                s = r["speedup_64_over_128"]
                tag = "✓" if s >= 1.05 else ("×" if s <= 0.95 else "·")
                cells.append(f"{s:.3f} {tag}")
            lines.append(f"| T={T} | " + " | ".join(cells) + " |")
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    log.info(f"Wrote {out_json}")
    log.info(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
