"""R45 probe: can we save the T=96 d_out>=3072 bad zone?

R44 accepted gate with T==96 only at d<=2048 because:
    d=3072 T=96 kBm=64 auto -> 0.913x (bad)
    d=4096 T=96 kBm=64 auto -> 0.524x (catastrophic)

But R44's kBn demote already forces kBn=8 at T in [32,96], so those
bad numbers ARE the post-demote speedups.  We now ask: is the kBn
demote itself wrong for T=96 at these d_out?  Specifically try all
(kBm, kBn) combos at T=96 d=3072 and T=96 d=4096 to see if any cell
saves the shape.

Methodology [[memory:bmmiahpl]]:
    50 warm-up + 3 x 100-iter windows + min-of-means.

Usage:
    python kernel/cuda_kernel/benchmarks/bench_r45_t96_probe.py
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
    log = logging.getLogger("bench_r45_t96_probe")
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


def _time_case(force_kbm, force_kbn, T, d_out, d_in, hp_ratio):
    if force_kbm is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = force_kbm
    if force_kbn is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBN", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBN"] = force_kbn
    inputs = _make_inputs(T, d_out, d_in, hp_ratio)
    fn = lambda: _run(inputs, d_out, d_in)
    return time_ms(fn, n_warmup=50, n_iter=100, n_repeat=3) * 1000


# R44 bad zone + some good-zone T for contrast.
PROBES = [
    # (T, d_out) triples at hp=0.05
    (96, 3072),
    (96, 4096),
    (64, 4096),   # good zone — R44 gate accepts
    (48, 4096),   # good zone
]
COMBOS = [
    ("128", "8"),   # baseline: legacy
    ("128", "32"),  # baseline: legacy
    ("128", "64"),  # baseline: legacy
    ("64",  "8"),   # R44 auto for T in [32,96]
    ("64",  "32"),
    ("64",  "64"),
]
HP_RATIOS = [0.0, 0.05]
D_IN = 4096


def main():
    torch.cuda.init()
    torch.backends.cudnn.benchmark = False
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_root = _THIS.parents[1] / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"bench_r45_t96_probe_{ts}.log"
    log = _setup_logging(log_file)

    dev = torch.cuda.get_device_name(0)
    log.info(f"Device: {dev}, d_in={D_IN}")
    log.info(f"Probes: {PROBES}")

    rows = []
    for hp in HP_RATIOS:
        for T, d_out in PROBES:
            log.info(f"--- hp={hp}  T={T}  d_out={d_out} ---")
            # Baseline: R44 auto.
            us_auto = _time_case(None, None, T, d_out, D_IN, hp)
            best_us = None
            best_combo = None
            for kbm, kbn in COMBOS:
                us = _time_case(kbm, kbn, T, d_out, D_IN, hp)
                key = f"kBm={kbm} kBn={kbn}"
                if best_us is None or us < best_us:
                    best_us = us
                    best_combo = key
                spd = us_auto / us if us > 0 else float("nan")
                log.info(f"  {key:<20}  {us:6.2f}us   (auto/this = {spd:.3f}x)")
                rows.append({
                    "hp_ratio": hp, "T": T, "d_out": d_out, "d_in": D_IN,
                    "kBm": kbm, "kBn": kbn, "us": us, "us_auto": us_auto,
                })
            auto_vs_best = us_auto / best_us if best_us > 0 else float("nan")
            log.info(f"  ==> auto={us_auto:6.2f}us  best={best_us:6.2f}us "
                     f"[{best_combo}]  auto/best={auto_vs_best:.3f}x")

    os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)
    os.environ.pop("HKUST_V9_FUSED_FORCE_KBN", None)

    out_json = log_root / f"bench_r45_t96_probe_{ts}.json"
    with out_json.open("w") as f:
        json.dump({"device": dev, "d_in": D_IN, "rows": rows}, f, indent=2)
    log.info(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
