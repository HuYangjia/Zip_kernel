"""Microbench for activation_quant_s4 kernel (stage-1 of V9 pipeline).

Why a separate file?
--------------------
`sweep_v9.py` measures the full 4-stage v9_linear and stage-1 quant cost
is only one column among many.  When iterating on act_quant specifically
we want:

  1. A tight feedback loop (~5s per run, not ~5min like sweep_v9).
  2. Per-shape breakdown with the exact HBM-bytes / effective-BW metric
     so we can see how close we are to the memory ceiling.
  3. A JSON/CSV artefact we can diff across optimisation iterations.

Measurement methodology
-----------------------
Follows the project-wide microbench convention:

  * >= 50 warm-up calls (RTX 4090 boost clock lock + Triton autotune settle)
  * >= 100 iterations per window
  * >= 3 windows, report min-of-means
  * All timings via `benchmarks._bench_util.time_ms`

Usage
-----
    python -m kernel.triton_kernel.benchmarks.bench_act_quant
    python -m kernel.triton_kernel.benchmarks.bench_act_quant --tag baseline
    python -m kernel.triton_kernel.benchmarks.bench_act_quant --tag try1 --compare baseline
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

import torch

from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.pack_utils import BCOL
from kernel.triton_kernel.benchmarks._bench_util import time_ms

# ---------------------------------------------------------------------------
# Shape matrix — keep it small (16 rows) so each full run stays < 60s.
# Covers:
#   decode-1, decode-16, small-64, mid-512, prefill-2K, prefill-8K   x
#   d_in in {4096, 11008, 14336}
# ---------------------------------------------------------------------------
_SHAPES = [
    # (T, d_in)   notes
    (1,     4096),   # pure launch-bound decode
    (1,    11008),   # Llama-2 FFN down-proj decode
    (1,    14336),   # Llama-3 FFN up-proj decode
    (16,    4096),
    (16,   11008),
    (16,   14336),
    (64,    4096),
    (64,   11008),
    (512,   4096),
    (512,  11008),
    (512,  14336),
    (2048,  4096),
    (2048, 11008),
    (8192,  4096),
    (8192, 11008),
    (8192, 14336),
]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _bytes_read(T: int, D: int) -> int:
    """Lower-bound HBM read volume for a 2-pass quant impl.

    Two full passes over X (fp16):  2 * T * D * 2 bytes
    perm lookup (int32) once per pass, negligible for D >> T.

    This is the reference for `effective_bw = bytes_read / time_s`.
    If the kernel does >2x we're wasting bandwidth (uncoalesced gather,
    L2 miss on re-read, etc).
    """
    return 2 * T * D * 2


def _bytes_write(T: int, D: int) -> int:
    """HBM write volume."""
    # X_s4: T * D/2 bytes (int8 packed)
    # scale_x: T * 2 bytes (fp16)
    # sum_X: T * (D/128) * 4 bytes (int32, per-group)
    n_groups = D // BCOL
    return T * (D // 2) + T * 2 + T * n_groups * 4


def bench_one(T: int, D: int, device: str = "cuda") -> dict:
    """Return dict with timing + derived metrics for one shape."""
    assert D % BCOL == 0, f"D={D} must be divisible by BCOL={BCOL}"
    torch.manual_seed(0)
    X = torch.randn(T, D, dtype=torch.float16, device=device) * 0.5
    perm = torch.randperm(D, dtype=torch.int32, device=device)

    def run():
        quantize_activation_s4(X, perm, bcol=BCOL)

    t_ms = time_ms(run, n_warmup=80, n_iter=200, n_repeat=5)
    t_us = t_ms * 1000.0
    b_r = _bytes_read(T, D)
    b_w = _bytes_write(T, D)
    bw_gb_s = (b_r + b_w) / (t_ms * 1e-3) / 1e9
    # 4090 HBM2e ~ 1008 GB/s.  Practical ceiling for a well-coalesced kernel ~850 GB/s.
    bw_pct = 100.0 * bw_gb_s / 1008.0
    return {
        "T": T,
        "D": D,
        "n_groups": D // BCOL,
        "time_us": t_us,
        "bytes_read": b_r,
        "bytes_write": b_w,
        "bw_gb_s": bw_gb_s,
        "bw_pct_of_peak": bw_pct,
    }


def run_full(tag: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU: {device_name}")
    print(f"Tag: {tag}")
    print(f"{'T':>6} {'D':>6} {'us':>8} {'GB/s':>8} {'%peak':>7}")
    out = {
        "tag": tag,
        "gpu": device_name,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "shapes": [],
    }
    for T, D in _SHAPES:
        row = bench_one(T, D)
        out["shapes"].append(row)
        print(f"{row['T']:>6} {row['D']:>6} {row['time_us']:>8.2f} "
              f"{row['bw_gb_s']:>8.1f} {row['bw_pct_of_peak']:>6.1f}%")
    return out


def save(results: dict, tag: str) -> Path:
    """Save both JSON (for diffing) and CSV (for spreadsheet)."""
    stem = f"act_quant_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{tag}"
    json_p = RESULTS_DIR / f"{stem}.json"
    csv_p = RESULTS_DIR / f"{stem}.csv"
    json_p.write_text(json.dumps(results, indent=2))
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["T", "D", "n_groups", "time_us",
                                          "bytes_read", "bytes_write",
                                          "bw_gb_s", "bw_pct_of_peak"])
        w.writeheader()
        for row in results["shapes"]:
            w.writerow(row)
    print(f"Saved:\n  {json_p}\n  {csv_p}")
    return json_p


def _latest_with_tag(tag: str) -> Path | None:
    cands = sorted(RESULTS_DIR.glob(f"act_quant_*_{tag}.json"))
    return cands[-1] if cands else None


def compare(curr: dict, base_tag: str) -> None:
    """Print a before/after table given a baseline tag name."""
    base_p = _latest_with_tag(base_tag)
    if base_p is None:
        print(f"(no baseline with tag={base_tag})")
        return
    base = json.loads(base_p.read_text())
    base_map = {(r["T"], r["D"]): r for r in base["shapes"]}
    print()
    print(f"Δ vs baseline ({base_p.name}):")
    print(f"{'T':>6} {'D':>6} {'base us':>10} {'curr us':>10} "
          f"{'Δ us':>8} {'Δ %':>8} {'curr GB/s':>10}")
    improved = regressed = 0
    for row in curr["shapes"]:
        k = (row["T"], row["D"])
        if k not in base_map:
            continue
        b = base_map[k]
        d = row["time_us"] - b["time_us"]
        dp = 100 * d / b["time_us"]
        flag = "↓" if dp < -2 else ("↑" if dp > 2 else " ")
        if dp < -2:
            improved += 1
        elif dp > 2:
            regressed += 1
        print(f"{row['T']:>6} {row['D']:>6} {b['time_us']:>10.2f} "
              f"{row['time_us']:>10.2f} {d:>+8.2f} {dp:>+7.1f}% {flag} "
              f"{row['bw_gb_s']:>10.1f}")
    print(f"\nSummary: {improved} improved, {regressed} regressed, "
          f"{len(curr['shapes']) - improved - regressed} flat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="run",
                    help="Label for this measurement set (e.g. 'baseline', 'try1')")
    ap.add_argument("--compare", type=str, default=None,
                    help="Baseline tag to diff against")
    args = ap.parse_args()
    results = run_full(args.tag)
    save(results, args.tag)
    if args.compare:
        compare(results, args.compare)


if __name__ == "__main__":
    main()
