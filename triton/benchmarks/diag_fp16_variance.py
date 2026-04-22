"""Diagnostic: quantify cuBLAS FP16 matmul variance at small batch.

Repeats each (bs) measurement 5 times to see:
  1. Run-to-run noise at bs=1 (launch overhead / clock jitter)
  2. Whether bs=1 is truly slower than bs=16 after tight warm-up
  3. Which cuBLAS kernel is picked at each bs (printed via torch profiler
     hook is avoided; instead we just eyeball the time pattern --
     gemv-like kernels show a clear time plateau at bs<=8)

Run with:
    python diag_fp16_variance.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "results" / "diag_fp16_variance.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("diag_fp16")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s",
                        datefmt="%H:%M:%S")
ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
fh = logging.FileHandler(LOG_PATH, mode="w"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
logger.addHandler(ch); logger.addHandler(fh)


def bench_fp16(bs: int, d_out: int, d_in: int,
               n_warmup: int = 50, n_iter: int = 200) -> float:
    W = torch.randn(d_out, d_in, dtype=torch.float16, device="cuda")
    X = torch.randn(bs, d_in, dtype=torch.float16, device="cuda")
    for _ in range(n_warmup):
        torch.matmul(X, W.t())
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        torch.matmul(X, W.t())
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter


def main():
    assert torch.cuda.is_available()
    logger.info("GPU: %s  torch=%s", torch.cuda.get_device_name(0), torch.__version__)
    d_out, d_in = 4096, 4096

    header = f"{'bs':>4} | " + " | ".join(f"run{i}(ms)" for i in range(1, 6)) + \
             " | " + f"{'min':>8} {'mean':>8} {'max':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    csv_rows = [["bs", "run1", "run2", "run3", "run4", "run5", "min", "mean", "max"]]
    for bs in [1, 2, 4, 8, 16, 32, 64, 128]:
        ts = [bench_fp16(bs, d_out, d_in) for _ in range(5)]
        mn, mx, avg = min(ts), max(ts), sum(ts) / len(ts)
        logger.info(
            "%4d | %s | %8.4f %8.4f %8.4f",
            bs, " | ".join(f"{t:>8.4f}" for t in ts), mn, avg, mx,
        )
        csv_rows.append([bs, *[f"{t:.6f}" for t in ts],
                         f"{mn:.6f}", f"{avg:.6f}", f"{mx:.6f}"])

    # Keep CSV ordering identical to the log table above.
    import csv as _csv
    csv_path = HERE / "results" / "diag_fp16_variance.csv"
    with csv_path.open("w", newline="") as f:
        _csv.writer(f).writerows(csv_rows)
    logger.info("csv written: %s", csv_path)


if __name__ == "__main__":
    main()
