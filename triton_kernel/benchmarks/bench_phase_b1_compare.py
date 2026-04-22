"""Phase B-1 micro-benchmark: prefill dense kernel vs cuBLAS FP16.

Purpose
-------
Sweep data (sweep_20260422_154306) showed that the prefill regime's
dominant bottleneck is the dense UINT4 x SINT4 GEMM: it accounts for
83-91% of v9_total, and our kernel is at 1.27x of cuBLAS FP16 at
median with HBM bandwidth utilisation of only 1.6-7%, i.e. clearly
Tensor-Core-occupancy limited.

Phase B-1 adds 5 new autotune configs targeting that regime. This
script directly measures the dense kernel (not the full pipeline) so
we can see the pure kernel-level delta without noise from the other
stages.

Measurement protocol
--------------------
Per the project's GPU micro-bench rule (kernel memory bmmiahpl):
  * >=50 warm-up iterations (GPU boost clock + Triton autotune cached)
  * >=100 iterations per measurement window
  * >=3 windows, report min-of-means
These are baked into benchmarks._bench_util.time_ms.

Output
------
- CSV `results/phase_b1_dense_{ts}.csv`
- Human MD  `results/phase_b1_dense_{ts}.md`
- Console log of each case
"""

from __future__ import annotations

import csv
import datetime as _dt
import sys
from pathlib import Path
from typing import List, Tuple

import torch

HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent.parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJ_ROOT))

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton_kernel.benchmarks._bench_util import time_ms  # noqa: E402


# Prefill-focused grid only (small bs is decode-regime, not our target here).
SHAPES: List[Tuple[int, int]] = [
    (4096, 4096),
    (11008, 4096),
    (4096, 11008),
    (14336, 4096),
    (4096, 14336),
    (8192, 8192),
    (28672, 4096),
]
PREFILL_BS: List[int] = [256, 512, 2048, 8192]


def _build_container(d_out: int, d_in: int):
    """Build a V9WeightContainer with hp_ratio=0 (dense only; sparse path is not
    the target of Phase B-1)."""
    nrow = d_out // BROW
    ncol = d_in // BCOL
    torch.manual_seed(0)
    device = "cuda"
    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device=device)
    scale_u4 = (torch.rand(d_out, ncol, device=device) * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, ncol), device=device).to(torch.float16)
    hp_indices = torch.empty((0, 2), dtype=torch.int32, device=device)
    Q_s8_blocks = torch.empty((0, BROW, BCOL), dtype=torch.int8, device=device)
    scale_s8 = torch.empty((0, BROW), dtype=torch.float16, device=device)
    perm = torch.arange(d_in, dtype=torch.int32, device=device)
    return pack_v9_weights({
        "Q_u4_permuted": Q_u4,
        "scale_u4_raw": scale_u4,
        "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8_blocks,
        "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices,
        "perm": perm,
    })


def _bench_dense_only(W, X_2d):
    """Time the dense kernel in isolation. Activation quant output is
    pre-computed once (it's not what we are measuring)."""
    X_s4, scale_x, sum_X = quantize_activation_s4(X_2d, W.perm, bcol=BCOL)
    torch.cuda.synchronize()

    def _once():
        return dense_gemm_u4_s4(
            W.W_low_packed, X_s4,
            W.scale_u4, W.zero_u4,
            sum_X, scale_x,
        )
    return time_ms(_once)


def _bench_fp16(d_out: int, d_in: int, bs: int) -> float:
    """cuBLAS FP16 reference: Y = X @ W^T."""
    X = torch.randn(bs, d_in, device="cuda", dtype=torch.float16)
    W_fp = torch.randn(d_out, d_in, device="cuda", dtype=torch.float16)

    def _once():
        return torch.nn.functional.linear(X, W_fp)
    return time_ms(_once)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"phase_b1_dense_{ts}.csv"
    md_path = RESULTS_DIR / f"phase_b1_dense_{ts}.md"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    import triton as _tri
    print(f"triton={_tri.__version__}  torch={torch.__version__}")
    print(f"Writing -> {csv_path}\n")

    header = ["d_out", "d_in", "bs", "dense_ms", "fp16_ms", "ratio", "bw_gbps", "hbm_util_pct"]
    # RTX 4090 HBM peak ~1008 GB/s
    HBM_PEAK = 1008.0
    rows = []
    print("{:>6} {:>6} {:>5} | {:>9} {:>9} {:>7} | {:>8} {:>7}".format(
        "d_out", "d_in", "bs", "dense(ms)", "fp16(ms)", "ratio", "GB/s", "%peak",
    ))

    for d_out, d_in in SHAPES:
        W = _build_container(d_out, d_in)
        for bs in PREFILL_BS:
            X = torch.randn(bs, d_in, device="cuda", dtype=torch.float16)
            X_2d = X.reshape(-1, d_in)
            try:
                t_dense = _bench_dense_only(W, X_2d)
            except Exception as e:
                print(f"dense failed @ ({d_out},{d_in},bs={bs}): {e}")
                t_dense = float("nan")
            t_fp16 = _bench_fp16(d_out, d_in, bs)

            # Effective weight bytes moved: d_out * d_in / 2 bytes of 4-bit weight
            # (scales / zeros / activations together are <3% and we ignore them
            # for this simple bandwidth metric).
            w_bytes = d_out * d_in // 2
            bw_gbps = (w_bytes / (t_dense * 1e-3)) / 1e9 if t_dense > 0 else 0.0
            hbm_util = 100.0 * bw_gbps / HBM_PEAK
            ratio = t_dense / t_fp16 if t_fp16 > 0 else float("inf")

            rows.append([d_out, d_in, bs, t_dense, t_fp16, ratio, bw_gbps, hbm_util])
            print("{:>6} {:>6} {:>5} | {:>9.4f} {:>9.4f} {:>6.2f}x | {:>8.1f} {:>6.1f}%".format(
                d_out, d_in, bs, t_dense, t_fp16, ratio, bw_gbps, hbm_util,
            ))

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # Build markdown summary
    md = ["# Phase B-1 dense kernel benchmark (prefill-focused)\n",
          f"- GPU: {torch.cuda.get_device_name(0)}",
          f"- torch {torch.__version__} / triton {_tri.__version__}",
          f"- Timestamp: {ts}",
          f"- Protocol: 50-warmup + 100-iter x 3-window min-of-means",
          "",
          "| d_out | d_in | bs | dense(ms) | fp16(ms) | ratio | GB/s | %HBM |",
          "|---|---|---|---|---|---|---|---|",
          ]
    for r in rows:
        md.append("| {} | {} | {} | {:.4f} | {:.4f} | **{:.2f}x** | {:.1f} | {:.1f}% |".format(*r))
    # Aggregates
    valid = [r for r in rows if r[5] == r[5]]  # drop NaN
    if valid:
        ratios = [r[5] for r in valid]
        md += ["", "## Aggregate",
               f"- median dense/fp16 ratio: {sorted(ratios)[len(ratios)//2]:.2f}x",
               f"- best  dense/fp16 ratio: {min(ratios):.2f}x",
               f"- worst dense/fp16 ratio: {max(ratios):.2f}x",
               ""]
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
