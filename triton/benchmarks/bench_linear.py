"""End-to-end benchmark: V9 v.s. cuBLAS FP16 v.s. fakequant PyTorch baseline.

Covers a 3x3 grid of Qwen3-typical layer shapes and batch*seq sizes.
Outputs are written to `results/bench_{timestamp}.csv` and `.md`.
All log strings are English.
"""

from __future__ import annotations

import csv
import datetime as _dt
import sys
from pathlib import Path
from typing import List, Tuple

import torch

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent.parent.parent))

from kernel.triton.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton.v9_linear import (  # noqa: E402
    reconstruct_w_fakequant_fp16,
    v9_linear_fakequant,
    v9_linear_forward,
)


SHAPES: List[Tuple[int, int]] = [(4096, 4096), (11008, 4096), (4096, 11008)]
BATCH_SEQS: List[int] = [1, 16, 256]


def _build_pack(d_out: int, d_in: int, hp_ratio: float = 0.05):
    nrow = d_out // BROW
    ncol = d_in // BCOL
    torch.manual_seed(0)
    device = "cuda"

    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device=device)
    scale_u4 = (torch.rand(d_out, ncol, device=device) * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, ncol), device=device).to(torch.float16)

    n_hp = max(1, int(nrow * ncol * hp_ratio))
    combined = torch.unique(torch.randint(0, nrow * ncol, (n_hp * 2,), device=device))[:n_hp]
    brs = (combined // ncol).to(torch.int32)
    bcs = (combined % ncol).to(torch.int32)
    hp_indices = torch.stack([brs, bcs], dim=-1)

    Q_s8_blocks = torch.randint(-64, 64, (len(brs), BROW, BCOL), dtype=torch.int8, device=device)
    scale_s8 = (torch.rand(len(brs), BROW, device=device) * 0.005 + 0.001).to(torch.float16)
    perm = torch.arange(d_in, dtype=torch.int32, device=device)

    return pack_v9_weights({
        "Q_u4_permuted": Q_u4, "scale_u4_raw": scale_u4, "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8_blocks, "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices, "perm": perm,
    })


def _time_ms(fn, n_warmup: int = 10, n_iter: int = 30) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"bench_{timestamp}.csv"
    md_path = RESULTS_DIR / f"bench_{timestamp}.md"

    rows = []
    header = ["d_out", "d_in", "bs", "v9_ms", "cublas_fp16_ms", "fakequant_ms",
              "speedup_vs_fp16", "speedup_vs_fakequant"]

    print("{:>8} {:>8} {:>6} {:>12} {:>14} {:>14} {:>10} {:>10}".format(*header))

    for d_out, d_in in SHAPES:
        W = _build_pack(d_out, d_in)
        W_fp = reconstruct_w_fakequant_fp16(W)

        for bs in BATCH_SEQS:
            X = torch.randn(bs, d_in, device="cuda", dtype=torch.float16)

            def v9():
                v9_linear_forward(X, W)

            def fp16():
                torch.matmul(X, W_fp.t())

            def fake():
                v9_linear_fakequant(X, W)

            try:
                t_v9 = _time_ms(v9)
            except Exception as e:
                t_v9 = float("nan")
                print(f"[warn] v9 failed on ({d_out},{d_in},{bs}): {e}")

            try:
                t_fp = _time_ms(fp16)
            except Exception as e:
                t_fp = float("nan")
                print(f"[warn] fp16 failed: {e}")

            try:
                # fakequant is slow; use smaller iteration count
                t_fake = _time_ms(fake, n_warmup=2, n_iter=5)
            except Exception as e:
                t_fake = float("nan")
                print(f"[warn] fakequant failed: {e}")

            sp_fp = (t_fp / t_v9) if (t_v9 and t_v9 > 0) else float("nan")
            sp_fake = (t_fake / t_v9) if (t_v9 and t_v9 > 0) else float("nan")

            rows.append([d_out, d_in, bs, t_v9, t_fp, t_fake, sp_fp, sp_fake])
            print("{:>8} {:>8} {:>6} {:>12.4f} {:>14.4f} {:>14.4f} {:>10.2f}x {:>10.2f}x".format(
                d_out, d_in, bs, t_v9, t_fp, t_fake, sp_fp, sp_fake))

    # ---- save CSV
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # ---- save Markdown
    with md_path.open("w") as f:
        f.write(f"# V9 End-to-End Linear Benchmark\n\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            def _fmt(v):
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            f.write("| " + " | ".join(_fmt(v) for v in r) + " |\n")

    print(f"\nResults saved to {csv_path}")
    print(f"Results saved to {md_path}")


if __name__ == "__main__":
    main()
