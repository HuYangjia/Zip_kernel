"""Large-scale sweep: V9 vs cuBLAS FP16 across (shape, batch, hp_ratio).

Goals
-----
1. Find the regime where V9 starts to beat cuBLAS FP16.
2. Per-call breakdown of the V9 pipeline:
   (1) activation quant  (2) dense u4 x s4 GEMM
   (3) sparse s4 x s4    (4) combine + transpose
   so we can see which stage dominates and is worth optimizing first.

Outputs (written next to this file under ./results/):
  - sweep_{ts}.csv   : full table, rows ordered identically to the plot order
  - sweep_{ts}.md    : human-readable summary
  - sweep_{ts}.log   : DEBUG-level full log

All log strings are in English.
"""

from __future__ import annotations

import csv
import datetime as _dt
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import torch

# ---------------------------------------------------------------------------
# absolute-path bootstrap (no CWD dependence)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent.parent           # /root
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJ_ROOT))

from kernel.triton_kernel.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton_kernel.sparse_s4s4_gemm import sparse_gemm_s4_s4  # noqa: E402
from kernel.triton_kernel.benchmarks._bench_util import time_ms as _time_ms  # noqa: E402


# ---------------------------------------------------------------------------
# logging: console INFO + file DEBUG
# ---------------------------------------------------------------------------
def _setup_logger(tag: str) -> logging.Logger:
    log_path = RESULTS_DIR / f"sweep_{tag}.log"
    logger = logging.getLogger(f"sweep_v9_{tag}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.debug("log file: %s", log_path)
    return logger


# ---------------------------------------------------------------------------
# sweep grid
# ---------------------------------------------------------------------------
SHAPES: List[Tuple[int, int]] = [
    (4096, 4096),       # Llama-7B attn proj
    (11008, 4096),      # Llama-7B gate/up
    (4096, 11008),      # Llama-7B down
    (14336, 4096),      # Llama-2-13B / Qwen2 ffn up
    (4096, 14336),      # Llama-2-13B / Qwen2 ffn down
    (8192, 8192),       # big square
    (28672, 4096),      # Llama-3-70B ffn up (tile)
]
BATCH_SEQS: List[int] = [1, 16, 64, 512, 2048, 8192]
HP_RATIOS: List[float] = [0.0, 0.02, 0.05, 0.10]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build_pack(d_out: int, d_in: int, hp_ratio: float):
    """Build a V9 weight pack with a given hp_ratio (fraction of hp blocks)."""
    nrow = d_out // BROW
    ncol = d_in // BCOL
    torch.manual_seed(0)
    device = "cuda"

    Q_u4 = torch.randint(0, 16, (d_out, d_in), dtype=torch.int8, device=device)
    scale_u4 = (torch.rand(d_out, ncol, device=device) * 0.01 + 0.001).to(torch.float16)
    zero_u4 = torch.randint(0, 16, (d_out, ncol), device=device).to(torch.float16)

    if hp_ratio > 0.0:
        n_hp = max(1, int(nrow * ncol * hp_ratio))
        combined = torch.unique(
            torch.randint(0, nrow * ncol, (n_hp * 2,), device=device)
        )[:n_hp]
        brs = (combined // ncol).to(torch.int32)
        bcs = (combined % ncol).to(torch.int32)
        hp_indices = torch.stack([brs, bcs], dim=-1)
        Q_s8_blocks = torch.randint(
            -64, 64, (len(brs), BROW, BCOL), dtype=torch.int8, device=device
        )
        scale_s8 = (torch.rand(len(brs), BROW, device=device) * 0.005 + 0.001).to(
            torch.float16
        )
    else:
        hp_indices = torch.empty((0, 2), dtype=torch.int32, device=device)
        Q_s8_blocks = torch.empty((0, BROW, BCOL), dtype=torch.int8, device=device)
        scale_s8 = torch.empty((0, BROW), dtype=torch.float16, device=device)

    perm = torch.arange(d_in, dtype=torch.int32, device=device)

    return pack_v9_weights({
        "Q_u4_permuted": Q_u4, "scale_u4_raw": scale_u4, "zero_u4_raw": zero_u4,
        "Q_s8_blocks": Q_s8_blocks, "scale_s8_per_block": scale_s8,
        "hp_block_indices": hp_indices, "perm": perm,
    })





def _bench_v9_stages(W, X_2d, has_hp: bool):
    """Measure each stage of V9 in isolation (best we can without re-hacking v9_linear)."""
    T = X_2d.shape[0]
    d_out = W.d_out

    # warm up once (also materializes outputs, so subsequent allocators reuse)
    X_s4, scale_x, sum_X = quantize_activation_s4(X_2d, W.perm, bcol=BCOL)
    Y_low = dense_gemm_u4_s4(W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x)
    if has_hp:
        Y_high = sparse_gemm_s4_s4(
            W.W_high_blocks_packed, W.hp_row_offsets, W.hp_col_indices,
            X_s4, W.scale_u4, scale_x, d_out=d_out, d_in=W.d_in,
        )
    else:
        Y_high = None

    def s1():
        quantize_activation_s4(X_2d, W.perm, bcol=BCOL)

    def s2():
        dense_gemm_u4_s4(W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x)

    def s3():
        if has_hp:
            sparse_gemm_s4_s4(
                W.W_high_blocks_packed, W.hp_row_offsets, W.hp_col_indices,
                X_s4, W.scale_u4, scale_x, d_out=d_out, d_in=W.d_in,
            )

    def s4():
        if has_hp:
            out = Y_low + 16.0 * Y_high
        else:
            out = Y_low
        out.transpose(0, 1).contiguous()

    t1 = _time_ms(s1)
    t2 = _time_ms(s2)
    t3 = _time_ms(s3) if has_hp else 0.0
    t4 = _time_ms(s4)
    return t1, t2, t3, t4


def _bench_v9_total(W, X_2d, has_hp: bool) -> float:
    """End-to-end V9 call time (what a user observes)."""
    T = X_2d.shape[0]
    d_out = W.d_out
    d_in = W.d_in

    def run():
        X_s4, scale_x, sum_X = quantize_activation_s4(X_2d, W.perm, bcol=BCOL)
        Y_low = dense_gemm_u4_s4(
            W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x
        )
        if has_hp:
            Y_high = sparse_gemm_s4_s4(
                W.W_high_blocks_packed, W.hp_row_offsets, W.hp_col_indices,
                X_s4, W.scale_u4, scale_x, d_out=d_out, d_in=d_in,
            )
            Y = Y_low + 16.0 * Y_high
        else:
            Y = Y_low
        Y.transpose(0, 1).contiguous()

    return _time_ms(run)


def _bench_cublas_fp16(W_fp: torch.Tensor, X: torch.Tensor) -> float:
    def run():
        torch.matmul(X, W_fp.t())

    return _time_ms(run)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = _setup_logger(ts)
    log.info("V9 vs cuBLAS FP16 large-scale sweep")
    log.info(
        "GPU: %s  |  torch=%s  triton=%s",
        torch.cuda.get_device_name(0),
        torch.__version__,
        getattr(__import__("triton"), "__version__", "?"),
    )

    csv_path = RESULTS_DIR / f"sweep_{ts}.csv"
    md_path = RESULTS_DIR / f"sweep_{ts}.md"

    header = [
        "d_out", "d_in", "bs", "hp_ratio",
        "v9_total_ms", "fp16_ms",
        "stage1_quant_ms", "stage2_dense_ms", "stage3_sparse_ms", "stage4_combine_ms",
        "speedup_vs_fp16",
    ]
    log.info(
        "{:>6} {:>6} {:>5} {:>5} {:>10} {:>9} {:>10} {:>10} {:>10} {:>10} {:>9}".format(
            "d_out", "d_in", "bs", "hp", "v9(ms)", "fp16(ms)",
            "quant(ms)", "dense(ms)", "sparse(ms)", "comb(ms)", "speedup",
        )
    )

    rows = []
    for d_out, d_in in SHAPES:
        for hp_ratio in HP_RATIOS:
            try:
                W = _build_pack(d_out, d_in, hp_ratio)
            except Exception as e:
                log.error("build_pack failed at (%d,%d,hp=%.2f): %s",
                          d_out, d_in, hp_ratio, e)
                continue
            has_hp = W.n_hp_blocks > 0

            # pre-build fp16 reference weight once per shape.
            # We simulate "quantized kernel weight -> fp16 dequant" cheaply as a
            # random fp16 tensor of the same shape, since cuBLAS timing depends
            # only on the shape (not the values).
            W_fp = torch.randn(d_out, d_in, device="cuda", dtype=torch.float16)

            for bs in BATCH_SEQS:
                X = torch.randn(bs, d_in, device="cuda", dtype=torch.float16)
                X_2d = X.reshape(-1, d_in)

                try:
                    t_v9 = _bench_v9_total(W, X_2d, has_hp)
                    t1, t2, t3, t4 = _bench_v9_stages(W, X_2d, has_hp)
                except Exception as e:
                    log.warning("v9 failed @ (%d,%d,bs=%d,hp=%.2f): %s",
                                d_out, d_in, bs, hp_ratio, e)
                    t_v9 = float("nan")
                    t1 = t2 = t3 = t4 = float("nan")

                try:
                    t_fp = _bench_cublas_fp16(W_fp, X)
                except Exception as e:
                    log.warning("fp16 failed @ (%d,%d,bs=%d): %s", d_out, d_in, bs, e)
                    t_fp = float("nan")

                sp = (t_fp / t_v9) if (t_v9 and t_v9 > 0) else float("nan")
                row = [d_out, d_in, bs, hp_ratio, t_v9, t_fp,
                       t1, t2, t3, t4, sp]
                rows.append(row)
                log.info(
                    "{:>6} {:>6} {:>5} {:>5.2f} {:>10.4f} {:>9.4f} "
                    "{:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.2f}x".format(
                        d_out, d_in, bs, hp_ratio, t_v9, t_fp,
                        t1, t2, t3, t4, sp,
                    )
                )
            del W
            torch.cuda.empty_cache()

    # ---- CSV (order identical to log) ----
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    log.info("csv saved to %s", csv_path)

    # ---- MD summary ----
    with md_path.open("w") as f:
        f.write(f"# V9 vs cuBLAS FP16 sweep ({ts})\n\n")
        f.write(f"GPU: {torch.cuda.get_device_name(0)}  ·  "
                f"torch={torch.__version__}\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")

        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)

        for r in rows:
            f.write("| " + " | ".join(_fmt(v) for v in r) + " |\n")

        # highlight wins
        wins = [r for r in rows if isinstance(r[-1], float) and r[-1] >= 1.0]
        f.write(f"\n## V9 >= FP16 cases ({len(wins)} / {len(rows)})\n\n")
        if wins:
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "|".join(["---"] * len(header)) + "|\n")
            for r in sorted(wins, key=lambda x: -x[-1]):
                f.write("| " + " | ".join(_fmt(v) for v in r) + " |\n")
    log.info("md saved to %s", md_path)


if __name__ == "__main__":
    main()
