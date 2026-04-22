"""NVTX-annotated driver for Nsight Systems profiling.

Runs a few V9 linear calls with each pipeline stage wrapped in an NVTX
range, so nsys / ncu can attribute kernel time to a stage. Also runs a
cuBLAS FP16 matmul for side-by-side comparison.

Environment variables:
  ZIP_PROFILE_BS     : override batch*seq (default 1)
  ZIP_PROFILE_DOUT   : override d_out      (default 11008)
  ZIP_PROFILE_DIN    : override d_in       (default 4096)
  ZIP_PROFILE_HP     : override hp_ratio   (default 0.0)
  ZIP_PROFILE_ITERS  : profiled iterations (default 30)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx

HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from kernel.triton.activation_quant import quantize_activation_s4  # noqa: E402
from kernel.triton.dense_u4s4_gemm import dense_gemm_u4_s4  # noqa: E402
from kernel.triton.pack_utils import BCOL, BROW, pack_v9_weights  # noqa: E402
from kernel.triton.sparse_s4s4_gemm import sparse_gemm_s4_s4  # noqa: E402


# logging: env-tag based log file in ./results/
def _setup_logger(tag: str) -> logging.Logger:
    results = HERE / "results"
    results.mkdir(parents=True, exist_ok=True)
    log_path = results / f"nvtx_driver_{tag}.log"
    logger = logging.getLogger(f"nvtx_driver_{tag}")
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


def _build_pack(d_out: int, d_in: int, hp_ratio: float):
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


def run_one_v9_call(W, X_2d, has_hp: bool):
    """One V9 forward pass with NVTX ranges around each stage."""
    d_out, d_in = W.d_out, W.d_in

    with torch.cuda.nvtx.range("v9_total"):
        with torch.cuda.nvtx.range("stage1_act_quant"):
            X_s4, scale_x, sum_X = quantize_activation_s4(X_2d, W.perm, bcol=BCOL)

        with torch.cuda.nvtx.range("stage2_dense_u4s4"):
            Y_low = dense_gemm_u4_s4(
                W.W_low_packed, X_s4, W.scale_u4, W.zero_u4, sum_X, scale_x
            )

        if has_hp:
            with torch.cuda.nvtx.range("stage3_sparse_s4s4"):
                Y_high = sparse_gemm_s4_s4(
                    W.W_high_blocks_packed, W.hp_row_offsets, W.hp_col_indices,
                    X_s4, W.scale_u4, scale_x, d_out=d_out, d_in=d_in,
                )
            with torch.cuda.nvtx.range("stage4_combine_add"):
                Y_low.add_(Y_high, alpha=16.0)

        with torch.cuda.nvtx.range("stage4_transpose_contig"):
            Y_low.transpose(0, 1).contiguous()


def run_one_fp16(W_fp, X):
    with torch.cuda.nvtx.range("cublas_fp16_matmul"):
        torch.matmul(X, W_fp.t())


def main():
    assert torch.cuda.is_available(), "needs CUDA"

    bs = int(os.environ.get("ZIP_PROFILE_BS", "1"))
    d_out = int(os.environ.get("ZIP_PROFILE_DOUT", "11008"))
    d_in = int(os.environ.get("ZIP_PROFILE_DIN", "4096"))
    hp_ratio = float(os.environ.get("ZIP_PROFILE_HP", "0.0"))
    iters = int(os.environ.get("ZIP_PROFILE_ITERS", "30"))

    tag = f"bs{bs}_do{d_out}_di{d_in}_hp{int(hp_ratio*100)}"
    log = _setup_logger(tag)
    log.info(
        "NVTX driver | bs=%d d_out=%d d_in=%d hp=%.2f iters=%d",
        bs, d_out, d_in, hp_ratio, iters,
    )
    log.info("GPU: %s  torch=%s", torch.cuda.get_device_name(0), torch.__version__)

    W = _build_pack(d_out, d_in, hp_ratio)
    has_hp = W.n_hp_blocks > 0
    W_fp = torch.randn(d_out, d_in, device="cuda", dtype=torch.float16)
    X = torch.randn(bs, d_in, device="cuda", dtype=torch.float16)
    X_2d = X.reshape(-1, d_in)

    # warm-up (autotune compile + fp16 CUBLAS warm-up).  These warm-up calls
    # are NOT wrapped in v9_total so nsys will not attribute their time to
    # the measured window, as long as the nsys --capture-range starts after.
    log.info("warm-up start")
    with torch.cuda.nvtx.range("WARMUP"):
        for _ in range(5):
            run_one_v9_call(W, X_2d, has_hp)
            run_one_fp16(W_fp, X)
    torch.cuda.synchronize()
    log.info("warm-up done")

    # Profiled region: explicit CUDA profiler start/stop so nsys can
    # use --capture-range=cudaProfilerApi to only save this window.
    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.nvtx.range("MEASURED"):
        for i in range(iters):
            with torch.cuda.nvtx.range(f"iter_{i}"):
                run_one_v9_call(W, X_2d, has_hp)
                run_one_fp16(W_fp, X)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    log.info("profiled region done (%d iters)", iters)


if __name__ == "__main__":
    main()
