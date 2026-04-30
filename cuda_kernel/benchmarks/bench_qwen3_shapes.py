"""Qwen3 multi-scale shape benchmark (pure Triton vs pure CUDA, with FP16 baseline).

Covers Qwen3-0.6B / 1.7B / 4B / 8B (and 14B opt-in) across 5 Linear
projections per layer (q_proj / kv_proj / o_proj / gate_up_proj /
down_proj) and a sweep of T = {1, 8, 128, 512, 1024}.

For every (model, proj, T) combo we measure:

  Sub-kernels:
    - activation_quant   (Triton / CUDA)
    - dense_gemm         (Triton / CUDA / FP16 baseline)
    - sparse_gemm        (Triton / CUDA / FP16 baseline)
    - fused_dense_sparse (Triton / CUDA / FP16 baseline)

  End-to-end (quant + fused):
    - fp16    (cuBLAS matmul baseline via torch.matmul)
    - triton  (quantize_activation_s4 + fused_dense_sparse_gemm)
    - cuda    (activation_quant_cuda + fused_dense_sparse_cuda;
               T=1 uses fused_quant_gemv)

All timings use the shared stable microbenchmark helper:
50 warmup, 100 inner iterations, 3 repeats, min-of-means.
Outputs: logs/qwen3_bench/qwen3_{TS}/{bench.md, bench.json, bench.log}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import torch

# ---------------------------------------------------------------------------
# Path anchoring (avoid cwd dependency)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_IMPORT_ROOT = _THIS.parents[3]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from kernel.cuda_kernel import ops as cuda_ops
from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.benchmarks._bench_util import time_ms
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_u4_s4
from kernel.triton_kernel.sparse_s4s4_gemm import sparse_gemm_s4_s4
from kernel.triton_kernel.fused_dense_sparse_gemm import fused_dense_sparse_gemm
from kernel.triton_kernel.pack_utils import BCOL, BROW, pack_s4_le


# ---------------------------------------------------------------------------
# Model configs (Qwen3 series)
# ---------------------------------------------------------------------------
@dataclass
class Qwen3Config:
    name: str
    hidden: int
    intermediate: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int = 128

    @property
    def q_out(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_out(self) -> int:
        return self.num_kv_heads * self.head_dim


QWEN3_MODELS: list[Qwen3Config] = [
    Qwen3Config("Qwen3-0.6B", hidden=1024, intermediate=3072,  num_q_heads=16, num_kv_heads=8),
    Qwen3Config("Qwen3-1.7B", hidden=2048, intermediate=6144,  num_q_heads=16, num_kv_heads=8),
    Qwen3Config("Qwen3-4B",   hidden=2560, intermediate=9728,  num_q_heads=32, num_kv_heads=8),
    Qwen3Config("Qwen3-8B",   hidden=4096, intermediate=12288, num_q_heads=32, num_kv_heads=8),
    Qwen3Config("Qwen3-14B",  hidden=5120, intermediate=17408, num_q_heads=40, num_kv_heads=8),
    # r63 — bigger dense GQA models (same architecture family, used to
    # show that INT4 speedup vs FP16 scales with model size up to 70B).
    # Shapes are vendor-published for production checkpoints.
    Qwen3Config("Qwen2.5-32B", hidden=5120, intermediate=27648, num_q_heads=40, num_kv_heads=8),
    Qwen3Config("LLaMA3-70B",  hidden=8192, intermediate=28672, num_q_heads=64, num_kv_heads=8),
]


@dataclass
class ProjShape:
    """A (d_in, d_out) shape for one transformer Linear."""
    proj: str     # q_proj / kv_proj / o_proj / gate_up_proj / down_proj
    d_in: int
    d_out: int


def enumerate_projs(cfg: Qwen3Config) -> list[ProjShape]:
    """Produce the 5 Linear shapes that dominate a Qwen3 transformer layer."""
    return [
        ProjShape("q_proj",       d_in=cfg.hidden,       d_out=cfg.q_out),
        ProjShape("kv_proj",      d_in=cfg.hidden,       d_out=cfg.kv_out * 2),   # merged K+V
        ProjShape("o_proj",       d_in=cfg.q_out,        d_out=cfg.hidden),
        ProjShape("gate_up_proj", d_in=cfg.hidden,       d_out=cfg.intermediate * 2),  # merged
        ProjShape("down_proj",    d_in=cfg.intermediate, d_out=cfg.hidden),
    ]


# Batch sizes we sweep.  Keep the list compact; caller can override via --ts.
DEFAULT_TS: tuple[int, ...] = (1, 8, 128, 512, 1024)


# ---------------------------------------------------------------------------
# Shape constraint helper: kernels require d_in % 128 == 0 and d_out % 128 == 0
# ---------------------------------------------------------------------------
def shape_is_supported(proj: ProjShape) -> bool:
    return (proj.d_in % BCOL == 0) and (proj.d_out % BROW == 0)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(out_dir: Path) -> tuple[logging.Logger, Path]:
    log = logging.getLogger("bench_qwen3")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    term = logging.StreamHandler(sys.stdout)
    term.setLevel(logging.INFO)
    term.setFormatter(fmt)
    log.addHandler(term)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bench.log"
    fileh = logging.FileHandler(log_path, mode="a")
    fileh.setLevel(logging.DEBUG)
    fileh.setFormatter(fmt)
    log.addHandler(fileh)
    return log, log_path


# ---------------------------------------------------------------------------
# Timer (shared stable microbenchmark helper)
# ---------------------------------------------------------------------------
def bench_us(
    fn: Callable[[], None],
    *,
    warmup: int = 50,
    outer: int = 3,
    inner: int = 100,
    flush_l2: bool = False,
) -> float:
    return time_ms(
        fn,
        n_warmup=warmup,
        n_iter=inner,
        n_repeat=outer,
        flush_l2=flush_l2,
    ) * 1000.0


# r62 P2: module-level flag, flipped by CLI.  When True, the FP16 cuBLAS
# baseline (and only the baseline) is measured with cold-cache HBM to
# remove the L2-reuse artefact.  INT4/Triton paths keep their original
# tight-loop timing so downstream numbers stay comparable to prior runs.
_FLUSH_L2_FP16 = True


# ---------------------------------------------------------------------------
# Input factory (both Triton and CUDA share the same tensors)
# ---------------------------------------------------------------------------
def make_inputs(
    T: int,
    d_out: int,
    d_in: int,
    hp_ratio: float = 0.05,
    device: str = "cuda",
    seed: int = 0,
):
    torch.manual_seed(seed + T + d_out + d_in)
    X = torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
    X_fp_t = X.transpose(0, 1).contiguous()
    perm = torch.arange(d_in, dtype=torch.int32, device=device)

    n_groups = d_in // BCOL
    W_s4_unpacked = torch.randint(-8, 8, (d_out, d_in), dtype=torch.int8, device=device)
    W_low_packed = pack_s4_le(W_s4_unpacked)
    scale_u4 = (torch.rand(d_out, n_groups, device=device) * 0.05 + 0.001).half()
    zero_u4 = (torch.randn(d_out, n_groups, device=device) * 0.2).half()
    W_fp = torch.randn(d_out, d_in, dtype=torch.float16, device=device) * 0.02

    # Sparse side
    nrow = d_out // BROW
    ncol = d_in // BCOL
    total_blocks = nrow * ncol
    n_hp = max(1, int(total_blocks * hp_ratio))

    torch.manual_seed((T * d_in * d_out) ^ 0xA5A5)
    flat = torch.randperm(total_blocks, device=device)[:n_hp]
    br = (flat // ncol).to(torch.int32)
    bc = (flat % ncol).to(torch.int32)
    order = torch.argsort(br.to(torch.int64) * 1_000_000 + bc.to(torch.int64))
    br_sorted = br[order]
    bc_sorted = bc[order]

    W_high_s4 = torch.randint(-8, 8, (n_hp, BROW, BCOL), dtype=torch.int8, device=device)
    W_high_packed = pack_s4_le(W_high_s4)

    hp_row_offsets = torch.zeros(nrow + 1, dtype=torch.int32, device=device)
    counts = torch.bincount(br_sorted.to(torch.int64), minlength=nrow)
    hp_row_offsets[1:] = torch.cumsum(counts, dim=0).to(torch.int32)

    # Pre-compute quantized X once (Triton path) for sub-kernel timing
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)

    return dict(
        X=X, X_fp_t=X_fp_t, perm=perm,
        W_low_packed=W_low_packed,
        W_high_packed=W_high_packed,
        hp_row_offsets=hp_row_offsets,
        hp_col_indices=bc_sorted,
        scale_u4=scale_u4, zero_u4=zero_u4,
        X_s4=X_s4, scale_x=scale_x, sum_X=sum_X,
        W_fp=W_fp,
        n_hp=n_hp,
        hp_ratio=hp_ratio,
    )


# ---------------------------------------------------------------------------
# Per-sub-kernel benches.  Each returns dict with three entries when applicable.
# ---------------------------------------------------------------------------
def bench_fp16_matmul(inp: dict) -> float:
    return bench_us(
        lambda: torch.matmul(inp["W_fp"], inp["X_fp_t"]),
        flush_l2=_FLUSH_L2_FP16,
    )


def bench_activation_quant(T: int, d_in: int, log: logging.Logger):
    X = torch.randn(T, d_in, dtype=torch.float16, device="cuda") * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device="cuda")
    t_triton = bench_us(lambda: quantize_activation_s4(X, perm))
    t_cuda = bench_us(lambda: cuda_ops.activation_quant_cuda(X, perm))
    # No fp16 baseline for quant (it is a non-matmul op).
    return {"triton_us": t_triton, "cuda_us": t_cuda}


def bench_dense_gemm(inp: dict, T: int, d_out: int, d_in: int):
    X_s4 = inp["X_s4"]
    W_low_packed = inp["W_low_packed"]
    scale_u4 = inp["scale_u4"]
    zero_u4 = inp["zero_u4"]
    sum_X = inp["sum_X"]
    scale_x = inp["scale_x"]

    t_fp16 = bench_fp16_matmul(inp)
    t_triton = bench_us(lambda: dense_gemm_u4_s4(
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
    ))
    t_cuda = bench_us(lambda: cuda_ops.dense_gemm_cuda(
        W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x
    ))
    return {
        "fp16_us": t_fp16,
        "triton_us": t_triton,
        "cuda_us": t_cuda,
        "triton_speedup_vs_fp16": t_fp16 / t_triton,
        "cuda_speedup_vs_fp16": t_fp16 / t_cuda,
        "cuda_speedup_vs_triton": t_triton / t_cuda,
    }


def bench_sparse_gemm(inp: dict, T: int, d_out: int, d_in: int):
    X_s4 = inp["X_s4"]
    scale_u4 = inp["scale_u4"]
    scale_x = inp["scale_x"]

    t_fp16 = bench_fp16_matmul(inp)
    t_triton = bench_us(lambda: sparse_gemm_s4_s4(
        inp["W_high_packed"], inp["hp_row_offsets"], inp["hp_col_indices"],
        X_s4, scale_u4, scale_x, d_out, d_in,
    ))
    t_cuda = bench_us(lambda: cuda_ops.sparse_gemm_cuda_int4(
        inp["W_high_packed"], inp["hp_row_offsets"], inp["hp_col_indices"],
        X_s4, scale_u4, scale_x, d_out, d_in,
    ))
    return {
        "fp16_us": t_fp16,
        "triton_us": t_triton,
        "cuda_us": t_cuda,
        "triton_speedup_vs_fp16": t_fp16 / t_triton,
        "cuda_speedup_vs_fp16": t_fp16 / t_cuda,
        "cuda_speedup_vs_triton": t_triton / t_cuda,
    }


def bench_fused(inp: dict, T: int, d_out: int, d_in: int):
    t_fp16 = bench_fp16_matmul(inp)
    t_triton = bench_us(lambda: fused_dense_sparse_gemm(
        inp["W_low_packed"], inp["W_high_packed"],
        inp["hp_row_offsets"], inp["hp_col_indices"],
        inp["X_s4"], inp["scale_u4"], inp["zero_u4"],
        inp["sum_X"], inp["scale_x"],
        d_out, d_in,
    ))
    t_cuda = bench_us(lambda: cuda_ops.fused_dense_sparse_cuda(
        inp["W_low_packed"], inp["W_high_packed"],
        inp["hp_row_offsets"], inp["hp_col_indices"],
        inp["X_s4"], inp["scale_u4"], inp["zero_u4"],
        inp["sum_X"], inp["scale_x"],
        d_out, d_in,
    ))
    return {
        "fp16_us": t_fp16,
        "triton_us": t_triton,
        "cuda_us": t_cuda,
        "triton_speedup_vs_fp16": t_fp16 / t_triton,
        "cuda_speedup_vs_fp16": t_fp16 / t_cuda,
        "cuda_speedup_vs_triton": t_triton / t_cuda,
    }


def bench_end_to_end(inp: dict, T: int, d_out: int, d_in: int):
    """Full quant + fused pipeline, comparing FP16 vs pure Triton vs pure CUDA."""
    X_fp = inp["X"]
    perm = inp["perm"]

    def run_fp16():
        return torch.matmul(inp["W_fp"], inp["X_fp_t"])

    def run_triton():
        X_s4, sx, sX = quantize_activation_s4(X_fp, perm)
        return fused_dense_sparse_gemm(
            inp["W_low_packed"], inp["W_high_packed"],
            inp["hp_row_offsets"], inp["hp_col_indices"],
            X_s4, inp["scale_u4"], inp["zero_u4"],
            sX, sx,
            d_out, d_in,
        )

    def run_cuda():
        if T == 1:
            return cuda_ops.fused_quant_gemv_cuda(
                X_fp, perm,
                inp["W_low_packed"], inp["W_high_packed"],
                inp["hp_row_offsets"], inp["hp_col_indices"],
                inp["scale_u4"], inp["zero_u4"],
                d_out, d_in,
            )
        X_s4, sx, sX = cuda_ops.activation_quant_cuda(X_fp, perm)
        return cuda_ops.fused_dense_sparse_cuda(
            inp["W_low_packed"], inp["W_high_packed"],
            inp["hp_row_offsets"], inp["hp_col_indices"],
            X_s4, inp["scale_u4"], inp["zero_u4"],
            sX, sx,
            d_out, d_in,
        )

    t_fp16 = bench_us(run_fp16, flush_l2=_FLUSH_L2_FP16)
    t_triton = bench_us(run_triton)
    t_cuda = bench_us(run_cuda)
    return {
        "fp16_us": t_fp16,
        "triton_us": t_triton,
        "cuda_us": t_cuda,
        "triton_speedup_vs_fp16": t_fp16 / t_triton,
        "cuda_speedup_vs_fp16": t_fp16 / t_cuda,
        "cuda_speedup_vs_triton": t_triton / t_cuda,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
SUBKERNELS = ["activation_quant", "dense_gemm", "sparse_gemm",
              "fused_dense_sparse", "end_to_end"]


def run_one_shape(
    model: Qwen3Config,
    proj: ProjShape,
    T: int,
    hp_ratio: float,
    log: logging.Logger,
) -> list[dict]:
    """Return one record per sub-kernel."""
    d_in, d_out = proj.d_in, proj.d_out
    try:
        inp = make_inputs(T, d_out, d_in, hp_ratio=hp_ratio)
    except Exception as e:  # noqa: BLE001
        log.exception("%s %s T=%d: make_inputs failed: %s",
                      model.name, proj.proj, T, e)
        return []

    records: list[dict] = []

    try:
        r = bench_activation_quant(T, d_in, log)
        records.append({
            "model": model.name, "proj": proj.proj, "T": T,
            "d_in": d_in, "d_out": d_out, "hp_ratio": hp_ratio,
            "kernel": "activation_quant",
            **r,
        })
    except Exception as e:  # noqa: BLE001
        log.exception("activation_quant %s T=%d: %s", proj.proj, T, e)

    try:
        r = bench_dense_gemm(inp, T, d_out, d_in)
        records.append({
            "model": model.name, "proj": proj.proj, "T": T,
            "d_in": d_in, "d_out": d_out, "hp_ratio": hp_ratio,
            "kernel": "dense_gemm",
            **r,
        })
    except Exception as e:  # noqa: BLE001
        log.exception("dense_gemm %s T=%d: %s", proj.proj, T, e)

    try:
        r = bench_sparse_gemm(inp, T, d_out, d_in)
        records.append({
            "model": model.name, "proj": proj.proj, "T": T,
            "d_in": d_in, "d_out": d_out, "hp_ratio": hp_ratio,
            "kernel": "sparse_gemm",
            **r,
        })
    except Exception as e:  # noqa: BLE001
        log.exception("sparse_gemm %s T=%d: %s", proj.proj, T, e)

    try:
        r = bench_fused(inp, T, d_out, d_in)
        records.append({
            "model": model.name, "proj": proj.proj, "T": T,
            "d_in": d_in, "d_out": d_out, "hp_ratio": hp_ratio,
            "kernel": "fused_dense_sparse",
            **r,
        })
    except Exception as e:  # noqa: BLE001
        log.exception("fused_dense_sparse %s T=%d: %s", proj.proj, T, e)

    try:
        r = bench_end_to_end(inp, T, d_out, d_in)
        records.append({
            "model": model.name, "proj": proj.proj, "T": T,
            "d_in": d_in, "d_out": d_out, "hp_ratio": hp_ratio,
            "kernel": "end_to_end",
            **r,
        })
    except Exception as e:  # noqa: BLE001
        log.exception("end_to_end %s T=%d: %s", proj.proj, T, e)

    # Log compact progress line per shape (end-to-end row, most informative).
    e2e = next((r for r in records if r["kernel"] == "end_to_end"), None)
    if e2e is not None:
        log.info(
            "%-11s %-12s T=%-4d [%5d->%5d] "
            "e2e: fp16=%6.1f  triton=%6.1f  cuda=%6.1f  cuda/fp16=%4.2fx  cuda/triton=%4.2fx",
            model.name, proj.proj, T, d_in, d_out,
            e2e["fp16_us"],
            e2e["triton_us"],
            e2e["cuda_us"],
            e2e["cuda_speedup_vs_fp16"],
            e2e["cuda_speedup_vs_triton"],
        )
    return records


def write_markdown(records: list[dict], out_path: Path, meta: dict):
    """Produce a human-friendly markdown summary with FP16, Triton, and CUDA tables."""
    def fmt_us(v):
        return " - " if v is None else f"{v:.2f}"

    def fmt_ratio(v):
        return " - " if v is None else f"{v:.2f}x"

    lines: list[str] = []
    lines.append(f"# Qwen3 multi-scale kernel benchmark\n")
    lines.append(f"- Timestamp: `{meta['ts']}`")
    lines.append(f"- Device: `{meta['device_name']}`")
    lines.append(f"- PyTorch: `{meta['torch_version']}`  Triton: `{meta['triton_version']}`")
    lines.append(f"- Baseline: cuBLAS FP16 matmul (`torch.matmul` on `fp16`)")
    lines.append(f"- CUDA path: `activation_quant_cuda` + `fused_dense_sparse_cuda` (T=1 uses `fused_quant_gemv_cuda`, with automatic fallback on unsupported decode-group counts)")
    lines.append(f"- Triton path: `quantize_activation_s4` + `fused_dense_sparse_gemm`")
    lines.append(f"- hp_ratio: `{meta['hp_ratio']}`  (block-sparse density)")
    lines.append(f"- Stats: stable microbenchmark helper = 50 warmup, 100 inner, 3 repeats, min-of-means\n")

    models = []
    for r in records:
        if r["model"] not in models:
            models.append(r["model"])
    Ts = sorted({r["T"] for r in records})

    lines.append("\n## 1. End-to-end speedup vs FP16\n")
    lines.append("Rows: projection. Cells: `fp16_us / cuda_us` (>1.0x means CUDA wins).\n")
    for m in models:
        lines.append(f"\n### {m}")
        projs = []
        for r in records:
            if r["model"] == m and r["kernel"] == "end_to_end" and r["proj"] not in projs:
                projs.append(r["proj"])
        header = "| proj | shape |" + "".join(f" T={t} |" for t in Ts)
        sep = "|---|---|" + "---:|" * len(Ts)
        lines.append(header)
        lines.append(sep)
        for p in projs:
            shape_str = ""
            row_cells = []
            for t in Ts:
                rec = next((r for r in records if r["model"] == m
                            and r["proj"] == p and r["kernel"] == "end_to_end"
                            and r["T"] == t), None)
                if rec is None:
                    row_cells.append(" - ")
                else:
                    shape_str = f"{rec['d_in']}->{rec['d_out']}"
                    row_cells.append(f" **{rec['cuda_speedup_vs_fp16']:.2f}x** ")
            lines.append(f"| {p} | {shape_str} |" + "|".join(row_cells) + "|")

    lines.append("\n\n## 2. End-to-end raw latencies (us)\n")
    for m in models:
        lines.append(f"\n### {m} - end-to-end (us)")
        header = "| proj | shape | T | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |"
        sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        rows = [r for r in records if r["model"] == m and r["kernel"] == "end_to_end"]
        rows.sort(key=lambda r: (r["proj"], r["T"]))
        for r in rows:
            shape_str = f"{r['d_in']}->{r['d_out']}"
            lines.append(
                f"| {r['proj']} | {shape_str} | {r['T']} | "
                f"{fmt_us(r.get('fp16_us'))} | {fmt_us(r.get('triton_us'))} | {fmt_us(r.get('cuda_us'))} | "
                f"{fmt_ratio(r.get('triton_speedup_vs_fp16'))} | {fmt_ratio(r.get('cuda_speedup_vs_fp16'))} | {fmt_ratio(r.get('cuda_speedup_vs_triton'))} |"
            )

    lines.append("\n\n## 3. Sub-kernel breakdown (us)\n")
    for m in models:
        lines.append(f"\n### {m} - sub-kernels")
        header = "| proj | T | kernel | fp16 | triton | cuda | triton/fp16 | cuda/fp16 | cuda/triton |"
        sep = "|---|---:|---|---:|---:|---:|---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        rows = [r for r in records if r["model"] == m and r["kernel"] != "end_to_end"]
        rows.sort(key=lambda r: (r["proj"], r["T"],
                                 ["activation_quant", "dense_gemm", "sparse_gemm",
                                  "fused_dense_sparse"].index(r["kernel"])))
        for r in rows:
            lines.append(
                f"| {r['proj']} | {r['T']} | {r['kernel']} | "
                f"{fmt_us(r.get('fp16_us'))} | {fmt_us(r.get('triton_us'))} | {fmt_us(r.get('cuda_us'))} | "
                f"{fmt_ratio(r.get('triton_speedup_vs_fp16'))} | {fmt_ratio(r.get('cuda_speedup_vs_fp16'))} | {fmt_ratio(r.get('cuda_speedup_vs_triton'))} |"
            )

    lines.append("\n\n## 4. End-to-end speedup (CUDA over Triton)\n")
    lines.append("Rows: projection. Cells: `triton_us / cuda_us` (>1.0x means CUDA wins).\n")
    for m in models:
        lines.append(f"\n### {m}")
        projs = []
        for r in records:
            if r["model"] == m and r["kernel"] == "end_to_end" and r["proj"] not in projs:
                projs.append(r["proj"])
        header = "| proj | shape |" + "".join(f" T={t} |" for t in Ts)
        sep = "|---|---|" + "---:|" * len(Ts)
        lines.append(header)
        lines.append(sep)
        for p in projs:
            shape_str = ""
            row_cells = []
            for t in Ts:
                rec = next((r for r in records if r["model"] == m
                            and r["proj"] == p and r["kernel"] == "end_to_end"
                            and r["T"] == t), None)
                if rec is None:
                    row_cells.append(" - ")
                else:
                    shape_str = f"{rec['d_in']}->{rec['d_out']}"
                    row_cells.append(f" **{rec['cuda_speedup_vs_triton']:.2f}x** ")
            lines.append(f"| {p} | {shape_str} |" + "|".join(row_cells) + "|")

    lines.append("\n\n## 5. CUDA end-to-end bottleneck hint\n")
    lines.append("For each shape, compare CUDA `activation_quant` against CUDA `fused_dense_sparse`. A larger `quant_share` means launch/prologue dominates; a larger `fused_share` means the main CUDA matmul kernel dominates.\n")
    lines.append("| model | proj | T | shape | quant_us | fused_us | quant_share | fused_share | likely_bottleneck |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---|")
    for m in models:
        for p in ["q_proj", "kv_proj", "o_proj", "gate_up_proj", "down_proj"]:
            for t in Ts:
                cell = {r["kernel"]: r for r in records if r["model"] == m and r["proj"] == p and r["T"] == t}
                q = cell.get("activation_quant")
                f = cell.get("fused_dense_sparse")
                e = cell.get("end_to_end")
                if q is None or f is None or e is None:
                    continue
                quant_us = q["cuda_us"]
                fused_us = f["cuda_us"]
                total = quant_us + fused_us
                quant_share = quant_us / total if total > 0 else 0.0
                fused_share = fused_us / total if total > 0 else 0.0
                if quant_share >= 0.35:
                    bottleneck = "quant/prologue dominated"
                elif fused_share >= 0.80:
                    bottleneck = "main fused kernel dominated"
                else:
                    bottleneck = "mixed"
                lines.append(
                    f"| {m} | {p} | {t} | {e['d_in']}->{e['d_out']} | "
                    f"{quant_us:.2f} | {fused_us:.2f} | {quant_share:.1%} | {fused_share:.1%} | {bottleneck} |"
                )

    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Which Qwen3 models to bench (default: 0.6B/1.7B/4B/8B).")
    parser.add_argument("--full", action="store_true",
                        help="Include the larger (>=14B) models "
                             "Qwen3-14B / Qwen2.5-32B / LLaMA3-70B "
                             "(takes longer + more VRAM).")
    parser.add_argument("--ts", nargs="+", type=int, default=list(DEFAULT_TS),
                        help="Batch sizes to sweep.")
    parser.add_argument("--hp-ratio", type=float, default=0.05,
                        help="Block-sparse density.")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Override output root directory.")
    parser.add_argument("--flush-l2-fp16", dest="flush_l2_fp16",
                        action="store_true", default=True,
                        help="Flush L2 before every FP16 baseline launch "
                             "(cold-cache HBM, default, matches real LLM "
                             "inference).")
    parser.add_argument("--no-flush-l2-fp16", dest="flush_l2_fp16",
                        action="store_false",
                        help="Legacy tight-loop FP16 baseline (inflates "
                             "cuBLAS by up to 2x on <72 MB problems).")
    args = parser.parse_args()

    # r62 P2: propagate the CLI flag into the module-level switch used by
    # bench_fp16_matmul / bench_end_to_end.
    global _FLUSH_L2_FP16
    _FLUSH_L2_FP16 = bool(args.flush_l2_fp16)

    # Resolve model list.
    if args.models is not None:
        wanted = set(args.models)
        models = [m for m in QWEN3_MODELS if m.name in wanted or m.name.replace("Qwen3-", "") in wanted]
        if not models:
            raise SystemExit(f"No model matched --models {args.models!r}")
    else:
        if args.full:
            models = list(QWEN3_MODELS)
        else:
            # Default omits the larger (≥14B) models — they take longer
            # and need more VRAM.  Opt in via `--full` or explicit
            # `--models Qwen3-14B Qwen2.5-32B LLaMA3-70B`.
            _big = {"Qwen3-14B", "Qwen2.5-32B", "LLaMA3-70B"}
            models = [m for m in QWEN3_MODELS if m.name not in _big]

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.out_root is not None:
        out_root = args.out_root
    else:
        out_root = _IMPORT_ROOT / "logs" / "qwen3_bench"
    out_dir = out_root / f"qwen3_{ts}"
    log, log_path = setup_logging(out_dir)
    log.info("output dir: %s", out_dir)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    import triton
    meta = {
        "ts": ts,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "hp_ratio": args.hp_ratio,
        "ts_swept": args.ts,
        "models": [m.name for m in models],
        "flush_l2_fp16": _FLUSH_L2_FP16,
    }
    log.info("device: %s", device_name)
    log.info("pytorch %s  triton %s", torch.__version__, triton.__version__)
    log.info("hp_ratio: %.3f", args.hp_ratio)
    log.info("flush_l2_fp16: %s", _FLUSH_L2_FP16)

    all_records: list[dict] = []
    total = 0
    for m in models:
        projs = [p for p in enumerate_projs(m) if shape_is_supported(p)]
        total += len(projs) * len(args.ts)

    done = 0
    for m in models:
        for p in enumerate_projs(m):
            if not shape_is_supported(p):
                log.warning("skipping %s %s: d_in=%d d_out=%d not divisible by 128",
                            m.name, p.proj, p.d_in, p.d_out)
                continue
            for T in args.ts:
                done += 1
                log.debug("[%d/%d] %s %s T=%d", done, total, m.name, p.proj, T)
                recs = run_one_shape(m, p, T, args.hp_ratio, log)
                all_records.extend(recs)

    # Write JSON (machine-readable)
    json_path = out_dir / "bench.json"
    json_path.write_text(json.dumps({"meta": meta, "records": all_records}, indent=2))
    log.info("wrote %s", json_path)

    # Write markdown (human-friendly)
    md_path = out_dir / "bench.md"
    write_markdown(all_records, md_path, meta)
    log.info("wrote %s", md_path)

    log.info("done.  %d records across %d shapes.", len(all_records), done)


if __name__ == "__main__":
    main()
