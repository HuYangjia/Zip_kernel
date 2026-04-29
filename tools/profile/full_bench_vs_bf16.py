"""Full-shape bench: CUDA INT4 W4A4 kernel vs BF16 cuBLAS GEMM vs Roofline.

Usage (from repo root, on GPU host):
    python -m kernel.tools.profile.full_bench_vs_bf16 \
        --output kernel/logs/full_bench_r54/report.md

What it does
------------
For every (d_out, d_in, T) triple in the sweep list:

1. Build representative W4A4 inputs (packed INT4 weight, packed INT4
   activation, per-(M, g) scale/zero, per-T activation scale, sum_X).
2. Measure the fused CUDA INT4 kernel via torch.cuda.Event with warmup
   + multi-window median, following the project benchmarking policy.
3. Measure a BF16 reference matmul `torch.matmul(Wbf, Xbf.T)` of the
   same (d_out, d_in) x (d_in, T) problem on the same device.
4. Compute Roofline reference times for both the INT4 and FP16/BF16
   paths using the same formulas as ``roofline_delta.py``.
5. Emit a markdown report summarising:
   - absolute times
   - INT4 / BF16 speed-up
   - INT4 roofline efficiency (t_roof / t_measured)
   - BF16 roofline efficiency

This script is tolerant of missing CUDA (emits "skip" for all rows) so
the import path stays clean on Mac for unit testing.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

# Ensure we can import the roofline helpers next door even when run as
# a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
from roofline_delta import cuda_roof_us, fp16_roof_us  # type: ignore  # noqa: E402

# Canonical Qwen-3 8B projection shapes at T in {1, 32, 128, 512}.
# We sample a handful that expose the per-n_groups/per-M sweep we care
# about. Feel free to extend.
DEFAULT_SHAPES = [
    # (d_out, d_in, T) — ordered by ng then M.
    (1024,  1024, 128),   # ng=8   small
    (2048,  2048, 128),   # ng=16  flagship
    (4096,  4096, 128),   # ng=32  flagship
    (1024,  4096, 128),   # ng=32  tall-thin
    (4096,  1024, 128),   # ng=8   wide-fat
    (2048,  4096, 128),   # ng=32
    (4096,  2048, 128),   # ng=16
    # Qwen-3-8B specific (approximate):
    (4096,  4096,  32),   # prefill-ish
    (4096,  4096,   1),   # decode
    (4096, 14336, 128),   # down_proj-ish
    (14336, 4096, 128),   # up/gate_proj-ish
]


@dataclass
class BenchRecord:
    d_out: int
    d_in: int
    T: int
    n_groups: int
    t_int4_us: float
    t_bf16_us: float
    roof_int4_us: float
    roof_bf16_us: float

    @property
    def speedup(self) -> float:
        return self.t_bf16_us / self.t_int4_us if self.t_int4_us > 0 else 0.0

    @property
    def int4_eff(self) -> float:
        return self.roof_int4_us / self.t_int4_us * 100 if self.t_int4_us > 0 else 0.0

    @property
    def bf16_eff(self) -> float:
        return self.roof_bf16_us / self.t_bf16_us * 100 if self.t_bf16_us > 0 else 0.0


def _bench_cuda_event(fn, warmup: int, outer: int, inner: int) -> float:
    """Median of `outer` timing windows, each averaging `inner` launches."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e) / inner * 1000.0)  # ms -> us
    return statistics.median(samples)


def run_sweep(shapes, warmup=500, outer=15, inner=200) -> List[BenchRecord]:
    import torch
    from kernel.cuda_kernel import ops as _ops
    os.environ.setdefault("HKUST_V9_USE_CUTLASS", "0")

    records: List[BenchRecord] = []
    for (d_out, d_in, T) in shapes:
        ng = max(d_in // 128, 1)
        assert d_in % 128 == 0, f"d_in={d_in} not a multiple of 128"

        torch.manual_seed(0)
        W = torch.randint(-8, 7, (d_out, d_in // 2), dtype=torch.int8, device="cuda")
        X = torch.randint(-8, 7, (T,     d_in // 2), dtype=torch.int8, device="cuda")
        s  = torch.rand(d_out, ng, dtype=torch.float16, device="cuda") * 0.1 + 0.01
        z  = torch.randint(-4, 4, (d_out, ng), device="cuda").to(torch.float16)
        sx = torch.rand(T, dtype=torch.float16, device="cuda") * 0.1 + 0.01
        sumX = torch.randint(-100, 100, (T, ng), dtype=torch.int32, device="cuda")
        Y  = torch.empty((d_out, T), dtype=torch.float16, device="cuda")
        Wh = torch.zeros(0, 128, 64, dtype=torch.int8, device="cuda")
        ro = torch.zeros(d_out // 128 + 1, dtype=torch.int32, device="cuda")
        ci = torch.zeros(0, dtype=torch.int32, device="cuda")

        def run_int4():
            _ops._ext.fused_dense_sparse_mma_int4_launch(
                W, Wh, ro, ci, X, s, z, sumX, sx, Y, d_out, d_in
            )

        t_int4 = _bench_cuda_event(run_int4, warmup, outer, inner)

        # BF16 reference: torch.matmul (cuBLAS) for the same logical
        # problem. We use random bf16 matrices of the target shape —
        # the absolute time depends on shape only (not values), which is
        # what cuBLAS heuristics key on.
        Wbf = torch.randn(d_out, d_in, dtype=torch.bfloat16, device="cuda")
        Xbf = torch.randn(T,     d_in, dtype=torch.bfloat16, device="cuda")

        def run_bf16():
            # (d_out, d_in) x (d_in, T) -> (d_out, T)
            torch.matmul(Wbf, Xbf.t())

        t_bf16 = _bench_cuda_event(run_bf16, warmup, outer, inner)

        rec = BenchRecord(
            d_out=d_out, d_in=d_in, T=T, n_groups=ng,
            t_int4_us=t_int4,
            t_bf16_us=t_bf16,
            roof_int4_us=cuda_roof_us(T, d_in, d_out),
            roof_bf16_us=fp16_roof_us(T, d_in, d_out),  # BF16 shares the FP16 vendor peak on Ada
        )
        records.append(rec)
        print(
            f"{d_out}x{d_in}x{T} ng={ng}: "
            f"int4={t_int4:7.2f}us  bf16={t_bf16:7.2f}us  "
            f"x{rec.speedup:.2f}  int4_eff={rec.int4_eff:.0f}%  "
            f"bf16_eff={rec.bf16_eff:.0f}%"
        )
    return records


def render_report(records: List[BenchRecord], title: str) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        "Full-shape bench of the CUDA INT4 W4A4 fused kernel (r54 stage-B.1) "
        "against a BF16 cuBLAS `torch.matmul` baseline on the same device, "
        "with Roofline efficiency computed per the canonical formulas in "
        "`kernel/tools/profile/roofline_delta.py` (RTX 4090 vendor peaks, "
        "ACHIEVABLE = 0.85)."
    )
    lines.append("")
    lines.append(
        "| shape (d_out×d_in×T) | ng | INT4 (μs) | BF16 (μs) | speed-up | "
        "roof_INT4 (μs) | INT4 eff | roof_FP16 (μs) | BF16 eff |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in records:
        lines.append(
            f"| {r.d_out}×{r.d_in}×{r.T} | {r.n_groups} | "
            f"{r.t_int4_us:.2f} | {r.t_bf16_us:.2f} | {r.speedup:.2f}× | "
            f"{r.roof_int4_us:.2f} | {r.int4_eff:.0f}% | "
            f"{r.roof_bf16_us:.2f} | {r.bf16_eff:.0f}% |"
        )

    # Aggregate stats.
    if records:
        sps = [r.speedup for r in records]
        e_int4 = [r.int4_eff for r in records]
        e_bf16 = [r.bf16_eff for r in records]
        lines.append("")
        lines.append("## Aggregate")
        lines.append("")
        lines.append(
            f"- median INT4/BF16 speed-up: **{statistics.median(sps):.2f}×** "
            f"(min {min(sps):.2f}× / max {max(sps):.2f}×)"
        )
        lines.append(
            f"- median INT4 roofline efficiency: **{statistics.median(e_int4):.1f}%** "
            f"(min {min(e_int4):.1f}% / max {max(e_int4):.1f}%)"
        )
        lines.append(
            f"- median BF16 roofline efficiency: **{statistics.median(e_bf16):.1f}%** "
            f"(min {min(e_bf16):.1f}% / max {max(e_bf16):.1f}%)"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True,
                   help="markdown report output path")
    p.add_argument("--json",   type=Path, default=None,
                   help="optional JSON dump of raw records")
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--outer",  type=int, default=15)
    p.add_argument("--inner",  type=int, default=200)
    p.add_argument("--title",  default="Full-shape bench vs BF16 + Roofline")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    records = run_sweep(
        DEFAULT_SHAPES,
        warmup=args.warmup, outer=args.outer, inner=args.inner,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(records, args.title))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            [asdict(r) for r in records], indent=2,
        ))


if __name__ == "__main__":
    main()
