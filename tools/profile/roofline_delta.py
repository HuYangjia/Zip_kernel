"""Roofline-delta reporter for eager vs optimised kernel paths.

Generalised from the R49 Step 1 one-off script.  Given a bench JSON in
the schema emitted by ``kernel/tools/profile/bench_cuda_graph_vs_eager.py``
(or any sibling bench with ``{"meta": {...}, "records": [{"tag",
"t_eager_med_us", "t_graph_med_us"}, ...]}`` shape), emit the markdown
body of a "§roofline delta" section that can be dropped straight into
the canonical ``logs/qwen3_bench/<run_id>/roofline_report.md`` file.

The roofline formulas are the ones specified in
``roofline_report.md`` §1 (CUDA INT4 path: separate quant + GEMM bytes,
vendor peaks scaled by ACHIEVABLE=0.85).  If roofline_report.md ever
revises those constants, update here in lock-step.

Typical usage (CLI)::

    python -m kernel.tools.profile.roofline_delta \\
        --bench cuda_kernel/logs/phase3_optimization/cuda_graph_bench/bench.json \\
        --title "R49 Step 1 - launch_sparse cluster"                           \\
        --eager-label eager --opt-label graph                                  \\
        --output /dev/stdout

This module is also importable; see :func:`render_delta_markdown` for
the programmatic entry.  Unit tests live next door in
``kernel/tools/profile/tests/test_roofline_delta.py``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TextIO

# ---------------------------------------------------------------------------
# Roofline constants — RTX 4090 vendor spec, ACHIEVABLE fraction = 0.85.
# Keep in sync with roofline_report.md §1.
# ---------------------------------------------------------------------------
HBM_GBS = 1008.0
INT4_TOPS = 660.6
FP16_TFLOPS = 165.2
ACHIEVABLE = 0.85
EFF_HBM = HBM_GBS * ACHIEVABLE
EFF_INT4 = INT4_TOPS * ACHIEVABLE
EFF_FP16 = FP16_TFLOPS * ACHIEVABLE
GROUP = 128


@dataclass(frozen=True)
class ShapeSpec:
    """Extracted (T, d_in, d_out) triple from a benchmark tag.

    The canonical tag layout used by R49/R50 benches is
    ``audit_<model>_<proj>_T<T>_<d_in>_<d_out>``.  Non-canonical tags
    fall through to :func:`parse_tag` which will raise ValueError.
    """

    T: int
    d_in: int
    d_out: int


def parse_tag(tag: str) -> ShapeSpec:
    """Parse ``audit_<model>_<proj>_T<T>_<d_in>_<d_out>`` into a ShapeSpec.

    Raises ValueError if the tag does not contain the ``T<N>`` marker
    and two integer fields after it.
    """
    parts = tag.split("_")
    t_field = next((p for p in parts if p.startswith("T") and p[1:].isdigit()), None)
    if t_field is None:
        raise ValueError(f"tag {tag!r} does not contain a T<n> field")
    idx = parts.index(t_field)
    try:
        T = int(t_field[1:])
        d_in = int(parts[idx + 1])
        d_out = int(parts[idx + 2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"tag {tag!r} missing d_in/d_out after {t_field}") from exc
    return ShapeSpec(T=T, d_in=d_in, d_out=d_out)


def cuda_roof_us(T: int, d_in: int, d_out: int) -> float:
    """CUDA INT4 W4A4 roofline in microseconds, per roofline_report.md §1.

    - T == 1 path treats activation-quant + GEMM as a single GEMV whose
      bytes include the activation scratch (no split).
    - T > 1 path splits activation-quant and GEMM kernels: quant bytes +
      max(gemm compute, gemm bytes).
    """
    ng = d_in // GROUP
    t_gemm_compute = 2 * T * d_in * d_out / (EFF_INT4 * 1e6)
    if T == 1:
        bytes_gv = 2 * d_in + 0.5 * d_in * d_out + 4 * d_out * ng + 2 * d_out
        return max(t_gemm_compute, bytes_gv / (EFF_HBM * 1e3))
    bytes_q = 2 * T * d_in + 0.5 * T * d_in + 2 * T + 4 * T * ng
    bytes_g = 0.5 * d_in * d_out + 0.5 * T * d_in + 4 * d_out * ng + 2 * T * d_out
    return bytes_q / (EFF_HBM * 1e3) + max(
        t_gemm_compute, bytes_g / (EFF_HBM * 1e3)
    )


def fp16_roof_us(T: int, d_in: int, d_out: int) -> float:
    """FP16 cuBLAS-style roofline in microseconds: max(flops/peak, bytes/BW)."""
    flops = 2 * T * d_in * d_out
    bytes_ = 2 * (d_in * d_out + T * d_in + T * d_out)
    return max(flops / (EFF_FP16 * 1e6), bytes_ / (EFF_HBM * 1e3))


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_delta_markdown(
    bench: dict,
    *,
    title: str,
    eager_label: str = "eager",
    opt_label: str = "graph",
    out: TextIO | None = None,
) -> str:
    """Render a markdown delta-report body to ``out`` and return it.

    ``bench`` schema:
        {
          "meta": {"run_id": str, "device": str, "warmup": int,
                   "outer": int, "inner": int, "K": int},
          "records": [{"tag": str,
                        "t_eager_med_us": float,
                        "t_graph_med_us": float}, ...]
        }

    The tag-field names ``t_eager_med_us`` / ``t_graph_med_us`` are kept
    literal for R49 bench compatibility.  For Step 2 (R50) benches using
    different field names, pre-process the JSON into this schema before
    calling.
    """
    lines: list[str] = []
    meta = bench.get("meta", {})
    lines.append(f"## {title} — roofline delta")
    lines.append("")
    lines.append(f"- bench run: `{meta.get('run_id', 'unknown')}` on `{meta.get('device', 'unknown')}`")
    lines.append(
        f"- timer: warmup={meta.get('warmup', '?')}, outer={meta.get('outer', '?')},"
        f" inner={meta.get('inner', '?')}, K={meta.get('K', '?')}"
    )
    lines.append("")
    lines.append(
        f"| shape | T | d_in | d_out | fp16_roof_us | cuda_roof_us |"
        f" {eager_label}_us | {eager_label}_eff | {opt_label}_us |"
        f" {opt_label}_eff | Δ eff (pp) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    rows: list[tuple[float, float, float, float, float]] = []
    for r in bench.get("records", []):
        spec = parse_tag(r["tag"])
        cr = cuda_roof_us(spec.T, spec.d_in, spec.d_out)
        fr = fp16_roof_us(spec.T, spec.d_in, spec.d_out)
        e = float(r["t_eager_med_us"])
        g = float(r["t_graph_med_us"])
        ee = cr / e * 100
        eg = cr / g * 100
        delta = eg - ee
        rows.append((e, g, ee, eg, delta))
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {r['tag']} | {spec.T} | {spec.d_in} | {spec.d_out} |"
            f" {fr:.2f} | {cr:.2f} |"
            f" {e:.2f} | {ee:.0f}% | {g:.2f} | {eg:.0f}% |"
            f" {sign}{delta:.0f}pp |"
        )
    lines.append("")

    if rows:
        eagers = [r[0] for r in rows]
        graphs = [r[1] for r in rows]
        eff_e = [r[2] for r in rows]
        eff_g = [r[3] for r in rows]
        deltas = [r[4] for r in rows]
        saved = sum(eagers) - sum(graphs)
        lines.extend([
            f"- {eager_label} cuda_eff: median **{statistics.median(eff_e):.1f}%** "
            f"(min {min(eff_e):.1f}% / max {max(eff_e):.1f}%)",
            f"- {opt_label} cuda_eff: median **{statistics.median(eff_g):.1f}%** "
            f"(min {min(eff_g):.1f}% / max {max(eff_g):.1f}%)",
            f"- median Δ cuda_eff: **+{statistics.median(deltas):.1f}pp**",
            f"- aggregate wall-time saved over {len(rows)} shapes: "
            f"**{saved:.1f}us / {sum(eagers):.1f}us "
            f"({saved/sum(eagers)*100:.1f}%)**",
        ])

    text = "\n".join(lines) + "\n"
    if out is not None:
        out.write(text)
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="roofline_delta",
        description="Render a roofline-delta markdown section for a bench JSON.",
    )
    p.add_argument(
        "--bench", required=True, type=Path,
        help="path to bench JSON with schema {meta:{...}, records:[{tag, "
             "t_eager_med_us, t_graph_med_us}, ...]}",
    )
    p.add_argument(
        "--title", default="Roofline delta",
        help="title for the emitted markdown section (default: 'Roofline delta')",
    )
    p.add_argument(
        "--eager-label", default="eager",
        help="column label for the baseline path (default: 'eager')",
    )
    p.add_argument(
        "--opt-label", default="graph",
        help="column label for the optimised path (default: 'graph')",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="output file; stdout if omitted",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    bench = json.loads(args.bench.read_text())
    if args.output is None:
        render_delta_markdown(
            bench,
            title=args.title,
            eager_label=args.eager_label,
            opt_label=args.opt_label,
            out=sys.stdout,
        )
    else:
        with args.output.open("w") as fh:
            render_delta_markdown(
                bench,
                title=args.title,
                eager_label=args.eager_label,
                opt_label=args.opt_label,
                out=fh,
            )


if __name__ == "__main__":
    main()
