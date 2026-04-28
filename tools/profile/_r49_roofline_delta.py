"""One-off script for R49 Step 1 roofline-delta table.

Loads cuda_graph_bench/bench.json, computes per-shape CUDA / FP16 roofline
times using the same formulas as ``logs/qwen3_bench/...roofline_report.md``
§1, then emits the markdown body for that report's new §8 section.

Not part of the production benching surface; safe to delete after the
R49 Step 1 retrospective lands.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

BENCH_JSON = (
    Path(__file__).resolve().parents[2]
    / "cuda_kernel"
    / "logs"
    / "phase3_optimization"
    / "cuda_graph_bench"
    / "bench.json"
)

# roofline constants (RTX 4090 vendor spec, ACHIEVABLE=0.85)
HBM_GBS = 1008.0
INT4_TOPS = 660.6
FP16_TFLOPS = 165.2
ACH = 0.85
EFF_HBM = HBM_GBS * ACH  # GB/s
EFF_INT4 = INT4_TOPS * ACH  # TOPS
EFF_FP16 = FP16_TFLOPS * ACH  # TFLOPS
GROUP = 128


def parse_tag(tag: str) -> tuple[int, int, int]:
    """audit_<model>_<proj>_T<T>_<d_in>_<d_out> -> (T, d_in, d_out)."""
    parts = tag.split("_")
    T = int(parts[3][1:])
    d_in = int(parts[4])
    d_out = int(parts[5])
    return T, d_in, d_out


def cuda_roof_us(T: int, d_in: int, d_out: int) -> float:
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
    flops = 2 * T * d_in * d_out
    bytes_ = 2 * (d_in * d_out + T * d_in + T * d_out)
    return max(flops / (EFF_FP16 * 1e6), bytes_ / (EFF_HBM * 1e3))


def main() -> None:
    d = json.loads(BENCH_JSON.read_text())
    print(f"## R49 Step 1 — launch_sparse cluster roofline delta")
    print()
    print(f"- bench run: `{d['meta']['run_id']}` on `{d['meta']['device']}`")
    print(
        f"- timer: warmup={d['meta']['warmup']}, outer={d['meta']['outer']},"
        f" inner={d['meta']['inner']}, K={d['meta']['K']}"
    )
    print()
    print(
        "| shape | T | d_in | d_out | fp16_roof_us | cuda_roof_us |"
        " eager_us | eager_eff | graph_us | graph_eff | Δ eff (pp) |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows: list[tuple[float, float, float, float, float]] = []
    for r in d["records"]:
        tag = r["tag"]
        T, d_in, d_out = parse_tag(tag)
        cr = cuda_roof_us(T, d_in, d_out)
        fr = fp16_roof_us(T, d_in, d_out)
        e = r["t_eager_med_us"]
        g = r["t_graph_med_us"]
        ee = cr / e * 100
        eg = cr / g * 100
        delta = eg - ee
        rows.append((e, g, ee, eg, delta))
        sign = "+" if delta >= 0 else ""
        print(
            f"| {tag} | {T} | {d_in} | {d_out} | {fr:.2f} | {cr:.2f} |"
            f" {e:.2f} | {ee:.0f}% | {g:.2f} | {eg:.0f}% |"
            f" {sign}{delta:.0f}pp |"
        )
    print()
    eagers = [r[0] for r in rows]
    graphs = [r[1] for r in rows]
    eff_e = [r[2] for r in rows]
    eff_g = [r[3] for r in rows]
    deltas = [r[4] for r in rows]
    saved = sum(eagers) - sum(graphs)
    print(
        f"- eager cuda_eff: median **{statistics.median(eff_e):.1f}%** "
        f"(min {min(eff_e):.1f}% / max {max(eff_e):.1f}%)"
    )
    print(
        f"- graph cuda_eff: median **{statistics.median(eff_g):.1f}%** "
        f"(min {min(eff_g):.1f}% / max {max(eff_g):.1f}%)"
    )
    print(
        f"- median Δ cuda_eff: **+{statistics.median(deltas):.1f}pp**"
    )
    print(
        f"- aggregate wall-time saved over 17 shapes: "
        f"**{saved:.1f}us / {sum(eagers):.1f}us "
        f"({saved/sum(eagers)*100:.1f}%)**"
    )


if __name__ == "__main__":
    main()
