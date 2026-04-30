"""r62 F2 Qwen3 roofline report generator.

Reads the cold-cache bench.json produced by
`kernel.cuda_kernel.benchmarks.bench_qwen3_shapes`, applies the RTX 4090
roofline model (HBM 1008 GB/s, INT4 TC peak 660.6 TOPS, FP16 TC peak
165.2 TFLOPS, ACHIEVABLE_FRACTION 0.85), and writes a markdown report
with four sections:

  §1 hardware constants + formulas
  §2 FP16 efficiency distribution by T
  §3 CUDA efficiency distribution by T / by proj
  §4 per-shape detail (cuda_us / cuda_roof_us / cuda_eff / speedup vs FP16)
  §5 worst CUDA-efficiency top-15 (implementation gap)
  §6 physics gap: cuda_roof vs fp16_roof (shapes where W4A4 cannot beat FP16)
  §7 conclusions

Only CUDA kernel numbers are used (Triton is not an evaluation target per
repo convention [memory 0d5nyof1]).  FP16 cuBLAS is kept as the reference
for the roofline comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Hardware constants (RTX 4090, vendor spec at boost clock)
# ============================================================
HBM_BW_GBS = 1008.0
FP16_TC_TFLOPS = 165.2
INT4_TC_TOPS = 660.6
ACHIEVABLE_FRACTION = 0.85

eff_hbm = HBM_BW_GBS * ACHIEVABLE_FRACTION                     # GB/s
eff_fp16 = FP16_TC_TFLOPS * ACHIEVABLE_FRACTION * 1e12          # FLOP/s
eff_int4 = INT4_TC_TOPS * ACHIEVABLE_FRACTION * 1e12            # OP/s


BCOL = 128  # int4 group size


def _bytes_fp16_gemm(T: int, d_in: int, d_out: int) -> float:
    """FP16 matmul HBM bytes (read W, read X, write Y)."""
    return 2.0 * (d_in * d_out + T * d_in + T * d_out)


def fp16_roof_us(T: int, d_in: int, d_out: int) -> float:
    flops = 2.0 * T * d_in * d_out
    t_compute = flops / eff_fp16           # s
    t_mem = _bytes_fp16_gemm(T, d_in, d_out) / (eff_hbm * 1e9)
    return max(t_compute, t_mem) * 1e6     # us


def cuda_roof_us(T: int, d_in: int, d_out: int) -> Dict[str, float]:
    """Return (t_quant, t_gemm, t_total) roofline in us, per §1 formulas."""
    n_groups = d_in // BCOL

    if T == 1:
        # T=1 fused GEMV path: traffic = read W_int4 + read X_int4 + meta + write Y
        bytes_fused = (
            0.5 * d_in * d_out       # W (int4 = 0.5 byte)
            + 2.0 * d_in             # X as fp16 (T=1 uses gemv path that reads fp16 X)
            + 4.0 * d_out * n_groups  # scale + zero (fp16 each = 2B × 2) × n_groups; approx as 4B
            + 2.0 * d_out            # Y fp16
        )
        t_mem = bytes_fused / (eff_hbm * 1e9)
        t_compute = 2.0 * d_in * d_out / eff_int4
        t_total = max(t_compute, t_mem) * 1e6
        return {"t_quant": 0.0, "t_gemm": t_total, "t_total": t_total}

    # T >= 2 — activation_quant + fused_dense_sparse in series
    # activation_quant: read X fp16, write X_s4 (int4), write scale_x (fp16), write sum_X (int32 per group)
    bytes_quant = (
        2.0 * T * d_in              # read X fp16
        + 0.5 * T * d_in            # write X_s4 (int4)
        + 2.0 * T                   # write scale_x fp16
        + 4.0 * T * n_groups        # write sum_X int32
    )
    t_quant = (bytes_quant / (eff_hbm * 1e9)) * 1e6

    # fused GEMM: HBM bytes
    bytes_gemm = (
        0.5 * d_in * d_out                     # W int4
        + 0.5 * T * d_in                       # X_s4 int4
        + 4.0 * d_out * n_groups               # scale+zero per (m, g): 2×fp16 × d_out × ng ≈ 4B
        + 4.0 * T * n_groups                   # sum_X int32 (t, g)
        + 2.0 * T                              # scale_x fp16
        + 2.0 * T * d_out                      # write Y fp16
    )
    t_gemm_mem = bytes_gemm / (eff_hbm * 1e9)
    t_gemm_cmp = 2.0 * T * d_in * d_out / eff_int4
    t_gemm = max(t_gemm_cmp, t_gemm_mem) * 1e6
    return {"t_quant": t_quant, "t_gemm": t_gemm, "t_total": t_quant + t_gemm}


@dataclass
class Row:
    model: str
    proj: str
    T: int
    d_in: int
    d_out: int
    fp16_us: float
    cuda_us: float
    fp16_roof: float
    cuda_roof: float
    t_quant_roof: float
    t_gemm_roof: float

    @property
    def fp16_eff(self) -> float:
        return self.fp16_roof / self.fp16_us if self.fp16_us > 0 else 0.0

    @property
    def cuda_eff(self) -> float:
        return self.cuda_roof / self.cuda_us if self.cuda_us > 0 else 0.0

    @property
    def actual_speedup(self) -> float:
        return self.fp16_us / self.cuda_us if self.cuda_us > 0 else 0.0

    @property
    def roof_speedup(self) -> float:
        return self.fp16_roof / self.cuda_roof if self.cuda_roof > 0 else 0.0

    @property
    def gemm_bound(self) -> str:
        mem = self.cuda_roof - self.t_quant_roof  # t_gemm
        bytes_cap = self.t_gemm_roof  # t_gemm already max(compute, mem)
        # Re-derive which side dominates
        t_cmp = 2.0 * self.T * self.d_in * self.d_out / eff_int4 * 1e6
        return "compute" if t_cmp >= self.t_gemm_roof - 1e-9 else "mem"


def extract_rows(bench_json: Path) -> List[Row]:
    data = json.loads(bench_json.read_text())
    recs = data["records"]
    # Group by (model, proj, T): cuda = end_to_end cuda_us, fp16 = end_to_end fp16_us
    # The bench records each shape once with both fp16_us and cuda_us on the e2e row.
    rows: List[Row] = []
    for r in recs:
        if r.get("kernel") != "end_to_end":
            continue
        T = r["T"]; d_in = r["d_in"]; d_out = r["d_out"]
        t_fp16 = r["fp16_us"]
        t_cuda = r["cuda_us"]
        fp16_r = fp16_roof_us(T, d_in, d_out)
        cr = cuda_roof_us(T, d_in, d_out)
        rows.append(Row(
            model=r["model"], proj=r["proj"], T=T, d_in=d_in, d_out=d_out,
            fp16_us=t_fp16, cuda_us=t_cuda,
            fp16_roof=fp16_r, cuda_roof=cr["t_total"],
            t_quant_roof=cr["t_quant"], t_gemm_roof=cr["t_gemm"],
        ))
    return rows


def _dist_table(pairs: List[tuple]) -> str:
    """pairs: list of (label, [values]) — render median / min / max."""
    hdr = "| label | n | min | median | max |\n|---|---:|---:|---:|---:|\n"
    out = []
    for lab, vs in pairs:
        if not vs:
            continue
        out.append(f"| {lab} | {len(vs)} | {min(vs)*100:.0f}% | "
                   f"{statistics.median(vs)*100:.0f}% | {max(vs)*100:.0f}% |")
    return hdr + "\n".join(out) + "\n"


def render_report(rows: List[Row], source: str, title: str) -> str:
    # Known parameter counts for scaling display in §0.1 / §4 / etc.
    # Unknown models fall to the end (sentinel=10_000) and are ordered
    # alphabetically after the known ones.
    PARAMS_B = {
        "Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-4B": 4.0,
        "Qwen3-8B": 8.0, "Qwen3-14B": 14.0,
        "Qwen2.5-32B": 32.0, "LLaMA3-70B": 70.0,
    }

    def model_sort_key(m: str):
        return (PARAMS_B.get(m, 10_000), m)

    L: List[str] = []
    L.append(f"# {title}")
    L.append("")
    L.append(f"Source: `{source}`")
    L.append("")
    L.append("GPU model: RTX 4090 (vendor spec, ACHIEVABLE_FRACTION=0.85)")
    L.append("")

    # =========================================================
    # §0 Executive Summary (headline numbers up top)
    # =========================================================
    L.append("## §0 Executive Summary")
    L.append("")
    spd_all = [r.actual_speedup for r in rows]
    ceff_all = [r.cuda_eff for r in rows]
    feff_all = [r.fp16_eff for r in rows]
    w10 = sum(1 for s in spd_all if s >= 1.0)
    w11 = sum(1 for s in spd_all if s >= 1.1)
    w20 = sum(1 for s in spd_all if s >= 2.0)
    lo = sum(1 for s in spd_all if s < 0.9)
    best = max(rows, key=lambda r: r.actual_speedup)
    worst = min(rows, key=lambda r: r.actual_speedup)
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append(f"| total shapes | **{len(rows)}** |")
    L.append(f"| models | "
             f"**{len(sorted({r.model for r in rows}))}** "
             f"({', '.join(sorted({r.model for r in rows}))}) |")
    L.append(f"| batch sizes (T) | "
             f"{sorted({r.T for r in rows})} |")
    L.append(f"| median speedup vs FP16 | "
             f"**{statistics.median(spd_all):.3f}×** |")
    L.append(f"| mean speedup vs FP16 | "
             f"**{statistics.mean(spd_all):.3f}×** |")
    L.append(f"| wins (≥ 1.00×) | "
             f"**{w10} / {len(rows)}** ({w10*100//len(rows)} %) |")
    L.append(f"| clear wins (≥ 1.10×) | {w11} / {len(rows)} |")
    L.append(f"| big wins (≥ 2.00×) | **{w20} / {len(rows)}** |")
    L.append(f"| losses (< 0.90×) | {lo} / {len(rows)} |")
    L.append(f"| peak speedup | "
             f"**{best.actual_speedup:.2f}×** — "
             f"{best.model} {best.proj} T={best.T} "
             f"({best.d_in}→{best.d_out}) |")
    L.append(f"| worst speedup | {worst.actual_speedup:.2f}× — "
             f"{worst.model} {worst.proj} T={worst.T} "
             f"({worst.d_in}→{worst.d_out}) |")
    L.append(f"| median INT4 eff (cuda_eff) | "
             f"{statistics.median(ceff_all)*100:.1f}% |")
    L.append(f"| peak INT4 eff | {max(ceff_all)*100:.1f}% |")
    L.append(f"| median FP16 eff (cuBLAS vs its own roof) | "
             f"{statistics.median(feff_all)*100:.1f}% |")
    L.append("")

    # Per-model table (only if >1 model)
    models = sorted({r.model for r in rows})
    if len(models) > 1:
        L.append("### §0.1 Per-model scaling (ordered by parameter count)")
        L.append("")
        L.append("| model | params | N | median | mean | wins | peak |")
        L.append("|:---|---:|---:|---:|---:|---:|---:|")
        for m in sorted(models, key=model_sort_key):
            mrows = [r for r in rows if r.model == m]
            msp = [r.actual_speedup for r in mrows]
            params = PARAMS_B.get(m)
            p_str = f"{params:.1f}B" if params is not None else "—"
            mwin = sum(1 for s in msp if s >= 1.0)
            L.append(
                f"| {m} | {p_str} | {len(msp)} | "
                f"{statistics.median(msp):.2f}× | "
                f"{statistics.mean(msp):.2f}× | "
                f"{mwin} / {len(msp)} | "
                f"{max(msp):.2f}× |"
            )
        L.append("")

    # Per-T roll-up (quick companion to §3.1 below)
    L.append("### §0.2 Per-T roll-up (across all models)")
    L.append("")
    L.append("| T | N | median | mean | wins |")
    L.append("|---:|---:|---:|---:|---:|")
    by_T = {}
    for r in rows:
        by_T.setdefault(r.T, []).append(r.actual_speedup)
    for T in sorted(by_T):
        sp = by_T[T]
        wn = sum(1 for s in sp if s >= 1.0)
        L.append(f"| {T} | {len(sp)} | "
                 f"{statistics.median(sp):.2f}× | "
                 f"{statistics.mean(sp):.2f}× | "
                 f"{wn} / {len(sp)} |")
    L.append("")

    # §1
    L.append("## §1 Hardware constants and formulas")
    L.append("")
    L.append("| Parameter | Value | Note |")
    L.append("| --- | --- | --- |")
    L.append("| HBM bandwidth | 1008 GB/s | RTX 4090 vendor spec |")
    L.append("| FP16/BF16 TC peak | 165.2 TFLOPS | boost clock, no sparsity |")
    L.append("| INT4 TC peak | 660.6 TOPS | boost clock |")
    L.append("| ACHIEVABLE_FRACTION | 0.85 | engineering derating |")
    L.append("")
    L.append("**Formulas** (all time in us, eff_* = peak × ACHIEVABLE_FRACTION):")
    L.append("")
    L.append("- **FP16 roofline**: `t = max(2·T·d_in·d_out / eff_fp16, "
             "(2·d_in·d_out + 2·T·d_in + 2·T·d_out) / eff_hbm)`")
    L.append("- **CUDA quant (T>=2)**: mem-bound, "
             "`bytes = 2·T·d_in + 0.5·T·d_in + 2·T + 4·T·n_groups`")
    L.append("- **CUDA GEMM (T>=2)**: "
             "`t = max(2·T·d_in·d_out / eff_int4, bytes_gemm / eff_hbm)` "
             "with `bytes_gemm = 0.5·d_in·d_out + 0.5·T·d_in + 4·d_out·n_g "
             "+ 4·T·n_g + 2·T + 2·T·d_out`")
    L.append("- **CUDA T=1 fused**: "
             "`bytes = 0.5·d_in·d_out + 2·d_in + 4·d_out·n_g + 2·d_out`, "
             "compute = `2·d_in·d_out / eff_int4`")
    L.append("- **CUDA e2e (T>=2)** = `t_quant + t_gemm` (serial)")
    L.append("")
    L.append("### Systematic biases")
    L.append("1. Kernel launch overhead (5-10us/launch) not in the roofline — "
             "T<=8 rows will under-estimate achievable time.")
    L.append("2. L2 cache reuse — in the cold-cache bench we explicitly "
             "L2-flush the FP16 side before each sample, so FP16 efficiency "
             "numbers here are honest (no warm-cache cheating).")
    L.append("3. Epilogue FMA cost (dequant) folded into the INT4 TC peak — "
             "slightly optimistic.")
    L.append("")

    # §2 FP16 eff distribution by T
    L.append("## §2 FP16 efficiency distribution (by T)")
    L.append("")
    L.append("`fp16_eff = fp16_roof_us / fp16_us` — how close cuBLAS "
             "(cold-cache) gets to the RTX 4090 physical limit.")
    L.append("")
    by_T: Dict[int, List[float]] = {}
    for r in rows:
        by_T.setdefault(r.T, []).append(r.fp16_eff)
    pairs = [(str(k), v) for k, v in sorted(by_T.items())]
    L.append(_dist_table(pairs))

    # §3 CUDA eff distribution
    L.append("## §3 CUDA efficiency distribution")
    L.append("")
    L.append("`cuda_eff = cuda_roof_us / cuda_us` — how close our W4A4 "
             "kernel gets to its own roofline.")
    L.append("")
    L.append("### §3.1 By T")
    L.append("")
    by_T_cuda: Dict[int, List[float]] = {}
    for r in rows:
        by_T_cuda.setdefault(r.T, []).append(r.cuda_eff)
    pairs = [(str(k), v) for k, v in sorted(by_T_cuda.items())]
    L.append(_dist_table(pairs))

    L.append("### §3.2 By proj")
    L.append("")
    by_proj: Dict[str, List[float]] = {}
    for r in rows:
        by_proj.setdefault(r.proj, []).append(r.cuda_eff)
    pairs = [(k, v) for k, v in by_proj.items()]
    L.append(_dist_table(pairs))

    # §4 per-shape detail
    L.append("## §4 Per-shape detail")
    L.append("")
    L.append("Per-row: measured time, roofline time, efficiency, and "
             "actual-vs-roof speedup against FP16.")
    L.append("")
    models = sorted({r.model for r in rows}, key=model_sort_key)
    for m in models:
        L.append(f"### {m}")
        L.append("")
        L.append("| proj | T | shape | fp16_us | fp16_roof | fp16_eff | "
                 "cuda_us | cuda_roof | cuda_eff | speedup (actual / roof) |")
        L.append("|:---|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|")
        mrows = [r for r in rows if r.model == m]
        # Sort by canonical proj order then T
        proj_ord = {p: i for i, p in enumerate(
            ["q_proj", "kv_proj", "o_proj", "gate_up_proj", "down_proj"])}
        mrows.sort(key=lambda r: (proj_ord.get(r.proj, 99), r.T))
        for r in mrows:
            L.append(
                f"| {r.proj} | {r.T} | {r.d_in}→{r.d_out} | "
                f"{r.fp16_us:.2f} | {r.fp16_roof:.2f} | {r.fp16_eff*100:.0f}% | "
                f"{r.cuda_us:.2f} | {r.cuda_roof:.2f} | {r.cuda_eff*100:.0f}% | "
                f"{r.actual_speedup:.2f}× ({r.roof_speedup:.2f}×) |"
            )
        L.append("")

    # §5 worst cuda_eff
    L.append("## §5 CUDA implementation-gap TOP-15 (worst cuda_efficiency)")
    L.append("")
    sorted_rows = sorted(rows, key=lambda r: r.cuda_eff)
    L.append("| rank | model | proj | T | shape | cuda_us | cuda_roof | "
             "cuda_eff | gemm_bound |")
    L.append("|---:|:---|:---|---:|:---:|---:|---:|---:|:---:|")
    for i, r in enumerate(sorted_rows[:15], 1):
        L.append(
            f"| {i} | {r.model} | {r.proj} | {r.T} | {r.d_in}→{r.d_out} | "
            f"{r.cuda_us:.2f} | {r.cuda_roof:.2f} | {r.cuda_eff*100:.0f}% | "
            f"{r.gemm_bound} |"
        )
    L.append("")

    # §6 physics gap
    L.append("## §6 Physics gap — `cuda_roof / fp16_roof`")
    L.append("")
    L.append("Rows where `cuda_roof / fp16_roof > 1.0` are shapes where "
             "**W4A4 cannot beat FP16 even at the physical limit** — these "
             "should route to FP16 via policy, no kernel work can rescue them.")
    L.append("")
    phys_lose = sum(1 for r in rows if r.roof_speedup < 1.0)
    L.append(f"- Rows with cuda_roof faster than fp16_roof (W4A4 can win at ceiling): "
             f"**{sum(1 for r in rows if r.roof_speedup >= 1.0)} / {len(rows)}**")
    L.append(f"- Rows where fp16_roof is faster (W4A4 loses at ceiling): "
             f"**{phys_lose} / {len(rows)}**")
    L.append("")

    # §7 conclusions
    L.append("## §7 Conclusions")
    L.append("")
    cnt_80 = sum(1 for r in rows if r.cuda_eff >= 0.80)
    cnt_50 = sum(1 for r in rows if r.cuda_eff < 0.50)
    cnt_actual_win = sum(1 for r in rows if r.actual_speedup >= 1.0)
    cnt_actual_lose = sum(1 for r in rows if r.actual_speedup < 1.0)
    roof_wins = sum(1 for r in rows if r.roof_speedup >= 1.0)
    gap_shapes = roof_wins - cnt_actual_win
    L.append(f"- Out of **{len(rows)}** shapes, **{cnt_80}** reach "
             f"`cuda_eff >= 0.80` — already near the physical limit.")
    L.append(f"- **{cnt_50}** shapes sit at `cuda_eff < 0.50` — real "
             f"implementation slack remains (§5 top offenders).")
    L.append(f"- **{phys_lose}** shapes have `cuda_roof >= fp16_roof` — "
             f"W4A4 loses at the ceiling; these must route to FP16 via "
             f"policy, no kernel work can rescue them.")
    L.append(f"- Measured today: **{cnt_actual_win} / {len(rows)}** shapes "
             f"actually beat FP16; **{cnt_actual_lose}** lose.  Of the "
             f"losing shapes, **{max(cnt_actual_lose - phys_lose, 0)}** are "
             f"*implementation-gap* losses (fixable) and **"
             f"{phys_lose}** are *physics* losses (unfixable).")
    L.append("")

    # Aggregate summary for quick glance
    spd = [r.actual_speedup for r in rows]
    ceffs = [r.cuda_eff for r in rows]
    L.append("### Aggregate stats (measured)")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---:|")
    L.append(f"| shapes | {len(rows)} |")
    L.append(f"| median speedup vs FP16 | {statistics.median(spd):.3f}× |")
    L.append(f"| mean speedup vs FP16 | {statistics.mean(spd):.3f}× |")
    L.append(f"| wins (≥ 1.00×) | {cnt_actual_win} / {len(rows)} |")
    L.append(f"| clear wins (≥ 1.10×) | "
             f"{sum(1 for s in spd if s >= 1.10)} / {len(rows)} |")
    L.append(f"| median INT4 efficiency | "
             f"{statistics.median(ceffs)*100:.1f}% |")
    L.append(f"| max INT4 efficiency | {max(ceffs)*100:.1f}% |")
    L.append("")

    return "\n".join(L)


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--bench-json", type=Path, required=True,
                   help="Path to bench.json produced by bench_qwen3_shapes")
    p.add_argument("--output", type=Path, required=True,
                   help="Output markdown path")
    p.add_argument("--title", default="r62 F2 Qwen3 roofline report")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    rows = extract_rows(args.bench_json)
    if not rows:
        raise SystemExit("No end_to_end records found in bench.json")
    text = render_report(rows, source=str(args.bench_json), title=args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote {args.output}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
