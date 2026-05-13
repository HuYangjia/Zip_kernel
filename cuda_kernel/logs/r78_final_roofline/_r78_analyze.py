#!/usr/bin/env python3
"""r78_final_roofline: r78 专用 bench + roofline 分析器

功能：
- 读 bench.json（--full 跑出来的 7 model × 5 proj × 7 T = 245 shape）
- §0 Executive Summary（全局 + per-model + per-T + per-proj 滚动）
- §0.x 挑出 top-wins / top-losses / ceiling-slower 子集
- 委托 `cuda_kernel.benchmarks.roofline_compare` 生成 §1-§7 详细分析
- 输出：
    r78_summary.md           -- §0 exec summary + 决策表
    roofline_report.md       -- 详细 roofline 分析（§1-§7）
    roofline_compare.csv     -- 逐行数据
    r78_SUMMARY.md           -- 最终复盘文档（人工 review 用）

r78 相对 r63_combined 的扩展：
- MODEL_ORDER 增加 Qwen2.5-32B / LLaMA3-70B
- T_ORDER 增加 32, 2048（完整 7 点：1,8,32,128,512,1024,2048）
- 滤掉 Triton 数据（CUDA-only production path, see memory 0d5nyof1）
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 允许直接 python _r78_analyze.py 运行（不必 python -m ...）
_HERE = Path(__file__).resolve().parent
_KERNEL_ROOT = _HERE.parent.parent.parent  # .../HKUST/kernel
if str(_KERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_KERNEL_ROOT))

from cuda_kernel.benchmarks import roofline_compare as rc  # noqa: E402


# r78 扩展的顺序
MODEL_ORDER_R78: Tuple[str, ...] = (
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
    "Qwen3-14B", "Qwen2.5-32B", "LLaMA3-70B",
)
PROJ_ORDER_R78: Tuple[str, ...] = (
    "q_proj", "kv_proj", "o_proj", "gate_up_proj", "down_proj",
)
T_ORDER_R78: Tuple[int, ...] = (1, 8, 32, 128, 512, 1024, 2048)

MODEL_PARAMS_B: Dict[str, float] = {
    "Qwen3-0.6B": 0.6, "Qwen3-1.7B": 1.7, "Qwen3-4B": 4.0,
    "Qwen3-8B": 8.0, "Qwen3-14B": 14.0, "Qwen2.5-32B": 32.0,
    "LLaMA3-70B": 70.0,
}


# ---------------------------------------------------------------------------
# 1) 预处理：把 r78 扩展的顺序 monkey-patch 到 roofline_compare，
#    这样调用 rc.render_report 就能渲染 32B/70B/T=2048 的行
# ---------------------------------------------------------------------------
def _patch_rc_orders() -> None:
    rc.MODEL_ORDER = MODEL_ORDER_R78
    rc.PROJ_ORDER = PROJ_ORDER_R78
    rc.T_ORDER = T_ORDER_R78


# ---------------------------------------------------------------------------
# 2) 统计工具
# ---------------------------------------------------------------------------
def _median(xs: List[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.median(xs) if xs else float("nan")


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.mean(xs) if xs else float("nan")


def _pct(xs: List[float], p: float) -> float:
    xs = sorted([x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)])
    if not xs:
        return float("nan")
    k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * p))))
    return xs[k]


def _fmt_speedup(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x:.2f}×"


def _fmt_eff(x: float) -> str:
    if isinstance(x, float) and math.isnan(x):
        return "n/a"
    return f"{x * 100:.1f}%"


# ---------------------------------------------------------------------------
# 3) §0 Executive Summary 渲染
# ---------------------------------------------------------------------------
def render_exec_summary(rows: List[Dict[str, Any]],
                        bench_json: Path,
                        freeze_commit: str) -> str:
    md: List[str] = []
    md.append("# r78 Final Roofline & Bench Report\n")
    md.append(f"**Source**: `{bench_json}`  ")
    md.append(f"**GPU**: NVIDIA RTX 4090 (RTX 4090 vendor spec, "
              f"ACHIEVABLE_FRACTION={rc.ACHIEVABLE_FRACTION:.2f})  ")
    md.append(f"**Freeze commit**: `{freeze_commit}`  ")
    md.append(f"**Kernel**: fused_quant_dense_sparse_mma_int4 (production), "
              f"`v9-final-dispatcher` dispatch policy  ")
    md.append(f"**Baseline**: FP16 cuBLAS + L2-flush (cold-cache, realistic inference)  ")
    md.append("**CUDA-only** evaluation (Triton path deprecated per project policy)\n")

    # ---- §0 Global ----
    md.append("## §0 Executive Summary\n")

    speedup = [r["cuda_vs_fp16_actual"] for r in rows]
    cuda_eff = [r["cuda_efficiency"] for r in rows]
    fp16_eff = [r["fp16_efficiency"] for r in rows]
    valid_speedup = [s for s in speedup if not math.isnan(s)]

    n_wins = sum(1 for s in valid_speedup if s >= 1.00)
    n_clear_wins = sum(1 for s in valid_speedup if s >= 1.10)
    n_big_wins = sum(1 for s in valid_speedup if s >= 2.00)
    n_losses = sum(1 for s in valid_speedup if s < 0.90)

    # speedup inverse: 原 rows 里是 cuda_us / fp16_us（越小越好），
    # 我们要的 speedup = fp16_us / cuda_us（越大越好），重新算
    real_speedup = []
    for r in rows:
        if r["fp16_us"] > 0 and r["cuda_us"] > 0:
            real_speedup.append(r["fp16_us"] / r["cuda_us"])
    n_wins = sum(1 for s in real_speedup if s >= 1.00)
    n_clear_wins = sum(1 for s in real_speedup if s >= 1.10)
    n_big_wins = sum(1 for s in real_speedup if s >= 2.00)
    n_losses = sum(1 for s in real_speedup if s < 0.90)

    peak_row = max(rows, key=lambda r: (r["fp16_us"] / r["cuda_us"])
                   if r["cuda_us"] > 0 else 0)
    worst_row = min(rows, key=lambda r: (r["fp16_us"] / r["cuda_us"])
                    if r["cuda_us"] > 0 else float("inf"))

    # ceiling-slower = W4A4 roofline 比 fp16 roofline 慢的 shape 数
    n_ceiling_slower = sum(1 for r in rows
                           if not math.isnan(r["cuda_vs_fp16_roofline"])
                           and r["cuda_vs_fp16_roofline"] > 1.0)

    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Total shapes | **{len(rows)}** |")
    md.append(f"| Models | **{len(MODEL_ORDER_R78)}** "
              f"({', '.join(MODEL_ORDER_R78)}) |")
    md.append(f"| Batch sizes T | {list(T_ORDER_R78)} |")
    md.append(f"| Projections | {list(PROJ_ORDER_R78)} |")
    md.append(f"| hp_ratio (density) | 0.05 |")
    md.append(f"| **Median speedup vs FP16** | **{_fmt_speedup(_median(real_speedup))}** |")
    md.append(f"| Mean speedup vs FP16 | {_fmt_speedup(_mean(real_speedup))} |")
    md.append(f"| Wins (≥ 1.00×) | **{n_wins} / {len(rows)}** "
              f"({100 * n_wins / len(rows):.0f}%) |")
    md.append(f"| Clear wins (≥ 1.10×) | {n_clear_wins} / {len(rows)} |")
    md.append(f"| Big wins (≥ 2.00×) | **{n_big_wins} / {len(rows)}** |")
    md.append(f"| Losses (< 0.90×) | {n_losses} / {len(rows)} |")
    md.append(f"| Peak speedup | **{_fmt_speedup(peak_row['fp16_us'] / peak_row['cuda_us'])}** "
              f"— {peak_row['model']} {peak_row['proj']} T={peak_row['T']} "
              f"({peak_row['d_in']}→{peak_row['d_out']}) |")
    md.append(f"| Worst speedup | {_fmt_speedup(worst_row['fp16_us'] / worst_row['cuda_us'])} "
              f"— {worst_row['model']} {worst_row['proj']} T={worst_row['T']} "
              f"({worst_row['d_in']}→{worst_row['d_out']}) |")
    md.append(f"| Median CUDA efficiency (vs W4A4 roofline) | "
              f"**{_fmt_eff(_median(cuda_eff))}** |")
    md.append(f"| Peak CUDA efficiency | {_fmt_eff(max(x for x in cuda_eff if not math.isnan(x)))} |")
    md.append(f"| Median FP16 efficiency (cuBLAS vs its roofline) | {_fmt_eff(_median(fp16_eff))} |")
    md.append(f"| W4A4 ceiling slower than FP16 ceiling | {n_ceiling_slower} / {len(rows)} "
              f"({100 * n_ceiling_slower / len(rows):.0f}%) — unfixable by kernel work |")
    md.append("")

    # ---- §0.1 per-model roll-up ----
    md.append("### §0.1 Per-model scaling (ordered by parameter count)\n")
    md.append("| Model | Params | N | median | mean | wins | peak | median cuda_eff |")
    md.append("|:---|---:|---:|---:|---:|---:|---:|---:|")
    for m in MODEL_ORDER_R78:
        m_rows = [r for r in rows if r["model"] == m]
        if not m_rows:
            continue
        m_speedup = [r["fp16_us"] / r["cuda_us"] for r in m_rows if r["cuda_us"] > 0]
        m_eff = [r["cuda_efficiency"] for r in m_rows]
        m_wins = sum(1 for s in m_speedup if s >= 1.00)
        peak = max(m_speedup) if m_speedup else float("nan")
        md.append(f"| {m} | {MODEL_PARAMS_B.get(m, 0):.1f}B | {len(m_rows)} | "
                  f"{_fmt_speedup(_median(m_speedup))} | "
                  f"{_fmt_speedup(_mean(m_speedup))} | "
                  f"{m_wins} / {len(m_rows)} | "
                  f"{_fmt_speedup(peak)} | {_fmt_eff(_median(m_eff))} |")
    md.append("")

    # ---- §0.2 per-T roll-up ----
    md.append("### §0.2 Per-T roll-up (across all models / projs)\n")
    md.append("| T | N | median | mean | wins | p25 | p75 |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|")
    for T in T_ORDER_R78:
        t_rows = [r for r in rows if r["T"] == T]
        if not t_rows:
            continue
        t_speedup = [r["fp16_us"] / r["cuda_us"] for r in t_rows if r["cuda_us"] > 0]
        t_wins = sum(1 for s in t_speedup if s >= 1.00)
        md.append(f"| {T} | {len(t_rows)} | "
                  f"{_fmt_speedup(_median(t_speedup))} | "
                  f"{_fmt_speedup(_mean(t_speedup))} | "
                  f"{t_wins} / {len(t_rows)} | "
                  f"{_fmt_speedup(_pct(t_speedup, 0.25))} | "
                  f"{_fmt_speedup(_pct(t_speedup, 0.75))} |")
    md.append("")

    # ---- §0.3 per-proj roll-up ----
    md.append("### §0.3 Per-proj roll-up (across all models / T)\n")
    md.append("| proj | N | median | mean | wins | median cuda_eff |")
    md.append("|:---|---:|---:|---:|---:|---:|")
    for p in PROJ_ORDER_R78:
        p_rows = [r for r in rows if r["proj"] == p]
        if not p_rows:
            continue
        p_speedup = [r["fp16_us"] / r["cuda_us"] for r in p_rows if r["cuda_us"] > 0]
        p_eff = [r["cuda_efficiency"] for r in p_rows]
        p_wins = sum(1 for s in p_speedup if s >= 1.00)
        md.append(f"| {p} | {len(p_rows)} | "
                  f"{_fmt_speedup(_median(p_speedup))} | "
                  f"{_fmt_speedup(_mean(p_speedup))} | "
                  f"{p_wins} / {len(p_rows)} | "
                  f"{_fmt_eff(_median(p_eff))} |")
    md.append("")

    # ---- §0.4 TOP-10 wins / losses ----
    md.append("### §0.4 TOP-10 biggest wins (by fp16_us/cuda_us)\n")
    md.append("| rank | model | proj | T | shape | fp16_us | cuda_us | speedup | cuda_eff | bound |")
    md.append("|---:|:---|:---|---:|:---:|---:|---:|---:|---:|:---:|")
    top_wins = sorted(rows,
                      key=lambda r: (r["fp16_us"] / r["cuda_us"]
                                     if r["cuda_us"] > 0 else 0),
                      reverse=True)[:10]
    for i, r in enumerate(top_wins, 1):
        s = r["fp16_us"] / r["cuda_us"]
        md.append(f"| {i} | {r['model']} | {r['proj']} | {r['T']} | "
                  f"{r['d_in']}→{r['d_out']} | "
                  f"{r['fp16_us']:.1f} | {r['cuda_us']:.1f} | "
                  f"**{_fmt_speedup(s)}** | {_fmt_eff(r['cuda_efficiency'])} | "
                  f"{r['cuda_gemm_bound']} |")
    md.append("")

    md.append("### §0.5 TOP-10 biggest losses (smallest speedup)\n")
    md.append("| rank | model | proj | T | shape | fp16_us | cuda_us | speedup | cuda_eff | ceiling |")
    md.append("|---:|:---|:---|---:|:---:|---:|---:|---:|---:|---:|")
    top_losses = sorted(rows,
                        key=lambda r: (r["fp16_us"] / r["cuda_us"]
                                       if r["cuda_us"] > 0 else float("inf")))[:10]
    for i, r in enumerate(top_losses, 1):
        s = r["fp16_us"] / r["cuda_us"]
        ceiling_tag = "✗ unfixable" if r["cuda_vs_fp16_roofline"] > 1.0 else "fixable"
        md.append(f"| {i} | {r['model']} | {r['proj']} | {r['T']} | "
                  f"{r['d_in']}→{r['d_out']} | "
                  f"{r['fp16_us']:.1f} | {r['cuda_us']:.1f} | "
                  f"{_fmt_speedup(s)} | {_fmt_eff(r['cuda_efficiency'])} | "
                  f"{ceiling_tag} |")
    md.append("")

    # ---- §0.6 Kernel vs roofline 的工程 gap ----
    md.append("### §0.6 Implementation gap taxonomy\n")
    n_near_roof = sum(1 for r in rows
                      if not math.isnan(r["cuda_efficiency"])
                      and r["cuda_efficiency"] >= 0.80)
    n_mid = sum(1 for r in rows
                if not math.isnan(r["cuda_efficiency"])
                and 0.50 <= r["cuda_efficiency"] < 0.80)
    n_low = sum(1 for r in rows
                if not math.isnan(r["cuda_efficiency"])
                and r["cuda_efficiency"] < 0.50)
    n_actual_lose = sum(1 for r in rows
                        if not math.isnan(r["cuda_vs_fp16_actual"])
                        and r["cuda_vs_fp16_actual"] > 1.0)
    md.append(f"- **Near-roofline** (`cuda_eff ≥ 80%`): **{n_near_roof}** shapes — 已贴近物理上限，继续调优 ROI 低")
    md.append(f"- **Mid gap** (`50% ≤ cuda_eff < 80%`): {n_mid} shapes — 仍有 kernel 层面改进空间")
    md.append(f"- **Large gap** (`cuda_eff < 50%`): {n_low} shapes — 实现离 roofline 差距显著（多数是 launch-bound T=1/8）")
    md.append(f"- **Kernel-fixable losses**: measured_lose ({n_actual_lose}) − ceiling_slower ({n_ceiling_slower}) "
              f"= **{n_actual_lose - n_ceiling_slower}** shapes 理论上还能被更好的 kernel 救回")
    md.append(f"- **Physics-bound losses**: {n_ceiling_slower} shapes 已超出任何 W4A4 kernel 的物理极限，dispatcher 应 fallback FP16")
    md.append("")

    return "\n".join(md)


# ---------------------------------------------------------------------------
# 4) Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-dir", type=Path, required=True,
                    help="r78 bench 输出子目录，内含 bench.json")
    ap.add_argument("--freeze-commit", type=str, default="29159da",
                    help="kernel freeze commit hash")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing outputs")
    args = ap.parse_args()

    bench_json = args.bench_dir / "bench.json"
    if not bench_json.is_file():
        print(f"[FATAL] bench.json not found: {bench_json}", file=sys.stderr)
        return 2

    # 步骤 1: patch orders 到 rc
    _patch_rc_orders()

    # 步骤 2: load + 构 rows
    e2e = rc.load_bench(bench_json)
    print(f"[info] loaded {len(e2e)} end_to_end records")
    rows = [rc.build_row(r) for r in e2e]
    print(f"[info] built {len(rows)} roofline rows")

    # 步骤 3: 写 §0 Executive Summary
    summary_md = render_exec_summary(rows, bench_json, args.freeze_commit)
    summary_path = args.bench_dir / "r78_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"[ok] wrote exec summary → {summary_path}")

    # 步骤 4: 委托 rc 写详细 §1-§7 + CSV
    csv_path = args.bench_dir / "roofline_compare.csv"
    md_path = args.bench_dir / "roofline_report.md"
    rc.write_csv(rows, csv_path, args.force)
    print(f"[ok] wrote detail CSV → {csv_path}")
    n_detail = rc.render_report(rows, md_path, args.force, bench_json)
    print(f"[ok] wrote detail report → {md_path} ({n_detail} shapes)")

    # 步骤 5: 合并 summary + detail = r78_SUMMARY.md（人工 review 用）
    detail_md = md_path.read_text(encoding="utf-8")
    # 去掉 detail_md 的顶部 "# Roofline theoretical vs measured report" 标题
    # 和几行元信息，保留 §1 起的正文
    split_marker = "## §1"
    if split_marker in detail_md:
        detail_body = split_marker + detail_md.split(split_marker, 1)[1]
    else:
        detail_body = detail_md
    combined = (summary_md
                + "\n\n---\n\n"
                + "# Detailed Roofline Analysis\n\n"
                + detail_body)
    final_path = args.bench_dir / "r78_SUMMARY.md"
    final_path.write_text(combined, encoding="utf-8")
    print(f"[ok] wrote combined SUMMARY → {final_path}")

    # 自检
    if n_detail != len(e2e):
        print(f"[FAIL] detail row self-check: rendered={n_detail}, expected={len(e2e)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
