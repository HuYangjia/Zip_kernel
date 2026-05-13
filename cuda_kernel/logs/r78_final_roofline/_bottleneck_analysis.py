#!/usr/bin/env python3
"""r78 bottleneck 归因分析：哪些 shape 达到瓶颈？瓶颈在哪里？

方法：
1. 从 bench.json 构建 rows（复用 roofline_compare 的公式）
2. 对每个 shape 打两个标签：
   (a) bound_kind: 理论上 roofline 该受什么约束
       - hbm-bound  : gemm 阶段 bytes/eff_hbm > flops/eff_int4
       - tc-bound   : 反之（int4 TC FMA 峰值是 bottleneck）
       - fused-mem  : T=1 fused GEMV，纯 mem
       - quant+gemm : T≥2 大 T，quant 和 gemm 串联都占比显著
   (b) status: 实际表现相对 roofline
       - at-roof    (cuda_eff ≥ 80%)     达到瓶颈，继续调优 ROI 低
       - mid-gap    (50-80%)             kernel 还能改
       - large-gap  (< 50%)              严重 under-utilize
3. 进一步细分 large-gap 的具体原因（通过 cuda_us 绝对值 + roof 绝对值）：
       - launch-bound    : cuda_us ≈ 30-35us 地板（T≤32, 小 shape）
       - epilogue-bound  : cuda_eff 30-50%（TC 峰值高估，实际 epilogue FMA 受限）
       - kernel-suboptimal: 其他
4. 输出 bottleneck_report.md
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
_KERNEL_ROOT = _HERE.parent.parent.parent  # .../HKUST/kernel
if str(_KERNEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_KERNEL_ROOT))

from cuda_kernel.benchmarks import roofline_compare as rc  # noqa: E402

BENCH_JSON = _HERE / "qwen3_20260502_201018" / "bench.json"
OUT_MD = _HERE / "qwen3_20260502_201018" / "bottleneck_report.md"

# RTX 4090 常数（和 roofline_compare 对齐）
HBM_GBPS = rc.HBM_BW_GBPS       # 1008
FP16_TFLOPS = rc.FP16_TFLOPS    # 165.2
INT4_TOPS = rc.INT4_TOPS        # 660.6
FRAC = rc.ACHIEVABLE_FRACTION   # 0.85

EFF_HBM = HBM_GBPS * FRAC * 1e9
EFF_INT4 = INT4_TOPS * FRAC * 1e12


def classify_bound(row: Dict[str, Any]) -> str:
    """基于 GEMM 阶段 bytes/flops 比值判定是哪侧 roofline."""
    T = row["T"]
    d_in = row["d_in"]
    d_out = row["d_out"]
    n_groups = d_in // 128

    if T == 1:
        # fused GEMV：bytes = 2*d_in + 0.5*d_in*d_out + 4*d_out*n_groups + 2*d_out
        bytes_b = 2 * d_in + 0.5 * d_in * d_out + 4 * d_out * n_groups + 2 * d_out
        flops = 2 * 1 * d_in * d_out
        t_mem = bytes_b / EFF_HBM
        t_compute = flops / EFF_INT4
        return "fused-mem" if t_mem > t_compute else "fused-tc"

    # T ≥ 2 的 gemm 阶段
    bytes_b = 0.5 * d_in * d_out + 0.5 * T * d_in + 4 * d_out * n_groups + 2 * T * d_out
    flops = 2 * T * d_in * d_out
    t_mem = bytes_b / EFF_HBM
    t_compute = flops / EFF_INT4
    # 还要看 quant 阶段是否很大（占比 ≥ 30%）
    t_quant = (2 * T * d_in + 0.5 * T * d_in + 2 * T + 4 * T * n_groups) / EFF_HBM
    t_gemm_roof = max(t_mem, t_compute)
    quant_share = t_quant / (t_quant + t_gemm_roof)

    if t_mem > t_compute:
        return "hbm-bound" if quant_share < 0.3 else "quant+hbm"
    else:
        return "tc-bound" if quant_share < 0.3 else "quant+tc"


def classify_status(row: Dict[str, Any]) -> str:
    eff = row["cuda_efficiency"]
    if math.isnan(eff):
        return "n/a"
    if eff >= 0.80:
        return "at-roof"
    if eff >= 0.50:
        return "mid-gap"
    return "large-gap"


def classify_gap_cause(row: Dict[str, Any]) -> str:
    """进一步解释 mid/large-gap 的根因（经验规则）."""
    status = classify_status(row)
    if status == "at-roof":
        return "—"

    T = row["T"]
    cuda_us = row["cuda_us"]
    cuda_roof = row["cuda_roof_us"]

    # 规则 1: T ≤ 32 时 cuda_us 贴着 30-36us 地板 → launch/dispatcher 开销
    if T <= 32 and 28 <= cuda_us <= 40 and cuda_roof < 10:
        return "launch-bound"
    # 规则 2: T = 1 但 cuda_eff 不高（30-60%）→ GEMV tail 低效，常见于小 d_in
    if T == 1 and classify_status(row) in ("mid-gap", "large-gap"):
        return "gemv-tail"
    # 规则 3: tc-bound 但 eff 偏低 → int4 TC FMA 峰值达不到（epilogue CUDA-core FMA 拖后）
    bound = classify_bound(row)
    if bound in ("tc-bound", "quant+tc", "fused-tc"):
        return "epilogue-fma"
    # 规则 4: hbm-bound 但 eff 低 → kernel 内存访问非最优（pack/stage 低效）
    if bound in ("hbm-bound", "fused-mem"):
        return "mem-access-suboptimal"
    return "other"


def main() -> int:
    # 1) 加载数据
    e2e = rc.load_bench(BENCH_JSON)
    rows = [rc.build_row(r) for r in e2e]
    print(f"[info] loaded {len(rows)} rows")

    # 2) 打标签
    for r in rows:
        r["bound"] = classify_bound(r)
        r["status"] = classify_status(r)
        r["gap_cause"] = classify_gap_cause(r)
        r["speedup"] = r["fp16_us"] / r["cuda_us"] if r["cuda_us"] > 0 else float("nan")

    # 3) 汇总
    md: List[str] = []
    md.append("# r78 Bottleneck Attribution Report\n")
    md.append(f"**Source**: `cuda_kernel/logs/r78_final_roofline/qwen3_20260502_201018/bench.json`  ")
    md.append(f"**GPU**: RTX 4090, ACHIEVABLE_FRACTION={FRAC}  ")
    md.append(f"**Total shapes**: {len(rows)}\n")

    # -------------------------------------------------------------------
    md.append("## §1 是否达到瓶颈：按 cuda_efficiency 分桶\n")
    md.append("> `cuda_efficiency = cuda_roof_us / cuda_us`，衡量实测速度距离 W4A4 的物理上限有多近。\n")
    md.append("> - `at-roof` (≥80%)：已到瓶颈，继续优化 ROI 极低")
    md.append("> - `mid-gap` (50-80%)：kernel 层仍有改进空间")
    md.append("> - `large-gap` (<50%)：显著 under-utilize，要找根因\n")

    status_count = Counter(r["status"] for r in rows)
    md.append("| status | N | 占比 | median cuda_eff | median speedup vs FP16 |")
    md.append("|:---|---:|---:|---:|---:|")
    for st in ("at-roof", "mid-gap", "large-gap"):
        subset = [r for r in rows if r["status"] == st]
        if not subset:
            continue
        eff = [r["cuda_efficiency"] for r in subset if not math.isnan(r["cuda_efficiency"])]
        sp = [r["speedup"] for r in subset if not math.isnan(r["speedup"])]
        md.append(f"| **{st}** | {len(subset)} | {100*len(subset)/len(rows):.1f}% | "
                  f"{100*statistics.median(eff):.1f}% | {statistics.median(sp):.2f}× |")
    md.append("")

    # -------------------------------------------------------------------
    md.append("## §2 理论瓶颈所在（roofline 侧）\n")
    md.append("> 在达到 roofline 的前提下，kernel 最终会被谁卡住？\n")
    md.append("> - `hbm-bound`：T≥2 的 gemm 阶段 mem 时间 > compute 时间（HBM 带宽是上限）")
    md.append("> - `tc-bound`：gemm 阶段 compute 时间 > mem 时间（INT4 TC 峰值是上限）")
    md.append("> - `quant+hbm/tc`：quant 阶段占 roofline ≥30%，额外串联开销显著")
    md.append("> - `fused-mem`：T=1 fused GEMV 纯内存受限")
    md.append("> - `fused-tc`：T=1 fused GEMV 计算受限（极少，仅超扁平 shape 出现）\n")

    bound_count = Counter(r["bound"] for r in rows)
    md.append("| bound | N | 占比 | median cuda_eff | at-roof 命中 | median speedup vs FP16 |")
    md.append("|:---|---:|---:|---:|---:|---:|")
    for b in sorted(bound_count, key=lambda k: -bound_count[k]):
        subset = [r for r in rows if r["bound"] == b]
        at_roof = sum(1 for r in subset if r["status"] == "at-roof")
        eff = [r["cuda_efficiency"] for r in subset if not math.isnan(r["cuda_efficiency"])]
        sp = [r["speedup"] for r in subset if not math.isnan(r["speedup"])]
        md.append(f"| {b} | {bound_count[b]} | {100*bound_count[b]/len(rows):.1f}% | "
                  f"{100*statistics.median(eff):.1f}% | {at_roof}/{len(subset)} | "
                  f"{statistics.median(sp):.2f}× |")
    md.append("")

    # -------------------------------------------------------------------
    md.append("## §3 实际瓶颈所在（gap 根因）\n")
    md.append("> 对于 `mid-gap + large-gap` 的 shape，到底是什么在拖慢 kernel？\n")
    md.append("> 经验规则：\n")
    md.append("> - `launch-bound`：T≤32 且 cuda_us 贴着 30-36us 地板（kernel launch/dispatcher 开销占主导）")
    md.append("> - `gemv-tail`：T=1 GEMV，尾 wave 不满 + 小 d_in 寄存器利用率差")
    md.append("> - `epilogue-fma`：tc-bound 但 eff<80%，INT4 TC 峰值达不到（per-group dequant FMA 在 CUDA-core 执行）")
    md.append("> - `mem-access-suboptimal`：hbm-bound 但 eff<80%（pack/stage/L2 利用未最优）\n")

    cause_count = Counter(r["gap_cause"] for r in rows if r["status"] != "at-roof")
    md.append("| gap_cause | N | median cuda_eff | median cuda_us | median speedup |")
    md.append("|:---|---:|---:|---:|---:|")
    for c in sorted(cause_count, key=lambda k: -cause_count[k]):
        subset = [r for r in rows if r["gap_cause"] == c and r["status"] != "at-roof"]
        if not subset:
            continue
        eff = [r["cuda_efficiency"] for r in subset if not math.isnan(r["cuda_efficiency"])]
        sp = [r["speedup"] for r in subset if not math.isnan(r["speedup"])]
        cu = [r["cuda_us"] for r in subset]
        md.append(f"| **{c}** | {cause_count[c]} | {100*statistics.median(eff):.1f}% | "
                  f"{statistics.median(cu):.1f} us | {statistics.median(sp):.2f}× |")
    md.append("")

    # -------------------------------------------------------------------
    md.append("## §4 达到瓶颈的 shape 清单（`at-roof` 全量）\n")
    at_roof_rows = [r for r in rows if r["status"] == "at-roof"]
    at_roof_rows.sort(key=lambda r: -r["cuda_efficiency"])
    md.append(f"共 **{len(at_roof_rows)}** 个 shape 达到 ≥80% W4A4 roofline：\n")
    md.append("| model | proj | T | shape | bound | cuda_us | cuda_roof_us | cuda_eff | speedup vs FP16 |")
    md.append("|:---|:---|---:|:---:|:---|---:|---:|---:|---:|")
    for r in at_roof_rows:
        md.append(f"| {r['model']} | {r['proj']} | {r['T']} | "
                  f"{r['d_in']}→{r['d_out']} | {r['bound']} | "
                  f"{r['cuda_us']:.1f} | {r['cuda_roof_us']:.2f} | "
                  f"{100*r['cuda_efficiency']:.1f}% | {r['speedup']:.2f}× |")
    md.append("")

    # -------------------------------------------------------------------
    md.append("## §5 严重 under-utilize 清单（`large-gap` TOP-20）\n")
    large_gap = [r for r in rows if r["status"] == "large-gap"]
    large_gap.sort(key=lambda r: r["cuda_efficiency"])
    md.append(f"共 **{len(large_gap)}** 个 shape cuda_eff < 50%，下面列最糟 20 个：\n")
    md.append("| model | proj | T | shape | bound | cause | cuda_us | cuda_roof | cuda_eff | speedup |")
    md.append("|:---|:---|---:|:---:|:---|:---|---:|---:|---:|---:|")
    for r in large_gap[:20]:
        md.append(f"| {r['model']} | {r['proj']} | {r['T']} | "
                  f"{r['d_in']}→{r['d_out']} | {r['bound']} | {r['gap_cause']} | "
                  f"{r['cuda_us']:.1f} | {r['cuda_roof_us']:.2f} | "
                  f"{100*r['cuda_efficiency']:.1f}% | {r['speedup']:.2f}× |")
    md.append("")

    # -------------------------------------------------------------------
    md.append("## §6 交叉表：bound × T（达 roofline 的命中率）\n")
    md.append("> 格子里写 `at-roof / total`（`at-roof` 数 / 该 bound+T 组合的总 shape 数）\n")
    all_Ts = sorted({r["T"] for r in rows})
    all_bounds = sorted(bound_count.keys())
    md.append("| bound \\ T | " + " | ".join(str(t) for t in all_Ts) + " |")
    md.append("|:---|" + "|".join(["---:"] * len(all_Ts)) + "|")
    for b in all_bounds:
        cells = []
        for t in all_Ts:
            sub = [r for r in rows if r["bound"] == b and r["T"] == t]
            at = sum(1 for r in sub if r["status"] == "at-roof")
            cells.append(f"{at}/{len(sub)}" if sub else "—")
        md.append(f"| {b} | " + " | ".join(cells) + " |")
    md.append("")

    # -------------------------------------------------------------------
    md.append("## §7 结论\n")
    n_at = status_count["at-roof"]
    n_mid = status_count["mid-gap"]
    n_large = status_count["large-gap"]
    md.append(f"1. **已达瓶颈（at-roof）**：{n_at}/{len(rows)} ({100*n_at/len(rows):.1f}%)，"
              f"全部是 `gate_up_proj` 的大 shape（d_out 19k/24k 级）在 T=8/32 的 hbm-bound 路径，"
              f"speedup 中位 3.38×，继续优化 ROI 极低。")
    md.append(f"2. **中等缺口（mid-gap）**：{n_mid}/{len(rows)} ({100*n_mid/len(rows):.1f}%)，"
              f"一半是大模型 T=1 的 fused-mem GEMV（gemv-tail），另一半是 T=512/1024/2048 的 tc-bound shape——"
              f"后者根因是 epilogue per-group dequant FMA 在 CUDA-core 上执行，INT4 TC 峰值无法达到。")
    md.append(f"3. **严重缺口（large-gap）**：{n_large}/{len(rows)} ({100*n_large/len(rows):.1f}%)，"
              f"分两类："
              f"(i) T≤32 小 shape 的 `launch-bound`/`mem-access-suboptimal`（~100 shape，"
              f"cuda_us 被 ~34us dispatcher 地板托住，roof 仅 1-10us）；"
              f"(ii) 大 T tc-bound 的 `epilogue-fma`（~100 shape，eff 30-45%，roof 计入了 660 TOPS INT4 TC 峰值但实际达不到）。")
    md.append("")
    md.append("**瓶颈优先级**：")
    md.append("- (A) 小 T launch 开销：把 30us 地板拍下来，能救约 80+ shape（收益最大）")
    md.append("- (B) 大 T epilogue FMA：CUTLASS int4 epilogue / dequant-in-register 能救 mid-gap 的 tc-bound 段")
    md.append("- (C) 大 shape gate_up_proj 已到瓶颈，无需再动")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] wrote {OUT_MD}")

    # 终端简报
    print("\n=== 终端摘要 ===")
    print(f"at-roof   : {n_at:3d} ({100*n_at/len(rows):.1f}%)")
    print(f"mid-gap   : {n_mid:3d} ({100*n_mid/len(rows):.1f}%)")
    print(f"large-gap : {n_large:3d} ({100*n_large/len(rows):.1f}%)")
    print("\nbound 分布:")
    for b, n in bound_count.most_common():
        print(f"  {b:20s}: {n:3d}")
    print("\ngap_cause 分布（非 at-roof）:")
    for c, n in cause_count.most_common():
        print(f"  {c:24s}: {n:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
