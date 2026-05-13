#!/usr/bin/env python3
"""r78 vs r63_combined comparison.

找到两次 bench 共同覆盖的 shape 集合 (model, proj, T, d_in, d_out)，
并对比 CUDA-side 的 cuda_us（kernel 本体进步）和 fp16_us（基线一致性，
应接近相等，如果相差 >5% 说明 4090 频率 / 上下文有变，要警惕）。

r63_combined: 140 shapes, T in {1, 32, 128, 512}, 7 models
r78:          245 shapes, T in {1, 8, 32, 128, 512, 1024, 2048}, 7 models

交集预期: 7 models × 5 proj × 4 T = 140 shapes
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def index(data):
    return {(r["model"], r["proj"], r["T"], r["d_in"], r["d_out"]): r
            for r in data["records"] if r.get("kernel") == "end_to_end"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r63", type=Path, required=True,
                    help="r63_combined bench.json path")
    ap.add_argument("--r78", type=Path, required=True,
                    help="r78_final_roofline bench.json path")
    ap.add_argument("--out", type=Path, required=True,
                    help="output markdown path")
    args = ap.parse_args()

    d63 = json.load(open(args.r63))
    d78 = json.load(open(args.r78))
    i63 = index(d63)
    i78 = index(d78)
    common = sorted(set(i63) & set(i78))
    print(f"r63 shapes: {len(i63)}, r78 shapes: {len(i78)}, common: {len(common)}")

    md = []
    md.append("# r63 → r78 regression & improvement report\n")
    md.append(f"**r63_combined**: {len(i63)} shapes  ")
    md.append(f"**r78_final_roofline**: {len(i78)} shapes  ")
    md.append(f"**Common overlap**: {len(common)} shapes  ")
    md.append("")
    md.append("## §1 FP16 baseline consistency\n")
    md.append("如果 cuBLAS 基线偏离 >5%，说明 GPU 频率 / 上下文变化，对比不可信。\n")

    fp16_deltas = []
    cuda_deltas = []
    cuda_speedup_delta = []   # 新 speedup - 旧 speedup
    per_proj_improve = defaultdict(list)
    per_t_improve = defaultdict(list)
    per_model_improve = defaultdict(list)

    for key in common:
        r_old = i63[key]
        r_new = i78[key]
        fp16_old = r_old["fp16_us"]
        fp16_new = r_new["fp16_us"]
        cuda_old = r_old["cuda_us"]
        cuda_new = r_new["cuda_us"]
        if fp16_old > 0:
            fp16_deltas.append((fp16_new - fp16_old) / fp16_old)
        if cuda_old > 0:
            cuda_deltas.append((cuda_new - cuda_old) / cuda_old)
            # speedup = fp16 / cuda, 变化
            s_old = fp16_old / cuda_old if cuda_old > 0 else float("nan")
            s_new = fp16_new / cuda_new if cuda_new > 0 else float("nan")
            d = s_new - s_old
            cuda_speedup_delta.append(d)
            per_proj_improve[key[1]].append(d)
            per_t_improve[key[2]].append(d)
            per_model_improve[key[0]].append(d)

    def _stats(xs):
        xs = [x for x in xs if not math.isnan(x)]
        if not xs:
            return (float("nan"),) * 4
        return (statistics.median(xs), statistics.mean(xs), min(xs), max(xs))

    fp_med, fp_mean, fp_min, fp_max = _stats(fp16_deltas)
    cu_med, cu_mean, cu_min, cu_max = _stats(cuda_deltas)
    sp_med, sp_mean, sp_min, sp_max = _stats(cuda_speedup_delta)

    md.append("| metric | median | mean | min | max |")
    md.append("|:---|---:|---:|---:|---:|")
    md.append(f"| fp16_us relative change | {fp_med:+.1%} | {fp_mean:+.1%} | {fp_min:+.1%} | {fp_max:+.1%} |")
    md.append(f"| cuda_us relative change | {cu_med:+.1%} | {cu_mean:+.1%} | {cu_min:+.1%} | {cu_max:+.1%} |")
    md.append(f"| **speedup Δ (r78 - r63)** | **{sp_med:+.2f}×** | {sp_mean:+.2f}× | {sp_min:+.2f}× | {sp_max:+.2f}× |")
    md.append("")
    if abs(fp_med) > 0.05:
        md.append(f"⚠️ **WARNING**: FP16 baseline median drift {fp_med:+.1%} > ±5%, "
                  "kernel-side comparison may be contaminated by GPU state drift.\n")
    else:
        md.append(f"✅ FP16 baseline drift ({fp_med:+.1%}) within ±5% — kernel comparison is reliable.\n")

    md.append("## §2 per-T improvement distribution\n")
    md.append("Positive Δ = r78 faster than r63 on that shape.\n")
    md.append("| T | N | median Δ | mean Δ | % improved |")
    md.append("|---:|---:|---:|---:|---:|")
    for T in sorted(per_t_improve):
        xs = per_t_improve[T]
        n_improved = sum(1 for d in xs if d > 0)
        md.append(f"| {T} | {len(xs)} | {statistics.median(xs):+.2f}× | "
                  f"{statistics.mean(xs):+.2f}× | {100 * n_improved / len(xs):.0f}% |")
    md.append("")

    md.append("## §3 per-proj improvement distribution\n")
    md.append("| proj | N | median Δ | mean Δ | % improved |")
    md.append("|:---|---:|---:|---:|---:|")
    for p in sorted(per_proj_improve):
        xs = per_proj_improve[p]
        n_improved = sum(1 for d in xs if d > 0)
        md.append(f"| {p} | {len(xs)} | {statistics.median(xs):+.2f}× | "
                  f"{statistics.mean(xs):+.2f}× | {100 * n_improved / len(xs):.0f}% |")
    md.append("")

    md.append("## §4 per-model improvement distribution\n")
    md.append("| model | N | median Δ | mean Δ | % improved |")
    md.append("|:---|---:|---:|---:|---:|")
    MODEL_ORDER = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
                   "Qwen3-14B", "Qwen2.5-32B", "LLaMA3-70B"]
    for m in sorted(per_model_improve,
                    key=lambda k: MODEL_ORDER.index(k) if k in MODEL_ORDER else 99):
        xs = per_model_improve[m]
        n_improved = sum(1 for d in xs if d > 0)
        md.append(f"| {m} | {len(xs)} | {statistics.median(xs):+.2f}× | "
                  f"{statistics.mean(xs):+.2f}× | {100 * n_improved / len(xs):.0f}% |")
    md.append("")

    md.append("## §5 TOP-10 shapes with biggest speedup improvement\n")
    md.append("| rank | model | proj | T | shape | r63 speedup | r78 speedup | Δ |")
    md.append("|---:|:---|:---|---:|:---:|---:|---:|---:|")
    ranked = []
    for key in common:
        r_old = i63[key]
        r_new = i78[key]
        if r_old["cuda_us"] > 0 and r_new["cuda_us"] > 0:
            s_old = r_old["fp16_us"] / r_old["cuda_us"]
            s_new = r_new["fp16_us"] / r_new["cuda_us"]
            ranked.append((s_new - s_old, key, s_old, s_new))
    ranked.sort(reverse=True)
    for i, (delta, key, s_old, s_new) in enumerate(ranked[:10], 1):
        m, p, T, d_in, d_out = key
        md.append(f"| {i} | {m} | {p} | {T} | {d_in}→{d_out} | "
                  f"{s_old:.2f}× | {s_new:.2f}× | **{delta:+.2f}×** |")
    md.append("")

    md.append("## §6 TOP-10 shapes with biggest regression\n")
    md.append("| rank | model | proj | T | shape | r63 speedup | r78 speedup | Δ |")
    md.append("|---:|:---|:---|---:|:---:|---:|---:|---:|")
    for i, (delta, key, s_old, s_new) in enumerate(ranked[-10:], 1):
        m, p, T, d_in, d_out = key
        md.append(f"| {i} | {m} | {p} | {T} | {d_in}→{d_out} | "
                  f"{s_old:.2f}× | {s_new:.2f}× | {delta:+.2f}× |")
    md.append("")

    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
