"""Compare r68_multiT_survey vs r63_combined historical roofline.

Difference drivers between the two:
  - r63 baseline: pre-Phase-C kernel (r63 main as of 2026-04-30).
  - r68 current : r63 + all Phase C dispatcher refinements (C.1..C.6v2)
                  plus Q.0-lite archive (probe macros reverted).

Scope of comparison (intersection only):
  - Models: Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen2.5-32B,
            LLaMA3-70B (Qwen3-0.6B excluded per user directive).
  - Batch sizes T: {1, 32, 128, 512} (the 4 Ts r63 has).
  - Projs: q_proj, kv_proj, o_proj, gate_up_proj, down_proj.

Output: one markdown table per (T, model) bucket with per-shape deltas
        plus roll-up stats (median, mean, wins, regressions).
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R63 = json.loads((ROOT / "logs/r63_combined/bench.json").read_text())
R68 = json.loads((ROOT / "logs/r68_multiT_survey/bench.json").read_text())


def index(bench):
    out = {}
    for r in bench["records"]:
        if r.get("kernel") != "end_to_end":
            continue
        key = (r["model"], r["proj"], r["T"], r["d_in"], r["d_out"])
        out[key] = r
    return out


A = index(R63)  # baseline
B = index(R68)  # current

# Filter: exclude Qwen3-0.6B and keep only Ts that r63 has (1, 32, 128, 512).
EXCLUDE_MODELS = {"Qwen3-0.6B"}
COMMON_TS = {1, 32, 128, 512}

keys = [k for k in A.keys()
        if k[0] not in EXCLUDE_MODELS and k[2] in COMMON_TS and k in B]
keys.sort()

print("=" * 96)
print("r63_combined → r68_multiT_survey comparison")
print("=" * 96)
print(f"Shapes in intersection: {len(keys)}")

rows = []
for k in keys:
    a, b = A[k], B[k]
    m, p, T, di, do = k
    sp63 = a["cuda_speedup_vs_fp16"]
    sp68 = b["cuda_speedup_vs_fp16"]
    us63 = a["cuda_us"]
    us68 = b["cuda_us"]
    d_us = us68 - us63
    d_pct = d_us / us63 * 100
    d_sp = sp68 - sp63
    rows.append({
        "model": m, "proj": p, "T": T, "d_in": di, "d_out": do,
        "us63": us63, "us68": us68, "d_us": d_us, "d_pct": d_pct,
        "sp63": sp63, "sp68": sp68, "d_sp": d_sp,
    })

# ================================================================
# Global roll-up
# ================================================================
sp63s = [r["sp63"] for r in rows]
sp68s = [r["sp68"] for r in rows]
w63 = sum(1 for s in sp63s if s >= 1.0)
w68 = sum(1 for s in sp68s if s >= 1.0)
bw63 = sum(1 for s in sp63s if s >= 2.0)
bw68 = sum(1 for s in sp68s if s >= 2.0)

print()
print("## Global")
print(f"{'metric':<34} {'r63':>9}  {'r68':>9}  {'delta':>9}")
print(f"{'median cuda_speedup_vs_fp16':<34} "
      f"{statistics.median(sp63s):>8.4f}x {statistics.median(sp68s):>8.4f}x "
      f"{statistics.median(sp68s)-statistics.median(sp63s):+9.4f}")
print(f"{'mean cuda_speedup_vs_fp16':<34} "
      f"{statistics.mean(sp63s):>8.4f}x {statistics.mean(sp68s):>8.4f}x "
      f"{statistics.mean(sp68s)-statistics.mean(sp63s):+9.4f}")
print(f"{'wins (sp >= 1.0x)':<34} "
      f"{w63:>5}/{len(rows)}   {w68:>5}/{len(rows)}   {w68-w63:+4d}")
print(f"{'big wins (sp >= 2.0x)':<34} "
      f"{bw63:>5}/{len(rows)}   {bw68:>5}/{len(rows)}   {bw68-bw63:+4d}")

# ================================================================
# By T bucket
# ================================================================
print()
print("## By T bucket")
print(f"{'T':>4} {'N':>4}  {'med r63':>10} {'med r68':>10} "
      f"{'Δ sp':>9}  {'wins r63':>10} {'wins r68':>10}")
by_T = defaultdict(list)
for r in rows:
    by_T[r["T"]].append(r)
for T in sorted(by_T):
    rs = by_T[T]
    m63 = statistics.median(r["sp63"] for r in rs)
    m68 = statistics.median(r["sp68"] for r in rs)
    w63 = sum(1 for r in rs if r["sp63"] >= 1.0)
    w68 = sum(1 for r in rs if r["sp68"] >= 1.0)
    print(f"{T:>4} {len(rs):>4}  {m63:>9.3f}x {m68:>9.3f}x "
          f"{m68-m63:+9.3f}  {w63:>4}/{len(rs):<4} {w68:>4}/{len(rs):<4}")

# ================================================================
# By model bucket
# ================================================================
PARAMS_B = {
    "Qwen3-1.7B": 1.7, "Qwen3-4B": 4.0, "Qwen3-8B": 8.0,
    "Qwen3-14B": 14.0, "Qwen2.5-32B": 32.0, "LLaMA3-70B": 70.0,
}
print()
print("## By model bucket (ordered by param count)")
print(f"{'model':<14} {'params':>7} {'N':>4}  "
      f"{'med r63':>10} {'med r68':>10} {'Δ sp':>9}  "
      f"{'wins r63':>10} {'wins r68':>10}")
by_m = defaultdict(list)
for r in rows:
    by_m[r["model"]].append(r)
for m in sorted(by_m, key=lambda x: (PARAMS_B.get(x, 999), x)):
    rs = by_m[m]
    m63 = statistics.median(r["sp63"] for r in rs)
    m68 = statistics.median(r["sp68"] for r in rs)
    w63 = sum(1 for r in rs if r["sp63"] >= 1.0)
    w68 = sum(1 for r in rs if r["sp68"] >= 1.0)
    p_str = f"{PARAMS_B.get(m, 0):.1f}B" if m in PARAMS_B else "-"
    print(f"{m:<14} {p_str:>7} {len(rs):>4}  "
          f"{m63:>9.3f}x {m68:>9.3f}x {m68-m63:+9.3f}  "
          f"{w63:>4}/{len(rs):<4} {w68:>4}/{len(rs):<4}")

# ================================================================
# Top movers (both directions)
# ================================================================
rows.sort(key=lambda r: r["d_pct"])
print()
print("## Top 10 improvements (largest cuda_us reduction vs r63)")
print(f"  {'shape':<44}  {'T':>3}  {'us63':>8}  {'us68':>8}  {'delta':>8}")
for r in rows[:10]:
    print(f"  {r['model']}/{r['proj']} {r['d_in']}->{r['d_out']:<6}  "
          f"{r['T']:>3}  {r['us63']:>7.2f}  {r['us68']:>7.2f}  "
          f"{r['d_pct']:+7.2f}%")

print()
print("## Top 10 regressions (largest cuda_us increase vs r63)")
rows.sort(key=lambda r: -r["d_pct"])
for r in rows[:10]:
    print(f"  {r['model']}/{r['proj']} {r['d_in']}->{r['d_out']:<6}  "
          f"{r['T']:>3}  {r['us63']:>7.2f}  {r['us68']:>7.2f}  "
          f"{r['d_pct']:+7.2f}%")

# ================================================================
# Count regressions and improvements above 3%
# ================================================================
imp3 = sum(1 for r in rows if r["d_pct"] <= -3.0)
reg3 = sum(1 for r in rows if r["d_pct"] >= +3.0)
print()
print(f"## Material changes (|Δ us| >= 3%)")
print(f"  improvements (>=3% faster): {imp3} / {len(rows)}")
print(f"  regressions  (>=3% slower): {reg3} / {len(rows)}")
