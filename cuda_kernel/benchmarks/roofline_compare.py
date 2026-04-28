#!/usr/bin/env python3
"""
roofline_compare.py
-------------------
Offline analysis: compare measured `fp16_us` / `cuda_us` from
`bench_qwen3_shapes.py` output (`bench.json`) against RTX 4090 roofline
theoretical lower-bounds. Writes a CSV and a markdown report into the same
directory as the input `bench.json`.

Usage:
    python tmp/roofline_compare.py <path-to-bench.json> [--force]

Pure python, no torch / CUDA dependency.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# Section 1  -  GPU hardware constants  (RTX 4090, boost clock, vendor spec)
# =============================================================================
# All numbers are factory-advertised boost-clock peaks, NOT including sparsity.
# Change this block only when retargeting a different GPU.
HBM_BW_GBPS: float = 1008.0          # HBM bandwidth, GB/s
FP16_TFLOPS: float = 165.2           # FP16/BF16 Tensor Core peak, TFLOPS
INT8_TOPS: float = 330.4             # INT8  Tensor Core peak, TOPS
INT4_TOPS: float = 660.6             # INT4  Tensor Core peak, TOPS
ACHIEVABLE_FRACTION: float = 0.85    # engineering derating factor


@dataclass(frozen=True)
class GpuSpec:
    hbm_bw_gbps: float = HBM_BW_GBPS
    fp16_tflops: float = FP16_TFLOPS
    int8_tops: float = INT8_TOPS
    int4_tops: float = INT4_TOPS
    achievable: float = ACHIEVABLE_FRACTION

    @property
    def eff_hbm_bps(self) -> float:
        return self.hbm_bw_gbps * 1e9 * self.achievable

    @property
    def eff_fp16_flops(self) -> float:
        return self.fp16_tflops * 1e12 * self.achievable

    @property
    def eff_int4_flops(self) -> float:
        return self.int4_tops * 1e12 * self.achievable


GPU = GpuSpec()
GROUP_SIZE: int = 128  # W4A4 group-wise quantization group size

# Ordering conventions (fixed for deterministic report layout)
MODEL_ORDER: Tuple[str, ...] = (
    "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
)
PROJ_ORDER: Tuple[str, ...] = (
    "q_proj", "kv_proj", "o_proj", "gate_up_proj", "down_proj",
)
T_ORDER: Tuple[int, ...] = (1, 8, 128, 512, 1024)

REQUIRED_E2E_KEYS = ("model", "proj", "T", "d_in", "d_out",
                     "kernel", "fp16_us", "cuda_us")

# =============================================================================
# Section 2  -  bench.json loader
# =============================================================================


def load_bench(path: Path) -> List[Dict[str, Any]]:
    """Read bench.json, validate required keys, return the list of
    `kernel == 'end_to_end'` records, sorted by (model, proj, T)."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if isinstance(data, dict) and "records" in data:
        records = data["records"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            f"Unexpected bench.json top-level schema: {type(data).__name__}")

    if not isinstance(records, list) or not records:
        raise ValueError("bench.json 'records' is empty or not a list")

    e2e = [r for r in records if r.get("kernel") == "end_to_end"]
    if not e2e:
        raise ValueError("No kernel=='end_to_end' records found in bench.json")

    missing_by_record: List[str] = []
    for idx, rec in enumerate(e2e):
        missing = [k for k in REQUIRED_E2E_KEYS if k not in rec]
        if missing:
            missing_by_record.append(f"record[{idx}] missing {missing}")
    if missing_by_record:
        raise KeyError(
            "bench.json end_to_end records have missing fields:\n  " +
            "\n  ".join(missing_by_record[:10]))

    def _sort_key(r: Dict[str, Any]) -> Tuple[int, int, int]:
        m_idx = MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 999
        p_idx = PROJ_ORDER.index(r["proj"]) if r["proj"] in PROJ_ORDER else 999
        t_idx = T_ORDER.index(r["T"]) if r["T"] in T_ORDER else 999
        return (m_idx, p_idx, t_idx)

    e2e.sort(key=_sort_key)
    return e2e


# =============================================================================
# Section 3  -  FP16 roofline
# =============================================================================


def fp16_roofline(T: int, d_in: int, d_out: int,
                  gpu: GpuSpec = GPU) -> Dict[str, Any]:
    """Single torch.matmul(W_fp, X_fp.T) roofline lower bound."""
    flops = 2.0 * T * d_in * d_out
    # HBM traffic: W + X + Y, all fp16 (2 bytes/elem)
    byts = (d_in * d_out + T * d_in + T * d_out) * 2.0

    t_compute_us = flops / gpu.eff_fp16_flops * 1e6
    t_mem_us = byts / gpu.eff_hbm_bps * 1e6
    t_roof_us = max(t_compute_us, t_mem_us)
    bound = "compute" if t_compute_us >= t_mem_us else "mem"
    return {
        "t_compute_us": t_compute_us,
        "t_mem_us": t_mem_us,
        "t_roof_us": t_roof_us,
        "bound": bound,
    }


# =============================================================================
# Section 4  -  CUDA W4A4 two-stage roofline (T >= 2)
# =============================================================================


def quant_roofline(T: int, d_in: int,
                   gpu: GpuSpec = GPU) -> float:
    """activation_quant: treat as pure memory-bound stage."""
    n_groups = d_in // GROUP_SIZE
    b_read_x = T * d_in * 2.0                  # fp16 X
    b_write_xq = T * d_in / 2.0                # int4 packed (2 per byte)
    b_write_sx = T * 2.0                       # fp16 scale_x
    b_write_sum = T * n_groups * 4.0           # int32 sum_X
    byts = b_read_x + b_write_xq + b_write_sx + b_write_sum
    return byts / gpu.eff_hbm_bps * 1e6


def gemm_roofline_mma(T: int, d_in: int, d_out: int,
                      gpu: GpuSpec = GPU) -> Dict[str, Any]:
    """fused_dense_sparse W4A4 GEMM roofline: max(INT4 compute, HBM mem)."""
    flops = 2.0 * T * d_in * d_out
    n_groups = d_in // GROUP_SIZE
    # W(int4)=0.5B/elem, X_s4(int4)=0.5B/elem,
    # scale+zero (fp16) x2 over d_out * n_groups, Y(fp16)=2B/elem
    b_w = d_in * d_out * 0.5
    b_x = T * d_in * 0.5
    b_sz = d_out * n_groups * 2.0 * 2.0  # scale and zero, each fp16
    b_y = T * d_out * 2.0
    byts = b_w + b_x + b_sz + b_y

    t_compute_us = flops / gpu.eff_int4_flops * 1e6
    t_mem_us = byts / gpu.eff_hbm_bps * 1e6
    t_roof_us = max(t_compute_us, t_mem_us)
    bound = "compute" if t_compute_us >= t_mem_us else "mem"
    return {
        "t_compute_us": t_compute_us,
        "t_mem_us": t_mem_us,
        "t_roof_us": t_roof_us,
        "bound": bound,
    }


def cuda_roofline_multi_T(T: int, d_in: int, d_out: int,
                          gpu: GpuSpec = GPU) -> Dict[str, Any]:
    """Two serial stages summed (no stream overlap in current ops.py)."""
    t_quant = quant_roofline(T, d_in, gpu)
    gemm = gemm_roofline_mma(T, d_in, d_out, gpu)
    return {
        "t_quant_roof_us": t_quant,
        "t_gemm_roof_us": gemm["t_roof_us"],
        "t_roof_us": t_quant + gemm["t_roof_us"],
        "gemm_bound": gemm["bound"],
    }


# =============================================================================
# Section 5  -  CUDA T=1 fused kernel roofline (single stage)
# =============================================================================


def cuda_roofline_T1(d_in: int, d_out: int,
                     gpu: GpuSpec = GPU) -> Dict[str, Any]:
    """T=1 fused_quant_gemv_cuda: quant + GEMV fused into a single kernel."""
    n_groups = d_in // GROUP_SIZE
    flops = 2.0 * d_in * d_out
    # traffic: read X (fp16), read W (int4), read scale+zero (fp16 x2),
    # write Y (fp16).  Quant intermediates stay in regs / smem, no HBM.
    byts = (d_in * 2.0
            + d_in * d_out * 0.5
            + d_out * n_groups * 2.0 * 2.0
            + d_out * 2.0)
    t_compute_us = flops / gpu.eff_int4_flops * 1e6
    t_mem_us = byts / gpu.eff_hbm_bps * 1e6
    t_roof_us = max(t_compute_us, t_mem_us)
    bound = "compute" if t_compute_us >= t_mem_us else "mem"
    return {
        "t_quant_roof_us": float("nan"),   # fused, no standalone stage
        "t_gemm_roof_us": t_roof_us,
        "t_roof_us": t_roof_us,
        "gemm_bound": "fused-" + bound,
    }


def cuda_roofline(T: int, d_in: int, d_out: int,
                  gpu: GpuSpec = GPU) -> Dict[str, Any]:
    if T == 1:
        return cuda_roofline_T1(d_in, d_out, gpu)
    return cuda_roofline_multi_T(T, d_in, d_out, gpu)


# =============================================================================
# Section 6  -  Row aggregation + CSV
# =============================================================================

CSV_COLUMNS: Tuple[str, ...] = (
    "model", "proj", "T", "d_in", "d_out",
    "fp16_us", "fp16_roof_us", "fp16_efficiency", "fp16_bound",
    "cuda_us", "cuda_quant_roof_us", "cuda_gemm_roof_us",
    "cuda_roof_us", "cuda_efficiency", "cuda_gemm_bound",
    "cuda_vs_fp16_actual", "cuda_vs_fp16_roofline",
)


def _fmt_float(x: float, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    return f"{x:.{digits}f}"


def build_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    T = int(rec["T"])
    d_in = int(rec["d_in"])
    d_out = int(rec["d_out"])
    fp16_us = float(rec["fp16_us"])
    cuda_us = float(rec["cuda_us"])

    fp = fp16_roofline(T, d_in, d_out)
    cu = cuda_roofline(T, d_in, d_out)

    fp16_eff = fp["t_roof_us"] / fp16_us if fp16_us > 0 else float("nan")
    cuda_eff = cu["t_roof_us"] / cuda_us if cuda_us > 0 else float("nan")

    return {
        "model": rec["model"],
        "proj": rec["proj"],
        "T": T,
        "d_in": d_in,
        "d_out": d_out,
        "fp16_us": fp16_us,
        "fp16_roof_us": fp["t_roof_us"],
        "fp16_efficiency": fp16_eff,
        "fp16_bound": fp["bound"],
        "cuda_us": cuda_us,
        "cuda_quant_roof_us": cu["t_quant_roof_us"],
        "cuda_gemm_roof_us": cu["t_gemm_roof_us"],
        "cuda_roof_us": cu["t_roof_us"],
        "cuda_efficiency": cuda_eff,
        "cuda_gemm_bound": cu["gemm_bound"],
        "cuda_vs_fp16_actual": (cuda_us / fp16_us) if fp16_us > 0 else float("nan"),
        "cuda_vs_fp16_roofline": (cu["t_roof_us"] / fp["t_roof_us"])
                                   if fp["t_roof_us"] > 0 else float("nan"),
    }


def write_csv(rows: List[Dict[str, Any]], out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out_path}. "
            f"Use --force to overwrite.")
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            out = {}
            for c in CSV_COLUMNS:
                v = row.get(c)
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        out[c] = ""
                    else:
                        out[c] = f"{v:.6f}"
                else:
                    out[c] = v
            writer.writerow(out)


# =============================================================================
# Section 7  -  Markdown report rendering
# =============================================================================


def _pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x * 100:.0f}%"


def _bucket_stats(values: List[float]) -> Tuple[float, float, float]:
    xs = [v for v in values if v is not None and not math.isnan(v)]
    if not xs:
        return (float("nan"),) * 3
    return (min(xs), statistics.median(xs), max(xs))


def render_section_1(md: List[str]) -> None:
    md.append("## §1 Hardware constants and formulas\n")
    md.append("| Parameter | Value | Note |")
    md.append("| --- | --- | --- |")
    md.append(f"| HBM bandwidth | {HBM_BW_GBPS:.0f} GB/s | RTX 4090 vendor spec |")
    md.append(f"| FP16/BF16 TC peak | {FP16_TFLOPS:.1f} TFLOPS | boost clock, no sparsity |")
    md.append(f"| INT8 TC peak | {INT8_TOPS:.1f} TOPS | boost clock |")
    md.append(f"| INT4 TC peak | {INT4_TOPS:.1f} TOPS | boost clock |")
    md.append(f"| ACHIEVABLE_FRACTION | {ACHIEVABLE_FRACTION:.2f} | engineering derating |")
    md.append("")
    md.append("**Formulas** (all time in microseconds, eff_* = peak * ACHIEVABLE_FRACTION):")
    md.append("")
    md.append("- **FP16 roofline** — `t = max(flops / eff_fp16, bytes / eff_hbm)` "
              "where `flops = 2*T*d_in*d_out`, `bytes = 2*(d_in*d_out + T*d_in + T*d_out)`.")
    md.append("- **CUDA `activation_quant` (T>=2)** — pure mem-bound, "
              "`bytes = 2*T*d_in + 0.5*T*d_in + 2*T + 4*T*n_groups`.")
    md.append("- **CUDA `fused_dense_sparse` (T>=2)** — "
              "`t = max(2*T*d_in*d_out / eff_int4, bytes / eff_hbm)` "
              "with `bytes = 0.5*d_in*d_out + 0.5*T*d_in + 4*d_out*n_groups + 2*T*d_out`.")
    md.append("- **CUDA end-to-end (T>=2)** — `t_cuda_roof = t_quant + t_gemm` "
              "(serial sum, because ops.py has no stream overlap).")
    md.append("- **CUDA T=1 fused** — single stage roofline with quant traffic "
              "merged into GEMV: `bytes = 2*d_in + 0.5*d_in*d_out + 4*d_out*n_groups + 2*d_out`.")
    md.append("")
    md.append("### Known systematic biases")
    md.append("")
    md.append("1. **Kernel launch overhead** (5-10us/launch) is *not* counted in any "
              "roofline; the model will systematically under-estimate achievable time "
              "for T<=8 small shapes.")
    md.append("2. **L2 cache reuse** — back-to-back benches may let W stay in L2, making "
              "real mem time 20-40% below the pure-HBM roofline. Our roofline is therefore "
              "a conservative *upper bound* on achievable time — it may actually be *too "
              "pessimistic* relative to what the kernel achieves with L2 reuse.")
    md.append("3. **Tensor-Core utilisation** — 165.2 / 660.6 TFLOPS are vendor peaks; "
              "cuBLAS / hand-written kernels on irregular / unaligned shapes typically "
              "reach 85-95%. `ACHIEVABLE_FRACTION < 1.0` is the mandatory conservative term.")
    md.append("4. **Reduce in activation_quant** — a CTA-wide max-abs reduce is not strictly "
              "pure mem-bound, but its cost is <1us and is neglected.")
    md.append("5. **Epilogue FMA cost** — each output point runs n_groups dequant FMAs on "
              "CUDA Cores; we fold this into the INT4 TC peak which is an *optimistic* "
              "simplification. In reality the GEMM stage is often CUDA-Core-FMA-bound, and "
              "this is the main reason measured `cuda_efficiency` falls below 1.0.")
    md.append("")
    md.append("> If `cuda_efficiency > 1.0` for some row, the row is tagged "
              "`⚠ L2/roof low-bound` in §4 — this indicates L2 hit or model pessimism.")
    md.append("")


def render_section_2(md: List[str], rows: List[Dict[str, Any]]) -> None:
    md.append("## §2 FP16 efficiency distribution (by T)\n")
    md.append("`fp16_efficiency = fp16_roof_us / fp16_us` — how close cuBLAS gets to 4090's physical limit.\n")
    md.append("| T | n | min | median | max |")
    md.append("| ---: | ---: | ---: | ---: | ---: |")
    for T in T_ORDER:
        vals = [r["fp16_efficiency"] for r in rows if r["T"] == T]
        mn, me, mx = _bucket_stats(vals)
        md.append(f"| {T} | {len(vals)} | {_pct(mn)} | {_pct(me)} | {_pct(mx)} |")
    md.append("")


def render_section_3(md: List[str], rows: List[Dict[str, Any]]) -> None:
    md.append("## §3 CUDA efficiency distribution\n")
    md.append("`cuda_efficiency = cuda_roof_us / cuda_us` — how close our W4A4 kernel gets to its own roofline.\n")
    md.append("### §3.1 By T\n")
    md.append("| T | n | min | median | max |")
    md.append("| ---: | ---: | ---: | ---: | ---: |")
    for T in T_ORDER:
        vals = [r["cuda_efficiency"] for r in rows if r["T"] == T]
        mn, me, mx = _bucket_stats(vals)
        md.append(f"| {T} | {len(vals)} | {_pct(mn)} | {_pct(me)} | {_pct(mx)} |")
    md.append("")
    md.append("### §3.2 By proj\n")
    md.append("| proj | n | min | median | max |")
    md.append("| :--- | ---: | ---: | ---: | ---: |")
    for p in PROJ_ORDER:
        vals = [r["cuda_efficiency"] for r in rows if r["proj"] == p]
        if not vals:
            continue
        mn, me, mx = _bucket_stats(vals)
        md.append(f"| {p} | {len(vals)} | {_pct(mn)} | {_pct(me)} | {_pct(mx)} |")
    md.append("")


def render_section_4(md: List[str], rows: List[Dict[str, Any]]) -> None:
    md.append("## §4 Per-shape detail tables (core section, 100 rows)\n")
    md.append("One subtable per Qwen3 model. One row per (proj, T) covering all "
              "bench.json end_to_end records.\n")

    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rows_by_model.setdefault(r["model"], []).append(r)

    detail_count = 0
    for model in MODEL_ORDER:
        mrows = rows_by_model.get(model)
        if not mrows:
            continue
        md.append(f"### {model}\n")
        md.append("| proj | T | shape | fp16_us | fp16_roof_us | fp16_eff | "
                  "cuda_us | cuda_roof_us | cuda_eff | cuda/fp16 actual (roof) |")
        md.append("| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")

        # Sort within model by (proj_order, T_order) — already globally sorted,
        # but be defensive in case only a subset is present.
        mrows.sort(key=lambda r: (
            PROJ_ORDER.index(r["proj"]) if r["proj"] in PROJ_ORDER else 999,
            T_ORDER.index(r["T"]) if r["T"] in T_ORDER else 999))

        for r in mrows:
            shape = f"{r['d_in']}→{r['d_out']}"
            eff_tag = " ⚠ L2/roof low-bound" if (
                isinstance(r["cuda_efficiency"], float)
                and not math.isnan(r["cuda_efficiency"])
                and r["cuda_efficiency"] > 1.0) else ""
            md.append(
                f"| {r['proj']} | {r['T']} | {shape} | "
                f"{_fmt_float(r['fp16_us'], 2)} | {_fmt_float(r['fp16_roof_us'], 2)} | "
                f"{_pct(r['fp16_efficiency'])} | "
                f"{_fmt_float(r['cuda_us'], 2)} | {_fmt_float(r['cuda_roof_us'], 2)} | "
                f"{_pct(r['cuda_efficiency'])}{eff_tag} | "
                f"{_fmt_float(r['cuda_vs_fp16_actual'], 2)}x "
                f"({_fmt_float(r['cuda_vs_fp16_roofline'], 2)}x) |")
            detail_count += 1
        md.append("")
    md.append(f"_Detail rows rendered: {detail_count}._\n")


def render_section_5(md: List[str], rows: List[Dict[str, Any]]) -> None:
    md.append("## §5 CUDA implementation-gap TOP-15 (worst cuda_efficiency)\n")
    md.append("Complement to §4 — zoom on the shapes furthest from their own roofline.\n")
    md.append("| rank | model | proj | T | shape | cuda_us | t_quant_roof | t_gemm_roof | cuda_roof | cuda_eff | gemm_bound |")
    md.append("| ---: | :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    top = sorted(
        (r for r in rows
         if isinstance(r["cuda_efficiency"], float)
         and not math.isnan(r["cuda_efficiency"])),
        key=lambda r: r["cuda_efficiency"])[:15]
    for i, r in enumerate(top, 1):
        shape = f"{r['d_in']}→{r['d_out']}"
        md.append(
            f"| {i} | {r['model']} | {r['proj']} | {r['T']} | {shape} | "
            f"{_fmt_float(r['cuda_us'], 2)} | "
            f"{_fmt_float(r['cuda_quant_roof_us'], 2)} | "
            f"{_fmt_float(r['cuda_gemm_roof_us'], 2)} | "
            f"{_fmt_float(r['cuda_roof_us'], 2)} | "
            f"{_pct(r['cuda_efficiency'])} | "
            f"{r['cuda_gemm_bound']} |")
    md.append("")


def render_section_6(md: List[str], rows: List[Dict[str, Any]]) -> None:
    md.append("## §6 Roofline-side CUDA vs FP16  (theoretical ceiling comparison)\n")
    md.append("`cuda_roof_us / fp16_roof_us` — shows which shapes **W4A4 cannot beat FP16 "
              "even at the physical limit**. Values >1.0 mean the FP16 ceiling is *faster*.\n")
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rows_by_model.setdefault(r["model"], []).append(r)
    for model in MODEL_ORDER:
        mrows = rows_by_model.get(model)
        if not mrows:
            continue
        md.append(f"### {model}\n")
        md.append("| proj | T | shape | fp16_roof_us | cuda_roof_us | cuda_roof / fp16_roof |")
        md.append("| :--- | ---: | :---: | ---: | ---: | :---: |")
        mrows.sort(key=lambda r: (
            PROJ_ORDER.index(r["proj"]) if r["proj"] in PROJ_ORDER else 999,
            T_ORDER.index(r["T"]) if r["T"] in T_ORDER else 999))
        for r in mrows:
            shape = f"{r['d_in']}→{r['d_out']}"
            ratio = r["cuda_vs_fp16_roofline"]
            tag = " ✗ W4A4 ceiling slower" if (
                isinstance(ratio, float) and not math.isnan(ratio) and ratio > 1.0) else ""
            md.append(
                f"| {r['proj']} | {r['T']} | {shape} | "
                f"{_fmt_float(r['fp16_roof_us'], 2)} | "
                f"{_fmt_float(r['cuda_roof_us'], 2)} | "
                f"{_fmt_float(ratio, 2)}x{tag} |")
        md.append("")


def render_section_7(md: List[str], rows: List[Dict[str, Any]]) -> None:
    md.append("## §7 Conclusions and next steps\n")
    # Empirical counts to feed the prose
    n_total = len(rows)
    n_cuda_near = sum(1 for r in rows
                      if isinstance(r["cuda_efficiency"], float)
                      and not math.isnan(r["cuda_efficiency"])
                      and r["cuda_efficiency"] >= 0.8)
    n_cuda_bad = sum(1 for r in rows
                     if isinstance(r["cuda_efficiency"], float)
                     and not math.isnan(r["cuda_efficiency"])
                     and r["cuda_efficiency"] < 0.5)
    n_ceiling_lose = sum(1 for r in rows
                         if isinstance(r["cuda_vs_fp16_roofline"], float)
                         and not math.isnan(r["cuda_vs_fp16_roofline"])
                         and r["cuda_vs_fp16_roofline"] > 1.0)
    n_actual_lose = sum(1 for r in rows
                        if isinstance(r["cuda_vs_fp16_actual"], float)
                        and not math.isnan(r["cuda_vs_fp16_actual"])
                        and r["cuda_vs_fp16_actual"] > 1.0)
    md.append(f"- Out of **{n_total}** shapes, **{n_cuda_near}** reach "
              f"`cuda_efficiency >= 0.8` — these are already near the physical "
              f"limit; further kernel tuning has diminishing ROI.")
    md.append(f"- **{n_cuda_bad}** shapes sit at `cuda_efficiency < 0.5` — these "
              f"have real implementation slack; cross-reference §5 for the worst "
              f"offenders and their mem/compute bound to pick the next kernel to fix.")
    md.append(f"- **{n_ceiling_lose}** shapes have `cuda_roof > fp16_roof` — W4A4 "
              f"loses even at the ceiling; these should fall back to FP16 via policy, "
              f"no kernel work can rescue them.")
    md.append(f"- Measured today, **{n_actual_lose}** shapes actually lose to FP16; "
              f"the delta `{n_actual_lose} - {n_ceiling_lose}` is the *implementation* "
              f"gap (fixable), the rest is *physics* (unfixable).")
    md.append("- Recommended next moves: (a) for the §5 top offenders whose "
              "`gemm_bound == mem`, audit packed-weight layout / cache reuse; "
              "(b) for those with `gemm_bound == compute` and narrow d_out, "
              "revisit CTA sizing (see R41-R46 iteration); (c) for shapes listed as "
              "`✗ W4A4 ceiling slower` in §6, route via policy.py to FP16.")
    md.append("")


def render_report(rows: List[Dict[str, Any]], out_path: Path, force: bool,
                  source_json: Path) -> int:
    if out_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out_path}. "
            f"Use --force to overwrite.")

    md: List[str] = []
    md.append("# Roofline theoretical vs measured report\n")
    md.append(f"Source: `{source_json}`\n")
    md.append(f"GPU model: RTX 4090 (vendor spec, ACHIEVABLE_FRACTION="
              f"{ACHIEVABLE_FRACTION:.2f})\n")
    md.append("")

    render_section_1(md)
    render_section_2(md, rows)
    render_section_3(md, rows)
    render_section_4(md, rows)
    render_section_5(md, rows)
    render_section_6(md, rows)
    render_section_7(md, rows)

    out_path.write_text("\n".join(md), encoding="utf-8")

    # self-check count (rows actually rendered into §4 tables)
    rendered = sum(1 for r in rows if r["model"] in MODEL_ORDER)
    return rendered


# =============================================================================
# Main
# =============================================================================


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_json", type=Path,
                        help="Path to bench.json produced by bench_qwen3_shapes.py")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing CSV / md output files.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("roofline")

    bench_path: Path = args.bench_json
    if not bench_path.is_file():
        log.error("bench.json not found: %s", bench_path)
        return 2

    out_dir = bench_path.parent
    csv_path = out_dir / "roofline_compare.csv"
    md_path = out_dir / "roofline_report.md"

    log.info("Loading bench records from %s", bench_path)
    e2e = load_bench(bench_path)
    log.info("Found %d end_to_end records", len(e2e))

    rows = [build_row(r) for r in e2e]
    log.info("Built %d roofline rows", len(rows))

    write_csv(rows, csv_path, args.force)
    log.info("CSV written: %s", csv_path)

    detail_rows = render_report(rows, md_path, args.force, bench_path)
    log.info("Markdown written: %s", md_path)
    log.info("detail_rows=%d", detail_rows)

    if detail_rows != len(e2e):
        log.error("Detail row self-check failed: rendered=%d, expected=%d",
                  detail_rows, len(e2e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
