"""Summarise nsys CSV exports into a single human-readable markdown.

Expected CSV files next to each base name (produced by `nsys stats`):
  <base>_cuda_gpu_kern_sum.csv   : GPU kernel totals
  <base>_nvtx_gpu_proj_sum.csv   : GPU time attributed to NVTX ranges
  <base>_cuda_api_sum.csv        : host-side CUDA API call totals

Input: --tags "decode|/path/nsys_ts_decode;prefill|/path/nsys_ts_prefill;..."
Output: a markdown file at --output.

Logging obeys team convention (logging module, INFO console + DEBUG file).
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path


def _setup_logger(output: Path, ts: str) -> logging.Logger:
    log_path = output.with_suffix(".summarize.log")
    logger = logging.getLogger(f"nsys_summary_{ts}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.debug("summary log: %s", log_path)
    return logger


def _read_csv(path: Path, log: logging.Logger):
    """Load an nsys stats CSV, skipping any leading comment-only lines."""
    if not path.exists():
        log.warning("missing csv: %s", path)
        return []
    with path.open() as f:
        # nsys CSV starts directly with a header row; no pre-amble expected in
        # --format csv mode, but be defensive.
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _to_float(x: str) -> float:
    try:
        return float(x.replace(",", ""))
    except Exception:
        return float("nan")


def _fmt_ns(ns: float) -> str:
    """Nanoseconds -> human-friendly ms/us."""
    if ns != ns:  # NaN
        return "n/a"
    if ns >= 1_000_000:
        return f"{ns/1e6:.3f} ms"
    if ns >= 1_000:
        return f"{ns/1e3:.2f} us"
    return f"{ns:.0f} ns"


# NVTX ranges we care about, in display order
STAGES = [
    "stage1_act_quant",
    "stage2_dense_u4s4",
    "stage3_sparse_s4s4",
    "stage4_combine_add",
    "stage4_transpose_contig",
    "v9_total",
    "cublas_fp16_matmul",
]


def _stage_stats(nvtx_rows, iters_guess: int):
    """Return {stage: (avg_gpu_ns_per_call, total_gpu_ns, n_instances)}.

    nsys nvtx_gpu_proj_sum CSV columns (2025.1):
      Range, Style, Total Proj Time (ns), Total Range Time (ns),
      Range Instances, Proj Avg (ns), Proj Med (ns), ...

    "Proj" = projected GPU time, i.e. the GPU activity that overlaps with
    the NVTX range on the CPU side. This is what we want.
    """
    out = {}
    for row in nvtx_rows:
        rng = row.get("Range") or row.get("Name") or ""
        # nsys prefixes ranges with their domain, e.g. ':v9_total'. Strip it.
        rng = rng.lstrip(":")
        if rng not in STAGES:
            continue
        total = _to_float(row.get("Total Proj Time (ns)", "nan"))
        inst = _to_float(row.get("Range Instances", "nan"))
        avg = _to_float(row.get("Proj Avg (ns)", "nan"))
        if avg != avg and inst and inst > 0:
            avg = total / inst
        out[rng] = (avg, total, inst)
    return out

def _top_kernels(gpu_rows, top_n: int = 8):
    """Top-N GPU kernels by total time, returning [(name, total_ns, pct, inst)].

    nsys cuda_gpu_kern_sum columns:
      Time (%), Total Time (ns), Instances, Avg (ns), ..., Name
    """
    all_kernels = []
    grand_total = 0.0
    for row in gpu_rows:
        name = row.get("Name") or row.get("Kernel") or ""
        total = _to_float(row.get("Total Time (ns)", "nan"))
        inst = _to_float(row.get("Instances", "nan"))
        if total != total:
            continue
        all_kernels.append((name, total, inst))
        grand_total += total
    all_kernels.sort(key=lambda x: -x[1])
    ret = []
    for name, total, inst in all_kernels[:top_n]:
        pct = 100.0 * total / grand_total if grand_total > 0 else float("nan")
        ret.append((name, total, pct, inst))
    return ret, grand_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True,
                    help="tag1|base1;tag2|base2;...")
    ap.add_argument("--output", required=True)
    ap.add_argument("--ts", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    log = _setup_logger(out_path, args.ts)
    log.info("writing summary to %s", out_path)

    pairs = []
    for item in args.tags.split(";"):
        if not item:
            continue
        tag, base = item.split("|", 1)
        pairs.append((tag, Path(base)))

    with out_path.open("w") as f:
        f.write(f"# Nsight Systems profiling summary ({args.ts})\n\n")
        f.write(
            "All V9 pipeline stages are wrapped in NVTX ranges. Values below "
            "come from `nsys stats --report nvtx_gpu_proj_sum`, which "
            "attributes **GPU time** (not host wall-clock) to each range.\n\n"
        )

        # -------- A. Per-stage GPU time averaged per call --------
        f.write("## A. Per-stage GPU time per call\n\n")
        f.write("| workload | " + " | ".join(STAGES) + " |\n")
        f.write("|" + "|".join(["---"] * (len(STAGES) + 1)) + "|\n")
        for tag, base in pairs:
            nvtx_csv = base.parent / f"{base.name}_nvtx_gpu_proj_sum.csv"
            rows = _read_csv(nvtx_csv, log)
            stats = _stage_stats(rows, iters_guess=50)
            cells = [tag]
            for s in STAGES:
                if s in stats:
                    avg, _, _ = stats[s]
                    cells.append(_fmt_ns(avg))
                else:
                    cells.append("-")
            f.write("| " + " | ".join(cells) + " |\n")
        f.write("\n")

        # -------- B. Stage share of V9 total --------
        f.write("## B. Stage share of `v9_total` GPU time\n\n")
        f.write("| workload | stage1 quant% | stage2 dense% | "
                "stage3 sparse% | stage4 add% | stage4 transpose% |\n")
        f.write("|---|---|---|---|---|---|\n")
        for tag, base in pairs:
            nvtx_csv = base.parent / f"{base.name}_nvtx_gpu_proj_sum.csv"
            rows = _read_csv(nvtx_csv, log)
            stats = _stage_stats(rows, iters_guess=50)
            if "v9_total" not in stats:
                f.write(f"| {tag} | n/a | n/a | n/a | n/a | n/a |\n")
                continue
            tot = stats["v9_total"][1]
            def _pct(name):
                if name not in stats or tot <= 0:
                    return "-"
                return f"{100.0 * stats[name][1] / tot:.1f}%"
            f.write(
                f"| {tag} | {_pct('stage1_act_quant')} | "
                f"{_pct('stage2_dense_u4s4')} | {_pct('stage3_sparse_s4s4')} | "
                f"{_pct('stage4_combine_add')} | {_pct('stage4_transpose_contig')} |\n"
            )
        f.write("\n")

        # -------- C. V9 total vs cuBLAS FP16 --------
        f.write("## C. V9 total vs cuBLAS FP16 (GPU time per call)\n\n")
        f.write("| workload | v9_total | cublas_fp16 | speedup |\n")
        f.write("|---|---|---|---|\n")
        for tag, base in pairs:
            nvtx_csv = base.parent / f"{base.name}_nvtx_gpu_proj_sum.csv"
            rows = _read_csv(nvtx_csv, log)
            stats = _stage_stats(rows, iters_guess=50)
            v9 = stats.get("v9_total", (float("nan"), 0, 0))[0]
            fp = stats.get("cublas_fp16_matmul", (float("nan"), 0, 0))[0]
            sp = (fp / v9) if (v9 and v9 > 0) else float("nan")
            f.write(
                f"| {tag} | {_fmt_ns(v9)} | {_fmt_ns(fp)} | "
                f"{'n/a' if sp != sp else f'{sp:.2f}x'} |\n"
            )
        f.write("\n")

        # -------- D. Top GPU kernels per workload --------
        f.write("## D. Top GPU kernels per workload\n\n")
        for tag, base in pairs:
            kern_csv = base.parent / f"{base.name}_cuda_gpu_kern_sum.csv"
            rows = _read_csv(kern_csv, log)
            top, grand = _top_kernels(rows, top_n=8)
            f.write(f"### {tag}  (total GPU time {_fmt_ns(grand)})\n\n")
            f.write("| # | kernel | total | % | instances |\n")
            f.write("|---|---|---|---|---|\n")
            for i, (name, total, pct, inst) in enumerate(top, 1):
                short = name if len(name) <= 90 else name[:87] + "..."
                f.write(f"| {i} | `{short}` | {_fmt_ns(total)} | "
                        f"{pct:.1f}% | {int(inst) if inst==inst else '-'} |\n")
            f.write("\n")

        # -------- E. host-side CUDA API cost --------
        f.write("## E. Host-side CUDA API cost (top 5 per workload)\n\n")
        for tag, base in pairs:
            api_csv = base.parent / f"{base.name}_cuda_api_sum.csv"
            rows = _read_csv(api_csv, log)
            # nsys cuda_api_sum columns:
            #   Time (%), Total Time (ns), Num Calls, Avg (ns), ..., Name
            trimmed = []
            for r in rows:
                name = r.get("Name", "")
                total = _to_float(r.get("Total Time (ns)", "nan"))
                calls = _to_float(r.get("Num Calls", "nan"))
                avg = _to_float(r.get("Avg (ns)", "nan"))
                if total != total:
                    continue
                trimmed.append((name, total, calls, avg))
            trimmed.sort(key=lambda x: -x[1])
            f.write(f"### {tag}\n\n")
            f.write("| api | total | calls | avg |\n")
            f.write("|---|---|---|---|\n")
            for name, total, calls, avg in trimmed[:5]:
                f.write(f"| `{name}` | {_fmt_ns(total)} | "
                        f"{int(calls) if calls==calls else '-'} | {_fmt_ns(avg)} |\n")
            f.write("\n")

    log.info("summary written (%d bytes)", out_path.stat().st_size)


if __name__ == "__main__":
    main()
