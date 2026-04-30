"""r62 F2 data-driven dispatch sweep.

For every (proj, T) shape in the Qwen3-8B canonical sweep, measure the
fused INT4 kernel under:
  1. auto (current dispatcher heuristics)
  2. all forced (kBm, kBn, split_k) combinations that compile

Emit a markdown report showing the best forced-config vs auto for each
shape so we can locate dispatch heuristic bugs with data, not guesses.

Usage (from repo root on autodl):
    PYTHONPATH=/root python -m kernel.tools.profile.dispatch_sweep \\
        --output /root/Zip_kernel/cuda_kernel/logs/r62_f2/dispatch_sweep.md \\
        --json   /root/Zip_kernel/cuda_kernel/logs/r62_f2/dispatch_sweep.json
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
from typing import List, Optional

import torch


# Qwen3-8B canonical 20 shapes (d_out, d_in, T).
# Order matches kernel/cuda_kernel/benchmarks/bench_qwen3_shapes.py.
QWEN3_8B_SHAPES = [
    # (proj, d_out, d_in, T)
    ("q_proj",       4096,  4096,    1),
    ("q_proj",       4096,  4096,   32),
    ("q_proj",       4096,  4096,  128),
    ("q_proj",       4096,  4096,  512),
    ("kv_proj",      2048,  4096,    1),
    ("kv_proj",      2048,  4096,   32),
    ("kv_proj",      2048,  4096,  128),
    ("kv_proj",      2048,  4096,  512),
    ("o_proj",       4096,  4096,    1),
    ("o_proj",       4096,  4096,   32),
    ("o_proj",       4096,  4096,  128),
    ("o_proj",       4096,  4096,  512),
    ("gate_up_proj",24576,  4096,    1),
    ("gate_up_proj",24576,  4096,   32),
    ("gate_up_proj",24576,  4096,  128),
    ("gate_up_proj",24576,  4096,  512),
    ("down_proj",    4096, 12288,    1),
    ("down_proj",    4096, 12288,   32),
    ("down_proj",    4096, 12288,  128),
    ("down_proj",    4096, 12288,  512),
]

# Config grid explored per shape.  Not exhaustive — we skip obviously-dead
# combos (e.g. kBn=8 when T>=128 is a known regression).
DEFAULT_CONFIGS = [
    # (label,     kBm,   kBn,   split_k)
    ("auto",      None,  None,  None),  # current dispatcher
    ("128/64/1",  "128", "64",  "1"),
    ("128/64/2",  "128", "64",  "2"),
    ("128/64/4",  "128", "64",  "4"),
    ("128/32/1",  "128", "32",  "1"),
    ("128/32/2",  "128", "32",  "2"),
    ("128/32/4",  "128", "32",  "4"),
    ("128/8/1",   "128", "8",   "1"),
    ("128/8/2",   "128", "8",   "2"),
    ("128/8/4",   "128", "8",   "4"),
    ("64/64/1",   "64",  "64",  "1"),
    ("64/64/2",   "64",  "64",  "2"),
    ("64/64/4",   "64",  "64",  "4"),
    ("64/32/1",   "64",  "32",  "1"),
    ("64/32/2",   "64",  "32",  "2"),
    ("64/32/4",   "64",  "32",  "4"),
    ("64/8/1",    "64",  "8",   "1"),
    ("64/8/2",    "64",  "8",   "2"),
    ("64/8/4",    "64",  "8",   "4"),
]


@dataclass
class Row:
    proj: str
    d_out: int
    d_in: int
    T: int
    n_groups: int
    auto_us: float
    best_label: str
    best_us: float
    results: List[dict]  # {label, kBm, kBn, split_k, us}

    @property
    def auto_vs_best(self) -> float:
        return self.auto_us / self.best_us if self.best_us > 0 else 1.0


def _set_env(kbm, kbn, sk):
    for k, v in [("HKUST_V9_FUSED_FORCE_KBM", kbm),
                 ("HKUST_V9_FUSED_FORCE_KBN", kbn),
                 ("HKUST_V9_FUSED_FORCE_SPLITK", sk)]:
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _clear_env():
    for k in ("HKUST_V9_FUSED_FORCE_KBM",
              "HKUST_V9_FUSED_FORCE_KBN",
              "HKUST_V9_FUSED_FORCE_SPLITK"):
        os.environ.pop(k, None)


def _bench_fn(fn, warmup=200, outer=10, inner=100, trials=3):
    """Median-of-trials over time_ms windows per GPU microbench spec."""
    from kernel.triton_kernel.benchmarks._bench_util import time_ms
    samples = []
    for _ in range(trials):
        samples.append(time_ms(fn, n_warmup=warmup,
                               n_iter=inner, n_repeat=outer) * 1000.0)
    return statistics.median(samples)


def _build_run(d_out, d_in, T):
    from kernel.cuda_kernel.tests.test_parity import _make_sparse_inputs
    from kernel.cuda_kernel import ops as cuda_ops

    # hp_ratio = 1/1024 is the canonical "dense-only" marker in bench
    # suite (small enough to collapse to hp=0 dispatch when we empty
    # the block arrays below).
    (W_low, _, _, _, X_s4, scale_u4, zero_u4, sum_X, scale_x) = \
        _make_sparse_inputs(T, d_out, d_in, n_hp_ratio=1 / 1024)
    nrow = d_out // 128
    W_hb = torch.zeros((0, 128, 64), dtype=torch.int8, device="cuda")
    hp_ro = torch.zeros(nrow + 1, dtype=torch.int32, device="cuda")
    hp_ci = torch.zeros((0,), dtype=torch.int32, device="cuda")

    def run():
        cuda_ops.fused_dense_sparse_cuda_int4(
            W_low, W_hb, hp_ro, hp_ci,
            X_s4, scale_u4, zero_u4, sum_X, scale_x,
            d_out, d_in,
        )

    return run


def run_sweep(configs=DEFAULT_CONFIGS, shapes=QWEN3_8B_SHAPES) -> List[Row]:
    rows: List[Row] = []
    for (proj, d_out, d_in, T) in shapes:
        run = _build_run(d_out, d_in, T)
        ng = d_in // 128
        results = []
        auto_us = None
        for (label, kbm, kbn, sk) in configs:
            _set_env(kbm, kbn, sk)
            # Warmup + time
            try:
                us = _bench_fn(run)
            except Exception as e:
                us = float("inf")
                results.append({
                    "label": label, "kBm": kbm, "kBn": kbn, "split_k": sk,
                    "us": None, "error": f"{type(e).__name__}: {e}"[:120],
                })
                continue
            results.append({
                "label": label, "kBm": kbm, "kBn": kbn, "split_k": sk,
                "us": us,
            })
            if label == "auto":
                auto_us = us
        _clear_env()

        # Find best among non-auto rows (auto is the baseline, not a choice).
        forced = [r for r in results if r["label"] != "auto" and r.get("us") is not None]
        if not forced:
            best = None
            best_label = "(none)"
            best_us = float("inf")
        else:
            best = min(forced, key=lambda r: r["us"])
            best_label = best["label"]
            best_us = best["us"]

        row = Row(
            proj=proj, d_out=d_out, d_in=d_in, T=T, n_groups=ng,
            auto_us=(auto_us or float("inf")),
            best_label=best_label, best_us=best_us, results=results,
        )
        rows.append(row)
        print(f"{proj:<13} T={T:<4} {d_out}x{d_in}: "
              f"auto={row.auto_us:6.2f}us  best={best_label:<10} "
              f"{best_us:6.2f}us  gap={row.auto_vs_best:.2f}x")
    return rows


def render_markdown(rows: List[Row], title: str) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(
        "Per-shape dispatch sweep: auto (current heuristic) vs exhaustive "
        "forced (kBm, kBn, split_k) grid.  Columns: auto_us | best_us | "
        "best_config | gap (auto/best).  Large gap => dispatch heuristic bug."
    )
    lines.append("")
    lines.append(
        "| proj | shape | T | ng | auto_us | best_us | best_cfg | gap |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---:|")
    for r in sorted(rows, key=lambda r: -r.auto_vs_best):
        shape_str = f"{r.d_out}×{r.d_in}"
        gap = f"{r.auto_vs_best:.2f}×"
        marker = " 🚨" if r.auto_vs_best >= 1.20 else (" ⚠" if r.auto_vs_best >= 1.10 else "")
        lines.append(
            f"| {r.proj} | {shape_str} | {r.T} | {r.n_groups} | "
            f"{r.auto_us:.2f} | {r.best_us:.2f} | `{r.best_label}` | {gap}{marker} |"
        )

    lines.append("")
    lines.append("## Per-shape full results (time in us)")
    lines.append("")
    for r in rows:
        lines.append(f"### {r.proj} {r.d_out}×{r.d_in}×{r.T} (ng={r.n_groups})")
        lines.append("")
        lines.append("| label | us |")
        lines.append("|---|---:|")
        for res in sorted(r.results, key=lambda res: res.get("us") or float("inf")):
            us = res.get("us")
            us_str = f"{us:.2f}" if us is not None else "FAIL"
            lines.append(f"| `{res['label']}` | {us_str} |")
        lines.append("")

    if rows:
        gaps = [r.auto_vs_best for r in rows]
        big_gap_shapes = [r for r in rows if r.auto_vs_best >= 1.10]
        lines.append("## Aggregate")
        lines.append("")
        lines.append(
            f"- shapes with ≥10% dispatch gap: **{len(big_gap_shapes)}/{len(rows)}**"
        )
        lines.append(
            f"- shapes with ≥20% dispatch gap: "
            f"**{sum(1 for g in gaps if g >= 1.20)}/{len(rows)}**"
        )
        lines.append(
            f"- median gap: {statistics.median(gaps):.2f}× | "
            f"max gap: {max(gaps):.2f}× "
            f"({max(rows, key=lambda r: r.auto_vs_best).proj}"
            f" T={max(rows, key=lambda r: r.auto_vs_best).T})"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True,
                   help="markdown report path")
    p.add_argument("--json", type=Path, default=None,
                   help="optional raw JSON dump")
    p.add_argument("--title", default="r62 F2 dispatch sweep — Qwen3-8B")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    t0 = time.time()
    rows = run_sweep()
    dt = time.time() - t0
    print(f"\nsweep done in {dt:.1f}s over {len(rows)} shapes "
          f"× {len(DEFAULT_CONFIGS)} configs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(rows, args.title))
    print(f"wrote {args.output}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            [asdict(r) for r in rows], indent=2
        ))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
