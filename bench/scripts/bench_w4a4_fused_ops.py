"""r79 bench driver: W4A4 CUDA 4-fused-op kernel time sweep.

Scope
-----
For each (model, phase, batch) in {Qwen3-4B,8B,14B} × {prefill,decode} ×
{4,8,16,32}, measure the kernel time of the 4 fused W4A4 projection
groups (qkv_fused / o_proj / gate_up_fused / down_proj) that the CUDA
kernel replaces in production inference.

Timing policy (per user Q2, 2026-05-06):

  * decode   →  STRICT               (warmup=500, outer=20, inner=200, trials=5)
  * prefill  →  adaptive, FLOPs-gated:
                  very heavy (> 2 ms/call)   → warmup=20, outer=5,  inner=5,  trials=5
                  heavy     (0.5–2 ms/call)  → warmup=30, outer=8,  inner=10, trials=5
                  light     (< 0.5 ms/call)  → ADAPTIVE_PREFILL
                                               (warmup=50, outer=10,inner=20, trials=5)

The adaptive rule keeps 14B prefill (the biggest shape) from dominating
wall-clock while staying well inside the <1% spread envelope that
ADAPTIVE_PREFILL already reliably hits for any op ≥100us on RTX 4090
(see bench/layer/timing.py docstring).

Kernel path taken (documented per row in the summary):

  * prefill (T = bs*2048):  T ≥ 2  →  legacy_mma
                           (activation_quant + fused_dense_sparse_cuda_int4)
  * decode  (T = bs*1):     T ≥ 2  →  legacy_mma (same as above)
    (B=1 decode is intentionally OUT of scope per user Q1, 2026-05-06.)

Outputs
-------
  <out-dir>/bench_w4a4_per_op.json   : full per-row data (24 × 4 = 96 rows)
  <out-dir>/bench_w4a4_summary.md    : grouped tables for eyeballing
  <out-dir>/run.log                  : stdout tee from run_bench_w4a4.sh
  <out-dir>/VALIDATION_LOG.md        : env + dispatch + spread sanity
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project-root on sys.path so the script can be launched from anywhere.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJ_ROOT = _THIS_FILE.parents[3]  # .../HKUST
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from kernel.bench.configs.qwen3_shapes import (  # noqa: E402
    BATCH_SIZES,
    PHASES,
    QWEN3_MODELS,
    Qwen3Config,
    PhaseConfig,
)
from kernel.bench.layer.qwen3_w4a4_ops import (  # noqa: E402
    W4A4OpBundle,
    build_four_op_callables,
)
from kernel.bench.layer.timing import (  # noqa: E402
    ADAPTIVE_PREFILL,
    STRICT,
    TimingStats,
    measure,
)

# ---------------------------------------------------------------------------
# Adaptive preset selection for prefill
# ---------------------------------------------------------------------------
# Rough peak: RTX 4090 @ 660 TOPS INT4 (dense MMA).  We use 330 TOPS as a
# realistic ceiling (half of peak accounts for memory + dequant overhead).
_INT4_PEAK_TOPS: float = 330.0e12  # 330 T ops/s

_HEAVY_MS = 2.0   # above this: tiny-loop preset
_MED_MS   = 0.5   # above this: medium-loop preset

_TINY_PREFILL_PRESET = dict(warmup=20, outer=5,  inner=5,  trials=5)
_MED_PREFILL_PRESET  = dict(warmup=30, outer=8,  inner=10, trials=5)

def _estimate_call_ms(T: int, d_in: int, d_out: int) -> float:
    """Rough est. of kernel time for a single W4A4 GEMM at (T,d_in,d_out)."""
    flops = 2.0 * T * d_in * d_out  # 1 MAC = 2 ops
    return flops / _INT4_PEAK_TOPS * 1.0e3  # ms

def _pick_preset(phase_name: str, bundles: list[W4A4OpBundle]) -> dict:
    """Pick timing preset based on phase + the *max* call-cost among the 4
    bundles.  We use max() so all 4 ops under one (model,phase,bs) share
    one preset — keeps the summary rows comparable.
    """
    if phase_name == "decode":
        return STRICT
    est_ms = max(_estimate_call_ms(b.T, b.d_in, b.d_out) for b in bundles)
    if est_ms > _HEAVY_MS:
        return _TINY_PREFILL_PRESET
    if est_ms > _MED_MS:
        return _MED_PREFILL_PRESET
    return ADAPTIVE_PREFILL

# ---------------------------------------------------------------------------
# One (model, phase, bs, op) row
# ---------------------------------------------------------------------------
@dataclass
class Row:
    model: str
    phase: str
    batch: int
    seqlen: int
    past_kv_len: int
    op: str
    d_in: int
    d_out: int
    T: int
    kernel_path: str          # "legacy_mma" / "gemv"
    preset_name: str          # "STRICT" / "ADAPTIVE_PREFILL" / "MED_PREFILL" / "TINY_PREFILL"
    est_call_ms: float
    stats: dict               # TimingStats.as_dict()

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
_PRESET_NAMES = {
    id(STRICT):                "STRICT",
    id(ADAPTIVE_PREFILL):      "ADAPTIVE_PREFILL",
    id(_TINY_PREFILL_PRESET):  "TINY_PREFILL",
    id(_MED_PREFILL_PRESET):   "MED_PREFILL",
}

def _preset_name(preset: dict) -> str:
    return _PRESET_NAMES.get(id(preset), "CUSTOM")

def _run_one_triple(
    cfg: Qwen3Config,
    phase: PhaseConfig,
    bs: int,
    *,
    device: torch.device,
) -> list[Row]:
    """Build + time all 4 ops for one (model, phase, bs) triple."""
    four = build_four_op_callables(
        cfg, batch=bs, seqlen=phase.seqlen, device=device
    )
    bundles = [b for _, b, _ in four.as_list()]
    preset = _pick_preset(phase.name, bundles)
    pname = _preset_name(preset)

    rows: list[Row] = []
    for op_name, bundle, fn in four.as_list():
        # smoke call to surface launch errors early (before the timing loop).
        _ = fn()
        torch.cuda.synchronize(device)

        stats: TimingStats = measure(fn, device=device, **preset)
        est_ms = _estimate_call_ms(bundle.T, bundle.d_in, bundle.d_out)
        rows.append(Row(
            model=cfg.name, phase=phase.name, batch=bs,
            seqlen=phase.seqlen, past_kv_len=phase.past_kv_len,
            op=op_name, d_in=bundle.d_in, d_out=bundle.d_out,
            T=bundle.T, kernel_path=bundle.path,
            preset_name=pname, est_call_ms=est_ms,
            stats=stats.as_dict(),
        ))
        print(
            f"  {cfg.name:<10} {phase.name:<7} bs={bs:<3} {op_name:<15} "
            f"T={bundle.T:<6} d_in={bundle.d_in:<6} d_out={bundle.d_out:<6} "
            f"path={bundle.path:<11} preset={pname:<17} "
            f"median={stats.median_us:8.2f}us  spread={stats.spread_pct:5.2f}%",
            flush=True,
        )
        # Drop bundle refs to free HBM for the next op (biggest bundles
        # are 14B gate_up @ T=8192: ~350 MiB).
        del fn, bundle
    del four, bundles
    torch.cuda.empty_cache()
    return rows

def _write_summary_md(rows: list[Row], out_path: Path) -> None:
    """Human-readable grouped table."""
    lines: list[str] = []
    lines.append("# W4A4 fused-op kernel time — r79 bench\n")
    lines.append("Timing contract: min-of-outer of mean-of-inner, median of "
                 "`trials` independent trials (see `bench/layer/timing.py`).\n")
    lines.append("| model | phase | bs | op | T | d_in | d_out | kernel | "
                 "preset | median_us | min_us | max_us | spread_% | est_ms |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        s = r.stats
        lines.append(
            f"| {r.model} | {r.phase} | {r.batch} | {r.op} | {r.T} | "
            f"{r.d_in} | {r.d_out} | {r.kernel_path} | {r.preset_name} | "
            f"{s['median_us']:.2f} | {s['min_us']:.2f} | {s['max_us']:.2f} | "
            f"{s['spread_pct']:.2f} | {r.est_call_ms:.3f} |\n"
        )
    # Per-(model,phase,bs) sum of 4 ops (useful for replacement-vs-BF16 comparison)
    lines.append("\n## Sum of 4 fused ops per (model, phase, bs)\n\n")
    lines.append("Replacement-ready number: this is the total kernel time "
                 "the W4A4 path contributes to a single decoder layer "
                 "(attention + MLP projections), NOT including attention "
                 "/ norms / RoPE / residuals.\n\n")
    lines.append("| model | phase | bs | Σ median_us |\n")
    lines.append("|---|---|---|---|\n")
    groups: dict[tuple[str, str, int], float] = {}
    for r in rows:
        key = (r.model, r.phase, r.batch)
        groups[key] = groups.get(key, 0.0) + r.stats["median_us"]
    for (model, phase, bs), tot in sorted(groups.items()):
        lines.append(f"| {model} | {phase} | {bs} | {tot:.2f} |\n")
    out_path.write_text("".join(lines))

def _write_validation_log(
    rows: list[Row], out_path: Path, meta: dict,
) -> None:
    spread_vals = [r.stats["spread_pct"] for r in rows]
    n_over_1pct = sum(1 for s in spread_vals if s > 1.0)
    n_over_5pct = sum(1 for s in spread_vals if s > 5.0)
    lines = [
        "# W4A4 bench validation log\n\n",
        "## Environment\n",
        f"- torch: {torch.__version__}\n",
        f"- cuda: {torch.version.cuda}\n",
        f"- device: {torch.cuda.get_device_name(0)}\n",
        f"- commit: {meta.get('commit','?')}\n",
        f"- rows: {len(rows)}\n\n",
        "## Spread summary (spread_pct = (max-min)/median · 100)\n",
        f"- rows with spread > 1.0%: {n_over_1pct} / {len(rows)}\n",
        f"- rows with spread > 5.0%: {n_over_5pct} / {len(rows)}\n",
        f"- max spread: {max(spread_vals):.2f}%\n",
        f"- median spread: {sorted(spread_vals)[len(spread_vals)//2]:.2f}%\n\n",
        "## Dispatch path distribution\n",
    ]
    paths: dict[str, int] = {}
    for r in rows:
        paths[r.kernel_path] = paths.get(r.kernel_path, 0) + 1
    for p, n in paths.items():
        lines.append(f"- {p}: {n}\n")
    out_path.write_text("".join(lines))

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory; must already exist (driver script creates it).",
    )
    ap.add_argument(
        "--models", nargs="*", default=None,
        help="Subset of Qwen3 model names to test (default: all 3).",
    )
    ap.add_argument(
        "--phases", nargs="*", default=None, choices=("prefill", "decode"),
        help="Subset of phases (default: both).",
    )
    ap.add_argument(
        "--batch-sizes", nargs="*", type=int, default=None,
        help="Subset of batch sizes (default: 4 8 16 32).",
    )
    ap.add_argument(
        "--device", type=str, default="cuda:0",
    )
    ap.add_argument(
        "--commit", type=str, default="?",
        help="Git commit for provenance; passed through from the driver.",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter the sweep
    models = [m for m in QWEN3_MODELS
              if args.models is None or m.name in args.models]
    phases = [p for p in PHASES
              if args.phases is None or p.name in args.phases]
    bses = list(args.batch_sizes) if args.batch_sizes else list(BATCH_SIZES)

    device = torch.device(args.device)
    assert torch.cuda.is_available(), "CUDA required"
    print(f"[w4a4-bench] device={device} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[w4a4-bench] sweep: {len(models)}m × {len(phases)}p × "
          f"{len(bses)}bs × 4op = {len(models)*len(phases)*len(bses)*4} rows",
          flush=True)

    all_rows: list[Row] = []
    t0 = time.time()
    for cfg in models:
        for phase in phases:
            for bs in bses:
                print(f"[{time.time()-t0:6.1f}s] === {cfg.name} / "
                      f"{phase.name} / bs={bs} ===", flush=True)
                rows = _run_one_triple(cfg, phase, bs, device=device)
                all_rows.extend(rows)
                # Flush JSON after every triple so a crash doesn't nuke
                # the run — same durability pattern as the BF16 bench.
                _dump_json(all_rows, out_dir / "bench_w4a4_per_op.json",
                           meta={"commit": args.commit})

    # Final deliverables
    _write_summary_md(all_rows, out_dir / "bench_w4a4_summary.md")
    _write_validation_log(
        all_rows, out_dir / "VALIDATION_LOG.md",
        meta={"commit": args.commit},
    )
    print(f"[w4a4-bench] done in {time.time()-t0:.1f}s; wrote {len(all_rows)} rows "
          f"→ {out_dir}", flush=True)
    return 0

def _dump_json(rows: list[Row], path: Path, *, meta: dict) -> None:
    payload = {
        "meta": {
            "commit": meta.get("commit", "?"),
            "torch":  torch.__version__,
            "cuda":   torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "n_rows": len(rows),
        },
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2))

if __name__ == "__main__":
    raise SystemExit(main())
