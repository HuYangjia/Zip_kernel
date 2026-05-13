"""W4A4 NAIVE-backend bench driver: same sweep as bench_w4a4_fused_ops.py.

For each (model, phase, batch) in {Qwen3-4B,8B,14B} × {prefill,decode} ×
{4,8,16,32}, measure the END-TO-END wall time of the naive 4-kernel
pipeline (activation_quant → dense_gemm → sparse_gemm → reduce_sum) on
each of the 4 fused projection groups (qkv_fused / o_proj /
gate_up_fused / down_proj).

The per-op timing number this script reports is the sum of all four
naive kernel launches for ONE projection call — directly comparable to
the optimised ``legacy_mma`` two-kernel path measured by
``bench_w4a4_fused_ops.py``.

Timing policy (mirrors bench_w4a4_fused_ops.py; adaptive per-shape):

  * decode   →  STRICT               (warmup=500, outer=20, inner=200, trials=5)
  * prefill  →  adaptive, FLOPs-gated (TINY / MED / ADAPTIVE_PREFILL)

Outputs
-------
  <out-dir>/bench_w4a4_naive_per_op.json
  <out-dir>/bench_w4a4_naive_summary.md
  <out-dir>/VALIDATION_LOG.md
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

_THIS_FILE = Path(__file__).resolve()
_PROJ_ROOT = _THIS_FILE.parents[3]  # .../HKUST
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from kernel.bench.configs.qwen3_shapes import (  # noqa: E402
    BATCH_SIZES, PHASES, QWEN3_MODELS, Qwen3Config, PhaseConfig,
)
from kernel.bench.layer.qwen3_w4a4_ops_naive import (  # noqa: E402
    NaiveOpBundle, build_four_op_callables_naive,
)
from kernel.bench.layer.timing import (  # noqa: E402
    ADAPTIVE_PREFILL, STRICT, TimingStats, measure,
)

# ---------------------------------------------------------------------------
# Preset selection — conservative for naive kernels (they are 10-100x
# slower than the optimised path, so "very heavy" threshold is tighter).
# ---------------------------------------------------------------------------
_INT4_NAIVE_OPS_PER_SEC: float = 20.0e12   # naive scalar IMAD ≈ 20 Tops/s
                                            # (empirical upper-bound estimate)
_HEAVY_MS = 20.0
_MED_MS   = 3.0

_TINY_PREFILL_PRESET = dict(warmup=10, outer=3,  inner=3,  trials=5)
_MED_PREFILL_PRESET  = dict(warmup=20, outer=5,  inner=8,  trials=5)


def _estimate_call_ms(T: int, d_in: int, d_out: int) -> float:
    """Rough estimate for one naive 4-kernel pipeline call."""
    # dense dominates; sparse adds ~5% density, reduce/quant are small.
    flops = 2.0 * T * d_in * d_out
    return flops / _INT4_NAIVE_OPS_PER_SEC * 1.0e3


def _pick_preset(phase_name: str, bundles: list[NaiveOpBundle]) -> dict:
    if phase_name == "decode":
        # Naive decode is still mostly small — keep strict for stability.
        return STRICT
    est_ms = max(_estimate_call_ms(b.T, b.d_in, b.d_out) for b in bundles)
    if est_ms > _HEAVY_MS:
        return _TINY_PREFILL_PRESET
    if est_ms > _MED_MS:
        return _MED_PREFILL_PRESET
    return ADAPTIVE_PREFILL


_PRESET_NAMES = {
    id(STRICT):                "STRICT",
    id(ADAPTIVE_PREFILL):      "ADAPTIVE_PREFILL",
    id(_TINY_PREFILL_PRESET):  "TINY_PREFILL",
    id(_MED_PREFILL_PRESET):   "MED_PREFILL",
}

def _preset_name(p: dict) -> str:
    return _PRESET_NAMES.get(id(p), "CUSTOM")


# ---------------------------------------------------------------------------
@dataclass
class Row:
    backend: str              # always "naive"
    model: str
    phase: str
    batch: int
    seqlen: int
    past_kv_len: int
    op: str                   # qkv_fused / o_proj / gate_up_fused / down_proj
    d_in: int
    d_out: int
    T: int
    kernel_path: str          # "naive_4kernel"
    preset_name: str
    est_call_ms: float
    stats: dict


def _run_one_triple(
    cfg: Qwen3Config, phase: PhaseConfig, bs: int,
    *, device: torch.device,
) -> list[Row]:
    four = build_four_op_callables_naive(
        cfg, batch=bs, seqlen=phase.seqlen, device=device
    )
    bundles = [b for _, b, _ in four.as_list()]
    preset = _pick_preset(phase.name, bundles)
    pname = _preset_name(preset)

    rows: list[Row] = []
    for op_name, bundle, fn in four.as_list():
        # Smoke call to surface launch errors early.
        _ = fn()
        torch.cuda.synchronize(device)

        stats: TimingStats = measure(fn, device=device, **preset)
        est_ms = _estimate_call_ms(bundle.T, bundle.d_in, bundle.d_out)
        rows.append(Row(
            backend="naive",
            model=cfg.name, phase=phase.name, batch=bs,
            seqlen=phase.seqlen, past_kv_len=phase.past_kv_len,
            op=op_name, d_in=bundle.d_in, d_out=bundle.d_out,
            T=bundle.T, kernel_path="naive_4kernel",
            preset_name=pname, est_call_ms=est_ms,
            stats=stats.as_dict(),
        ))
        print(
            f"  [naive] {cfg.name:<10} {phase.name:<7} bs={bs:<3} "
            f"{op_name:<15} T={bundle.T:<6} d_in={bundle.d_in:<6} "
            f"d_out={bundle.d_out:<6} preset={pname:<17} "
            f"median={stats.median_us:10.2f}us  spread={stats.spread_pct:5.2f}%",
            flush=True,
        )
        del fn, bundle
    del four, bundles
    torch.cuda.empty_cache()
    return rows


def _write_summary_md(rows: list[Row], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# W4A4 NAIVE-backend kernel time\n")
    lines.append("Per-op number = wall time of the 4-kernel naive pipeline "
                 "(activation_quant_naive → dense_gemm_naive → "
                 "sparse_gemm_naive → reduce_sum_naive).  Directly "
                 "comparable to the optimised `legacy_mma` path.\n")
    lines.append("| model | phase | bs | op | T | d_in | d_out | "
                 "preset | median_us | min_us | max_us | spread_% | est_ms |\n")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        s = r.stats
        lines.append(
            f"| {r.model} | {r.phase} | {r.batch} | {r.op} | {r.T} | "
            f"{r.d_in} | {r.d_out} | {r.preset_name} | "
            f"{s['median_us']:.2f} | {s['min_us']:.2f} | "
            f"{s['max_us']:.2f} | {s['spread_pct']:.2f} | "
            f"{r.est_call_ms:.3f} |\n"
        )
    lines.append("\n## Sum of 4 fused ops per (model, phase, bs)\n\n")
    lines.append("| model | phase | bs | Σ median_us |\n|---|---|---|---|\n")
    groups: dict[tuple[str, str, int], float] = {}
    for r in rows:
        key = (r.model, r.phase, r.batch)
        groups[key] = groups.get(key, 0.0) + r.stats["median_us"]
    for (m, p, bs), tot in sorted(groups.items()):
        lines.append(f"| {m} | {p} | {bs} | {tot:.2f} |\n")
    out_path.write_text("".join(lines))


def _write_validation_log(rows: list[Row], out_path: Path, meta: dict) -> None:
    spread_vals = [r.stats["spread_pct"] for r in rows] or [0.0]
    n_over_1 = sum(1 for s in spread_vals if s > 1.0)
    n_over_5 = sum(1 for s in spread_vals if s > 5.0)
    sorted_s = sorted(spread_vals)
    out_path.write_text(
        "# W4A4 NAIVE bench validation log\n\n"
        "## Environment\n"
        f"- torch: {torch.__version__}\n"
        f"- cuda: {torch.version.cuda}\n"
        f"- device: {torch.cuda.get_device_name(0)}\n"
        f"- commit: {meta.get('commit','?')}\n"
        f"- backend: naive (csrc_naive, no MMA, no cp.async)\n"
        f"- sparsity: 5.0%\n"
        f"- rows: {len(rows)}\n\n"
        "## Spread summary\n"
        f"- rows with spread > 1.0%: {n_over_1} / {len(rows)}\n"
        f"- rows with spread > 5.0%: {n_over_5} / {len(rows)}\n"
        f"- max spread: {max(spread_vals):.2f}%\n"
        f"- median spread: {sorted_s[len(sorted_s)//2]:.2f}%\n"
    )


def _dump_json(rows: list[Row], path: Path, *, meta: dict) -> None:
    payload = {
        "meta": {
            "commit": meta.get("commit", "?"),
            "torch":  torch.__version__,
            "cuda":   torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "backend": "naive",
            "sparsity_pct": 5.0,
            "n_rows": len(rows),
        },
        "rows": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--phases", nargs="*", default=None,
                    choices=("prefill", "decode"))
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--commit", type=str, default="?")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m for m in QWEN3_MODELS
              if args.models is None or m.name in args.models]
    phases = [p for p in PHASES
              if args.phases is None or p.name in args.phases]
    bses = list(args.batch_sizes) if args.batch_sizes else list(BATCH_SIZES)

    device = torch.device(args.device)
    assert torch.cuda.is_available(), "CUDA required"
    print(f"[naive-bench] device={device} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[naive-bench] sweep: {len(models)}m × {len(phases)}p × "
          f"{len(bses)}bs × 4op = "
          f"{len(models)*len(phases)*len(bses)*4} rows", flush=True)

    all_rows: list[Row] = []
    t0 = time.time()
    for cfg in models:
        for phase in phases:
            for bs in bses:
                print(f"[{time.time()-t0:6.1f}s] === {cfg.name} / "
                      f"{phase.name} / bs={bs} ===", flush=True)
                rows = _run_one_triple(cfg, phase, bs, device=device)
                all_rows.extend(rows)
                _dump_json(
                    all_rows,
                    out_dir / "bench_w4a4_naive_per_op.json",
                    meta={"commit": args.commit},
                )

    _write_summary_md(all_rows, out_dir / "bench_w4a4_naive_summary.md")
    _write_validation_log(
        all_rows, out_dir / "VALIDATION_LOG.md",
        meta={"commit": args.commit},
    )
    print(f"[naive-bench] done in {time.time()-t0:.1f}s; "
          f"wrote {len(all_rows)} rows → {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
