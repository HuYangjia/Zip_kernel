"""Bench (D4 smoke / D3 pre-flight): CUDA Graph replay vs eager on
the 17 launch_sparse audit shapes.

Contract (per shape):
  t_eager = _min_of_means_us(eager_fn, warmup=200, outer=10, inner=100)
  t_graph = _min_of_means_us(graph_fn, warmup=200, outer=10, inner=100)

Each of (eager, graph) is measured K=5 times in *alternating* order
per trial (graph-first on even trials, eager-first on odd) to absorb
per-trial clock transients per the GPU microbench memo.  The median
across trials is the reported number; [min, max] serves as a
stability indicator.

Verdict (per shape):
  speedup = t_eager_med / t_graph_med
  improvement_pct = (1 - t_graph_med / t_eager_med) * 100

Global pass (for D4 sign-off, NOT this script):
  - >=15 of 17 shapes reach improvement_pct >= 50%
  - max_abs_diff <= 1e-3 between eager and graph output (checked
    by ``test_cuda_graph_cluster_parity.py``; this bench assumes it)

Output:
  cuda_kernel/logs/phase3_optimization/cuda_graph_bench/bench.json
  cuda_kernel/logs/phase3_optimization/cuda_graph_bench/bench.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

# ---- path anchoring -------------------------------------------------------
_THIS = Path(__file__).resolve()
KERNEL_ROOT = _THIS.parents[2]
IMPORT_ROOT = KERNEL_ROOT.parent
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

from kernel.backend import graph_cache as gc_mod        # noqa: E402
from kernel.backend import v9_linear_forward            # noqa: E402
from kernel.tools.profile._phase1_shapes import (       # noqa: E402
    build_shape_inputs,
)
from kernel.tools.profile.audit_launch_tax import (     # noqa: E402
    AUDIT_SHAPES,
    _register_audit_shapes,
)

# Register the 17 audit tags in PHASE1_SHAPES_BY_TAG so build_shape_inputs
# accepts them (same side-effect the audit itself relies on).
_register_audit_shapes()


# ---------------------------------------------------------------------------
# Timer — strict compliance with the GPU microbench memo
# ---------------------------------------------------------------------------

def _min_of_means_us(
    fn: Callable[[], None],
    *,
    warmup: int,
    outer: int,
    inner: int,
) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    means: List[float] = []
    for _ in range(outer):
        start_ev.record()
        for _ in range(inner):
            fn()
        end_ev.record()
        torch.cuda.synchronize()
        means.append(start_ev.elapsed_time(end_ev) * 1000.0 / inner)
    return min(means)


# ---------------------------------------------------------------------------
# Per-shape bench
# ---------------------------------------------------------------------------

@dataclass
class BenchRecord:
    t_eager_med_us: float
    t_eager_min_us: float
    t_eager_max_us: float
    t_graph_med_us: float
    t_graph_min_us: float
    t_graph_max_us: float
    speedup: float
    improvement_pct: float
    trials: int


def _bench_one(tag: str, K: int, warmup: int, outer: int, inner: int) -> Dict:
    bundle = build_shape_inputs(tag)
    X, W = bundle.X, bundle.W

    # Eager fn.
    def eager_fn() -> None:
        v9_linear_forward(X, W)

    # Graph fn.  Force policy so eligibility never bails.
    prev_pol = gc_mod.set_cuda_graph_policy("force")
    try:
        # Prime the cache once outside the timed region.
        _ = gc_mod.v9_linear_forward_cuda_graph(X, W)

        def graph_fn() -> None:
            gc_mod.v9_linear_forward_cuda_graph(X, W)

        eager_trials: List[float] = []
        graph_trials: List[float] = []
        for k in range(K):
            if k % 2 == 0:
                # graph-first
                t_g = _min_of_means_us(
                    graph_fn, warmup=warmup, outer=outer, inner=inner
                )
                t_e = _min_of_means_us(
                    eager_fn, warmup=warmup, outer=outer, inner=inner
                )
            else:
                t_e = _min_of_means_us(
                    eager_fn, warmup=warmup, outer=outer, inner=inner
                )
                t_g = _min_of_means_us(
                    graph_fn, warmup=warmup, outer=outer, inner=inner
                )
            eager_trials.append(t_e)
            graph_trials.append(t_g)
    finally:
        # Always clean up per-bench so adjacent shapes don't see stale
        # LRU state.
        gc_mod.clear_cuda_graph_cache()
        gc_mod.set_cuda_graph_policy(prev_pol)

    t_eager_med = statistics.median(eager_trials)
    t_graph_med = statistics.median(graph_trials)
    rec = BenchRecord(
        t_eager_med_us=t_eager_med,
        t_eager_min_us=min(eager_trials),
        t_eager_max_us=max(eager_trials),
        t_graph_med_us=t_graph_med,
        t_graph_min_us=min(graph_trials),
        t_graph_max_us=max(graph_trials),
        speedup=t_eager_med / t_graph_med if t_graph_med > 0 else float("inf"),
        improvement_pct=(1.0 - t_graph_med / t_eager_med) * 100.0
        if t_eager_med > 0 else 0.0,
        trials=K,
    )
    return asdict(rec) | {"tag": tag, "eager_trials_us": eager_trials,
                          "graph_trials_us": graph_trials}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _render_md(rows: List[Dict], meta: Dict) -> str:
    lines: List[str] = []
    lines.append("# CUDA Graph vs Eager — launch_sparse cluster bench")
    lines.append("")
    lines.append(f"- run_id: `{meta['run_id']}`")
    lines.append(f"- device: `{meta['device']}`")
    lines.append(f"- schedule: warmup={meta['warmup']}, outer={meta['outer']},"
                 f" inner={meta['inner']}, K={meta['K']}")
    lines.append(f"- shapes evaluated: {len(rows)}")
    lines.append("")

    # Summary table.
    lines.append("| shape | T | d_in | d_out | eager_us | graph_us | speedup | improvement |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        a = next(a for a in AUDIT_SHAPES if a.tag == r["tag"])
        lines.append(
            f"| {r['tag']} | {a.T} | {a.d_in} | {a.d_out} | "
            f"{r['t_eager_med_us']:.2f} | {r['t_graph_med_us']:.2f} | "
            f"{r['speedup']:.2f}x | {r['improvement_pct']:+.1f}% |"
        )
    lines.append("")

    # Distribution.
    pct_vals = [r["improvement_pct"] for r in rows]
    above_50 = sum(1 for p in pct_vals if p >= 50.0)
    above_30 = sum(1 for p in pct_vals if p >= 30.0)
    regress  = sum(1 for p in pct_vals if p < 0.0)
    lines.append("## Distribution")
    lines.append(f"- improvement >= 50%: **{above_50} / {len(rows)}**")
    lines.append(f"- improvement >= 30%: {above_30} / {len(rows)}")
    lines.append(f"- regressions (<0%): {regress} / {len(rows)}")
    if pct_vals:
        lines.append(f"- median improvement: {statistics.median(pct_vals):+.1f}%")
        lines.append(f"- min / max improvement: "
                     f"{min(pct_vals):+.1f}% / {max(pct_vals):+.1f}%")
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=str(
        KERNEL_ROOT / "cuda_kernel" / "logs" / "phase3_optimization"
                      / "cuda_graph_bench"
    ))
    ap.add_argument("--only", default="",
                    help="comma-sep subset of tags; default = all 17")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--outer",  type=int, default=10)
    ap.add_argument("--inner",  type=int, default=100)
    ap.add_argument("--K",      type=int, default=5)
    ap.add_argument("--smoke",  action="store_true",
                    help="Fast smoke: single shape, K=2, warmup=50.")
    args = ap.parse_args()

    if args.smoke:
        args.warmup = 50
        args.outer = 3
        args.inner = 100
        args.K = 2
        shapes = (AUDIT_SHAPES[0],)
    elif args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        shapes = tuple(a for a in AUDIT_SHAPES if a.tag in want)
    else:
        shapes = AUDIT_SHAPES

    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    t_start = _dt.datetime.now().isoformat(timespec="seconds")
    for a in shapes:
        print(f"[bench] {a.tag} ... ", end="", flush=True)
        rec = _bench_one(a.tag, K=args.K, warmup=args.warmup,
                         outer=args.outer, inner=args.inner)
        print(f"eager={rec['t_eager_med_us']:.1f}us "
              f"graph={rec['t_graph_med_us']:.1f}us "
              f"({rec['improvement_pct']:+.1f}%)")
        rows.append(rec)
    t_end = _dt.datetime.now().isoformat(timespec="seconds")

    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    device = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    meta = {
        "run_id": run_id,
        "t_start": t_start,
        "t_end": t_end,
        "device": device,
        "warmup": args.warmup,
        "outer": args.outer,
        "inner": args.inner,
        "K": args.K,
        "smoke": args.smoke,
    }
    payload = {"meta": meta, "records": rows}
    (out_dir / "bench.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "bench.md").write_text(_render_md(rows, meta))
    print(f"\n[bench] wrote {out_dir/'bench.json'} and {out_dir/'bench.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
