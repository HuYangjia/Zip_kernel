"""Audit: re-validate ``launch_sparse`` cluster under tightened timer.

Motivation
----------
The Phase 2 roadmap currently assigns 17 shapes to the ``launch_sparse``
cluster based on ``phase1_measure_launch_tax.py``'s per-shape
``launch_tax_pct_of_plain > 50%`` verdict.  That script used
``(warmup=50, outer=3, inner=100)`` — the same budget that was later
proven to fabricate two false clusters (``x_zero_anomaly`` and
``epilogue_fma_bound``).  Before committing 3-4 engineering days to a
CUDA-Graph based production integration, we want to know:

  "Is the >=50% launch-tax verdict still real under the tightened
   (warmup=200, outer=10, inner=100) × K=5 trials schedule?"

Method
------
For each of the 17 shapes in the cluster, we measure two wall-clock
quantities under identical timer settings::

    t_plain_us  = wall-clock of eager v9_linear_forward(X, W)
    t_graph_us  = wall-clock of CUDA Graph .replay() of the same call

and report::

    launch_tax_us  = t_plain - t_graph
    launch_tax_pct = launch_tax_us / t_plain * 100

Each is timed with ``_min_of_means_us(warmup=200, outer=10, inner=100)``.
The whole measurement is repeated ``K = 5`` times in alternating
(plain, graph) order per trial, to absorb per-trial clock transients
(see memo on GPU microbench hygiene).  We report the median as the
verdict, with [min, max] as a stability indicator.

Verdict rules (applied per shape)::

    pct_median >= 50%             -> CONFIRMED       (stays in cluster)
    30% <= pct_median < 50%       -> DEGRADED        (stays, lower ROI)
    pct_median < 30%              -> REJECTED        (reclassify)
    |pct_max - pct_min| >= 20pp   -> UNSTABLE flag   (audit-only)

Scope
-----
This script is a one-shot audit.  It does NOT update
``shape_clusters.csv`` or ``shape_targets.csv`` — re-classification is
a separate step gated on the audit outcome.  The script writes its
results to::

    cuda_kernel/logs/phase2_microscope/audit_launch_tax/launch_tax_audit.json
    cuda_kernel/logs/phase2_microscope/audit_launch_tax/launch_tax_audit.md

Shape set
---------
The 17 cluster members are hard-coded below (read from
``shape_clusters.csv`` at the time this audit was designed).  We do
NOT re-parse the CSV at runtime — we want the audit input to be
auditable too.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# --- path anchoring
_THIS = Path(__file__).resolve()
KERNEL_ROOT = _THIS.parents[2]
IMPORT_ROOT = KERNEL_ROOT.parent
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

from kernel.tools.profile._phase1_shapes import (  # noqa: E402
    PHASE1_SHAPES_BY_TAG,
    PhaseShape,
    build_shape_inputs,
)

DEFAULT_OUT_DIR = (
    KERNEL_ROOT / "cuda_kernel" / "logs" / "phase2_microscope" / "audit_launch_tax"
)


# ---------------------------------------------------------------------------
# Shape set — 17 members of the launch_sparse cluster per
# shape_clusters.csv (commit 0dbf1bf).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AuditShape:
    tag: str
    T: int
    d_in: int
    d_out: int
    model: str
    proj: str


AUDIT_SHAPES: Tuple[AuditShape, ...] = (
    AuditShape("audit_0p6B_q_T1_1024_2048",      1,  1024,  2048, "Qwen3-0.6B", "q_proj"),
    AuditShape("audit_0p6B_q_T8_1024_2048",      8,  1024,  2048, "Qwen3-0.6B", "q_proj"),
    AuditShape("audit_0p6B_kv_T1_1024_2048",     1,  1024,  2048, "Qwen3-0.6B", "kv_proj"),
    AuditShape("audit_0p6B_kv_T8_1024_2048",     8,  1024,  2048, "Qwen3-0.6B", "kv_proj"),
    AuditShape("audit_0p6B_o_T8_2048_1024",      8,  2048,  1024, "Qwen3-0.6B", "o_proj"),
    AuditShape("audit_0p6B_gu_T1_1024_6144",     1,  1024,  6144, "Qwen3-0.6B", "gate_up_proj"),
    AuditShape("audit_0p6B_gu_T8_1024_6144",     8,  1024,  6144, "Qwen3-0.6B", "gate_up_proj"),
    AuditShape("audit_1p7B_q_T1_2048_2048",      1,  2048,  2048, "Qwen3-1.7B", "q_proj"),
    AuditShape("audit_1p7B_q_T8_2048_2048",      8,  2048,  2048, "Qwen3-1.7B", "q_proj"),
    AuditShape("audit_1p7B_kv_T1_2048_2048",     1,  2048,  2048, "Qwen3-1.7B", "kv_proj"),
    AuditShape("audit_1p7B_kv_T8_2048_2048",     8,  2048,  2048, "Qwen3-1.7B", "kv_proj"),
    AuditShape("audit_1p7B_o_T1_2048_2048",      1,  2048,  2048, "Qwen3-1.7B", "o_proj"),
    AuditShape("audit_1p7B_o_T8_2048_2048",      8,  2048,  2048, "Qwen3-1.7B", "o_proj"),
    AuditShape("audit_1p7B_gu_T1_2048_12288",    1,  2048, 12288, "Qwen3-1.7B", "gate_up_proj"),
    AuditShape("audit_1p7B_gu_T8_2048_12288",    8,  2048, 12288, "Qwen3-1.7B", "gate_up_proj"),
    AuditShape("audit_4B_q_T1_2560_4096",        1,  2560,  4096, "Qwen3-4B",   "q_proj"),
    AuditShape("audit_4B_gu_T1_2560_19456",      1,  2560, 19456, "Qwen3-4B",   "gate_up_proj"),
)


def _register_audit_shapes() -> None:
    """Monkey-patch the 17 audit shapes into PHASE1_SHAPES_BY_TAG so
    ``build_shape_inputs`` accepts them.  Idempotent.
    """
    for a in AUDIT_SHAPES:
        if a.tag in PHASE1_SHAPES_BY_TAG:
            continue
        ps = PhaseShape(
            tag=a.tag,
            T=a.T,
            d_in=a.d_in,
            d_out=a.d_out,
            hp_ratio=0.05,
            model=a.model,
            proj=a.proj,
            note=f"launch_sparse audit (commit 0dbf1bf cluster member)",
        )
        PHASE1_SHAPES_BY_TAG[a.tag] = ps


# ---------------------------------------------------------------------------
# Timer — strict compliance with the memo
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
# Graph capture helper
# ---------------------------------------------------------------------------
def _build_graph_replayer(X, W) -> Tuple[Optional[Callable[[], None]], Optional[str]]:
    from kernel.backend import v9_linear_forward

    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                v9_linear_forward(X, W)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=pool):
            y = v9_linear_forward(X, W)
        del y
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return (lambda: g.replay()), None


# ---------------------------------------------------------------------------
# Per-shape K-trial audit
# ---------------------------------------------------------------------------
@dataclass
class TrialRecord:
    t_plain_us: float
    t_graph_us: float
    tax_us: float
    tax_pct: float


def _audit_one(
    shape: AuditShape,
    *,
    K: int,
    warmup: int,
    outer: int,
    inner: int,
) -> Dict[str, Any]:
    from kernel.backend import v9_linear_forward

    b = build_shape_inputs(shape.tag)
    X, W = b.X, b.W

    plain_fn = lambda: v9_linear_forward(X, W)
    replay_fn, capture_error = _build_graph_replayer(X, W)

    if replay_fn is None:
        return {
            "tag": shape.tag,
            "T": shape.T, "d_in": shape.d_in, "d_out": shape.d_out,
            "model": shape.model, "proj": shape.proj,
            "graph_capture_error": capture_error,
            "trials": [],
            "verdict": "CAPTURE_FAILED",
        }

    trials: List[TrialRecord] = []
    for k in range(K):
        # Alternate plain-first / graph-first across trials to avoid
        # any systematic ordering bias.
        if k % 2 == 0:
            tp = _min_of_means_us(plain_fn, warmup=warmup, outer=outer, inner=inner)
            tg = _min_of_means_us(replay_fn, warmup=warmup, outer=outer, inner=inner)
        else:
            tg = _min_of_means_us(replay_fn, warmup=warmup, outer=outer, inner=inner)
            tp = _min_of_means_us(plain_fn, warmup=warmup, outer=outer, inner=inner)
        tax_us = tp - tg
        tax_pct = tax_us / tp * 100.0 if tp > 0 else 0.0
        trials.append(TrialRecord(tp, tg, tax_us, tax_pct))

    pct_vals = [t.tax_pct for t in trials]
    tax_vals = [t.tax_us for t in trials]
    plain_vals = [t.t_plain_us for t in trials]
    graph_vals = [t.t_graph_us for t in trials]

    pct_med = statistics.median(pct_vals)
    pct_min = min(pct_vals)
    pct_max = max(pct_vals)
    range_pp = pct_max - pct_min

    if pct_med >= 50.0:
        verdict = "CONFIRMED"
    elif pct_med >= 30.0:
        verdict = "DEGRADED"
    else:
        verdict = "REJECTED"

    flags: List[str] = []
    if range_pp >= 20.0:
        flags.append(f"UNSTABLE(range={range_pp:.1f}pp)")

    return {
        "tag": shape.tag,
        "T": shape.T, "d_in": shape.d_in, "d_out": shape.d_out,
        "model": shape.model, "proj": shape.proj,
        "trials": [t.__dict__ for t in trials],
        "tax_pct_median": round(pct_med, 2),
        "tax_pct_min": round(pct_min, 2),
        "tax_pct_max": round(pct_max, 2),
        "tax_pct_range_pp": round(range_pp, 2),
        "t_plain_us_median": round(statistics.median(plain_vals), 3),
        "t_graph_us_median": round(statistics.median(graph_vals), 3),
        "tax_us_median": round(statistics.median(tax_vals), 3),
        "verdict": verdict,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
def _render_md(summary: Dict[str, Any]) -> str:
    records = summary["records"]
    lines: List[str] = []
    lines.append("# Launch-tax audit under tightened timer")
    lines.append("")
    lines.append(
        f"_Generated {summary['timestamp_utc']}; "
        f"K={summary['K']} trials × (warmup={summary['warmup']}, "
        f"outer={summary['outer']}, inner={summary['inner']}); "
        f"alternating plain/graph order per trial._"
    )
    lines.append("")
    lines.append(
        "## Motivation\n\n"
        "The Phase 2 roadmap assigned 17 shapes to the ``launch_sparse`` "
        "cluster based on ``launch_tax_pct >= 50%`` measured under "
        "``(warmup=50, outer=3, inner=100)`` — the same budget that was "
        "later proven to fabricate the ``x_zero_anomaly`` and "
        "``epilogue_fma_bound`` clusters.  This audit re-measures the "
        "verdict under the tightened schedule before R49 commits to "
        "CUDA-Graph based optimisation."
    )
    lines.append("")
    lines.append(
        "## Verdict rule\n\n"
        "* ``CONFIRMED``: median launch_tax_pct >= 50% — stays in cluster.\n"
        "* ``DEGRADED``: 30% <= median launch_tax_pct < 50% — stays, lower ROI.\n"
        "* ``REJECTED``: median launch_tax_pct < 30% — reclassify.\n"
        "* ``UNSTABLE`` flag: max-min range >= 20pp — audit-only (still "
        "applies the median-based verdict).\n"
    )

    # Aggregate counts
    vcount: Dict[str, int] = {}
    for r in records:
        vcount[r["verdict"]] = vcount.get(r["verdict"], 0) + 1
    lines.append("## Summary")
    lines.append("")
    lines.append("| verdict | n |")
    lines.append("|---|---:|")
    for v in ("CONFIRMED", "DEGRADED", "REJECTED", "CAPTURE_FAILED"):
        lines.append(f"| {v} | {vcount.get(v, 0)} |")
    lines.append("")

    lines.append("## Per-shape verdict")
    lines.append("")
    lines.append(
        "| tag | T | d_in | d_out | "
        "plain (us) | graph (us) | tax (us) | "
        "tax% median | [min,max] | verdict | flags |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
    )
    for r in sorted(records, key=lambda x: (x.get("tax_pct_median", -1))):
        if r["verdict"] == "CAPTURE_FAILED":
            lines.append(
                f"| `{r['tag']}` | {r['T']} | {r['d_in']} | {r['d_out']} "
                f"| - | - | - | - | - | CAPTURE_FAILED "
                f"| {r.get('graph_capture_error', '')} |"
            )
            continue
        lines.append(
            f"| `{r['tag']}` | {r['T']} | {r['d_in']} | {r['d_out']} "
            f"| {r['t_plain_us_median']:.2f} | {r['t_graph_us_median']:.2f} "
            f"| {r['tax_us_median']:.2f} "
            f"| {r['tax_pct_median']:.1f}% "
            f"| [{r['tax_pct_min']:.1f}, {r['tax_pct_max']:.1f}] "
            f"| {r['verdict']} | {', '.join(r['flags']) if r['flags'] else '-'} |"
        )
    lines.append("")

    # Conclusion hint (mechanical; operator interprets).
    n_total = len(records)
    n_confirmed = vcount.get("CONFIRMED", 0)
    n_degraded = vcount.get("DEGRADED", 0)
    n_rejected = vcount.get("REJECTED", 0)
    lines.append("## Cluster stability")
    lines.append("")
    lines.append(
        f"Of {n_total} cluster members: {n_confirmed} CONFIRMED, "
        f"{n_degraded} DEGRADED, {n_rejected} REJECTED."
    )
    if n_rejected == 0 and n_degraded == 0:
        lines.append(
            "\nCluster is **stable under tightened measurement** — R49 "
            "may proceed with CUDA-Graph integration as the #1 ROI lever."
        )
    elif n_rejected >= n_total // 2:
        lines.append(
            "\nCluster is **materially reduced under tightened measurement** — "
            "R49 must reclassify the rejected shapes before committing."
        )
    else:
        lines.append(
            "\nCluster is **partially stable**: CUDA-Graph work remains "
            "worthwhile for the confirmed subset, but ROI must be "
            "recomputed over the reduced shape count."
        )
    lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trials", type=int, default=5,
                        help="K = independent trials per shape")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--outer", type=int, default=10)
    parser.add_argument("--inner", type=int, default=100)
    parser.add_argument("--only", nargs="+", default=None,
                        help="subset of audit tags")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _register_audit_shapes()

    shapes = AUDIT_SHAPES
    if args.only:
        want = set(args.only)
        shapes = tuple(s for s in AUDIT_SHAPES if s.tag in want)
    if not shapes:
        print("No audit shapes selected; exiting.")
        return 1

    print(
        f"Auditing {len(shapes)} shapes "
        f"(K={args.trials}, warmup={args.warmup}, outer={args.outer}, "
        f"inner={args.inner})..."
    )

    t0 = time.time()
    records: List[Dict[str, Any]] = []
    for i, s in enumerate(shapes):
        print(f"  [{i+1}/{len(shapes)}] {s.tag}  T={s.T} {s.d_in}->{s.d_out}",
              flush=True)
        rec = _audit_one(
            s, K=args.trials,
            warmup=args.warmup, outer=args.outer, inner=args.inner,
        )
        if rec["verdict"] == "CAPTURE_FAILED":
            print(f"      capture failed: {rec['graph_capture_error']}")
        else:
            print(
                f"      plain={rec['t_plain_us_median']:.2f}us  "
                f"graph={rec['t_graph_us_median']:.2f}us  "
                f"tax%={rec['tax_pct_median']:.1f}% "
                f"[{rec['tax_pct_min']:.1f}, {rec['tax_pct_max']:.1f}] "
                f"=> {rec['verdict']}"
                + (f"  {','.join(rec['flags'])}" if rec['flags'] else "")
            )
        records.append(rec)

    summary: Dict[str, Any] = {
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "K": args.trials,
        "warmup": args.warmup,
        "outer": args.outer,
        "inner": args.inner,
        "wall_seconds": round(time.time() - t0, 1),
        "records": records,
    }

    out_json = args.out_dir / "launch_tax_audit.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path = args.out_dir / "launch_tax_audit.md"
    md_path.write_text(_render_md(summary), encoding="utf-8")

    print()
    print(f"Written: {out_json}")
    print(f"Written: {md_path}")
    print(f"Total wall-clock: {summary['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
