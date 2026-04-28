"""Phase 1.4: CUDA-Graph based launch-tax measurement.

Goal (requirements.md §2.4)
---------------------------
For each representative shape, answer "how much of the wall-clock time
is *pure kernel-launch API overhead* as opposed to real kernel body
work?"  We use three numbers per shape::

    T_plain_us       = min-of-means over eager v9_linear_forward(X, W)
    T_graph_us       = min-of-means over graph.replay() of the same call
    T_kernel_body_us = GPU kernel accumulated time from nsys sqlite,
                       divided by iter count (optional; filled if the
                       --sqlite-dir flag points to a Phase 1 timeline run)

Then::

    launch_tax_us = T_plain_us - T_graph_us

is the kernel-launch API cost per forward (the graph replay amortises
all N kernel launches into a single graph-launch).  We additionally
report::

    body_vs_plain_ratio = T_kernel_body_us / T_plain_us

to cross-check: if this ratio is ~1.0 then the plain path is
launch-free already (no optimisation headroom via Graph); if <0.5 the
path is launch-dominated.

Output
------
``<out_dir>/launch_tax.json`` with one record per shape, plus a
``launch_tax.md`` human-readable summary.  Does NOT emit any nsys
artefact (that is Phase 1.2's job).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# --- path anchoring
_THIS = Path(__file__).resolve()
KERNEL_ROOT = _THIS.parents[2]
IMPORT_ROOT = KERNEL_ROOT.parent
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

from kernel.tools.profile._phase1_shapes import (  # noqa: E402
    PHASE1_SHAPES,
    PhaseShape,
    build_shape_inputs,
)

DEFAULT_OUT_DIR = KERNEL_ROOT / "cuda_kernel" / "logs" / "phase1_timeline"


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------

def _min_of_means_us(fn, *, warmup: int = 50, outer: int = 3, inner: int = 100) -> float:
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


def _time_plain(X, W) -> float:
    from kernel.backend import v9_linear_forward
    return _min_of_means_us(lambda: v9_linear_forward(X, W))


def _time_graph(X, W) -> Optional[Dict[str, Any]]:
    """Capture the forward under a CUDA Graph and time ``replay()``.

    Returns ``None`` with a diagnostic entry if capture fails (common
    causes: stream-ordered allocator hits, unsupported ops inside the
    forward).  A failure is not fatal — plain-only numbers are still
    useful.
    """
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
        # Provide a generous pool to avoid re-allocating inside capture.
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=pool):
            y = v9_linear_forward(X, W)
        del y  # discard; we only care about replay wall-clock
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "error": repr(exc),
        }

    def _do():
        g.replay()

    try:
        t = _min_of_means_us(_do)
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": f"replay failed: {exc!r}"}
    return {"ok": True, "t_graph_us": t}


# ---------------------------------------------------------------------------
# nsys sqlite: accumulated GPU kernel time per iteration
# ---------------------------------------------------------------------------

def _kernel_body_us_from_sqlite(
    sqlite_path: Path,
    inner_calls: int,
) -> Optional[float]:
    """Return the total GPU kernel runtime (in us) averaged per forward.

    nsys exports kernel events into the CUPTI_ACTIVITY_KIND_KERNEL table.
    We sum (end - start) of every kernel event, then divide by the number
    of profiled forward iterations that the inner driver executed
    (warmup iterations were also profiled; their kernel time will be in
    the sum as well, which is why we subtract them out here).
    """
    if not sqlite_path.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='CUPTI_ACTIVITY_KIND_KERNEL'"
        )
        if cur.fetchone() is None:
            return None
        cur.execute(
            "SELECT COALESCE(SUM(end - start), 0) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
        total_ns = int(cur.fetchone()[0])
        con.close()
    except sqlite3.Error:
        return None

    # Inner driver does N_WARMUP=50 warmup + `inner_calls` profiled iters.
    total_iters = 50 + inner_calls
    if total_iters == 0:
        return None
    return total_ns / total_iters / 1000.0  # ns -> us per forward


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _one_shape(shape: PhaseShape, out_dir: Path, inner_calls: int) -> Dict[str, Any]:
    b = build_shape_inputs(shape.tag)
    X, W = b.X, b.W

    # 1) Plain path
    t_plain = _time_plain(X, W)

    # 2) Graph path (may fail — that's OK).
    graph_info = _time_graph(X, W)

    # 3) Kernel body time from the matching nsys sqlite, if available.
    sqlite_path = out_dir / shape.tag / "report.sqlite"
    t_body = _kernel_body_us_from_sqlite(sqlite_path, inner_calls)

    entry: Dict[str, Any] = {
        "tag": shape.tag,
        "T": shape.T,
        "d_in": shape.d_in,
        "d_out": shape.d_out,
        "t_plain_us": round(t_plain, 3),
    }
    if graph_info is not None and graph_info.get("ok"):
        t_graph = graph_info["t_graph_us"]
        entry["t_graph_us"] = round(t_graph, 3)
        entry["launch_tax_us"] = round(t_plain - t_graph, 3)
        entry["launch_tax_pct_of_plain"] = round(
            (t_plain - t_graph) / t_plain * 100.0, 2
        )
    else:
        entry["graph_capture_error"] = (graph_info or {}).get("error", "unknown")

    if t_body is not None:
        entry["t_kernel_body_us"] = round(t_body, 3)
        entry["body_vs_plain_ratio"] = round(t_body / t_plain, 3)

    return entry


def _render_md(records: List[Dict[str, Any]], path: Path) -> None:
    lines: List[str] = []
    lines.append("# Phase 1 — Launch-Tax via CUDA Graph replay")
    lines.append("")
    lines.append(
        f"_Generated {_dt.datetime.utcnow().isoformat()}Z; "
        f"min-of-means (warmup=50, 3×100 windows)._"
    )
    lines.append("")
    lines.append(
        "Columns:  \n"
        "* `t_plain` — eager path wall-clock per forward.  \n"
        "* `t_graph` — CUDA Graph replay wall-clock per forward "
        "(kernel-launch API amortised).  \n"
        "* `launch_tax = t_plain - t_graph` — aggregate kernel-launch overhead "
        "per forward.  \n"
        "* `t_body` — GPU kernel accumulated time per forward from nsys "
        "sqlite; **gold-standard kernel-body time**.  \n"
        "* `body/plain` — how much of wall-clock is real kernel work vs "
        "launch/gap."
    )
    lines.append("")
    lines.append(
        "| tag | T | plain (us) | graph (us) | launch_tax (us) | tax % of plain | t_body (us) | body/plain |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in records:
        lines.append(
            "| {tag} | {T} | {p} | {g} | {tax} | {pct} | {body} | {ratio} |".format(
                tag=r["tag"],
                T=r["T"],
                p=r.get("t_plain_us", "n/a"),
                g=r.get("t_graph_us", "n/a"),
                tax=r.get("launch_tax_us", "n/a"),
                pct=f'{r["launch_tax_pct_of_plain"]}%'
                if "launch_tax_pct_of_plain" in r else "n/a",
                body=r.get("t_kernel_body_us", "n/a"),
                ratio=r.get("body_vs_plain_ratio", "n/a"),
            )
        )
    lines.append("")
    # Add graph-capture error section if any shape failed to capture.
    failed = [r for r in records if "graph_capture_error" in r]
    if failed:
        lines.append("## CUDA Graph capture failures")
        for r in failed:
            lines.append(f"- {r['tag']}: {r['graph_capture_error']}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="directory containing per-shape nsys reports and where to "
             "write launch_tax.{json,md}",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="only measure these shape tags",
    )
    parser.add_argument(
        "--inner-calls", type=int, default=10,
        help="must match --calls used when collecting the Phase 1 timeline "
             "(needed to average kernel body time correctly)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    shapes = PHASE1_SHAPES
    if args.only:
        wanted = set(args.only)
        shapes = tuple(s for s in PHASE1_SHAPES if s.tag in wanted)

    print(f"Measuring launch tax for {len(shapes)} shapes...")
    records: List[Dict[str, Any]] = []
    t0 = time.time()
    for s in shapes:
        print(f"  * {s.tag}", flush=True)
        rec = _one_shape(s, args.out_dir, args.inner_calls)
        print(
            f"    plain={rec.get('t_plain_us')}us  "
            f"graph={rec.get('t_graph_us', 'n/a')}us  "
            f"tax={rec.get('launch_tax_us', 'n/a')}us  "
            f"body={rec.get('t_kernel_body_us', 'n/a')}us"
        )
        records.append(rec)

    out_json = args.out_dir / "launch_tax.json"
    out_json.write_text(
        json.dumps(
            {
                "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
                "shapes": records,
                "wall_seconds": round(time.time() - t0, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _render_md(records, args.out_dir / "launch_tax.md")
    print(f"Written: {out_json}")
    print(f"Written: {args.out_dir / 'launch_tax.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
