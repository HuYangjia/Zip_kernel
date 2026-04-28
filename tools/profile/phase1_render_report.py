"""Phase 1.6: render the timeline attribution report from nsys sqlite data.

For each representative shape that has a matching ``report.sqlite``,
this script computes the per-iteration time spent inside each of the
four NVTX ranges emitted by ``nvtx_shim``::

    ops.linear_forward        (whole forward)
    dispatcher.select_impl    (policy + backend resolution)
    cuda.activation_quant     (activation SINT4 quant kernel)
    cuda.fused_dense_sparse   (fused dense+sparse MMA/GEMV kernel)

The outer ``ops.linear_forward`` range is the authoritative total; the
other three are inner sub-ranges.  Their sum is compared against the
outer total and any leftover is attributed to ``dispatcher_python``
(host-side cost that is not covered by any named inner range).

Additionally we use ``CUPTI_ACTIVITY_KIND_KERNEL`` to compute the
per-iteration on-GPU kernel time, and cross-check against the
``launch_tax.json`` produced by the CUDA Graph driver.

Output
------
``<out_dir>/phase1_timeline_report.md`` — human-readable report per
requirements.md §2.5.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
KERNEL_ROOT = _THIS.parents[2]
IMPORT_ROOT = KERNEL_ROOT.parent
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

from kernel.tools.profile._phase1_shapes import (  # noqa: E402
    PHASE1_SHAPES,
    PhaseShape,
)

DEFAULT_DIR = KERNEL_ROOT / "cuda_kernel" / "logs" / "phase1_timeline"

# Iteration-bracket NVTX ranges emitted by phase1_inner_driver (each
# profiled forward sits inside exactly one ``phase1.iter_<i>``).
ITER_RANGE_PREFIX = "phase1.iter_"

# Inner named ranges we care about.
NAMED_RANGES = (
    "ops.linear_forward",
    "dispatcher.select_impl",
    "cuda.activation_quant",
    "cuda.fused_dense_sparse",
)


# ---------------------------------------------------------------------------
# SQLite accessors
# ---------------------------------------------------------------------------

def _connect(sqlite_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    cur = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def _string_id(con: sqlite3.Connection, value: str) -> Optional[int]:
    """Return the StringIds.id for a given NVTX range text."""
    if not _table_exists(con, "StringIds"):
        return None
    cur = con.execute("SELECT id FROM StringIds WHERE value=? LIMIT 1", (value,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def _iter_windows(con: sqlite3.Connection) -> List[Tuple[int, int, int]]:
    """Return [(iter_index, start_ns, end_ns)] for phase1.iter_* ranges."""
    if not _table_exists(con, "NVTX_EVENTS"):
        return []
    cur = con.execute("SELECT id, value FROM StringIds WHERE value LIKE ?",
                      (f"{ITER_RANGE_PREFIX}%",))
    id_to_name: Dict[int, str] = {int(i): s for (i, s) in cur.fetchall()}
    if not id_to_name:
        return []
    placeholders = ",".join("?" * len(id_to_name))
    cur = con.execute(
        f"SELECT start, end, textId FROM NVTX_EVENTS "
        f"WHERE textId IN ({placeholders}) AND end IS NOT NULL",
        tuple(id_to_name.keys()),
    )
    rows = cur.fetchall()
    out: List[Tuple[int, int, int]] = []
    for start, end, tid in rows:
        name = id_to_name[int(tid)]
        try:
            idx = int(name.split("_")[-1])
        except ValueError:
            continue
        out.append((idx, int(start), int(end)))
    out.sort()
    return out


def _range_durations(
    con: sqlite3.Connection, range_name: str, windows: List[Tuple[int, int, int]]
) -> List[int]:
    """Return per-iter total duration (ns) of ``range_name`` within each window.

    We join NVTX_EVENTS against the iter windows by start-time containment:
    any event whose ``start`` is inside ``[iter_start, iter_end]`` is
    attributed to that iter.
    """
    sid = _string_id(con, range_name)
    if sid is None:
        return [0] * len(windows)
    cur = con.execute(
        "SELECT start, end FROM NVTX_EVENTS WHERE textId=? AND end IS NOT NULL",
        (sid,),
    )
    events = cur.fetchall()
    buckets = [0] * len(windows)
    # windows is already sorted by iter index, but we need to search by
    # time; build a time-sorted view.
    time_sorted = sorted(range(len(windows)), key=lambda i: windows[i][1])
    for s, e in events:
        # Linear scan is fine — windows <= 20 in all our experiments.
        for i in time_sorted:
            _, ws, we = windows[i]
            if ws <= s <= we:
                buckets[i] += int(e) - int(s)
                break
    return buckets


def _kernel_durations_per_iter(
    con: sqlite3.Connection, windows: List[Tuple[int, int, int]]
) -> List[int]:
    """Total GPU kernel runtime (ns) per iter from CUPTI kernel events.

    Attribution is by event start containment in the iter window (same
    as NVTX attribution).  Kernel events launched before the window
    (or completed after) are excluded, giving us a clean per-iter body
    count unaffected by warmup kernels.
    """
    if not _table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return [0] * len(windows)
    cur = con.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE end IS NOT NULL"
    )
    events = cur.fetchall()
    buckets = [0] * len(windows)
    time_sorted = sorted(range(len(windows)), key=lambda i: windows[i][1])
    for s, e in events:
        for i in time_sorted:
            _, ws, we = windows[i]
            if ws <= s <= we:
                buckets[i] += int(e) - int(s)
                break
    return buckets


# ---------------------------------------------------------------------------
# Per-shape analysis
# ---------------------------------------------------------------------------

def _analyse_shape(shape: PhaseShape, out_dir: Path) -> Optional[Dict[str, Any]]:
    sqlite_path = out_dir / shape.tag / "report.sqlite"
    if not sqlite_path.exists():
        return {"tag": shape.tag, "error": "sqlite missing"}
    con = _connect(sqlite_path)
    try:
        windows = _iter_windows(con)
        if not windows:
            return {"tag": shape.tag, "error": "no phase1.iter_* windows found"}

        ranges_ns: Dict[str, List[int]] = {
            name: _range_durations(con, name, windows) for name in NAMED_RANGES
        }
        kernel_ns = _kernel_durations_per_iter(con, windows)
    finally:
        con.close()

    n = len(windows)
    iter_total_ns = [e - s for (_, s, e) in windows]

    def _mean_us(vals: List[int]) -> float:
        return sum(vals) / n / 1000.0 if n else 0.0

    total_us = _mean_us(iter_total_ns)
    forward_us = _mean_us(ranges_ns["ops.linear_forward"])
    dispatcher_us = _mean_us(ranges_ns["dispatcher.select_impl"])
    quant_us = _mean_us(ranges_ns["cuda.activation_quant"])
    fused_us = _mean_us(ranges_ns["cuda.fused_dense_sparse"])
    kernel_body_us = _mean_us(kernel_ns)

    # Outside-of-forward (the iter window includes 1 torch.cuda.synchronize()
    # after the forward and the NVTX push/pop themselves).
    outside_forward_us = max(0.0, total_us - forward_us)

    # Inside forward: forward = named_inner_sum + (host_python_glue + gaps).
    inner_named_us = quant_us + fused_us + dispatcher_us
    python_glue_us = max(0.0, forward_us - inner_named_us)
    # Inter-kernel gap on the GPU timeline:
    inter_kernel_gap_us = max(0.0, forward_us - kernel_body_us - python_glue_us)

    entry = {
        "tag": shape.tag,
        "iters": n,
        "total_us": round(total_us, 3),
        "forward_us": round(forward_us, 3),
        "outside_forward_us": round(outside_forward_us, 3),
        "dispatcher_us": round(dispatcher_us, 3),
        "activation_quant_us": round(quant_us, 3),
        "fused_dense_sparse_us": round(fused_us, 3),
        "kernel_body_us": round(kernel_body_us, 3),
        "python_glue_us": round(python_glue_us, 3),
        "inter_kernel_gap_us": round(inter_kernel_gap_us, 3),
    }
    return entry


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _pct(part: float, whole: float) -> str:
    if whole <= 0:
        return "n/a"
    return f"{part / whole * 100:.1f}%"


def _render_md(
    records: List[Dict[str, Any]],
    launch_tax: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Phase 1 — Timeline Attribution Report")
    lines.append("")
    lines.append(f"_Generated {_dt.datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append(
        "Every representative shape was run under `nsys profile -t cuda,nvtx,osrt` "
        "with the inner driver emitting one `phase1.iter_<i>` NVTX range per "
        "profiled forward, plus four fine-grained sub-ranges instrumented via "
        "`HKUST_V9_PROFILE=1`:  \n"
        "* `ops.linear_forward` — whole forward, outermost range.  \n"
        "* `dispatcher.select_impl` — backend resolution (per-kernel).  \n"
        "* `cuda.activation_quant` — SINT4 activation quantisation body.  \n"
        "* `cuda.fused_dense_sparse` — fused dense+sparse MMA/GEMV body.  \n"
        "Each row below is the mean over all profiled iterations (n=10 unless "
        "otherwise noted)."
    )
    lines.append("")

    # Headline attribution table -------------------------------------------
    lines.append("## 1. Attribution Table (per forward, microseconds)")
    lines.append("")
    lines.append(
        "| shape | T | forward | disp | quant | fused | kernel_body | "
        "python_glue | inter_kernel_gap |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in records:
        if "error" in r:
            lines.append(f"| {r['tag']} | — | **{r['error']}** | | | | | | |")
            continue
        shape = next((s for s in PHASE1_SHAPES if s.tag == r["tag"]), None)
        T = shape.T if shape else "?"
        lines.append(
            "| {tag} | {T} | {fwd:.2f} | {d:.2f} | {q:.2f} | {f:.2f} | "
            "{k:.2f} | {glue:.2f} | {gap:.2f} |".format(
                tag=r["tag"], T=T, fwd=r["forward_us"],
                d=r["dispatcher_us"], q=r["activation_quant_us"],
                f=r["fused_dense_sparse_us"], k=r["kernel_body_us"],
                glue=r["python_glue_us"], gap=r["inter_kernel_gap_us"],
            )
        )
    lines.append("")

    # Percent-of-forward breakdown ----------------------------------------
    lines.append("## 2. Fraction of forward (percentages)")
    lines.append("")
    lines.append(
        "Each row sums to ≥100%: `quant + fused` is the on-device kernel work, "
        "`python_glue` is host-side time inside the forward, and the `forward - "
        "kernel_body` gap accounts for CUDA API + dispatcher overhead."
    )
    lines.append("")
    lines.append(
        "| shape | quant% | fused% | kernel_body% | python_glue% | "
        "inter_kernel_gap% |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in records:
        if "error" in r:
            continue
        fwd = r["forward_us"]
        lines.append(
            "| {tag} | {q} | {f} | {k} | {g} | {gap} |".format(
                tag=r["tag"],
                q=_pct(r["activation_quant_us"], fwd),
                f=_pct(r["fused_dense_sparse_us"], fwd),
                k=_pct(r["kernel_body_us"], fwd),
                g=_pct(r["python_glue_us"], fwd),
                gap=_pct(r["inter_kernel_gap_us"], fwd),
            )
        )
    lines.append("")

    # Cross-check against the Graph-replay launch tax ----------------------
    lines.append("## 3. Cross-check against CUDA-Graph launch-tax measurement")
    lines.append("")
    lines.append(
        "The Graph-replay driver amortises all kernel launches into a single "
        "`cudaGraphLaunch`.  Difference = aggregate kernel-launch API overhead."
    )
    lines.append("")
    lines.append(
        "| shape | plain_us | graph_us | launch_tax_us | tax %% | "
        "nvtx_kernel_body_us |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in records:
        if "error" in r:
            continue
        lt = launch_tax.get(r["tag"], {})
        lines.append(
            "| {tag} | {p} | {g} | {tax} | {pct} | {body} |".format(
                tag=r["tag"],
                p=lt.get("t_plain_us", "n/a"),
                g=lt.get("t_graph_us", "n/a"),
                tax=lt.get("launch_tax_us", "n/a"),
                pct=(f"{lt['launch_tax_pct_of_plain']}%"
                     if "launch_tax_pct_of_plain" in lt else "n/a"),
                body=r["kernel_body_us"],
            )
        )
    lines.append("")

    # Verdict --------------------------------------------------------------
    lines.append("## 4. Verdict — bottleneck class per shape")
    lines.append("")
    lines.append(
        "Threshold per requirements.md §2.6: if `launch_tax / plain > 30%` the "
        "shape is marked `launch_bound` and CUDA Graph is the recommended first "
        "optimisation.  Otherwise the shape is `kernel_bound` and the Phase 2 "
        "microscope should drive the next step."
    )
    lines.append("")
    lines.append("| shape | launch_tax % | class | recommended next step |")
    lines.append("|---|---:|---|---|")
    for r in records:
        if "error" in r:
            continue
        lt = launch_tax.get(r["tag"], {})
        pct = lt.get("launch_tax_pct_of_plain")
        if pct is None:
            verdict = "unknown"
            rec = "re-run launch tax"
        elif pct > 30.0:
            verdict = "**launch_bound**"
            rec = "CUDA Graph capture (Phase 3 cluster A)"
        else:
            verdict = "kernel_bound"
            rec = "Phase 2 microscope (SASS + microbench bisection)"
        lines.append(
            f"| {r['tag']} | {pct if pct is not None else 'n/a'}% | "
            f"{verdict} | {rec} |"
        )
    lines.append("")

    lines.append("## 5. Known measurement caveats")
    lines.append("")
    lines.append(
        "* `inter_kernel_gap_us` can be slightly negative in theory because of "
        "NVTX push/pop timestamps not being perfectly aligned with CUDA stream "
        "timestamps; we clamp at 0.  \n"
        "* `nsys` has ~1% CUPTI sampling overhead; raw `t_plain_us` from the "
        "launch-tax driver is the authoritative wall-clock.  \n"
        "* Kernel body time in this table comes from CUPTI kernel events "
        "*inside* each `phase1.iter_<i>` window — warmup kernels are excluded "
        "automatically by the attribution logic, so this number is "
        "cleaner than what `launch_tax.md`'s `t_body` column reports.  \n"
        "* GPU-metric sampling was **not** enabled (container PMU blocked, "
        "ERR_NVGPUCTRPERM).  Phase 2 will substitute with microbench "
        "bisection + SASS static analysis."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="only analyse these shape tags",
    )
    args = parser.parse_args()

    if not args.dir.exists():
        sys.stderr.write(f"directory not found: {args.dir}\n")
        return 2

    shapes = PHASE1_SHAPES
    if args.only:
        wanted = set(args.only)
        shapes = tuple(s for s in PHASE1_SHAPES if s.tag in wanted)

    records: List[Dict[str, Any]] = []
    for s in shapes:
        rec = _analyse_shape(s, args.dir)
        if rec is None:
            records.append({"tag": s.tag, "error": "no data"})
        else:
            records.append(rec)
        if "error" in rec:
            print(f"  ! {s.tag}: {rec['error']}")
        else:
            print(
                f"  * {s.tag}: fwd={rec['forward_us']}us "
                f"quant={rec['activation_quant_us']}us "
                f"fused={rec['fused_dense_sparse_us']}us "
                f"body={rec['kernel_body_us']}us"
            )

    launch_tax_path = args.dir / "launch_tax.json"
    launch_tax: Dict[str, Dict[str, Any]] = {}
    if launch_tax_path.exists():
        data = json.loads(launch_tax_path.read_text())
        launch_tax = {e["tag"]: e for e in data.get("shapes", [])}

    out_path = args.dir / "phase1_timeline_report.md"
    _render_md(records, launch_tax, out_path)

    # Also emit a machine-readable JSON for downstream Phase 3 clustering.
    json_path = args.dir / "phase1_attribution.json"
    json_path.write_text(
        json.dumps(
            {
                "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
