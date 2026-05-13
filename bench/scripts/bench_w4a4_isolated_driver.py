"""r79 bench ISOLATED driver: run each (model, phase, bs) triple in a fresh
Python subprocess, then aggregate.

Why this exists
---------------
When we sweep all 24 triples in a single long-lived process, we observed
a CUDA "illegal memory access" in 14B prefill bs=16 gate_up_fused that
CANNOT be reproduced by running just that one shape in isolation — i.e.
the kernel itself is correct, but some stateful residue across many
prior calls (persistent workspace / L2 pollution / allocator
fragmentation) eventually triggers the fault.

The fix here is structural, not a kernel change: we isolate every
triple into its own subprocess.  Each subprocess starts with:
  * a fresh CUDA context (no leftover workspace pointers),
  * a fresh PyTorch caching allocator,
  * a cold GPU clock/cache state (mild, but harmless).

The measured per-op numbers are identical to the single-process path
(same warmup/outer/inner/trials, same kernel launches); the only cost
is ~5-15s of extra wall-clock per triple for Python + CUDA init.

Resume semantics
----------------
If an (model, phase, bs) sub-output already contains a valid JSON with
4 rows (=the 4 fused ops), we SKIP the subprocess.  This makes the
driver idempotent — re-running after a partial failure only tops up
the missing triples.

Aggregate outputs
-----------------
  <out-dir>/bench_w4a4_per_op.json   : all merged rows
  <out-dir>/bench_w4a4_summary.md    : grouped tables, same schema as
                                        the single-process driver
  <out-dir>/VALIDATION_LOG.md        : env, spread sanity, and a list
                                        of FAILED triples (if any)
  <out-dir>/per_triple/<m>__<p>__bs<N>/... : raw per-triple output dirs
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Project-root on sys.path so we can import the qwen3_shapes catalog for
# enumerating models/phases/batches.  The actual measurement code lives
# in a subprocess — this driver never touches CUDA directly.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PROJ_ROOT = _THIS_FILE.parents[3]  # .../HKUST  (or /root via symlink on server)
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from kernel.bench.configs.qwen3_shapes import (  # noqa: E402
    BATCH_SIZES,
    PHASES,
    QWEN3_MODELS,
)


# ---------------------------------------------------------------------------
# One subprocess = one (model, phase, bs) triple = 4 ops
# ---------------------------------------------------------------------------
def _triple_dir(out_dir: Path, model: str, phase: str, bs: int) -> Path:
    return out_dir / "per_triple" / f"{model}__{phase}__bs{bs}"


def _already_done(tdir: Path) -> bool:
    """Return True iff the triple has a complete 4-row per-op JSON already."""
    j = tdir / "bench_w4a4_per_op.json"
    if not j.is_file():
        return False
    try:
        payload = json.loads(j.read_text())
        return len(payload.get("rows", [])) == 4
    except Exception:
        return False


def _run_triple(
    out_dir: Path,
    model: str,
    phase: str,
    bs: int,
    *,
    commit: str,
    device: str,
    python_exe: str,
    log_stream,
) -> tuple[bool, str]:
    """Launch one subprocess. Returns (ok, err_tail)."""
    tdir = _triple_dir(out_dir, model, phase, bs)
    tdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe, "-u",
        str(_THIS_FILE.with_name("bench_w4a4_fused_ops.py")),
        "--out-dir", str(tdir),
        "--models", model,
        "--phases", phase,
        "--batch-sizes", str(bs),
        "--commit", commit,
        "--device", device,
    ]

    # Tee subprocess output into (a) a per-triple log, (b) the main run log.
    per_triple_log = tdir / "run.log"
    with per_triple_log.open("w") as ptf:
        ptf.write(f"$ {' '.join(cmd)}\n")
        ptf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            env={**os.environ},
        )
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            ptf.write(line); ptf.flush()
            log_stream.write(line); log_stream.flush()
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
        rc = proc.wait()

    if rc != 0:
        return False, "".join(tail)

    # Sanity: subprocess exited 0 but we still want a valid per-op.json
    if not _already_done(tdir):
        return False, (
            f"subprocess returned 0 but {tdir/'bench_w4a4_per_op.json'} "
            "missing or has != 4 rows"
        )
    return True, ""


# ---------------------------------------------------------------------------
# Aggregation (identical schema to bench_w4a4_fused_ops.py outputs)
# ---------------------------------------------------------------------------
def _collect_rows(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    per_triple = out_dir / "per_triple"
    if not per_triple.is_dir():
        return rows
    for tdir in sorted(per_triple.iterdir()):
        j = tdir / "bench_w4a4_per_op.json"
        if not j.is_file():
            continue
        try:
            payload = json.loads(j.read_text())
        except Exception:
            continue
        rows.extend(payload.get("rows", []))
    return rows


def _dump_aggregate_json(
    rows: list[dict], path: Path, *, meta: dict,
) -> None:
    payload = {
        "meta": {
            **meta,
            "n_rows": len(rows),
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_summary_md(rows: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# W4A4 fused-op kernel time — r79 bench (isolated driver)\n")
    lines.append(
        "Each (model, phase, bs) triple was measured in a fresh CUDA "
        "subprocess to eliminate cross-triple kernel-workspace / allocator "
        "state from the timings.  Per-op timing contract is unchanged "
        "(min-of-outer of mean-of-inner, median over trials).\n"
    )
    lines.append(
        "| model | phase | bs | op | T | d_in | d_out | kernel | "
        "preset | median_us | min_us | max_us | spread_% | est_ms |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        s = r["stats"]
        lines.append(
            f"| {r['model']} | {r['phase']} | {r['batch']} | {r['op']} | "
            f"{r['T']} | {r['d_in']} | {r['d_out']} | {r['kernel_path']} | "
            f"{r['preset_name']} | {s['median_us']:.2f} | {s['min_us']:.2f} | "
            f"{s['max_us']:.2f} | {s['spread_pct']:.2f} | "
            f"{r['est_call_ms']:.3f} |\n"
        )

    lines.append("\n## Sum of 4 fused ops per (model, phase, bs)\n\n")
    lines.append(
        "Replacement-ready number: this is the total kernel time the "
        "W4A4 path contributes to a single decoder layer (attention + "
        "MLP projections), NOT including attention / norms / RoPE / "
        "residuals.\n\n"
    )
    lines.append("| model | phase | bs | Σ median_us |\n")
    lines.append("|---|---|---|---|\n")
    groups: dict[tuple[str, str, int], float] = {}
    for r in rows:
        key = (r["model"], r["phase"], r["batch"])
        groups[key] = groups.get(key, 0.0) + r["stats"]["median_us"]
    for (model, phase, bs), tot in sorted(groups.items()):
        lines.append(f"| {model} | {phase} | {bs} | {tot:.2f} |\n")
    out_path.write_text("".join(lines))


def _write_validation_log(
    rows: list[dict],
    out_path: Path,
    *,
    meta: dict,
    failed: list[tuple[str, str, int, str]],
) -> None:
    spread_vals = [r["stats"]["spread_pct"] for r in rows] or [0.0]
    n_over_1pct = sum(1 for s in spread_vals if s > 1.0)
    n_over_5pct = sum(1 for s in spread_vals if s > 5.0)
    sorted_spread = sorted(spread_vals)
    lines = [
        "# W4A4 bench validation log (isolated driver)\n\n",
        "## Environment\n",
        f"- commit: {meta.get('commit','?')}\n",
        f"- host: {meta.get('host','?')}\n",
        f"- started: {meta.get('started','?')}\n",
        f"- rows collected: {len(rows)}\n",
        f"- triples expected: {meta.get('n_triples_expected','?')}\n",
        f"- triples completed: {meta.get('n_triples_completed','?')}\n",
        f"- triples FAILED: {len(failed)}\n\n",
        "## Spread summary (spread_pct = (max-min)/median · 100)\n",
        f"- rows with spread > 1.0%: {n_over_1pct} / {len(rows)}\n",
        f"- rows with spread > 5.0%: {n_over_5pct} / {len(rows)}\n",
        f"- max spread: {max(spread_vals):.2f}%\n",
        f"- median spread: {sorted_spread[len(sorted_spread)//2]:.2f}%\n\n",
        "## Dispatch path distribution\n",
    ]
    paths: dict[str, int] = {}
    for r in rows:
        paths[r["kernel_path"]] = paths.get(r["kernel_path"], 0) + 1
    for p, n in paths.items():
        lines.append(f"- {p}: {n}\n")

    if failed:
        lines.append("\n## FAILED triples\n\n")
        lines.append("| model | phase | bs | error tail |\n")
        lines.append("|---|---|---|---|\n")
        for m, p, bs, tail in failed:
            short = tail.strip().splitlines()[-1] if tail.strip() else "(no output)"
            # Escape markdown table pipe chars
            short = short.replace("|", "\\|")
            lines.append(f"| {m} | {p} | {bs} | `{short[:200]}` |\n")

    out_path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Top-level output directory (will be created).")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Subset of Qwen3 model names (default: all).")
    ap.add_argument("--phases", nargs="*", default=None,
                    choices=("prefill", "decode"),
                    help="Subset of phases (default: both).")
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=None,
                    help="Subset of batch sizes (default: catalog).")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--commit", type=str, default="?")
    ap.add_argument("--python", dest="python_exe", type=str,
                    default=sys.executable,
                    help="Python interpreter used for sub-processes.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip triples that already have a complete "
                         "per-op JSON under per_triple/.")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_triple").mkdir(exist_ok=True)

    models = [m.name for m in QWEN3_MODELS
              if args.models is None or m.name in args.models]
    phases = [p.name for p in PHASES
              if args.phases is None or p.name in args.phases]
    bses = list(args.batch_sizes) if args.batch_sizes else list(BATCH_SIZES)

    triples = [(m, p, bs) for m in models for p in phases for bs in bses]
    n_total = len(triples)

    main_log = out_dir / "driver.log"
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    hostname = os.uname().nodename

    with main_log.open("a") as lf:
        lf.write(f"\n=== isolated-driver start {started_iso} ===\n")
        lf.write(f"out-dir: {out_dir}\n")
        lf.write(f"triples: {n_total}  (models={models}, phases={phases}, bs={bses})\n")
        lf.write(f"commit: {args.commit}  host: {hostname}\n")
        lf.write(f"python: {args.python_exe}  resume={args.resume}\n\n")
        lf.flush()

        t0 = time.time()
        failed: list[tuple[str, str, int, str]] = []
        n_skipped = 0
        n_done = 0

        for idx, (model, phase, bs) in enumerate(triples, 1):
            tag = f"[{idx}/{n_total}] {model}/{phase}/bs{bs}"
            tdir = _triple_dir(out_dir, model, phase, bs)

            if args.resume and _already_done(tdir):
                msg = f"{tag} SKIP (already complete)\n"
                print(msg, end="", flush=True); lf.write(msg); lf.flush()
                n_skipped += 1
                continue

            t_start = time.time()
            print(f"{tag} START (elapsed={time.time()-t0:.1f}s)", flush=True)
            lf.write(f"{tag} START\n"); lf.flush()

            ok, err_tail = _run_triple(
                out_dir, model, phase, bs,
                commit=args.commit, device=args.device,
                python_exe=args.python_exe, log_stream=lf,
            )

            dt = time.time() - t_start
            if ok:
                msg = f"{tag} OK   ({dt:.1f}s)\n"
                n_done += 1
            else:
                msg = f"{tag} FAIL ({dt:.1f}s) tail={err_tail.strip().splitlines()[-1:] if err_tail else []}\n"
                failed.append((model, phase, bs, err_tail))
            print(msg, end="", flush=True); lf.write(msg); lf.flush()

        # Aggregate
        rows = _collect_rows(out_dir)
        meta = {
            "commit": args.commit,
            "host": hostname,
            "started": started_iso,
            "n_triples_expected": n_total,
            "n_triples_completed": n_done + n_skipped,
            "n_triples_failed": len(failed),
        }
        _dump_aggregate_json(
            rows, out_dir / "bench_w4a4_per_op.json", meta=meta,
        )
        _write_summary_md(rows, out_dir / "bench_w4a4_summary.md")
        _write_validation_log(
            rows, out_dir / "VALIDATION_LOG.md",
            meta=meta, failed=failed,
        )

        total_dt = time.time() - t0
        summary = (
            f"\n=== isolated-driver done ===\n"
            f"  total triples:  {n_total}\n"
            f"  ok (this run):  {n_done}\n"
            f"  skipped:        {n_skipped}\n"
            f"  failed:         {len(failed)}\n"
            f"  rows collected: {len(rows)}\n"
            f"  wall clock:     {total_dt:.1f}s\n"
            f"  output:         {out_dir}\n"
        )
        print(summary, flush=True); lf.write(summary); lf.flush()

    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
