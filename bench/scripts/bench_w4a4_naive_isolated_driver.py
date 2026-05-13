"""r80 NAIVE-backend ISOLATED driver: subprocess-per-triple for cross-shape
state isolation.

Sibling of ``bench_w4a4_isolated_driver.py`` — same wall-clock isolation
rationale (fresh CUDA context + allocator per triple) — but drives the
naive 4-kernel pipeline (bench_w4a4_naive_fused_ops.py) instead of the
optimised fused path.

Why this exists
---------------
The same cross-triple CUDA-workspace / allocator residue that bit the
optimised bench (14B prefill bs=16 gate_up_fused illegal memory access
after N warm triples) can in principle hit the naive bench too — naive
kernels use larger shmem tiles and can leave PyTorch allocator
fragmentation in a worse state than the compact MMA path.  Running each
(model, phase, bs) triple in its own Python subprocess is the same
structural fix.

Resume semantics
----------------
A triple is "done" iff its per-triple JSON contains 16 rows
(= 4 ops × 1 backend).  Re-running with --resume only tops up missing
triples.

Aggregate outputs
-----------------
  <out-dir>/bench_w4a4_naive_per_op.json
  <out-dir>/bench_w4a4_naive_summary.md
  <out-dir>/VALIDATION_LOG.md
  <out-dir>/per_triple/<model>__<phase>__bs<N>/...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJ_ROOT = _THIS_FILE.parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from kernel.bench.configs.qwen3_shapes import (  # noqa: E402
    BATCH_SIZES, PHASES, QWEN3_MODELS,
)

_PER_OP_JSON = "bench_w4a4_naive_per_op.json"
_SUMMARY_MD  = "bench_w4a4_naive_summary.md"


def _triple_dir(out_dir: Path, model: str, phase: str, bs: int) -> Path:
    return out_dir / "per_triple" / f"{model}__{phase}__bs{bs}"


def _already_done(tdir: Path) -> bool:
    j = tdir / _PER_OP_JSON
    if not j.is_file():
        return False
    try:
        payload = json.loads(j.read_text())
        return len(payload.get("rows", [])) == 4
    except Exception:
        return False


def _run_triple(
    out_dir: Path, model: str, phase: str, bs: int,
    *, commit: str, device: str, python_exe: str, log_stream,
) -> tuple[bool, str]:
    tdir = _triple_dir(out_dir, model, phase, bs)
    tdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe, "-u",
        str(_THIS_FILE.with_name("bench_w4a4_naive_fused_ops.py")),
        "--out-dir", str(tdir),
        "--models", model,
        "--phases", phase,
        "--batch-sizes", str(bs),
        "--commit", commit,
        "--device", device,
    ]
    per_triple_log = tdir / "run.log"
    with per_triple_log.open("w") as ptf:
        ptf.write(f"$ {' '.join(cmd)}\n"); ptf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, env={**os.environ},
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
    if not _already_done(tdir):
        return False, (
            f"subprocess returned 0 but {tdir/_PER_OP_JSON} missing "
            "or has != 4 rows"
        )
    return True, ""


def _collect_rows(out_dir: Path) -> list[dict]:
    rows: list[dict] = []
    per_triple = out_dir / "per_triple"
    if not per_triple.is_dir():
        return rows
    for tdir in sorted(per_triple.iterdir()):
        j = tdir / _PER_OP_JSON
        if not j.is_file():
            continue
        try:
            payload = json.loads(j.read_text())
        except Exception:
            continue
        rows.extend(payload.get("rows", []))
    return rows


def _dump_aggregate_json(rows: list[dict], path: Path, *, meta: dict) -> None:
    payload = {"meta": {**meta, "n_rows": len(rows)}, "rows": rows}
    path.write_text(json.dumps(payload, indent=2))


def _write_summary_md(rows: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# W4A4 NAIVE-backend kernel time — isolated driver\n")
    lines.append(
        "Every (model, phase, bs) triple was measured in a fresh CUDA "
        "subprocess.  Per-op number = wall time of the naive 4-kernel "
        "pipeline (quant → dense → sparse → reduce_sum) for one "
        "projection call.\n"
    )
    lines.append(
        "| model | phase | bs | op | T | d_in | d_out | preset | "
        "median_us | min_us | max_us | spread_% | est_ms |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        s = r["stats"]
        lines.append(
            f"| {r['model']} | {r['phase']} | {r['batch']} | {r['op']} | "
            f"{r['T']} | {r['d_in']} | {r['d_out']} | "
            f"{r['preset_name']} | {s['median_us']:.2f} | "
            f"{s['min_us']:.2f} | {s['max_us']:.2f} | "
            f"{s['spread_pct']:.2f} | {r['est_call_ms']:.3f} |\n"
        )

    lines.append("\n## Sum of 4 fused ops per (model, phase, bs)\n\n")
    lines.append("| model | phase | bs | Σ median_us |\n|---|---|---|---|\n")
    groups: dict[tuple[str, str, int], float] = {}
    for r in rows:
        key = (r["model"], r["phase"], r["batch"])
        groups[key] = groups.get(key, 0.0) + r["stats"]["median_us"]
    for (model, phase, bs), tot in sorted(groups.items()):
        lines.append(f"| {model} | {phase} | {bs} | {tot:.2f} |\n")
    out_path.write_text("".join(lines))


def _write_validation_log(
    rows: list[dict], out_path: Path, *,
    meta: dict, failed: list[tuple[str, str, int, str]],
) -> None:
    spread_vals = [r["stats"]["spread_pct"] for r in rows] or [0.0]
    n_over_1 = sum(1 for s in spread_vals if s > 1.0)
    n_over_5 = sum(1 for s in spread_vals if s > 5.0)
    sorted_s = sorted(spread_vals)
    lines = [
        "# W4A4 NAIVE bench validation log (isolated driver)\n\n",
        "## Environment\n",
        f"- commit: {meta.get('commit','?')}\n",
        f"- host: {meta.get('host','?')}\n",
        f"- started: {meta.get('started','?')}\n",
        f"- backend: naive (csrc_naive, no MMA, no cp.async)\n",
        f"- sparsity: 5.0%\n",
        f"- rows collected: {len(rows)}\n",
        f"- triples expected: {meta.get('n_triples_expected','?')}\n",
        f"- triples completed: {meta.get('n_triples_completed','?')}\n",
        f"- triples FAILED: {len(failed)}\n\n",
        "## Spread summary\n",
        f"- rows with spread > 1.0%: {n_over_1} / {len(rows)}\n",
        f"- rows with spread > 5.0%: {n_over_5} / {len(rows)}\n",
        f"- max spread: {max(spread_vals):.2f}%\n",
        f"- median spread: {sorted_s[len(sorted_s)//2]:.2f}%\n",
    ]
    if failed:
        lines.append("\n## FAILED triples\n\n")
        lines.append("| model | phase | bs | error tail |\n|---|---|---|---|\n")
        for m, p, bs, tail in failed:
            short = tail.strip().splitlines()[-1] if tail.strip() else "(no output)"
            short = short.replace("|", "\\|")
            lines.append(f"| {m} | {p} | {bs} | `{short[:200]}` |\n")
    out_path.write_text("".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--phases", nargs="*", default=None,
                    choices=("prefill", "decode"))
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--commit", type=str, default="?")
    ap.add_argument("--python", dest="python_exe", type=str,
                    default=sys.executable)
    ap.add_argument("--resume", action="store_true")
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
        lf.write(f"\n=== naive isolated-driver start {started_iso} ===\n")
        lf.write(f"out-dir: {out_dir}\n")
        lf.write(f"triples: {n_total} (models={models}, phases={phases}, "
                 f"bs={bses})\n")
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
                msg = (f"{tag} FAIL ({dt:.1f}s) "
                       f"tail={err_tail.strip().splitlines()[-1:] if err_tail else []}\n")
                failed.append((model, phase, bs, err_tail))
            print(msg, end="", flush=True); lf.write(msg); lf.flush()

        rows = _collect_rows(out_dir)
        meta = {
            "commit": args.commit, "host": hostname,
            "started": started_iso, "backend": "naive",
            "sparsity_pct": 5.0,
            "n_triples_expected": n_total,
            "n_triples_completed": n_done + n_skipped,
            "n_triples_failed": len(failed),
        }
        _dump_aggregate_json(rows, out_dir / _PER_OP_JSON, meta=meta)
        _write_summary_md(rows, out_dir / _SUMMARY_MD)
        _write_validation_log(
            rows, out_dir / "VALIDATION_LOG.md", meta=meta, failed=failed,
        )

        total_dt = time.time() - t0
        summary = (
            f"\n=== naive isolated-driver done ===\n"
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
