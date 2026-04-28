"""Phase 1 orchestrator: run ``nsys profile`` for each representative shape.

For every shape in ``PHASE1_SHAPES`` this script invokes one subprocess::

    nsys profile -t cuda,nvtx,osrt -f true -o <out>/report \\
        python phase1_inner_driver.py <shape_tag>

The inner driver runs 50 warmups + 10 profiled iterations, each wrapped
in an NVTX range so the later sqlite post-processor can bucketise time
by ``ops.linear_forward``, ``cuda.activation_quant``,
``cuda.fused_dense_sparse``, ``dispatcher.select_impl``.

Rationale for this separation
-----------------------------
* The **outer** script (this one) is responsible for orchestration,
  cwd, env vars, and artefact layout.
* The **inner** script (``phase1_inner_driver.py``) is the one actually
  measured by nsys; it only emits NVTX ranges and keeps allocations
  minimal so the timeline stays readable.

Artefacts
---------
Per shape::

    <out_dir>/<tag>/
        report.nsys-rep    # full timeline (binary; gitignored)
        report.sqlite      # exported SQL (gitignored)
        run_meta.json      # git sha, nsys version, timings, nvidia-smi

A top-level ``run_index.json`` summarises all shapes that were covered.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# --- path anchoring: <kernel>/tools/profile/phase1_collect_timeline.py
_THIS = Path(__file__).resolve()
KERNEL_ROOT = _THIS.parents[2]      # <kernel>
IMPORT_ROOT = KERNEL_ROOT.parent    # /root on autodl, <repo> locally
if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))

from kernel.tools.profile._phase1_shapes import (  # noqa: E402
    PHASE1_SHAPES,
    PhaseShape,
)


DEFAULT_OUT_DIR = KERNEL_ROOT / "cuda_kernel" / "logs" / "phase1_timeline"


def _run(cmd: List[str], **kw: Any) -> subprocess.CompletedProcess:
    """subprocess.run wrapper that never raises and captures everything."""
    return subprocess.run(
        cmd, capture_output=True, text=True, check=False, **kw
    )


def _get_git_sha() -> str:
    res = _run(["git", "rev-parse", "HEAD"], cwd=str(KERNEL_ROOT))
    return res.stdout.strip()[:12] if res.returncode == 0 else "n/a"


def _nsys_version() -> str:
    res = _run(["nsys", "--version"])
    first = (res.stdout or res.stderr).splitlines()
    return first[0] if first else "unknown"


def _nvidia_smi_head() -> str:
    res = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ]
    )
    return res.stdout.strip() if res.returncode == 0 else "n/a"


def _collect_one(
    shape: PhaseShape,
    out_dir: Path,
    calls: int,
    force: bool,
) -> Dict[str, Any]:
    """Invoke nsys profile for a single shape.  Returns a summary dict."""
    shape_dir = out_dir / shape.tag
    rep_file = shape_dir / "report.nsys-rep"
    sqlite_file = shape_dir / "report.sqlite"

    if shape_dir.exists() and not force:
        if rep_file.exists():
            return {
                "tag": shape.tag,
                "skipped": True,
                "reason": "report exists (use --force to overwrite)",
                "report": str(rep_file),
            }
    shape_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HKUST_V9_PROFILE"] = "1"
    # Always put /usr/local/cuda/bin on PATH so nsys can find cuobjdump etc.
    env["PATH"] = "/usr/local/cuda/bin:" + env.get("PATH", "")

    inner = str(KERNEL_ROOT / "tools" / "profile" / "phase1_inner_driver.py")
    # NB: --gpu-metrics-device is intentionally NOT used; probe reported
    # ERR_NVGPUCTRPERM in this container, so we rely on CUDA API + NVTX
    # traces (which never touch the PMU) instead.
    cmd = [
        "nsys",
        "profile",
        "-t", "cuda,nvtx,osrt",
        "-f", "true",
        "-o", str(shape_dir / "report"),
        "--stats=false",
        sys.executable, inner, shape.tag, "--calls", str(calls),
    ]
    start = time.time()
    res = _run(cmd, env=env, cwd=str(IMPORT_ROOT))
    wall_s = time.time() - start

    entry: Dict[str, Any] = {
        "tag": shape.tag,
        "cmd": " ".join(cmd),
        "rc": res.returncode,
        "wall_seconds": round(wall_s, 2),
        "stdout_tail": (res.stdout or "")[-500:],
        "stderr_tail": (res.stderr or "")[-500:],
        "report_exists": rep_file.exists(),
        "report_size_bytes": (
            rep_file.stat().st_size if rep_file.exists() else 0
        ),
    }

    if rep_file.exists():
        # Export sqlite for post-processing.  Uses nsys built-in exporter.
        sql_cmd = [
            "nsys", "export",
            "-t", "sqlite",
            "-f", "true",
            "-o", str(sqlite_file),
            str(rep_file),
        ]
        sql_res = _run(sql_cmd, env=env)
        entry["sqlite_export_rc"] = sql_res.returncode
        entry["sqlite_exists"] = sqlite_file.exists()
        entry["sqlite_size_bytes"] = (
            sqlite_file.stat().st_size if sqlite_file.exists() else 0
        )
    else:
        entry["sqlite_export_rc"] = -1
        entry["sqlite_exists"] = False

    # Per-shape run_meta.json
    meta = {
        "shape": {
            "tag": shape.tag,
            "T": shape.T,
            "d_in": shape.d_in,
            "d_out": shape.d_out,
            "hp_ratio": shape.hp_ratio,
            "model": shape.model,
            "proj": shape.proj,
            "note": shape.note,
        },
        "git_sha": _get_git_sha(),
        "gpu": _nvidia_smi_head(),
        "nsys_version": _nsys_version(),
        "start_time_utc": _dt.datetime.utcfromtimestamp(start).isoformat() + "Z",
        "end_time_utc": _dt.datetime.utcfromtimestamp(start + wall_s).isoformat() + "Z",
        "cmd_line": entry["cmd"],
        "inner_calls": calls,
        "rc": res.returncode,
        "report_exists": entry["report_exists"],
        "sqlite_exists": entry["sqlite_exists"],
    }
    (shape_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="artefact root directory",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="only capture these shape tags",
    )
    parser.add_argument(
        "--calls", type=int, default=10,
        help="profiled iterations per shape (inner driver)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing reports",
    )
    args = parser.parse_args()

    if shutil.which("nsys") is None:
        sys.stderr.write("nsys not found on PATH; aborting.\n")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    shapes = PHASE1_SHAPES
    if args.only:
        wanted = set(args.only)
        shapes = tuple(s for s in PHASE1_SHAPES if s.tag in wanted)
        missing = wanted - {s.tag for s in shapes}
        if missing:
            sys.stderr.write(f"unknown tags: {sorted(missing)}\n")
            return 2

    print(f"Collecting Phase 1 timelines for {len(shapes)} shapes -> {args.out_dir}")
    summary: List[Dict[str, Any]] = []
    t0 = time.time()
    for s in shapes:
        print(f"  * {s.tag} ...", flush=True)
        entry = _collect_one(s, args.out_dir, calls=args.calls, force=args.force)
        summary.append(entry)
        status = "OK" if entry.get("report_exists") else "FAIL"
        print(
            f"    -> {status} (rc={entry.get('rc')}, "
            f"report={entry.get('report_size_bytes', 0) // 1024}KB, "
            f"sqlite={entry.get('sqlite_size_bytes', 0) // 1024}KB, "
            f"wall={entry.get('wall_seconds')}s)"
        )

    index = {
        "timestamp_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "git_sha": _get_git_sha(),
        "nsys_version": _nsys_version(),
        "gpu": _nvidia_smi_head(),
        "shapes": summary,
        "total_wall_seconds": round(time.time() - t0, 2),
    }
    (args.out_dir / "run_index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    n_ok = sum(1 for e in summary if e.get("report_exists"))
    print(f"done: {n_ok}/{len(summary)} shapes captured; index -> {args.out_dir / 'run_index.json'}")
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
