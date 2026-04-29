"""Deprecated shim — superseded by :mod:`kernel.tools.profile.roofline_delta`.

The R49 Step 1 one-off script has been generalised.  This file now
exists only as a compatibility shim in case any external caller still
imports it by the old name; new code must use
``kernel.tools.profile.roofline_delta``.

Running the module directly replays the original R49 Step 1 report
(title fixed, bench path fixed), for anyone bisecting a past result.
It produces identical output to the pre-deprecation behaviour.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

from kernel.tools.profile.roofline_delta import render_delta_markdown

warnings.warn(
    "`_r49_roofline_delta` is deprecated; import "
    "`kernel.tools.profile.roofline_delta` instead.",
    DeprecationWarning,
    stacklevel=2,
)

BENCH_JSON = (
    Path(__file__).resolve().parents[2]
    / "cuda_kernel"
    / "logs"
    / "phase3_optimization"
    / "cuda_graph_bench"
    / "bench.json"
)


def main() -> None:
    import sys

    bench = json.loads(BENCH_JSON.read_text())
    render_delta_markdown(
        bench,
        title="R49 Step 1 — launch_sparse cluster",
        eager_label="eager",
        opt_label="graph",
        out=sys.stdout,
    )


if __name__ == "__main__":
    main()
