"""Inner process launched under ``nsys profile`` for Phase 1 timeline capture.

This script is deliberately minimal: it imports the production forward,
builds one Phase 1 shape, warms up under NVTX range ``warmup``, then
emits ``N_TIMELINE_CALLS`` forwards each wrapped in NVTX range
``iter_<i>``.  The outer orchestrator (``phase1_collect_timeline.py``)
is the one that invokes ``nsys profile -- python <this> <tag>``.

No reporting, no stats — that is the orchestrator's job after the
``.nsys-rep`` / ``.sqlite`` files are produced.

Usage
-----
    HKUST_V9_PROFILE=1 python phase1_inner_driver.py <shape_tag>

Requires
--------
- Environment variable ``HKUST_V9_PROFILE=1`` must be set so the
  NVTX shim in ops.py / dispatcher.py becomes active.  We assert this
  at startup to avoid silently producing a useless nsys-rep.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


# Make sure the 'kernel' namespace is importable regardless of cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_KERNEL_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _KERNEL_PARENT not in sys.path:
    sys.path.insert(0, _KERNEL_PARENT)


N_WARMUP = 50
N_TIMELINE_CALLS = 10  # keep small — nsys timeline wants a few clear iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shape_tag", help="Phase 1 shape tag to drive")
    parser.add_argument(
        "--calls",
        type=int,
        default=N_TIMELINE_CALLS,
        help="number of post-warmup iterations to profile",
    )
    args = parser.parse_args()

    if os.environ.get("HKUST_V9_PROFILE") != "1":
        sys.stderr.write(
            "phase1_inner_driver: HKUST_V9_PROFILE=1 must be set or NVTX "
            "ranges will not fire; refusing to run.\n"
        )
        return 2

    # Imports are done *after* the env-var check so a misconfigured run
    # fails before we allocate any CUDA resources.
    from kernel.tools.profile._phase1_shapes import build_shape_inputs
    from kernel.tools.profile.nvtx_shim import nvtx_range
    from kernel.backend import v9_linear_forward

    inputs = build_shape_inputs(args.shape_tag)
    X, W = inputs.X, inputs.W

    # Warmup under its own range so the orchestrator can filter it out.
    with nvtx_range("phase1.warmup"):
        for _ in range(N_WARMUP):
            v9_linear_forward(X, W)
    torch.cuda.synchronize()

    for i in range(args.calls):
        with nvtx_range(f"phase1.iter_{i}"):
            v9_linear_forward(X, W)
        torch.cuda.synchronize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
