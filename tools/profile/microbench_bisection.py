"""Phase 2 microbench bisection (task-item.md step 10).

For each of the 8 Phase-1/2 shapes we run a short battery of
"what-if" microbenchmarks that each mutate *one* variable.  The
resulting delta in wall-clock vs a reference measurement is the
bisection signal used to attribute the primary bottleneck.

Only the three experiments that require **no kernel code change**
are wired in here (D.1, D.2, D.3 from the plan).  D.4 / D.5 require
altered launcher signatures and are deferred to Phase 3.

Timing contract
---------------
We import :func:`kernel.tools.profile._phase1_shapes.time_forward_us`
which already satisfies the long-term memo's "3-piece microbench"
rule: >=50 warm-up iterations, >=100 iter per window, >=3 windows,
and min-of-means return.

Experiments
-----------
- **base**   : the vanilla inputs returned by :func:`build_shape_inputs`.
             This is the reference that D.1/D.2/D.3 deltas are compared
             against.
- **L2 hot** : before timing, we pre-touch the weights 200 times so
             they are warm in L2; if ``base`` is HBM-bound we expect a
             noticeable speed-up here.  If the delta is < 3% the
             bottleneck is *not* HBM bandwidth on the weights.
- **X = 0**  : overwrite ``X`` with zeros.  Activation-quant cost is
             unchanged (same shape, same HBM traffic), but the
             ``fused_dense_sparse`` MMA body sees only zero operands
             -> if the GPU executes a zero-short-circuit path or the
             FMA chain benefits from trivial-operand forwarding we'll
             see a small but detectable delta.  More importantly this
             isolates any *data-dependent* latency (none expected).
- **scale=1**: overwrite ``scale_u4`` and ``zero_u4`` with 1 and 0.
             The epilogue FMA per output point degenerates to a
             plain accumulate.  If this gives a meaningful speed-up
             the kernel is FMA-epilogue-bound (matches the SASS
             finding of FMA% >> TC%).

Output
------
Writes one JSON per shape at
``cuda_kernel/logs/phase2_microscope/<tag>/bisection.json`` with
the four timings plus derived deltas in %.

Usage::

    python -m kernel.tools.profile.microbench_bisection
    # or
    python -m kernel.tools.profile.microbench_bisection --shape mid_T128_kv_2560_2048
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# This script has two modes:
#   1. outer mode: spawn one subprocess per shape (default)
#   2. inner mode: ``--inner <tag>`` -> run the 4 experiments for one shape
# The split keeps per-shape GPU state hermetic (import order, JIT caches).


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# INNER mode
# ---------------------------------------------------------------------------
# Budget notes
# ------------
# The earlier (warmup=80, outer=4, inner=100) schedule produced a spurious
# -27.9% ``d_xzero`` signal on ``mid_T128_kv_2560_2048`` that did not
# reproduce under (warmup=200, outer=10, inner=200) in
# :mod:`kernel.tools.profile.xzero_probe`.  See that script's report for
# the ablation.  Since the whole bisection takes only a few seconds per
# shape anyway, we standardise on the stronger schedule to keep the
# deltas below the |Δ| < 3% noise floor stated in the rendered summary.
def _time_variant(X, W, warmup: int = 200, outer: int = 10, inner: int = 100) -> float:
    from kernel.tools.profile._phase1_shapes import time_forward_us  # noqa: WPS433
    return time_forward_us(X, W, warmup=warmup, outer=outer, inner=inner)


def _l2_prewarm_weights(W, rounds: int = 200) -> None:
    """Repeatedly read W weight tensors to park them in L2.

    A simple ``.sum()`` over the packed low-bits and scale arrays is
    enough to force a full HBM->L2 staircase; on a 6 MB L2 the large
    weight tensors don't fit entirely but we at least force the
    working-set blocks that the kernel will touch next to be hot.
    """
    import torch  # noqa: WPS433

    accum = torch.zeros(1, dtype=torch.float32, device=W.W_low_packed.device)
    for _ in range(rounds):
        accum += W.W_low_packed.to(torch.int32).sum()
        accum += W.scale_u4.sum().float()
        accum += W.zero_u4.sum().float()
    # ``accum`` is never read - we only want the traffic.
    torch.cuda.synchronize(W.W_low_packed.device)


def run_inner(tag: str, out_root: Path) -> Dict[str, object]:
    import torch  # noqa: WPS433
    from kernel.tools.profile._phase1_shapes import build_shape_inputs  # noqa: WPS433

    print(f"[bisection] tag={tag} building inputs ...", flush=True)
    b = build_shape_inputs(tag)

    results: Dict[str, float] = {}

    # A. base
    t_base = _time_variant(b.X, b.W)
    results["base_us"] = t_base
    print(f"[bisection]   base        = {t_base:.3f} us", flush=True)

    # D.1 L2 hot W — prewarm then measure (without rebuilding)
    _l2_prewarm_weights(b.W, rounds=200)
    t_l2 = _time_variant(b.X, b.W)
    results["l2_hot_us"] = t_l2
    print(f"[bisection]   l2_hot      = {t_l2:.3f} us", flush=True)

    # D.2 X = 0
    X_zero = torch.zeros_like(b.X)
    t_xzero = _time_variant(X_zero, b.W)
    results["x_zero_us"] = t_xzero
    print(f"[bisection]   x_zero      = {t_xzero:.3f} us", flush=True)

    # D.3 scale = 1, zero = 0 -> epilogue becomes trivial
    W_flat = _weights_with_identity_scales(b.W)
    t_scale = _time_variant(b.X, W_flat)
    results["scale_one_us"] = t_scale
    print(f"[bisection]   scale_one   = {t_scale:.3f} us", flush=True)

    # Derived deltas (positive = faster than base).
    def _pct(vs: float) -> float:
        return (t_base - vs) / t_base * 100.0

    deltas = {
        "l2_hot_delta_pct":    _pct(t_l2),
        "x_zero_delta_pct":    _pct(t_xzero),
        "scale_one_delta_pct": _pct(t_scale),
    }

    shape_meta = {
        "tag": tag,
        "T": b.shape.T,
        "d_in": b.shape.d_in,
        "d_out": b.shape.d_out,
        "model": b.shape.model,
        "proj": b.shape.proj,
        "n_hp_blocks": b.meta["n_hp_blocks"],
    }
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bisection.json"
    payload = {"shape": shape_meta, "timings_us": results, "deltas_pct": deltas}
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[bisection]   wrote {out_path}", flush=True)
    return payload


def _weights_with_identity_scales(W):
    """Clone W with scale=1, zero=0 so epilogue FMA degenerates."""
    import torch  # noqa: WPS433
    from kernel.triton_kernel.pack_utils import V9WeightContainer  # noqa: WPS433

    flat_scale = torch.ones_like(W.scale_u4)
    flat_zero = torch.zeros_like(W.zero_u4)
    return V9WeightContainer(
        W_low_packed=W.W_low_packed,
        W_high_blocks_packed=W.W_high_blocks_packed,
        scale_u4=flat_scale,
        zero_u4=flat_zero,
        hp_row_offsets=W.hp_row_offsets,
        hp_col_indices=W.hp_col_indices,
        perm=W.perm,
        block_shape=W.block_shape,
        d_out=W.d_out,
        d_in=W.d_in,
    )


# ---------------------------------------------------------------------------
# OUTER mode
# ---------------------------------------------------------------------------
def run_outer(tags: List[str], out_root: Path) -> None:
    me = Path(__file__).resolve()
    for tag in tags:
        print(f"\n===== {tag} =====", flush=True)
        cmd = [
            sys.executable,
            "-m", "kernel.tools.profile.microbench_bisection",
            "--inner", tag,
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", "/root")  # cwd is caller's
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print(f"[outer] tag={tag} subprocess failed rc={res.returncode}")


def _render_summary(root: Path) -> None:
    """Render a consolidated bisection table across all shapes."""
    rows: List[Dict[str, object]] = []
    for sub in sorted(root.iterdir()):
        p = sub / "bisection.json"
        if p.is_file():
            rows.append(json.loads(p.read_text()))
    if not rows:
        print("[bisection] no bisection.json files found under", root)
        return

    out = []
    out.append("# Phase 2 microbench bisection summary\n")
    out.append(
        "Each row is a representative shape.  `base` is a fresh build;\n"
        "`l2_hot / x_zero / scale_one` are the three no-code-change\n"
        "experiments.  `Δ%` = (base - variant) / base * 100 (positive\n"
        "= faster).  |Δ| >= 3 % is considered meaningful (below that\n"
        "is dominated by timer jitter even with min-of-means).\n"
    )
    out.append(
        "| shape | T | base_us | l2_hot_us | Δ_l2% | x_zero_us | Δ_xzero% | scale1_us | Δ_scale1% |"
    )
    out.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in rows:
        s = r["shape"]
        t = r["timings_us"]
        d = r["deltas_pct"]
        out.append(
            f"| {s['tag']} | {s['T']} | "
            f"{t['base_us']:.2f} | {t['l2_hot_us']:.2f} | {d['l2_hot_delta_pct']:+.1f} | "
            f"{t['x_zero_us']:.2f} | {d['x_zero_delta_pct']:+.1f} | "
            f"{t['scale_one_us']:.2f} | {d['scale_one_delta_pct']:+.1f} |"
        )
    out.append("")
    out.append("## Interpretation cheat-sheet\n")
    out.append("- **Δ_l2 >= 3 %**  → kernel is HBM-weight bound (L2 hot fetch helped).")
    out.append("- **Δ_xzero >= 3 %** → data-dependent latency on activation path (rare).")
    out.append("- **Δ_scale1 >= 3 %** → epilogue FMA is a real consumer; scale/zero path dominates tail.")
    out.append("- all three small and base close to roofline mem → memory-bound with no headroom.")
    out.append("- all three small and base far from roofline      → compute-bound / occupancy-bound (TC_underutil).")
    out.append("")
    summary_path = root / "bisection_summary.md"
    summary_path.write_text("\n".join(out))
    print(f"[bisection] wrote {summary_path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--inner",
        type=str,
        default=None,
        help="If given, run inner mode for this single shape tag and exit.",
    )
    ap.add_argument(
        "--shape",
        type=str,
        default=None,
        help="Restrict outer mode to a single shape tag (for debugging).",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=_repo_root() / "cuda_kernel/logs/phase2_microscope",
    )
    args = ap.parse_args(argv)

    if args.inner is not None:
        run_inner(args.inner, args.out_root)
        return 0

    # Outer mode: spawn a subprocess per shape.
    from kernel.tools.profile._phase1_shapes import PHASE_ALL_SHAPES  # noqa: WPS433
    if args.shape:
        tags = [args.shape]
    else:
        tags = [s.tag for s in PHASE_ALL_SHAPES]
    args.out_root.mkdir(parents=True, exist_ok=True)
    run_outer(tags, args.out_root)
    _render_summary(args.out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
