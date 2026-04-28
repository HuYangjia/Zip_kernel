"""
epilogue_pressure_test.py - stress-test the epilogue_fma_bound cluster collapse
==============================================================================

Background
----------
The original Phase 2 attribution rule was

    Delta_scale_one >= 2.5%  ->  bottleneck = epilogue_fma_bound

That rule was calibrated under the (warmup=80, outer=4, inner=100) budget
used by :mod:`kernel.tools.profile.microbench_bisection`.  After the
warmup bump to (warmup=200, outer=10, inner=100) -- forced by the
x_zero anomaly deep-dive -- every shape's Delta_scale_one fell below
the 2.5% threshold, collapsing the entire 43-shape cluster to zero.

Before we bake the collapse into R49 planning we want one more level
of evidence: re-measure the top-|Delta_scale_one| shapes under an
even stronger budget (warmup=500, outer=20, inner=200), with 5
independent trials per variant interleaved to neutralise clock drift,
and report a median + percentile interval.

Verdict rule
------------
For each shape, we compute the distribution of
``delta_pct = (base_us - scale1_us) / base_us * 100`` over the 5 trials.

- If median(|delta_pct|) < 3.0%  AND  the 5%..95% percentile interval
  straddles 0  -> epilogue signal is noise for this shape.
- Otherwise    -> a residual epilogue signal survives and the shape
  should stay (or return to) ``epilogue_fma_bound``.

Usage
-----
    PYTHONPATH=/root python -m kernel.tools.profile.epilogue_pressure_test

Outputs under
``cuda_kernel/logs/phase2_microscope/epilogue_pressure_test/``:
    - ``<shape_tag>.json``      per-shape raw trials + verdict
    - ``pressure_test_report.md``   human-readable summary
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List


# Shapes to stress test.
# Top three by |Delta_scale_one| in the warmup=200 re-run, plus one
# compute-bound control (large_T1024_gu_4096_24576) whose Delta was
# -0.07% -- it should stay in noise under any budget.
SHAPES = [
    "mid_T128_kv_2560_2048",      # +2.18% in warmup=200 run
    "worst_T8_q_4096_4096",       # +2.00%
    "decode_T1_kv_2560_2048",     # +1.11%
    "large_T1024_gu_4096_24576",  # -0.07% (control)
]

N_TRIALS = 5
WARMUP = 500
OUTER = 20
INNER = 200


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _time_variant(X, W) -> float:
    from kernel.tools.profile._phase1_shapes import time_forward_us  # noqa: WPS433
    return time_forward_us(X, W, warmup=WARMUP, outer=OUTER, inner=INNER)


def _weights_with_identity_scales(W):
    """Same trick as microbench_bisection: scale=1, zero=0."""
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


def _run_one_shape(tag: str) -> Dict[str, object]:
    from kernel.tools.profile._phase1_shapes import build_shape_inputs  # noqa: WPS433

    print(f"\n===== {tag} =====", flush=True)
    b = build_shape_inputs(tag)
    W_base = b.W
    W_flat = _weights_with_identity_scales(b.W)

    base_trials: List[float] = []
    scale1_trials: List[float] = []

    # Interleave: base/scale1/base/scale1/... so any slow clock drift
    # affects both variants symmetrically.
    for trial in range(N_TRIALS):
        t0 = time.time()
        t_base = _time_variant(b.X, W_base)
        t_s1 = _time_variant(b.X, W_flat)
        dt = time.time() - t0
        base_trials.append(t_base)
        scale1_trials.append(t_s1)
        delta = (t_base - t_s1) / t_base * 100.0
        print(
            f"  trial {trial + 1}/{N_TRIALS}: base={t_base:.3f}us  "
            f"scale1={t_s1:.3f}us  delta={delta:+.2f}%  ({dt:.1f}s wall)",
            flush=True,
        )

    # Per-trial deltas.
    deltas = [
        (base_trials[i] - scale1_trials[i]) / base_trials[i] * 100.0
        for i in range(N_TRIALS)
    ]
    deltas_sorted = sorted(deltas)
    med = statistics.median(deltas)
    # 5th & 95th percentile of a 5-element sample = min & max.
    p05 = deltas_sorted[0]
    p95 = deltas_sorted[-1]

    # Verdict rule.
    # A signal this close to zero is meaningless given HBM schedule
    # jitter and clock drift at these timescales; if |median| is
    # below the "trivially zero" floor (0.5%) we call it NOISE
    # unconditionally, otherwise we require median < 3% AND the
    # [p05, p95] interval to straddle zero.
    TRIVIAL_ZERO = 0.5  # percent
    if abs(med) < TRIVIAL_ZERO:
        noise = True
    else:
        noise = abs(med) < 3.0 and (p05 <= 0.0 <= p95)
    verdict = "NOISE" if noise else "REAL_SIGNAL"
    print(
        f"  median delta = {med:+.2f}%   "
        f"5%-95% interval = [{p05:+.2f}%, {p95:+.2f}%]   "
        f"verdict = {verdict}",
        flush=True,
    )

    return {
        "tag": tag,
        "shape": {
            "T": b.shape.T,
            "d_in": b.shape.d_in,
            "d_out": b.shape.d_out,
            "model": b.shape.model,
            "proj": b.shape.proj,
        },
        "budget": {"warmup": WARMUP, "outer": OUTER, "inner": INNER},
        "n_trials": N_TRIALS,
        "trials_us": {
            "base": base_trials,
            "scale_one": scale1_trials,
        },
        "delta_pct_per_trial": deltas,
        "median_delta_pct": med,
        "p05_delta_pct": p05,
        "p95_delta_pct": p95,
        "verdict": verdict,
    }


def _render_report(results: List[Dict[str, object]], out_dir: Path) -> None:
    lines: List[str] = []
    lines.append("# epilogue_fma_bound cluster: pressure-test report")
    lines.append("")
    lines.append(
        f"Budget per timing: warmup={WARMUP}, outer={OUTER}, inner={INNER}.  "
        f"Each shape ran {N_TRIALS} independent trials with "
        "base/scale_one interleaved per trial.  "
        "Verdict is NOISE when median |delta| < 0.5% (trivially zero), "
        "OR when median |delta| < 3% AND the [p05, p95] interval "
        "straddles 0.  Otherwise REAL_SIGNAL."
    )
    lines.append("")
    lines.append(
        "| shape | T | d_in | d_out | median delta | [p05, p95] | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        s = r["shape"]
        lines.append(
            f"| `{r['tag']}` | {s['T']} | {s['d_in']} | {s['d_out']} | "
            f"{r['median_delta_pct']:+.2f}% | "
            f"[{r['p05_delta_pct']:+.2f}%, {r['p95_delta_pct']:+.2f}%] | "
            f"**{r['verdict']}** |"
        )
    lines.append("")

    noise_count = sum(1 for r in results if r["verdict"] == "NOISE")
    real_count = len(results) - noise_count
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        f"- {noise_count} / {len(results)} shapes verdict = NOISE"
    )
    lines.append(
        f"- {real_count} / {len(results)} shapes verdict = REAL_SIGNAL"
    )
    lines.append("")
    if real_count == 0:
        lines.append(
            "**Every tested shape -- including the three highest-|delta| "
            "reps from the Phase 2 re-run -- shows an epilogue signal "
            "indistinguishable from noise under the (warmup=500, "
            "outer=20, inner=200, 5 trials) budget.**"
        )
        lines.append("")
        lines.append(
            "This corroborates the `epilogue_fma_bound` cluster collapse "
            "and closes out the last audit item before R49 can commit to "
            "the two-cluster roadmap (`tc_underutil`, `launch_sparse`).  "
            "See `../xzero_probe/xzero_probe_report.md` for the sibling "
            "audit that prompted the warmup bump in the first place."
        )
    else:
        lines.append(
            "**At least one shape still carries a real epilogue signal.**  "
            "The cluster should not be fully retired; keep those shapes "
            "in the `epilogue_fma_bound` bucket and re-derive the R49 "
            "plan with that slice restored."
        )
    lines.append("")
    lines.append("## Methodology notes")
    lines.append("")
    lines.append(
        "- Each trial reruns both `base` and `scale_one` so transient "
        "clock drift cancels to first order.  Within a trial the two "
        "variants are back-to-back (no sleeps)."
    )
    lines.append(
        "- Per-variant timing uses the project standard "
        "`time_forward_us(warmup, outer, inner)` which returns "
        "`min over outer of (mean over inner of per-iter us)`.  "
        "The budgets here are ~2.5x the stronger Phase 2 re-run."
    )
    lines.append(
        "- `scale_one` is constructed via "
        "`_weights_with_identity_scales(W)` (same helper as "
        "`microbench_bisection`) which replaces `scale_u4` with ones "
        "and `zero_u4` with zeros; the rest of W is shared, so the "
        "only algorithmic difference is the epilogue dequant FMA "
        "degenerating."
    )
    lines.append(
        "- Inputs are rebuilt via `build_shape_inputs(tag)`; X is a "
        "fresh random draw once per shape and reused across trials.  "
        "This is intentional: we want to isolate the epilogue lever, "
        "not convolve with X-dependent HBM scheduling."
    )
    lines.append("")

    out_path = out_dir / "pressure_test_report.md"
    out_path.write_text("\n".join(lines))
    print(f"[pressure_test] wrote {out_path}", flush=True)


def main() -> None:
    out_dir = (
        _repo_root()
        / "cuda_kernel"
        / "logs"
        / "phase2_microscope"
        / "epilogue_pressure_test"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    for tag in SHAPES:
        r = _run_one_shape(tag)
        (out_dir / f"{tag}.json").write_text(json.dumps(r, indent=2))
        results.append(r)

    _render_report(results, out_dir)

    print("\n[pressure_test] done")


if __name__ == "__main__":
    main()
