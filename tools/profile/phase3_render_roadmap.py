"""Render Phase 3 rewrite roadmap (task-item.md step 14).

Consumes:
  - ``cuda_kernel/logs/phase2_microscope/shape_clusters.csv``
  - ``cuda_kernel/logs/phase2_microscope/shape_targets.csv``
  - ``cuda_kernel/logs/phase2_microscope/cluster_summary.json``
  - ``cuda_kernel/logs/phase2_microscope/phase2_kernel_microscope_report.md``

Emits:
  - ``.codebuddy/plan/kernel_eff_diagnosis_and_rewrite/phase3_roadmap.md``

Per-cluster fields (per requirements.md §5.1):
  cluster_id / bottleneck_type / member_shapes_count /
  current_eff_median / target_eff_median /
  proposed_technique / expected_eff_after / effort_days

Clusters are sorted by ROI = ``members * eff_gain / effort_days`` descending.

Each cluster entry also lists:
  - new dependencies (CUTLASS, cp.async, CUDA Graphs, ...) + risk notes
  - "how to verify with microbench after implementation" recipe
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
P2_DIR = REPO / "cuda_kernel/logs/phase2_microscope"
ROADMAP_PATH = REPO.parent / ".codebuddy/plan/kernel_eff_diagnosis_and_rewrite/phase3_roadmap.md"

# NOTE: the planning tree lives at ``<workspace>/.codebuddy/plan/...`` while
# this script is at ``<workspace>/kernel/tools/profile/...``, so going up
# three times (``parents[2]``) lands in ``kernel/``; one more ``parent``
# gets us to the workspace root.  If the layout ever moves we'll fail
# fast at write time rather than silently writing to the wrong place.


# ---------------------------------------------------------------------------
# Static knowledge base — per-cluster rewrite plan
# ---------------------------------------------------------------------------
@dataclass
class ClusterPlan:
    bottleneck: str
    proposed_technique: str
    # expected_eff is a *post-rewrite* estimate.  We take a conservative
    # fraction of the roofline:
    #   - launch_sparse    : 0.75 (CUDA Graph removes 70% of launch tax
    #                              but still leaves some dispatcher cost)
    #   - tc_underutil     : 0.60 (switching to real HMMA + proper tile
    #                              should reach 60% with current weight
    #                              layout; 80% needs CUTLASS persistent)
    #   - epilogue_fma_bound: 0.70 (fused mul-add epilogue + vectorised
    #                              scale broadcast)
    #   - x_zero_anomaly   : 0.70 (fix the branch, then re-cluster into
    #                              whichever canonical bucket it falls in)
    expected_eff_after: float
    effort_days: float
    dependencies: List[str] = field(default_factory=list)
    risk_notes: str = ""
    verification_recipe: str = ""


PLANS: Dict[str, ClusterPlan] = {
    "launch_sparse": ClusterPlan(
        bottleneck="launch_sparse",
        proposed_technique=(
            "Capture `activation_quant -> fused_dense_sparse -> dequant` as a single "
            "**CUDA Graph** per (shape, dtype) signature; cache in the dispatcher "
            "keyed on `(T, d_in, d_out, hp_ratio)` and replay instead of issuing "
            "three individual launches.  Secondary: merge activation_quant and the "
            "dense GEMV into one fused kernel for T<=8 decode so the body itself "
            "is a single launch."
        ),
        expected_eff_after=0.75,
        effort_days=4.0,
        dependencies=[
            "torch.cuda.CUDAGraph (stock in torch 2.8)",
            "dispatcher cache that keys on (shape, dtype, device) and invalidates "
            "on weight-ptr change",
        ],
        risk_notes=(
            "CUDA Graph capture forbids any dynamic allocation / host-sync in the "
            "captured region.  `activation_quant` currently allocates scratch — "
            "need to pre-allocate.  Graph replay also breaks if the caller passes "
            "a different tensor identity per forward; we must wire the "
            "graph replay behind an opt-in flag that the outer Python layer honours."
        ),
        verification_recipe=(
            "Re-run `microbench_bisection.py --shape decode_T1_q_2048_2048` with "
            "and without CUDA Graph; `Δ_graph >= 60%` proves launch-tax removal.  "
            "Check Phase 1 `launch_tax.json`: `launch_tax_pct_of_plain` should "
            "drop from ~70% to <15%."
        ),
    ),
    "tc_underutil": ClusterPlan(
        bottleneck="tc_underutil",
        proposed_technique=(
            "Rewrite `fused_dense_sparse_mma_int4` to actually emit **HMMA.16816.F16** "
            "or **IMMA.8816.S8** via `cutlass::arch::Mma` templates.  Current "
            "SASS shows TC% < 2% across all 42 kernels; the FMA chain from "
            "dequant-then-multiply forces CUDA-core execution.  Plan: "
            "(a) dequant into shared-memory tile in a separate warp, "
            "(b) producer/consumer async pipeline with `cp.async.bulk` so HBM "
            "loads overlap TC compute, (c) use CUTLASS 3.x `CollectiveBuilder` "
            "for tile-level orchestration to avoid a hand-rolled pipeline."
        ),
        expected_eff_after=0.60,
        effort_days=10.0,
        dependencies=[
            "CUTLASS 3.5+ headers (vendored or git submodule)",
            "sm_89 target + `-arch=sm_89` so `cp.async.bulk` is available",
            "new weight layout: interleave W_low + W_scale so a single async "
            "copy fills both sub-tiles",
        ],
        risk_notes=(
            "CUTLASS 3.x adds ~10MB of headers and slows incremental build by "
            "~20s; we gate the new path behind `HKUST_V9_USE_CUTLASS=1` and "
            "keep the current kernel as fallback for sm<89.  Accuracy risk: "
            "the dequant-before-MMA pattern changes accumulation order vs "
            "current dequant-after-FMA — run the full unit-test suite in "
            "`triton_kernel/tests/` as a regression gate (re-purpose them as "
            "numerical equivalence tests against FP16)."
        ),
        verification_recipe=(
            "Re-run SASS analyser on the new `.so`; `TC% > 20%` on all "
            "`*_mma_int4_kernel` entries.  Re-run roofline bench on the 8 "
            "representative shapes; `cuda_efficiency` must reach >=0.50 on all "
            "`tc_underutil` cluster members or we're regressing the plan."
        ),
    ),
    "epilogue_fma_bound": ClusterPlan(
        bottleneck="epilogue_fma_bound",
        proposed_technique=(
            "Fuse the `scale * acc + zero` epilogue into the main MMA inner loop "
            "so it runs in registers without a separate FMA pass.  Additionally "
            "vectorise the per-block `scale_u4/zero_u4` broadcast via "
            "`__ldg` + `half2` unpack so the epilogue touches HBM once per 32 "
            "outputs instead of once per 1.  For the down_proj shapes (narrow "
            "d_out) also add an **inter-SM reduction** via splitK=4 to amortise "
            "the epilogue cost over more warps."
        ),
        expected_eff_after=0.70,
        effort_days=6.0,
        dependencies=[
            "no new external dependency; pure kernel refactor",
            "new kernel launcher arg `scale_broadcast_mode ∈ {per_row, per_block}` "
            "so we can A/B test without touching callers",
        ],
        risk_notes=(
            "splitK + epilogue fusion together means the partial-sum reduction "
            "kernel now also needs to run the scale/zero correction.  Risk of "
            "double-applying the correction if the splitK merge runs the "
            "epilogue once and the main kernel also runs it — add a compile-time "
            "`kEpilogueInMain` flag and assert the two kernels are compiled "
            "with opposite values."
        ),
        verification_recipe=(
            "Re-run `microbench_bisection.py` D.3 (scale=1) on the 43 cluster "
            "members; `Δ_scale1 < 1%` after rewrite proves the epilogue no "
            "longer dominates.  Roofline re-bench should lift the "
            "cluster's median `cuda_efficiency` from current 0.28 to >=0.55."
        ),
    ),
    "x_zero_anomaly": ClusterPlan(
        bottleneck="x_zero_anomaly",
        proposed_technique=(
            "Investigate `mid_T128_kv_2560_2048` specifically: X=0 is **27.9% "
            "SLOWER** than random input, which is the opposite of the usual "
            "zero-short-circuit.  Hypothesis: `activation_quant` computes "
            "`scale_x = max_abs(X) / 7` and when `max_abs=0` we fall into a "
            "degenerate epsilon branch, OR the sparse path's `sum_X==0` "
            "triggers a code path that skips a pipeline stage and serialises.  "
            "Fix: add a fast-path that returns zeros directly when `scale_x < "
            "eps`, and audit the sparse reduction for zero-input branches."
        ),
        expected_eff_after=0.70,  # after fix, inherits epilogue_fma_bound profile
        effort_days=1.5,
        dependencies=[],
        risk_notes=(
            "Pure investigation first.  Once the branch is identified the fix "
            "is a few lines.  Risk: the anomaly may actually be an HBM "
            "pre-fetcher artefact of the random-input case (i.e. random input "
            "is *faster* than expected, not that zero input is slower); then "
            "there is nothing to fix and we reclassify the shape."
        ),
        verification_recipe=(
            "Re-run D.2 (X=0) microbench post-fix on `mid_T128_kv_2560_2048`; "
            "`|Δ_xzero| < 3%` as the success criterion."
        ),
    ),
    "physics_loss": ClusterPlan(
        bottleneck="physics_loss",
        proposed_technique=(
            "Route these shapes to FP16 cuBLAS via `policy.py`; the W4A4 "
            "theoretical floor is below the FP16 floor so kernel optimisation "
            "cannot help."
        ),
        expected_eff_after=0.0,  # N/A - not a kernel target
        effort_days=0.5,
        dependencies=["kernel/backend/policy.py dispatch entry"],
        risk_notes="None; the FP16 path is already production-stable.",
        verification_recipe="policy.py unit test asserts these shapes land on FP16.",
    ),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class ShapeRec:
    model: str
    proj: str
    T: int
    d_in: int
    d_out: int
    cuda_eff: float
    fp16_eff: float
    target_eff: float
    gap_pp: float
    status: str
    primary_bottleneck: str
    nearest_rep_tag: str


def _load_targets_csv(path: Path) -> List[ShapeRec]:
    # Skip the 3 comment lines at the top (they all start with "#").
    with path.open() as f:
        rows = [l for l in f.read().splitlines() if not l.startswith("#")]
    reader = csv.DictReader(rows)
    out: List[ShapeRec] = []
    for r in reader:
        out.append(ShapeRec(
            model=r["model"],
            proj=r["proj"],
            T=int(r["T"]),
            d_in=int(r["d_in"]),
            d_out=int(r["d_out"]),
            cuda_eff=float(r["cuda_eff"]),
            fp16_eff=float(r["fp16_eff"]),
            target_eff=float(r["target_eff"]),
            gap_pp=float(r["gap_pp"]),
            status=r["status"],
            primary_bottleneck=r["primary_bottleneck"],
            nearest_rep_tag=r["nearest_rep_tag"],
        ))
    return out


# ---------------------------------------------------------------------------
# Aggregate per cluster
# ---------------------------------------------------------------------------
def _aggregate(recs: List[ShapeRec]) -> Dict[str, Dict[str, object]]:
    by_b: Dict[str, List[ShapeRec]] = {}
    for r in recs:
        by_b.setdefault(r.primary_bottleneck, []).append(r)

    agg: Dict[str, Dict[str, object]] = {}
    for bn, members in by_b.items():
        plan = PLANS.get(bn)
        cur_med = statistics.median(m.cuda_eff for m in members)
        tgt_med = statistics.median(
            m.target_eff if m.status != "physics_loss" else 0.0 for m in members
        )
        exp_after = plan.expected_eff_after if plan else cur_med
        eff_gain = max(exp_after - cur_med, 0.0)
        effort = plan.effort_days if plan else 1.0
        roi = (len(members) * eff_gain / effort) if effort else 0.0

        agg[bn] = {
            "bottleneck": bn,
            "members": members,
            "count": len(members),
            "current_eff_median": cur_med,
            "target_eff_median": tgt_med,
            "expected_eff_after": exp_after,
            "eff_gain": eff_gain,
            "effort_days": effort,
            "roi": roi,
            "plan": plan,
        }
    return agg


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
def render(recs: List[ShapeRec]) -> str:
    agg = _aggregate(recs)
    # Sort by ROI descending; physics_loss always last.
    ordered = sorted(
        agg.values(),
        key=lambda a: (a["bottleneck"] == "physics_loss", -float(a["roi"])),
    )

    # Status counts for the header summary.
    status_counts: Dict[str, int] = {}
    for r in recs:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    out: List[str] = []
    out.append("# Phase 3 — Kernel Rewrite Roadmap")
    out.append("")
    out.append(
        "_Generated by `kernel/tools/profile/phase3_render_roadmap.py` from "
        "Phase 2 artefacts._"
    )
    out.append("")
    out.append("## 0. Current state")
    out.append("")
    out.append(
        f"- Total shapes: **{len(recs)}**  "
        f"(on_track: {status_counts.get('on_track',0)}, "
        f"gap_small: {status_counts.get('gap_small',0)}, "
        f"gap_large: {status_counts.get('gap_large',0)}, "
        f"physics_loss: {status_counts.get('physics_loss',0)})"
    )
    med_gap = statistics.median(
        r.gap_pp for r in recs if r.status != "physics_loss"
    )
    out.append(f"- Median gap to `target_eff = min(0.80, fp16_eff * 0.90)`: **{med_gap:+.1f} pp**")
    out.append(
        "- `target_eff` definition: see requirements.md §1.1.  Buckets: "
        "`on_track` (cuda_eff >= target), `gap_small` (<20 pp), "
        "`gap_large` (>=20 pp), `physics_loss` (cuda_roof > fp16_roof)."
    )
    out.append("")

    # -- Cluster summary table --
    out.append("## 1. Cluster priority (ROI ordered)")
    out.append("")
    out.append(
        "| # | cluster | bottleneck | members | cur_eff_med | target_eff_med | "
        "expected_after | Δeff | effort_d | ROI |"
    )
    out.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for i, a in enumerate(ordered, 1):
        out.append(
            f"| {i} | `{a['bottleneck']}` | "
            f"{_bottleneck_label(a['bottleneck'])} | "
            f"{a['count']} | {a['current_eff_median']:.3f} | "
            f"{a['target_eff_median']:.3f} | {a['expected_eff_after']:.3f} | "
            f"{a['eff_gain']:+.3f} | {a['effort_days']:.1f} | {a['roi']:.2f} |"
        )
    out.append("")
    out.append(
        "> ROI = `members * Δeff / effort_days`.  It does not monetise the "
        "cost of regressing a production path — interpret alongside each "
        "cluster's risk notes below."
    )
    out.append("")

    # -- Per-cluster detail --
    out.append("## 2. Per-cluster rewrite plan")
    out.append("")
    for i, a in enumerate(ordered, 1):
        plan: Optional[ClusterPlan] = a["plan"]  # type: ignore[assignment]
        bn = a["bottleneck"]
        out.append(f"### 2.{i} `{bn}` — {a['count']} shapes")
        out.append("")
        if plan is None:
            out.append("_No rewrite plan defined for this bottleneck._")
            out.append("")
            continue
        out.append(f"**Proposed technique.** {plan.proposed_technique}")
        out.append("")
        out.append(
            f"**Expected eff after rewrite:** median {plan.expected_eff_after:.2f} "
            f"(current median {a['current_eff_median']:.2f}, "
            f"Δ = {a['eff_gain']:+.2f})."
        )
        out.append("")
        if plan.dependencies:
            out.append("**New dependencies:**")
            for d in plan.dependencies:
                out.append(f"  - {d}")
            out.append("")
        if plan.risk_notes:
            out.append(f"**Risk notes.** {plan.risk_notes}")
            out.append("")
        if plan.verification_recipe:
            out.append(f"**Verification recipe.** {plan.verification_recipe}")
            out.append("")

        # Member shape sample (top 10 by gap_pp so the reader sees the
        # worst offenders first).
        members: List[ShapeRec] = a["members"]  # type: ignore[assignment]
        sample = sorted(members, key=lambda m: -m.gap_pp)[:10]
        out.append(f"**Sample member shapes (top 10 of {len(members)} by gap):**")
        out.append("")
        out.append("| model | proj | T | d_in | d_out | cuda_eff | target | gap_pp |")
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for m in sample:
            out.append(
                f"| {m.model} | {m.proj} | {m.T} | {m.d_in} | {m.d_out} | "
                f"{m.cuda_eff:.3f} | {m.target_eff:.3f} | {m.gap_pp:+.1f} |"
            )
        out.append("")

    # -- Verification matrix --
    out.append("## 3. Verification matrix (\"done\" criteria)")
    out.append("")
    out.append("| cluster | metric | source | threshold |")
    out.append("|---|---|---|---|")
    for a in ordered:
        bn = a["bottleneck"]
        if bn == "launch_sparse":
            out.append(f"| `{bn}` | `launch_tax_pct_of_plain` | Phase 1 `launch_tax.json` | <15% |")
        elif bn == "tc_underutil":
            out.append(f"| `{bn}` | SASS `TC%` on MMA kernels | `sass_profile.json` | >=20% |")
            out.append(f"| `{bn}` | median `cuda_efficiency` | new roofline_report | >=0.50 |")
        elif bn == "epilogue_fma_bound":
            out.append(f"| `{bn}` | `Δ_scale1` in bisection | `bisection.json` | <1% |")
            out.append(f"| `{bn}` | median `cuda_efficiency` | new roofline_report | >=0.55 |")
        elif bn == "x_zero_anomaly":
            out.append(f"| `{bn}` | `|Δ_xzero|` in bisection | `bisection.json` | <3% |")
        elif bn == "physics_loss":
            out.append(f"| `{bn}` | policy.py routing test | pytest | passes |")
    out.append("")

    # -- Next step (Task 17 anchor) --
    top = ordered[0]
    out.append("## 4. Immediate next R49 task")
    out.append("")
    out.append(
        f"The highest-ROI cluster is **`{top['bottleneck']}`** "
        f"({top['count']} shapes, ROI {top['roi']:.2f}).  R49 should start "
        f"with the rewrite technique described in §2.1 and use the "
        f"verification recipe in §3 as the exit criterion."
    )
    out.append("")
    on_track = status_counts.get("on_track", 0)
    out.append(
        f"Current `on_track` count: **{on_track} / {len(recs)}**.  "
        f"Path to 90/100: execute clusters 1+2 in §1 (projected combined lift "
        f"on {ordered[0]['count'] + (ordered[1]['count'] if len(ordered)>1 else 0)} "
        f"shapes)."
    )
    out.append("")
    if on_track < 90:
        gap_large = [r for r in recs if r.status == "gap_large"]
        out.append(
            f"### Gap-large shapes not yet on track ({len(gap_large)}):"
        )
        out.append("")
        # Group by cluster for readability.
        by_bn: Dict[str, List[ShapeRec]] = {}
        for r in gap_large:
            by_bn.setdefault(r.primary_bottleneck, []).append(r)
        for bn in sorted(by_bn, key=lambda k: -len(by_bn[k])):
            members = by_bn[bn]
            out.append(f"- `{bn}`: {len(members)} shapes (see §2 for rewrite plan)")
        out.append("")

    return "\n".join(out)


def _bottleneck_label(bn: str) -> str:
    return {
        "launch_sparse":       "Python/launch tax dominates",
        "tc_underutil":        "Tensor Core not emitted",
        "epilogue_fma_bound":  "Dequant FMA epilogue tail",
        "x_zero_anomaly":      "Data-dependent slowdown (1 shape)",
        "physics_loss":        "W4A4 roof below FP16 roof",
    }.get(bn, bn)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--targets",
        type=Path,
        default=P2_DIR / "shape_targets.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROADMAP_PATH,
    )
    args = ap.parse_args()
    if not args.targets.is_file():
        raise SystemExit(
            f"[phase3] missing input: {args.targets} — run cluster_all_shapes first"
        )
    recs = _load_targets_csv(args.targets)
    text = render(recs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[phase3] wrote {args.out}  ({len(text)} chars, {len(recs)} shapes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
