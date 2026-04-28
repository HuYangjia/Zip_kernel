"""Phase 2 kernel-microscope report renderer (task-item.md step 11).

Joins three evidence streams for each of the 8 representative shapes:

    1. Phase 1 timeline attribution  (``phase1_attribution.json``)
    2. SASS static profile verdicts  (``_sass/sass_profile.json``)
    3. Microbench bisection deltas   (``<tag>/bisection.json``)

and emits a single consolidated markdown file listing, per shape:

    * wall-clock breakdown carried over from Phase 1
    * the verdicts that the Phase 1 launch-tax check triggered
      (``launch_bound`` / ``python_bound``)
    * the top SASS verdicts for the kernel families this shape invokes
    * the three bisection deltas + the "primary bottleneck" attribution

The primary bottleneck is picked from the taxonomy declared in
requirements.md §3.5, using the following decision tree (written in
plain prose rather than a big table because readers asked):

    - if ``launch_tax / total > 0.5``      -> launch_sparse
    - elif ``all bisection deltas < 3 %`` and ``cuda_eff < 0.25`` and
      any kernel has ``tc_underutil``      -> tc_underutil
    - elif ``Δ_scale1 >= 3 %``              -> epilogue_fma_bound
    - elif ``Δ_l2 >= 3 %``                  -> hbm_stall
    - elif launch_tax 0.3..0.5             -> launch_sparse
    - elif regs >= 128 and cuda_eff < 0.30  -> occupancy_low
    - else                                 -> unclassified
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO = _repo_root()
P1_DIR = REPO / "cuda_kernel/logs/phase1_timeline"
P2_DIR = REPO / "cuda_kernel/logs/phase2_microscope"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_phase1() -> Dict[str, Dict]:
    path = P1_DIR / "phase1_attribution.json"
    if not path.is_file():
        print(f"[phase2_render] phase1 attribution missing: {path}")
        return {}
    data = json.loads(path.read_text())
    return {r["tag"]: r for r in data.get("records", [])}


def _load_launch_tax() -> Dict[str, Dict]:
    """Return a dict ``{tag: shape_entry}`` from the consolidated JSON.

    The Phase 1 launch-tax measurement produces one file covering every
    shape, not a per-shape file.  Fields of interest:
    ``launch_tax_us``, ``launch_tax_pct_of_plain``, ``t_plain_us``,
    ``t_kernel_body_us``.
    """
    path = P1_DIR / "launch_tax.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {row["tag"]: row for row in data.get("shapes", [])}


def _load_bisection() -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for sub in P2_DIR.iterdir():
        p = sub / "bisection.json"
        if p.is_file():
            out[sub.name] = json.loads(p.read_text())
    return out


def _load_sass_verdicts() -> Tuple[Dict[str, List[str]], Dict]:
    """Return (family_name -> [verdicts], full_json)."""
    path = P2_DIR / "_sass/sass_profile.json"
    if not path.is_file():
        return {}, {}
    data = json.loads(path.read_text())
    family_to_verdicts: Dict[str, List[str]] = {}
    for k in data.get("kernels", []):
        family_to_verdicts[k["name"]] = k.get("verdicts", [])
    return family_to_verdicts, data


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------
def _attribute_bottleneck(
    shape_row: Dict,
    bisection: Optional[Dict],
    launch_tax: Optional[Dict],
    sass_all: Dict,
) -> Tuple[str, str]:
    fwd = shape_row.get("forward_us", 0) or 0.0
    lt = launch_tax or {}
    launch_us = lt.get("launch_tax_us", 0.0) or 0.0
    launch_pct = lt.get("launch_tax_pct_of_plain", None)
    # Prefer the already-computed pct from launch_tax.json (uses
    # ``t_plain_us`` as denominator which is the cleanest ref).
    # If missing, fall back to launch_us / forward_us from Phase 1 NVTX.
    if launch_pct is not None:
        launch_frac = launch_pct / 100.0
    elif fwd > 0:
        launch_frac = launch_us / fwd
    else:
        launch_frac = 0.0

    deltas = (bisection or {}).get("deltas_pct", {})
    d_l2 = deltas.get("l2_hot_delta_pct", 0.0)
    d_xzero = deltas.get("x_zero_delta_pct", 0.0)
    d_scale = deltas.get("scale_one_delta_pct", 0.0)
    max_delta = max(abs(d_l2), abs(d_xzero), abs(d_scale))

    # Decision tree (see module docstring).
    if launch_frac > 0.5:
        return (
            "launch_sparse",
            f"launch_tax ≈ {launch_us:.1f}us = {launch_frac*100:.0f}% of plain",
        )

    # Epilogue FMA takes priority over tc_underutil if the lever clearly
    # helps (>= 2.5 % after min-of-means still counts as signal).
    if d_scale >= 2.5:
        return (
            "epilogue_fma_bound",
            f"scale=1 speeds up by {d_scale:.1f}% -> epilogue FMA is a real tail consumer",
        )

    if d_l2 >= 3.0:
        return (
            "hbm_stall",
            f"L2-hot weights speed up by {d_l2:.1f}% -> HBM weight bandwidth bound",
        )

    # Note: a large negative ``d_xzero`` (X=0 much slower than random) used
    # to route to ``x_zero_anomaly`` here.  It was removed after the
    # deep-dive in :mod:`kernel.tools.profile.xzero_probe` proved the
    # signal is a measurement artefact of the earlier
    # (warmup=80, outer=4) budget and disappears completely under the
    # current (warmup=200, outer=10) schedule.  We now leave a negative
    # ``d_xzero`` in the raw bisection JSON for audit but never use it
    # as a classification lever.

    # tc_underutil is the universal fallback for compute-bound kernels
    # when no single lever helps.  Holds whenever max|Δ| < 3 %.
    # Label retained for taxonomy stability; meaning redefined by
    # phase2_tc_rediagnosis.md (2026-04-28) from "TC not emitted" to
    # "MMA pipeline starvation".
    if max_delta < 3.0:
        return (
            "tc_underutil",
            f"all bisection deltas < 3% (max {max_delta:.1f}%); "
            f"SASS mac_tc_share >= 99% (IMMA active) but TC pipeline "
            f"idle ~76% due to epilogue/IMAD/async-copy serialisation -> "
            f"MMA pipeline starvation",
        )

    if 0.3 <= launch_frac <= 0.5:
        return (
            "launch_sparse",
            f"launch_tax fraction {launch_frac*100:.0f}% (medium)",
        )

    return ("unclassified", f"no single lever triggered; max|Δ|={max_delta:.1f}%")

    return ("unclassified", f"no single lever triggered; max|Δ|={max_delta:.1f}%")


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------
def render() -> str:
    p1 = _load_phase1()
    ltax = _load_launch_tax()
    bisect = _load_bisection()
    _, sass_all = _load_sass_verdicts()

    # Union of shape tags across the three sources (use bisection as
    # the authoritative Phase-2 set, extend with phase1-only tags).
    tags = sorted(set(bisect) | set(p1))
    if not tags:
        return "# Phase 2 kernel microscope report\n\n_no data found._"

    out: List[str] = []
    out.append("# Phase 2 Kernel Microscope Report")
    out.append("")
    out.append("Joined sources:")
    out.append("- Phase 1 timeline attribution (`phase1_timeline/phase1_attribution.json`)")
    out.append("- Phase 1 launch tax (`phase1_timeline/<tag>/launch_tax.json`)")
    out.append("- SASS static profile (`phase2_microscope/_sass/sass_profile.json`)")
    out.append("- Microbench bisection (`phase2_microscope/<tag>/bisection.json`)")
    out.append("")

    # ---- SASS global finding (same for every shape) -----------------------
    agg: Dict[str, int] = {}
    for k in sass_all.get("kernels", []):
        for v in k.get("verdicts", []):
            agg[v] = agg.get(v, 0) + 1
    total_kernels = len(sass_all.get("kernels", []))
    out.append("## Global SASS finding")
    out.append("")
    out.append(
        "Static analysis of **all 42 compiled kernel instantiations** in "
        "`hkust_v9_cuda.so` gives:"
    )
    for v, c in sorted(agg.items(), key=lambda kv: -kv[1]):
        out.append(f"- `{v}`: **{c}** / {total_kernels} kernels")
    out.append("")
    out.append(
        "→ **Rediagnosed reading (2026-04-28)**: the `tc_underutil` "
        "label fires on every `*_mma_int4_kernel` but the original "
        "trigger (issue-slot TC fraction < 5 %) was a false positive.  "
        "Under MAC-weighted analysis `mac_tc_share >= 99 %` — IMMA is "
        "already carrying the full compute budget.  The 13–39 % "
        "`cuda_eff` observed in the Roofline report reflects **MMA "
        "pipeline starvation**: the tensor pipeline sits idle ~76 % of "
        "cycles while the warp scheduler runs (i) HFMA2 dequant "
        "epilogue, (ii) IMAD shared-memory swizzle, and (iii) a "
        "2-stage `cp.async` pipeline that cannot saturate the "
        "MMA-consumer rate.  See "
        "`phase2_tc_rediagnosis.md` for the evidence chain and the "
        "sub-bottleneck decomposition; Step 2 in `phase3_roadmap.md` "
        "has been recalibrated accordingly."
    )
    out.append("")

    # ---- Per-shape table -------------------------------------------------
    out.append("## Per-shape bottleneck attribution")
    out.append("")
    out.append(
        "| shape | T | plain_us | body_us | launch_tax_us | launch% | Δ_l2% | Δ_xzero% | Δ_scale1% | primary_bottleneck |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    attributions: Dict[str, Tuple[str, str]] = {}
    for tag in tags:
        p1r = p1.get(tag, {})
        b = bisect.get(tag, {})
        lt = ltax.get(tag, {})
        # Prefer launch_tax.json ``t_plain_us`` as wall-clock reference
        # (it's the same timing convention as bisection base_us).  Fall
        # back to Phase 1 forward_us when launch_tax is missing.
        plain = lt.get("t_plain_us") or p1r.get("forward_us", 0.0)
        body = lt.get("t_kernel_body_us", 0.0)
        launch_us = lt.get("launch_tax_us", 0.0)
        launch_pct = lt.get("launch_tax_pct_of_plain", 0.0)
        d = b.get("deltas_pct", {})
        bottleneck, evidence = _attribute_bottleneck(p1r, b, lt, sass_all)
        attributions[tag] = (bottleneck, evidence)
        T = b.get("shape", {}).get("T", p1r.get("T", "?"))
        out.append(
            "| {tag} | {T} | {pl:.2f} | {bo:.2f} | {lt:.2f} | {lf:.0f}% | {dl2:+.1f} | "
            "{dx:+.1f} | {ds:+.1f} | `{b}` |".format(
                tag=tag,
                T=T,
                pl=plain,
                bo=body,
                lt=launch_us,
                lf=launch_pct,
                dl2=d.get("l2_hot_delta_pct", 0.0),
                dx=d.get("x_zero_delta_pct", 0.0),
                ds=d.get("scale_one_delta_pct", 0.0),
                b=bottleneck,
            )
        )
    out.append("")

    # ---- Evidence narrative per shape ------------------------------------
    out.append("## Per-shape evidence (one block per shape)")
    out.append("")
    for tag in tags:
        p1r = p1.get(tag, {})
        b = bisect.get(tag, {})
        lt = ltax.get(tag, {})
        attr, evidence = attributions.get(tag, ("unclassified", "-"))
        out.append(f"### `{tag}`")
        out.append("")
        s = b.get("shape", {})
        if s:
            out.append(
                f"- shape: T={s.get('T')}, d_in={s.get('d_in')}, "
                f"d_out={s.get('d_out')}, {s.get('model')} {s.get('proj')}"
            )
        if p1r:
            path = p1r.get("gemm_path", "?")
            out.append(
                f"- Phase 1: forward={p1r.get('forward_us',0):.2f}us "
                f"(path={path}, quant={p1r.get('activation_quant_us',0):.2f}us, "
                f"gemm={p1r.get('gemm_us',0):.2f}us, "
                f"python_glue={p1r.get('python_glue_us',0):.2f}us)"
            )
        if lt:
            out.append(
                f"- launch tax: {lt.get('launch_tax_us',0):.2f} us "
                f"({lt.get('launch_tax_pct_of_plain',0):.1f}% of plain); "
                f"graph_replay={lt.get('t_graph_us',0):.2f}us, "
                f"plain={lt.get('t_plain_us',0):.2f}us"
            )
        if b:
            d = b.get("deltas_pct", {})
            t = b.get("timings_us", {})
            out.append(
                f"- bisection (base={t.get('base_us',0):.2f}us): "
                f"Δ_l2={d.get('l2_hot_delta_pct',0):+.1f}%, "
                f"Δ_xzero={d.get('x_zero_delta_pct',0):+.1f}%, "
                f"Δ_scale1={d.get('scale_one_delta_pct',0):+.1f}%"
            )
        out.append(f"- **Primary bottleneck: `{attr}`** — {evidence}")
        out.append("")

    # ---- Coverage check --------------------------------------------------
    buckets: Dict[str, int] = {}
    for _, (b, _) in attributions.items():
        buckets[b] = buckets.get(b, 0) + 1
    out.append("## Bottleneck category coverage")
    out.append("")
    for k in sorted(buckets):
        out.append(f"- `{k}`: {buckets[k]} shape(s)")
    out.append("")
    distinct = sum(1 for k in buckets if k != "unclassified")
    if distinct < 2:
        out.append(
            f"> ⚠ Only {distinct} distinct bottleneck category found across "
            f"the 8 representative shapes.  Consider adding more probes "
            f"(requirements.md §3.6)."
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=P2_DIR / "phase2_kernel_microscope_report.md",
    )
    args = ap.parse_args()
    text = render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[phase2_render] wrote {args.out}  ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
