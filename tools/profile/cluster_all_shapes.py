"""Propagate Phase 2 bottleneck labels to all 100 shapes + compute targets.

.. note::
   **Rediagnosis 2026-04-28** — the ``tc_underutil`` label produced by
   this pipeline retains its name for downstream taxonomy stability
   but its *meaning* has been corrected from "Tensor Core not emitted"
   to **"MMA pipeline starvation"**.  See
   ``cuda_kernel/logs/phase2_microscope/phase2_tc_rediagnosis.md``
   for the full evidence chain (MAC-weighted SASS share, sub-bottleneck
   decomposition, and recalibrated Step 2 expected gain).  All
   ``POST_AUDIT_OVERRIDES`` entries and all 84 cluster members under
   ``tc_underutil`` now mean *pipeline starvation*, not *no TC*.

Implements task-item.md steps 12 & 13:

Step 12 — cluster every shape in ``roofline_compare.csv`` (100 rows) by
nearest-neighbour match against the 8 Phase 2 representatives, inheriting
their ``primary_bottleneck`` label.  The feature vector is::

    (T_log, d_in_log, d_out_log, cuda_eff, roof_ratio)

where ``roof_ratio = cuda_roof_us / fp16_roof_us`` (always ``< 1`` by
requirement §1.3 if ``cuda_roof`` actually beats fp16_roof; otherwise the
shape is flagged ``physics_loss``).  Features are z-score normalised
against the 8 representatives' distribution so no single column dominates
the Euclidean distance.  Output: ``shape_clusters.csv``.

Step 13 — for every shape compute::

    target_eff = min(0.80, fp16_efficiency * 0.90)
    status in {
        "physics_loss",  # cuda_roof_us > fp16_roof_us
        "on_track",      # cuda_eff >= target_eff
        "gap_small",     # 0 <= target_eff - cuda_eff < 0.20
        "gap_large",     # target_eff - cuda_eff >= 0.20
    }

Output: ``shape_targets.csv`` (also prints the 4-bucket summary to stdout).

Both CSVs are written side-by-side in
``cuda_kernel/logs/phase2_microscope/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
ROOFLINE_CSV = REPO / "cuda_kernel/logs/qwen3_20260428_111515/roofline_compare.csv"
P2_DIR = REPO / "cuda_kernel/logs/phase2_microscope"
P2_REPORT_JSON = P2_DIR / "phase2_kernel_microscope_report.md"  # exists, not used directly
BISECT_DIR = P2_DIR  # each <tag>/bisection.json


# ---------------------------------------------------------------------------
# Post-audit cluster overrides
# ---------------------------------------------------------------------------
# Each entry explicitly reclassifies a single shape whose
# nearest-neighbour assignment was later invalidated by a targeted
# audit (under the tightened (warmup=200, outer=10) x K>=5 schedule).
# Keeping these as a data table — rather than post-hoc CSV edits —
# lets the pipeline stay idempotent: re-running this script always
# reproduces the exact same clusters/targets on disk.
#
# Keys are ``(model, proj, T, d_in, d_out)``.
#
# ``nearest_rep_tag`` is the rep whose label the overridden shape
# should now inherit.  ``cluster_id`` is resolved automatically from
# the new bottleneck label and the cluster_id assignment order.
#
# ``audit_ref`` is a short human-readable pointer to the report that
# justifies the override, written out as a new CSV column for
# traceability.
POST_AUDIT_OVERRIDES: Dict[Tuple[str, str, int, int, int], Dict[str, str]] = {
    # Qwen3-4B gate_up_proj T=1 2560->19456
    # Phase-2 NN classifier put this under ``launch_sparse`` (nearest
    # rep decode_T1_q_2048_2048), but the launch_sparse cluster audit
    # (cuda_kernel/logs/phase2_microscope/audit_launch_tax/
    # launch_tax_audit.md) measured median launch_tax_pct = 1.8%
    # [1.6, 2.1] under (warmup=200, outer=10, inner=100) x K=5 trials.
    # The kernel body (~47.6us) dominates entirely -- this is a
    # tc_underutil shape, not a launch-bound one.
    ("Qwen3-4B", "gate_up_proj", 1, 2560, 19456): {
        "bottleneck": "tc_underutil",
        "nearest_rep_tag": "large_T1024_gu_4096_24576",
        "audit_ref": "audit_launch_tax/launch_tax_audit.md",
    },
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class ShapeRow:
    model: str
    proj: str
    T: int
    d_in: int
    d_out: int
    fp16_us: float
    fp16_eff: float
    cuda_us: float
    cuda_eff: float
    cuda_roof_us: float
    fp16_roof_us: float
    roof_ratio: float  # cuda_roof / fp16_roof

    @property
    def shape_key(self) -> str:
        return f"{self.T}_{self.d_in}_{self.d_out}"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_roofline(path: Path) -> List[ShapeRow]:
    rows: List[ShapeRow] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ShapeRow(
                    model=r["model"],
                    proj=r["proj"],
                    T=int(r["T"]),
                    d_in=int(r["d_in"]),
                    d_out=int(r["d_out"]),
                    fp16_us=float(r["fp16_us"]),
                    fp16_eff=float(r["fp16_efficiency"]),
                    cuda_us=float(r["cuda_us"]),
                    cuda_eff=float(r["cuda_efficiency"]),
                    cuda_roof_us=float(r["cuda_roof_us"]),
                    fp16_roof_us=float(r["fp16_roof_us"]),
                    roof_ratio=float(r["cuda_vs_fp16_roofline"]),
                )
            )
    return rows


def _load_representatives() -> Dict[str, str]:
    """Return ``{tag: primary_bottleneck}`` by re-running the Phase 2 decision.

    We reuse :mod:`kernel.tools.profile.phase2_render_report`'s loaders and
    classifier directly so the attribution stays in sync with the report.
    """
    # Late import to avoid a cycle if this module is ever itself imported by
    # the renderer.  Use the local module path so the script works both
    # when launched via ``python -m tools.profile.cluster_all_shapes`` from
    # the ``kernel/`` directory and via ``python -m kernel.tools.profile...``
    # when ``kernel`` is on ``sys.path`` (e.g. on the AutoDL box).
    try:
        from kernel.tools.profile.phase2_render_report import (  # noqa: WPS433
            _attribute_bottleneck,
            _load_bisection,
            _load_launch_tax,
            _load_phase1,
            _load_sass_verdicts,
        )
    except ModuleNotFoundError:
        from tools.profile.phase2_render_report import (  # type: ignore[no-redef]  # noqa: WPS433
            _attribute_bottleneck,
            _load_bisection,
            _load_launch_tax,
            _load_phase1,
            _load_sass_verdicts,
        )

    p1 = _load_phase1()
    ltax = _load_launch_tax()
    bisect = _load_bisection()
    _, sass_all = _load_sass_verdicts()

    out: Dict[str, str] = {}
    # Representatives are exactly the 8 shapes that have a bisection.json.
    for tag, b in bisect.items():
        bottleneck, _evidence = _attribute_bottleneck(
            p1.get(tag, {}), b, ltax.get(tag, {}), sass_all
        )
        out[tag] = bottleneck
    return out


def _representative_rows(reps: Dict[str, str], all_rows: List[ShapeRow]) -> List[Tuple[ShapeRow, str]]:
    """Resolve each ``tag`` back to its ShapeRow via the bisection JSON.

    Using :mod:`_phase1_shapes` would be cleaner, but that module needs
    torch + the custom CUDA extension to even import, which is not
    available on the host that renders this report.  The bisection JSON
    already stores ``T / d_in / d_out / model / proj``, so we key off
    that directly.
    """
    # Load each rep's shape metadata from its bisection.json.
    rep_meta: Dict[str, Dict[str, object]] = {}
    for tag in reps:
        p = BISECT_DIR / tag / "bisection.json"
        if p.is_file():
            rep_meta[tag] = json.loads(p.read_text()).get("shape", {})

    by_key = {f"{r.T}_{r.d_in}_{r.d_out}_{r.model}_{r.proj}": r for r in all_rows}
    resolved: List[Tuple[ShapeRow, str]] = []
    for tag, bottleneck in reps.items():
        s = rep_meta.get(tag)
        if not s:
            print(f"[cluster] WARNING: no bisection.json shape for tag {tag!r}", file=sys.stderr)
            continue
        key = f"{s.get('T')}_{s.get('d_in')}_{s.get('d_out')}_{s.get('model')}_{s.get('proj')}"
        row = by_key.get(key)
        if row is None:
            print(
                f"[cluster] WARNING: no roofline row for tag {tag!r} "
                f"(T={s.get('T')}, d_in={s.get('d_in')}, d_out={s.get('d_out')}, "
                f"model={s.get('model')})",
                file=sys.stderr,
            )
            continue
        resolved.append((row, bottleneck))
    return resolved


# ---------------------------------------------------------------------------
# Nearest-neighbour cluster assignment
# ---------------------------------------------------------------------------
def _feature_vec(r: ShapeRow) -> List[float]:
    # log-domain for T/d_in/d_out to compress the 1..1024 axis.
    return [
        math.log1p(r.T),
        math.log1p(r.d_in),
        math.log1p(r.d_out),
        r.cuda_eff,
        r.roof_ratio,
    ]


def _zscore_params(vecs: List[List[float]]) -> Tuple[List[float], List[float]]:
    d = len(vecs[0])
    means = [statistics.fmean(v[j] for v in vecs) for j in range(d)]
    stds = []
    for j in range(d):
        col = [v[j] for v in vecs]
        stds.append(statistics.pstdev(col) or 1.0)
    return means, stds


def _apply_zscore(v: List[float], means: List[float], stds: List[float]) -> List[float]:
    return [(v[j] - means[j]) / stds[j] for j in range(len(v))]


def _euclid(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _cluster_shapes(
    all_rows: List[ShapeRow],
    reps: List[Tuple[ShapeRow, str]],
) -> List[Dict[str, object]]:
    """Return ``[{row, primary_bottleneck, cluster_id, nearest_rep_tag}]``.

    Historical note: an earlier version of this function carved out an
    ``x_zero_anomaly`` bucket for the single ``mid_T128_kv_2560_2048``
    shape whose D.2 (X=0) bisection was -27.9% vs random.  The
    deep-dive in :mod:`kernel.tools.profile.xzero_probe` proved the
    signal was a measurement artefact of the earlier (warmup=80,
    outer=4) budget and vanishes under (warmup=200, outer=10).  We
    therefore no longer special-case that shape — it falls through
    to its real nearest-neighbour cluster like every other shape.
    The attribution code in :func:`phase2_render_report._attribute_bottleneck`
    was updated in lock-step so no rep is ever labelled
    ``x_zero_anomaly`` again.
    """
    # Defensive filter: if any rep still carries a stale anomaly label
    # (e.g. an outdated bisection.json on disk), drop it from the pool
    # so the taxonomy stays {hbm_stall, tc_underutil, epilogue_fma_bound,
    # launch_sparse}.  After the attribution update this is an invariant,
    # but keeping the filter makes the pipeline idempotent during the
    # transition.
    candidates = [(r, bn) for r, bn in reps if bn != "x_zero_anomaly"]
    if not candidates:
        raise RuntimeError("no non-anomaly representatives left after filtering")

    # Build feature matrix over the candidate reps -> derive z-score scale.
    rep_feats = [_feature_vec(r) for r, _ in candidates]
    means, stds = _zscore_params(rep_feats)
    rep_feats_z = [_apply_zscore(v, means, stds) for v in rep_feats]

    # Stable cluster_id per unique bottleneck, in the order first seen.
    seen: Dict[str, int] = {}
    for _, bn in candidates:
        seen.setdefault(bn, len(seen))

    # For each of the 100 shapes, compute nearest rep.
    out: List[Dict[str, object]] = []
    for row in all_rows:
        if row.cuda_roof_us > row.fp16_roof_us:
            out.append({
                "row": row,
                "primary_bottleneck": "physics_loss",
                "cluster_id": -1,
                "nearest_rep_tag": None,
                "distance": 0.0,
            })
            continue
        v = _apply_zscore(_feature_vec(row), means, stds)
        best_d = float("inf")
        best_i = 0
        for i, rv in enumerate(rep_feats_z):
            d = _euclid(v, rv)
            if d < best_d:
                best_d = d
                best_i = i
        rep_row, rep_bn = candidates[best_i]
        # Tag for the representative — re-derive from bisection JSON meta.
        rep_tag = None
        for tag in sorted((BISECT_DIR).iterdir()):
            p = tag / "bisection.json"
            if not p.is_file():
                continue
            meta = json.loads(p.read_text()).get("shape", {})
            if (meta.get("T"), meta.get("d_in"), meta.get("d_out"),
                meta.get("model"), meta.get("proj")) == \
               (rep_row.T, rep_row.d_in, rep_row.d_out, rep_row.model, rep_row.proj):
                rep_tag = tag.name
                break
        out.append({
            "row": row,
            "primary_bottleneck": rep_bn,
            "cluster_id": seen[rep_bn],
            "nearest_rep_tag": rep_tag,
            "distance": best_d,
            "audit_ref": "",
        })

    # ------------------------------------------------------------------
    # Apply POST_AUDIT_OVERRIDES
    # ------------------------------------------------------------------
    # Build a tag->rep_row lookup so we can redirect both the label
    # AND the nearest_rep_tag at once.
    tag_to_rep: Dict[str, ShapeRow] = {}
    for tag in sorted(BISECT_DIR.iterdir()):
        p = tag / "bisection.json"
        if not p.is_file():
            continue
        meta = json.loads(p.read_text()).get("shape", {})
        for rep_row, _bn in candidates:
            if (meta.get("T"), meta.get("d_in"), meta.get("d_out"),
                meta.get("model"), meta.get("proj")) == \
               (rep_row.T, rep_row.d_in, rep_row.d_out, rep_row.model, rep_row.proj):
                tag_to_rep[tag.name] = rep_row
                break

    n_overrides_applied = 0
    for rec in out:
        row = rec["row"]  # type: ignore[assignment]
        key = (row.model, row.proj, row.T, row.d_in, row.d_out)
        ovr = POST_AUDIT_OVERRIDES.get(key)
        if not ovr:
            continue
        new_bn = ovr["bottleneck"]
        new_tag = ovr["nearest_rep_tag"]
        if new_bn not in seen:
            raise RuntimeError(
                f"POST_AUDIT_OVERRIDES for {key} targets unknown cluster {new_bn!r}; "
                f"known clusters: {sorted(seen.keys())}"
            )
        if new_tag not in tag_to_rep:
            raise RuntimeError(
                f"POST_AUDIT_OVERRIDES for {key} targets unknown rep tag {new_tag!r}; "
                f"known rep tags: {sorted(tag_to_rep.keys())}"
            )
        # Recompute feature distance against the named rep so the
        # ``feature_distance`` column stays meaningful.
        v = _apply_zscore(_feature_vec(row), means, stds)
        target_rv = _apply_zscore(_feature_vec(tag_to_rep[new_tag]), means, stds)
        rec["primary_bottleneck"] = new_bn
        rec["cluster_id"] = seen[new_bn]
        rec["nearest_rep_tag"] = new_tag
        rec["distance"] = _euclid(v, target_rv)
        rec["audit_ref"] = ovr.get("audit_ref", "override")
        n_overrides_applied += 1

    if n_overrides_applied:
        print(
            f"[cluster] applied {n_overrides_applied} post-audit "
            f"override(s) from POST_AUDIT_OVERRIDES"
        )
    return out


# ---------------------------------------------------------------------------
# Target status (step 13)
# ---------------------------------------------------------------------------
def _status_for(row: ShapeRow, bottleneck: str) -> Tuple[float, str]:
    """Return ``(target_eff, status)`` per requirements.md §1."""
    if bottleneck == "physics_loss":
        return (0.0, "physics_loss")
    target = min(0.80, row.fp16_eff * 0.90)
    gap = target - row.cuda_eff
    if gap <= 0:
        return (target, "on_track")
    if gap < 0.20:
        return (target, "gap_small")
    return (target, "gap_large")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def _write_clusters_csv(
    path: Path,
    records: List[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "proj", "T", "d_in", "d_out",
            "cuda_eff", "fp16_eff", "roof_ratio",
            "primary_bottleneck", "cluster_id",
            "nearest_rep_tag", "feature_distance",
            "audit_ref",
        ])
        for rec in records:
            row: ShapeRow = rec["row"]  # type: ignore[assignment]
            w.writerow([
                row.model, row.proj, row.T, row.d_in, row.d_out,
                f"{row.cuda_eff:.4f}", f"{row.fp16_eff:.4f}", f"{row.roof_ratio:.4f}",
                rec["primary_bottleneck"], rec["cluster_id"],
                rec.get("nearest_rep_tag") or "-",
                f"{rec['distance']:.4f}",
                rec.get("audit_ref") or "",
            ])


def _write_targets_csv(
    path: Path,
    records: List[Dict[str, object]],
    summary_counts: Dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(
        "# target_eff = min(0.80, fp16_efficiency * 0.90)  (requirements.md §1.1)"
    )
    lines.append(
        "# status buckets: on_track (cuda_eff >= target), gap_small (<20pp), gap_large (>=20pp), physics_loss"
    )
    lines.append("# Summary: " + ", ".join(
        f"{k}={v}" for k, v in sorted(summary_counts.items())
    ))
    header = [
        "model", "proj", "T", "d_in", "d_out",
        "cuda_eff", "fp16_eff", "target_eff", "gap_pp",
        "status", "primary_bottleneck", "nearest_rep_tag",
        "audit_ref",
    ]
    lines.append(",".join(header))
    for rec in records:
        row: ShapeRow = rec["row"]  # type: ignore[assignment]
        target, status = _status_for(row, rec["primary_bottleneck"])  # type: ignore[arg-type]
        gap_pp = (target - row.cuda_eff) * 100.0 if status != "physics_loss" else 0.0
        lines.append(
            ",".join([
                row.model, row.proj, str(row.T), str(row.d_in), str(row.d_out),
                f"{row.cuda_eff:.4f}", f"{row.fp16_eff:.4f}",
                f"{target:.4f}", f"{gap_pp:+.2f}",
                status, str(rec["primary_bottleneck"]),
                str(rec.get("nearest_rep_tag") or "-"),
                str(rec.get("audit_ref") or ""),
            ])
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--roofline",
        type=Path,
        default=ROOFLINE_CSV,
        help="Input roofline_compare.csv (100 shapes).",
    )
    ap.add_argument("--out-dir", type=Path, default=P2_DIR)
    args = ap.parse_args()

    if not args.roofline.is_file():
        print(f"[cluster] ERROR: roofline CSV missing: {args.roofline}", file=sys.stderr)
        return 2

    rows = _load_roofline(args.roofline)
    print(f"[cluster] loaded {len(rows)} shapes from {args.roofline.name}")

    rep_map = _load_representatives()
    print(f"[cluster] representatives: {len(rep_map)} with bottleneck labels")
    for tag, bn in sorted(rep_map.items()):
        print(f"          {tag:40s} -> {bn}")

    reps = _representative_rows(rep_map, rows)
    if len(reps) < 3:
        print(
            f"[cluster] ERROR: only {len(reps)} representatives resolved; "
            "cannot build a meaningful NN classifier.",
            file=sys.stderr,
        )
        return 2

    records = _cluster_shapes(rows, reps)

    # Status summary for step 13.
    status_counts: Dict[str, int] = {}
    bn_counts: Dict[str, int] = {}
    for rec in records:
        row: ShapeRow = rec["row"]  # type: ignore[assignment]
        _tgt, status = _status_for(row, rec["primary_bottleneck"])  # type: ignore[arg-type]
        status_counts[status] = status_counts.get(status, 0) + 1
        bn = str(rec["primary_bottleneck"])
        bn_counts[bn] = bn_counts.get(bn, 0) + 1

    clusters_csv = args.out_dir / "shape_clusters.csv"
    targets_csv = args.out_dir / "shape_targets.csv"
    _write_clusters_csv(clusters_csv, records)
    _write_targets_csv(targets_csv, records, status_counts)

    # Also a small JSON summary for downstream renderers.
    summary_json = args.out_dir / "cluster_summary.json"
    summary_json.write_text(json.dumps({
        "total": len(rows),
        "representatives": rep_map,
        "bottleneck_counts": bn_counts,
        "status_counts": status_counts,
    }, indent=2))

    print()
    print("[cluster] bottleneck coverage:")
    for k, v in sorted(bn_counts.items(), key=lambda kv: -kv[1]):
        print(f"          {k:24s}: {v} shapes")
    print()
    print("[cluster] target status:")
    for k, v in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"          {k:14s}: {v}")
    print()
    print(f"[cluster] wrote {clusters_csv}")
    print(f"[cluster] wrote {targets_csv}")
    print(f"[cluster] wrote {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
