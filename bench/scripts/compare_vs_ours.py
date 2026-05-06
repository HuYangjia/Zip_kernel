"""3-way comparison: BF16 (full layer) vs Atom baseline vs YOUR W4A4 kernel.

Inputs
------
* ``--bf16-json``     bench_bf16_per_op.json (already produced)
* ``--baseline-json`` baseline_layer.json    (from compose_baseline_layer.py)
* ``--ours-json``     ours_layer.json        (your r79 layer composition; same
                                              schema as baseline_layer.json)

The schema requirement for ``--ours-json`` is intentionally minimal: each
record must carry at least

    {model, phase, batch, baseline_layer_us}   # 'baseline_layer_us' is
                                               # the misleadingly-named
                                               # composed total — we treat
                                               # it as 'your composed total'.

i.e. you can produce ``ours_layer.json`` by running compose_baseline_layer
with ``--atom-json`` swapped for your kernel's per-GEMM JSON (same shape
contract).

Output
------
* ``comparison_table.md``  — markdown table:
       | model | phase | B | bf16_full | atom_layer | ours_layer |
       | speedup_atom_vs_bf16 | speedup_ours_vs_bf16 | speedup_ours_vs_atom |
       | atom_amdahl_ceil | ours_amdahl_ceil | verdict |

Verdict rules:
  * ``ours < atom``  → "✅ ours wins"
  * ``ours within +5% of atom`` → "≈ tie"
  * ``ours > atom by >5%`` → "❌ ours loses"
  * any input missing → "—"

We also emit ``comparison_table.json`` so downstream plots can pick it up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _key(rec: dict) -> tuple[str, str, int]:
    return (rec["model"], rec["phase"], rec["batch"])


def _amdahl_ceil(layer_us: float, replaced_us: float) -> float:
    """Return the layer speedup ceiling if the replaced region took 0 us."""
    if layer_us <= 0:
        return float("nan")
    p = replaced_us / layer_us
    if not (0 < p < 1):
        return float("inf") if p >= 1 else 1.0
    return 1.0 / (1.0 - p)


def _verdict(ours_us: float, atom_us: float) -> str:
    if not (ours_us > 0 and atom_us > 0):
        return "—"
    rel = ours_us / atom_us
    if rel < 0.95:
        return "✅ ours wins"
    if rel <= 1.05:
        return "≈ tie"
    return "❌ ours loses"


def _bf16_full_us(rec: dict) -> float | None:
    s = rec.get("sanity") or {}
    return s.get("full_layer_us")


def _bf16_replaced_region_us(rec: dict) -> float:
    """In the BF16 record: cost of the 7 GEMMs + 2 RMSNorms (= what gets replaced)."""
    s = rec.get("sanity") or {}
    gemm = s.get("gemm_sum_us", 0.0)
    rms_in = (rec.get("ops") or {}).get("input_rmsnorm", {}).get("median_us", 0.0)
    rms_post = (rec.get("ops") or {}).get("post_rmsnorm", {}).get("median_us", 0.0)
    return gemm + rms_in + rms_post


def main() -> int:
    p = argparse.ArgumentParser(description="3-way comparison: BF16 vs Atom vs Ours")
    p.add_argument("--bf16-json", required=True, type=Path)
    p.add_argument("--baseline-json", required=True, type=Path,
                   help="from compose_baseline_layer.py (Atom side)")
    p.add_argument("--ours-json", required=True, type=Path,
                   help="same schema as baseline-json, but holding your kernel's numbers")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    bf16 = json.loads(args.bf16_json.read_text())
    base = json.loads(args.baseline_json.read_text())
    ours = json.loads(args.ours_json.read_text())

    bf16_by_key = {_key(r): r for r in bf16["records"]}
    base_by_key = {_key(r): r for r in base["records"]}
    ours_by_key = {_key(r): r for r in ours["records"]}

    rows: list[dict] = []
    for k in sorted(bf16_by_key.keys(), key=lambda t: (t[0], t[1], t[2])):
        bf16_rec = bf16_by_key[k]
        full_us = _bf16_full_us(bf16_rec)
        replaced_us = _bf16_replaced_region_us(bf16_rec)
        atom_rec = base_by_key.get(k)
        ours_rec = ours_by_key.get(k)
        atom_us = (atom_rec or {}).get("baseline_layer_us")
        ours_us = (ours_rec or {}).get("baseline_layer_us")

        row = {
            "model": k[0], "phase": k[1], "batch": k[2],
            "bf16_full_us": full_us,
            "atom_layer_us": atom_us,
            "ours_layer_us": ours_us,
            "speedup_atom_vs_bf16": (full_us / atom_us) if (full_us and atom_us) else None,
            "speedup_ours_vs_bf16": (full_us / ours_us) if (full_us and ours_us) else None,
            "speedup_ours_vs_atom": (atom_us / ours_us) if (atom_us and ours_us) else None,
            "amdahl_ceil_bf16": _amdahl_ceil(full_us, replaced_us) if full_us else None,
            "atom_status": (atom_rec or {}).get("status", "missing"),
            "ours_status": (ours_rec or {}).get("status", "missing"),
            "verdict": _verdict(ours_us or 0.0, atom_us or 0.0),
        }
        rows.append(row)

    # Markdown.
    lines = [
        "# 3-way Comparison: BF16 vs Atom W4A4 vs Ours\n",
        "| model | phase | B | bf16_full (us) | atom (us) | ours (us) "
        "| s_atom/bf16 | s_ours/bf16 | s_ours/atom | amdahl_ceil | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        def _f(v, fmt=".2f"):
            return ("—" if v is None else format(v, fmt))
        lines.append(
            f"| {r['model']} | {r['phase']} | {r['batch']} "
            f"| {_f(r['bf16_full_us'])} "
            f"| {_f(r['atom_layer_us'])} "
            f"| {_f(r['ours_layer_us'])} "
            f"| {_f(r['speedup_atom_vs_bf16'], '.2f')}× "
            f"| {_f(r['speedup_ours_vs_bf16'], '.2f')}× "
            f"| {_f(r['speedup_ours_vs_atom'], '.2f')}× "
            f"| {_f(r['amdahl_ceil_bf16'], '.2f')}× "
            f"| {r['verdict']} |"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison_table.md").write_text("\n".join(lines) + "\n")
    (args.out_dir / "comparison_table.json").write_text(
        json.dumps({"rows": rows}, indent=2)
    )
    print(f"[out] wrote comparison_table.{{md,json}} to {args.out_dir}")

    wins = sum(1 for r in rows if r["verdict"].startswith("✅"))
    ties = sum(1 for r in rows if r["verdict"].startswith("≈"))
    loses = sum(1 for r in rows if r["verdict"].startswith("❌"))
    print(f"[verdict] ours wins {wins}, ties {ties}, loses {loses} (of {len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
