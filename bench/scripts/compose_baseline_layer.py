"""Compose baseline-layer timing from Atom W4A4 GEMM + BF16 non-replaced ops.

The composition formula (per discussion 2026-05-06):

    T_baseline_layer
        = Σ T_atom_gemm[qkv,o,gate_up,down]            # from atom_w4a4_gemm.json
        + Σ T_atom_quant[pre_qkv,pre_o,pre_gu,pre_dn]  # idem
        + T_attention_bf16                             # from bench_bf16_per_op.json
        + T_qk_norm_bf16  + T_rope_bf16                # ditto (decode-only effectively)
        + T_layer_residual_bf16                        # = full_layer_bf16
                                                       #   - Σ bf16_gemm_us
                                                       #   - T_attn - T_qknorm - T_rope
                                                       #   - T_input_rmsnorm - T_post_rmsnorm
                                                       # i.e. residual adds + GQA broadcast
                                                       # + reshape + KV concat (exact-by-defn)

Why we drop the BF16 RMSNorms from the residual:
  * Atom replaces both ``input_layernorm`` and ``post_attention_layernorm``
    with its own fused rmsnorm+quant kernel; that cost is captured in
    ``T_atom_quant[pre_qkv]`` and ``T_atom_quant[pre_gu]`` respectively.
  * If we kept the BF16 rmsnorm inside the residual we'd double-count.

Why we *keep* QK-norm and RoPE in BF16:
  * Atom's modeling_llama doesn't have QK-norm at all (Llama doesn't);
    Qwen3 has it.  Keeping QK-norm at BF16 is the *most charitable* model
    of "Atom on Qwen3": we assume Atom would do exactly what we do for
    those head-dim ops.  Same for RoPE.

Output schema mirrors ``bench_bf16_per_op.json`` (one record per
(model, phase, batch)) so the downstream comparison script can iterate
the same way.

Usage
-----
    python -m kernel.bench.scripts.compose_baseline_layer \\
        --bf16-json kernel/bench/logs/bf16_<ts>/bench_bf16_per_op.json \\
        --atom-json kernel/bench/logs/baseline_<ts>/atom_w4a4_gemm.json \\
        --out-dir   kernel/bench/logs/baseline_<ts>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

# Make "kernel.*" importable.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Op groups — must stay in sync with the producer scripts.
BF16_GEMM_OPS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
# BF16 ops kept "as-is" inside the baseline layer (Atom doesn't replace these
# on Qwen3 in our model).
BF16_KEPT_OPS = ("q_norm", "k_norm", "rope", "attention")
# BF16 ops *replaced* by Atom (their cost moves into atom_quant_*).
BF16_REPLACED_OPS = ("input_rmsnorm", "post_rmsnorm")

ATOM_GEMM_OPS = (
    "gemm_qkv_fused", "gemm_o", "gemm_gate_up_fused", "gemm_down",
)
ATOM_QUANT_OPS = (
    "rmsnorm_quant_pre_qkv", "reorder_quant_pre_o",
    "rmsnorm_quant_pre_gate_up", "activate_quant_pre_down",
)


def _key(rec: dict) -> tuple[str, str, int]:
    return (rec["model"], rec["phase"], rec["batch"])


def _safe_get(record: dict, op: str) -> float | None:
    """Return median_us if op is present, else None."""
    ops = record.get("ops", {})
    if op not in ops:
        return None
    return float(ops[op]["median_us"])


def _compose_one(bf16_rec: dict, atom_rec: dict) -> dict:
    """Build one composed baseline record.

    Returns a dict with
      * ``baseline_layer_us``    — composed total (= what we'd report)
      * ``atom_gemm_sum_us``     — Σ INT4 GEMM
      * ``atom_quant_sum_us``    — Σ Atom quant ops
      * ``bf16_kept_sum_us``     — Σ QK-norm + RoPE + attention (kept bf16)
      * ``bf16_layer_residual_us`` — exact black-box leftover
                                     (full_layer_bf16
                                       − Σ bf16_gemm_us
                                       − Σ bf16_kept_sum_us
                                       − Σ bf16_replaced_us)
        i.e. residual adds + KV concat + GQA broadcast + reshape.
      * ``status``               — "ok" | "missing_atom" | "missing_bf16" | ...

    All numbers in microseconds.
    """
    out: dict = {
        "model": bf16_rec["model"],
        "phase": bf16_rec["phase"],
        "batch": bf16_rec["batch"],
    }

    full_us = bf16_rec["sanity"]["full_layer_us"]
    bf16_gemm_us = bf16_rec["sanity"]["gemm_sum_us"]

    # Atom side may be missing (invalid shape) or partial.
    if atom_rec is None or atom_rec.get("status", "ok") in (
        "skipped_all_gemm_invalid", "baseline_unavailable",
    ):
        out["status"] = atom_rec.get("status", "missing_atom") if atom_rec else "missing_atom"
        out["bf16_full_layer_us"] = full_us
        return out

    atom_gemm = {op: _safe_get(atom_rec, op) for op in ATOM_GEMM_OPS}
    atom_quant = {op: _safe_get(atom_rec, op) for op in ATOM_QUANT_OPS}

    if any(v is None for v in atom_gemm.values()):
        out["status"] = "atom_partial_gemm"
        out["atom_missing"] = [k for k, v in atom_gemm.items() if v is None]
        out["bf16_full_layer_us"] = full_us
        return out

    atom_gemm_sum = sum(atom_gemm.values())
    # Quant ops may be entirely absent if atom was run with --gemm-only.
    have_quant = all(v is not None for v in atom_quant.values())
    atom_quant_sum = sum(atom_quant.values()) if have_quant else 0.0

    # BF16 kept & replaced sums.
    bf16_kept_sum = 0.0
    for op in BF16_KEPT_OPS:
        v = _safe_get(bf16_rec, op)
        if v is None:
            out["status"] = f"bf16_missing_{op}"
            out["bf16_full_layer_us"] = full_us
            return out
        bf16_kept_sum += v
    bf16_replaced_sum = 0.0
    for op in BF16_REPLACED_OPS:
        v = _safe_get(bf16_rec, op)
        if v is None:
            v = 0.0  # not fatal; just means the residual will absorb it
        bf16_replaced_sum += v

    bf16_layer_residual = (
        full_us - bf16_gemm_us - bf16_kept_sum - bf16_replaced_sum
    )
    # The residual should be small but nonnegative.  If it isn't, that's
    # a measurement artifact (per-op was over-timed somewhere) and we
    # clamp to zero with a warning flag rather than producing a negative
    # composed time.
    residual_clamped = max(0.0, bf16_layer_residual)

    baseline_layer_us = (
        atom_gemm_sum
        + atom_quant_sum
        + bf16_kept_sum
        + residual_clamped
    )

    out.update({
        "status": "ok" if have_quant else "ok_gemm_only",
        "bf16_full_layer_us": full_us,
        "bf16_gemm_sum_us": bf16_gemm_us,
        "bf16_kept_sum_us": bf16_kept_sum,
        "bf16_replaced_sum_us": bf16_replaced_sum,
        "bf16_layer_residual_us": bf16_layer_residual,
        "bf16_layer_residual_clamped_us": residual_clamped,
        "atom_gemm_sum_us": atom_gemm_sum,
        "atom_quant_sum_us": atom_quant_sum,
        "atom_per_gemm": atom_gemm,
        "atom_per_quant": atom_quant if have_quant else {},
        "baseline_layer_us": baseline_layer_us,
        "speedup_vs_bf16": full_us / baseline_layer_us if baseline_layer_us > 0 else float("nan"),
        "warning_residual_negative": bf16_layer_residual < 0,
    })
    return out


def _format_summary_md(composed: list[dict]) -> str:
    lines = [
        "# Baseline composition (Atom W4A4 + BF16 non-replaced)\n",
        "Each row composes the timings as:",
        "`baseline_layer_us = atom_gemm_sum + atom_quant_sum + bf16_kept_sum + bf16_layer_residual`",
        "where residual = `full_bf16 − Σ bf16_gemm − Σ bf16_kept − Σ bf16_replaced` (exact).\n",
        "**Caveat**: a `negR` flag in the warn column means the BF16 measurement",
        "yielded a negative residual (per-op sum > full_layer due to noise);",
        "we clamped it to 0 in the layer total, which makes that row's",
        "`speedup_vs_bf16` an over-estimate.  Re-run that BF16 point at strict",
        "timing if it appears.\n",
        "| model | phase | B | bf16_full | atom_gemm | atom_quant | bf16_kept | residual | baseline_layer | **speedup** | warn | status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in composed:
        if r["status"].startswith("missing") or r["status"].startswith("atom_") or r["status"].startswith("bf16_missing"):
            lines.append(
                f"| {r['model']} | {r['phase']} | {r['batch']} "
                f"| {r.get('bf16_full_layer_us', float('nan')):.2f} "
                f"| — | — | — | — | — | — | — | {r['status']} |"
            )
            continue
        warn = "⚠negR" if r.get("warning_residual_negative") else ""
        lines.append(
            f"| {r['model']} | {r['phase']} | {r['batch']} "
            f"| {r['bf16_full_layer_us']:.2f} "
            f"| {r['atom_gemm_sum_us']:.2f} "
            f"| {r['atom_quant_sum_us']:.2f} "
            f"| {r['bf16_kept_sum_us']:.2f} "
            f"| {r['bf16_layer_residual_clamped_us']:.2f} "
            f"| **{r['baseline_layer_us']:.2f}** "
            f"| **{r['speedup_vs_bf16']:.3f}×** "
            f"| {warn} "
            f"| {r['status']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Compose Atom + BF16 baseline layer")
    p.add_argument("--bf16-json", required=True, type=Path,
                   help="path to bench_bf16_per_op.json")
    p.add_argument("--atom-json", required=True, type=Path,
                   help="path to atom_w4a4_gemm.json")
    p.add_argument("--out-dir", required=True, type=Path)
    args = p.parse_args()

    bf16_data = json.loads(args.bf16_json.read_text())
    atom_data = json.loads(args.atom_json.read_text())

    bf16_by_key = {_key(r): r for r in bf16_data["records"]}
    atom_by_key = {_key(r): r for r in atom_data["records"]}

    composed: list[dict] = []
    for k, br in bf16_by_key.items():
        ar = atom_by_key.get(k)
        composed.append(_compose_one(br, ar))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = {
        "composed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "bf16_source": str(args.bf16_json),
        "atom_source": str(args.atom_json),
        "bf16_env": bf16_data.get("env", {}),
        "atom_env": atom_data.get("env", {}),
        "records": composed,
    }
    (args.out_dir / "baseline_layer.json").write_text(json.dumps(out_json, indent=2))
    (args.out_dir / "baseline_layer_summary.md").write_text(_format_summary_md(composed))
    print(f"[out] wrote baseline_layer.json + baseline_layer_summary.md to {args.out_dir}")

    # Quick stdout digest.
    ok = [r for r in composed if r["status"].startswith("ok")]
    if ok:
        speedups = [r["speedup_vs_bf16"] for r in ok]
        print(
            f"[summary] {len(ok)}/{len(composed)} points composed | "
            f"speedup vs bf16: min={min(speedups):.2f}x "
            f"median={sorted(speedups)[len(speedups)//2]:.2f}x "
            f"max={max(speedups):.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
