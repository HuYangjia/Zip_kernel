"""r79 replacement bench — BF16 side (S3 of the plan).

Measures, for each (model, phase, batch) in MODEL_SELECTION.md §2, the
median-of-trials timing of:
  * 7 un-fused Linears : q / k / v / o / gate / up / down
  * 2 RMSNorms         : input_layernorm, post_attention_layernorm
  * QK-norm (head_dim) : q_norm, k_norm
  * RoPE apply
  * Attention (SDPA)
  * Full layer (sanity check)

Outputs
-------
  <out-dir>/bench_bf16_per_op.json  -- per-op numbers, all trials
  <out-dir>/bench_bf16_summary.md   -- pretty markdown table, one section
                                       per (model, phase, batch)
  <out-dir>/VALIDATION_LOG.md       -- Amdahl accounting (GEMM share,
                                       Amdahl ceiling, non-GEMM residual).
                                       The replacement-method only needs
                                       full_layer and Σ GEMM to be precise;
                                       everything else is captured as an
                                       exact-by-definition residual.

Usage
-----
    python -m kernel.bench.scripts.bench_layer_per_op_bf16 \
           --models Qwen3-4B Qwen3-8B Qwen3-14B \
           --phases prefill decode \
           --batches 4 8 16 32 \
           --timing strict \
           --out-dir kernel/bench/logs/$(date +%Y%m%d_%H%M)

Use ``--timing light`` for a quick smoke test (warmup=200, outer=10, trials=3).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import socket
import sys
from pathlib import Path

import torch

# Make "kernel.*" importable when invoked directly from any cwd.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernel.bench.configs.qwen3_shapes import (  # noqa: E402
    BATCH_SIZES,
    DECODE,
    PHASES,
    PREFILL,
    QWEN3_BY_NAME,
    QWEN3_MODELS,
    enumerate_unfused_projs,
)
from kernel.bench.layer.qwen3_layer_bf16 import (  # noqa: E402
    Qwen3LayerBF16,
    build_per_op_callables,
)
from kernel.bench.layer.timing import (  # noqa: E402
    ADAPTIVE_PREFILL,
    LIGHT,
    STRICT,
    TimingStats,
    measure,
)


# Per-op list — order matches report columns.
OP_ORDER: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "input_rmsnorm", "post_rmsnorm",
    "q_norm", "k_norm", "rope",
    "attention",
    "full_layer",
)

# -----------------------------------------------------------------------------
# Sanity / Amdahl accounting (rationale 2026-05-06)
# -----------------------------------------------------------------------------
# The replacement-method analysis only cares about THREE quantities:
#
#   * full_layer_bf16        -- the denominator of the speedup
#   * Σ GEMM_bf16            -- the set of ops that WILL be replaced by W4A4
#   * Σ GEMM_w4a4 (later)    -- the set of ops AFTER replacement
#
# Every other op (RMSNorms, QK-norm, RoPE, SDPA, KV concat, GQA broadcast,
# residual adds, silu*up) is NOT replaced in this study; their total cost is
# therefore captured as a single black-box residual  (full_layer − Σ GEMM)
# and that residual is, BY CONSTRUCTION, exact -- no per-op timing needed.
#
# Consequently the sanity check is weak on purpose: we only require
#   (a)  Σ GEMM  ≤  full_layer       (trivial correctness)
#   (b)  GEMM share in a plausible band  [3%, 70%]
#
# The per-op timings for the non-GEMM ops are STILL measured (see OP_ORDER),
# because they are cheap to collect and may be useful for a future finer
# decomposition (e.g. "what if we also fuse RMSNorm into the GEMM epilogue?").
# They just don't drive the verdict here.
GEMM_OPS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Informational only -- the "fine-grained" sum that used to be the sanity
# denominator.  Kept so the JSON consumer can still inspect the gap.
FINE_GRAINED_SUM_OPS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "input_rmsnorm", "post_rmsnorm",
    "q_norm", "k_norm", "rope",
    "attention",
)


def _env_info() -> dict:
    dev_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev_idx)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "gpu_sm": f"{props.major}.{props.minor}",
        "gpu_vram_gb": round(props.total_memory / 1024**3, 1),
        "bf16": True,
    }


def _run_one_point(
    *,
    model_name: str,
    phase_name: str,
    batch: int,
    timing_kwargs: dict,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    """Measure every op for one (model, phase, batch) triple.

    Returns a dict suitable for JSON serialisation.
    """
    cfg = QWEN3_BY_NAME[model_name]
    phase = PREFILL if phase_name == "prefill" else DECODE

    print(
        f"\n=== {model_name} | {phase_name} | B={batch} | "
        f"S={phase.seqlen} | past_kv={phase.past_kv_len} ===",
        flush=True,
    )

    # ---- build the layer + per-op callables -----------------------------
    torch.manual_seed(0xBEEF + hash((model_name, phase_name, batch)) & 0xFFFF)
    layer = Qwen3LayerBF16(cfg, device=device, dtype=dtype).eval()
    for p in layer.parameters():
        p.requires_grad_(False)

    with torch.inference_mode():
        ops = build_per_op_callables(
            layer,
            batch=batch,
            seqlen=phase.seqlen,
            past_kv_len=phase.past_kv_len,
        )

        result_ops: dict[str, TimingStats] = {}
        for op_name in OP_ORDER:
            fn = getattr(ops, op_name)
            stats = measure(fn, **timing_kwargs, device=device)
            result_ops[op_name] = stats
            print(
                f"  {op_name:>16s}: median={stats.median_us:>10.3f} us "
                f"(spread {stats.spread_pct:>5.1f}%)",
                flush=True,
            )

    # ---- shape record per op (for report annotation) --------------------
    proj_shapes = {p.proj: (p.d_in, p.d_out) for p in enumerate_unfused_projs(cfg)}

    full_us = result_ops["full_layer"].median_us
    gemm_us = sum(result_ops[k].median_us for k in GEMM_OPS)
    fine_us = sum(result_ops[k].median_us for k in FINE_GRAINED_SUM_OPS)
    non_gemm_residual_us = full_us - gemm_us  # black-box remainder, exact by defn
    gemm_share = gemm_us / full_us if full_us > 0 else float("nan")
    fine_gap_pct = (full_us - fine_us) / full_us * 100.0 if full_us > 0 else float("nan")

    return {
        "model": model_name,
        "phase": phase_name,
        "batch": batch,
        "seqlen": phase.seqlen,
        "past_kv_len": phase.past_kv_len,
        "hidden": cfg.hidden,
        "intermediate": cfg.intermediate,
        "num_q_heads": cfg.num_q_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "head_dim": cfg.head_dim,
        "proj_shapes": {
            k: {"d_in": v[0], "d_out": v[1]} for k, v in proj_shapes.items()
        },
        "ops": {k: v.as_dict() for k, v in result_ops.items()},
        "sanity": {
            # primary Amdahl accounting (what drives the verdict)
            "full_layer_us": full_us,
            "gemm_sum_us": gemm_us,
            "non_gemm_residual_us": non_gemm_residual_us,
            "gemm_share": gemm_share,
            # informational: old fine-grained sum, kept for diagnostics only
            "fine_grained_sum_us": fine_us,
            "fine_grained_gap_pct": fine_gap_pct,
        },
    }


def _format_summary_md(records: list[dict], env: dict) -> str:
    lines: list[str] = []
    lines.append("# r79 BF16 Per-Op Timing Summary\n")
    lines.append(f"- Host: `{env['hostname']}` | GPU: **{env['gpu']}** (SM {env['gpu_sm']}, {env['gpu_vram_gb']} GB)")
    lines.append(f"- torch {env['torch']} / CUDA {env['cuda']} / Python {env['python']}")
    lines.append(f"- dtype: **BF16** | timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}\n")

    for r in records:
        lines.append(
            f"## {r['model']} · {r['phase']} · B={r['batch']} "
            f"(S={r['seqlen']}, past_kv={r['past_kv_len']})\n"
        )
        lines.append(
            "| op | shape (d_in → d_out) | median (us) | min (us) | max (us) | spread % |"
        )
        lines.append("|---|---|---:|---:|---:|---:|")
        for op in OP_ORDER:
            s = r["ops"][op]
            shape_str = "—"
            proj_key_map = {
                "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
                "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
            }
            if op in proj_key_map:
                sh = r["proj_shapes"][proj_key_map[op]]
                shape_str = f"{sh['d_in']} → {sh['d_out']}"
            lines.append(
                f"| {op} | {shape_str} "
                f"| {s['median_us']:.3f} | {s['min_us']:.3f} | {s['max_us']:.3f} "
                f"| {s['spread_pct']:.1f} |"
            )
        sane = r["sanity"]
        lines.append("")
        lines.append(
            f"**Amdahl accounting**: full_layer = **{sane['full_layer_us']:.2f} us**, "
            f"Σ GEMM (7 Linears) = **{sane['gemm_sum_us']:.2f} us** "
            f"({sane['gemm_share']*100:.1f}% of layer), "
            f"non-GEMM residual = **{sane['non_gemm_residual_us']:.2f} us** "
            f"(black-box, exact by subtraction)."
        )
        lines.append(
            f"_Info only_ — fine-grained sum (13 non-add ops) = "
            f"{sane['fine_grained_sum_us']:.2f} us, "
            f"gap vs full_layer = {sane['fine_grained_gap_pct']:+.2f}% "
            "(the residual absorbs KV-concat / GQA-broadcast / reshape / residual-adds, "
            "which are not individually timed).\n"
        )
    return "\n".join(lines) + "\n"


def _format_validation_md(records: list[dict]) -> str:
    """Amdahl-style validation log.

    We only need TWO numbers to be trustworthy for the replacement-method
    speedup analysis: ``full_layer_us`` and ``gemm_sum_us``.  Every other op
    is captured correctly as the black-box residual ``full_layer − Σ GEMM``.

    Verdict rules
    -------------
    * ``gemm_sum > full_layer``           → ❌  bug (per-op exceeds layer)
    * ``gemm_share < 3%`` or ``> 70%``    → ⚠️  implausible share, investigate
    * otherwise                           → ✅  ok (gap numbers informational)
    """
    lines: list[str] = [
        "# r79 BF16 Replacement-Method Validation (Amdahl accounting)\n",
        "This bench answers: **if we replace the 7 BF16 GEMMs with W4A4, how much",
        "can one transformer layer speed up?**  The analysis only needs",
        "`full_layer_us` and `Σ GEMM_us`; the non-GEMM part enters as a single",
        "black-box residual `full_layer − Σ GEMM`, which is exact by construction.\n",
        "Verdict rules:",
        "* `Σ GEMM > full_layer` → ❌ bug",
        "* `GEMM share` outside `[3%, 70%]` → ⚠️ implausible (investigate)",
        "* otherwise → ✅ ok\n",
        "The `fine-grained gap` column is informational only: it measures how much",
        "of the layer is *not* covered by the 13 individually-timed ops.  Large gaps",
        "(e.g. +10% to +20% in decode) are EXPECTED because KV-concat, GQA broadcast,",
        "output reshape, residual adds and silu*up are not individually timed.\n",
        "| model | phase | batch | Σ GEMM (us) | full_layer (us) | GEMM share | Amdahl ceil | non-GEMM residual (us) | fine gap % | verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        s = r["sanity"]
        share = s["gemm_share"]
        # Theoretical upper bound on layer speedup if GEMMs took 0 time.
        amdahl_ceil = 1.0 / (1.0 - share) if 0.0 < share < 1.0 else float("inf")
        if s["gemm_sum_us"] > s["full_layer_us"]:
            verdict = "❌ Σ GEMM > full"
        elif not (0.03 <= share <= 0.70):
            verdict = "⚠️ share out of band"
        else:
            verdict = "✅ ok"
        lines.append(
            f"| {r['model']} | {r['phase']} | {r['batch']} "
            f"| {s['gemm_sum_us']:.2f} | {s['full_layer_us']:.2f} "
            f"| {share*100:.1f}% | {amdahl_ceil:.2f}× "
            f"| {s['non_gemm_residual_us']:.2f} "
            f"| {s['fine_grained_gap_pct']:+.2f} | {verdict} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="r79 BF16 per-op replacement bench")
    p.add_argument(
        "--models", nargs="+", default=[c.name for c in QWEN3_MODELS],
        help="model names (default: all 3 in MODEL_SELECTION.md)",
    )
    p.add_argument(
        "--phases", nargs="+", default=[ph.name for ph in PHASES],
        choices=["prefill", "decode"],
    )
    p.add_argument(
        "--batches", nargs="+", type=int, default=list(BATCH_SIZES),
    )
    p.add_argument(
        "--timing", choices=["light", "strict", "adaptive"], default="adaptive",
        help="timing preset. 'adaptive' (default) = STRICT for decode + "
             "ADAPTIVE_PREFILL for prefill (ms-level GEMMs don't need inner=200). "
             "'light' = smoke test. 'strict' = warmup=500/outer=20/inner=200 "
             "for every op (slow: ~2.5h for 24 points).",
    )
    p.add_argument(
        "--out-dir", type=str, required=True,
        help="output directory (will be created if missing)",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[fatal] CUDA not available", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    if args.timing == "strict":
        timing_kwargs_prefill = dict(STRICT)
        timing_kwargs_decode = dict(STRICT)
    elif args.timing == "light":
        timing_kwargs_prefill = dict(LIGHT)
        timing_kwargs_decode = dict(LIGHT)
    else:  # "adaptive"
        timing_kwargs_prefill = dict(ADAPTIVE_PREFILL)
        timing_kwargs_decode = dict(STRICT)
    dtype = torch.bfloat16

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _env_info()
    print(f"[env] {env}")
    print(f"[timing] preset={args.timing}")
    print(f"[timing]   prefill kwargs = {timing_kwargs_prefill}")
    print(f"[timing]   decode  kwargs = {timing_kwargs_decode}")
    print(f"[out] {out_dir}")

    # Validate model names early.
    for m in args.models:
        if m not in QWEN3_BY_NAME:
            print(f"[fatal] unknown model {m!r}; valid: {sorted(QWEN3_BY_NAME)}", file=sys.stderr)
            return 2

    records: list[dict] = []
    t0 = _dt.datetime.now()
    try:
        for m in args.models:
            for ph in args.phases:
                for b in args.batches:
                    timing_kwargs = (
                        timing_kwargs_prefill if ph == "prefill"
                        else timing_kwargs_decode
                    )
                    rec = _run_one_point(
                        model_name=m, phase_name=ph, batch=b,
                        timing_kwargs=timing_kwargs,
                        dtype=dtype, device=device,
                    )
                    records.append(rec)
                    # Flush-save after every point so a mid-run crash keeps
                    # previous results.
                    with open(out_dir / "bench_bf16_per_op.json", "w") as f:
                        json.dump(
                            {
                                "env": env,
                                "timing_preset": args.timing,
                                "timing_prefill": timing_kwargs_prefill,
                                "timing_decode": timing_kwargs_decode,
                                "records": records,
                            },
                            f, indent=2,
                        )
    finally:
        elapsed = (_dt.datetime.now() - t0).total_seconds()
        print(f"\n[done] {len(records)} points in {elapsed:.1f}s")

        # Always write whatever we have to markdown too.
        if records:
            with open(out_dir / "bench_bf16_summary.md", "w") as f:
                f.write(_format_summary_md(records, env))
            with open(out_dir / "VALIDATION_LOG.md", "w") as f:
                f.write(_format_validation_md(records))
            print(f"[out] wrote summary + validation to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
