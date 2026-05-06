"""r79 baseline — Atom W4A4 per-GEMM timing on Qwen3 shapes.

Output schema mirrors ``bench_bf16_per_op.json`` so ``compose_baseline_layer.py``
can join the two cleanly.  See ``MODEL_SELECTION.md §4`` for the
Amdahl interpretation.

For each (model, phase, batch) point we time:

  * ``rmsnorm_quant_pre_qkv``   — fused RMSNorm + activation quant, K=hidden
  * ``gemm_qkv_fused``          — INT4 GEMM, M=B*S, N=q_out+2*kv_out, K=hidden
  * ``reorder_quant_pre_o``     — fp16 reorder + quant, K=q_out
  * ``gemm_o``                  — INT4 GEMM, K=q_out, N=hidden
  * ``rmsnorm_quant_pre_gate_up`` — fused RMSNorm + quant, K=hidden
  * ``gemm_gate_up_fused``      — INT4 GEMM, K=hidden, N=2*intermediate
  * ``activate_quant_pre_down`` — silu(gate)*up + quant, K=intermediate
  * ``gemm_down``               — INT4 GEMM, K=intermediate, N=hidden

Some shapes will be invalid for the Atom kernel (e.g. K=hidden=2560 for
Qwen3-4B → K-keeper=2432, divisible by 128 ✓, M=batch ≥ 16 for prefill
all OK; for decode M=batch*1 = 4/8/16/32 — M=4/8 fails the M%16==0
constraint).  Such points are NOT silently skipped: they're recorded
in the JSON with ``status: "invalid"`` and a human-readable reason,
and counted in the validation log.

Usage
-----
    python -m kernel.bench.scripts.bench_baseline_w4a4_gemm \\
        --models Qwen3-4B Qwen3-8B Qwen3-14B \\
        --phases prefill decode \\
        --batches 4 8 16 32 \\
        --timing strict \\
        --out-dir kernel/bench/logs/baseline_$(date +%Y%m%d_%H%M)

Add ``--gemm-only`` to skip the quant-side ops (faster, but the resulting
layer composition will assume zero quant overhead — only use for
sanity / upper-bound estimates).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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

from kernel.bench.baselines import BaselineUnavailable  # noqa: E402
from kernel.bench.baselines.shapes import (  # noqa: E402
    AtomGemmShape,
    enumerate_atom_shapes,
)
from kernel.bench.configs.qwen3_shapes import (  # noqa: E402
    BATCH_SIZES,
    DECODE,
    PHASES,
    PREFILL,
    QWEN3_BY_NAME,
    QWEN3_MODELS,
)
from kernel.bench.layer.timing import LIGHT, STRICT, TimingStats, measure  # noqa: E402


# Op order — the keys we emit per point.  Layout chosen so the markdown
# table reads top-to-bottom in execution order of one Qwen3 layer.
OP_ORDER: tuple[str, ...] = (
    "rmsnorm_quant_pre_qkv",
    "gemm_qkv_fused",
    "reorder_quant_pre_o",
    "gemm_o",
    "rmsnorm_quant_pre_gate_up",
    "gemm_gate_up_fused",
    "activate_quant_pre_down",
    "gemm_down",
)

GEMM_OPS: tuple[str, ...] = (
    "gemm_qkv_fused", "gemm_o", "gemm_gate_up_fused", "gemm_down",
)
QUANT_OPS: tuple[str, ...] = (
    "rmsnorm_quant_pre_qkv", "reorder_quant_pre_o",
    "rmsnorm_quant_pre_gate_up", "activate_quant_pre_down",
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
        "kernel_backend": "atom_punica",
    }


def _build_callables(
    shapes_by_proj: dict[str, AtomGemmShape],
    *,
    cfg_intermediate: int,
    cfg_hidden: int,
    M: int,
    device: torch.device,
    gemm_only: bool,
) -> dict[str, callable]:
    """Wire each OP_ORDER name to a zero-arg callable.

    Imports of ``atom_punica`` happen here so the module can still be
    loaded for ``--help`` on a host without the Atom kernel.
    """
    from kernel.bench.baselines import atom_punica as ap

    qkv = shapes_by_proj["qkv_fused"]
    o = shapes_by_proj["o"]
    gu = shapes_by_proj["gate_up_fused"]
    dn = shapes_by_proj["down"]

    fns: dict[str, callable] = {
        "gemm_qkv_fused":     ap.build_gemm_callable(qkv.M, qkv.N, qkv.K, device=device),
        "gemm_o":             ap.build_gemm_callable(o.M, o.N, o.K, device=device),
        "gemm_gate_up_fused": ap.build_gemm_callable(gu.M, gu.N, gu.K, device=device),
        "gemm_down":          ap.build_gemm_callable(dn.M, dn.N, dn.K, device=device),
    }
    if not gemm_only:
        # K for the quant ops:
        #   pre_qkv  / pre_gate_up : K = hidden  (input to RMSNorm)
        #   pre_o                  : K = q_out   (== o_proj's d_in, == o.K)
        #   pre_down               : K = intermediate (gate/up output width)
        fns["rmsnorm_quant_pre_qkv"]     = ap.build_quantize_via_rmsnorm_callable(M, cfg_hidden, device=device)
        fns["reorder_quant_pre_o"]       = ap.build_quantize_via_reorder_callable(M, o.K, device=device)
        fns["rmsnorm_quant_pre_gate_up"] = ap.build_quantize_via_rmsnorm_callable(M, cfg_hidden, device=device)
        fns["activate_quant_pre_down"]   = ap.build_quantize_via_activate_callable(M, cfg_intermediate, device=device)
    return fns


def _validate_shapes(
    shapes: list[AtomGemmShape],
) -> tuple[dict[str, AtomGemmShape], list[dict]]:
    """Split shapes into ``valid`` / ``invalid`` lists.

    Returns
    -------
    by_proj : dict
        ``proj_name -> AtomGemmShape`` for valid shapes only.
    invalids : list of dict
        Each dict has ``proj``, ``M``, ``N``, ``K``, ``reason``.
    """
    by_proj: dict[str, AtomGemmShape] = {}
    invalids: list[dict] = []
    for s in shapes:
        try:
            s.assert_valid_for_atom()
        except ValueError as e:
            invalids.append({
                "proj": s.proj, "M": s.M, "N": s.N, "K": s.K,
                "reason": str(e),
            })
        else:
            by_proj[s.proj] = s
    return by_proj, invalids


def _run_one_point(
    *,
    model_name: str,
    phase_name: str,
    batch: int,
    timing_kwargs: dict,
    device: torch.device,
    gemm_only: bool,
) -> dict:
    cfg = QWEN3_BY_NAME[model_name]
    phase = PREFILL if phase_name == "prefill" else DECODE

    print(
        f"\n=== [Atom] {model_name} | {phase_name} | B={batch} | "
        f"S={phase.seqlen} | M={batch * phase.seqlen} ===",
        flush=True,
    )

    shapes = enumerate_atom_shapes(cfg, phase, batch)
    by_proj, invalids = _validate_shapes(shapes)

    if invalids:
        for inv in invalids:
            print(f"  [skip-invalid] {inv['proj']}: {inv['reason']}", flush=True)

    record: dict = {
        "model": model_name,
        "phase": phase_name,
        "batch": batch,
        "seqlen": phase.seqlen,
        "M": batch * phase.seqlen,
        "hidden": cfg.hidden,
        "intermediate": cfg.intermediate,
        "shapes": {s.proj: {"M": s.M, "N": s.N, "K": s.K} for s in shapes},
        "invalid_shapes": invalids,
        "ops": {},
        "status": "ok" if not invalids else "partial",
    }

    if len(by_proj) < 4:
        record["status"] = "skipped_all_gemm_invalid"
        return record

    try:
        fns = _build_callables(
            by_proj,
            cfg_intermediate=cfg.intermediate,
            cfg_hidden=cfg.hidden,
            M=batch * phase.seqlen,
            device=device,
            gemm_only=gemm_only,
        )
    except BaselineUnavailable as e:
        record["status"] = "baseline_unavailable"
        record["error"] = str(e)
        return record

    for op_name in OP_ORDER:
        if op_name not in fns:  # gemm-only mode skips quant ops
            continue
        fn = fns[op_name]
        stats = measure(fn, **timing_kwargs, device=device)
        record["ops"][op_name] = stats.as_dict()
        print(
            f"  {op_name:>26s}: median={stats.median_us:>10.3f} us "
            f"(spread {stats.spread_pct:>5.1f}%)",
            flush=True,
        )

    # Aggregate sums.
    gemm_us = sum(record["ops"][k]["median_us"] for k in GEMM_OPS if k in record["ops"])
    quant_us = sum(record["ops"][k]["median_us"] for k in QUANT_OPS if k in record["ops"])
    record["sanity"] = {
        "gemm_sum_us": gemm_us,
        "quant_sum_us": quant_us,
        "replaced_region_us": gemm_us + quant_us,
    }
    return record


def _format_summary_md(records: list[dict], env: dict) -> str:
    lines: list[str] = [
        "# Baseline (Atom W4A4 via punica-atom) per-GEMM timing\n",
        f"- Host: `{env['hostname']}` | GPU: **{env['gpu']}** (SM {env['gpu_sm']})",
        f"- torch {env['torch']} / CUDA {env['cuda']} / Python {env['python']}",
        f"- backend: **{env['kernel_backend']}** (Atom paper config: keeper=128, INT8 outlier)",
        f"- timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}\n",
    ]
    for r in records:
        lines.append(
            f"## {r['model']} · {r['phase']} · B={r['batch']} (M={r['M']})  "
            f"— status: **{r['status']}**\n"
        )
        if r['invalid_shapes']:
            lines.append("Invalid shapes (skipped):")
            for inv in r['invalid_shapes']:
                lines.append(f"  - `{inv['proj']}` (M={inv['M']}, N={inv['N']}, K={inv['K']}): {inv['reason']}")
            lines.append("")
        if not r['ops']:
            lines.append("_No timings collected._\n")
            continue
        lines.append("| op | M | N | K | median (us) | spread % |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for op in OP_ORDER:
            if op not in r['ops']:
                continue
            s = r['ops'][op]
            shape_key = {
                "gemm_qkv_fused": "qkv_fused",
                "gemm_o": "o",
                "gemm_gate_up_fused": "gate_up_fused",
                "gemm_down": "down",
            }.get(op)
            if shape_key and shape_key in r['shapes']:
                sh = r['shapes'][shape_key]
                shape_str = f"| {sh['M']} | {sh['N']} | {sh['K']} "
            else:
                shape_str = "| — | — | — "
            lines.append(
                f"| {op} {shape_str}"
                f"| {s['median_us']:.3f} | {s['spread_pct']:.1f} |"
            )
        if "sanity" in r:
            sane = r["sanity"]
            lines.append("")
            lines.append(
                f"**Replaced-region cost** = Σ GEMM + Σ quant = "
                f"**{sane['gemm_sum_us']:.2f} + {sane['quant_sum_us']:.2f} = "
                f"{sane['replaced_region_us']:.2f} us**\n"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="r79 Atom W4A4 baseline per-GEMM bench")
    p.add_argument("--models", nargs="+", default=[c.name for c in QWEN3_MODELS])
    p.add_argument("--phases", nargs="+", default=[ph.name for ph in PHASES],
                   choices=["prefill", "decode"])
    p.add_argument("--batches", nargs="+", type=int, default=list(BATCH_SIZES))
    p.add_argument("--timing", choices=["light", "strict"], default="strict")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--gemm-only", action="store_true",
        help="Time only the 4 INT4 GEMMs, skip quant ops "
             "(produces an over-optimistic baseline; use only for sanity).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[fatal] CUDA not available", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    timing_kwargs = dict(STRICT if args.timing == "strict" else LIGHT)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _env_info()
    print(f"[env] {env}")
    print(f"[timing] preset={args.timing} kwargs={timing_kwargs} gemm_only={args.gemm_only}")
    print(f"[out] {out_dir}")

    for m in args.models:
        if m not in QWEN3_BY_NAME:
            print(f"[fatal] unknown model {m!r}", file=sys.stderr)
            return 2

    records: list[dict] = []
    t0 = _dt.datetime.now()
    try:
        for m in args.models:
            for ph in args.phases:
                for b in args.batches:
                    rec = _run_one_point(
                        model_name=m, phase_name=ph, batch=b,
                        timing_kwargs=timing_kwargs,
                        device=device, gemm_only=args.gemm_only,
                    )
                    records.append(rec)
                    with open(out_dir / "atom_w4a4_gemm.json", "w") as f:
                        json.dump(
                            {"env": env, "timing": timing_kwargs,
                             "gemm_only": args.gemm_only,
                             "records": records},
                            f, indent=2,
                        )
    finally:
        elapsed = (_dt.datetime.now() - t0).total_seconds()
        print(f"\n[done] {len(records)} points in {elapsed:.1f}s")
        if records:
            with open(out_dir / "atom_w4a4_gemm_summary.md", "w") as f:
                f.write(_format_summary_md(records, env))
            print(f"[out] wrote summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
