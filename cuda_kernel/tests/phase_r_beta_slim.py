"""Phase R β-scan (SLIMMED): smart-pruned knob scan.

We already know from phase_r_probe:
  - 14B gu T=2048: splitk=2 wins (1.033x)
  - 14B/32B kv T=2048: default already wins (1.03x) -- bench FP16 baseline
    was artifically low
  - 4B dn T=1024: splitk=2 wins (0.96x, marginal)
  - 70B gu / 32B gu / 70B kv: no knob helped

So we ONLY need to:
  1. Verify splitk=2 on 14B/32B/70B gu at T in {2048, 4096, 8192}
     (does it generalise?)
  2. Verify splitk=2 does NOT regress winners (8B gu, 14B q, 4B q)
  3. Try a few more creative knobs on the hopeless 70B/32B gu T=2048
     (maybe kBn=32 with splitk=2?)

Protocol: warmup=300, outer=5, inner=150, median of 1 trial per knob.
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})


def bench_us(fn, warmup=300, outer=5, inner=150):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / inner)
    return best


def bench_fp16_flushed(W_fp, X_fp_t, warmup=150, outer=3, inner=80,
                      flush_mb=96):
    flush = torch.empty(flush_mb * 1024 * 256, dtype=torch.int8, device="cuda")
    def _flush_once():
        flush.zero_()
    for _ in range(warmup):
        _flush_once(); torch.matmul(W_fp, X_fp_t)
    torch.cuda.synchronize()
    best = float('inf')
    for _ in range(outer):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            _flush_once(); torch.matmul(W_fp, X_fp_t)
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / inner)
    for _ in range(warmup):
        _flush_once()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(inner):
        _flush_once()
    e.record()
    torch.cuda.synchronize()
    flush_us = s.elapsed_time(e) * 1000.0 / inner
    return max(0.0, best - flush_us)


def set_env(kbm=None, kbn=None, splitk=None):
    for k in ("HKUST_V9_FUSED_FORCE_KBM",
              "HKUST_V9_FUSED_FORCE_KBN",
              "HKUST_V9_FUSED_FORCE_SPLITK"):
        os.environ.pop(k, None)
    if kbm is not None: os.environ["HKUST_V9_FUSED_FORCE_KBM"] = str(kbm)
    if kbn is not None: os.environ["HKUST_V9_FUSED_FORCE_KBN"] = str(kbn)
    if splitk is not None: os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(splitk)


def run_e2e(X, perm, b, d_out, d_in):
    X_s4, sx, sX = ops.activation_quant_cuda(X, perm)
    return ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        X_s4, b["scale_u4"], b["zero_u4"], sX, sx, d_out, d_in,
    )


# Smart-pruned knob set
KNOBS = [
    ("default",        None, None, None),
    ("sk2",            None, None, 2),
    ("sk4",            None, None, 4),
    ("kBn32",          None, 32,   None),
    ("kBn32/sk2",      None, 32,   2),
    ("kBn16",          None, 16,   None),
]

# Loser shapes
LOSERS = [
    # gate_up LARGE
    ("14B gu",  2048, 5120, 34816),
    ("14B gu",  4096, 5120, 34816),
    ("14B gu",  8192, 5120, 34816),
    ("32B gu",  2048, 5120, 55296),
    ("32B gu",  4096, 5120, 55296),
    ("70B gu",  2048, 8192, 57344),
    ("70B gu",  4096, 8192, 57344),
    # kv LARGE
    ("14B kv",  2048, 5120, 2048),
    ("14B kv",  8192, 5120, 2048),
    ("32B kv",  2048, 5120, 2048),
    ("70B kv",  1024, 8192, 2048),
    ("70B kv",  2048, 8192, 2048),
    # dn SMALL
    ("1.7B dn", 1024, 6144, 2048),
    ("4B dn",   1024, 9728, 2560),
]

# Winner shapes (must not regress)
WINNERS = [
    ("8B gu",   2048, 4096, 24576),
    ("8B gu",   4096, 4096, 24576),
    ("14B q",   2048, 5120,  5120),
    ("4B q",    2048, 2560,  4096),
    # Phase C.6-v2 rule already sets splitk=2 on these -- verify it doesn't change
    ("32B dn",  2048, 27648, 5120),
    ("14B dn",  2048, 17408, 5120),
]


def run_shape(label_s, T, d_in, d_out, tag):
    b = make_inputs(T, d_out, d_in, hp_ratio=0.0,
                    device="cuda", seed=T + d_in + d_out)
    W_fp = b["W_fp"]; X_fp_t = b["X_fp_t"]
    X = b["X"]; perm = b["perm"]
    t_fp16 = bench_fp16_flushed(W_fp, X_fp_t)
    results = {}
    for knob_name, kbm, kbn, sk in KNOBS:
        set_env(kbm, kbn, sk)
        try:
            t = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in))
        except Exception:
            t = float('inf')
        results[knob_name] = t
    set_env()
    default = results["default"]
    best_knob = min(results, key=lambda k: results[k])
    best_us = results[best_knob]
    row = f"  [{tag}] {label_s:<8} T={T:<5}  fp16={t_fp16:>8.2f}  "
    for knob_name, _, _, _ in KNOBS:
        us = results[knob_name]
        delta = (us - default) / default * 100 if default < float('inf') else 0
        marker = "!" if us < default * 0.98 else ("-" if us > default * 1.02 else " ")
        row += f"{knob_name}={us:>7.2f}({delta:+4.1f}%){marker} "
    row += f" best={best_knob}({best_us:.2f} {t_fp16/best_us:.3f}x"
    row += (" ✓)" if t_fp16/best_us >= 1.0 else ")")
    PR(row)
    return results, t_fp16


PR("=" * 180)
PR("Phase R β-scan SLIMMED: 6 knobs × 20 shapes")
PR("Legend: '!' = this knob is >2% faster than default; '-' = >2% slower")
PR("=" * 180)
PR()
PR("### LOSER shapes")
loser_results = {}
for label_s, T, d_in, d_out in LOSERS:
    res, fp16 = run_shape(label_s, T, d_in, d_out, "L")
    loser_results[(label_s, T)] = (res, fp16)

PR()
PR("### WINNER shapes (regression sanity)")
winner_results = {}
for label_s, T, d_in, d_out in WINNERS:
    res, fp16 = run_shape(label_s, T, d_in, d_out, "W")
    winner_results[(label_s, T)] = (res, fp16)

# ==================================================
# Analysis: for each knob, summarise loser wins and winner regressions
# ==================================================
PR()
PR("=" * 180)
PR("ANALYSIS: per-knob impact summary")
PR("=" * 180)
PR()
for knob_name, _, _, _ in KNOBS:
    if knob_name == "default":
        continue
    loser_wins = []
    loser_regs = []
    winner_regs = []
    for (ls, T), (res, fp16) in loser_results.items():
        d = res["default"]
        k = res[knob_name]
        if k == float('inf') or d == float('inf'):
            continue
        d_sp = fp16 / d
        k_sp = fp16 / k
        delta = (k - d) / d * 100
        if k_sp >= 1.0 and d_sp < 1.0:
            loser_wins.append((ls, T, d_sp, k_sp))
        if delta > 2.0:
            loser_regs.append((ls, T, delta))
    for (ws, T), (res, fp16) in winner_results.items():
        d = res["default"]; k = res[knob_name]
        if k == float('inf') or d == float('inf'):
            continue
        delta = (k - d) / d * 100
        if delta > 2.0:
            winner_regs.append((ws, T, delta))
    PR(f"\n### Knob: {knob_name}")
    PR(f"  Loser wins (< 1.0x → ≥ 1.0x): {len(loser_wins)}")
    for ls, T, d_sp, k_sp in loser_wins:
        PR(f"    {ls} T={T}: {d_sp:.3f}x → {k_sp:.3f}x")
    PR(f"  Loser regressions (>2%): {len(loser_regs)}")
    for ls, T, d in loser_regs:
        PR(f"    {ls} T={T}: +{d:.2f}%")
    PR(f"  Winner regressions (>2%): {len(winner_regs)}")
    for ws, T, d in winner_regs:
        PR(f"    {ws} T={T}: +{d:.2f}%")
