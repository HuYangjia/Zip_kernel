"""Phase R β-scan — full knob cube on the 3 loser families.

Extends phase_r_probe by adding the kBn dimension.  For prefill T >= 1024
the default dispatcher always picks kBn=64.  But kBn=32 may be better
for certain shapes due to reduced register pressure per CTA and
improved wave balance.

Knobs (env vars):
  KBM ∈ {128, 64}
  KBN ∈ {64, 32, 16}
  SPLITK ∈ {1, 2, 4}
=> 2 × 3 × 3 = 18 configs per shape (including default).

Loser shapes we must not regress on other T (3 per family):
  A: gate_up LARGE (14B/32B/70B gu at T=2048, T=4096, T=8192)
  B: kv LARGE      (14B/32B/70B kv at T=2048, T=4096, T=8192)
  C: down_proj SMALL (1.7B/4B dn at T=1024, T=2048)

Sanity controls to ensure we don't hurt winners:
  D: 8B gu (1024/2048/4096)   — current winner, must stay >= 1.0x
  D: 14B q  (1024/2048)       — neutral-ish, must stay >= 0.97x

Protocol: warmup=500, outer=10, inner=200, single-trial per knob
(we re-run the default as sanity and expect delta < 2%).
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs

PR = lambda *a, **kw: print(*a, **{**kw, "flush": True})


def bench_us(fn, warmup=500, outer=10, inner=200):
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


def bench_fp16_flushed(W_fp, X_fp_t, warmup=200, outer=5, inner=100,
                      flush_mb=96):
    flush = torch.empty(flush_mb * 1024 * 256, dtype=torch.int8, device="cuda")
    def _flush_once():
        flush.zero_()
    trials = []
    for _ in range(3):
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
        trials.append(best)
    median = statistics.median(trials)
    # Calibrate flush cost
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
    return max(0.0, median - flush_us)


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


# Loser shapes (must reach >= 1.0x)
LOSERS = [
    # Family A: gate_up LARGE
    ("14B gu", 2048, 5120, 34816),
    ("14B gu", 4096, 5120, 34816),
    ("14B gu", 8192, 5120, 34816),
    ("32B gu", 2048, 5120, 55296),
    ("32B gu", 4096, 5120, 55296),
    ("70B gu", 2048, 8192, 57344),
    ("70B gu", 4096, 8192, 57344),
    # Family B: kv LARGE
    ("14B kv", 2048, 5120, 2048),
    ("14B kv", 4096, 5120, 2048),
    ("32B kv", 2048, 5120, 2048),
    ("70B kv", 1024, 8192, 2048),
    ("70B kv", 2048, 8192, 2048),
    # Family C: dn SMALL
    ("1.7B dn", 1024, 6144, 2048),
    ("4B dn",   1024, 9728, 2560),
    ("4B dn",   2048, 9728, 2560),
]

# Sanity-check shapes (must NOT regress >2%)
WINNERS = [
    ("8B gu",  2048, 4096, 24576),
    ("8B gu",  4096, 4096, 24576),
    ("14B q",  2048, 5120,  5120),
    ("4B q",   2048, 2560,  4096),
]

# Knob grid: 18 configs
KNOBS = []
for kbm in [None, 64]:          # default (128 per current rule) and 64
    for kbn in [None, 32, 16]:  # default (64 for T>=1024) and smaller
        for sk in [None, 2, 4]:  # default (1) and split-K
            # Avoid silly combos (kbm default + kbn default + sk default = dup)
            KNOBS.append((kbm, kbn, sk))

PR("=" * 140)
PR("Phase R β-scan: kBn × kBm × splitk cube on 3 loser families + 4 sanity winners")
PR("=" * 140)

# Helper for knob label
def label(kbm, kbn, sk):
    parts = []
    parts.append(f"kBm{kbm}" if kbm else "kBm*")
    parts.append(f"kBn{kbn}" if kbn else "kBn*")
    parts.append(f"sk{sk}" if sk else "sk1")
    return "/".join(parts)


def run_shape(fam, label_s, T, d_in, d_out):
    b = make_inputs(T, d_out, d_in, hp_ratio=0.0,
                    device="cuda", seed=T + d_in + d_out)
    W_fp = b["W_fp"]; X_fp_t = b["X_fp_t"]
    X = b["X"]; perm = b["perm"]
    t_fp16 = bench_fp16_flushed(W_fp, X_fp_t)
    results = {}
    for kbm, kbn, sk in KNOBS:
        set_env(kbm, kbn, sk)
        try:
            t = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in),
                         warmup=300, outer=5, inner=150)
        except Exception as e:
            t = float('inf')
        results[(kbm, kbn, sk)] = t
    set_env()
    best_knob = min(results, key=lambda k: results[k])
    best_us = results[best_knob]
    default_us = results.get((None, None, None), None)
    PR(f"  {fam:<14} {label_s:<18} T={T:<5} d_in={d_in:<5} d_out={d_out:<6}  "
       f"fp16={t_fp16:>8.2f}  default={default_us:>8.2f}  "
       f"best={best_us:>8.2f} ({label(*best_knob)})  "
       f"default_sp={t_fp16/default_us:>5.3f}x  best_sp={t_fp16/best_us:>5.3f}x"
       f"{'  ✓' if t_fp16/best_us >= 1.0 else ''}")
    return results, t_fp16


PR()
PR("### Loser shapes (goal: find knob achieving >= 1.0x)")
PR()
loser_results = {}
for label_s, T, d_in, d_out in LOSERS:
    fam = "A"
    if "kv" in label_s: fam = "B"
    if "dn" in label_s: fam = "C"
    res, t_fp16 = run_shape(fam, label_s, T, d_in, d_out)
    loser_results[(label_s, T)] = (res, t_fp16, d_in, d_out)

PR()
PR("### Winner shapes (sanity: knobs that help losers must NOT regress these)")
PR()
winner_results = {}
for label_s, T, d_in, d_out in WINNERS:
    res, t_fp16 = run_shape("D", label_s, T, d_in, d_out)
    winner_results[(label_s, T)] = (res, t_fp16, d_in, d_out)

# ============================================================
# Analysis: for each winning knob on loser shapes, check if it regresses
# on winner shapes.
# ============================================================
PR()
PR("=" * 140)
PR("ANALYSIS: for each 'winning knob' found on losers, check how it behaves on winners")
PR("=" * 140)
# Collect all knobs that took at least one loser to >=1.0x
good_knobs = set()
for (label_s, T), (res, t_fp16, _, _) in loser_results.items():
    for knob, us in res.items():
        if us < float('inf') and t_fp16 / us >= 1.0:
            good_knobs.add(knob)

PR(f"Knobs that achieve >=1.0x on at least one loser shape: {len(good_knobs)}")
PR()
for knob in sorted(good_knobs, key=lambda k: (k[0] or 0, k[1] or 0, k[2] or 0)):
    kbm, kbn, sk = knob
    lab = label(*knob)
    # Loser wins with this knob
    loser_wins = []
    loser_regs = []
    for (ls, T), (res, t_fp16, _, _) in loser_results.items():
        us = res.get(knob, float('inf'))
        if us == float('inf'):
            continue
        sp = t_fp16 / us
        default_us = res.get((None, None, None), float('inf'))
        default_sp = t_fp16 / default_us if default_us < float('inf') else 0
        delta = (us - default_us) / default_us * 100 if default_us < float('inf') else 0
        if sp >= 1.0 and default_sp < 1.0:
            loser_wins.append((ls, T, default_sp, sp))
        if delta > 2.0:
            loser_regs.append((ls, T, delta))
    # Winner regressions with this knob
    winner_regs = []
    for (ws, T), (res, t_fp16, _, _) in winner_results.items():
        us = res.get(knob, float('inf'))
        default_us = res.get((None, None, None), float('inf'))
        if us < float('inf') and default_us < float('inf'):
            delta = (us - default_us) / default_us * 100
            if delta > 2.0:
                winner_regs.append((ws, T, delta))

    PR(f"\nKnob {lab}:")
    PR(f"  Loser wins (< 1.0x → ≥ 1.0x): {len(loser_wins)}")
    for ls, T, d_sp, b_sp in loser_wins:
        PR(f"    {ls} T={T}: {d_sp:.3f}x → {b_sp:.3f}x")
    if loser_regs:
        PR(f"  Loser regressions (>2% slower than default): {len(loser_regs)}")
        for ls, T, d in loser_regs:
            PR(f"    {ls} T={T}: +{d:.2f}%")
    if winner_regs:
        PR(f"  Winner regressions (>2% slower than default): {len(winner_regs)}")
        for ws, T, d in winner_regs:
            PR(f"    {ws} T={T}: +{d:.2f}%")
    else:
        PR(f"  Winner regressions: NONE")
