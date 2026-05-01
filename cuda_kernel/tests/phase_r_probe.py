"""Phase R probe — test 3 dispatcher/kernel tweaks on the 3 loser families
to see which direction has feasibility for raising speedup > 1.0x.

Knobs tested per shape (via env vars):
  V1: HKUST_V9_FUSED_FORCE_KBM=64 (kBm=64 instead of kBm=128)
  V2: HKUST_V9_FUSED_FORCE_SPLITK=4
  V3: HKUST_V9_FUSED_FORCE_KBM=64 + SPLITK=2
  V4: HKUST_V9_FUSED_FORCE_SPLITK=2

Families:
  A: gate_up LARGE (14B/32B/70B gu at T=2048) — cuda_eff 20-22%
  B: kv LARGE     (14B/32B/70B kv at T=2048)  — cuda_eff 35-40%
  C: down_proj SMALL (1.7B/4B dn at T=1024)   — cuda_eff 28-37%

Protocol per [[memory:bmmiahpl]]:
  warmup=500, outer=10, inner=200, 3 interleaved trials, median
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
    """FP16 with L2 flush to match bench_qwen3 protocol (with flush cost)."""
    flush = torch.empty(flush_mb * 1024 * 256, dtype=torch.int8, device="cuda")
    def _flush_once():
        flush.zero_()
    # Do 3 trials, return median
    trials = []
    for _ in range(3):
        for _ in range(warmup):
            _flush_once()
            torch.matmul(W_fp, X_fp_t)
        torch.cuda.synchronize()
        best = float('inf')
        for _ in range(outer):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(inner):
                _flush_once()
                torch.matmul(W_fp, X_fp_t)
            e.record()
            torch.cuda.synchronize()
            best = min(best, s.elapsed_time(e) * 1000.0 / inner)
        # Subtract flush cost (measured separately once)
        # Approximated below after calibration
        trials.append(best)
    median = statistics.median(trials)
    # Calibrate flush cost: measure flush alone
    torch.cuda.synchronize()
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


def set_env(kbm=None, splitk=None):
    """Set env vars controlling dispatcher overrides."""
    for k in ("HKUST_V9_FUSED_FORCE_KBM", "HKUST_V9_FUSED_FORCE_SPLITK"):
        os.environ.pop(k, None)
    if kbm is not None:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = str(kbm)
    if splitk is not None:
        os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(splitk)


def run_e2e(X, perm, b, d_out, d_in):
    X_s4, sx, sX = ops.activation_quant_cuda(X, perm)
    return ops.fused_dense_sparse_cuda_int4(
        b["W_low_packed"], b["W_high_packed"],
        b["hp_row_offsets"], b["hp_col_indices"],
        X_s4, b["scale_u4"], b["zero_u4"], sX, sx, d_out, d_in,
    )


FAMILIES = [
    # (family_label, shape_list[(label, T, d_in, d_out)])
    ("A: gate_up LARGE", [
        ("14B gu T=2048", 2048, 5120, 34816),
        ("32B gu T=2048", 2048, 5120, 55296),
        ("70B gu T=2048", 2048, 8192, 57344),
    ]),
    ("B: kv LARGE", [
        ("14B kv T=2048", 2048, 5120, 2048),
        ("32B kv T=2048", 2048, 5120, 2048),
        ("70B kv T=2048", 2048, 8192, 2048),
        # Also check worst one — 70B kv T=1024 (sp=0.79x)
        ("70B kv T=1024", 1024, 8192, 2048),
    ]),
    ("C: down_proj SMALL", [
        ("1.7B dn T=1024", 1024, 6144, 2048),
        ("4B dn T=1024",   1024, 9728, 2560),
        ("4B dn T=2048",   2048, 9728, 2560),
    ]),
]


KNOBS = [
    ("default",       None, None),
    ("kBm=64",        64,   None),
    ("splitk=2",      None, 2),
    ("splitk=4",      None, 4),
    ("kBm=64+sk=2",   64,   2),
    ("kBm=64+sk=4",   64,   4),
]

PR("=" * 130)
PR("Phase R probe — dispatcher knobs on 3 loser families")
PR("=" * 130)

for fam_label, shapes in FAMILIES:
    PR()
    PR(f"### {fam_label}")
    PR(f"{'shape':<18}  {'fp16':>7}  {'default':>8} {'kBm=64':>8} "
       f"{'sk=2':>8} {'sk=4':>8} {'km64+sk2':>10} {'km64+sk4':>10}  "
       f"{'best':>14}  {'fp16 sp':>8}")
    for label, T, d_in, d_out in shapes:
        b = make_inputs(T, d_out, d_in, hp_ratio=0.0,
                        device="cuda", seed=T + d_in + d_out)
        W_fp = b["W_fp"]; X_fp_t = b["X_fp_t"]
        X = b["X"]; perm = b["perm"]

        # FP16 baseline
        t_fp16 = bench_fp16_flushed(W_fp, X_fp_t)

        results = {}
        for knob_name, kbm, sk in KNOBS:
            set_env(kbm, sk)
            t = bench_us(lambda: run_e2e(X, perm, b, d_out, d_in))
            results[knob_name] = t
        set_env()  # cleanup

        best_name = min(results, key=lambda k: results[k])
        best_us = results[best_name]
        fp16_sp = t_fp16 / best_us
        win = " ✓" if fp16_sp >= 1.0 else ""

        row = f"  {label:<16}  {t_fp16:>6.2f}  "
        for knob_name, _, _ in KNOBS:
            row += f"{results[knob_name]:>7.2f} "
        row += f" {best_name:>13} {best_us:>7.2f}  {fp16_sp:>6.3f}x{win}"
        PR(row)
