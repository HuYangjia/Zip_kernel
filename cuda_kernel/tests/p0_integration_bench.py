"""P0 integration in-process A/B bench.

Compares:
  path L (legacy):  activation_quant_cuda + fused_dense_sparse_cuda (two-step)
  path P (P0):      fused_dense_sparse_e2e_cuda with P0 enabled

On the shapes where P0 is supported (T in [2,128], d_out % 128 == 0,
hp=0), we expect the P0 path to save the ~16us activation_quant
launch floor.

Per repo measurement discipline [[memory:bmmiahpl]]:
  warmup=500, outer=10, inner=200, 4 interleaved trials, median.
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs


dev = torch.device("cuda:0")


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


def path_legacy(inp, d_out, d_in):
    """Two-step: activation_quant + fused_dense_sparse.  Matches what
    bench_qwen3_shapes currently benches under the cuda_e2e column."""
    X_s4, sx, sX = ops.activation_quant_cuda(inp["X"], inp["perm"])
    return ops.fused_dense_sparse_cuda(
        inp["W_low_packed"], inp["W_high_packed"],
        inp["hp_row_offsets"], inp["hp_col_indices"],
        X_s4, inp["scale_u4"], inp["zero_u4"],
        sX, sx, d_out, d_in,
    )


def path_p0(inp, d_out, d_in):
    """P0: fp16 in, fused quant + MMA in one launch."""
    return ops.fused_dense_sparse_e2e_cuda(
        inp["X"], inp["perm"],
        inp["W_low_packed"], inp["W_high_packed"],
        inp["hp_row_offsets"], inp["hp_col_indices"],
        inp["scale_u4"], inp["zero_u4"],
        d_out, d_in,
    )


# Targets: shapes where P0 is expected to help (T in [2,128], hp=0).
# Focus on previously-identified P3b (T=32) and P3d (T=128) losers.
TARGETS_HELP = [
    # (model, T, d_in, d_out)
    ("0.6B q_proj T=32",     32,  1024, 2048),
    ("0.6B o_proj T=32",     32,  2048, 1024),
    ("1.7B q_proj T=32",     32,  2048, 2048),
    ("1.7B kv_proj T=32",    32,  2048, 2048),
    ("4B kv_proj T=32",      32,  2560, 2048),
    ("8B kv_proj T=32",      32,  4096, 2048),
    ("8B q_proj T=32",       32,  4096, 4096),
    ("8B gu T=32",           32,  4096, 24576),
    # T=128
    ("0.6B q_proj T=128",   128,  1024, 2048),
    ("1.7B q_proj T=128",   128,  2048, 2048),
    ("4B q_proj T=128",     128,  2560, 4096),
    ("4B o_proj T=128",     128,  4096, 2560),
    ("8B q_proj T=128",     128,  4096, 4096),
    ("8B gu T=128",         128,  4096, 24576),
    ("14B q_proj T=128",    128,  5120, 5120),
    ("14B gu T=128",        128,  5120, 34816),
]

# Guards: shapes where P0 should NOT be used (T=512, hp!=0, etc.).
# Just verify that the e2e path selects legacy here and doesn't regress.
TARGETS_GUARD = [
    # T=512 → should fall back to legacy
    ("8B q_proj T=512",      512, 4096, 4096),
    ("8B gu T=512",          512, 4096, 24576),
    # hp=0.05 — still supported for P0 IF P0.3 adds sparse, but right
    # now P0 gate should refuse hp>0 and fall back to legacy.
    # Note: make_inputs uses hp_ratio=0.05 by default already; we test
    # P0 support under hp=0 separately below.
]


def run_shape(label, T, d_in, d_out, use_hp):
    hp = 0.05 if use_hp else 0.0
    inp = make_inputs(T, d_out, d_in, hp_ratio=hp, device="cuda",
                      seed=T + d_in + d_out)

    # Parity check first
    os.environ["HKUST_V9_P0_MODE"] = "0"
    Y_p0 = path_p0(inp, d_out, d_in)
    Y_leg = path_legacy(inp, d_out, d_in)
    diff = (Y_p0.float() - Y_leg.float()).abs()
    mad = float(diff.max())
    rel = float((diff / (Y_leg.float().abs() + 1e-6)).max())
    if mad >= 2e-3 and rel >= 0.05:
        return {"label": label, "T": T, "d_in": d_in, "d_out": d_out,
                "hp": hp, "parity": f"FAIL mad={mad:.4g} rel={rel:.4g}",
                "leg_us": 0, "p0_us": 0, "delta_pct": 0}

    # A/B bench: 4 interleaved trials
    legs, p0s = [], []
    for _ in range(4):
        legs.append(bench_us(lambda: path_legacy(inp, d_out, d_in)))
        p0s.append(bench_us(lambda: path_p0(inp, d_out, d_in)))
    leg = statistics.median(legs)
    p0_ = statistics.median(p0s)
    delta = (p0_ - leg) / leg * 100
    return {"label": label, "T": T, "d_in": d_in, "d_out": d_out,
            "hp": hp, "parity": "PASS",
            "leg_us": leg, "p0_us": p0_, "delta_pct": delta}


print("=" * 90)
print("P0 integration A/B: dense (hp=0), where P0 is active")
print("=" * 90)
print(f"{'shape':<28} {'legacy us':>10} {'P0 us':>9} {'delta':>9} {'parity':>8}")
p0_gains = []
for label, T, d_in, d_out in TARGETS_HELP:
    r = run_shape(label, T, d_in, d_out, use_hp=False)
    mark = ""
    if r["parity"] == "PASS":
        if r["delta_pct"] < -5:
            p0_gains.append(-r["delta_pct"])
            mark = "  WIN"
        elif r["delta_pct"] > 5:
            mark = "  REGRESS"
    print(f"  {r['label']:<26} {r['leg_us']:>9.2f}  {r['p0_us']:>8.2f}  "
          f"{r['delta_pct']:+8.2f}% {r['parity']:>8}{mark}")

print()
print("=" * 90)
print("P0 integration A/B: hp=0.05 (P0 gate should refuse, legacy path on both)")
print("=" * 90)
print(f"{'shape':<28} {'legacy us':>10} {'P0 us':>9} {'delta':>9} {'parity':>8}")
for label, T, d_in, d_out in [
    ("8B q_proj T=32 hp=5%",   32, 4096, 4096),
    ("8B q_proj T=128 hp=5%", 128, 4096, 4096),
]:
    r = run_shape(label, T, d_in, d_out, use_hp=True)
    mark = "  OK(expected same)" if abs(r["delta_pct"]) < 3 else "  UNEXPECTED"
    print(f"  {r['label']:<26} {r['leg_us']:>9.2f}  {r['p0_us']:>8.2f}  "
          f"{r['delta_pct']:+8.2f}% {r['parity']:>8}{mark}")

print()
print("=" * 90)
print("SUMMARY")
print("=" * 90)
if p0_gains:
    print(f"P0 wins ≥5%: {len(p0_gains)}/{len(TARGETS_HELP)}  "
          f"median uplift: {statistics.median(p0_gains):+.2f}%  "
          f"mean: {statistics.mean(p0_gains):+.2f}%")
else:
    print("P0 did not achieve ≥5% win on any shape")
