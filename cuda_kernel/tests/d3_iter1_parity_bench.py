"""D.3 Iter1 parity — compare kInterleaveFold=false vs =true on 10 shapes.

Launch twice per shape: once with INTERLEAVE=0 (r66 bit-exact fold), once
with INTERLEAVE=1 (re-formed fmaf+nzs fold).  Under IEEE 754, fmaf is
MORE accurate than separate mul+sub (single rounding vs double), so the
new path's output will differ by ~1 ulp for some elements.  We accept
relative error < 5e-3 as a parity PASS.

Also logs perf side-by-side on 3 HFMA-critical shapes (Qwen3-8B gu T=128,
gu T=512, o_proj T=512) so we can see immediately if Iter1 produced any
speedup.
"""
import os
import statistics
import time

import torch
import kernel.cuda_kernel.ops as ops


dev = torch.device("cuda:0")


def prep(T, d_in, d_out, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(T, d_in, dtype=torch.float16, device=dev) * 0.1
    perm = torch.randperm(d_in, device=dev).to(torch.int32)
    W = torch.randint(0, 16, (d_out, d_in // 2), dtype=torch.int8, device=dev)
    n_g = d_in // 128
    su = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 0.01 + 0.001).contiguous()
    zu = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 14.0).contiguous()
    hpb = torch.zeros((0, 128, 64), dtype=torch.int8, device=dev)
    hpro = torch.zeros((d_out // 128) + 1, dtype=torch.int32, device=dev)
    hpci = torch.zeros(0, dtype=torch.int32, device=dev)
    X_s4, sx, sum_X = ops.activation_quant_cuda(X, perm)
    return W, hpb, hpro, hpci, X_s4, su, zu, sum_X, sx


def run(W, hpb, hpro, hpci, X_s4, su, zu, sum_X, sx, d_out, d_in):
    return ops.fused_dense_sparse_cuda_int4(
        W, hpb, hpro, hpci, X_s4, su, zu, sum_X, sx, d_out, d_in,
    )


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


PARITY_SHAPES = [
    (1,   4096, 4096,  "T=1   q_proj"),
    (32,  4096, 4096,  "T=32  q_proj"),
    (32,  4096, 24576, "T=32  gu"),
    (128, 4096, 4096,  "T=128 q_proj"),
    (128, 4096, 24576, "T=128 gu"),
    (512, 4096, 4096,  "T=512 q_proj"),
    (512, 4096, 24576, "T=512 gu"),
    (512, 12288, 4096, "T=512 dn"),
    (32,  14336, 4096, "T=32  14B-dn"),
    (128, 5120, 34816, "T=128 14B-gu"),
]

BENCH_SHAPES = [
    (128, 4096, 24576, "8B gu T=128"),
    (512, 4096, 24576, "8B gu T=512"),
    (512, 4096, 4096,  "8B o T=512"),
]


print("=" * 78)
print("D.3 Iter1 PARITY TEST")
print("=" * 78)
print(f"{'shape':<28}  {'max abs err':>12}  {'max rel err':>12}  {'verdict':>8}")

max_rel_overall = 0.0
pass_count = 0
for T, d_in, d_out, label in PARITY_SHAPES:
    inputs = prep(T, d_in, d_out, seed=T + d_in + d_out)

    os.environ["HKUST_V9_INTERLEAVE"] = "0"
    Y_base = run(*inputs, d_out, d_in).clone()

    os.environ["HKUST_V9_INTERLEAVE"] = "1"
    Y_new = run(*inputs, d_out, d_in).clone()

    diff = (Y_new.float() - Y_base.float()).abs()
    abs_err = float(diff.max())
    # Relative tolerance with a tiny epsilon to avoid division by zero.
    denom = Y_base.float().abs().clamp_min(1e-3)
    rel = (diff / denom).max()
    rel_err = float(rel)
    max_rel_overall = max(max_rel_overall, rel_err)
    verdict = "PASS" if rel_err < 5e-3 else "FAIL"
    if verdict == "PASS":
        pass_count += 1
    print(f"  {label:<28}  {abs_err:>12.6f}  {rel_err:>12.6f}  {verdict:>8}")

print()
print(f"OVERALL PARITY: {pass_count}/{len(PARITY_SHAPES)}  max rel err {max_rel_overall:.6f}")
print()
print("=" * 78)
print("D.3 Iter1 PERF A/B (interleaved trials)")
print("=" * 78)

for T, d_in, d_out, label in BENCH_SHAPES:
    inputs = prep(T, d_in, d_out, seed=T + d_in + d_out + 1)

    def _run():
        run(*inputs, d_out, d_in)

    # Interleaved trials: [base, new, base, new, ...] ×4
    bs, ns = [], []
    for _ in range(4):
        os.environ["HKUST_V9_INTERLEAVE"] = "0"
        bs.append(bench_us(_run))
        os.environ["HKUST_V9_INTERLEAVE"] = "1"
        ns.append(bench_us(_run))
    bm = statistics.median(bs)
    nm = statistics.median(ns)
    pct = (nm - bm) / bm * 100
    print(f"  {label:<20}  base={bm:>7.2f}us  new={nm:>7.2f}us  delta={pct:+6.2f}%")
