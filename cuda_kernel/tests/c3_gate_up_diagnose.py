"""C.3 — diagnose Qwen3-8B gate_up T=128 (4096 -> 24576).

Goal: understand WHY today this shape sits at 48% INT4 eff, then
sweep tuning axes (split_k, kBm, kBn, cache, cp_async) to find the
real ceiling.  Also cover the sibling Qwen3-14B gu T=128
(5120 -> 34816) which has the same pathology.

Shape facts:
  d_in = 4096,  d_out = 24576 (or 34816 for 14B)
  n_groups = 32  (14B: 40)
  n_cta_m at kBm=128 = 192  (14B: 272)
  T=128 means 4 N-tiles @ kBn=32, 2 @ kBn=64, 8 @ kBn=16
  Grid is ENORMOUS (192-272 M-tiles * 4-8 N-tiles) — already saturates all SMs many waves deep
  → wave-fill is not the bottleneck; per-CTA efficiency is.
"""
import os
import json
import statistics
import random
from pathlib import Path

import torch
import kernel.cuda_kernel.ops as ops

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "sweep_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bench_us(fn, warmup=500, outer=10, inner=200):
    # Project norm bmmiahpl: long warmup + multiple trials; we'll call
    # this via the trial-randomised driver below.
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
        us = s.elapsed_time(e) * 1000.0 / inner
        best = min(best, us)
    return best


# ----- configuration axes to sweep -----
# For each shape we want the Cartesian product of:
#   kBm       ∈ {64, 128}
#   kBn       ∈ {8, 16, 32, 64}
#   split_k   ∈ {1, 2, 4}   (limited by n_groups divisibility)
#   cache     ∈ {auto, force_on, force_off}
#   cp_async  (follow default — no env hook)

SHAPES = [
    dict(model="Qwen3-8B",  proj="gate_up", T=128, d_in=4096, d_out=24576),
    dict(model="Qwen3-14B", proj="gate_up", T=128, d_in=5120, d_out=34816),
    # For context comparisons:
    dict(model="Qwen3-4B",  proj="gate_up", T=128, d_in=2560, d_out=18432),  # smaller variant
    dict(model="Qwen3-8B",  proj="gate_up", T=512, d_in=4096, d_out=24576),  # bigger T
]


def set_env(kbm=None, kbn=None, split_k=None, cache=None):
    for var in ("HKUST_V9_FUSED_FORCE_KBM", "HKUST_V9_FUSED_FORCE_KBN",
                "HKUST_V9_FUSED_FORCE_SPLITK", "HKUST_V9_FUSED_FORCE_CACHE"):
        os.environ.pop(var, None)
    if kbm is not None:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = str(kbm)
    if kbn is not None:
        os.environ["HKUST_V9_FUSED_FORCE_KBN"] = str(kbn)
    if split_k is not None:
        os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(split_k)
    if cache is not None:
        os.environ["HKUST_V9_FUSED_FORCE_CACHE"] = str(cache)


def prep_shape(sh):
    dev = torch.device("cuda:0")
    T, d_in, d_out = sh["T"], sh["d_in"], sh["d_out"]
    torch.manual_seed(0)
    X = torch.randn(T, d_in, dtype=torch.float16, device=dev) * 0.1
    perm = torch.randperm(d_in, device=dev).to(torch.int32)
    W_low = torch.randint(0, 16, (d_out, d_in // 2), dtype=torch.int8, device=dev)
    n_g = d_in // 128
    scale_u4 = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 0.01 + 0.001).contiguous()
    zero_u4  = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 14.0).contiguous()
    empty_hpb = torch.zeros((0, 128, 64), dtype=torch.int8, device=dev)
    hp_ro = torch.zeros((d_out // 128) + 1, dtype=torch.int32, device=dev)
    hp_ci = torch.zeros(0, dtype=torch.int32, device=dev)
    X_s4, scale_x, sum_X = ops.activation_quant_cuda(X, perm)
    def run():
        ops.fused_dense_sparse_cuda_int4(
            W_low, empty_hpb, hp_ro, hp_ci,
            X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
        )
    return run


def axis_sweep(sh, axis_configs, label):
    """Run every config in axis_configs.  Randomised-trial median."""
    run = prep_shape(sh)
    N_TRIALS = 3
    results = {}
    plan = [(name, t) for name in axis_configs for t in range(N_TRIALS)]
    random.Random(20260430).shuffle(plan)
    acc = {name: [] for name in axis_configs}
    for name, _ in plan:
        set_env(**axis_configs[name])
        us = bench_us(run, warmup=200, outer=5, inner=150)
        acc[name].append(us)
    set_env()  # reset
    for name in axis_configs:
        results[name] = statistics.median(acc[name])
    auto_us = results.get("auto", None)
    print(f"\n## {label}  —  shape={sh}")
    print(f"  {'config':<18}  {'us':>7}  {'Δ vs auto':>10}")
    for name, us in sorted(results.items(), key=lambda kv: kv[1]):
        delta = (us - auto_us) / auto_us * 100 if auto_us else 0
        tag = " ←".ljust(3) if us == min(results.values()) else ""
        print(f"  {name:<18}  {us:>7.2f}  {delta:>+9.1f}%  {tag}")
    return results


def main():
    for sh in SHAPES:
        print(f"\n=== SHAPE {sh['model']} {sh['proj']} T={sh['T']} "
              f"({sh['d_in']}→{sh['d_out']}) ===")

        # Axis 1 — kBm
        axis_sweep(sh, {
            "auto":       dict(),
            "kBm=64":     dict(kbm=64),
            "kBm=128":    dict(kbm=128),
        }, "A1 kBm")

        # Axis 2 — kBn × cache
        axis_sweep(sh, {
            "auto":                 dict(),
            "kBn=16":               dict(kbn=16),
            "kBn=32":               dict(kbn=32),
            "kBn=64":               dict(kbn=64),
            "kBn=64+cache_on":      dict(kbn=64, cache=1),
            "kBn=32+cache_on":      dict(kbn=32, cache=1),
        }, "A2 kBn + cache")

        # Axis 3 — split_k (n_groups=32 divisible by 1/2/4/8; 40 by 1/2/4/5/8/10)
        n_g = sh["d_in"] // 128
        sk_options = [1, 2, 4]
        if n_g % 8 == 0:
            sk_options.append(8)
        cfgs = {"auto": dict()}
        for sk in sk_options:
            cfgs[f"sk={sk}"] = dict(split_k=sk)
        axis_sweep(sh, cfgs, "A3 split_k")

        # Axis 4 — kBm × kBn joint (small grid)
        axis_sweep(sh, {
            "auto":                 dict(),
            "kBm=128,kBn=32":       dict(kbm=128, kbn=32),
            "kBm=128,kBn=64":       dict(kbm=128, kbn=64),
            "kBm=64, kBn=64":       dict(kbm=64,  kbn=64),
            "kBm=64, kBn=32":       dict(kbm=64,  kbn=32),
        }, "A4 (kBm, kBn) joint")


if __name__ == "__main__":
    main()
