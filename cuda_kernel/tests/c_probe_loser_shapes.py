"""C-probe v2: scan kBn × kBm × cache config with REALISTIC hp_ratio=0.05.

v1 used hp=0 (dense only) which does NOT match r66 bench conditions
(bench_qwen3_shapes.py uses hp_ratio=0.05).  r66's dispatcher was tuned
against hp=0.05 and R52 gate comment explicitly documents
"d_out=4096 at T=128 with kBm=64 is 0.95x at hp=0.05".  Re-probe at
hp=0.05 to get actionable dispatcher-tuning data.
"""
import os
import statistics

import torch
import kernel.cuda_kernel.ops as ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs


dev = torch.device("cuda:0")


def prep(T, d_in, d_out, hp_ratio=0.05, seed=0):
    return make_inputs(T, d_out, d_in, hp_ratio=hp_ratio, device="cuda", seed=seed)


def run(bundle, d_out, d_in):
    return ops.fused_dense_sparse_cuda_int4(
        bundle["W_low_packed"],
        bundle["W_high_packed"],
        bundle["hp_row_offsets"],
        bundle["hp_col_indices"],
        bundle["X_s4"],
        bundle["scale_u4"],
        bundle["zero_u4"],
        bundle["sum_X"],
        bundle["scale_x"],
        d_out,
        d_in,
    )


def bench_us(fn, warmup=300, outer=5, inner=100):
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


TARGETS = [
    (128, 4096, 4096,  "8B q_proj T=128",  0.91),
    (128, 4096, 4096,  "8B o_proj T=128",  0.91),
    (128, 2560, 4096,  "4B q_proj T=128",  0.79),
    (128, 4096, 2560,  "4B o_proj T=128",  0.70),
    (128, 6144, 2048,  "1.7B down T=128",  0.81),
    (128, 2048, 2048,  "1.7B q_proj T=128", 0.39),
    (128, 2560, 2048,  "4B kv_proj T=128", 0.45),
]

CONFIGS = [
    ("default",             None, None, None),
    ("kBn=64",              "64", None, None),
    ("kBn=32",              "32", None, None),
    ("kBn=16",              "16", None, None),
    ("kBm=64",              None, "64", None),
    ("kBm=64 kBn=32",       "32", "64", None),
    ("kBm=64 kBn=16",       "16", "64", None),
    ("cache=1",             None, None, "1"),
    ("cache=0",             None, None, "0"),
    ("kBm=64 cache=1",      None, "64", "1"),
    ("kBm=64 cache=0",      None, "64", "0"),
    ("kBm=128 kBn=32 cache=1", "32", "128", "1"),
]


def apply_env(kbn, kbm, cache):
    os.environ.pop("HKUST_V9_FUSED_FORCE_KBN", None)
    os.environ.pop("HKUST_V9_FUSED_FORCE_KBM", None)
    os.environ.pop("HKUST_V9_FUSED_FORCE_CACHE", None)
    if kbn is not None:
        os.environ["HKUST_V9_FUSED_FORCE_KBN"] = kbn
    if kbm is not None:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = kbm
    if cache is not None:
        os.environ["HKUST_V9_FUSED_FORCE_CACHE"] = cache


print("=" * 90)
print("C-probe v2 (hp=0.05): dispatcher scan for T=128 loser shapes")
print("=" * 90)

all_results = []

for T, d_in, d_out, label, r66_sp in TARGETS:
    print()
    print(f"## {label}  d_in={d_in} d_out={d_out}  r66 sp={r66_sp:.2f}x")
    bundle = prep(T, d_in, d_out)

    def _run():
        run(bundle, d_out, d_in)

    config_us = {}
    for cfg_label, kbn, kbm, cache in CONFIGS:
        apply_env(kbn, kbm, cache)
        trials = [bench_us(_run) for _ in range(3)]
        config_us[cfg_label] = statistics.median(trials)

    default_us = config_us["default"]
    sorted_cfgs = sorted(config_us.items(), key=lambda kv: kv[1])

    print(f"  {'config':<26}  {'us':>8}  {'vs default':>11}")
    for cfg_label, us in sorted_cfgs:
        pct = (us - default_us) / default_us * 100
        marker = "  *" if cfg_label != "default" and pct < -2.0 else ""
        print(f"  {cfg_label:<26}  {us:>7.2f}  {pct:+10.2f}%{marker}")

    best_cfg, best_us = sorted_cfgs[0]
    if best_cfg != "default":
        uplift = (default_us - best_us) / default_us * 100
        print(f"  ==> BEST: '{best_cfg}', {best_us:.2f}us, {uplift:+.2f}% vs default")
    else:
        print(f"  ==> No non-default config beats default ({default_us:.2f}us)")

    all_results.append({
        "label": label, "T": T, "d_in": d_in, "d_out": d_out,
        "default_us": default_us,
        "best_cfg": best_cfg, "best_us": best_us,
    })

print()
print("=" * 90)
print("SUMMARY (hp=0.05)")
print("=" * 90)
print(f"{'shape':<24}  {'default_us':>10}  {'best_cfg':<28} {'best_us':>8}  {'uplift':>8}")
total_uplift = 0
wins = 0
for r in all_results:
    upl = (r["default_us"] - r["best_us"]) / r["default_us"] * 100
    mark = "  WIN" if upl > 2.0 and r["best_cfg"] != "default" else ""
    if mark:
        wins += 1
        total_uplift += upl
    print(f"  {r['label']:<24}  {r['default_us']:>9.2f}  {r['best_cfg']:<28} "
          f"{r['best_us']:>8.2f}  {upl:+7.2f}%{mark}")
print()
print(f"Wins: {wins}/{len(all_results)}  avg uplift on wins: "
      f"{total_uplift/max(wins,1):+.2f}%")
