"""C-probe: scan kBn × kBm × cache config for r66 loser shapes.

Targets: shapes where r66 speedup < 1.0× at T=128 (most promising for
dispatcher-level rescue; the T=512 losers are mostly architectural and
not dispatcher-fixable per Phase 4 analysis).

For each target shape, measure cuda_us under:
  - default dispatcher (no env overrides) — our baseline
  - all combinations of kBn ∈ {8, 16, 32, 64} × kBm ∈ {default, 64, 128}
    × cache ∈ {default, 0, 1}

Report: best config per shape, uplift vs default.  If any shape has
>= +5% uplift from a non-default config, we have a dispatcher win.
"""
import os
import statistics

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


def run(bundle, d_out, d_in):
    W, hpb, hpro, hpci, X_s4, su, zu, sum_X, sx = bundle
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


# Target shapes: r66 T=128 losers where dispatcher might be wrong
TARGETS = [
    (128, 4096, 4096,  "8B q_proj T=128",  0.91),
    (128, 4096, 4096,  "8B o_proj T=128",  0.91),  # same as q but keep as separate probe
    (128, 2560, 4096,  "4B q_proj T=128",  0.79),
    (128, 4096, 2560,  "4B o_proj T=128",  0.70),
    (128, 6144, 2048,  "1.7B down T=128",  0.81),
    (128, 2048, 2048,  "1.7B q_proj T=128", 0.39),  # extreme loser
    (128, 2560, 2048,  "4B kv_proj T=128", 0.45),
]

# Config space
CONFIGS = [
    # (label, kbn_env, kbm_env, cache_env)
    ("default",     None, None, None),
    ("kBn=64",      "64", None, None),
    ("kBn=32",      "32", None, None),
    ("kBn=16",      "16", None, None),
    ("kBn=8",        "8", None, None),
    ("kBm=64",      None, "64", None),
    ("kBm=64 kBn=32", "32","64", None),
    ("kBm=64 kBn=16", "16","64", None),
    ("cache=1",     None, None, "1"),
    ("cache=0",     None, None, "0"),
    ("kBn=16 cache=1",  "16", None, "1"),
    ("kBn=32 cache=1",  "32", None, "1"),
    ("kBm=64 kBn=32 cache=1", "32", "64", "1"),
]


def apply_env(label, kbn, kbm, cache):
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
print("C-probe: dispatcher config scan for T=128 loser shapes")
print("=" * 90)

all_results = []

for T, d_in, d_out, label, r66_sp in TARGETS:
    print()
    print(f"## {label}  d_in={d_in} d_out={d_out}  r66 sp={r66_sp:.2f}x")
    bundle = prep(T, d_in, d_out, seed=T + d_in + d_out)

    def _run():
        run(bundle, d_out, d_in)

    # Per config: 3 independent trials, median
    config_us = {}
    for cfg_label, kbn, kbm, cache in CONFIGS:
        apply_env(cfg_label, kbn, kbm, cache)
        trials = [bench_us(_run, warmup=300, outer=5, inner=100) for _ in range(3)]
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

# Summary
print()
print("=" * 90)
print("SUMMARY")
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
