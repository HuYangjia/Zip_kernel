"""T=512 dispatcher probe (Step 2A).

Analogous to c_probe_loser_shapes.py but targeting T=512 where the
r67 bench shows median 0.87× and 25/35 losers.  The goal is to
identify whether any dispatcher-level knob (kBn, kBm, cache,
split_k, ldmatrix) unlocks a measurable win on the loser shapes.

Target groups (picked from r67 T=512 loser list in
logs/r67_c5/_dispatch_audit.py output):

  G1 Mid compute (d_out <= 4096, speedup 0.6-0.85):
    4B kv T=512 (2560, 2048)    sp=0.62
    8B kv T=512 (4096, 2048)    sp=0.79
    14B kv T=512 (5120, 2048)   sp=0.84
    32B kv T=512 (5120, 2048)   sp=0.83
    LLaMA-70B kv T=512 (8192, 2048) sp=0.88

  G2 Big d_out compute-bound (the hardest region, per D.1 analysis):
    14B gu T=512 (5120, 34816)  sp=0.79
    32B gu T=512 (5120, 55296)  sp=0.72
    LLaMA-70B gu T=512 (8192, 57344) sp=0.71

  G3 Big d_in down_proj:
    4B dn T=512 (9728, 2560)    sp=0.84
    14B dn T=512 (17408, 5120)  sp=0.80
    32B dn T=512 (27648, 5120)  sp=0.71
    70B dn T=512 (28672, 8192)  sp=1.01 (borderline)

  G4 Mid q/o T=512:
    14B q T=512 (5120, 5120)    sp=0.99
    70B q T=512 (8192, 8192)    sp=1.09

Protocol: warmup=500, outer=10, inner=200, 3 trials median
(per repo measurement discipline).
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
        bundle["W_low_packed"], bundle["W_high_packed"],
        bundle["hp_row_offsets"], bundle["hp_col_indices"],
        bundle["X_s4"], bundle["scale_u4"], bundle["zero_u4"],
        bundle["sum_X"], bundle["scale_x"], d_out, d_in,
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


# Targets — focused subset of the r67 T=512 loser list.
TARGETS = [
    # G1 Mid kv (d_out=2048)
    ("G1 4B kv",    2560,  2048),
    ("G1 8B kv",    4096,  2048),
    ("G1 14B kv",   5120,  2048),
    ("G1 70B kv",   8192,  2048),
    # G2 Big d_out
    ("G2 14B gu",   5120, 34816),
    ("G2 32B gu",   5120, 55296),
    ("G2 70B gu",   8192, 57344),
    # G3 Big d_in dn
    ("G3 4B dn",    9728,  2560),
    ("G3 14B dn",  17408,  5120),
    ("G3 32B dn",  27648,  5120),
    # G4 Mid q/o
    ("G4 14B q",    5120,  5120),
    ("G4 70B q",    8192,  8192),
]

# Config space: probe the key knobs the dispatcher can tune at T=512.
#   split_k ∈ {default, 1, 2, 4} — default rule sets sk=1 for T=512
#     but maybe we missed cases where sk=2/4 helps.
#   kBn     ∈ {default, 64, 32, 16}
#   kBm     ∈ {default, 64, 128}
#   cache   ∈ {default, 0, 1}
CONFIGS = [
    ("default",             None, None, None, None),
    ("sk=2",                None, None, None, "2"),
    ("sk=4",                None, None, None, "4"),
    ("kBm=64",              None, "64", None, None),
    ("kBm=64 sk=2",         None, "64", None, "2"),
    ("kBn=32",              "32", None, None, None),
    ("kBn=32 sk=2",         "32", None, None, "2"),
    ("kBn=16",              "16", None, None, None),
    ("cache=1",             None, None, "1",  None),
    ("cache=0",             None, None, "0",  None),
    ("kBm=64 cache=1",      None, "64", "1",  None),
    ("kBm=64 cache=0",      None, "64", "0",  None),
]


def apply_env(kbn, kbm, cache, sk):
    for k in ("HKUST_V9_FUSED_FORCE_KBN", "HKUST_V9_FUSED_FORCE_KBM",
              "HKUST_V9_FUSED_FORCE_CACHE", "HKUST_V9_FUSED_FORCE_SPLITK"):
        os.environ.pop(k, None)
    if kbn   is not None: os.environ["HKUST_V9_FUSED_FORCE_KBN"]    = kbn
    if kbm   is not None: os.environ["HKUST_V9_FUSED_FORCE_KBM"]    = kbm
    if cache is not None: os.environ["HKUST_V9_FUSED_FORCE_CACHE"]  = cache
    if sk    is not None: os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = sk


T = 512

print("=" * 96)
print(f"T=512 dispatcher probe — {len(TARGETS)} target shapes × {len(CONFIGS)} configs")
print("=" * 96)

all_results = []
for label, d_in, d_out in TARGETS:
    print()
    print(f"## {label}   d_in={d_in} d_out={d_out}   T={T}")
    b = prep(T, d_in, d_out, seed=T + d_in + d_out)

    def _run():
        run(b, d_out, d_in)

    us_by_cfg = {}
    for cfg_lbl, kbn, kbm, cache, sk in CONFIGS:
        apply_env(kbn, kbm, cache, sk)
        trials = [bench_us(_run, warmup=300, outer=5, inner=150) for _ in range(3)]
        us_by_cfg[cfg_lbl] = statistics.median(trials)

    default_us = us_by_cfg["default"]
    sorted_cfgs = sorted(us_by_cfg.items(), key=lambda kv: kv[1])
    print(f"  {'config':<24}  {'us':>9}  {'vs default':>11}")
    for cfg_lbl, us in sorted_cfgs:
        pct = (us - default_us) / default_us * 100
        marker = "  *" if cfg_lbl != "default" and pct < -2.0 else ""
        print(f"  {cfg_lbl:<24}  {us:>8.2f}  {pct:+10.2f}%{marker}")
    best_cfg, best_us = sorted_cfgs[0]
    uplift = (default_us - best_us) / default_us * 100
    if best_cfg != "default":
        print(f"  ==> BEST: '{best_cfg}', {best_us:.2f}us, {uplift:+.2f}% vs default")
    else:
        print(f"  ==> default is already best ({default_us:.2f}us)")

    all_results.append({
        "label": label, "d_in": d_in, "d_out": d_out,
        "default_us": default_us, "best_cfg": best_cfg, "best_us": best_us,
        "uplift": uplift, "all_us": us_by_cfg,
    })


print()
print("=" * 96)
print("SUMMARY")
print("=" * 96)
print(f"{'shape':<20}  {'default':>9}  {'best_cfg':<24} {'best us':>9}  {'uplift':>8}")
total_uplift = 0
wins = 0
for r in all_results:
    mark = "  WIN" if r["uplift"] > 2.0 and r["best_cfg"] != "default" else ""
    if mark:
        wins += 1
        total_uplift += r["uplift"]
    print(f"  {r['label']:<18}  {r['default_us']:>8.2f}  {r['best_cfg']:<24} "
          f"{r['best_us']:>8.2f}  {r['uplift']:+7.2f}%{mark}")
print()
print(f"Wins ≥2%: {wins}/{len(all_results)}  "
      f"avg uplift on wins: {total_uplift/max(wins,1):+.2f}%")

# Which config label wins most often?  Good signal for dispatcher rule.
from collections import Counter
winner_cfg_count = Counter(r["best_cfg"] for r in all_results if r["uplift"] > 2.0)
if winner_cfg_count:
    print()
    print("Winner config distribution:")
    for cfg, cnt in winner_cfg_count.most_common():
        print(f"  {cfg:<24}  {cnt} shape(s)")
