"""C.2 — kBn sweep with trial-randomised in-process timing.

Per memory bmmiahpl: single-point in-process timings can have 20-50%
outlier rounds from transients.  Mitigation: run each (shape, mode)
as K=5 alternating trials in randomised order, take median over the
5 trials' bests.

Much faster than subprocess (~2 min vs 40 min), and the
randomisation breaks the "cache/clock hot from previous mode"
correlation that fooled the v1 sweep.
"""
import os
import json
import random
import statistics
from pathlib import Path

import torch
import kernel.cuda_kernel.ops as ops

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "sweep_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bench_us(fn, warmup=300, outer=10, inner=150):
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


SHAPES = []
def add(model, proj, T, d_in, d_out):
    SHAPES.append(dict(model=model, proj=proj, T=T, d_in=d_in, d_out=d_out))

for T in (32, 128):
    add("Qwen3-0.6B", "q",  T, 1024, 2048)
    add("Qwen3-0.6B", "o",  T, 2048, 1024)
    add("Qwen3-0.6B", "gu", T, 1024, 6144)
    add("Qwen3-0.6B", "dn", T, 3072, 1024)
    add("Qwen3-1.7B", "q",  T, 2048, 2048)
    add("Qwen3-1.7B", "gu", T, 2048, 12288)
    add("Qwen3-1.7B", "dn", T, 6144, 2048)
    add("Qwen3-4B",   "q",  T, 2560, 4096)
    add("Qwen3-4B",   "gu", T, 2560, 18432)
    add("Qwen3-4B",   "dn", T, 9216, 2560)
    add("Qwen3-8B",   "q",  T, 4096, 4096)
    add("Qwen3-8B",   "kv", T, 4096, 2048)
    add("Qwen3-8B",   "gu", T, 4096, 24576)
    add("Qwen3-8B",   "dn", T, 14336, 4096)
    add("Qwen3-14B",  "q",  T, 5120, 5120)
    add("Qwen3-14B",  "gu", T, 5120, 34816)
    add("Qwen3-14B",  "dn", T, 17408, 5120)

MODES = [("auto", None), ("k8", "8"), ("k16", "16"), ("k32", "32"), ("k64", "64")]
N_TRIALS = 5


def set_mode(mode):
    env_val = dict(MODES)[mode]
    if env_val is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBN", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBN"] = env_val


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


def main():
    rng = random.Random(20260430)
    rows = []
    header = (f"{'model':<10} {'proj':<4} {'T':>4} {'d_in':>5} {'d_out':>6} " +
              " ".join(f"{lbl:>7}" for lbl, _ in MODES) +
              f"  {'best':>6} {'gain%':>6}")
    print(header)
    print("-" * len(header))
    for i, sh in enumerate(SHAPES):
        run = prep_shape(sh)

        # Collect N_TRIALS timings per mode in INTERLEAVED random order.
        per_mode = {lbl: [] for lbl, _ in MODES}
        plan = [(lbl, t) for lbl, _ in MODES for t in range(N_TRIALS)]
        rng.shuffle(plan)
        for lbl, _ in plan:
            set_mode(lbl)
            us = bench_us(run, warmup=200, outer=6, inner=150)
            per_mode[lbl].append(us)

        us_by_mode = {lbl: statistics.median(per_mode[lbl]) for lbl, _ in MODES}
        auto_us = us_by_mode["auto"]
        forced = {k: v for k, v in us_by_mode.items() if k != "auto"}
        best_mode = min(forced, key=lambda k: forced[k])
        gain = (auto_us - forced[best_mode]) / auto_us * 100
        row = {**sh, "us_by_mode": us_by_mode,
               "per_trial_medians": {k: per_mode[k] for k in per_mode},
               "best_forced": best_mode, "gain_pct_vs_auto": gain}
        rows.append(row)
        print(f"{sh['model']:<10} {sh['proj']:<4} {sh['T']:>4} {sh['d_in']:>5} {sh['d_out']:>6} " +
              " ".join(f"{us_by_mode[lbl]:>7.2f}" for lbl, _ in MODES) +
              f"  {best_mode:>6} {gain:>+5.1f}%", flush=True)

    out_path = OUT_DIR / "c2_kbn_sweep_trials.json"
    out_path.write_text(json.dumps({"rows": rows, "n": len(rows)}, indent=2))
    print(f"\nWrote {out_path}")

    # Summary
    wins_any = [r for r in rows if r["gain_pct_vs_auto"] > 3.0]
    print(f"\nShapes where any forced mode beats auto by >3%: {len(wins_any)}")
    for r in sorted(wins_any, key=lambda r: -r["gain_pct_vs_auto"]):
        print(f"  {r['model']:<10} {r['proj']:<4} T={r['T']:>3} d=({r['d_in']},{r['d_out']}) "
              f"best={r['best_forced']} gain {r['gain_pct_vs_auto']:+.1f}%")


if __name__ == "__main__":
    main()
