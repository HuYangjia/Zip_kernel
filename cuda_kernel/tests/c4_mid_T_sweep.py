"""C.4 — measure T ∈ {48, 64, 96} holes.

The r62 F2 dispatcher has explicit gates for T ∈ {48, 64, 96} but
we never measured these T values in the 140-shape bench.  This
script covers them on the Qwen3-8B representative shapes, which is
the model family with the best speedup behaviour.

Output: per-shape auto timing + comparison with the R44 demote
bands, to confirm the existing gates are still calibrated (or
surface new ones).
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


def bench_us(fn, warmup=300, outer=6, inner=150):
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


# Qwen3-8B shapes at the unmeasured T values.
SHAPES = []
for T in (48, 64, 96):
    SHAPES += [
        dict(model="Qwen3-8B", proj="q",  T=T, d_in=4096, d_out=4096),
        dict(model="Qwen3-8B", proj="kv", T=T, d_in=4096, d_out=2048),
        dict(model="Qwen3-8B", proj="o",  T=T, d_in=4096, d_out=4096),
        dict(model="Qwen3-8B", proj="gu", T=T, d_in=4096, d_out=24576),
        dict(model="Qwen3-8B", proj="dn", T=T, d_in=14336, d_out=4096),
    ]


def set_env(kbm=None, kbn=None, cache=None):
    for v in ("HKUST_V9_FUSED_FORCE_KBM", "HKUST_V9_FUSED_FORCE_KBN",
              "HKUST_V9_FUSED_FORCE_CACHE"):
        os.environ.pop(v, None)
    if kbm is not None:
        os.environ["HKUST_V9_FUSED_FORCE_KBM"] = str(kbm)
    if kbn is not None:
        os.environ["HKUST_V9_FUSED_FORCE_KBN"] = str(kbn)
    if cache is not None:
        os.environ["HKUST_V9_FUSED_FORCE_CACHE"] = str(cache)


def prep(sh):
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


MODES = {
    "auto":     dict(),
    "kBm=128":  dict(kbm=128),
    "kBm=64":   dict(kbm=64),
    "k32":      dict(kbn=32),
    "k64":      dict(kbn=64),
    "k8":       dict(kbn=8),
}


def main():
    rng = random.Random(20260430)
    rows = []
    print(f"{'model':<10} {'proj':<4} {'T':>3} {'d_in':>5} {'d_out':>6}  " +
          " ".join(f"{k:>8}" for k in MODES) +
          f"  {'best':>6} {'gain%':>6}")
    for sh in SHAPES:
        run = prep(sh)
        N = 3
        per = {k: [] for k in MODES}
        plan = [(k, i) for k in MODES for i in range(N)]
        rng.shuffle(plan)
        for k, _ in plan:
            set_env(**MODES[k])
            per[k].append(bench_us(run))
        set_env()
        med = {k: statistics.median(per[k]) for k in MODES}
        auto_us = med["auto"]
        best = min((k for k in MODES if k != "auto"), key=lambda k: med[k])
        gain = (auto_us - med[best]) / auto_us * 100
        rows.append({**sh, "us": med, "best": best, "gain_pct_vs_auto": gain})
        print(f"{sh['model']:<10} {sh['proj']:<4} {sh['T']:>3} {sh['d_in']:>5} {sh['d_out']:>6}  " +
              " ".join(f"{med[k]:>8.2f}" for k in MODES) +
              f"  {best:>6} {gain:>+5.1f}%", flush=True)

    (OUT_DIR / "c4_mid_T_sweep.json").write_text(json.dumps({"rows": rows}, indent=2))
    print()
    flagged = [r for r in rows if r["gain_pct_vs_auto"] > 3.0]
    print(f"Shapes where any forced mode beats auto >3%: {len(flagged)}")
    for r in sorted(flagged, key=lambda r: -r["gain_pct_vs_auto"]):
        print(f"  {r['model']} {r['proj']} T={r['T']}  best={r['best']}  gain {r['gain_pct_vs_auto']:+.1f}%")


if __name__ == "__main__":
    main()
