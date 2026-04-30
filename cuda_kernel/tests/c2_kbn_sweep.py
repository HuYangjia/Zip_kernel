"""C.2 — kBn sweep including the new kBn=16 tile size.

Goal: discover shapes where kBn=16 beats auto by >3%.  Focus on the
T regime where kBn=16 gives exactly 1 tile (T=16) or 2 tiles (T=32)
or 8 tiles (T=128) — between kBn=8's over-fragmentation and kBn=32's
wave-starvation.

Method: in-process env sweep over HKUST_V9_FUSED_FORCE_KBN ∈
{"", "8", "16", "32", "64"}.
"""
import os
import json
from pathlib import Path

import torch
import kernel.cuda_kernel.ops as ops

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "sweep_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bench_us(fn, warmup=500, outer=20, inner=200):
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

# Qwen3 T=32 and T=128 (the regime where kBn=16 might help).
# Skip T=1 (GEMV) and T=512 (already wave-saturated).
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

MODES = [
    ("auto", None),
    ("k8",   "8"),
    ("k16",  "16"),
    ("k32",  "32"),
    ("k64",  "64"),
]


def run_shape(sh, kbn_env):
    if kbn_env is None:
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBN", None)
    else:
        os.environ["HKUST_V9_FUSED_FORCE_KBN"] = kbn_env
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
    try:
        return bench_us(run)
    except Exception as ex:
        return float("nan")


def main():
    rows = []
    header = f"{'model':<10} {'proj':<4} {'T':>4} {'d_in':>5} {'d_out':>6} " + \
             " ".join(f"{lbl:>7}" for lbl, _ in MODES) + \
             f"  {'best':>6} {'gain%':>6}"
    print(header)
    print("-" * len(header))
    for sh in SHAPES:
        us_by_mode = {}
        for lbl, env in MODES:
            us_by_mode[lbl] = run_shape(sh, env)
        auto_us = us_by_mode["auto"]
        # best among forced values only (auto is the reference)
        forced = {k: v for k, v in us_by_mode.items() if k != "auto"}
        best_mode = min(forced, key=lambda k: forced[k] if forced[k] == forced[k] else 1e9)
        gain = (auto_us - forced[best_mode]) / auto_us * 100 if auto_us == auto_us else 0
        row = {**sh, "us_by_mode": us_by_mode, "best_forced": best_mode,
               "gain_pct_vs_auto": gain}
        rows.append(row)
        print(f"{sh['model']:<10} {sh['proj']:<4} {sh['T']:>4} {sh['d_in']:>5} {sh['d_out']:>6} " +
              " ".join(f"{us_by_mode[lbl]:>7.2f}" for lbl, _ in MODES) +
              f"  {best_mode:>6} {gain:>+5.1f}%", flush=True)

    out_path = OUT_DIR / "c2_kbn_sweep.json"
    out_path.write_text(json.dumps({"rows": rows, "n": len(rows)}, indent=2))
    print(f"\nWrote {out_path}")

    # Summary — shapes that prefer kBn=16 over auto by >3%
    k16_wins = [r for r in rows if r["best_forced"] == "k16" and r["gain_pct_vs_auto"] > 3.0]
    print(f"\nShapes where kBn=16 beats auto by >3%: {len(k16_wins)}")
    for r in sorted(k16_wins, key=lambda r: -r["gain_pct_vs_auto"]):
        print(f"  {r['model']:<10} {r['proj']:<4} T={r['T']:>3} d=({r['d_in']},{r['d_out']})  "
              f"gain {r['gain_pct_vs_auto']:+.1f}% (auto {r['us_by_mode']['auto']:.2f} → "
              f"k16 {r['us_by_mode']['k16']:.2f})")

    # Also flag any NON-k16 wins (k8/k32/k64 beats auto by >3%) — dispatcher oversight
    other_wins = [r for r in rows if r["best_forced"] != "k16" and r["gain_pct_vs_auto"] > 3.0]
    print(f"\nDispatcher oversights (non-k16 mode beats auto >3%): {len(other_wins)}")
    for r in sorted(other_wins, key=lambda r: -r["gain_pct_vs_auto"]):
        print(f"  {r['model']:<10} {r['proj']:<4} T={r['T']:>3} d=({r['d_in']},{r['d_out']}) "
              f"best={r['best_forced']} gain {r['gain_pct_vs_auto']:+.1f}%")


if __name__ == "__main__":
    main()
