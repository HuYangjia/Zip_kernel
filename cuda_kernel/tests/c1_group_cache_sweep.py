"""C.1 — Group-cache gate sweep at T=128 (in-process, fast).

The kernel reads HKUST_V9_FUSED_FORCE_CACHE via std::getenv on every
launch() call, so we can toggle os.environ in-process between bench
runs.  This avoids the 15-20 s JIT-import cost per subprocess.

Output: CSV + human table of (auto / force_on / force_off) kernel times
at T=128 across 25 production shapes.
"""
import os
import json
import sys
from pathlib import Path

import torch

import kernel.cuda_kernel.ops as ops

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "sweep_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def bench_us(fn, warmup=500, outer=20, inner=200):
    # Strict timing (project memory bmmiahpl).
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


# Qwen3 production shapes, T=128 focus
SHAPES = []
def add(model, proj, T, d_in, d_out):
    SHAPES.append(dict(model=model, proj=proj, T=T, d_in=d_in, d_out=d_out))

add("Qwen3-0.6B", "q",  128, 1024, 2048)
add("Qwen3-0.6B", "kv", 128, 1024, 2048)
add("Qwen3-0.6B", "o",  128, 2048, 1024)
add("Qwen3-0.6B", "gu", 128, 1024, 6144)
add("Qwen3-0.6B", "dn", 128, 3072, 1024)
add("Qwen3-1.7B", "q",  128, 2048, 2048)
add("Qwen3-1.7B", "kv", 128, 2048, 2048)
add("Qwen3-1.7B", "o",  128, 2048, 2048)
add("Qwen3-1.7B", "gu", 128, 2048, 12288)
add("Qwen3-1.7B", "dn", 128, 6144, 2048)
add("Qwen3-4B",   "q",  128, 2560, 4096)
add("Qwen3-4B",   "kv", 128, 2560, 2048)
add("Qwen3-4B",   "o",  128, 4096, 2560)
add("Qwen3-4B",   "gu", 128, 2560, 18432)
add("Qwen3-4B",   "dn", 128, 9216, 2560)
add("Qwen3-8B",   "q",  128, 4096, 4096)
add("Qwen3-8B",   "kv", 128, 4096, 2048)
add("Qwen3-8B",   "o",  128, 4096, 4096)
add("Qwen3-8B",   "gu", 128, 4096, 24576)
add("Qwen3-8B",   "dn", 128, 14336, 4096)
add("Qwen3-14B",  "q",  128, 5120, 5120)
add("Qwen3-14B",  "kv", 128, 5120, 1024)
add("Qwen3-14B",  "o",  128, 5120, 5120)
add("Qwen3-14B",  "gu", 128, 5120, 34816)
add("Qwen3-14B",  "dn", 128, 17408, 5120)


def run_mode(sh, mode):
    if mode == "auto":
        os.environ.pop("HKUST_V9_FUSED_FORCE_CACHE", None)
    elif mode == "force_on":
        os.environ["HKUST_V9_FUSED_FORCE_CACHE"] = "1"
    elif mode == "force_off":
        os.environ["HKUST_V9_FUSED_FORCE_CACHE"] = "0"
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
    return bench_us(run)


def main():
    out_rows = []
    print(f"{'model':<10} {'proj':<4} {'d_in':>5} {'d_out':>6} {'n_g':>4} {'gM':>4} "
          f"{'auto':>7} {'on':>7} {'off':>7} {'best':>10} {'gain':>6}")
    print("-" * 84)
    for i, sh in enumerate(SHAPES):
        n_g = sh["d_in"] // 128
        grid_M = (sh["d_out"] + 127) // 128
        try:
            us_auto = run_mode(sh, "auto")
            us_on   = run_mode(sh, "force_on")
            us_off  = run_mode(sh, "force_off")
        except Exception as ex:
            print(f"{sh['model']} {sh['proj']} FAIL: {ex}", flush=True)
            continue
        trio = {"auto": us_auto, "on": us_on, "off": us_off}
        best_mode = min(trio, key=trio.get)
        gain = (us_auto - trio[best_mode]) / us_auto * 100
        row = {**sh, "n_groups": n_g, "grid_M": grid_M,
               "auto_us": us_auto, "on_us": us_on, "off_us": us_off,
               "best_mode": best_mode, "gain_pct_vs_auto": gain}
        out_rows.append(row)
        print(f"{sh['model']:<10} {sh['proj']:<4} {sh['d_in']:>5} {sh['d_out']:>6} "
              f"{n_g:>4} {grid_M:>4} {us_auto:>7.2f} {us_on:>7.2f} {us_off:>7.2f} "
              f"{best_mode:>10} {gain:>+5.1f}%", flush=True)

    out_path = OUT_DIR / "c1_group_cache_sweep.json"
    out_path.write_text(json.dumps({"rows": out_rows, "n": len(out_rows)}, indent=2))
    print(f"\nWrote {out_path}")

    # Summary
    gainers = sorted([r for r in out_rows if r["gain_pct_vs_auto"] > 3.0],
                     key=lambda r: -r["gain_pct_vs_auto"])
    print(f"\nShapes where non-auto mode beats auto by >3%: "
          f"{len(gainers)} / {len(out_rows)}")
    for r in gainers:
        print(f"  {r['model']:<10} {r['proj']:<4} d=({r['d_in']},{r['d_out']}) "
              f"n_g={r['n_groups']:>3} grid_M={r['grid_M']:>3}  "
              f"best={r['best_mode']:<10} gain={r['gain_pct_vs_auto']:+.1f}%")


if __name__ == "__main__":
    main()
