"""C.10-v2 组合扫描探针：kBn × split_k 对 32B/70B gu loser 的影响.

前置：C.10 诊断已证明 split_k 单独扫描无效（1/2/4/8 时间几乎相同），
       且 32B/70B gu T≥512 路径已被 C.8.1(a) 锁死 kBm=128.

本脚本目标：在已锁定 kBm=128 的前提下，联合扫 (kBn, split_k) 组合，
确认 auto 选出的 kBn 是否真的是最优，或换成更小/更大的 kBn 能否让
MMA pipeline 更饱满（从而缓解 pipeline starvation）。

测试矩阵:
  Shapes (6):
     32B_gu T=512/1024/2048,  70B_gu T=512/1024/2048
  kBn  (4):  8, 16, 32, 64
  sk   (3):  1, 2, 4    (C.10 已证明 >4 无效)

  每 shape × 12 组合 = 72 测量点
  T=512 : warmup=300/outer=15/inner=100, trials=3  (~5s/point → 60s/shape)
  T=1024: warmup=200/outer=10/inner=100, trials=3  (~6s/point → 72s/shape)
  T=2048: warmup=200/outer=10/inner=50,  trials=3  (~8s/point → 96s/shape)
  总计 ~ 7-9 min.

输出:
   logs/r72_c10v2/combo_probe.json
   stdout: 对每个 shape 打印 heatmap + 标注 best(kBn,sk)
"""
import json
import os
import statistics
import sys
import time
sys.path.insert(0, '/root/Zip_kernel')

import torch

from kernel.cuda_kernel import ops
from kernel.triton_kernel.activation_quant import quantize_activation_s4
from kernel.triton_kernel.pack_utils import BCOL, pack_s4_le


def make_inputs(T, d_out, d_in, seed=0xBEEF, device="cuda"):
    torch.manual_seed(seed)
    X = torch.randn(T, d_in, dtype=torch.float16, device=device) * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device=device)
    X_s4, scale_x, sum_X = quantize_activation_s4(X, perm)
    n_groups = d_in // BCOL
    W_low_s4 = torch.randint(-8, 8, (d_out, d_in), dtype=torch.int8, device=device)
    W_low_packed = pack_s4_le(W_low_s4)
    scale_u4 = (torch.rand(d_out, n_groups, device=device) * 0.05 + 0.001).to(torch.float16)
    zero_u4  = (torch.randn(d_out, n_groups, device=device) * 0.2).to(torch.float16)
    return W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, X


def _time_inner(fn, warmup, outer, inner):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    means = []
    for _ in range(outer):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        means.append(s.elapsed_time(e) * 1000.0 / inner)
    return min(means)


def bench_median(fn, n_trials, warmup, outer, inner):
    samples = []
    for _ in range(n_trials):
        samples.append(_time_inner(fn, warmup, outer, inner))
    return statistics.median(samples), samples


def run_cuda(args):
    return ops.dense_gemm_cuda_int4(*args)


def run_fp16(W_deq, X_fp16):
    return torch.matmul(W_deq, X_fp16.t())


SHAPES = [
    # label,           T,    d_out,  d_in
    ("32B_gu_T512",    512,  55296,  5120),
    ("32B_gu_T1024", 1024,   55296,  5120),
    ("32B_gu_T2048", 2048,   55296,  5120),
    ("70B_gu_T512",    512,  57344,  8192),
    ("70B_gu_T1024", 1024,   57344,  8192),
    ("70B_gu_T2048", 2048,   57344,  8192),
]

KBN_LIST = [8, 16, 32, 64]
SK_LIST  = [1, 2, 4]


def timing_spec(T):
    if T >= 2048:
        return dict(warmup=200, outer=10, inner=50, n_trials=3)
    elif T >= 1024:
        return dict(warmup=200, outer=10, inner=100, n_trials=3)
    else:
        return dict(warmup=300, outer=15, inner=100, n_trials=3)


def main():
    results = []
    t0 = time.time()
    for label, T, d_out, d_in in SHAPES:
        spec = timing_spec(T)
        n_groups = d_in // BCOL
        print(f"\n### {label} (T={T}, d_out={d_out}, d_in={d_in}, n_g={n_groups})")
        print(f"    timing: {spec}")
        try:
            W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, X_fp16 = \
                make_inputs(T, d_out, d_in)
        except torch.cuda.OutOfMemoryError:
            print("    OOM skip"); continue

        cuda_args = (W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x)

        # FP16 baseline
        Wg = W_low_packed.view(d_out, d_in // 2, 1)
        low  = (Wg & 0x0F).to(torch.int8)
        high = ((Wg >> 4) & 0x0F).to(torch.int8)
        q_W = torch.cat([low, high], dim=-1).view(d_out, d_in // BCOL, BCOL).float()
        W_deq = ((q_W - zero_u4.unsqueeze(-1).float()) *
                 scale_u4.unsqueeze(-1).float()).view(d_out, d_in).to(torch.float16)
        del Wg, low, high, q_W
        fp16_us, _ = bench_median(
            lambda: run_fp16(W_deq, X_fp16), **spec)
        print(f"    FP16: {fp16_us:8.2f} us")
        del W_deq; torch.cuda.empty_cache()

        # AUTO baseline (no env override)
        os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBN", None)
        auto_us, _ = bench_median(lambda: run_cuda(cuda_args), **spec)
        print(f"    auto: {auto_us:8.2f} us  sp={fp16_us/auto_us:.3f}x")

        row = {"label": label, "T": T, "d_out": d_out, "d_in": d_in,
               "n_groups": n_groups, "fp16_us": fp16_us,
               "auto_us": auto_us, "auto_sp": fp16_us / auto_us,
               "combos": []}

        # Combo sweep
        print(f"    {'kBn':>4} | " + " ".join(f"sk={s:>1}:us,sp" for s in SK_LIST))
        for kbn in KBN_LIST:
            os.environ["HKUST_V9_FUSED_FORCE_KBN"] = str(kbn)
            line = f"    {kbn:>4} | "
            for sk in SK_LIST:
                # sk must divide n_groups
                if n_groups % sk != 0:
                    row["combos"].append({"kBn": kbn, "sk": sk, "us": None,
                                         "sp": None, "skipped": True})
                    line += f"         -       "
                    continue
                os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(sk)
                us, samples = bench_median(lambda: run_cuda(cuda_args), **spec)
                sp = fp16_us / us
                row["combos"].append({"kBn": kbn, "sk": sk, "us": us,
                                     "sp": sp, "samples": samples})
                line += f" {us:6.1f}us,{sp:.2f}x "
            print(line)
        os.environ.pop("HKUST_V9_FUSED_FORCE_KBN", None)
        os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)

        # Pick best combo
        valid = [c for c in row["combos"] if c.get("sp") is not None]
        best = max(valid, key=lambda c: c["sp"])
        gain = best["sp"] - row["auto_sp"]
        flag = "🚀" if gain > 0.02 else "~" if gain > -0.02 else "❌"
        print(f"    best: kBn={best['kBn']} sk={best['sk']} sp={best['sp']:.3f}x "
              f"(auto {row['auto_sp']:.3f}x, {flag}{gain:+.3f})")
        row["best_combo"] = {k: best[k] for k in ("kBn", "sk", "us", "sp")}
        row["best_gain"]  = gain
        results.append(row)

        del W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, X_fp16
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n=== Total runtime {elapsed:.1f}s ===")

    # Summary
    print("\n### Summary (C.10-v2 combo scan)")
    print(f"{'shape':<18} {'auto_sp':>8} {'best_combo':>14} {'best_sp':>8} {'gain':>7}")
    for r in results:
        bc = r["best_combo"]
        combo = f"kBn={bc['kBn']},sk={bc['sk']}"
        print(f"{r['label']:<18} {r['auto_sp']:>8.3f} {combo:>14} "
              f"{bc['sp']:>8.3f} {r['best_gain']:>+7.3f}")

    outpath = os.environ.get("HKUST_V9_C10V2_OUT",
                             "/root/Zip_kernel/kernel/cuda_kernel/logs/r72_c10v2/combo_probe.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON dumped to {outpath}")


if __name__ == "__main__":
    main()
