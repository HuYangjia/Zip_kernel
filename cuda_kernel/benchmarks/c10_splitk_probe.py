"""C.10 Split-K deep-dive probe: 32B / 70B gate_up at prefill T.

Goal: find out whether raising split_k from 1/2 → 4 or 8 can rescue the
0.70x speedup cliff on Qwen2.5-32B / LLaMA3-70B gate_up T≥1024 shapes.

Test matrix:
  Models : Qwen2.5-32B (5120 → 55296, n_g=80),
           LLaMA3-70B  (8192 → 57344, n_g=128)
  T      : 512, 1024, 2048
  split_k: 1, 2, 4, 8           (via HKUST_V9_FUSED_FORCE_SPLITK)

Timing : min-over-outer of (mean-over-inner of per-iter us),
         median over 5 independent trials.
         warmup=500, outer=20, inner=200  (per GPU micro-bench spec).

FP16 baseline is also recorded for sp computation.
"""
import json
import os
import statistics
import sys
import time
sys.path.insert(0, '/root')

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
    """Return min-over-outer of mean-over-inner per-iter us."""
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


def bench_median(fn, n_trials=5, warmup=500, outer=20, inner=200):
    samples = []
    for _ in range(n_trials):
        samples.append(_time_inner(fn, warmup, outer, inner))
    return statistics.median(samples), samples


def run_cuda(args):
    return ops.dense_gemm_cuda_int4(*args)


def run_fp16(W_deq, X_fp16):
    return torch.matmul(W_deq, X_fp16.t())


SHAPES = [
    # label,               T,    d_out,  d_in
    ("32B_gu_T512",         512, 55296,  5120),
    ("32B_gu_T1024",       1024, 55296,  5120),
    ("32B_gu_T2048",       2048, 55296,  5120),
    ("70B_gu_T512",         512, 57344,  8192),
    ("70B_gu_T1024",       1024, 57344,  8192),
    ("70B_gu_T2048",       2048, 57344,  8192),
]

SPLIT_KS = [1, 2, 4, 8]


def main():
    # Small warmup parameters for T=2048 to keep total runtime reasonable.
    # T=2048 × (4 splits × 5 trials × (500 warmup + 20×200 inner)) =
    #   each trial ~4500 calls of a ~15ms kernel ≈ 68s, × 4 × 5 = 22 min
    # Scale down for T=2048 to (warmup=200, outer=10, inner=50).
    results = []
    t0 = time.time()

    for label, T, d_out, d_in in SHAPES:
        if T >= 2048:
            warmup, outer, inner = 200, 10, 50
            n_trials = 3
        elif T >= 1024:
            warmup, outer, inner = 300, 15, 100
            n_trials = 3
        else:
            warmup, outer, inner = 500, 20, 200
            n_trials = 3
        print(f"\n### {label} (T={T}, d_out={d_out}, d_in={d_in}) n_g={d_in//BCOL}")
        print(f"    timing: warmup={warmup} outer={outer} inner={inner} trials={n_trials}")

        try:
            W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, X_fp16 = \
                make_inputs(T, d_out, d_in)
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM, skipping")
            continue
        cuda_args = (W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x)

        # FP16 baseline: dequantise W first.
        Wg = W_low_packed.view(d_out, d_in // 2, 1)
        low  = (Wg & 0x0F).to(torch.int8)
        high = ((Wg >> 4) & 0x0F).to(torch.int8)
        q_W = torch.cat([low, high], dim=-1).view(d_out, d_in // BCOL, BCOL).float()
        W_deq = ((q_W - zero_u4.unsqueeze(-1).float()) *
                 scale_u4.unsqueeze(-1).float()).view(d_out, d_in).to(torch.float16)
        # Free int tmp to give room to FP16 W
        del Wg, low, high, q_W

        fp16_us, fp16_samples = bench_median(
            lambda: run_fp16(W_deq, X_fp16),
            n_trials=n_trials, warmup=warmup, outer=outer, inner=inner)
        print(f"    FP16           : {fp16_us:8.2f} us   samples={[f'{s:.1f}' for s in fp16_samples]}")

        del W_deq
        torch.cuda.empty_cache()

        row = {"label": label, "T": T, "d_out": d_out, "d_in": d_in,
               "n_groups": d_in // BCOL, "fp16_us": fp16_us}

        # Auto dispatch (no env override)
        os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)
        auto_us, _ = bench_median(lambda: run_cuda(cuda_args),
                                  n_trials=n_trials, warmup=warmup, outer=outer, inner=inner)
        row["auto_us"] = auto_us
        row["auto_sp"] = fp16_us / auto_us
        print(f"    auto           : {auto_us:8.2f} us   sp={fp16_us/auto_us:.3f}x")

        for sk in SPLIT_KS:
            os.environ["HKUST_V9_FUSED_FORCE_SPLITK"] = str(sk)
            us, samples = bench_median(lambda: run_cuda(cuda_args),
                                       n_trials=n_trials, warmup=warmup, outer=outer, inner=inner)
            sp = fp16_us / us
            row[f"sk{sk}_us"] = us
            row[f"sk{sk}_sp"] = sp
            row[f"sk{sk}_samples"] = samples
            marker = " ★" if sp == max(fp16_us/row.get(f'sk{s}_us', 1e9) for s in SPLIT_KS if f'sk{s}_us' in row) else "  "
            print(f"    sk={sk:<2d}          : {us:8.2f} us   sp={sp:.3f}x{marker}"
                  f"   samples={[f'{s:.1f}' for s in samples]}")

        os.environ.pop("HKUST_V9_FUSED_FORCE_SPLITK", None)
        results.append(row)

        # Free buffers
        del W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x, X_fp16
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n--- Total runtime: {elapsed:.1f}s ---\n")

    # Summary
    print("\n### Summary")
    print(f"{'shape':<20} {'fp16':>8} {'auto':>8} {'sp':>6}  "
          f"{'sk=1':>8} {'sk=2':>8} {'sk=4':>8} {'sk=8':>8}  best_sk")
    for r in results:
        best_sk = max(SPLIT_KS, key=lambda s: r.get(f'sk{s}_sp', 0))
        best_sp = r.get(f'sk{best_sk}_sp', 0)
        gain = best_sp - r['auto_sp']
        flag = "🚀" if gain > 0.02 else "~" if gain > -0.02 else "❌"
        print(f"{r['label']:<20} {r['fp16_us']:>8.1f} {r['auto_us']:>8.1f} "
              f"{r['auto_sp']:>6.3f}  "
              f"{r.get('sk1_us', 0):>8.1f} {r.get('sk2_us', 0):>8.1f} "
              f"{r.get('sk4_us', 0):>8.1f} {r.get('sk8_us', 0):>8.1f}  "
              f"sk={best_sk} sp={best_sp:.3f}x ({flag}{gain:+.3f})")

    # Dump JSON
    outpath = os.environ.get("HKUST_V9_C10_OUT", "/root/Zip_kernel/kernel/cuda_kernel/logs/c10_probe.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON dumped to {outpath}")


if __name__ == "__main__":
    main()
