"""Stage F.6 — Decomposition profiling for speedup<1 shapes.

Sweeps kBn × kBm × cache × cpAsync combinations for each of the 5
speedup<1 shapes and prints a compact matrix of (t_us, speedup_vs_def).

Conclusions guide the next optimisation step:
  * if one (kBn, kBm) config beats the default significantly, the
    dispatch heuristic is suboptimal (cheap fix).
  * if no config helps, the shape is genuinely architecture-bound.
"""
from __future__ import annotations
import os, sys, statistics, torch
sys.path.insert(0, "/root")
from kernel.cuda_kernel import ops  # noqa: E402


def bench_one(d_out, d_in, T, kbn=None, kbm=None, cache=None, iters=2000, warm=800):
    env_vars = {
        "HKUST_V9_FUSED_FORCE_KBN": kbn,
        "HKUST_V9_FUSED_FORCE_KBM": kbm,
        "HKUST_V9_FUSED_FORCE_CACHE": cache,
    }
    for k, v in env_vars.items():
        if v is None or v == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    ng = d_in // 128
    torch.manual_seed(0)
    W = torch.randint(-8, 7, (d_out, d_in // 2), dtype=torch.int8, device="cuda")
    X = torch.randint(-8, 7, (T, d_in // 2), dtype=torch.int8, device="cuda")
    s = torch.rand(d_out, ng, dtype=torch.float16, device="cuda") * 0.1 + 0.01
    z = torch.randint(-4, 4, (d_out, ng), device="cuda").to(torch.float16)
    sx = torch.rand(T, dtype=torch.float16, device="cuda") * 0.1 + 0.01
    sumX = torch.randint(-100, 100, (T, ng), dtype=torch.int32, device="cuda")
    Y = torch.empty((d_out, T), dtype=torch.float16, device="cuda")
    Wh = torch.zeros(0, 128, 64, dtype=torch.int8, device="cuda")
    ro = torch.zeros(d_out // 128 + 1, dtype=torch.int32, device="cuda")
    ci = torch.zeros(0, dtype=torch.int32, device="cuda")

    def call():
        ops._ext.fused_dense_sparse_mma_int4_launch(
            W, Wh, ro, ci, X, s, z, sumX, sx, Y, d_out, d_in
        )

    for _ in range(warm):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(7):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(iters):
            call()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / iters * 1000)
    return statistics.median(ts)


def main():
    shapes = [
        (2048, 2048, 128),
        (4096, 4096, 128),
        (1024, 4096, 128),
        (4096, 1024, 128),
        (2048, 4096, 128),
        (4096, 2048, 128),
        (4096, 4096, 32),
        (4096, 4096, 1),
    ]
    configs = [
        ("def", None, None, None),
        ("kbn8_kbm128", "8", "128", None),
        ("kbn32_kbm128", "32", "128", None),
        ("kbn64_kbm128", "64", "128", None),
        ("kbn8_kbm64", "8", "64", None),
        ("kbn32_kbm64", "32", "64", None),
        ("kbn8_cON", "8", "128", "1"),
        ("kbn32_cON", "32", "128", "1"),
        ("kbn64_cON", "64", "128", "1"),
    ]
    print(f"{'shape':>22} " + " ".join(f"{c[0]:>13}" for c in configs))
    for shape in shapes:
        t_def = bench_one(*shape, *configs[0][1:])
        row = [f"{t_def:6.2f}us/--"]
        for name, kbn, kbm, cache in configs[1:]:
            t = bench_one(*shape, kbn=kbn, kbm=kbm, cache=cache)
            row.append(f"{t:6.2f}/{t_def/t:.2f}x")
        print(f"{str(shape):>22} " + " ".join(f"{c:>13}" for c in row))


if __name__ == "__main__":
    main()
