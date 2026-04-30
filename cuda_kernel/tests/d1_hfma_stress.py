"""D.1 HFMA stress diagnostic.

Measures cuda_us on representative compute-bound shapes with 8 extra
HFMA ops injected into fold_dense critical path.  Compares to clean
r66 baseline (which we captured from r66 bench.json) to determine if
HFMA2 is on the critical path.

Run this AFTER building the tree with the HFMA injection patch.
"""
import torch
import kernel.cuda_kernel.ops as ops

dev = torch.device("cuda:0")


def bench(T, d_in, d_out, label):
    torch.manual_seed(0)
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

    def run():
        ops.fused_dense_sparse_cuda_int4(
            W, hpb, hpro, hpci, X_s4, su, zu, sum_X, sx, d_out, d_in,
        )
    for _ in range(500):
        run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(10):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(200):
            run()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) * 1000.0 / 200)
    print(f"{label}: {best:.2f} us", flush=True)
    return best


# r66 reference values from logs/r66_path_c/bench.json
R66 = {
    "8B gu T=512":  458.3,
    "14B gu T=512": 1513.4,
    "8B gu T=128":  148.5,
    "8B gu T=32":    66.9,
    "8B gu T=1":     94.9,
}

cases = [
    (512, 4096, 24576, "8B gu T=512"),
    (512, 5120, 34816, "14B gu T=512"),
    (128, 4096, 24576, "8B gu T=128"),
    (32,  4096, 24576, "8B gu T=32"),
    (1,   4096, 24576, "8B gu T=1"),
]
print(f"{'shape':<16}  {'r66':>7}  {'stress':>7}  {'slowdown':>9}")
for T, d_in, d_out, lbl in cases:
    us = bench(T, d_in, d_out, lbl)
    ref = R66[lbl]
    pct = (us - ref) / ref * 100
    print(f"  -> {lbl:<14} ref={ref:.1f}  stress={us:.1f}  +{pct:.1f}%", flush=True)
