"""Probe (1024,4096,128) regression: a kBm × kBn sweep to pick the right config."""
import os, sys, statistics, torch
sys.path.insert(0, '/root')
from kernel.cuda_kernel import ops


def bench(d_out, d_in, T, **env):
    for k in ['HKUST_V9_FUSED_FORCE_KBN', 'HKUST_V9_FUSED_FORCE_KBM', 'HKUST_V9_FUSED_FORCE_CACHE']:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = str(v)
    ng = d_in // 128
    W = torch.randint(-8, 7, (d_out, d_in // 2), dtype=torch.int8, device='cuda')
    X = torch.randint(-8, 7, (T, d_in // 2), dtype=torch.int8, device='cuda')
    s = torch.rand(d_out, ng, dtype=torch.float16, device='cuda') * 0.1 + 0.01
    z = torch.randint(-4, 4, (d_out, ng), device='cuda').to(torch.float16)
    sx = torch.rand(T, dtype=torch.float16, device='cuda') * 0.1 + 0.01
    sumX = torch.randint(-100, 100, (T, ng), dtype=torch.int32, device='cuda')
    Y = torch.empty((d_out, T), dtype=torch.float16, device='cuda')
    Wh = torch.zeros(0, 128, 64, dtype=torch.int8, device='cuda')
    ro = torch.zeros(d_out // 128 + 1, dtype=torch.int32, device='cuda')
    ci = torch.zeros(0, dtype=torch.int32, device='cuda')

    def call():
        ops._ext.fused_dense_sparse_mma_int4_launch(W, Wh, ro, ci, X, s, z, sumX, sx, Y, d_out, d_in)

    for _ in range(600):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(7):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(2000):
            call()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / 2000 * 1000)
    return statistics.median(ts)


def main():
    shape = (1024, 4096, 128)
    print(str(shape))
    for cfg in [
        {},
        {'HKUST_V9_FUSED_FORCE_KBM': 128},
        {'HKUST_V9_FUSED_FORCE_KBM': 64},
        {'HKUST_V9_FUSED_FORCE_KBN': 32, 'HKUST_V9_FUSED_FORCE_KBM': 128},
        {'HKUST_V9_FUSED_FORCE_KBN': 64, 'HKUST_V9_FUSED_FORCE_KBM': 128},
        {'HKUST_V9_FUSED_FORCE_KBN': 32, 'HKUST_V9_FUSED_FORCE_KBM': 64},
        {'HKUST_V9_FUSED_FORCE_KBN': 8, 'HKUST_V9_FUSED_FORCE_KBM': 64},
    ]:
        t = bench(*shape, **cfg)
        print(f'  {cfg} -> {t:7.2f}us')


if __name__ == '__main__':
    main()
