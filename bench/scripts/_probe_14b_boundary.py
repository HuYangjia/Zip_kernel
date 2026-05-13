"""Boundary probe: bisect (T, d_out) grid to find exact failure envelope."""
import os, sys
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.insert(0, '/root')
import torch
from kernel.bench.layer.qwen3_w4a4_ops import _build_bundle, _make_callable

device = torch.device('cuda:0')

def try_shape(T, d_in, d_out):
    try:
        b = _build_bundle(d_in=d_in, d_out=d_out, T=T, device=device, name='probe')
        fn = _make_callable(b)
        y = fn()
        torch.cuda.synchronize()
        print(f'  OK  T={T:<6} d_in={d_in:<6} d_out={d_out:<6}  y.shape={tuple(y.shape)}', flush=True)
        del y, b, fn
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f'  FAIL T={T:<6} d_in={d_in:<6} d_out={d_out:<6}  {type(e).__name__}: {str(e)[:120]}', flush=True)
        return False

print('=== sweep T at d_out=34816 ===')
for T in [16384, 20480, 24576, 28672, 32768]:
    try_shape(T, 5120, 34816)

print()
print('=== sweep d_out at T=32768 ===')
for d_out in [17408, 24576, 28672, 32768, 34816]:
    try_shape(32768, 5120, d_out)
