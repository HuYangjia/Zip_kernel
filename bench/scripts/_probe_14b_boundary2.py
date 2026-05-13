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
        print(f'  OK   T={T:<6} d_in={d_in:<6} d_out={d_out:<6}', flush=True)
        del y, b, fn
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f'  FAIL T={T:<6} d_in={d_in:<6} d_out={d_out:<6}  {str(e)[:60]}', flush=True)
        return False

print('=== vary d_in at T=32768 d_out=17408 ===')
# 4B d_in=2560, 8B d_in=4096, 14B d_in=5120; also try 5120 smaller Ts
for d_in in [2560, 3072, 4096, 4608, 5120, 5632]:
    try_shape(32768, d_in, 17408)

print()
print('=== 14B d_in=5120 at smaller T (find boundary) ===')
for T in [28672, 30720, 31744, 32256, 32512, 32640, 32768]:
    try_shape(T, 5120, 17408)

print()
print('=== cross-check: 14B d_in=5120 at prefill bs=32 (T=65536) ===')
try_shape(65536, 5120, 17408)
