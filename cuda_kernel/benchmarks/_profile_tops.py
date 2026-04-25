import torch, sys, os, time
sys.path.insert(0, "/root")
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
from kernel.cuda_kernel import ops
from kernel.triton_kernel.pack_utils import pack_s4_le, BCOL

torch.manual_seed(0)

def bench_ev(fn, warmup=30, rep=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/rep*1000.0

# Current dispatch for T=128 is kBn=32.  Let's monkey-patch by re-launching
# manually... actually simpler: measure T=64 (kBn=32) vs T=256 (kBn=64, just
# above the T<=128 cutoff) per-CTA work.

configs = [
    ("T=128  kBn=32 (current)", 128, 4096, 4096),
    ("T=256  kBn=64", 256, 4096, 4096),
]
for (name, T, d_out, d_in) in configs:
    n_groups = d_in // BCOL
    X = torch.randn(T, d_in, dtype=torch.float16, device="cuda") * 0.4
    perm = torch.arange(d_in, dtype=torch.int32, device="cuda")
    Xs, sx, sX = ops.activation_quant_cuda(X, perm)
    W_low = pack_s4_le(torch.randint(-8,8,(d_out,d_in),dtype=torch.int8,device="cuda"))
    scale_u4 = (torch.rand(d_out,n_groups,device="cuda")*0.05+0.001).half()
    zero_u4  = (torch.randn(d_out,n_groups,device="cuda")*0.2).half()
    t = bench_ev(lambda: ops.dense_gemm_cuda_int4(W_low, Xs, scale_u4, zero_u4, sX, sx))
    flops = 2.0 * T * d_out * d_in
    tflops = flops / (t * 1e-6) / 1e12
    print(f"{name:30s} t={t:7.2f}us  {tflops:6.2f} TOPS")