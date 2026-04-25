import torch, sys, os
sys.path.insert(0, "/root")
from kernel.cuda_kernel import ops
from kernel.triton_kernel.pack_utils import pack_s4_le, BCOL, BROW

torch.manual_seed(0)

def bench_ev(fn, warmup=30, rep=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s=torch.cuda.Event(True); e=torch.cuda.Event(True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)/rep*1000.0

# T=1 decode  4k_4k  --  end_to_end path (fused_quant_gemv)
T, d_out, d_in = 1, 4096, 4096
n_groups = d_in // BCOL
X = torch.randn(T, d_in, dtype=torch.float16, device="cuda") * 0.4
perm = torch.arange(d_in, dtype=torch.int32, device="cuda")
W_low = pack_s4_le(torch.randint(-8,8,(d_out,d_in),dtype=torch.int8,device="cuda"))
W_fp = torch.randn(d_out, d_in, dtype=torch.float16, device="cuda") * 0.02
scale_u4 = (torch.rand(d_out,n_groups,device="cuda")*0.05+0.001).half()
zero_u4  = (torch.randn(d_out,n_groups,device="cuda")*0.2).half()
nrow = d_out // BROW
ncol = d_in // BCOL
n_hp = max(1, int(nrow*ncol*0.05))
flat = torch.randperm(nrow*ncol, device="cuda")[:n_hp]
br = (flat//ncol).to(torch.int32); bc = (flat%ncol).to(torch.int32)
order = torch.argsort(br.to(torch.int64)*1000000 + bc.to(torch.int64))
br = br[order]; bc = bc[order]
W_high = pack_s4_le(torch.randint(-8,8,(n_hp,BROW,BCOL),dtype=torch.int8,device="cuda"))
hp_row_off = torch.zeros(nrow+1, dtype=torch.int32, device="cuda")
hp_row_off[1:] = torch.cumsum(torch.bincount(br.to(torch.int64), minlength=nrow),0).to(torch.int32)

# Stage 1: pure FP16 reference
t_fp = bench_ev(lambda: torch.matmul(W_fp, X.t()))
print(f"torch.matmul(W_fp, X.t()): {t_fp:.2f} us")

# Stage 2: fused_quant_gemv_cuda (e2e path for T=1)
t_fused = bench_ev(lambda: ops.fused_quant_gemv_cuda(
    X, perm, W_low, W_high, hp_row_off, bc, scale_u4, zero_u4, d_out, d_in
))
print(f"fused_quant_gemv:          {t_fused:.2f} us")

# HBM-bound roofline:
# W bytes = 4096 * 2048 = 8 MB packed
# W_high bytes = n_hp * 16 * 64 = ~210 KB packed  (~0.2 MB)
# Total = ~8.2 MB  vs  FP16 W = 32 MB
# RTX 4090 HBM BW = 1008 GB/s
# int4 ideal = 8.2 MB / 1008 GB/s = 8.1 us
# fp16 ideal = 32 MB / 1008 GB/s = 31 us  (but cublas achieves ~17us -- L2 cached)
print(f"HBM roofline (int4 weight read only): {(4096*2048+n_hp*16*64)/1008e9 * 1e6:.1f} us")
print(f"HBM roofline (fp16 weight read only): {(4096*4096*2)/1008e9 * 1e6:.1f} us")