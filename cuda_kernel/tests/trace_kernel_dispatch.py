"""Runtime verification: for every (T, shape) the prefill bench ran,
which CUDA kernel was actually launched?

Uses CUDA profiler to capture kernel names for a sampled set of shapes.
"""
import torch
import kernel.cuda_kernel.ops as cuda_ops
from kernel.cuda_kernel.benchmarks.bench_qwen3_shapes import make_inputs
from torch.profiler import profile, ProfilerActivity


def trace_one(T, d_in, d_out, label):
    b = make_inputs(T, d_out, d_in, hp_ratio=0.05, device="cuda", seed=T+d_in+d_out)
    X = b["X"]; perm = b["perm"]

    # Warmup
    for _ in range(3):
        if T == 1:
            y = cuda_ops.fused_quant_gemv_cuda(
                X, perm, b["W_low_packed"], b["W_high_packed"],
                b["hp_row_offsets"], b["hp_col_indices"],
                b["scale_u4"], b["zero_u4"], d_out, d_in)
        else:
            X_s4, sx, sX = cuda_ops.activation_quant_cuda(X, perm)
            y = cuda_ops.fused_dense_sparse_cuda(
                b["W_low_packed"], b["W_high_packed"],
                b["hp_row_offsets"], b["hp_col_indices"],
                X_s4, b["scale_u4"], b["zero_u4"],
                sX, sx, d_out, d_in)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        if T == 1:
            y = cuda_ops.fused_quant_gemv_cuda(
                X, perm, b["W_low_packed"], b["W_high_packed"],
                b["hp_row_offsets"], b["hp_col_indices"],
                b["scale_u4"], b["zero_u4"], d_out, d_in)
        else:
            X_s4, sx, sX = cuda_ops.activation_quant_cuda(X, perm)
            y = cuda_ops.fused_dense_sparse_cuda(
                b["W_low_packed"], b["W_high_packed"],
                b["hp_row_offsets"], b["hp_col_indices"],
                X_s4, b["scale_u4"], b["zero_u4"],
                sX, sx, d_out, d_in)
        torch.cuda.synchronize()

    print(f"=== {label}: T={T} shape={d_in}->{d_out} ===")
    for e in prof.key_averages():
        # Try both APIs (older/newer torch): cuda_time_total / device_time_total
        t = getattr(e, 'device_time_total', None)
        if t is None:
            t = getattr(e, 'cuda_time_total', 0)
        if t > 0 and not e.key.startswith('aten::'):
            print(f"  {e.key:<70}  count={e.count} total_us={t:.1f}")
    print()


# Sample shapes covering the 4 prefill Ts and different models
SAMPLES = [
    ("Qwen3-1.7B q T=1024",  1024, 2048, 2048),
    ("Qwen3-4B gu T=2048",   2048, 2560, 19456),
    ("Qwen3-8B q T=4096",    4096, 4096, 4096),
    ("Qwen3-14B gu T=8192",  8192, 5120, 34816),
    ("Qwen2.5-32B dn T=2048", 2048, 27648, 5120),
    ("LLaMA3-70B gu T=2048", 2048, 8192, 57344),
    ("LLaMA3-70B kv T=1024", 1024, 8192, 2048),
    # Also test T=1 for sanity (different path)
    ("Qwen3-8B q T=1",       1, 4096, 4096),
]

for label, T, d_in, d_out in SAMPLES:
    trace_one(T, d_in, d_out, label)
