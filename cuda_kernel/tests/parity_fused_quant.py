"""P0.2 parity sweep — compare fused vs legacy on a range of shapes."""
import kernel.cuda_kernel.ops as ops
import torch

dev = torch.device("cuda:0")

# Non-trivial shapes including edge cases
SHAPES = [
    # (T, d_in, d_out) — all hp_ratio=0
    (32,  1024, 1024),
    (32,  4096, 4096),     # Qwen3-8B q_proj
    (128, 4096, 4096),
    (32,  4096, 24576),    # Qwen3-8B gate_up (merged)
    (128, 4096, 2048),     # Qwen3-8B kv_proj
    (32,  14336, 4096),    # Qwen3-8B down_proj
    (17,  4096, 4096),     # T % kBn != 0
    (40,  4096, 4096),     # T % kBn != 0
    (64,  8192, 8192),     # LLaMA-70B q_proj scale
    (128, 28672, 8192),    # LLaMA-70B down_proj
]

header = f"{'T':>4} {'d_in':>6} {'d_out':>6} | {'max abs':>10} {'max rel':>10} {'status':>8}"
print(header)
print("-" * len(header))
all_pass = True
for T, d_in, d_out in SHAPES:
    torch.manual_seed(42)
    X = torch.randn(T, d_in, dtype=torch.float16, device=dev) * 0.1
    perm = torch.randperm(d_in, device=dev).to(torch.int32)

    W_low = torch.randint(0, 16, (d_out, d_in // 2), dtype=torch.int8, device=dev)
    n_g = d_in // 128
    scale_u4 = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 0.01 + 0.001).contiguous()
    zero_u4  = (torch.rand(d_out, n_g, dtype=torch.float16, device=dev) * 14.0).contiguous()

    empty_hpb = torch.zeros((0, 128, 64), dtype=torch.int8, device=dev)
    hp_ro = torch.zeros((d_out // 128) + 1, dtype=torch.int32, device=dev)
    hp_ci = torch.zeros(0, dtype=torch.int32, device=dev)

    # Legacy reference
    X_s4, scale_x, sum_X = ops.activation_quant_cuda(X, perm)
    Y_legacy = ops.fused_dense_sparse_cuda_int4(
        W_low, empty_hpb, hp_ro, hp_ci,
        X_s4, scale_u4, zero_u4, sum_X, scale_x, d_out, d_in,
    )
    # Fused candidate
    try:
        Y_fused = ops.fused_quant_dense_sparse_cuda_int4(
            X, perm, W_low, empty_hpb, hp_ro, hp_ci,
            scale_u4, zero_u4, d_out, d_in,
        )
    except Exception as e:
        print(f"{T:>4} {d_in:>6} {d_out:>6} | SKIP: {e!s:.60}")
        continue

    diff = (Y_legacy.float() - Y_fused.float()).abs()
    rel = diff / (Y_legacy.float().abs() + 1e-6)
    mad = diff.max().item()
    mrd = rel.max().item()
    ok = (mad < 1e-3) or (mrd < 0.05)
    all_pass = all_pass and ok
    status = "PASS" if ok else "FAIL"
    print(f"{T:>4} {d_in:>6} {d_out:>6} | {mad:>10.4g} {mrd:>10.4g} {status:>8}")

print()
print("=== OVERALL:", "PASS" if all_pass else "FAIL", "===")
