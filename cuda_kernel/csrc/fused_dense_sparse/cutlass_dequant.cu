// Copyright (c) 2026 HKUST R50 — Stage A1 dequant kernel (bit-exact).
//
// Consumes the per-group int32 workspace produced by n_groups CUTLASS
// Int4Gemm calls (one per K-slice of width BCOL = 128) and writes the
// fp16 Y_total tensor using the V9 dequant formula:
//
//   y_fp16[m, t] = sum_g (acc_s32[g, m, t] - zero_u4[m, g] * sum_X[t, g])
//                        * scale_u4[m, g] * scale_x[t]
//
// Workspace layout:
//   acc_s32 : (n_groups, d_out, T) int32 row-major
//   sum_X   : (T, n_groups) int32
//   scale_u4: (d_out, n_groups) fp16
//   zero_u4 : (d_out, n_groups) fp16
//   scale_x : (T,) fp16
//   Y_total : (d_out, T) fp16
//
// Parallelism: one thread per (m, t) output element.  The kernel
// itself is memory-bound on HBM bandwidth (reads n_groups int32 +
// all the metadata, writes 1 fp16).  For the worst-case 4096x4096
// shape with 32 groups, this is ~4 MiB of int32 acc + ~1 MiB meta
// vs the CUTLASS kernel's ~2 MiB weight stream — so the dequant
// kernel is expected to run ≤ 30% of the Stage B CUTLASS time, i.e.
// the full (CUTLASS + dequant) wall time stays within budget.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9 {
namespace cutlass_dequant {

namespace {

// Threads per block: 256 gives 4 CTAs for d_out*T=4096*128=512K.
constexpr int kThreads = 256;

__global__ void dequant_kernel(
    const int32_t* __restrict__ acc,      // (n_groups, d_out, T) int32
    const __half*  __restrict__ scale_u4, // (d_out, n_groups) fp16
    const __half*  __restrict__ zero_u4,  // (d_out, n_groups) fp16
    const int32_t* __restrict__ sum_X,    // (T, n_groups) int32
    const __half*  __restrict__ scale_x,  // (T,) fp16
    __half*        __restrict__ y_total,  // (d_out, T) row-major fp16
    int d_out, int T, int n_groups,
    int64_t acc_group_stride              // elements between acc groups
) {
    const int mt = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = d_out * T;
    if (mt >= total) return;

    const int m = mt / T;
    const int t = mt - m * T;
    const float ax = __half2float(scale_x[t]);

    float y = 0.0f;
    #pragma unroll 1
    for (int g = 0; g < n_groups; ++g) {
        const float acc_g = static_cast<float>(
            acc[g * acc_group_stride + m * T + t]);
        const float s_u4 = __half2float(scale_u4[m * n_groups + g]);
        const float z_u4 = __half2float(zero_u4[m * n_groups + g]);
        const float sx_g = static_cast<float>(sum_X[t * n_groups + g]);
        y += (acc_g - z_u4 * sx_g) * s_u4 * ax;
    }
    y_total[m * T + t] = __float2half(y);
}

}  // namespace

/// Public API — called from the CUTLASS launcher after all `n_groups`
/// Int4Gemm calls have populated the (n_groups, d_out, T) int32 workspace.
void launch_dequant(
    torch::Tensor acc_int32,   // (n_groups, d_out, T) int32 workspace
    torch::Tensor scale_u4,    // (d_out, n_groups) fp16
    torch::Tensor zero_u4,     // (d_out, n_groups) fp16
    torch::Tensor sum_X,       // (T, n_groups) int32
    torch::Tensor scale_x,     // (T,) fp16
    torch::Tensor Y_total,     // (d_out, T) fp16 — output
    int d_out, int T, int n_groups
) {
    const int total = d_out * T;
    const int blocks = (total + kThreads - 1) / kThreads;

    // Stride between groups in the int32 workspace. Caller guarantees
    // contiguous row-major (n_groups, d_out, T), so stride = d_out*T.
    const int64_t acc_group_stride =
        static_cast<int64_t>(d_out) * static_cast<int64_t>(T);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_kernel<<<blocks, kThreads, 0, stream>>>(
        acc_int32.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
        sum_X.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Y_total.data_ptr<at::Half>()),
        d_out, T, n_groups,
        acc_group_stride
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cutlass_dequant
}  // namespace hkust_v9