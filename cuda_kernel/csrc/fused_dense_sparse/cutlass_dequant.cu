// Copyright (c) 2026 HKUST R50 — Stage A1 / A3 dequant kernels.
//
// A1 (original): single-kernel reduce over the 3D int32 workspace
//   acc_s32[g, m, t] -> Y_fp16[m, t] with the V9 dequant formula:
//
//     y[m, t] = sum_g (acc[g,m,t] - zero[m,g]*sum_X[t,g]) * scale[m,g] * scale_x[t]
//
//   This was clean but needed an (n_groups, d_out, T) int32 buffer.
//   For large shapes (e.g. gate_up T=512) the buffer exceeds 1 GB and
//   the extra HBM round-trip dominates — kernel ends up 8x slower
//   than the hand-rolled fused kernel.  See r62 F4 diagnosis.
//
// A3 (this file, 2026-04-30): split into two kernels so the launcher
//   can run GEMM + dequant streamed per-group and retire the workspace
//   entirely:
//
//     dequant_accum_kernel(acc_per_group, g, Y_fp32)
//        acc_per_group : (d_out, T) int32 (reused across groups)
//        Y_fp32        : (d_out, T) fp32 accumulator (reused across groups)
//        g             : current group index; scale/zero/sum_X/scale_x
//                        are indexed by g
//
//     finalize_fp32_to_fp16_kernel(Y_fp32, Y_total)
//        converts the accumulator to fp16 once, after all groups are done.
//
//   The (d_out, T) buffer is typically a few MB and L2-resident, so
//   HBM traffic collapses to what the legacy kernel pays.

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

// -------------------------------------------------------------------------
// A3 kernel 1: consume one group's int32 acc, update Y_fp32 accumulator.
// -------------------------------------------------------------------------
//
// Each thread owns one (m, t) output element.  For group ``g``, we read
//   acc[m, t]           — int32 from the per-group workspace
//   scale_u4[m, g]      — fp16 broadcast along t
//   zero_u4[m, g]       — fp16 broadcast along t
//   sum_X[t, g]         — int32 broadcast along m
// and add the g-th dequant contribution to Y_fp32[m, t].
// scale_x[t] is applied only at the end in the finalize kernel; we can
// do that because scale_x is t-only (not g-dependent), so it factors
// out of the per-group sum:
//   y[m,t] = scale_x[t] * sum_g { (acc[g,m,t] - zero[m,g]*sum_X[t,g]) * scale[m,g] }
__global__ void dequant_accum_kernel(
    const int32_t* __restrict__ acc_g,    // (d_out, T) int32 for this group
    const __half*  __restrict__ scale_u4, // (d_out, n_groups) fp16
    const __half*  __restrict__ zero_u4,  // (d_out, n_groups) fp16
    const int32_t* __restrict__ sum_X,    // (T, n_groups) int32
    float*         __restrict__ y_fp32,   // (d_out, T) fp32 accumulator
    int d_out, int T, int n_groups,
    int g                                  // current group index
) {
    const int mt = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = d_out * T;
    if (mt >= total) return;

    const int m = mt / T;
    const int t = mt - m * T;

    const float acc_v = static_cast<float>(acc_g[m * T + t]);
    const float s_u4  = __half2float(scale_u4[m * n_groups + g]);
    const float z_u4  = __half2float(zero_u4 [m * n_groups + g]);
    const float sx_g  = static_cast<float>(sum_X[t * n_groups + g]);

    // Partial contribution of group g (scale_x[t] applied in finalize).
    const float add = (acc_v - z_u4 * sx_g) * s_u4;

    // For g == 0 we overwrite (avoids needing a prior cudaMemsetAsync).
    if (g == 0) {
        y_fp32[m * T + t] = add;
    } else {
        y_fp32[m * T + t] += add;
    }
}

// -------------------------------------------------------------------------
// A3 kernel 2: scale by scale_x[t], cast to fp16, write to Y_total.
// -------------------------------------------------------------------------
__global__ void finalize_fp32_to_fp16_kernel(
    const float*  __restrict__ y_fp32,    // (d_out, T) fp32
    const __half* __restrict__ scale_x,   // (T,) fp16
    __half*       __restrict__ y_total,   // (d_out, T) fp16
    int d_out, int T
) {
    const int mt = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = d_out * T;
    if (mt >= total) return;

    const int m = mt / T;
    const int t = mt - m * T;
    const float ax = __half2float(scale_x[t]);
    y_total[m * T + t] = __float2half(y_fp32[m * T + t] * ax);
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

/// A3: update fp32 accumulator with the g-th group's dequant contribution.
/// Launcher must call this exactly once per g in [0, n_groups), in any order
/// (addition is commutative) AS LONG AS g==0 is called first or the
/// accumulator is zeroed by the caller beforehand.  The simpler pattern
/// is "call with g in order 0,1,...,n_groups-1 on an uninitialised
/// buffer" — g==0 overwrites, rest add.
void launch_dequant_accum(
    torch::Tensor acc_int32_g, // (d_out, T) int32 — this group's workspace
    torch::Tensor scale_u4,    // (d_out, n_groups) fp16
    torch::Tensor zero_u4,     // (d_out, n_groups) fp16
    torch::Tensor sum_X,       // (T, n_groups) int32
    torch::Tensor Y_fp32,      // (d_out, T) fp32 accumulator
    int d_out, int T, int n_groups, int g
) {
    const int total = d_out * T;
    const int blocks = (total + kThreads - 1) / kThreads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_accum_kernel<<<blocks, kThreads, 0, stream>>>(
        acc_int32_g.data_ptr<int32_t>(),
        reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
        sum_X.data_ptr<int32_t>(),
        Y_fp32.data_ptr<float>(),
        d_out, T, n_groups, g
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/// A3: scale by per-token scale_x, cast to fp16, store into Y_total.
void launch_finalize_fp32_to_fp16(
    torch::Tensor Y_fp32,      // (d_out, T) fp32
    torch::Tensor scale_x,     // (T,) fp16
    torch::Tensor Y_total,     // (d_out, T) fp16 — output
    int d_out, int T
) {
    const int total = d_out * T;
    const int blocks = (total + kThreads - 1) / kThreads;
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    finalize_fp32_to_fp16_kernel<<<blocks, kThreads, 0, stream>>>(
        Y_fp32.data_ptr<float>(),
        reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Y_total.data_ptr<at::Half>()),
        d_out, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace cutlass_dequant
}  // namespace hkust_v9