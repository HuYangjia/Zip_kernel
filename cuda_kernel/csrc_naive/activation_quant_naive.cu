// Naive per-token SINT4 activation quantization (CUDA, SM89).
//
// This is the *textbook* implementation kept as a reference baseline
// for the multi-iteration optimised kernel in
// ``csrc/activation_quant/activation_quant.cu``.
//
// Contract (must match the optimised version up to fp16 ulp):
//   scale_fp32 = max(|X[t, perm[:]]|) / 7
//   scale_fp16 = fp16(scale_fp32)                (single fp16 rounding)
//   scale_math = fp32(scale_fp16)
//   q = clamp(rint(x / scale_math), -8, 7)
//   sum_X[t, g] = sum_{k in group_g} q[t, k]
//   X_s4[t, j]  = (q[t, 2j+1] << 4) | (q[t, 2j] & 0x0F)    (LE pack)
//
// "Naive" means:
//   * 1 CTA per token (grid.x = T).
//   * ``blockDim.x = 128`` threads; shmem-only reductions
//     (no warp shuffles, no cooperative_groups).
//   * Plain IEEE ``x / scale`` (not ``__fdividef``).
//   * Packing handled by giving each thread ownership of a
//     contiguous (k_even, k_odd) pair — no atomics, no CAS.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9_naive {
namespace activation_quant {

static constexpr int kBCOL = 128;
static constexpr int kBlockThreads = 128;

// One CTA = one token.
//   Grid  : (T, 1, 1)
//   Block : (128, 1, 1)
__global__ void activation_quant_naive_kernel(
    const __half* __restrict__ X,      // (T, D) fp16, contiguous
    const int*    __restrict__ perm,   // (D,) int32
    int8_t*       __restrict__ X_s4,   // (T, D/2) packed int8 (LE)
    __half*       __restrict__ scale_x,// (T,) fp16
    int*          __restrict__ sum_X,  // (T, D/128) int32
    int T, int D
) {
    const int t   = blockIdx.x;
    const int tid = threadIdx.x;
    if (t >= T) return;

    const int n_groups = D / kBCOL;
    const __half* Xt     = X    + (int64_t)t * D;
    int8_t*       X_s4_t = X_s4 + (int64_t)t * (D / 2);
    int*          sum_t  = sum_X + (int64_t)t * n_groups;

    __shared__ float  s_maxabs[kBlockThreads];
    __shared__ float  s_scale_math;
    __shared__ int    s_scale_is_zero;
    // Per-group sum accumulator.  Naive impl: one int per group.
    // D up to 17408 → n_groups up to 136; cap at 160 (covers all Qwen3).
    constexpr int kMaxGroups = 160;
    __shared__ int s_group_sum[kMaxGroups];

    // ---------------------------------------------------------------
    // Pass 1: find max(|X[t, :]|) — strided gather via perm.
    // ---------------------------------------------------------------
    float local_max = 0.0f;
    for (int k = tid; k < D; k += kBlockThreads) {
        int pk = perm[k];
        float v = fabsf(__half2float(Xt[pk]));
        if (v > local_max) local_max = v;
    }
    s_maxabs[tid] = local_max;
    __syncthreads();

    // Shmem tree reduction (no warp shuffles).
    for (int stride = kBlockThreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float a = s_maxabs[tid];
            float b = s_maxabs[tid + stride];
            s_maxabs[tid] = (a > b) ? a : b;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float max_abs = s_maxabs[0];
        float scale_fp32 = max_abs / 7.0f;
        __half s_h = __float2half(scale_fp32);
        s_scale_math    = __half2float(s_h);
        s_scale_is_zero = (s_scale_math == 0.0f) ? 1 : 0;
        scale_x[t]      = s_h;
    }
    // Zero out ALL group-sum slots (n_groups can exceed blockDim,
    // e.g. Qwen3-14B down_proj has n_groups=136 > 128 threads per CTA).
    for (int i = tid; i < kMaxGroups; i += kBlockThreads) {
        s_group_sum[i] = 0;
    }
    __syncthreads();

    const float s_math = s_scale_math;
    const bool  s_zero = (s_scale_is_zero != 0);

    // ---------------------------------------------------------------
    // Pass 2: quant + pack (each thread owns a byte = 2 adjacent k's)
    //          and accumulate per-group sums via shmem atomicAdd.
    // ---------------------------------------------------------------
    //
    // Byte index b ∈ [0, D/2) covers (k_low=2b, k_high=2b+1).  We give
    // each thread a strided slice of byte indices — no two threads
    // ever write the same byte, so packing needs no atomics.
    //
    // Group index for byte b: g = (2b) / 128 = b / 64 (since both
    // k_low and k_high always fall into the same group — groups are
    // 128 elements = 64 bytes wide).
    const int n_bytes = D / 2;
    for (int b = tid; b < n_bytes; b += kBlockThreads) {
        int k_lo = 2 * b;
        int k_hi = 2 * b + 1;

        int p_lo = perm[k_lo];
        int p_hi = perm[k_hi];

        float x_lo = __half2float(Xt[p_lo]);
        float x_hi = __half2float(Xt[p_hi]);

        int q_lo, q_hi;
        if (s_zero) {
            q_lo = 0;
            q_hi = 0;
        } else {
            float qf_lo = rintf(x_lo / s_math);
            float qf_hi = rintf(x_hi / s_math);
            if (qf_lo > 7.0f)  qf_lo = 7.0f;
            if (qf_lo < -8.0f) qf_lo = -8.0f;
            if (qf_hi > 7.0f)  qf_hi = 7.0f;
            if (qf_hi < -8.0f) qf_hi = -8.0f;
            q_lo = static_cast<int>(qf_lo);
            q_hi = static_cast<int>(qf_hi);
        }

        // Pack: low nibble = q_lo, high nibble = q_hi.  Store as
        // int8 (value range -128..127).  Equivalent LE packing to
        // the optimised kernel.
        uint8_t byte = static_cast<uint8_t>(
            (static_cast<uint8_t>(q_lo) & 0x0Fu) |
            ((static_cast<uint8_t>(q_hi) & 0x0Fu) << 4));
        X_s4_t[b] = static_cast<int8_t>(byte);

        // Per-group sum: each group is 128 elements = 64 bytes.
        int g = b / 64;
        atomicAdd(&s_group_sum[g], q_lo + q_hi);
    }
    __syncthreads();

    // Write out per-group sums (n_groups can exceed blockDim_x = 128).
    for (int g = tid; g < n_groups; g += kBlockThreads) {
        sum_t[g] = s_group_sum[g];
    }
}

// ---------------------------------------------------------------------
// Host launcher (ABI-identical to the optimised version's launch()).
// ---------------------------------------------------------------------
void launch(torch::Tensor X_fp16, torch::Tensor perm,
            torch::Tensor X_s4, torch::Tensor scale_x,
            torch::Tensor sum_X,
            int T, int D, int bcol)
{
    TORCH_CHECK(bcol == kBCOL, "naive activation_quant hard-codes BCOL=128");
    TORCH_CHECK(X_fp16.is_cuda() && X_fp16.dtype() == torch::kHalf);
    TORCH_CHECK(perm.dtype() == torch::kInt32);
    TORCH_CHECK(X_s4.dtype() == torch::kInt8);
    TORCH_CHECK(scale_x.dtype() == torch::kHalf);
    TORCH_CHECK(sum_X.dtype() == torch::kInt32);
    TORCH_CHECK(D % kBCOL == 0, "D must be divisible by 128");
    TORCH_CHECK(D % 2 == 0, "D must be even for 4-bit packing");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 grid(T), block(kBlockThreads);

    activation_quant_naive_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        perm.data_ptr<int>(),
        X_s4.data_ptr<int8_t>(),
        reinterpret_cast<__half*>(scale_x.data_ptr<at::Half>()),
        sum_X.data_ptr<int>(),
        T, D
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace activation_quant
}  // namespace hkust_v9_naive
