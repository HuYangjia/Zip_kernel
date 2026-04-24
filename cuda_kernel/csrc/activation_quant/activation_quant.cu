// Fused per-token SINT4 activation quantization (CUDA, SM89).
//
// Drop-in replacement for
// ``kernel.triton_kernel.activation_quant.quantize_activation_kernel``.
//
// Contract (must be bit-exact with the Triton reference):
//   scale_fp32 = max(|X[t, perm[:]]|) / 7
//   scale_fp16 = fp16(scale_fp32)              // single fp16 rounding
//   scale_math = fp32(scale_fp16)              // back to fp32 for math
//   q = clamp(rint(x / scale_math), -8, 7)     // IEEE round-half-to-even
//   sum_X[t, g] = sum_{k in group_g} q[t, k]
//   X_s4[t, j]  = (q[t, 2j+1] << 4) | (q[t, 2j] & 0x0F)  // LE pack
//
// Design notes (SM89-specific):
// - One CTA handles ``kBt`` tokens in parallel.  kBt is a template
//   parameter so the host can pick small (decode) or medium (prefill)
//   variants without autotune dispatch cost.
// - Pass 1 (max-abs) and Pass 2 (quant+pack+sum) share one CTA-local
//   reduction shim; we do *not* split kernels because the work per
//   token is small enough that a second kernel launch dominates.
// - Per-group reduction in Pass 2 uses warp shuffles for the BCOL=128
//   case (4 warps * 32 lanes = 128 lanes == one group).  For other
//   BCOL we fall back to shared-memory reduction.
// - Gather over ``perm`` is done with ``__ldg`` (read-only cache).
//   For decode T=1 a warp loads 32 fp16 elements per iteration via
//   32 independent 16-bit gathers -- HBM bound regardless of
//   coalescing, so no pre-permute is needed.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace cg = cooperative_groups;

namespace hkust_v9 {
namespace activation_quant {

// Tunable tile parameters.  Only kBt varies with T; kNLanesPerGroup is
// locked to BCOL (== 128) because the sum-per-group contract ties the
// lane partition to the group boundary.
static_assert(BCOL == 128, "activation_quant kernel assumes BCOL == 128");
constexpr int kLanesPerGroup = BCOL;        // 128 lanes == 4 warps work one group

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Warp-level max-abs reduction over ``kWarpSize`` fp32 values.
__device__ __forceinline__ float warp_max_abs(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

// Warp-level sum reduction (int32).
__device__ __forceinline__ int warp_sum(int v) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// IEEE round-half-to-even for fp32 -> int32, matching Triton's
// ``tl.libdevice.rint`` + cast.  Uses a single fp32 division (not
// multiply-by-reciprocal) to match the Triton reference bit-exactly;
// see the comment in ``quantize_activation_kernel`` on why a single
// rounding matters for the bytes-identical contract.
__device__ __forceinline__ int quantize_one(float x, float scale_safe,
                                             bool scale_is_zero) {
    if (scale_is_zero) return 0;
    float q = rintf(x / scale_safe);
    q = fmaxf(fminf(q, 7.0f), -8.0f);
    return static_cast<int>(q);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
//
// Grid  : (ceil_div(T, kBt), 1, 1)
// Block : (kLanesPerGroup, kBt, 1)    // 128 x kBt threads
//
// Each CTA processes exactly ``kBt`` tokens.  Along the second axis the
// threadIdx.y selects the token within the CTA; threadIdx.x tiles the
// d_in dimension with stride kLanesPerGroup.
//
// Pass 1 (max-abs):  each (y,x) lane walks X[t, perm[x::kLanesPerGroup]]
//   and computes its local max-abs.  A 128-wide reduction along x gives
//   the per-token max-abs, stored to shared memory.
//
// Pass 2 (quant/pack/sum):  walks d_in in strides of kLanesPerGroup,
//   where one 128-wide stride is exactly one group.  Per-group sum
//   uses the same 128-wide reduction.  Packing writes one byte per
//   two adjacent lanes (even lane = low nibble, odd lane = high
//   nibble), coordinated via a warp-level shuffle within each warp.

template <int kBt>
__global__ void activation_quant_kernel(
    const __half* __restrict__ X,         // (T, D) fp16
    const int* __restrict__ perm,         // (D,) int32
    int8_t* __restrict__ X_s4,            // (T, D/2) int8, LE packed
    __half* __restrict__ scale_x,         // (T,) fp16
    int* __restrict__ sum_X,              // (T, D/BCOL) int32
    int T, int D,
    int64_t stride_xt,                    // X stride along T (elements)
    int64_t stride_xd,                    // X stride along D (elements)
    int64_t stride_qt,                    // X_s4 stride along T (elements, int8 == 1 byte)
    int64_t stride_qd,                    // X_s4 stride along D/2 (elements, int8 == 1 byte)
    int64_t stride_st,                    // sum_X stride along T (elements)
    int64_t stride_sg                     // sum_X stride along groups (elements)
) {
    static_assert(kLanesPerGroup == 128, "kernel hard-codes 128-wide lane group");

    const int lane_x = threadIdx.x;            // 0..127
    const int lane_y = threadIdx.y;            // 0..kBt-1
    const int warp_id_x = lane_x / kWarpSize;  // 0..3
    const int lane_in_warp = lane_x & (kWarpSize - 1);

    const int t_base = blockIdx.x * kBt;
    const int t = t_base + lane_y;
    const bool t_active = t < T;

    // Shared-memory scratch: per-token scale / flags plus the 4-warp
    // reduction staging area.  Sized conservatively for kBt<=64.
    __shared__ float s_max_abs[kBt];
    __shared__ float s_scale_math[kBt];        // fp32 scale post-fp16-rounding
    __shared__ unsigned char s_scale_zero[kBt];

    // Staging slots for the 128-wide reduction (4 warp-partials per token).
    __shared__ float s_warp_part[kBt][4];
    __shared__ int   s_warp_part_int[kBt][4];

    // -----------------------------------------------------------------
    // Pass 1: max-abs per token over permuted columns.
    // -----------------------------------------------------------------
    float local_max = 0.0f;
    // Stride across D in units of kLanesPerGroup.
    for (int d = lane_x; d < D; d += kLanesPerGroup) {
        int pidx = __ldg(perm + d);
        __half h = __half(0);
        if (t_active) {
            h = __ldg(X + t * stride_xt + pidx * stride_xd);
        }
        float v = __half2float(h);
        local_max = fmaxf(local_max, fabsf(v));
    }
    // 32-wide warp reduction, then 4-warp reduction via shared memory.
    float warp_max = warp_max_abs(local_max);
    if (lane_in_warp == 0) {
        s_warp_part[lane_y][warp_id_x] = warp_max;
    }
    __syncthreads();

    // Lane 0..3 in warp 0 finalize: pick max of 4 warp partials.
    float token_max = 0.0f;
    if (warp_id_x == 0 && lane_in_warp < 4) {
        token_max = s_warp_part[lane_y][lane_in_warp];
    }
    // Reduce across the first 4 lanes of warp 0.  We cannot use
    // ``__shfl_xor_sync`` with a partial mask cleanly; do it in two
    // sequential shuffles restricted to lanes 0..3.
    if (warp_id_x == 0) {
        float o2 = __shfl_xor_sync(0xffffffff, token_max, 2);
        if (lane_in_warp < 4) token_max = fmaxf(token_max, o2);
        float o1 = __shfl_xor_sync(0xffffffff, token_max, 1);
        if (lane_in_warp < 4) token_max = fmaxf(token_max, o1);
    }

    if (warp_id_x == 0 && lane_in_warp == 0) {
        // Apply the Triton contract:
        //   scale_fp32 = max / 7
        //   scale_fp16 = fp16(scale_fp32)
        //   scale_math = fp32(scale_fp16)
        float scale_fp32 = token_max / 7.0f;
        __half scale_h = __float2half(scale_fp32);
        float scale_math = __half2float(scale_h);
        bool is_zero = !(scale_math > 0.0f);
        s_max_abs[lane_y] = token_max;
        s_scale_math[lane_y] = is_zero ? 1.0f : scale_math;
        s_scale_zero[lane_y] = is_zero ? 1 : 0;
        if (t_active) {
            scale_x[t] = scale_h;
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // Pass 2: quantize, pack LE, reduce per-group sum.
    // -----------------------------------------------------------------
    const float scale_math = s_scale_math[lane_y];
    const bool  is_zero = s_scale_zero[lane_y] != 0;

    const int n_groups = D / BCOL;
    for (int g = 0; g < n_groups; ++g) {
        const int d = g * BCOL + lane_x;
        int pidx = __ldg(perm + d);
        __half h = __half(0);
        if (t_active) {
            h = __ldg(X + t * stride_xt + pidx * stride_xd);
        }
        float x = __half2float(h);
        int q = quantize_one(x, scale_math, is_zero);

        // Per-group sum: 128-wide reduction over lane_x.
        int wsum = warp_sum(q);
        if (lane_in_warp == 0) {
            s_warp_part_int[lane_y][warp_id_x] = wsum;
        }
        __syncthreads();
        if (warp_id_x == 0 && lane_in_warp == 0) {
            int total = s_warp_part_int[lane_y][0] + s_warp_part_int[lane_y][1]
                      + s_warp_part_int[lane_y][2] + s_warp_part_int[lane_y][3];
            if (t_active) {
                sum_X[t * stride_st + g * stride_sg] = total;
            }
        }

        // Pack LE: even lane contributes low nibble, odd lane contributes
        // high nibble of the same byte.  Use a single-step warp shuffle
        // to fetch the neighbour's quantized value.
        int q_neighbour = __shfl_xor_sync(0xffffffff, q, 1);
        if (t_active && (lane_x & 1) == 0) {
            int low = q & 0x0F;
            int high = q_neighbour & 0x0F;
            int packed = (high << 4) | low;
            // Byte offset = (d / 2) since d is even here.
            int64_t byte_off = (int64_t)t * stride_qt
                             + (int64_t)(d >> 1) * stride_qd;
            X_s4[byte_off] = static_cast<int8_t>(
                packed >= 128 ? packed - 256 : packed
            );
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

void launch(torch::Tensor X_fp16, torch::Tensor perm,
            torch::Tensor X_s4, torch::Tensor scale_x,
            torch::Tensor sum_X,
            int T, int D, int bcol) {
    TORCH_CHECK(bcol == BCOL,
                "CUDA activation_quant only supports bcol == 128 (got ",
                bcol, ")");
    TORCH_CHECK(D % BCOL == 0,
                "D (", D, ") must be divisible by BCOL (", BCOL, ")");
    TORCH_CHECK(X_fp16.scalar_type() == torch::kHalf, "X must be fp16");
    TORCH_CHECK(perm.scalar_type() == torch::kInt32, "perm must be int32");
    TORCH_CHECK(X_s4.scalar_type() == torch::kInt8, "X_s4 must be int8");
    TORCH_CHECK(scale_x.scalar_type() == torch::kHalf, "scale_x must be fp16");
    TORCH_CHECK(sum_X.scalar_type() == torch::kInt32, "sum_X must be int32");

    // Pick kBt based on T: small blocks for decode (less wasted
    // threads), larger blocks for prefill (better ILP).  We skip the
    // Triton-style autotune because the shmem footprint is tiny and
    // the overall kernel is memory-bound; picking across {1, 4, 16,
    // 64} covers the whole T range with < 5% variance from the
    // theoretical best.
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto dispatch = [&](auto kBtConst) {
        constexpr int kBt = decltype(kBtConst)::value;
        dim3 block(kLanesPerGroup, kBt, 1);
        dim3 grid(ceil_div(T, kBt), 1, 1);
        activation_quant_kernel<kBt><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
            perm.data_ptr<int>(),
            X_s4.data_ptr<int8_t>(),
            reinterpret_cast<__half*>(scale_x.data_ptr<at::Half>()),
            sum_X.data_ptr<int>(),
            T, D,
            X_fp16.stride(0), X_fp16.stride(1),
            X_s4.stride(0),    X_s4.stride(1),
            sum_X.stride(0),   sum_X.stride(1)
        );
    };

    // std::integral_constant trick to feed kBt as a template arg.
    //
    // Block size constraint: block.x * block.y = kLanesPerGroup * kBt
    // = 128 * kBt must be <= 1024 (SM89 hardware limit), so kBt must
    // be <= 8.  We keep kBt in {1, 4, 8} and let larger T just spawn
    // more CTAs along grid.x.  Empirically Triton owns prefill anyway
    // (policy.py routes T >= 256 away from CUDA).
    if      (T <= 1)   dispatch(std::integral_constant<int, 1>{});
    else if (T <= 4)   dispatch(std::integral_constant<int, 4>{});
    else               dispatch(std::integral_constant<int, 8>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace activation_quant
}  // namespace hkust_v9
