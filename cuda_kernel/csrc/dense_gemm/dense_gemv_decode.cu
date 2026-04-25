// Dense UINT4 x SINT4 batched GEMV  (T <= 16 decode/small-batch, SM89).
//
// Round 13: dedicated kernel for T=1 (dp4a, 1 warp per m-row).
// Round 28: generalised to T <= 16 ("batched GEMV").  For each group,
//   the warp loads one W byte pair and performs T dp4a's against T
//   different X byte pairs.  W-register reuse + L1/shmem X reuse.
//
//   MMA kBn=8 path (previously used for T=8..16) is wave-underfilled:
//   grid = (d_out/128) * ceil(T/8) = 32..64 CTAs at d_out=4096
//   -> <= 0.5 waves on SM89 (128 SMs).  Batched-GEMV with kBm=8 gives
//   grid = d_out/8 = 512 CTAs = 4 waves, filling the device properly.
//
// Kernel shape:
//   Grid : (ceil(d_out / kBm),)   CTAs
//   Block: (32, kBm) threads     (1 warp per m-row, kBm m-rows per CTA)
//   Template kBT in {1,4,8,16} picked by host based on T.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace dense_gemv_decode {

__device__ __forceinline__ uint32_t unpack_s4_to_s8x4(uint16_t packed) {
    auto s4_sext = [](unsigned nib) -> int {
        int v = nib & 0x0F;
        return v - ((v & 0x08) << 1);
    };
    int v0 = s4_sext( packed        & 0x0F);
    int v1 = s4_sext((packed >>  4) & 0x0F);
    int v2 = s4_sext((packed >>  8) & 0x0F);
    int v3 = s4_sext((packed >> 12) & 0x0F);
    return (static_cast<uint32_t>(static_cast<uint8_t>(v3)) << 24) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v2)) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v1)) <<  8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(v0))      );
}

template <int kBm, int kBT>
__global__ void dense_gemv_decode_kernel(
    const uint8_t* __restrict__ W,
    const uint8_t* __restrict__ X,
    const __half* __restrict__ scale_u4,
    const __half* __restrict__ zero_u4,
    const int* __restrict__ sum_X,
    const __half* __restrict__ scale_x,
    __half* __restrict__ Y,
    int d_out, int d_in, int T, int n_groups,
    int64_t stride_w_m, int64_t stride_w_k,
    int64_t stride_x_t, int64_t stride_x_k,
    int64_t stride_su_m, int64_t stride_su_g,
    int64_t stride_zu_m, int64_t stride_zu_g,
    int64_t stride_sx_t,
    int64_t stride_sumx_t, int64_t stride_sumx_g,
    int64_t stride_y_m, int64_t stride_y_t
) {
    constexpr int kWarpSize = 32;
    constexpr int kGroupBytes = BCOL >> 1;   // 64 packed bytes per group

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int m = blockIdx.x * kBm + warp_id;

    __shared__ alignas(16) uint8_t s_X[kBT][kGroupBytes];
    __shared__ int s_sum_X[kBT][128];
    __shared__ __half s_scale_x[kBT];

    const int flat_tid = warp_id * kWarpSize + lane;
    const int cta_threads = kBm * kWarpSize;

    // Prefetch sum_X for all T rows.
    #pragma unroll 1
    for (int i = flat_tid; i < kBT * n_groups; i += cta_threads) {
        int t = i / n_groups;
        int g = i - t * n_groups;
        if (t < T) {
            s_sum_X[t][g] = sum_X[(int64_t)t * stride_sumx_t
                                + (int64_t)g * stride_sumx_g];
        }
    }
    if (flat_tid < kBT) {
        s_scale_x[flat_tid] = (flat_tid < T)
            ? scale_x[(int64_t)flat_tid * stride_sx_t]
            : __float2half(0.0f);
    }
    __syncthreads();

    if (m >= d_out) return;

    float y_acc[kBT];
    #pragma unroll
    for (int t = 0; t < kBT; ++t) y_acc[t] = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        // Cooperatively load T * 64 B of X into shmem.
        int total_u32 = kBT * (kGroupBytes / 4);
        #pragma unroll 1
        for (int i = flat_tid; i < total_u32; i += cta_threads) {
            int t = i / (kGroupBytes / 4);
            int u = i - t * (kGroupBytes / 4);
            if (t < T) {
                int64_t off = (int64_t)t * stride_x_t
                            + (int64_t)(g * kGroupBytes + u * 4) * stride_x_k;
                reinterpret_cast<uint32_t*>(&s_X[t][0])[u] =
                    *reinterpret_cast<const uint32_t*>(X + off);
            }
        }
        __syncthreads();

        int64_t w_off = (int64_t)m * stride_w_m
                      + (int64_t)(g * kGroupBytes + lane * 2) * stride_w_k;
        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(W + w_off);
        uint32_t w_s8x4 = unpack_s4_to_s8x4(w_packed);

        int dot[kBT];
        #pragma unroll
        for (int t = 0; t < kBT; ++t) {
            if (t < T) {
                uint16_t x_packed =
                    *reinterpret_cast<const uint16_t*>(&s_X[t][lane * 2]);
                uint32_t x_s8x4 = unpack_s4_to_s8x4(x_packed);
                dot[t] = __dp4a(static_cast<int>(w_s8x4),
                                static_cast<int>(x_s8x4), 0);
            } else {
                dot[t] = 0;
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            #pragma unroll
            for (int t = 0; t < kBT; ++t) {
                dot[t] += __shfl_xor_sync(0xFFFFFFFF, dot[t], off);
            }
        }

        if (lane == 0) {
            float s = __half2float(scale_u4[(int64_t)m * stride_su_m
                                          + (int64_t)g * stride_su_g]);
            float z = __half2float(zero_u4 [(int64_t)m * stride_zu_m
                                          + (int64_t)g * stride_zu_g]);
            #pragma unroll
            for (int t = 0; t < kBT; ++t) {
                if (t < T) {
                    float sumxn = static_cast<float>(s_sum_X[t][g]);
                    y_acc[t] += s * (static_cast<float>(dot[t]) - z * sumxn);
                }
            }
        }

        __syncthreads();
    }

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < kBT; ++t) {
            if (t < T) {
                float sxn = __half2float(s_scale_x[t]);
                Y[(int64_t)m * stride_y_m + (int64_t)t * stride_y_t] =
                    __float2half(y_acc[t] * sxn);
            }
        }
    }
}

void launch(
    torch::Tensor W_low, torch::Tensor X_s4,
    torch::Tensor scale_u4, torch::Tensor zero_u4,
    torch::Tensor sum_X, torch::Tensor scale_x,
    torch::Tensor Y_low
) {
    TORCH_CHECK(W_low.dtype() == torch::kInt8);
    TORCH_CHECK(X_s4.dtype() == torch::kInt8);
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf);
    TORCH_CHECK(zero_u4.dtype() == torch::kHalf);
    TORCH_CHECK(sum_X.dtype() == torch::kInt32);
    TORCH_CHECK(scale_x.dtype() == torch::kHalf);
    TORCH_CHECK(Y_low.dtype() == torch::kHalf);

    const int d_out = W_low.size(0);
    const int d_in_half = W_low.size(1);
    const int d_in = d_in_half * 2;
    const int T = X_s4.size(0);
    TORCH_CHECK(d_in % BCOL == 0);
    TORCH_CHECK(T >= 1 && T <= 16, "dense_gemv_decode requires T in [1,16]");
    const int n_groups = d_in / BCOL;
    TORCH_CHECK(n_groups <= 128, "n_groups > 128 unsupported");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kBm = 8;

    auto do_launch = [&](auto kBT_c) {
        constexpr int kBT = decltype(kBT_c)::value;
        dim3 block(32, kBm, 1);
        dim3 grid(ceil_div(d_out, kBm), 1, 1);
        dense_gemv_decode_kernel<kBm, kBT><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(W_low.data_ptr<int8_t>()),
            reinterpret_cast<const uint8_t*>(X_s4.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
            sum_X.data_ptr<int>(),
            reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(Y_low.data_ptr<at::Half>()),
            d_out, d_in, T, n_groups,
            W_low.stride(0), W_low.stride(1),
            X_s4.stride(0), X_s4.stride(1),
            scale_u4.stride(0), scale_u4.stride(1),
            zero_u4.stride(0), zero_u4.stride(1),
            scale_x.stride(0),
            sum_X.stride(0), sum_X.stride(1),
            Y_low.stride(0), Y_low.stride(1)
        );
    };

    if      (T <= 1)  do_launch(std::integral_constant<int, 1>{});
    else if (T <= 4)  do_launch(std::integral_constant<int, 4>{});
    else if (T <= 8)  do_launch(std::integral_constant<int, 8>{});
    else              do_launch(std::integral_constant<int, 16>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dense_gemv_decode
}  // namespace hkust_v9
