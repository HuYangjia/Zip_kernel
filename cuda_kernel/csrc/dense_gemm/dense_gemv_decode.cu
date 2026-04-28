// Dense UINT4 x SINT4 GEMV  (T=1 decode specialisation, SM89).
//
// Round 13: dedicated kernel for the T=1 path.  The general INT4 MMA
// kernel wastes 7/8 of its N-slice when T=1, so a specialised GEMV
// that uses dp4a (4090 full-rate) with 1 warp per output row is a
// better fit.
//
// Kernel shape
// ------------
//   Grid : (ceil(d_out / kBm),)   CTAs
//   Block: (32, kBm) threads     (1 warp per m-row, kBm m-rows per CTA)
//
// Each warp computes:
//   y[m] = scale_x[0] * sum_{g=0..n_groups-1}
//            scale_u4[m,g] * ( <W[m, g*128 : (g+1)*128], X[0, g*128:]> - zero_u4[m,g] * sum_X[0,g] )
//
// Per-warp work per group:
//   - Load 64 packed bytes of W row  (= 128 s4)
//   - Load 64 packed bytes of X row  (= 128 s4)   -- shared across all warps
//   - Unpack 4 bytes -> 8 s4 -> 2x int32 (8 s8)
//   - Four dp4a(W_reg[i], X_reg[i], acc) calls accumulate 16 s8 pairs = 32 s4 pairs
//   - Per thread: 4 packed bytes per group (128 s4 / 32 threads / 2 s4-per-nibble... wait re-derive)
//
// Derivation:
//   - Group has 128 s4.  Warp has 32 threads.  So 4 s4 per thread per group.
//   - 4 s4 = 2 packed bytes = 1 uint16.
//   - Unpack 2 packed bytes -> 4 s8 (= 1 uint32).
//   - 1 dp4a per thread per group.
//   - 32 threads * 1 dp4a = 32 dp4a instructions per warp per group.
//
// For d_in = 4096 => n_groups = 32 => 32*32 = 1024 dp4a per warp.
// One warp = one m-row, so the full kernel issues d_out * 32 dp4a total
// (d_out * n_groups dp4a per warp if we count d_in/group instead).
//
// Shared memory:
//   - s_X_packed[128]  : packed bytes of X for current group (reused by all warps in CTA)
//   - s_scale_x        : 1 fp16
//   - s_sum_X_g[32]    : per-group int32 sum_X[0, g] (prefetched once, n_groups <= 32)
//
// No MMA, no shmem-staged W (W is already 1 packed byte per thread per
// iteration; direct GMEM read with L2 cache hits is fine).
//
// Epilogue: per warp, lane 0 writes y[m] to GMEM (after warp reduce).

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace dense_gemv_decode {

constexpr int kMaxGroups = 160;

// s4 -> s8 unpack: 2 packed bytes (4 s4 values) -> one int32 (4 s8 lanes).
// Input layout (little-endian within each byte):
//   packed = [b1 b0]  (uint16)
//   b0 = (s1 << 4) | (s0 & 0x0F)
//   b1 = (s3 << 4) | (s2 & 0x0F)
// Output int32 layout for dp4a:
//   [s3, s2, s1, s0]  (MSB..LSB stored as 4 s8 lanes)
//   dp4a reads a uint32 as 4 signed int8 starting from LSB:
//     dp4a(a, b, c) = c + a.0*b.0 + a.1*b.1 + a.2*b.2 + a.3*b.3
//   where a.i is bits [8i+7 : 8i] sign-extended.
__device__ __forceinline__ uint32_t unpack_s4_to_s8x4(uint16_t packed) {
    // Extract 4 nibbles, each a s4 value in {-8..7}.
    auto s4_sext = [](unsigned nib) -> int {
        int v = nib & 0x0F;
        return v - ((v & 0x08) << 1);   // sign extend bit 3
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

template <int kBm>
__global__ void dense_gemv_decode_kernel(
    const uint8_t* __restrict__ W,         // (d_out, d_in/2) packed s4
    const uint8_t* __restrict__ X,         // (1, d_in/2)     packed s4
    const __half* __restrict__ scale_u4,   // (d_out, n_groups)
    const __half* __restrict__ zero_u4,    // (d_out, n_groups)
    const int* __restrict__ sum_X,         // (1, n_groups)
    const __half* __restrict__ scale_x,    // (1,)
    __half* __restrict__ Y,                // (d_out, 1)
    int d_out, int d_in, int n_groups,
    int64_t stride_w_m, int64_t stride_w_k,
    int64_t stride_x_k,
    int64_t stride_su_m, int64_t stride_su_g,
    int64_t stride_zu_m, int64_t stride_zu_g,
    int64_t stride_y_m
) {
    constexpr int kWarpSize = 32;
    constexpr int kGroupBytes = BCOL >> 1;   // 64 packed bytes per group

    const int lane = threadIdx.x;           // 0..31 within warp
    const int warp_id = threadIdx.y;        // 0..kBm-1 within CTA
    const int m = blockIdx.x * kBm + warp_id;

    // Per-thread workload per group: 4 s4 = 2 packed bytes = 1 uint16.
    // At lane l in {0..31}, this thread covers s4 cols [l*4, l*4+4) within
    // the current group's 128 s4 columns -> packed bytes [l*2, l*2+2).

    // Shmem: stage X for the current group (128 s4 = 64 packed bytes),
    // reused by all kBm warps in the CTA.  Also stage sum_X per group
    // (current Qwen3 decode shapes fit under kMaxGroups = 160).
    __shared__ alignas(16) uint8_t s_X[kGroupBytes];
    __shared__ int s_sum_X[kMaxGroups];
    __shared__ __half s_scale_x_val;

    // Prefetch sum_X (all groups) once.  First warp cooperatively loads.
    const int flat_tid = warp_id * kWarpSize + lane;
    if (flat_tid < n_groups) {
        s_sum_X[flat_tid] = sum_X[flat_tid];
    }
    if (flat_tid == 0) {
        s_scale_x_val = scale_x[0];
    }
    __syncthreads();

    if (m >= d_out) return;

    // Per-warp FP32 accumulator for y[m].
    float y_acc = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        // Cooperatively load X for this group into shmem.
        //   64 packed bytes = 16 uint32; 32 threads per warp * kBm warps.
        //   Distribute uint32 writes across warp 0 (lane 0..15).
        if (warp_id == 0 && lane < (kGroupBytes / 4)) {
            int64_t off = (int64_t)(g * kGroupBytes + lane * 4) * stride_x_k;
            reinterpret_cast<uint32_t*>(s_X)[lane] =
                *reinterpret_cast<const uint32_t*>(X + off);
        }
        __syncthreads();

        // This warp's W byte pair for row m, columns [lane*4..lane*4+4) in s4.
        int64_t w_off = (int64_t)m * stride_w_m
                      + (int64_t)(g * kGroupBytes + lane * 2) * stride_w_k;
        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(W + w_off);
        uint16_t x_packed = *reinterpret_cast<const uint16_t*>(&s_X[lane * 2]);

        uint32_t w_s8x4 = unpack_s4_to_s8x4(w_packed);
        uint32_t x_s8x4 = unpack_s4_to_s8x4(x_packed);

        // Per-thread partial dot product (signed s8 * s8 -> s32).
        int dot = 0;
        dot = __dp4a(static_cast<int>(w_s8x4), static_cast<int>(x_s8x4), dot);

        // Warp-level reduce (32 lanes -> lane 0 holds the group sum).
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
        }

        // Lane 0 applies per-group scale/zero correction and folds into y_acc.
        if (lane == 0) {
            float s = __half2float(scale_u4[(int64_t)m * stride_su_m
                                          + (int64_t)g * stride_su_g]);
            float z = __half2float(zero_u4 [(int64_t)m * stride_zu_m
                                          + (int64_t)g * stride_zu_g]);
            float sumxn = static_cast<float>(s_sum_X[g]);
            y_acc += s * (static_cast<float>(dot) - z * sumxn);
        }

        __syncthreads();  // before overwriting s_X for next group
    }

    if (lane == 0) {
        float sxn = __half2float(s_scale_x_val);
        Y[(int64_t)m * stride_y_m] = __float2half(y_acc * sxn);
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

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
    TORCH_CHECK(X_s4.size(0) == 1, "dense_gemv_decode requires T == 1");

    const int d_out = W_low.size(0);
    const int d_in_half = W_low.size(1);
    const int d_in = d_in_half * 2;
    TORCH_CHECK(d_in % BCOL == 0);
    const int n_groups = d_in / BCOL;
    TORCH_CHECK(n_groups <= kMaxGroups,
                "n_groups (", n_groups, ") > kMaxGroups (", kMaxGroups,
                ") unsupported in dense_gemv_decode");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kBm = 8;   // 8 warps per CTA = 256 threads
    dim3 block(32, kBm, 1);
    dim3 grid(ceil_div(d_out, kBm), 1, 1);

    dense_gemv_decode_kernel<kBm><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W_low.data_ptr<int8_t>()),
        reinterpret_cast<const uint8_t*>(X_s4.data_ptr<int8_t>()),
        reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
        sum_X.data_ptr<int>(),
        reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Y_low.data_ptr<at::Half>()),
        d_out, d_in, n_groups,
        W_low.stride(0), W_low.stride(1),
        X_s4.stride(1),
        scale_u4.stride(0), scale_u4.stride(1),
        zero_u4.stride(0), zero_u4.stride(1),
        Y_low.stride(0)
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dense_gemv_decode
}  // namespace hkust_v9
