// Fused Activation Quantization + Dense+Sparse GEMV (T=1 decode, SM89).
//
// Round 15c: All kBm warps cooperate on act_quant (parallel across groups),
// then each warp computes its own GEMV row.  This eliminates the serial
// bottleneck of the Round 15b design (warp 0 only).
//
// Architecture:
//   Grid:  (ceil_div(d_out, kBm),)
//   Block: (32, kBm)  -- 1 warp per output row
//
//   Phase A (act_quant, all warps cooperate):
//     A1. Max-abs: each warp handles d_in/kBm elements, CTA-wide reduce.
//     A2. Quant+pack+sum: warp w handles groups [w, w+kBm, w+2*kBm, ...].
//         Each warp does 4 passes of 32 lanes over its 128-element group.
//         Writes X_s4 and sum_X to shmem.
//
//   Phase B (GEMV, each warp independent):
//     Dense + sparse dot product using shmem X_s4 and sum_X.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9 {
namespace fused_quant_gemv {

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

__device__ __forceinline__ float warp_max_abs_f(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, off));
    return v;
}

__device__ __forceinline__ int warp_sum_i(int v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

constexpr int kMaxGroups = 128;

template <int kBm>
__global__ void fused_quant_gemv_kernel(
    const __half* __restrict__ X_fp16,
    const int*    __restrict__ perm,
    int64_t stride_xd,
    const uint8_t* __restrict__ W_low,
    int64_t stride_w_m, int64_t stride_w_k,
    const __half* __restrict__ scale_u4,
    const __half* __restrict__ zero_u4,
    int64_t stride_su_m, int64_t stride_su_g,
    int64_t stride_zu_m, int64_t stride_zu_g,
    const uint8_t* __restrict__ W_high_blocks,
    const int*     __restrict__ hp_row_offsets,
    const int*     __restrict__ hp_col_indices,
    int64_t stride_wb_blk, int64_t stride_wb_r, int64_t stride_wb_k,
    __half* __restrict__ Y,
    int64_t stride_y_m,
    int d_out, int d_in, int n_groups
) {
    constexpr int kGroupBytes = BCOL >> 1;  // 64 packed bytes per group
    constexpr int kCTASize = 32 * kBm;      // total threads in CTA

    const int lane    = threadIdx.x;   // 0..31
    const int warp_id = threadIdx.y;   // 0..kBm-1
    const int flat_tid = warp_id * 32 + lane;
    const int m = blockIdx.x * kBm + warp_id;
    const int br = (blockIdx.x * kBm) / BROW;
    const int m_in_br = m - br * BROW;

    // ----------------------------------------------------------------
    // Shared memory
    // ----------------------------------------------------------------
    __shared__ int8_t  s_X_s4[kMaxGroups * (BCOL / 2)];  // packed INT4
    __shared__ int     s_sum_X[kMaxGroups];
    __shared__ __half  s_scale_x_h;
    __shared__ float   s_scale_math;
    __shared__ int     s_is_zero;
    // Per-warp max-abs partial (kBm floats).
    __shared__ float   s_warp_max[kBm];
    // Round 26: prefetch scale_u4 and zero_u4 per (m, g) to shmem so
    // the dense GEMV loop does not go to HBM for each group.  Each
    // warp owns exactly one m-row, so layout is s_scale_u4[warp][g].
    __shared__ __half  s_scale_u4_w[kBm][kMaxGroups];
    __shared__ __half  s_zero_u4_w [kBm][kMaxGroups];

    // ================================================================
    // Phase A1: Max-abs (all warps cooperate, each handles d_in/kBm elems)
    // ================================================================
    float local_max = 0.0f;
    // Stride over d_in with step kCTASize (all threads together).
    for (int d = flat_tid; d < d_in; d += kCTASize) {
        int pidx = __ldg(perm + d);
        float v = __half2float(__ldg(X_fp16 + (int64_t)pidx * stride_xd));
        local_max = fmaxf(local_max, fabsf(v));
    }

    // Round 26: prefetch scale_u4 / zero_u4 for ALL rows of this CTA
    // in parallel with the max-abs reduction.  Issued as independent
    // loads that overlap the reduction's shuffle chain.
    //   Rows owned by this CTA: warp_id in [0, kBm).
    //   Each thread loads (kBm * n_groups) / kCTASize entries.
    for (int idx = flat_tid; idx < kBm * n_groups; idx += kCTASize) {
        int w = idx / n_groups;
        int g = idx - w * n_groups;
        int m_g = blockIdx.x * kBm + w;
        if (m_g < d_out) {
            s_scale_u4_w[w][g] = __ldg(scale_u4 + (int64_t)m_g * stride_su_m
                                                + (int64_t)g * stride_su_g);
            s_zero_u4_w [w][g] = __ldg(zero_u4  + (int64_t)m_g * stride_zu_m
                                                + (int64_t)g * stride_zu_g);
        } else {
            s_scale_u4_w[w][g] = __half(0);
            s_zero_u4_w [w][g] = __half(0);
        }
    }

    // Warp-level reduce.
    float wmax = warp_max_abs_f(local_max);
    if (lane == 0) s_warp_max[warp_id] = wmax;
    __syncthreads();

    // Thread 0 finalises global max and writes scale.
    if (flat_tid == 0) {
        float gmax = 0.0f;
        #pragma unroll
        for (int w = 0; w < kBm; ++w) gmax = fmaxf(gmax, s_warp_max[w]);
        float scale_fp32 = gmax / 7.0f;
        __half scale_h = __float2half(scale_fp32);
        float scale_math = __half2float(scale_h);
        bool iz = !(scale_math > 0.0f);
        s_scale_x_h  = scale_h;
        s_scale_math = iz ? 1.0f : scale_math;
        s_is_zero    = iz ? 1 : 0;
    }
    __syncthreads();

    // ================================================================
    // Phase A2: Quant + pack + sum (each warp handles its own groups)
    // ================================================================
    const float scale_math = s_scale_math;
    const bool  is_zero    = s_is_zero != 0;

    // Warp w handles groups: w, w+kBm, w+2*kBm, ...
    for (int g = warp_id; g < n_groups; g += kBm) {
        int d_base = g * BCOL;
        int group_sum = 0;

        // Each lane handles BCOL/kWarpSize = 4 elements per group.
        // We iterate 4 times (k = lane, lane+32, lane+64, lane+96).
        for (int k = lane; k < BCOL; k += kWarpSize) {
            int pidx = __ldg(perm + d_base + k);
            float x = __half2float(__ldg(X_fp16 + (int64_t)pidx * stride_xd));
            float qf = is_zero ? 0.0f : rintf(__fdividef(x, scale_math));
            qf = fmaxf(fminf(qf, 7.0f), -8.0f);
            int q = static_cast<int>(qf);
            group_sum += q;

            // Pack: pair (k, k+1) -> 1 byte.
            // k = lane + i*32; pairs are (lane, lane+1) within same i.
            int q_nb = __shfl_xor_sync(0xFFFFFFFF, q, 1);
            if ((k & 1) == 0) {
                int packed = ((q_nb & 0x0F) << 4) | (q & 0x0F);
                s_X_s4[g * kGroupBytes + (k >> 1)] = static_cast<int8_t>(
                    packed >= 128 ? packed - 256 : packed);
            }
        }

        // Reduce group_sum across warp (4 partial sums -> 1).
        int wsum = warp_sum_i(group_sum);
        if (lane == 0) s_sum_X[g] = wsum;
    }
    __syncthreads();

    // ================================================================
    // Phase B: GEMV (each warp computes its own output row)
    // ================================================================
    if (m >= d_out) return;

    float y_acc = 0.0f;

    // Dense branch.
    for (int g = 0; g < n_groups; ++g) {
        int64_t w_off = (int64_t)m * stride_w_m
                      + (int64_t)(g * kGroupBytes + lane * 2) * stride_w_k;
        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(W_low + w_off);
        uint16_t x_packed = *reinterpret_cast<const uint16_t*>(
            &s_X_s4[g * kGroupBytes + lane * 2]);

        uint32_t w_s8 = unpack_s4_to_s8x4(w_packed);
        uint32_t x_s8 = unpack_s4_to_s8x4(x_packed);

        int dot = 0;
        dot = __dp4a(static_cast<int>(w_s8), static_cast<int>(x_s8), dot);

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

        if (lane == 0) {
            // Round 26: scale/zero from shmem (prefetched in Phase A1).
            float s = __half2float(s_scale_u4_w[warp_id][g]);
            float z = __half2float(s_zero_u4_w [warp_id][g]);
            float sumxn = static_cast<float>(s_sum_X[g]);
            y_acc += s * (static_cast<float>(dot) - z * sumxn);
        }
    }

    // Sparse branch.
    const int blk_start = hp_row_offsets[br];
    const int blk_end   = hp_row_offsets[br + 1];

    for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
        const int bc = __ldg(&hp_col_indices[block_idx]);

        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(
            W_high_blocks + (int64_t)block_idx * stride_wb_blk
                          + (int64_t)m_in_br * stride_wb_r
                          + (int64_t)(lane * 2) * stride_wb_k);
        uint16_t x_packed = *reinterpret_cast<const uint16_t*>(
            &s_X_s4[bc * kGroupBytes + lane * 2]);

        uint32_t w_s8 = unpack_s4_to_s8x4(w_packed);
        uint32_t x_s8 = unpack_s4_to_s8x4(x_packed);

        int dot = 0;
        dot = __dp4a(static_cast<int>(w_s8), static_cast<int>(x_s8), dot);

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

        if (lane == 0) {
            // Round 26: scale from shmem.
            float s = __half2float(s_scale_u4_w[warp_id][bc]);
            y_acc += 16.0f * static_cast<float>(dot) * s;
        }
    }

    if (lane == 0) {
        float sxn = __half2float(s_scale_x_h);
        Y[(int64_t)m * stride_y_m] = __float2half(y_acc * sxn);
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

void launch(
    torch::Tensor X_fp16, torch::Tensor perm,
    torch::Tensor W_low, torch::Tensor W_high_blocks,
    torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
    torch::Tensor scale_u4, torch::Tensor zero_u4,
    torch::Tensor Y_total,
    int d_out, int d_in
) {
    TORCH_CHECK(X_fp16.dtype() == torch::kHalf);
    TORCH_CHECK(perm.dtype() == torch::kInt32);
    TORCH_CHECK(W_low.dtype() == torch::kInt8);
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf);
    TORCH_CHECK(zero_u4.dtype() == torch::kHalf);
    TORCH_CHECK(Y_total.dtype() == torch::kHalf);
    TORCH_CHECK(X_fp16.size(0) == 1, "fused_quant_gemv requires T == 1");
    TORCH_CHECK(d_in % BCOL == 0);

    const int n_groups = d_in / BCOL;
    TORCH_CHECK(n_groups <= kMaxGroups,
                "n_groups (", n_groups, ") > kMaxGroups (", kMaxGroups, ")");

    if (W_high_blocks.numel() == 0) {
        W_high_blocks = torch::zeros(
            {0, BROW, BCOL / 2},
            torch::TensorOptions().dtype(torch::kInt8).device(W_low.device())
        );
    }
    TORCH_CHECK(W_high_blocks.dtype() == torch::kInt8);
    const int nrow = (d_out + BROW - 1) / BROW;
    TORCH_CHECK(hp_row_offsets.numel() == nrow + 1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kBm = 8;
    static_assert(BROW % kBm == 0);

    dim3 block(32, kBm, 1);
    dim3 grid(ceil_div(d_out, kBm), 1, 1);

    fused_quant_gemv_kernel<kBm><<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
        perm.data_ptr<int>(),
        X_fp16.stride(1),
        reinterpret_cast<const uint8_t*>(W_low.data_ptr<int8_t>()),
        W_low.stride(0), W_low.stride(1),
        reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
        scale_u4.stride(0), scale_u4.stride(1),
        zero_u4.stride(0),  zero_u4.stride(1),
        reinterpret_cast<const uint8_t*>(W_high_blocks.data_ptr<int8_t>()),
        hp_row_offsets.data_ptr<int>(),
        hp_col_indices.data_ptr<int>(),
        W_high_blocks.stride(0), W_high_blocks.stride(1), W_high_blocks.stride(2),
        reinterpret_cast<__half*>(Y_total.data_ptr<at::Half>()),
        Y_total.stride(0),
        d_out, d_in, n_groups
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fused_quant_gemv
}  // namespace hkust_v9
