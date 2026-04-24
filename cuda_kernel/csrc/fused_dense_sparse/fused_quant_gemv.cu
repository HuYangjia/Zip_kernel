// Fused Activation Quantization + Dense+Sparse GEMV (T=1 decode, SM89).
//
// Round 15b: Fuse activation_quant + fused_gemv_decode into a single kernel
// to eliminate the second kernel launch overhead and avoid writing/reading
// X_s4, scale_x, sum_X to/from HBM.
//
// For T=1 decode on 4k->4k:
//   - activation_quant alone: ~14us (dominated by kernel launch overhead)
//   - fused_gemv_decode alone: ~16us
//   - fused together: ~16us (one launch, X read once)
//
// Architecture:
//   Grid:  (ceil_div(d_out, kBm),)  -- same as fused_gemv_decode
//   Block: (32, kBm)                -- 1 warp per output row
//
//   Each CTA:
//     1. Warp 0 (lane_y=0) performs activation quant for each group g:
//        - Gather X[perm[g*128 : (g+1)*128]] into shmem
//        - Compute max-abs, scale, quantize, pack X_s4, compute sum_X
//        - Write X_s4 and sum_X to shmem (shared with all warps)
//     2. All warps (lane_y=0..kBm-1) read X_s4 and sum_X from shmem
//        and compute their output row's dot product (dense + sparse).
//
// Shmem layout per CTA:
//   s_X_s4[n_groups][BCOL/2]  -- packed INT4 activation (int8)
//   s_sum_X[n_groups]         -- per-group sum (int32)
//   s_scale_x                 -- fp16 activation scale
//   s_scale_math              -- fp32 scale for quant math
//   s_is_zero                 -- bool
//   s_X_fp16[BCOL]            -- scratch for current group gather (fp16)
//   s_warp_part[4]            -- max-abs warp reduction scratch (float)
//   s_warp_part_int[4]        -- sum warp reduction scratch (int)
//
// Constraint: n_groups <= kMaxGroups (128), D <= 16384.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9 {
namespace fused_quant_gemv {

// Reuse the same s4 unpacker as fused_gemv_decode.
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

constexpr int kMaxGroups = 128;   // max n_groups supported

template <int kBm>
__global__ void fused_quant_gemv_kernel(
    // Activation input (fp16, T=1)
    const __half* __restrict__ X_fp16,   // (1, D)
    const int*    __restrict__ perm,     // (D,)
    int64_t stride_xd,                   // X stride along D (usually 1)
    // Weight (dense, packed INT4)
    const uint8_t* __restrict__ W_low,   // (d_out, d_in/2)
    int64_t stride_w_m, int64_t stride_w_k,
    // Weight scale/zero (dense)
    const __half* __restrict__ scale_u4, // (d_out, n_groups)
    const __half* __restrict__ zero_u4,  // (d_out, n_groups)
    int64_t stride_su_m, int64_t stride_su_g,
    int64_t stride_zu_m, int64_t stride_zu_g,
    // Sparse BSR
    const uint8_t* __restrict__ W_high_blocks,  // (n_hp, BROW, BCOL/2)
    const int*     __restrict__ hp_row_offsets, // (nrow+1,)
    const int*     __restrict__ hp_col_indices, // (n_hp,)
    int64_t stride_wb_blk, int64_t stride_wb_r, int64_t stride_wb_k,
    // Output
    __half* __restrict__ Y,              // (d_out, 1)
    int64_t stride_y_m,
    // Dims
    int d_out, int d_in, int n_groups
) {
    constexpr int kGroupBytes = BCOL >> 1;  // 64 packed bytes per group

    const int lane = threadIdx.x;      // 0..31
    const int warp_id = threadIdx.y;   // 0..kBm-1
    const int m = blockIdx.x * kBm + warp_id;
    const int br = (blockIdx.x * kBm) / BROW;
    const int m_in_br = m - br * BROW;

    // ----------------------------------------------------------------
    // Shared memory layout
    // ----------------------------------------------------------------
    // s_X_s4:       n_groups * (BCOL/2) bytes  (packed INT4)
    // s_sum_X:      n_groups * 4 bytes          (int32)
    // s_scale_x:    4 bytes                     (fp16, padded to 4)
    // s_scale_math: 4 bytes                     (fp32)
    // s_is_zero:    4 bytes                     (int, 0 or 1)
    // s_X_fp16:     BCOL * 2 bytes              (fp16 scratch for gather)
    // s_warp_part:  4 * 4 bytes                 (float, max-abs reduction)
    // s_warp_part_int: 4 * 4 bytes              (int, sum reduction)
    // Total for n_groups=32 (D=4096): 32*64 + 32*4 + 4+4+4 + 256 + 16 + 16
    //   = 2048 + 128 + 12 + 256 + 32 = 2476 bytes  (very small)

    __shared__ int8_t  s_X_s4[kMaxGroups * (BCOL / 2)];
    __shared__ int     s_sum_X[kMaxGroups];
    __shared__ __half  s_scale_x_h;
    __shared__ float   s_scale_math;
    __shared__ int     s_is_zero;
    __shared__ __half  s_X_fp16[BCOL];
    __shared__ float   s_warp_part[4];
    __shared__ int     s_warp_part_int[4];

    // ----------------------------------------------------------------
    // Step 1: Warp 0 performs activation quantization, group by group.
    //         Other warps wait at the end of each group.
    // ----------------------------------------------------------------

    // First, compute global max-abs across all groups (warp 0 only).
    // We do this in a single pass over all D elements.
    if (warp_id == 0) {
        float local_max = 0.0f;
        for (int d = lane; d < d_in; d += kWarpSize) {
            int pidx = __ldg(perm + d);
            float v = __half2float(__ldg(X_fp16 + (int64_t)pidx * stride_xd));
            local_max = fmaxf(local_max, fabsf(v));
        }
        float wmax = warp_max_abs_f(local_max);
        if (lane == 0) {
            float scale_fp32 = wmax / 7.0f;
            __half scale_h = __float2half(scale_fp32);
            float scale_math = __half2float(scale_h);
            bool iz = !(scale_math > 0.0f);
            s_scale_x_h  = scale_h;
            s_scale_math = iz ? 1.0f : scale_math;
            s_is_zero    = iz ? 1 : 0;
        }
    }
    __syncthreads();

    // Now quantize group by group, writing X_s4 and sum_X to shmem.
    const float scale_math = s_scale_math;
    const bool  is_zero    = s_is_zero != 0;

    for (int g = 0; g < n_groups; ++g) {
        // Warp 0 gathers X[perm[g*BCOL .. (g+1)*BCOL-1]] into s_X_fp16.
        if (warp_id == 0) {
            int d_base = g * BCOL;
            for (int k = lane; k < BCOL; k += kWarpSize) {
                int pidx = __ldg(perm + d_base + k);
                s_X_fp16[k] = __ldg(X_fp16 + (int64_t)pidx * stride_xd);
            }
        }
        __syncthreads();

        // Warp 0 quantizes and packs.
        if (warp_id == 0) {
            // Each lane handles 4 elements (lane * 4 .. lane*4+3).
            // But BCOL=128, kWarpSize=32, so each lane handles 4 elements.
            // We need to compute sum over all 128 elements.
            // Use 4 passes of 32 lanes each.
            int group_sum = 0;
            for (int k = lane; k < BCOL; k += kWarpSize) {
                float x = __half2float(s_X_fp16[k]);
                float qf = is_zero ? 0.0f : rintf(__fdividef(x, scale_math));
                qf = fmaxf(fminf(qf, 7.0f), -8.0f);
                int q = static_cast<int>(qf);
                group_sum += q;

                // Pack: even k = low nibble, odd k = high nibble.
                // We need to pair k with k^1.
                // Use shfl to get neighbour within the warp.
                // But k increments by kWarpSize, so consecutive k values
                // in this loop are NOT adjacent in the array.
                // We must handle packing differently:
                // Write q to a temp location and pack after the loop.
                // For simplicity, write to s_X_s4 byte by byte using
                // a two-step: even k writes low nibble, odd k writes high.
                // Since k = lane + i*32, consecutive lanes are adjacent.
                // Within one iteration, lane 0 handles k=0,32,64,96;
                // lane 1 handles k=1,33,65,97; etc.
                // Pairs are (k, k+1) = (lane, lane+1) within same iteration.
                // Use shfl_xor(1) to get the neighbour's q.
                int q_nb = __shfl_xor_sync(0xFFFFFFFF, q, 1);
                if ((k & 1) == 0) {
                    int low  = q    & 0x0F;
                    int high = q_nb & 0x0F;
                    int packed = (high << 4) | low;
                    s_X_s4[g * (BCOL/2) + (k >> 1)] = static_cast<int8_t>(
                        packed >= 128 ? packed - 256 : packed);
                }
            }
            // Reduce group_sum across warp.
            int wsum = warp_sum_i(group_sum);
            if (lane == 0) {
                s_sum_X[g] = wsum;
            }
        }
        __syncthreads();
    }

    // ----------------------------------------------------------------
    // Step 2: All warps compute their output row (same as fused_gemv_decode).
    // ----------------------------------------------------------------
    if (m >= d_out) return;

    float y_acc = 0.0f;

    // Dense branch.
    for (int g = 0; g < n_groups; ++g) {
        // Load W_low for this row and group.
        int64_t w_off = (int64_t)m * stride_w_m
                      + (int64_t)(g * (BCOL/2) + lane * 2) * stride_w_k;
        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(W_low + w_off);
        uint16_t x_packed = *reinterpret_cast<const uint16_t*>(&s_X_s4[g * (BCOL/2) + lane * 2]);

        uint32_t w_s8 = unpack_s4_to_s8x4(w_packed);
        uint32_t x_s8 = unpack_s4_to_s8x4(x_packed);

        int dot = 0;
        dot = __dp4a(static_cast<int>(w_s8), static_cast<int>(x_s8), dot);

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

        if (lane == 0) {
            float s = __half2float(scale_u4[(int64_t)m * stride_su_m
                                          + (int64_t)g * stride_su_g]);
            float z = __half2float(zero_u4 [(int64_t)m * stride_zu_m
                                          + (int64_t)g * stride_zu_g]);
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
            &s_X_s4[bc * (BCOL/2) + lane * 2]);

        uint32_t w_s8 = unpack_s4_to_s8x4(w_packed);
        uint32_t x_s8 = unpack_s4_to_s8x4(x_packed);

        int dot = 0;
        dot = __dp4a(static_cast<int>(w_s8), static_cast<int>(x_s8), dot);

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

        if (lane == 0) {
            float s = __half2float(scale_u4[(int64_t)m * stride_su_m
                                          + (int64_t)bc * stride_su_g]);
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
