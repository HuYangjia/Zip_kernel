// Fused Dense + Sparse UINT4/SINT4 GEMV  (T=1 decode specialisation, SM89).
//
// Round 14: T=1 specialisation of fused_dense_sparse, analogous to
// dense_gemv_decode.cu (Round 13) but with the sparse BSR branch folded
// into the same warp-per-row kernel.
//
// Kernel shape
// ------------
//   kBm warps per CTA, one warp per output row m.
//   Grid: (ceil(d_out / kBm),) CTAs.
//   Multiple CTAs share the same BSR block-row (br = blockIdx.x / (BROW/kBm)).
//
// For each warp m:
//   y[m]  = dense_part(m) + 16 * sparse_part(m)
//   dense_part(m)  = sum_{g=0..n_groups-1} scale_u4[m,g] *
//                    ( <W_low[m, g*128:], X_s4[0, g*128:]> - zero_u4[m,g] * sum_X[0,g] )
//   sparse_part(m) = sum_{bl in BSR row br} scale_u4[m, bc(bl)] *
//                    <W_high_blocks[bl, m%BROW, :], X_s4[0, bc(bl)*128:]>
//   y_final[m]     = scale_x[0] * y[m]
//
// All shared structures (X, sum_X) are prefetched once to shmem.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace fused_gemv_decode {

// Same s4 -> 4 s8 unpacker used by dense_gemv_decode.cu.
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

template <int kBm>
__global__ void fused_gemv_decode_kernel(
    const uint8_t* __restrict__ W_low,       // (d_out, d_in/2)
    const uint8_t* __restrict__ X,           // (1, d_in/2)
    const __half* __restrict__ scale_u4,     // (d_out, n_groups)
    const __half* __restrict__ zero_u4,      // (d_out, n_groups)
    const int* __restrict__ sum_X,           // (1, n_groups)
    const __half* __restrict__ scale_x,      // (1,)
    const uint8_t* __restrict__ W_high_blocks,   // (n_hp, BROW, BCOL/2)
    const int* __restrict__ hp_row_offsets,  // (nrow + 1,)
    const int* __restrict__ hp_col_indices,  // (n_hp,)
    __half* __restrict__ Y,                  // (d_out, 1)
    int d_out, int d_in, int n_groups,
    int64_t stride_w_m,   int64_t stride_w_k,
    int64_t stride_x_k,
    int64_t stride_su_m,  int64_t stride_su_g,
    int64_t stride_zu_m,  int64_t stride_zu_g,
    int64_t stride_sx_g,
    int64_t stride_wb_blk, int64_t stride_wb_r, int64_t stride_wb_k,
    int64_t stride_y_m
) {
    constexpr int kWarpSize = 32;
    constexpr int kGroupBytes = BCOL >> 1;  // 64 packed bytes per group

    const int lane = threadIdx.x;           // 0..31
    const int warp_id = threadIdx.y;        // 0..kBm-1
    const int m = blockIdx.x * kBm + warp_id;

    // BSR block-row this CTA belongs to (multiple CTAs share one br).
    const int br = (blockIdx.x * kBm) / BROW;
    const int m_in_br = m - br * BROW;      // row offset within BROW (0..BROW-1)

    // Shmem: X for current group (64 packed bytes), sum_X cache, scale_x.
    __shared__ alignas(16) uint8_t s_X[kGroupBytes];
    __shared__ int s_sum_X[128];            // upper bound on n_groups
    __shared__ __half s_scale_x_val;

    const int flat_tid = warp_id * kWarpSize + lane;
    if (flat_tid < n_groups) {
        s_sum_X[flat_tid] = sum_X[flat_tid];
    }
    if (flat_tid == 0) {
        s_scale_x_val = scale_x[0];
    }
    __syncthreads();

    if (m >= d_out) return;

    // ============================================================
    // DENSE BRANCH (identical math to dense_gemv_decode)
    // ============================================================
    float y_acc = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        if (warp_id == 0 && lane < (kGroupBytes / 4)) {
            int64_t off = (int64_t)(g * kGroupBytes + lane * 4) * stride_x_k;
            reinterpret_cast<uint32_t*>(s_X)[lane] =
                *reinterpret_cast<const uint32_t*>(X + off);
        }
        __syncthreads();

        int64_t w_off = (int64_t)m * stride_w_m
                      + (int64_t)(g * kGroupBytes + lane * 2) * stride_w_k;
        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(W_low + w_off);
        uint16_t x_packed = *reinterpret_cast<const uint16_t*>(&s_X[lane * 2]);

        uint32_t w_s8 = unpack_s4_to_s8x4(w_packed);
        uint32_t x_s8 = unpack_s4_to_s8x4(x_packed);

        int dot = 0;
        dot = __dp4a(static_cast<int>(w_s8), static_cast<int>(x_s8), dot);

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
        }

        if (lane == 0) {
            float s = __half2float(scale_u4[(int64_t)m * stride_su_m
                                          + (int64_t)g * stride_su_g]);
            float z = __half2float(zero_u4 [(int64_t)m * stride_zu_m
                                          + (int64_t)g * stride_zu_g]);
            float sumxn = static_cast<float>(s_sum_X[g]);
            y_acc += s * (static_cast<float>(dot) - z * sumxn);
        }

        __syncthreads();
    }

    // ============================================================
    // SPARSE BRANCH
    // ============================================================
    const int blk_start = hp_row_offsets[br];
    const int blk_end   = hp_row_offsets[br + 1];

    for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
        const int bc = __ldg(&hp_col_indices[block_idx]);

        // Re-load X for column group bc (shared across CTA).
        if (warp_id == 0 && lane < (kGroupBytes / 4)) {
            int64_t off = (int64_t)(bc * kGroupBytes + lane * 4) * stride_x_k;
            reinterpret_cast<uint32_t*>(s_X)[lane] =
                *reinterpret_cast<const uint32_t*>(X + off);
        }
        __syncthreads();

        // W_high_blocks[block_idx, m_in_br, lane*2 : lane*2+2].
        int64_t wb_off = (int64_t)block_idx * stride_wb_blk
                       + (int64_t)m_in_br * stride_wb_r
                       + (int64_t)(lane * 2) * stride_wb_k;
        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(W_high_blocks + wb_off);
        uint16_t x_packed = *reinterpret_cast<const uint16_t*>(&s_X[lane * 2]);

        uint32_t w_s8 = unpack_s4_to_s8x4(w_packed);
        uint32_t x_s8 = unpack_s4_to_s8x4(x_packed);

        int dot = 0;
        dot = __dp4a(static_cast<int>(w_s8), static_cast<int>(x_s8), dot);

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);
        }

        if (lane == 0) {
            float s = __half2float(scale_u4[(int64_t)m * stride_su_m
                                          + (int64_t)bc * stride_su_g]);
            y_acc += 16.0f * static_cast<float>(dot) * s;
        }

        __syncthreads();
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
    torch::Tensor W_low, torch::Tensor W_high_blocks,
    torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
    torch::Tensor X_s4,
    torch::Tensor scale_u4, torch::Tensor zero_u4,
    torch::Tensor sum_X, torch::Tensor scale_x,
    torch::Tensor Y_total,
    int d_out, int d_in
) {
    TORCH_CHECK(W_low.dtype() == torch::kInt8);
    TORCH_CHECK(X_s4.dtype() == torch::kInt8);
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf);
    TORCH_CHECK(zero_u4.dtype() == torch::kHalf);
    TORCH_CHECK(sum_X.dtype() == torch::kInt32);
    TORCH_CHECK(scale_x.dtype() == torch::kHalf);
    TORCH_CHECK(Y_total.dtype() == torch::kHalf);
    TORCH_CHECK(W_low.stride(1) == 1);
    TORCH_CHECK(X_s4.stride(1) == 1);
    TORCH_CHECK(X_s4.size(0) == 1, "fused_gemv_decode requires T == 1");

    const int d_in_half = W_low.size(1);
    TORCH_CHECK(d_in_half * 2 == d_in);
    TORCH_CHECK(d_in % BCOL == 0);
    const int n_groups = d_in / BCOL;
    TORCH_CHECK(n_groups <= 128, "n_groups > 128 unsupported in GEMV path");

    if (W_high_blocks.numel() == 0) {
        W_high_blocks = torch::zeros(
            {0, BROW, BCOL / 2},
            torch::TensorOptions().dtype(torch::kInt8).device(W_low.device())
        );
    }
    TORCH_CHECK(W_high_blocks.dtype() == torch::kInt8);
    TORCH_CHECK(W_high_blocks.stride(2) == 1);
    const int nrow = (d_out + BROW - 1) / BROW;
    TORCH_CHECK(hp_row_offsets.numel() == nrow + 1);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kBm = 8;   // 8 warps per CTA = 256 threads
    //   We must have kBm divide BROW so that every CTA falls cleanly
    //   into exactly one BSR block-row.  BROW is typically 128, so kBm
    //   in {1,2,4,8,16,32,64,128} all work.
    static_assert(BROW % kBm == 0, "kBm must divide BROW");

    dim3 block(32, kBm, 1);
    dim3 grid(ceil_div(d_out, kBm), 1, 1);

    fused_gemv_decode_kernel<kBm><<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W_low.data_ptr<int8_t>()),
        reinterpret_cast<const uint8_t*>(X_s4.data_ptr<int8_t>()),
        reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
        sum_X.data_ptr<int>(),
        reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
        reinterpret_cast<const uint8_t*>(W_high_blocks.data_ptr<int8_t>()),
        hp_row_offsets.data_ptr<int>(),
        hp_col_indices.data_ptr<int>(),
        reinterpret_cast<__half*>(Y_total.data_ptr<at::Half>()),
        d_out, d_in, n_groups,
        W_low.stride(0), W_low.stride(1),
        X_s4.stride(1),
        scale_u4.stride(0), scale_u4.stride(1),
        zero_u4.stride(0), zero_u4.stride(1),
        sum_X.stride(1),
        W_high_blocks.stride(0), W_high_blocks.stride(1), W_high_blocks.stride(2),
        Y_total.stride(0)
    );
    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fused_gemv_decode
}  // namespace hkust_v9
