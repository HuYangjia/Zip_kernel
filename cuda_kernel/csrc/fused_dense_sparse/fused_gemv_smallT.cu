// Fused Dense + Sparse UINT4/SINT4 GEMV for small T (T=2..16), SM89.
//
// Round 16: extends the T=1 GEMV architecture to small batch (T=2..16).
// The INT4 MMA kernel at T=8 fills only 1/8 of its N=8 slice, wasting
// 87% of the Tensor Core N dimension; a dp4a kernel that loops over
// T columns inside one warp achieves 100% arithmetic utilisation.
//
// Per warp (one output row m):
//   For each group g:
//     Load W[m, g] once (64 packed bytes, per-lane 2 bytes = 4 s8).
//     For each t in [0, T):
//       Load X[t, g] into shmem (shared across all warps in CTA).
//       Compute dp4a, warp reduce, accumulate y_acc[t].
//
// This keeps the same BSR architecture as fused_gemv_decode and gets
// extended dispatch for T up to kMaxT.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9 {
namespace fused_gemv_smallT {

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

constexpr int kMaxT = 16;

template <int kBm, int kT>
__global__ void fused_gemv_smallT_kernel(
    const uint8_t* __restrict__ W_low,       // (d_out, d_in/2)
    const uint8_t* __restrict__ X,           // (T, d_in/2)
    const __half* __restrict__ scale_u4,     // (d_out, n_groups)
    const __half* __restrict__ zero_u4,      // (d_out, n_groups)
    const int* __restrict__ sum_X,           // (T, n_groups)
    const __half* __restrict__ scale_x,      // (T,)
    const uint8_t* __restrict__ W_high_blocks,
    const int* __restrict__ hp_row_offsets,
    const int* __restrict__ hp_col_indices,
    __half* __restrict__ Y,                  // (d_out, T)
    int d_out, int d_in, int n_groups, int T,
    int64_t stride_w_m,   int64_t stride_w_k,
    int64_t stride_xt,    int64_t stride_xk,
    int64_t stride_su_m,  int64_t stride_su_g,
    int64_t stride_zu_m,  int64_t stride_zu_g,
    int64_t stride_sX_t,  int64_t stride_sX_g,
    int64_t stride_wb_blk, int64_t stride_wb_r, int64_t stride_wb_k,
    int64_t stride_y_m,   int64_t stride_y_n
) {
    constexpr int kGroupBytes = BCOL >> 1;  // 64 packed bytes per group

    const int lane = threadIdx.x;       // 0..31
    const int warp_id = threadIdx.y;    // 0..kBm-1
    const int m = blockIdx.x * kBm + warp_id;
    const int br = (blockIdx.x * kBm) / BROW;
    const int m_in_br = m - br * BROW;

    // Shmem:
    //   s_X[kT][kGroupBytes]  : staged X for current group, all T rows.
    //   s_sum_X[kT][n_groups] : sum_X prefetched (n_groups <= 128 guaranteed).
    //   s_scale_x[kT]         : scale_x prefetched.
    __shared__ alignas(16) uint8_t s_X[kT][kGroupBytes];
    __shared__ int s_sum_X[kT][128];
    __shared__ __half s_scale_x[kT];

    const int flat_tid = warp_id * kWarpSize + lane;
    const int cta_size = kBm * kWarpSize;

    // Prefetch sum_X and scale_x.  sum_X has T*n_groups entries.
    for (int i = flat_tid; i < T * n_groups; i += cta_size) {
        int t_i = i / n_groups;
        int g_i = i - t_i * n_groups;
        s_sum_X[t_i][g_i] = sum_X[(int64_t)t_i * stride_sX_t
                                + (int64_t)g_i * stride_sX_g];
    }
    if (flat_tid < T) {
        s_scale_x[flat_tid] = scale_x[flat_tid];
    }
    __syncthreads();

    if (m >= d_out) return;

    // Per-warp (one m row) fp32 accumulators for each of T columns.
    float y_acc[kMaxT];
    #pragma unroll
    for (int t = 0; t < kT; ++t) y_acc[t] = 0.0f;

    // ============================================================
    // DENSE BRANCH
    // ============================================================
    for (int g = 0; g < n_groups; ++g) {
        // Cooperatively load X[t, g] for t in [0, T) into shmem.
        // Each T-row has 64 bytes = 16 uint32.
        // Total uint32 writes = T * 16.  Distribute among all CTA threads.
        const int total_u32 = T * (kGroupBytes / 4);
        for (int i = flat_tid; i < total_u32; i += cta_size) {
            int t_i = i / (kGroupBytes / 4);
            int u_i = i - t_i * (kGroupBytes / 4);
            int64_t off = (int64_t)t_i * stride_xt
                        + (int64_t)(g * kGroupBytes + u_i * 4) * stride_xk;
            reinterpret_cast<uint32_t*>(s_X[t_i])[u_i] =
                *reinterpret_cast<const uint32_t*>(X + off);
        }
        __syncthreads();

        // Load W[m, g] once per warp.
        int64_t w_off = (int64_t)m * stride_w_m
                      + (int64_t)(g * kGroupBytes + lane * 2) * stride_w_k;
        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(W_low + w_off);
        uint32_t w_s8 = unpack_s4_to_s8x4(w_packed);

        // Load per-group scale/zero for this m, g (once per group).
        float s_val = __half2float(scale_u4[(int64_t)m * stride_su_m
                                          + (int64_t)g * stride_su_g]);
        float z_val = __half2float(zero_u4 [(int64_t)m * stride_zu_m
                                          + (int64_t)g * stride_zu_g]);

        // Loop over T columns: load X[t, g], dp4a, reduce, accumulate.
        #pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t >= T) break;
            uint16_t x_packed = *reinterpret_cast<const uint16_t*>(
                &s_X[t][lane * 2]);
            uint32_t x_s8 = unpack_s4_to_s8x4(x_packed);

            int dot = 0;
            dot = __dp4a(static_cast<int>(w_s8), static_cast<int>(x_s8), dot);

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

            if (lane == 0) {
                float sumxn = static_cast<float>(s_sum_X[t][g]);
                y_acc[t] += s_val * (static_cast<float>(dot) - z_val * sumxn);
            }
        }
        __syncthreads();   // before overwriting s_X for next g
    }

    // ============================================================
    // SPARSE BRANCH
    // ============================================================
    const int blk_start = hp_row_offsets[br];
    const int blk_end   = hp_row_offsets[br + 1];

    for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
        const int bc = __ldg(&hp_col_indices[block_idx]);

        // Reload X for column group bc.
        const int total_u32 = T * (kGroupBytes / 4);
        for (int i = flat_tid; i < total_u32; i += cta_size) {
            int t_i = i / (kGroupBytes / 4);
            int u_i = i - t_i * (kGroupBytes / 4);
            int64_t off = (int64_t)t_i * stride_xt
                        + (int64_t)(bc * kGroupBytes + u_i * 4) * stride_xk;
            reinterpret_cast<uint32_t*>(s_X[t_i])[u_i] =
                *reinterpret_cast<const uint32_t*>(X + off);
        }
        __syncthreads();

        // W_high_blocks[block_idx, m_in_br, lane*2 : lane*2+2].
        int64_t wb_off = (int64_t)block_idx * stride_wb_blk
                       + (int64_t)m_in_br * stride_wb_r
                       + (int64_t)(lane * 2) * stride_wb_k;
        uint16_t w_packed = *reinterpret_cast<const uint16_t*>(W_high_blocks + wb_off);
        uint32_t w_s8 = unpack_s4_to_s8x4(w_packed);

        float s_val = __half2float(scale_u4[(int64_t)m * stride_su_m
                                          + (int64_t)bc * stride_su_g]);

        #pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t >= T) break;
            uint16_t x_packed = *reinterpret_cast<const uint16_t*>(
                &s_X[t][lane * 2]);
            uint32_t x_s8 = unpack_s4_to_s8x4(x_packed);

            int dot = 0;
            dot = __dp4a(static_cast<int>(w_s8), static_cast<int>(x_s8), dot);

            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xFFFFFFFF, dot, off);

            if (lane == 0) {
                y_acc[t] += 16.0f * static_cast<float>(dot) * s_val;
            }
        }
        __syncthreads();
    }

    // Writeback: lane 0 writes all T outputs for row m.
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < kT; ++t) {
            if (t >= T) break;
            float sxn = __half2float(s_scale_x[t]);
            Y[(int64_t)m * stride_y_m + (int64_t)t * stride_y_n] =
                __float2half(y_acc[t] * sxn);
        }
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

    const int d_in_half = W_low.size(1);
    TORCH_CHECK(d_in_half * 2 == d_in);
    TORCH_CHECK(d_in % BCOL == 0);
    const int T = X_s4.size(0);
    TORCH_CHECK(T >= 1 && T <= kMaxT,
                "smallT path supports 1 <= T <= ", kMaxT, " (got ", T, ")");

    const int n_groups = d_in / BCOL;
    TORCH_CHECK(n_groups <= 128);

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

    auto launch_kT = [&](auto kTConst) {
        constexpr int kT = decltype(kTConst)::value;
        fused_gemv_smallT_kernel<kBm, kT><<<grid, block, 0, stream>>>(
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
            d_out, d_in, n_groups, T,
            W_low.stride(0), W_low.stride(1),
            X_s4.stride(0),  X_s4.stride(1),
            scale_u4.stride(0), scale_u4.stride(1),
            zero_u4.stride(0),  zero_u4.stride(1),
            sum_X.stride(0),    sum_X.stride(1),
            W_high_blocks.stride(0), W_high_blocks.stride(1), W_high_blocks.stride(2),
            Y_total.stride(0), Y_total.stride(1)
        );
    };

    // Ceil-up to next power-of-2 kT template to minimise instantiations.
    if      (T <= 2)  launch_kT(std::integral_constant<int, 2>{});
    else if (T <= 4)  launch_kT(std::integral_constant<int, 4>{});
    else if (T <= 8)  launch_kT(std::integral_constant<int, 8>{});
    else              launch_kT(std::integral_constant<int, 16>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fused_gemv_smallT
}  // namespace hkust_v9
