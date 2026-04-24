// Block-sparse SINT4 x SINT4 GEMM -- INT4 Tensor Core version (SM89).
//
// Round 12 optimisations:
//   - kBn capped at 64 (eliminates 255-reg spill at kBn=128).
//   - Per-block scale_u4 prefetched to shmem: for each BSR block, the
//     kBm scale values (scale_u4[m_tile:m_tile+kBm, bc]) are cooperatively
//     loaded once and reused by all lanes in the epilogue.
//
// Uses mma.m16n8k64.s4.s4.s32 (Tensor Core on Ada SM89).

#include "common/arch.cuh"
#include "common/mma_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace sparse_gemm_mma_int4 {

template <int kBn>
__global__ void sparse_gemm_mma_int4_kernel(
    const uint8_t* __restrict__ W_high_blocks,
    const int* __restrict__ hp_row_offsets,
    const int* __restrict__ hp_col_indices,
    const uint8_t* __restrict__ X,
    const __half* __restrict__ scale_u4,
    const __half* __restrict__ scale_x,
    __half* __restrict__ Y,
    int d_out, int d_in, int T,
    int64_t stride_wb_blk, int64_t stride_wb_r, int64_t stride_wb_k,
    int64_t stride_x_n,  int64_t stride_x_k,
    int64_t stride_su_m, int64_t stride_su_g,
    int64_t stride_sx_n,
    int64_t stride_y_m,  int64_t stride_y_n
) {
    constexpr int kBm = BROW;
    constexpr int kBk = BCOL;
    constexpr int kMmaK = 64;
    constexpr int kKSteps = kBk / kMmaK;      // 2
    constexpr int kMsubPerWarp = 2;
    constexpr int kNsubPerCta = (kBn + 7) / 8;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int br = blockIdx.x;
    const int m_tile = br * kBm;
    const int n_tile = blockIdx.y * kBn;
    const int bytes_per_group = BCOL >> 1;

    __shared__ alignas(16) uint8_t sW[2][kBm][bytes_per_group];
    __shared__ alignas(16) uint8_t sX[2][kBn][bytes_per_group];
    __shared__ __half s_scale_x[kBn];
    __shared__ __half s_scale_block[kBm];   // scale_u4[m_tile:, bc] per block

    if (tid < kBn) {
        int n = n_tile + tid;
        s_scale_x[tid] = (n < T) ? scale_x[(int64_t)n * stride_sx_n] : __half(0);
    }

    float y_fp[kMsubPerWarp][kNsubPerCta][4];
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im)
      #pragma unroll
      for (int in = 0; in < kNsubPerCta; ++in)
        #pragma unroll
        for (int r = 0; r < 4; ++r) y_fp[im][in][r] = 0.0f;

    const int blk_start = hp_row_offsets[br];
    const int blk_end   = hp_row_offsets[br + 1];

    auto issue_w_load = [&](int block_idx, int buf) {
        const uint8_t* src = W_high_blocks
                           + (int64_t)block_idx * stride_wb_blk
                           + (int64_t)tid * stride_wb_r;
        uint8_t* dst = &sW[buf][tid][0];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint4 v = *reinterpret_cast<const uint4*>(src + i * 16);
            *reinterpret_cast<uint4*>(dst + i * 16) = v;
        }
    };

    auto issue_x_load = [&](int bc, int buf) {
        const int total_quads = kBn * 4;
        for (int q = tid; q < total_quads; q += kBm) {
            int row = q >> 2;
            int quad = q & 3;
            int n = n_tile + row;
            uint4 v;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(bc * bytes_per_group + quad * 16) * stride_x_k;
                v = *reinterpret_cast<const uint4*>(X + off);
            } else {
                v = make_uint4(0, 0, 0, 0);
            }
            *reinterpret_cast<uint4*>(&sX[buf][row][quad * 16]) = v;
        }
    };

    // Cooperative load of scale_u4[m_tile + 0..kBm, bc] into shmem.
    auto issue_scale_load = [&](int bc) {
        int m = m_tile + tid;
        s_scale_block[tid] = (m < d_out)
            ? scale_u4[(int64_t)m * stride_su_m + (int64_t)bc * stride_su_g]
            : __half(0);
    };

    __syncthreads();

    if (blk_start < blk_end) {
        int bc0 = __ldg(&hp_col_indices[blk_start]);
        issue_w_load(blk_start, 0);
        issue_x_load(bc0, 0);
    }
    __syncthreads();

    for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
        const int bc = __ldg(&hp_col_indices[block_idx]);
        const int buf = (block_idx - blk_start) & 1;
        if (block_idx + 1 < blk_end) {
            int bc_next = __ldg(&hp_col_indices[block_idx + 1]);
            issue_w_load(block_idx + 1, buf ^ 1);
            issue_x_load(bc_next, buf ^ 1);
        }
        issue_scale_load(bc);

        int d_acc[kMsubPerWarp][kNsubPerCta][4];
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im)
          #pragma unroll
          for (int in = 0; in < kNsubPerCta; ++in)
            #pragma unroll
            for (int r = 0; r < 4; ++r) d_acc[im][in][r] = 0;

        #pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
            const int kpb_base = ks * 32;

            uint32_t a_regs[kMsubPerWarp][4];
            #pragma unroll
            for (int im = 0; im < kMsubPerWarp; ++im) {
                int msub_base = warp_id * 32 + im * 16;
                int row0 = msub_base + (lane >> 2);
                int row1 = row0 + 8;
                int col_low  = kpb_base + (lane & 3) * 4;
                int col_high = col_low + 16;
                uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                if (row0 < kBm) {
                    a0 = *reinterpret_cast<const uint32_t*>(&sW[buf][row0][col_low]);
                    a2 = *reinterpret_cast<const uint32_t*>(&sW[buf][row0][col_high]);
                }
                if (row1 < kBm) {
                    a1 = *reinterpret_cast<const uint32_t*>(&sW[buf][row1][col_low]);
                    a3 = *reinterpret_cast<const uint32_t*>(&sW[buf][row1][col_high]);
                }
                a_regs[im][0] = a0;
                a_regs[im][1] = a1;
                a_regs[im][2] = a2;
                a_regs[im][3] = a3;
            }

            uint32_t b_regs[kNsubPerCta][2];
            #pragma unroll
            for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                int n_row_in_sub = lane >> 2;
                int n_row = in_sub * 8 + n_row_in_sub;
                int col_low  = kpb_base + (lane & 3) * 4;
                int col_high = col_low + 16;
                uint32_t b0 = 0, b1 = 0;
                if (n_row < kBn) {
                    b0 = *reinterpret_cast<const uint32_t*>(&sX[buf][n_row][col_low]);
                    b1 = *reinterpret_cast<const uint32_t*>(&sX[buf][n_row][col_high]);
                }
                b_regs[in_sub][0] = b0;
                b_regs[in_sub][1] = b1;
            }

            #pragma unroll
            for (int im = 0; im < kMsubPerWarp; ++im) {
                #pragma unroll
                for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                    mma_m16n8k64_s4s4s32(
                        d_acc[im][in_sub][0], d_acc[im][in_sub][1],
                        d_acc[im][in_sub][2], d_acc[im][in_sub][3],
                        a_regs[im][0], a_regs[im][1], a_regs[im][2], a_regs[im][3],
                        b_regs[in_sub][0], b_regs[in_sub][1],
                        d_acc[im][in_sub][0], d_acc[im][in_sub][1],
                        d_acc[im][in_sub][2], d_acc[im][in_sub][3]
                    );
                }
            }
        }

        // Epilogue: scale via shmem, not HBM.
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im) {
            int msub_base = warp_id * 32 + im * 16;
            #pragma unroll
            for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                int nsub_base = in_sub * 8;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int row_local = (lane >> 2) + ((r >> 1) ? 8 : 0);
                    int col_local = (lane & 3) * 2 + (r & 1);
                    int m_local = msub_base + row_local;
                    int m_global = m_tile + m_local;
                    int n_local = nsub_base + col_local;
                    if (n_local >= kBn) continue;
                    int n_global = n_tile + n_local;
                    if (m_global >= d_out) continue;
                    if (n_global >= T) continue;
                    float s = __half2float(s_scale_block[m_local]);
                    float sxn = __half2float(s_scale_x[n_local]);
                    y_fp[im][in_sub][r] += static_cast<float>(d_acc[im][in_sub][r])
                                         * s * sxn;
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im) {
        int msub_base = warp_id * 32 + im * 16;
        #pragma unroll
        for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
            int nsub_base = in_sub * 8;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                int row_local = (lane >> 2) + ((r >> 1) ? 8 : 0);
                int col_local = (lane & 3) * 2 + (r & 1);
                int m_global = m_tile + msub_base + row_local;
                int n_local = nsub_base + col_local;
                if (n_local >= kBn) continue;
                int n_global = n_tile + n_local;
                if (m_global >= d_out) continue;
                if (n_global >= T) continue;
                int64_t y_off = (int64_t)m_global * stride_y_m
                              + (int64_t)n_global * stride_y_n;
                Y[y_off] = __float2half(y_fp[im][in_sub][r]);
            }
        }
    }
}

void launch(
    torch::Tensor W_high_blocks,
    torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
    torch::Tensor X_s4, torch::Tensor scale_u4, torch::Tensor scale_x,
    torch::Tensor Y_high,
    int d_out, int d_in
) {
    TORCH_CHECK(W_high_blocks.dtype() == torch::kInt8);
    TORCH_CHECK(hp_row_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(hp_col_indices.dtype() == torch::kInt32);
    TORCH_CHECK(X_s4.dtype() == torch::kInt8);
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf);
    TORCH_CHECK(scale_x.dtype() == torch::kHalf);
    TORCH_CHECK(Y_high.dtype() == torch::kHalf);
    TORCH_CHECK(X_s4.stride(1) == 1);
    TORCH_CHECK(W_high_blocks.stride(2) == 1);

    const int T = X_s4.size(0);
    const int nrow = (d_out + BROW - 1) / BROW;
    TORCH_CHECK(hp_row_offsets.numel() == nrow + 1);
    Y_high.zero_();
    if (W_high_blocks.size(0) == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kBm = BROW;

    auto do_launch = [&](auto kBn_c) {
        constexpr int kBn = decltype(kBn_c)::value;
        dim3 block(kBm, 1, 1);
        dim3 grid(ceil_div(d_out, kBm), ceil_div(T, kBn), 1);
        sparse_gemm_mma_int4_kernel<kBn><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(W_high_blocks.data_ptr<int8_t>()),
            hp_row_offsets.data_ptr<int>(),
            hp_col_indices.data_ptr<int>(),
            reinterpret_cast<const uint8_t*>(X_s4.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(Y_high.data_ptr<at::Half>()),
            d_out, d_in, T,
            W_high_blocks.stride(0), W_high_blocks.stride(1), W_high_blocks.stride(2),
            X_s4.stride(0), X_s4.stride(1),
            scale_u4.stride(0), scale_u4.stride(1),
            scale_x.stride(0),
            Y_high.stride(0), Y_high.stride(1)
        );
    };

    // Round 12: cap kBn at 64.
    if      (T <= 8)    do_launch(std::integral_constant<int, 8>{});
    else if (T <= 32)   do_launch(std::integral_constant<int, 32>{});
    else                do_launch(std::integral_constant<int, 64>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sparse_gemm_mma_int4
}  // namespace hkust_v9
