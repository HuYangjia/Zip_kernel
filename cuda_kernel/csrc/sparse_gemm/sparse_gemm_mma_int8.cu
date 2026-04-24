// Block-sparse SINT4 x SINT4 GEMM -- INT8 Tensor Core version (SM89).
// Single-buffered shmem to fit SM89's 48 KB static shmem budget.

#include "common/arch.cuh"
#include "common/mma_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace sparse_gemm_mma_int8 {

template <int kBn>
__global__ void sparse_gemm_mma_int8_kernel(
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
    constexpr int kMmaK = 32;
    constexpr int kKSteps = kBk / kMmaK;      // 4
    constexpr int kMsubPerWarp = 2;
    constexpr int kNsubPerCta = (kBn + 7) / 8;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int br = blockIdx.x;
    const int m_tile = br * kBm;
    const int n_tile = blockIdx.y * kBn;
    const int bytes_per_group = BCOL >> 1;

    __shared__ alignas(16) int8_t sW[kBm][kBk];
    __shared__ alignas(16) int8_t sX[kBn][kBk];
    __shared__ __half s_scale_x[kBn];

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

    auto issue_w_load = [&](int block_idx) {
        const uint8_t* src = W_high_blocks
                           + (int64_t)block_idx * stride_wb_blk
                           + (int64_t)tid * stride_wb_r;
        int8_t* dst = &sW[tid][0];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t w0 = reinterpret_cast<const uint32_t*>(src)[4*i + 0];
            uint32_t w1 = reinterpret_cast<const uint32_t*>(src)[4*i + 1];
            int s0, s1;
            unpack_s4_to_s8_x8(w0, s0, s1);
            reinterpret_cast<int*>(dst)[4*i + 0] = s0;
            reinterpret_cast<int*>(dst)[4*i + 1] = s1;
            unpack_s4_to_s8_x8(w1, s0, s1);
            reinterpret_cast<int*>(dst)[4*i + 2] = s0;
            reinterpret_cast<int*>(dst)[4*i + 3] = s1;
        }
    };

    auto issue_x_load = [&](int bc) {
        const int chunks_per_row = 16;
        const int total_chunks = kBn * chunks_per_row;
        for (int q = tid; q < total_chunks; q += kBm) {
            int row = q / chunks_per_row;
            int ck  = q % chunks_per_row;
            int n = n_tile + row;
            uint32_t packed4;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(bc * bytes_per_group + ck * 4) * stride_x_k;
                packed4 = *reinterpret_cast<const uint32_t*>(X + off);
            } else {
                packed4 = 0;
            }
            int s0, s1;
            unpack_s4_to_s8_x8(packed4, s0, s1);
            int8_t* dst = &sX[row][ck * 8];
            reinterpret_cast<int*>(dst)[0] = s0;
            reinterpret_cast<int*>(dst)[1] = s1;
        }
    };

    __syncthreads();

    for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
        const int bc = __ldg(&hp_col_indices[block_idx]);

        issue_w_load(block_idx);
        issue_x_load(bc);
        __syncthreads();

        int d_acc[kMsubPerWarp][kNsubPerCta][4];
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im)
          #pragma unroll
          for (int in = 0; in < kNsubPerCta; ++in)
            #pragma unroll
            for (int r = 0; r < 4; ++r) d_acc[im][in][r] = 0;

        #pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
            const int k_base = ks * kMmaK;

            uint32_t a_regs[kMsubPerWarp][4];
            #pragma unroll
            for (int im = 0; im < kMsubPerWarp; ++im) {
                int msub_base = warp_id * 32 + im * 16;
                int row0 = msub_base + (lane >> 2);
                int row1 = row0 + 8;
                int col0 = k_base + (lane & 3) * 4;
                int col2 = col0 + 16;
                uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                if (row0 < kBm) {
                    a0 = *reinterpret_cast<const uint32_t*>(&sW[row0][col0]);
                    a2 = *reinterpret_cast<const uint32_t*>(&sW[row0][col2]);
                }
                if (row1 < kBm) {
                    a1 = *reinterpret_cast<const uint32_t*>(&sW[row1][col0]);
                    a3 = *reinterpret_cast<const uint32_t*>(&sW[row1][col2]);
                }
                a_regs[im][0] = a0;
                a_regs[im][1] = a1;
                a_regs[im][2] = a2;
                a_regs[im][3] = a3;
            }

            uint32_t b_regs[kNsubPerCta][2];
            #pragma unroll
            for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                int n_base = in_sub * 8;
                int n_row_in_sub = lane >> 2;
                int k0 = k_base + (lane & 3) * 4;
                int k1 = k0 + 16;
                uint32_t b0 = 0, b1 = 0;
                if (n_base + n_row_in_sub < kBn) {
                    b0 = *reinterpret_cast<const uint32_t*>(&sX[n_base + n_row_in_sub][k0]);
                    b1 = *reinterpret_cast<const uint32_t*>(&sX[n_base + n_row_in_sub][k1]);
                }
                b_regs[in_sub][0] = b0;
                b_regs[in_sub][1] = b1;
            }

            #pragma unroll
            for (int im = 0; im < kMsubPerWarp; ++im) {
                #pragma unroll
                for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                    mma_m16n8k32_s8s8s32(
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

        // Sparse epilogue.
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
                    float s = __half2float(
                        scale_u4[(int64_t)m_global * stride_su_m
                               + (int64_t)bc * stride_su_g]
                    );
                    float sxn = __half2float(s_scale_x[n_local]);
                    y_fp[im][in_sub][r] += static_cast<float>(d_acc[im][in_sub][r])
                                         * s * sxn;
                }
            }
        }

        __syncthreads();
    }

    // Writeback.
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
        sparse_gemm_mma_int8_kernel<kBn><<<grid, block, 0, stream>>>(
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

    if      (T <= 8)    do_launch(std::integral_constant<int, 8>{});
    else if (T <= 64)   do_launch(std::integral_constant<int, 64>{});
    else                do_launch(std::integral_constant<int, 128>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sparse_gemm_mma_int8
}  // namespace hkust_v9
