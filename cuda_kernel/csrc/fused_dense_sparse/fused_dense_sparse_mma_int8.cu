// Fused Dense+Sparse GEMM -- INT8 Tensor Core version (SM89).
//
// Y_total[m,n] = Y_low[m,n] + 16 * Y_high[m,n]
// Uses mma.m16n8k32.s8.s8.s32 for both branches.  Dense branch sweeps
// all n_groups; sparse branch iterates the BSR row for this block-row.
// A single FP32 accumulator per output entry carries both branches'
// contributions so the fused kernel avoids a full intermediate store.

#include "common/arch.cuh"
#include "common/mma_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace fused_dense_sparse_mma_int8 {

template <int kBn>
__global__ void fused_dense_sparse_mma_int8_kernel(
    const uint8_t* __restrict__ W_low,
    const uint8_t* __restrict__ X,
    const __half* __restrict__ scale_u4,
    const __half* __restrict__ zero_u4,
    const int* __restrict__ sum_X,
    const __half* __restrict__ scale_x,
    const uint8_t* __restrict__ W_high_blocks,
    const int* __restrict__ hp_row_offsets,
    const int* __restrict__ hp_col_indices,
    __half* __restrict__ Y,
    int d_out, int d_in, int T,
    int n_groups,
    int64_t stride_w_m,   int64_t stride_w_k,
    int64_t stride_x_n,   int64_t stride_x_k,
    int64_t stride_su_m,  int64_t stride_su_g,
    int64_t stride_zu_m,  int64_t stride_zu_g,
    int64_t stride_sx_n,  int64_t stride_sx_g,
    int64_t stride_wb_blk, int64_t stride_wb_r, int64_t stride_wb_k,
    int64_t stride_y_m,   int64_t stride_y_n
) {
    constexpr int kBm = BROW;                 // 128
    constexpr int kBk = BCOL;                 // 128
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

    __shared__ alignas(16) int8_t sW[2][kBm][kBk];
    __shared__ alignas(16) int8_t sX[2][kBn][kBk];
    __shared__ __half s_scale_x[kBn];
    __shared__ int s_sum_X[2][kBn];

    if (tid < kBn) {
        int n = n_tile + tid;
        s_scale_x[tid] = (n < T) ? scale_x[n] : __half(0);
    }

    float y_fp[kMsubPerWarp][kNsubPerCta][4];
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im)
      #pragma unroll
      for (int in = 0; in < kNsubPerCta; ++in)
        #pragma unroll
        for (int r = 0; r < 4; ++r) y_fp[im][in][r] = 0.0f;

    // Dense loaders.
    auto issue_w_dense_load = [&](int g, int buf) {
        int m = m_tile + tid;
        int8_t* dst = &sW[buf][tid][0];
        if (m < d_out) {
            const uint8_t* src = W_low + (int64_t)m * stride_w_m
                                       + (int64_t)(g * bytes_per_group) * stride_w_k;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_packed0 = reinterpret_cast<const uint32_t*>(src)[4*i + 0];
                uint32_t w_packed1 = reinterpret_cast<const uint32_t*>(src)[4*i + 1];
                int s0, s1;
                unpack_s4_to_s8_x8(w_packed0, s0, s1);
                reinterpret_cast<int*>(dst)[4*i + 0] = s0;
                reinterpret_cast<int*>(dst)[4*i + 1] = s1;
                unpack_s4_to_s8_x8(w_packed1, s0, s1);
                reinterpret_cast<int*>(dst)[4*i + 2] = s0;
                reinterpret_cast<int*>(dst)[4*i + 3] = s1;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kBk / 4; ++i) {
                reinterpret_cast<int*>(dst)[i] = 0;
            }
        }
    };

    auto issue_x_load = [&](int g_or_bc, int buf) {
        const int chunks_per_row = 16;
        const int total_chunks = kBn * chunks_per_row;
        for (int q = tid; q < total_chunks; q += kBm) {
            int row = q / chunks_per_row;
            int ck  = q % chunks_per_row;
            int n = n_tile + row;
            uint32_t packed4;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(g_or_bc * bytes_per_group + ck * 4) * stride_x_k;
                packed4 = *reinterpret_cast<const uint32_t*>(X + off);
            } else {
                packed4 = 0;
            }
            int s0, s1;
            unpack_s4_to_s8_x8(packed4, s0, s1);
            int8_t* dst = &sX[buf][row][ck * 8];
            reinterpret_cast<int*>(dst)[0] = s0;
            reinterpret_cast<int*>(dst)[1] = s1;
        }
    };

    auto issue_sum_X_load = [&](int g, int buf) {
        for (int nk = tid; nk < kBn; nk += kBm) {
            int n = n_tile + nk;
            s_sum_X[buf][nk] = (n < T) ?
                sum_X[(int64_t)n * stride_sx_n + (int64_t)g * stride_sx_g] : 0;
        }
    };

    auto issue_w_sparse_load = [&](int block_idx, int buf) {
        const uint8_t* src = W_high_blocks
                           + (int64_t)block_idx * stride_wb_blk
                           + (int64_t)tid * stride_wb_r;
        int8_t* dst = &sW[buf][tid][0];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t w_packed0 = reinterpret_cast<const uint32_t*>(src)[4*i + 0];
            uint32_t w_packed1 = reinterpret_cast<const uint32_t*>(src)[4*i + 1];
            int s0, s1;
            unpack_s4_to_s8_x8(w_packed0, s0, s1);
            reinterpret_cast<int*>(dst)[4*i + 0] = s0;
            reinterpret_cast<int*>(dst)[4*i + 1] = s1;
            unpack_s4_to_s8_x8(w_packed1, s0, s1);
            reinterpret_cast<int*>(dst)[4*i + 2] = s0;
            reinterpret_cast<int*>(dst)[4*i + 3] = s1;
        }
    };

    // Helper: run a single MMA pass on the currently loaded sW/sX[buf].
    // ``fold_fn`` is a lambda (int d, int m_global, int n_local, int g_or_bc) -> void
    // that folds the int32 MMA output into y_fp with branch-specific epilogue.
    auto run_mma_pass = [&](int buf, auto fold_fn, int g_or_bc) {
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
                    a0 = *reinterpret_cast<const uint32_t*>(&sW[buf][row0][col0]);
                    a2 = *reinterpret_cast<const uint32_t*>(&sW[buf][row0][col2]);
                }
                if (row1 < kBm) {
                    a1 = *reinterpret_cast<const uint32_t*>(&sW[buf][row1][col0]);
                    a3 = *reinterpret_cast<const uint32_t*>(&sW[buf][row1][col2]);
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
                int k_row_base0 = (lane & 3) * 4;
                int k_row_base1 = k_row_base0 + 16;
                if (n_base + n_row_in_sub < kBn) {
                    const int8_t* p0 = &sX[buf][n_base + n_row_in_sub][k_base + k_row_base0];
                    const int8_t* p1 = &sX[buf][n_base + n_row_in_sub][k_base + k_row_base1];
                    b_regs[in_sub][0] = *reinterpret_cast<const uint32_t*>(p0);
                    b_regs[in_sub][1] = *reinterpret_cast<const uint32_t*>(p1);
                } else {
                    b_regs[in_sub][0] = 0;
                    b_regs[in_sub][1] = 0;
                }
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

        // Fold into y_fp via caller's epilogue lambda.
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
                    fold_fn(d_acc[im][in_sub][r], m_global, n_local, g_or_bc,
                            im, in_sub, r, buf);
                }
            }
        }
    };

    __syncthreads();

    // =================================================================
    // DENSE BRANCH
    // =================================================================
    issue_w_dense_load(0, 0);
    issue_x_load(0, 0);
    issue_sum_X_load(0, 0);
    __syncthreads();

    for (int g = 0; g < n_groups; ++g) {
        const int buf = g & 1;
        if (g + 1 < n_groups) {
            issue_w_dense_load(g + 1, buf ^ 1);
            issue_x_load(g + 1, buf ^ 1);
            issue_sum_X_load(g + 1, buf ^ 1);
        }

        auto fold_dense = [&](int d_val, int m_global, int n_local,
                              int gg, int im, int in_sub, int r, int bb) {
            float z = __half2float(
                zero_u4[(int64_t)m_global * stride_zu_m
                      + (int64_t)gg * stride_zu_g]
            );
            float s = __half2float(
                scale_u4[(int64_t)m_global * stride_su_m
                       + (int64_t)gg * stride_su_g]
            );
            float sxn = __half2float(s_scale_x[n_local]);
            float sumxn = static_cast<float>(s_sum_X[bb][n_local]);
            float corrected = static_cast<float>(d_val) - z * sumxn;
            y_fp[im][in_sub][r] += corrected * s * sxn;
        };
        run_mma_pass(buf, fold_dense, g);

        __syncthreads();
    }

    // =================================================================
    // SPARSE BRANCH
    // =================================================================
    const int blk_start = hp_row_offsets[br];
    const int blk_end   = hp_row_offsets[br + 1];

    if (blk_start < blk_end) {
        int bc0 = __ldg(&hp_col_indices[blk_start]);
        issue_w_sparse_load(blk_start, 0);
        issue_x_load(bc0, 0);
        __syncthreads();

        for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
            const int bc = __ldg(&hp_col_indices[block_idx]);
            const int buf = (block_idx - blk_start) & 1;
            if (block_idx + 1 < blk_end) {
                int bc_next = __ldg(&hp_col_indices[block_idx + 1]);
                issue_w_sparse_load(block_idx + 1, buf ^ 1);
                issue_x_load(bc_next, buf ^ 1);
            }

            auto fold_sparse = [&](int d_val, int m_global, int n_local,
                                   int bc_idx, int im, int in_sub, int r, int bb) {
                float s = __half2float(
                    scale_u4[(int64_t)m_global * stride_su_m
                           + (int64_t)bc_idx * stride_su_g]
                );
                float sxn = __half2float(s_scale_x[n_local]);
                y_fp[im][in_sub][r] += 16.0f * static_cast<float>(d_val) * s * sxn;
            };
            run_mma_pass(buf, fold_sparse, bc);

            __syncthreads();
        }
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
    TORCH_CHECK(scale_x.stride(0) == 1);

    const int d_in_half = W_low.size(1);
    TORCH_CHECK(d_in_half * 2 == d_in);
    const int T = X_s4.size(0);
    TORCH_CHECK(d_in % BCOL == 0);
    const int n_groups = d_in / BCOL;

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
    constexpr int kBm = BROW;

    auto do_launch = [&](auto kBn_c) {
        constexpr int kBn = decltype(kBn_c)::value;
        dim3 block(kBm, 1, 1);
        dim3 grid(ceil_div(d_out, kBm), ceil_div(T, kBn), 1);
        fused_dense_sparse_mma_int8_kernel<kBn><<<grid, block, 0, stream>>>(
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
            d_out, d_in, T, n_groups,
            W_low.stride(0), W_low.stride(1),
            X_s4.stride(0), X_s4.stride(1),
            scale_u4.stride(0), scale_u4.stride(1),
            zero_u4.stride(0), zero_u4.stride(1),
            sum_X.stride(0), sum_X.stride(1),
            W_high_blocks.stride(0), W_high_blocks.stride(1), W_high_blocks.stride(2),
            Y_total.stride(0), Y_total.stride(1)
        );
    };

    if      (T <= 8)    do_launch(std::integral_constant<int, 8>{});
    else if (T <= 64)   do_launch(std::integral_constant<int, 64>{});
    else                do_launch(std::integral_constant<int, 128>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fused_dense_sparse_mma_int8
}  // namespace hkust_v9
