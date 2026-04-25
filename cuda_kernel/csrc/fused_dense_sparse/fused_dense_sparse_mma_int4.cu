// Fused Dense+Sparse GEMM -- INT4 Tensor Core version (SM89).
//
// Round 12 optimisations:
//   - kBn capped at 64 (eliminate 255-reg spill).
//   - Dense-branch scale_u4/zero_u4 cached in shmem when n_groups <= 32.
//   - Sparse-branch scale_u4[m_tile:, bc] cached in shmem per BSR block.
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
namespace fused_dense_sparse_mma_int4 {

template <int kBn>
__global__ void fused_dense_sparse_mma_int4_kernel(
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
    constexpr int kBm = BROW;
    constexpr int kBk = BCOL;
    constexpr int kMmaK = 64;
    constexpr int kKSteps = kBk / kMmaK;
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
    __shared__ alignas(16) __half s_scale_u4[kBm][32];  // used if cache_sz
    __shared__ alignas(16) __half s_zero_u4 [kBm][32];
    __shared__ __half s_scale_x[kBn];
    __shared__ __half s_scale_block[kBm];               // per BSR block
    __shared__ int s_sum_X[2][kBn];

    const bool cache_sz = (n_groups <= 32);

    if (tid < kBn) {
        int n = n_tile + tid;
        s_scale_x[tid] = (n < T) ? scale_x[n] : __half(0);
    }

    if (cache_sz) {
        for (int idx = tid; idx < kBm * n_groups; idx += kBm) {
            int m_local = idx / n_groups;
            int g       = idx - m_local * n_groups;
            int m = m_tile + m_local;
            if (m < d_out) {
                s_scale_u4[m_local][g] = scale_u4[(int64_t)m * stride_su_m
                                                + (int64_t)g * stride_su_g];
                s_zero_u4 [m_local][g] = zero_u4 [(int64_t)m * stride_zu_m
                                                + (int64_t)g * stride_zu_g];
            } else {
                s_scale_u4[m_local][g] = __half(0);
                s_zero_u4 [m_local][g] = __half(0);
            }
        }
    }

    float y_fp[kMsubPerWarp][kNsubPerCta][4];
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im)
      #pragma unroll
      for (int in = 0; in < kNsubPerCta; ++in)
        #pragma unroll
        for (int r = 0; r < 4; ++r) y_fp[im][in][r] = 0.0f;

    auto issue_w_dense_load = [&](int g, int buf) {
        int m = m_tile + tid;
        uint8_t* dst = &sW[buf][tid][0];
        if (m < d_out) {
            const uint8_t* src = W_low + (int64_t)m * stride_w_m
                                       + (int64_t)(g * bytes_per_group) * stride_w_k;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint4 v = *reinterpret_cast<const uint4*>(src + i * 16);
                *reinterpret_cast<uint4*>(dst + i * 16) = v;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                *reinterpret_cast<uint4*>(dst + i * 16) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    auto issue_x_load = [&](int g_or_bc, int buf) {
        const int total_quads = kBn * 4;
        for (int q = tid; q < total_quads; q += kBm) {
            int row = q >> 2;
            int quad = q & 3;
            int n = n_tile + row;
            uint4 v;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(g_or_bc * bytes_per_group + quad * 16) * stride_x_k;
                v = *reinterpret_cast<const uint4*>(X + off);
            } else {
                v = make_uint4(0, 0, 0, 0);
            }
            *reinterpret_cast<uint4*>(&sX[buf][row][quad * 16]) = v;
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
        uint8_t* dst = &sW[buf][tid][0];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint4 v = *reinterpret_cast<const uint4*>(src + i * 16);
            *reinterpret_cast<uint4*>(dst + i * 16) = v;
        }
    };

    auto issue_scale_block_load = [&](int bc) {
        int m = m_tile + tid;
        s_scale_block[tid] = (m < d_out)
            ? scale_u4[(int64_t)m * stride_su_m + (int64_t)bc * stride_su_g]
            : __half(0);
    };

    auto run_mma_pass = [&](int buf, auto fold_fn, auto prefetch_fn, int g_or_bc) {
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

        // Round 22b: per-m-row prefetch of any (z, s, scale_block, ...) that
        // the fold function will consume.  This eliminates redundant
        // __half2float calls that NVCC cannot hoist out of a lambda boundary.
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im) {
            int msub_base = warp_id * 32 + im * 16;
            int mrow0 = msub_base + (lane >> 2);
            int mrow1 = mrow0 + 8;
            // Prefetch closure returns per-row scalars; ABI is fold-specific.
            auto pr = prefetch_fn(mrow0, mrow1, g_or_bc);
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
                    fold_fn(d_acc[im][in_sub][r], m_global, m_local, n_local,
                            g_or_bc, im, in_sub, r, buf, pr);
                }
            }
        }
    };

    __syncthreads();

    // DENSE BRANCH
    issue_w_dense_load(0, 0);
    issue_x_load(0, 0);
    issue_sum_X_load(0, 0);
    __syncthreads();

    // Round 23: pre-convert s_scale_x[n_local] (fp16 -> fp32) once per CTA.
    //   Invariant across both DENSE and SPARSE passes, so compute once
    //   and reuse.  Indexed by (in_sub, r&1).
    float sxn_cache[kNsubPerCta][2];
    #pragma unroll
    for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
        int nsub_base = in_sub * 8;
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
            int col_local = (lane & 3) * 2 + cc;
            int n_local = nsub_base + col_local;
            if (n_local < kBn) {
                sxn_cache[in_sub][cc] = __half2float(s_scale_x[n_local]);
            } else {
                sxn_cache[in_sub][cc] = 0.0f;
            }
        }
    }

    for (int g = 0; g < n_groups; ++g) {
        const int buf = g & 1;
        if (g + 1 < n_groups) {
            issue_w_dense_load(g + 1, buf ^ 1);
            issue_x_load(g + 1, buf ^ 1);
            issue_sum_X_load(g + 1, buf ^ 1);
        }

        // Round 24: per-g sumxn_cache for dense branch.
        //   sum_X depends on (n_local, g).  Same thread sees only 2 * kNsubPerCta
        //   distinct n_local values.  Lift the int->float conversion out of the
        //   fold loop.
        float sumxn_cache[kNsubPerCta][2];
        #pragma unroll
        for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
            int nsub_base = in_sub * 8;
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                int col_local = (lane & 3) * 2 + cc;
                int n_local = nsub_base + col_local;
                sumxn_cache[in_sub][cc] = (n_local < kBn)
                    ? static_cast<float>(s_sum_X[buf][n_local])
                    : 0.0f;
            }
        }

        // Dense prefetch: (z0, s0, z1, s1) for the two m-rows this thread owns.
        auto prefetch_dense = [&](int mrow0, int mrow1, int gg) {
            struct { float z0, s0, z1, s1; } v{0.0f, 0.0f, 0.0f, 0.0f};
            if (cache_sz) {
                if (mrow0 < kBm) {
                    v.z0 = __half2float(s_zero_u4 [mrow0][gg]);
                    v.s0 = __half2float(s_scale_u4[mrow0][gg]);
                }
                if (mrow1 < kBm) {
                    v.z1 = __half2float(s_zero_u4 [mrow1][gg]);
                    v.s1 = __half2float(s_scale_u4[mrow1][gg]);
                }
            } else {
                int m_g0 = m_tile + mrow0;
                int m_g1 = m_tile + mrow1;
                if (m_g0 < d_out) {
                    v.z0 = __half2float(zero_u4 [(int64_t)m_g0 * stride_zu_m + (int64_t)gg * stride_zu_g]);
                    v.s0 = __half2float(scale_u4[(int64_t)m_g0 * stride_su_m + (int64_t)gg * stride_su_g]);
                }
                if (m_g1 < d_out) {
                    v.z1 = __half2float(zero_u4 [(int64_t)m_g1 * stride_zu_m + (int64_t)gg * stride_zu_g]);
                    v.s1 = __half2float(scale_u4[(int64_t)m_g1 * stride_su_m + (int64_t)gg * stride_su_g]);
                }
            }
            return v;
        };

        auto fold_dense = [&](int d_val, int m_global, int m_local, int n_local,
                              int gg, int im, int in_sub, int r, int bb, auto pr) {
            float z = (r >> 1) ? pr.z1 : pr.z0;
            float s = (r >> 1) ? pr.s1 : pr.s0;
            float sxn = sxn_cache[in_sub][r & 1];  // R23: register cache
            float sumxn = sumxn_cache[in_sub][r & 1];  // R24: register cache
            float corrected = static_cast<float>(d_val) - z * sumxn;
            y_fp[im][in_sub][r] += corrected * s * sxn;
        };
        run_mma_pass(buf, fold_dense, prefetch_dense, g);

        __syncthreads();
    }

    // SPARSE BRANCH
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
            issue_scale_block_load(bc);

            auto prefetch_sparse = [&](int mrow0, int mrow1, int bc_idx) {
                struct { float s0, s1; } v{0.0f, 0.0f};
                if (mrow0 < kBm) v.s0 = __half2float(s_scale_block[mrow0]);
                if (mrow1 < kBm) v.s1 = __half2float(s_scale_block[mrow1]);
                return v;
            };
            auto fold_sparse = [&](int d_val, int m_global, int m_local, int n_local,
                                   int bc_idx, int im, int in_sub, int r, int bb, auto pr) {
                float s = (r >> 1) ? pr.s1 : pr.s0;
                float sxn = sxn_cache[in_sub][r & 1];  // R23: register cache
                y_fp[im][in_sub][r] += 16.0f * static_cast<float>(d_val) * s * sxn;
            };
            run_mma_pass(buf, fold_sparse, prefetch_sparse, bc);

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
        fused_dense_sparse_mma_int4_kernel<kBn><<<grid, block, 0, stream>>>(
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

    // Round 18: kBn=32 bucket extended to T<=128.
    if      (T <= 8)    do_launch(std::integral_constant<int, 8>{});
    else if (T <= 128)  do_launch(std::integral_constant<int, 32>{});
    else                do_launch(std::integral_constant<int, 64>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fused_dense_sparse_mma_int4
}  // namespace hkust_v9
