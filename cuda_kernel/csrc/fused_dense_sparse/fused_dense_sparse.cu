// Fused Dense + Sparse GEMM (CUDA, SM89).
//
// Drop-in replacement for
// ``kernel.triton_kernel.fused_dense_sparse_gemm.fused_dense_sparse_kernel``.
//
// Semantics (bit-accurate to Triton reference within FP32 rounding):
//   Y_total[m, n] = Y_low[m, n] + 16 * Y_high[m, n]
// where Y_low and Y_high are exactly the outputs of the dense and
// sparse kernels above.  See those files for the per-branch
// derivation; this kernel simply fuses the two K-loops into one so
// that:
//   - we launch one kernel instead of two (save ~5-10us)
//   - we avoid a full (d_out, T) FP16 store + read between them
//   - the accumulator stays in registers across both branches
//
// Layout and thread mapping mirror the dense kernel exactly (one CTA
// per (BM=128, kBn) tile, one thread per m row), with the extra BSR
// loop appended after the dense K-loop.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace fused_dense_sparse {

__device__ __forceinline__ int8_t s4_lo(uint8_t b) {
    int v = b & 0x0F;
    return static_cast<int8_t>(v - ((v & 0x08) << 1));
}
__device__ __forceinline__ int8_t s4_hi(uint8_t b) {
    int v = (b >> 4) & 0x0F;
    return static_cast<int8_t>(v - ((v & 0x08) << 1));
}

__device__ __forceinline__ void unpack_4bytes_to_2int32(
    uint32_t packed4, int& out0, int& out1
) {
    uint8_t b0 = (packed4      ) & 0xFF;
    uint8_t b1 = (packed4 >>  8) & 0xFF;
    uint8_t b2 = (packed4 >> 16) & 0xFF;
    uint8_t b3 = (packed4 >> 24) & 0xFF;
    int8_t c0 = s4_lo(b0), c1 = s4_hi(b0);
    int8_t c2 = s4_lo(b1), c3 = s4_hi(b1);
    int8_t c4 = s4_lo(b2), c5 = s4_hi(b2);
    int8_t c6 = s4_lo(b3), c7 = s4_hi(b3);
    out0 = (static_cast<int>(static_cast<uint8_t>(c3)) << 24)
         | (static_cast<int>(static_cast<uint8_t>(c2)) << 16)
         | (static_cast<int>(static_cast<uint8_t>(c1)) <<  8)
         | (static_cast<int>(static_cast<uint8_t>(c0))      );
    out1 = (static_cast<int>(static_cast<uint8_t>(c7)) << 24)
         | (static_cast<int>(static_cast<uint8_t>(c6)) << 16)
         | (static_cast<int>(static_cast<uint8_t>(c5)) <<  8)
         | (static_cast<int>(static_cast<uint8_t>(c4))      );
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int kBn>
__global__ void fused_dense_sparse_kernel(
    const uint8_t* __restrict__ W_low,             // (d_out, d_in/2)
    const uint8_t* __restrict__ X,                 // (T, d_in/2)
    const __half* __restrict__ scale_u4,           // (d_out, n_groups)
    const __half* __restrict__ zero_u4,            // (d_out, n_groups)
    const int* __restrict__ sum_X,                 // (T, n_groups)
    const __half* __restrict__ scale_x,            // (T,)
    const uint8_t* __restrict__ W_high_blocks,     // (n_hp, BROW, BCOL/2)
    const int* __restrict__ hp_row_offsets,        // (nrow + 1,)
    const int* __restrict__ hp_col_indices,        // (n_hp,)
    __half* __restrict__ Y,                        // (d_out, T)
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
    constexpr int kBm = BROW;      // 128; one block-row per CTA, matches sparse path
    const int tid = threadIdx.x;
    const int br = blockIdx.x;                  // block-row index
    const int m = br * kBm + tid;
    const bool m_active = m < d_out;

    const int n_base = blockIdx.y * kBn;
    const int bytes_per_group = BCOL >> 1;      // 64

    // -----------------------------------------------------------------
    // Per-N hoist: scale_x, staged in shmem
    // -----------------------------------------------------------------
    __shared__ __half s_scale_x[kBn];
    if (tid < kBn) {
        int n = n_base + tid;
        // scale_x is 1D contiguous (T,) stride=1 (enforced by
        // launcher).  Do not reuse stride_sx_n which is sum_X's row
        // stride (n_groups) here.
        s_scale_x[tid] = (n < T) ? scale_x[n] : __half(0);
    }

    float y_acc[kBn];
    #pragma unroll
    for (int k = 0; k < kBn; ++k) y_acc[k] = 0.0f;

    __shared__ uint8_t sX[kBn][64];
    __shared__ int s_sum_X[kBn];

    __syncthreads();  // publish s_scale_x

    // =================================================================
    // DENSE branch : full K sweep
    // =================================================================
    for (int g = 0; g < n_groups; ++g) {
        const int total_x_bytes = kBn * 64;
        for (int idx = tid; idx < total_x_bytes; idx += kBm) {
            int row = idx >> 6;
            int col = idx & 63;
            int n = n_base + row;
            uint8_t v = 0;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(g * bytes_per_group + col) * stride_x_k;
                v = X[off];
            }
            sX[row][col] = v;
        }
        if (tid < kBn) {
            int n = n_base + tid;
            s_sum_X[tid] = (n < T) ?
                sum_X[(int64_t)n * stride_sx_n + (int64_t)g * stride_sx_g] : 0;
        }
        __syncthreads();

        if (m_active) {
            // 128-bit (uint4) loads; see dense_gemm.cu Round 4 note.
            uint32_t w_words[16];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int64_t off_w = (int64_t)m * stride_w_m
                              + (int64_t)(g * bytes_per_group + i * 16) * stride_w_k;
                uint4 v = __ldg(
                    reinterpret_cast<const uint4*>(W_low + off_w)
                );
                w_words[4*i    ] = v.x;
                w_words[4*i + 1] = v.y;
                w_words[4*i + 2] = v.z;
                w_words[4*i + 3] = v.w;
            }
            int w_dp4a[32];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int w0, w1;
                unpack_4bytes_to_2int32(w_words[i], w0, w1);
                w_dp4a[2*i    ] = w0;
                w_dp4a[2*i + 1] = w1;
            }

            float scale_g = __half2float(
                scale_u4[(int64_t)m * stride_su_m + (int64_t)g * stride_su_g]
            );
            float zero_g = __half2float(
                zero_u4[(int64_t)m * stride_zu_m + (int64_t)g * stride_zu_g]
            );

            // ILP-friendly loop swap (see dense_gemm.cu for rationale):
            // K outside, N inside, so kBn independent dp4a chains feed
            // the scheduler at once and hide the 4-6 cycle dp4a latency.
            int acc_n[kBn];
            #pragma unroll
            for (int nk = 0; nk < kBn; ++nk) acc_n[nk] = 0;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int x0_n[kBn], x1_n[kBn];
                #pragma unroll
                for (int nk = 0; nk < kBn; ++nk) {
                    uint32_t xp = reinterpret_cast<const uint32_t*>(&sX[nk][0])[i];
                    unpack_4bytes_to_2int32(xp, x0_n[nk], x1_n[nk]);
                }
                int w0 = w_dp4a[2*i];
                int w1 = w_dp4a[2*i + 1];
                #pragma unroll
                for (int nk = 0; nk < kBn; ++nk) {
                    acc_n[nk] = __dp4a(w0, x0_n[nk], acc_n[nk]);
                    acc_n[nk] = __dp4a(w1, x1_n[nk], acc_n[nk]);
                }
            }

            #pragma unroll
            for (int nk = 0; nk < kBn; ++nk) {
                int n = n_base + nk;
                if (n >= T) break;
                float corrected = static_cast<float>(acc_n[nk])
                                - zero_g * static_cast<float>(s_sum_X[nk]);
                float sxn = __half2float(s_scale_x[nk]);
                y_acc[nk] += corrected * scale_g * sxn;
            }
        }

        __syncthreads();
    }

    // =================================================================
    // SPARSE branch    // SPARSE branch : BSR loop for this block-row, contributes
    // 16 * dot(W_high, X_s4_bc) * scale[m, bc] * scale_x[n] to y_acc.
    // =================================================================
    const int blk_start = hp_row_offsets[br];
    const int blk_end   = hp_row_offsets[br + 1];

    for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
        const int bc = hp_col_indices[block_idx];

        const int total_x_bytes = kBn * 64;
        for (int idx = tid; idx < total_x_bytes; idx += kBm) {
            int row = idx >> 6;
            int col = idx & 63;
            int n = n_base + row;
            uint8_t v = 0;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(bc * bytes_per_group + col) * stride_x_k;
                v = X[off];
            }
            sX[row][col] = v;
        }
        __syncthreads();

        if (m_active) {
            // 128-bit (uint4) loads; see dense_gemm.cu Round 4 note.
            uint32_t w_words[16];
            int64_t w_row_base = (int64_t)block_idx * stride_wb_blk
                               + (int64_t)tid * stride_wb_r;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int64_t off_w = w_row_base + (int64_t)(i * 16) * stride_wb_k;
                uint4 v = __ldg(
                    reinterpret_cast<const uint4*>(W_high_blocks + off_w)
                );
                w_words[4*i    ] = v.x;
                w_words[4*i + 1] = v.y;
                w_words[4*i + 2] = v.z;
                w_words[4*i + 3] = v.w;
            }
            int w_dp4a[32];
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int w0, w1;
                unpack_4bytes_to_2int32(w_words[i], w0, w1);
                w_dp4a[2*i    ] = w0;
                w_dp4a[2*i + 1] = w1;
            }

            float scale_bc = __half2float(
                scale_u4[(int64_t)m * stride_su_m + (int64_t)bc * stride_su_g]
            );

            // ILP-friendly K-outside / N-inside loop (see dense kernel).
            int acc_n[kBn];
            #pragma unroll
            for (int nk = 0; nk < kBn; ++nk) acc_n[nk] = 0;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                int x0_n[kBn], x1_n[kBn];
                #pragma unroll
                for (int nk = 0; nk < kBn; ++nk) {
                    uint32_t xp = reinterpret_cast<const uint32_t*>(&sX[nk][0])[i];
                    unpack_4bytes_to_2int32(xp, x0_n[nk], x1_n[nk]);
                }
                int w0 = w_dp4a[2*i];
                int w1 = w_dp4a[2*i + 1];
                #pragma unroll
                for (int nk = 0; nk < kBn; ++nk) {
                    acc_n[nk] = __dp4a(w0, x0_n[nk], acc_n[nk]);
                    acc_n[nk] = __dp4a(w1, x1_n[nk], acc_n[nk]);
                }
            }

            #pragma unroll
            for (int nk = 0; nk < kBn; ++nk) {
                int n = n_base + nk;
                if (n >= T) break;
                float sxn = __half2float(s_scale_x[nk]);
                // 16x factor: matches Triton fused kernel's
                // ``y_acc += 16.0 * acc_block * scale_bc * sxn``.
                y_acc[nk] += 16.0f * static_cast<float>(acc_n[nk]) * scale_bc * sxn;
            }
        }

        __syncthreads();
    }

    // =================================================================
    // Store Y_total (d_out, T) fp16
    // =================================================================
    if (m_active) {
        #pragma unroll
        for (int nk = 0; nk < kBn; ++nk) {
            int n = n_base + nk;
            if (n < T) {
                int64_t y_off = (int64_t)m * stride_y_m + (int64_t)n * stride_y_n;
                Y[y_off] = __float2half(y_acc[nk]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
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
    TORCH_CHECK(W_low.dtype() == torch::kInt8, "W_low must be int8");
    TORCH_CHECK(X_s4.dtype() == torch::kInt8, "X_s4 must be int8");
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf, "scale_u4 must be fp16");
    TORCH_CHECK(zero_u4.dtype() == torch::kHalf, "zero_u4 must be fp16");
    TORCH_CHECK(sum_X.dtype() == torch::kInt32, "sum_X must be int32");
    TORCH_CHECK(scale_x.dtype() == torch::kHalf, "scale_x must be fp16");
    TORCH_CHECK(Y_total.dtype() == torch::kHalf, "Y_total must be fp16");
    TORCH_CHECK(W_low.stride(1) == 1, "W_low must be K-contiguous");
    TORCH_CHECK(X_s4.stride(1) == 1, "X_s4 must be K-contiguous");
    TORCH_CHECK(scale_x.stride(0) == 1, "scale_x must be contiguous");

    const int d_in_half = W_low.size(1);
    TORCH_CHECK(d_in_half * 2 == d_in, "W_low d_in mismatch");
    const int T = X_s4.size(0);
    TORCH_CHECK(d_in % BCOL == 0, "d_in must be divisible by BCOL=128");
    const int n_groups = d_in / BCOL;

    // Make sure W_high_blocks and BSR tensors are always valid even
    // when there are no high-precision blocks.  The fused kernel's
    // BSR loop will iterate 0 times and the kernel behaves as
    // dense-only; in that case the caller is expected to use the
    // dense kernel instead, but we must not crash.
    if (W_high_blocks.numel() == 0) {
        W_high_blocks = torch::zeros(
            {0, BROW, BCOL / 2},
            torch::TensorOptions().dtype(torch::kInt8).device(W_low.device())
        );
    }
    TORCH_CHECK(W_high_blocks.dtype() == torch::kInt8, "W_high_blocks must be int8");
    TORCH_CHECK(W_high_blocks.stride(2) == 1, "W_high_blocks must be K-contiguous");
    TORCH_CHECK(hp_row_offsets.dtype() == torch::kInt32, "hp_row_offsets must be int32");
    TORCH_CHECK(hp_col_indices.dtype() == torch::kInt32, "hp_col_indices must be int32");
    // grid.x == ceil_div(d_out, BROW) == nrow; indexes hp_row_offsets in [0, nrow).
    const int nrow = (d_out + BROW - 1) / BROW;
    TORCH_CHECK(hp_row_offsets.numel() == nrow + 1,
                "hp_row_offsets length must be nrow+1 (got ",
                hp_row_offsets.numel(), ", expected ", nrow + 1, ")");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kBm = BROW;

    auto do_launch = [&](auto kBn_c) {
        constexpr int kBn = decltype(kBn_c)::value;
        dim3 block(kBm, 1, 1);
        dim3 grid(ceil_div(d_out, kBm), ceil_div(T, kBn), 1);
        fused_dense_sparse_kernel<kBn><<<grid, block, 0, stream>>>(
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

    // See dense_gemm.cu iter-Round 3 notes: keep kBn <= 4 to avoid
    // register spill of the acc_n/x0_n/x1_n register arrays that were
    // introduced by the K-outside/N-inside loop swap.
    if      (T <= 1)   do_launch(std::integral_constant<int, 1>{});
    else if (T <= 16)  do_launch(std::integral_constant<int, 2>{});
    else               do_launch(std::integral_constant<int, 4>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fused_dense_sparse
}  // namespace hkust_v9
