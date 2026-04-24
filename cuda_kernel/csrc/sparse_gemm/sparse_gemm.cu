// Block-sparse SINT4 x SINT4 GEMM (CUDA, SM89).
//
// Drop-in replacement for
// ``kernel.triton_kernel.sparse_s4s4_gemm.sparse_gemm_kernel``.
//
// Contract (must match the Triton reference up to FP32->FP16 rounding):
//   Y_high[m, n] = sum over {block_idx : hp_row_offsets[br] <= block_idx
//                             < hp_row_offsets[br+1]} of
//                   dot(W_high[block_idx, m_in_blk, :],
//                       X_s4[n, bc*BCOL : (bc+1)*BCOL])
//                   * scale_u4[m, bc] * scale_x[n]
// where br = m // BROW and m_in_blk = m % BROW.  No zero, no sum_X:
// the SINT8 high-bit path is already symmetric around zero.
//
// If the block-row for this program has no high-precision blocks the
// loop iterates zero times and we store zeros.
//
// Strategy
// --------
// Same SIMT dp4a scheme as dense_gemm.  The inner loop is identical
// apart from:
//   - W is indexed through (block_idx, row_in_blk, col) instead of
//     (m, g*bytes_per_group + col);
//   - the group index used for scale_u4 is the data-dependent ``bc``
//     rather than the K-loop index;
//   - no zero*sum_X correction in the epilogue.
//
// This kernel intentionally launches one CTA per (BM=128 block-row,
// BN) output tile; for empty rows the CTA cost is just the prologue
// + single tl.store of zeros.  A persistent-queue variant is future
// work (see analysis notes).

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace sparse_gemm {

// Same sign-extend + unpack as dense_gemm.  We duplicate the helper
// deliberately so each .cu is self-contained; the compiler will inline
// either way, and having a single header would force a coupling we'd
// regret once MMA PTX replaces dp4a in one kernel but not the other.

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
//
// Grid  : (ceil_div(d_out, BM=128), ceil_div(T, kBn), 1)
// Block : 128 threads (4 warps); tid -> m_in_blk.

template <int kBn>
__global__ void sparse_gemm_kernel(
    const uint8_t* __restrict__ W_high_blocks,  // (n_hp, BROW, BCOL/2)
    const int* __restrict__ hp_row_offsets,     // (nrow + 1,)
    const int* __restrict__ hp_col_indices,     // (n_hp,)
    const uint8_t* __restrict__ X,              // (T, d_in/2)
    const __half* __restrict__ scale_u4,        // (d_out, n_groups)
    const __half* __restrict__ scale_x,         // (T,)
    __half* __restrict__ Y,                     // (d_out, T)
    int d_out, int d_in, int T,
    int64_t stride_wb_blk, int64_t stride_wb_r, int64_t stride_wb_k,
    int64_t stride_x_n,  int64_t stride_x_k,
    int64_t stride_su_m, int64_t stride_su_g,
    int64_t stride_sx_n,
    int64_t stride_y_m,  int64_t stride_y_n
) {
    constexpr int kBm = BROW;   // 128; one block-row per CTA
    const int tid = threadIdx.x;
    const int br = blockIdx.x;                   // block-row == pid_m
    const int m = br * kBm + tid;
    const bool m_active = m < d_out;

    const int n_base = blockIdx.y * kBn;
    const int bytes_per_group = BCOL >> 1;       // 64
    const int d_in_half = d_in >> 1;

    // Stage per-N scale_x into shmem once.
    __shared__ __half s_scale_x[kBn];
    if (tid < kBn) {
        int n = n_base + tid;
        s_scale_x[tid] = (n < T) ? scale_x[(int64_t)n * stride_sx_n] : __half(0);
    }

    // Accumulators, one per output column owned by this thread.
    float y_acc[kBn];
    #pragma unroll
    for (int k = 0; k < kBn; ++k) y_acc[k] = 0.0f;

    __shared__ uint8_t sX[kBn][64];

    // Fetch BSR slice for this block-row.
    const int blk_start = hp_row_offsets[br];
    const int blk_end   = hp_row_offsets[br + 1];

    __syncthreads();  // publish s_scale_x

    for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
        const int bc = hp_col_indices[block_idx];

        // --- Stage X rows for group bc (same 64 bytes per row as dense) ---
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
            // Load the W_high block row for this thread's m_in_blk (== tid).
            // Layout: W_high_blocks[block_idx, tid, 0..63].
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

            // ILP-friendly K-outside / N-inside loop (see dense_gemm.cu).
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
                y_acc[nk] += static_cast<float>(acc_n[nk]) * scale_bc * sxn;
            }
        }

        __syncthreads();
    }

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
    torch::Tensor W_high_blocks,
    torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
    torch::Tensor X_s4, torch::Tensor scale_u4, torch::Tensor scale_x,
    torch::Tensor Y_high,
    int d_out, int d_in
) {
    TORCH_CHECK(W_high_blocks.dtype() == torch::kInt8, "W_high_blocks must be int8");
    TORCH_CHECK(hp_row_offsets.dtype() == torch::kInt32, "hp_row_offsets must be int32");
    TORCH_CHECK(hp_col_indices.dtype() == torch::kInt32, "hp_col_indices must be int32");
    TORCH_CHECK(X_s4.dtype() == torch::kInt8, "X_s4 must be int8");
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf, "scale_u4 must be fp16");
    TORCH_CHECK(scale_x.dtype() == torch::kHalf, "scale_x must be fp16");
    TORCH_CHECK(Y_high.dtype() == torch::kHalf, "Y_high must be fp16");
    TORCH_CHECK(X_s4.stride(1) == 1, "X_s4 must be K-contiguous");
    // W_high_blocks layout: (n_hp, BROW, BCOL/2), require
    // stride(2) == 1 (K-contiguous) for uint32 casts.
    TORCH_CHECK(W_high_blocks.dim() == 3, "W_high_blocks must be 3D");
    TORCH_CHECK(W_high_blocks.stride(2) == 1, "W_high_blocks must be K-contiguous");

    const int T = X_s4.size(0);
    // grid.x == ceil_div(d_out, BROW) == nrow by construction, so
    // ``blockIdx.x`` indexes hp_row_offsets in [0, nrow) regardless of
    // whether d_out is a multiple of BROW; extra M lanes are masked.
    const int nrow = (d_out + BROW - 1) / BROW;
    TORCH_CHECK(hp_row_offsets.numel() == nrow + 1,
                "hp_row_offsets length must be nrow+1 (got ",
                hp_row_offsets.numel(), ", expected ", nrow + 1, ")");
    // Zero out output first (the kernel may skip whole block-rows).
    Y_high.zero_();
    if (W_high_blocks.size(0) == 0) {
        return;  // nothing to do; Y_high is zero already.
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kBm = BROW;

    auto do_launch = [&](auto kBn_c) {
        constexpr int kBn = decltype(kBn_c)::value;
        dim3 block(kBm, 1, 1);
        dim3 grid(ceil_div(d_out, kBm), ceil_div(T, kBn), 1);
        sparse_gemm_kernel<kBn><<<grid, block, 0, stream>>>(
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

    // See dense_gemm.cu iter-Round 3 notes: keep kBn <= 4 to avoid
    // register spill.
    if      (T <= 1)   do_launch(std::integral_constant<int, 1>{});
    else if (T <= 8)   do_launch(std::integral_constant<int, 2>{});
    else               do_launch(std::integral_constant<int, 4>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace sparse_gemm
}  // namespace hkust_v9
