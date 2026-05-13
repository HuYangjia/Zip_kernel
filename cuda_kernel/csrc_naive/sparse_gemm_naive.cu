// Naive block-sparse SINT4 x SINT4 GEMM -- L1 ("Tensor-Core naive").
//
// Upgraded from the L0 scalar-IMAD kernel to mma.m16n8k64.s4.s4.s32 so
// the naive vs optimised comparison isolates the cost of pipelining
// and fusion, not the cost of not-using TC at all.
//
// Deliberately NAIVE at the MMA level:
//   * 4 kernels stay separate -- no fusion with quant / dense / add.
//   * Single-buffered shmem (no cp.async, no double-buffer of W/X).
//   * No per-block scale prefetch into shmem (read HBM each block).
//   * Fixed kBn=32 (no T-bucket dispatcher).
//   * Plain LDG into shmem + __syncthreads + MMA; operand regs are
//     assembled by direct lane-indexed 32-bit reads from shmem
//     (no ldmatrix).
//
// Semantic contract (matches L0 sparse naive and optimised mma.int4):
//   Y[m,n] = scale_x[n] * sum_{blk in row_block m_block} [
//               acc_blk * scale_u4[m, bc(blk)]
//            ]
// W_high has no zero-point (sign-extended s4), which is exactly what
// mma.s4.s4 reads natively.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9_naive {
namespace sparse_gemm {

// ---------------------------------------------------------------------------
// Inlined MMA intrinsic (kept local to this TU; naive tree is
// intentionally header-free).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k64_s4s4s32(
    int& d0, int& d1, int& d2, int& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int c0, int c1, int c2, int c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}

// ---------------------------------------------------------------------------
// Kernel configuration.
//
// BSR is parameterised by BROW=128 (fixed by the packer), so kBm=128
// naturally aligns 128 threads per CTA = 4 warps.  kBn=32 matches the
// optimised kernel's T<=96 bucket and keeps register pressure moderate.
// ---------------------------------------------------------------------------
static constexpr int kBm          = 128;          // = BROW
static constexpr int kBn          = 32;
static constexpr int kBk          = 128;          // = BCOL, one group
static constexpr int kBkB         = kBk / 2;      // 64 packed bytes
static constexpr int kMmaK        = 64;
static constexpr int kKSteps      = kBk / kMmaK;  // 2
static constexpr int kMsubPerWarp = 2;            // 2 * 16 = 32 rows per warp
static constexpr int kNsubPerCta  = kBn / 8;      // 4

// Grid  : (d_out/BROW, ceil(T/kBn), 1)
// Block : (kBm, 1, 1) == 128 threads (4 warps)
__global__ void sparse_gemm_naive_kernel(
    const uint8_t* __restrict__ W_high_blocks,  // (n_blocks, 128, 64)
    const int*     __restrict__ hp_row_offsets, // (d_out/128 + 1,)
    const int*     __restrict__ hp_col_indices, // (n_blocks,)
    const uint8_t* __restrict__ X,              // (T, d_in/2)
    const __half*  __restrict__ scale_u4,       // (d_out, n_groups)
    const __half*  __restrict__ scale_x,        // (T,)
    __half*        __restrict__ Y,              // (d_out, T)
    int d_out, int d_in, int T
) {
    const int n_groups = d_in / kBk;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int br     = blockIdx.x;
    const int m_tile = br * kBm;
    const int n_tile = blockIdx.y * kBn;

    // Shared-memory tiles.
    //   sW : 128 rows * 64 bytes = 8 KiB
    //   sX :  32 rows * 64 bytes = 2 KiB
    __shared__ alignas(16) uint8_t sW[kBm][kBkB];
    __shared__ alignas(16) uint8_t sX[kBn][kBkB];

    // Per-thread fp32 output accumulator.
    float y_fp[kMsubPerWarp][kNsubPerCta][4];
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im)
        #pragma unroll
        for (int in = 0; in < kNsubPerCta; ++in)
            #pragma unroll
            for (int r = 0; r < 4; ++r) y_fp[im][in][r] = 0.0f;

    const int blk_start = hp_row_offsets[br];
    const int blk_end   = hp_row_offsets[br + 1];

    for (int blk = blk_start; blk < blk_end; ++blk) {
        const int bc = hp_col_indices[blk];   // K-group index

        // -------------------------------------------------------------------
        // Load W_high_blocks[blk, 0:128, 0:64] -> sW.
        //   Layout: (n_blocks, 128 rows, 64 bytes).  tid owns one row.
        // -------------------------------------------------------------------
        {
            const uint8_t* src = W_high_blocks
                + (int64_t)blk * kBm * kBkB
                + (int64_t)tid * kBkB;
            uint8_t* dst = &sW[tid][0];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint4 v = *reinterpret_cast<const uint4*>(src + i * 16);
                *reinterpret_cast<uint4*>(dst + i * 16) = v;
            }
        }

        // -------------------------------------------------------------------
        // Load X[n_tile:n_tile+32, group bc] -> sX (128 loads = 1/thread).
        // -------------------------------------------------------------------
        {
            int row  = tid >> 2;     // 0..31
            int quad = tid & 3;      // 0..3
            int n = n_tile + row;
            uint4 v;
            if (n < T) {
                int64_t off = (int64_t)n * (d_in / 2)
                            + (int64_t)bc * kBkB
                            + (int64_t)quad * 16;
                v = *reinterpret_cast<const uint4*>(X + off);
            } else {
                v = make_uint4(0, 0, 0, 0);
            }
            *reinterpret_cast<uint4*>(&sX[row][quad * 16]) = v;
        }

        __syncthreads();

        // -------------------------------------------------------------------
        // Inner K-loop: 2 MMA_K=64 steps on the 128-wide group.
        // -------------------------------------------------------------------
        int d_acc[kMsubPerWarp][kNsubPerCta][4];
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im)
            #pragma unroll
            for (int in = 0; in < kNsubPerCta; ++in)
                #pragma unroll
                for (int r = 0; r < 4; ++r) d_acc[im][in][r] = 0;

        #pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
            const int kpb_base = ks * 32;     // byte offset within row

            // A operand regs from sW.
            uint32_t a_regs[kMsubPerWarp][4];
            #pragma unroll
            for (int im = 0; im < kMsubPerWarp; ++im) {
                int msub_base = warp_id * 32 + im * 16;
                int row0 = msub_base + (lane >> 2);
                int row1 = row0 + 8;
                int col_low  = kpb_base + (lane & 3) * 4;
                int col_high = col_low + 16;
                uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sW[row0][col_low]);
                uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sW[row1][col_low]);
                uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sW[row0][col_high]);
                uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sW[row1][col_high]);
                a_regs[im][0] = a0;
                a_regs[im][1] = a1;
                a_regs[im][2] = a2;
                a_regs[im][3] = a3;
            }

            // B operand regs from sX.
            uint32_t b_regs[kNsubPerCta][2];
            #pragma unroll
            for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                int n_row = in_sub * 8 + (lane >> 2);
                int col_low  = kpb_base + (lane & 3) * 4;
                int col_high = col_low + 16;
                uint32_t b0 = 0, b1 = 0;
                if (n_row < kBn) {
                    b0 = *reinterpret_cast<const uint32_t*>(&sX[n_row][col_low]);
                    b1 = *reinterpret_cast<const uint32_t*>(&sX[n_row][col_high]);
                }
                b_regs[in_sub][0] = b0;
                b_regs[in_sub][1] = b1;
            }

            // Issue the MMA tile (kMsubPerWarp * kNsubPerCta) = 8 MMAs.
            #pragma unroll
            for (int im = 0; im < kMsubPerWarp; ++im) {
                #pragma unroll
                for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                    mma_m16n8k64_s4s4s32(
                        d_acc[im][in_sub][0], d_acc[im][in_sub][1],
                        d_acc[im][in_sub][2], d_acc[im][in_sub][3],
                        a_regs[im][0], a_regs[im][1],
                        a_regs[im][2], a_regs[im][3],
                        b_regs[in_sub][0], b_regs[in_sub][1],
                        d_acc[im][in_sub][0], d_acc[im][in_sub][1],
                        d_acc[im][in_sub][2], d_acc[im][in_sub][3]
                    );
                }
            }
        }

        // -------------------------------------------------------------------
        // Per-block fold:  y_fp += 16 * acc * scale_u4[m, bc]
        //   The 16x factor comes from the bit-level W_low/W_high split:
        //     q_s8 = (q_s8 & 0x0F)        (stored in W_low, as UINT4)
        //          + 16 * (q_s8 >> 4)     (stored in W_high, as SINT4)
        //   so the dequantized contribution of the W_high branch carries
        //   an implicit scale of 16.  This matches
        //   csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu
        //   (which has "16.0f * d_val * s" in its sparse epilogue).
        //   The standalone csrc/sparse_gemm/sparse_gemm_mma_int4.cu
        //   omits this factor and expects the caller to apply it,
        //   but in our 4-kernel naive pipeline there is no such caller,
        //   so we fold it in here.
        // -------------------------------------------------------------------
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
                    int n_local = nsub_base + col_local;
                    int m_global = m_tile + m_local;
                    int n_global = n_tile + n_local;
                    if (m_global >= d_out || n_global >= T) continue;

                    float s = __half2float(
                        scale_u4[(int64_t)m_global * n_groups + bc]);
                    y_fp[im][in_sub][r] += 16.0f *
                        static_cast<float>(d_acc[im][in_sub][r]) * s;
                }
            }
        }

        __syncthreads();
    }

    // =======================================================================
    // Final per-N scale_x multiply and writeback.
    // =======================================================================
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
                int n_global = n_tile + nsub_base + col_local;
                if (m_global >= d_out || n_global >= T) continue;

                float sxn = __half2float(scale_x[n_global]);
                Y[(int64_t)m_global * T + n_global] =
                    __float2half(y_fp[im][in_sub][r] * sxn);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher -- ABI identical to L0.
// ---------------------------------------------------------------------------
void launch(torch::Tensor W_high_blocks,
            torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
            torch::Tensor X_s4, torch::Tensor scale_u4, torch::Tensor scale_x,
            torch::Tensor Y_high,
            int d_out, int d_in)
{
    TORCH_CHECK(W_high_blocks.dtype() == torch::kInt8);
    TORCH_CHECK(hp_row_offsets.dtype() == torch::kInt32);
    TORCH_CHECK(hp_col_indices.dtype() == torch::kInt32);
    TORCH_CHECK(X_s4.dtype() == torch::kInt8);
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf);
    TORCH_CHECK(scale_x.dtype() == torch::kHalf);
    TORCH_CHECK(Y_high.dtype() == torch::kHalf);
    TORCH_CHECK(d_out % kBm == 0, "d_out must be divisible by 128");
    TORCH_CHECK(d_in  % kBk == 0, "d_in  must be divisible by 128");
    TORCH_CHECK(X_s4.stride(1) == 1, "X_s4 must be K-contiguous");

    const int T = X_s4.size(0);
    Y_high.zero_();
    if (W_high_blocks.size(0) == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 block(kBm, 1, 1);                       // 128 threads = 4 warps
    dim3 grid(d_out / kBm, (T + kBn - 1) / kBn, 1);

    sparse_gemm_naive_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W_high_blocks.data_ptr<int8_t>()),
        hp_row_offsets.data_ptr<int>(),
        hp_col_indices.data_ptr<int>(),
        reinterpret_cast<const uint8_t*>(X_s4.data_ptr<int8_t>()),
        reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Y_high.data_ptr<at::Half>()),
        d_out, d_in, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace sparse_gemm
}  // namespace hkust_v9_naive
