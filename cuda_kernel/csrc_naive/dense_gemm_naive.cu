// Naive UINT4 x SINT4 dense GEMM -- L1 ("Tensor-Core naive") baseline.
//
// Upgraded from the scalar-IMAD L0 version to exercise the INT4 Tensor
// Core (mma.m16n8k64.s4.s4.s32) so the naive vs optimised comparison
// isolates the cost of kernel fusion & pipelining, not the cost of
// not-using TC at all.
//
// Deliberately NAIVE at the MMA level:
//   * 4 kernels stay separate -- no fusion with quant / sparse / add.
//   * Single-buffered shmem (no cp.async, no double-buffer).
//   * No scale / zero prefetch into shmem (read from HBM each group).
//   * No kBn / kBm dispatcher (fixed kBm=128, kBn=32).
//   * No sum_X prefetch, no sxn caching (read from HBM / shmem per group).
//   * Plain __ldg-based LDG into shmem + __syncthreads + MMA; the
//     MMA operand registers are assembled by direct lane-indexed
//     32-bit reads from shmem (no ldmatrix).
//
// Semantic contract (bit-equivalent to L0 naive and to the optimised
// mma.int4 kernel, modulo fp32 accumulation order):
//   Y[m,n] = scale_x[n] * sum_g [ ( acc_g - z[m,g] * sum_X[n,g] ) * s[m,g] ]
// where acc_g is the INT4 Tensor-Core dot-product over one 128-wide
// quant group.  W_low is packed as uint4 (low=col 2k, high=col 2k+1)
// but fed to mma.s4.s4 directly -- the zero-point absorbs the implicit
// +8 offset (same convention as csrc/dense_gemm/dense_gemm_mma_int4.cu).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9_naive {
namespace dense_gemm {

// ---------------------------------------------------------------------------
// Inlined MMA intrinsic (copy of csrc/common/mma_utils.cuh to avoid
// cross-extension header dependencies -- naive source tree is
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
// kBm = 128 rows per CTA, kBn = 32 cols per CTA, kBk = 128 (= one quant
// group).  128 threads per CTA = 4 warps; each warp handles 2 MMA_M=16
// row-subtiles (covers 32 rows); across 4 warps that covers kBm=128.
// Each warp also iterates kNsubPerCta = 4 MMA_N=8 column-subtiles to
// cover kBn=32.  kKSteps = 2 because one MMA eats 64 cols of K and a
// group is 128 cols wide.
// ---------------------------------------------------------------------------
static constexpr int kBm = 128;
static constexpr int kBn = 32;
static constexpr int kBk = 128;             // one quant group
static constexpr int kBkB = kBk / 2;        // 64 packed bytes
static constexpr int kMmaK = 64;
static constexpr int kKSteps = kBk / kMmaK; // 2
static constexpr int kMsubPerWarp = 2;      // 2 * 16 = 32 rows per warp
static constexpr int kNsubPerCta  = kBn / 8;

// Grid  : (ceil(d_out/kBm), ceil(T/kBn), 1)
// Block : (kBm, 1, 1)   == 128 threads  (4 warps)
__global__ void dense_gemm_naive_kernel(
    const uint8_t* __restrict__ W,         // (d_out, d_in/2)   U4
    const uint8_t* __restrict__ X,         // (T,     d_in/2)   S4
    const __half*  __restrict__ scale_u4,  // (d_out, n_groups) fp16
    const __half*  __restrict__ zero_u4,   // (d_out, n_groups) fp16
    const int*     __restrict__ sum_X,     // (T,     n_groups) int32
    const __half*  __restrict__ scale_x,   // (T,)              fp16
    __half*        __restrict__ Y,         // (d_out, T)        fp16
    int d_out, int d_in, int T
) {
    const int n_groups = d_in / kBk;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;        // 0..3
    const int lane    = tid & 31;        // 0..31

    const int m_tile = blockIdx.x * kBm;
    const int n_tile = blockIdx.y * kBn;

    // --- shmem tiles (single-buffered on purpose) ---
    //   sW : 128 rows * 64 bytes = 8 KiB
    //   sX :  32 rows * 64 bytes = 2 KiB
    //   Total ~10 KiB per CTA, well under the 48 KiB static budget.
    __shared__ alignas(16) uint8_t sW[kBm][kBkB];
    __shared__ alignas(16) uint8_t sX[kBn][kBkB];

    // Per-thread fp32 output accumulator: (kMsubPerWarp, kNsubPerCta, 4).
    float y_fp[kMsubPerWarp][kNsubPerCta][4];
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im)
        #pragma unroll
        for (int in = 0; in < kNsubPerCta; ++in)
            #pragma unroll
            for (int r = 0; r < 4; ++r) y_fp[im][in][r] = 0.0f;

    // =======================================================================
    // Loop over K-groups.
    // =======================================================================
    for (int g = 0; g < n_groups; ++g) {

        // -------------------------------------------------------------------
        // Load W[m_tile:m_tile+128, g] -> sW.
        //   Each of the 128 threads owns exactly one row (tid == row).
        //   Each row = 64 bytes = 4 * uint4 (16B).  4 LDG per thread.
        // -------------------------------------------------------------------
        {
            int m = m_tile + tid;
            uint8_t* dst = &sW[tid][0];
            if (m < d_out) {
                const uint8_t* src = W
                    + (int64_t)m * (d_in / 2)
                    + (int64_t)g * kBkB;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint4 v = *reinterpret_cast<const uint4*>(src + i * 16);
                    *reinterpret_cast<uint4*>(dst + i * 16) = v;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *reinterpret_cast<uint4*>(dst + i * 16)
                        = make_uint4(0, 0, 0, 0);
                }
            }
        }

        // -------------------------------------------------------------------
        // Load X[n_tile:n_tile+32, g] -> sX.
        //   32 rows * 4 (uint4)s per row = 128 loads total; 1 per thread.
        // -------------------------------------------------------------------
        {
            int row  = tid >> 2;            // 0..31
            int quad = tid & 3;             // 0..3
            int n = n_tile + row;
            uint4 v;
            if (n < T) {
                int64_t off = (int64_t)n * (d_in / 2)
                            + (int64_t)g * kBkB
                            + (int64_t)quad * 16;
                v = *reinterpret_cast<const uint4*>(X + off);
            } else {
                v = make_uint4(0, 0, 0, 0);
            }
            *reinterpret_cast<uint4*>(&sX[row][quad * 16]) = v;
        }

        __syncthreads();

        // -------------------------------------------------------------------
        // Inner K-loop: 2 MMA_K=64 steps.
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
            // Byte offset into each row of sW / sX for this MMA_K slice.
            //   one mma.m16n8k64.s4 consumes 64 s4 values = 32 bytes per row,
            //   so kpb_base (bytes) = ks * 32.
            const int kpb_base = ks * 32;

            // ----- Assemble A operand regs (W tile) -----
            //   A layout per-thread for mma.m16n8k64.row.col:
            //     4 regs per thread, each reg = 8 s4 values (32 bits).
            //     Row indices:  row0 = (lane>>2)          row1 = row0 + 8
            //     Col-byte offs: col_low = kpb_base + (lane&3)*4
            //                    col_high = col_low + 16
            //     Regs: a0 = sW[row0][col_low],  a1 = sW[row1][col_low],
            //           a2 = sW[row0][col_high], a3 = sW[row1][col_high].
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
                    a0 = *reinterpret_cast<const uint32_t*>(&sW[row0][col_low]);
                    a2 = *reinterpret_cast<const uint32_t*>(&sW[row0][col_high]);
                }
                if (row1 < kBm) {
                    a1 = *reinterpret_cast<const uint32_t*>(&sW[row1][col_low]);
                    a3 = *reinterpret_cast<const uint32_t*>(&sW[row1][col_high]);
                }
                a_regs[im][0] = a0;
                a_regs[im][1] = a1;
                a_regs[im][2] = a2;
                a_regs[im][3] = a3;
            }

            // ----- Assemble B operand regs (X tile) -----
            //   B layout: 2 regs per thread, col-major.
            //     n_row = in_sub * 8 + (lane>>2)
            //     b0 = sX[n_row][col_low]
            //     b1 = sX[n_row][col_high]
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

            // ----- Issue MMAs -----
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
        // Per-group fold:
        //   y_fp += ( acc - z[m,g] * sum_X[n,g] ) * s[m,g]
        // Everything straight from HBM -- no shmem cache (keeps L1 naive).
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

                    float z  = __half2float(
                        zero_u4 [(int64_t)m_global * n_groups + g]);
                    float s  = __half2float(
                        scale_u4[(int64_t)m_global * n_groups + g]);
                    float sx = static_cast<float>(
                        sum_X[(int64_t)n_global * n_groups + g]);

                    float corrected = static_cast<float>(d_acc[im][in_sub][r])
                                    - z * sx;
                    y_fp[im][in_sub][r] += corrected * s;
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
// Host launcher -- ABI identical to the original L0 launcher.
// ---------------------------------------------------------------------------
void launch(
    torch::Tensor W_low, torch::Tensor X_s4,
    torch::Tensor scale_u4, torch::Tensor zero_u4,
    torch::Tensor sum_X, torch::Tensor scale_x,
    torch::Tensor Y_low
) {
    TORCH_CHECK(W_low.is_cuda() && W_low.dtype() == torch::kInt8);
    TORCH_CHECK(X_s4.is_cuda() && X_s4.dtype() == torch::kInt8);
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf);
    TORCH_CHECK(zero_u4.dtype() == torch::kHalf);
    TORCH_CHECK(sum_X.dtype() == torch::kInt32);
    TORCH_CHECK(scale_x.dtype() == torch::kHalf);
    TORCH_CHECK(Y_low.dtype() == torch::kHalf);
    TORCH_CHECK(W_low.stride(1) == 1, "W_low must be K-contiguous");
    TORCH_CHECK(X_s4.stride(1) == 1,  "X_s4 must be K-contiguous");

    int d_out   = W_low.size(0);
    int d_in_2  = W_low.size(1);
    int d_in    = d_in_2 * 2;
    int T       = X_s4.size(0);
    int n_groups = d_in / kBk;

    TORCH_CHECK(X_s4.size(1) == d_in_2, "X_s4 d_in mismatch");
    TORCH_CHECK(scale_u4.size(0) == d_out && scale_u4.size(1) == n_groups);
    TORCH_CHECK(zero_u4.size(0) == d_out && zero_u4.size(1) == n_groups);
    TORCH_CHECK(sum_X.size(0) == T && sum_X.size(1) == n_groups);
    TORCH_CHECK(scale_x.size(0) == T);
    TORCH_CHECK(Y_low.size(0) == d_out && Y_low.size(1) == T);
    TORCH_CHECK(d_in % kBk == 0, "d_in must be divisible by 128");

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dim3 block(kBm, 1, 1);                     // 128 threads = 4 warps
    dim3 grid((d_out + kBm - 1) / kBm,
              (T     + kBn - 1) / kBn, 1);

    dense_gemm_naive_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(W_low.data_ptr<int8_t>()),
        reinterpret_cast<const uint8_t*>(X_s4.data_ptr<int8_t>()),
        reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
        sum_X.data_ptr<int>(),
        reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Y_low.data_ptr<at::Half>()),
        d_out, d_in, T
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace dense_gemm
}  // namespace hkust_v9_naive
