// Dense UINT4 x SINT4 GEMM -- INT4 Tensor Core version (SM89).
//
// Round 12 optimisations:
//   - kBn capped at 64 (eliminates 255-reg spill at kBn=128).
//   - scale_u4 / zero_u4 prefetched to shmem: 128 rows x n_groups fp16
//     each, loaded once at CTA entry. Epilogue reads shmem instead of
//     HBM, removing n_groups HBM transactions per output element.
//   - Double-buffered W/X kept (total shmem for kBn=64: 16+8+8+8+~=40KB
//     which fits SM89's 48 KB static-shmem budget).
//
// Semantic contract: bit-exact match with dense_gemm Triton reference.
// Uses mma.m16n8k64.s4.s4.s32 (Tensor Core, accelerated on Ada SM89).

#include "common/arch.cuh"
#include "common/mma_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace dense_gemm_mma_int4 {

// Hard upper bound on n_groups we will ever see here: d_in <= 16384
// covers all current shapes (Qwen3 / Llama 7B/13B).  n_groups = d_in/128
// so kMaxGroups = 128.  Scale/zero shmem uses ceil_div path-wise max.
constexpr int kMaxGroups = 128;

template <int kBn>
__global__ void dense_gemm_mma_int4_kernel(
    const uint8_t* __restrict__ W,         // (d_out, d_in/2)
    const uint8_t* __restrict__ X,         // (T, d_in/2)
    const __half* __restrict__ scale_u4,   // (d_out, n_groups)
    const __half* __restrict__ zero_u4,    // (d_out, n_groups)
    const int* __restrict__ sum_X,         // (T, n_groups)
    const __half* __restrict__ scale_x,    // (T,)
    __half* __restrict__ Y,                // (d_out, T)
    int d_out, int d_in, int T,
    int n_groups,
    int64_t stride_w_m,   int64_t stride_w_k,
    int64_t stride_x_n,   int64_t stride_x_k,
    int64_t stride_su_m,  int64_t stride_su_g,
    int64_t stride_zu_m,  int64_t stride_zu_g,
    int64_t stride_sx_n,  int64_t stride_sx_g,
    int64_t stride_y_m,   int64_t stride_y_n
) {
    constexpr int kBm = 128;
    constexpr int kBk = 128;                 // one group
    constexpr int kMmaK = 64;
    constexpr int kKSteps = kBk / kMmaK;     // 2
    constexpr int kMsubPerWarp = 2;
    constexpr int kNsubPerCta = (kBn + 7) / 8;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int m_tile = blockIdx.x * kBm;
    const int n_tile = blockIdx.y * kBn;

    const int bytes_per_group = BCOL >> 1;   // 64

    // Shared memory layout
    //   sW           : (2, kBm, bytes_per_group) packed uint8 -- 2*8KB = 16KB
    //   sX           : (2, kBn, bytes_per_group) packed uint8 -- kBn=64 -> 8KB
    //   s_scale_u4   : (kBm, kMaxGroups) fp16                   -- 128*128*2 = 32KB ❌
    //
    // 32KB for scale alone would break the budget.  In practice n_groups
    // for Qwen3-like shapes is 32 (d_in=4096) or 86 (d_in=11008).  We
    // therefore allocate exactly (kBm, n_groups_ub) where n_groups_ub is
    // chosen at kernel launch time via a small dispatcher: when
    // n_groups <= 32 we use the ``kGrpBuf=32`` specialisation; when
    // n_groups <= 64 we use ``kGrpBuf=64``; otherwise we fall back to a
    // non-cached kernel (HBM epilogue).
    //
    // For this first implementation we conservatively allocate 32
    // groups (= 8 KB scale + 8 KB zero); callers with n_groups > 32
    // use a separate launch path that omits the prefetch.
    __shared__ alignas(16) uint8_t sW[2][kBm][bytes_per_group];
    __shared__ alignas(16) uint8_t sX[2][kBn][bytes_per_group];
    __shared__ alignas(16) __half s_scale_u4[kBm][32];
    __shared__ alignas(16) __half s_zero_u4 [kBm][32];
    __shared__ __half s_scale_x[kBn];
    __shared__ int s_sum_X[2][kBn];

    const bool cache_sz = (n_groups <= 32);

    if (tid < kBn) {
        int n = n_tile + tid;
        s_scale_x[tid] = (n < T) ? scale_x[n] : __half(0);
    }

    // Prefetch scale_u4 / zero_u4 into shmem.
    //   Each (m, g) is one fp16.  Total entries = kBm * n_groups (<=32).
    //   128 threads * 32 groups = 4096 entries worst case.
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

    // Per-thread FP32 accumulator for writeback.
    float y_fp[kMsubPerWarp][kNsubPerCta][4];
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im)
      #pragma unroll
      for (int in = 0; in < kNsubPerCta; ++in)
        #pragma unroll
        for (int r = 0; r < 4; ++r) y_fp[im][in][r] = 0.0f;

    // Round 23: pre-convert scale_x (fp16) to fp32 once per CTA.
    //   scale_x depends only on n_local, which is invariant across the
    //   g-loop.  Previously we paid 1 __half2float per (g, im, in_sub, r)
    //   iteration = n_groups * kMsubPerWarp * kNsubPerCta * 4 calls per
    //   thread.  Now we pay 2 * kNsubPerCta per thread, lifted outside
    //   the g loop.  Each thread owns (r & 1) ∈ {0, 1} columns, i.e.
    //   indices n_local = in_sub*8 + (r&1) for in_sub in [0, kNsubPerCta).
    //
    //   Register budget: kNsubPerCta * 2 floats per thread. For kBn=64
    //   this is 8*2=16 floats = 64B (trivial).
    //
    //   Also cache sum_X base pointer here, but sum_X itself still needs
    //   per-g access; leave it as shmem load in the loop.

    // ------------------------------------------------------------
    // Loaders (packed bytes, no unpack needed for INT4 MMA).
    // ------------------------------------------------------------
    auto issue_w_load = [&](int g, int buf) {
        int m = m_tile + tid;
        uint8_t* dst = &sW[buf][tid][0];
        if (m < d_out) {
            const uint8_t* src = W + (int64_t)m * stride_w_m
                                   + (int64_t)(g * bytes_per_group) * stride_w_k;
            // 64 bytes = 4 uint4
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

    auto issue_x_load = [&](int g, int buf) {
        const int total_quads = kBn * 4;
        for (int q = tid; q < total_quads; q += kBm) {
            int row = q >> 2;
            int quad = q & 3;
            int n = n_tile + row;
            uint4 v;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(g * bytes_per_group + quad * 16) * stride_x_k;
                v = *reinterpret_cast<const uint4*>(X + off);
            } else {
                v = make_uint4(0, 0, 0, 0);
            }
            *reinterpret_cast<uint4*>(&sX[buf][row][quad * 16]) = v;
        }
        for (int nk = tid; nk < kBn; nk += kBm) {
            int n = n_tile + nk;
            s_sum_X[buf][nk] = (n < T) ?
                sum_X[(int64_t)n * stride_sx_n + (int64_t)g * stride_sx_g] : 0;
        }
    };

    __syncthreads();
    issue_w_load(0, 0);
    issue_x_load(0, 0);
    __syncthreads();

    // Round 23: pre-convert s_scale_x[n_local] (fp16 -> fp32) once per CTA.
    //   Each thread owns 2 * kNsubPerCta slots indexed by:
    //     col_local = r & 1 (in {0, 1})
    //     n_local   = in_sub * 8 + col_local
    //   OOB handled as 0.0f (s_scale_x was zero-filled for n >= T above).
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
            issue_w_load(g + 1, buf ^ 1);
            issue_x_load(g + 1, buf ^ 1);
        }

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

        // Per-group fold to y_fp.
        //   Round 22: hoist (z, s) to per-row load.
        //   Round 24: also hoist sumxn to per-thread register array
        //   (depends on (in_sub, r&1) only, invariant over im loop).
        //
        //   sumxn_cache[in_sub][r&1] = (float)s_sum_X[buf][n_local]
        //
        //   Was: fetched once per (im, in_sub, r) = 2 * kNsubPerCta * 4
        //        = 16-64 fetches per group.
        //   Now: fetched once per (in_sub, r&1)    = 2 * kNsubPerCta
        //        = 4-16 fetches per group.  Halved.
        float sumxn_cache[kNsubPerCta][2];
        #pragma unroll
        for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
            int nsub_base = in_sub * 8;
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                int col_local = (lane & 3) * 2 + cc;
                int n_local = nsub_base + col_local;
                if (n_local < kBn) {
                    sumxn_cache[in_sub][cc] = static_cast<float>(
                        s_sum_X[buf][n_local]);
                } else {
                    sumxn_cache[in_sub][cc] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im) {
            int msub_base = warp_id * 32 + im * 16;
            // Pre-load (z, s) for the two m-rows this thread will touch.
            //   Row0: msub_base + (lane>>2)
            //   Row1: msub_base + (lane>>2) + 8
            int mrow0 = msub_base + (lane >> 2);
            int mrow1 = mrow0 + 8;
            float z0 = 0.0f, s0 = 0.0f, z1 = 0.0f, s1 = 0.0f;
            if (cache_sz) {
                if (mrow0 < kBm) {
                    z0 = __half2float(s_zero_u4 [mrow0][g]);
                    s0 = __half2float(s_scale_u4[mrow0][g]);
                }
                if (mrow1 < kBm) {
                    z1 = __half2float(s_zero_u4 [mrow1][g]);
                    s1 = __half2float(s_scale_u4[mrow1][g]);
                }
            } else {
                int m_g0 = m_tile + mrow0;
                int m_g1 = m_tile + mrow1;
                if (m_g0 < d_out) {
                    z0 = __half2float(zero_u4 [(int64_t)m_g0 * stride_zu_m + (int64_t)g * stride_zu_g]);
                    s0 = __half2float(scale_u4[(int64_t)m_g0 * stride_su_m + (int64_t)g * stride_su_g]);
                }
                if (m_g1 < d_out) {
                    z1 = __half2float(zero_u4 [(int64_t)m_g1 * stride_zu_m + (int64_t)g * stride_zu_g]);
                    s1 = __half2float(scale_u4[(int64_t)m_g1 * stride_su_m + (int64_t)g * stride_su_g]);
                }
            }

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
                    // Select the pre-loaded (z, s) for this m-row.
                    float z = (r >> 1) ? z1 : z0;
                    float s = (r >> 1) ? s1 : s0;
                    // Round 23: sxn from per-thread register cache.
                    float sxn = sxn_cache[in_sub][r & 1];
                    // Round 24: sumxn from per-thread register cache.
                    float sumxn = sumxn_cache[in_sub][r & 1];
                    float corrected = static_cast<float>(d_acc[im][in_sub][r])
                                    - z * sumxn;
                    y_fp[im][in_sub][r] += corrected * s * sxn;
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

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

void launch(
    torch::Tensor W_low, torch::Tensor X_s4,
    torch::Tensor scale_u4, torch::Tensor zero_u4,
    torch::Tensor sum_X, torch::Tensor scale_x,
    torch::Tensor Y_low
) {
    TORCH_CHECK(W_low.dtype() == torch::kInt8, "W_low must be int8");
    TORCH_CHECK(X_s4.dtype() == torch::kInt8, "X_s4 must be int8");
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf, "scale_u4 must be fp16");
    TORCH_CHECK(zero_u4.dtype() == torch::kHalf, "zero_u4 must be fp16");
    TORCH_CHECK(sum_X.dtype() == torch::kInt32, "sum_X must be int32");
    TORCH_CHECK(scale_x.dtype() == torch::kHalf, "scale_x must be fp16");
    TORCH_CHECK(Y_low.dtype() == torch::kHalf, "Y_low must be fp16");
    TORCH_CHECK(W_low.stride(1) == 1, "W_low must be K-contiguous");
    TORCH_CHECK(X_s4.stride(1) == 1, "X_s4 must be K-contiguous");
    TORCH_CHECK(scale_x.stride(0) == 1, "scale_x must be contiguous");

    const int d_out = W_low.size(0);
    const int d_in_half = W_low.size(1);
    const int d_in = d_in_half * 2;
    const int T = X_s4.size(0);
    TORCH_CHECK(X_s4.size(1) == d_in_half, "X_s4/W_low d_in mismatch");
    TORCH_CHECK(d_in % BCOL == 0, "d_in must be divisible by BCOL=128");
    const int n_groups = d_in / BCOL;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int kBm = 128;

    auto do_launch = [&](auto kBn_c) {
        constexpr int kBn = decltype(kBn_c)::value;
        dim3 block(kBm, 1, 1);
        dim3 grid(ceil_div(d_out, kBm), ceil_div(T, kBn), 1);
        dense_gemm_mma_int4_kernel<kBn><<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(W_low.data_ptr<int8_t>()),
            reinterpret_cast<const uint8_t*>(X_s4.data_ptr<int8_t>()),
            reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
            sum_X.data_ptr<int>(),
            reinterpret_cast<const __half*>(scale_x.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(Y_low.data_ptr<at::Half>()),
            d_out, d_in, T, n_groups,
            W_low.stride(0), W_low.stride(1),
            X_s4.stride(0), X_s4.stride(1),
            scale_u4.stride(0), scale_u4.stride(1),
            zero_u4.stride(0), zero_u4.stride(1),
            sum_X.stride(0), sum_X.stride(1),
            Y_low.stride(0), Y_low.stride(1)
        );
    };

    // Round 18: T=128 also benefits from kBn=32 dispatch.
    //   Profile 4k->4k (bench_20260424_18*):
    //     T=128 kBn=64: 110us  (fixed cost of 2 T-tiles * kBn=64 MMAs)
    //     T=128 kBn=32: 4 T-tiles * kBn=32 MMAs, grid=(32, 4)=128 CTAs
    //                   -> fits a single wave on SM89 (128 SMs).
    //   Extending kBn=32 bucket to T<=128 is expected to help T in
    //   (96, 128].
    if      (T <= 8)    do_launch(std::integral_constant<int, 8>{});
    else if (T <= 128)  do_launch(std::integral_constant<int, 32>{});
    else                do_launch(std::integral_constant<int, 64>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dense_gemm_mma_int4
}  // namespace hkust_v9
