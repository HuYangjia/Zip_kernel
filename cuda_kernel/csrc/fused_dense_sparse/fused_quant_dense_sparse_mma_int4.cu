// Fused Activation Quantization + Dense+Sparse GEMM (T>1, SM89).
//
// P0 (plan docs/P0_QUANT_FUSION_SPIKE.md): folds activation_quant into
// the prologue of fused_dense_sparse_mma_int4.  Removes the ~16us
// launch-overhead floor that dominates small/mid-shape kernels.
//
// P0.2 (correctness-first):
//   - Supports kBm=128 only (common mid-shape case).
//   - Supports hp_nnz==0 (dense-only) first; sparse branch to be
//     ported from legacy kernel in P0.3 once parity is stable.
//   - No cp.async, no group-cache, no split-K.  These optimisations
//     layer on top of a correct baseline in P0.4+.
//   - Uses the legacy kernel's helpers (mma_m16n8k64_s4s4s32, ldmatrix)
//     and replicates the exact per-group dequant-fold math so parity
//     is bit-identical to the split-launch pipeline.
//
// CTA layout:
//   blockDim = (kBm=128, 1, 1)  — 4 warps per CTA.
//   gridDim  = (ceil(d_out/128), ceil(T/kBn=32), 1)
//
// Prologue algorithm (bit-exact with activation_quant):
//   Phase 1 — per-token max-abs:
//     For each of kBn tokens owned by this CTA, the 128 threads scan
//     X[t_global, perm[0..D)] in strides of 128 along D.  Warp-tree
//     reduce + 4-warp smem reduce → s_scale_x[n_local] (fp16 round-trip
//     to match the scale chain in activation_quant_kernel).
//   Phase 2 — per-group quantize + pack + sum (merged into main loop):
//     For each group g:
//       128 threads cooperatively read 128 fp16 cols of each of kBn
//       tokens, quantize to s4, pair-wise pack to bytes in sX[buf][...],
//       compute per-(token, group) sum reduction for s_sum_X[buf][...].
//     Then run_mma_pass(buf) as in the legacy kernel.

#include "common/arch.cuh"
#include "common/mma_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9 {
namespace fused_quant_dense_sparse_mma_int4 {

// -------------------------------------------------------------------
// Kernel tuning constants (match legacy kernel's taxonomy)
// -------------------------------------------------------------------
static_assert(BCOL == 128, "kernel assumes BCOL == 128");
static_assert(BROW == 128, "kernel assumes BROW == 128");
constexpr int kBm            = 128;            // fixed for P0.2
constexpr int kWarpCount     = kBm / kWarpSize;
constexpr int bytes_per_group = BCOL / 2;      // == 64

// -------------------------------------------------------------------
// Device helpers (same math as activation_quant.cu and fused_quant_gemv.cu)
// -------------------------------------------------------------------
__device__ __forceinline__ float warp_max_abs_f(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFF, v, off));
    return v;
}

__device__ __forceinline__ int warp_sum_i(int v) {
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    return v;
}

__device__ __forceinline__ int quantize_one(float x, float scale_safe,
                                             bool scale_is_zero) {
    // Must match activation_quant.cu::quantize_one bit-for-bit:
    //   uses __fdividef (div.approx.f32) to match Triton's fp32 div.
    if (scale_is_zero) return 0;
    float q = rintf(__fdividef(x, scale_safe));
    q = fmaxf(fminf(q, 7.0f), -8.0f);
    return static_cast<int>(q);
}

// -------------------------------------------------------------------
// MMA pass runner (copied from legacy kernel, specialised for kBm=128)
// -------------------------------------------------------------------
//
// Layout contract (m16n8k64.s4, as per mma_utils.cuh):
//   A = 16x64 s4 per sub-tile (row-major in sW[buf][row][col]).
//   B = 64x8  s4 per sub-tile (column-major; k along row of sX).
//   C = 16x8  int32.
//
// Each warp owns m-band [warp_id*32, (warp_id+1)*32); m-sub = 16 rows.
// Each CTA owns kBn cols; n-sub = 8 cols (kNsubPerCta = kBn / 8).
//
// A group contains BCOL=128 s4 = 64 bytes.  One mma.m16n8k64 covers
// k=64 s4 = 32 bytes.  So each group needs kKSteps = BCOL / 64 = 2
// MMA inner iterations (ks=0 covers bytes 0..31; ks=1 covers 32..63).

constexpr int kKSteps = BCOL / 64;  // == 2 for BCOL=128

template <int kBn>
__device__ __forceinline__ void run_mma_pass_kbm128(
    uint8_t sW[2][kBm][bytes_per_group],
    uint8_t sX[2][kBn][bytes_per_group],
    int warp_id, int lane,
    int buf,
    int d_acc[/*kMsubPerWarp*/2][/*kNsubPerCta*/(kBn + 7) / 8][4]
) {
    constexpr int kMsubPerWarp = 2;
    constexpr int kNsubPerCta  = (kBn + 7) / 8;

    #pragma unroll
    for (int ks = 0; ks < kKSteps; ++ks) {
        const int kpb_base = ks * 32;  // byte offset within the 64-byte group

        // ---- Load A (4 regs/thread per m-sub, no-ldmatrix path) ----
        uint32_t a_regs[kMsubPerWarp][4];
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im) {
            const int msub_base = warp_id * 32 + im * 16;
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

        // ---- Load B (2 regs/thread per n-sub) ----
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

        // ---- Issue mma.m16n8k64.s4.s4.s32 ----
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
}

// -------------------------------------------------------------------
// Main kernel (P0.2 — dense-only, no cp.async, no group-cache)
// -------------------------------------------------------------------
template <int kBn>
__global__ void fused_quant_dense_mma_int4_kernel(
    const __half* __restrict__ X,         // (T, D) fp16
    int64_t stride_x_t, int64_t stride_x_d,
    const int* __restrict__ perm,         // (D,)
    const uint8_t* __restrict__ W_low,    // (d_out, d_in/2) uint4 packed
    int64_t stride_w_m, int64_t stride_w_k,
    const __half* __restrict__ scale_u4,  // (d_out, n_groups)
    const __half* __restrict__ zero_u4,
    int64_t stride_su_m, int64_t stride_su_g,
    int64_t stride_zu_m, int64_t stride_zu_g,
    __half* __restrict__ Y,               // (d_out, T)
    int64_t stride_y_m, int64_t stride_y_n,
    int d_out, int d_in, int T, int n_groups
) {
    constexpr int kMsubPerWarp = 2;
    constexpr int kNsubPerCta  = (kBn + 7) / 8;

    const int tid     = threadIdx.x;
    const int lane    = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;

    const int m_tile = blockIdx.x * kBm;
    const int n_tile = blockIdx.y * kBn;

    // -----------------------------------------------------------------
    // Shared memory (layout matches legacy kernel for mainloop compat)
    // -----------------------------------------------------------------
    __shared__ alignas(16) uint8_t sW[2][kBm][bytes_per_group];
    __shared__ alignas(16) uint8_t sX[2][kBn][bytes_per_group];
    __shared__ __half  s_scale_x[kBn];
    __shared__ int     s_sum_X[2][kBn];
    // Phase-1 reduction staging (kBn tokens × kWarpCount=4 warps).
    __shared__ float   s_max_partial[kBn][kWarpCount];
    __shared__ float   s_scale_math[kBn];
    __shared__ uint8_t s_scale_zero[kBn];

    int y_int[kMsubPerWarp][kNsubPerCta][4];
    float y_fp[kMsubPerWarp][kNsubPerCta][4];
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im)
      #pragma unroll
      for (int in = 0; in < kNsubPerCta; ++in)
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
          y_int[im][in][r] = 0;
          y_fp [im][in][r] = 0.0f;
        }

    // =================================================================
    // PHASE 1 — per-token max-abs → s_scale_x[n_local]
    // =================================================================
    //
    // Thread remap for prologue: each 128-thread CTA covers
    //   (n_local ∈ [0, kBn), col_quad ∈ [0, 4))
    //     tid = n_local * 4 + col_quad      // 32 × 4 = 128
    //     col_quad's 32 cols = col_quad*32 .. col_quad*32+31
    //
    // Phase 1: thread walks its 32 cols per "k-chunk" of 128 cols,
    // slides across D in strides of 128.  Each n_local=tid/4 accumulates
    // max over its share; then 4-way shfl reduce (tid%4) across col_quad
    // merges into a single per-token max.
    //
    // The 4-way reduce lives inside one warp because (tid / 32) is
    // the warp id and within a warp, lanes {0..3, 4..7, ..., 28..31}
    // each form an independent 4-lane group owning (n_local, *).
    // -----------------------------------------------------------------
    const int qn_local  = tid >> 2;         // 0..31 (the n_local this thread serves)
    const int qcol_quad = tid & 3;          // 0..3 (which 32-col chunk of the 128 stride)
    const int n_global_for_q = n_tile + qn_local;
    const bool n_active_q    = (n_global_for_q < T);

    float local_max = 0.0f;
    // Slide through D with stride 128.  Each thread handles
    // 32 cols per 128-col chunk, starting at col_quad*32 + lane_in_chunk.
    // Total cols per thread = D / 4 (= D/128 chunks × 32 cols).
    for (int d_base = 0; d_base < d_in; d_base += BCOL) {
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            int d = d_base + qcol_quad * 32 + k;
            int pidx = __ldg(perm + d);
            __half h(0);
            if (n_active_q) {
                h = __ldg(X + (int64_t)n_global_for_q * stride_x_t
                            + (int64_t)pidx * stride_x_d);
            }
            float v = __half2float(h);
            local_max = fmaxf(local_max, fabsf(v));
        }
    }
    // 4-way reduce across qcol_quad using warp shuffles.
    //   threads (n_local*4+0 .. n_local*4+3) live in the same warp's
    //   4-lane group; XOR 1 then XOR 2 composes the max.
    {
        float o = __shfl_xor_sync(0xFFFFFFFF, local_max, 1);
        local_max = fmaxf(local_max, o);
        o = __shfl_xor_sync(0xFFFFFFFF, local_max, 2);
        local_max = fmaxf(local_max, o);
    }
    // Only lane with col_quad == 0 holds the final value.  Write to smem.
    if (qcol_quad == 0) {
        float scale_fp32 = local_max / 7.0f;
        __half scale_h   = __float2half(scale_fp32);
        float  scale_math = __half2float(scale_h);
        bool   iz = !(scale_math > 0.0f);
        s_scale_x   [qn_local] = scale_h;
        s_scale_math[qn_local] = iz ? 1.0f : scale_math;
        s_scale_zero[qn_local] = iz ? 1 : 0;
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // Preload scale_x into the kernel's s_scale_x (already done above).
    // -----------------------------------------------------------------

    // =================================================================
    // PHASE 2 — merged per-group quant + W prefetch + MMA
    // =================================================================
    //
    // Helpers for the W dense load (128 threads each handle 64 bytes
    // of row = one group's worth of packed uint4).
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

    // Quantize+pack+sum one group g into sX[buf][...] and s_sum_X[buf][...].
    //
    // Thread remap (same as Phase 1):
    //   qn_local  = tid >> 2  ∈ [0, 32)   — token this thread serves
    //   qcol_quad = tid & 3   ∈ [0, 4)    — 32-col chunk within the group
    //
    // Each thread handles 32 cols (k = 0..31) of one (token, group).
    // For packing: within a thread's 32 cols, even-k and (k+1) form a
    // nibble pair → thread packs its own byte (no cross-thread shfl needed!).
    //
    // For per-(token, group) sum: 4 threads (same qn_local, different
    // qcol_quad) need to be combined.  Their qcol_quad values are 0..3
    // and they live in the SAME warp (because tid/32 = qn_local/8 is
    // same for 8 consecutive qn_locals).  A 4-way shfl_xor with offs
    // {1, 2} does the reduce.
    auto quant_one_group = [&](int g, int buf) {
        const int n_global = n_tile + qn_local;
        const bool n_active = (n_global < T);
        const float scale_math = s_scale_math[qn_local];
        const bool  is_zero    = (s_scale_zero[qn_local] != 0);

        int thread_sum = 0;
        // Each thread processes 32 consecutive cols within its col_quad.
        // Pack pairs (k, k+1) into 1 byte at sX[buf][qn_local][col_quad*16 + k/2].
        #pragma unroll
        for (int k = 0; k < 32; k += 2) {
            int d_lo = g * BCOL + qcol_quad * 32 + k;
            int d_hi = d_lo + 1;
            int p_lo = __ldg(perm + d_lo);
            int p_hi = __ldg(perm + d_hi);
            __half h_lo(0), h_hi(0);
            if (n_active) {
                h_lo = __ldg(X + (int64_t)n_global * stride_x_t
                               + (int64_t)p_lo * stride_x_d);
                h_hi = __ldg(X + (int64_t)n_global * stride_x_t
                               + (int64_t)p_hi * stride_x_d);
            }
            int q_lo = quantize_one(__half2float(h_lo), scale_math, is_zero);
            int q_hi = quantize_one(__half2float(h_hi), scale_math, is_zero);
            thread_sum += q_lo + q_hi;

            if (n_active) {
                int packed = ((q_hi & 0x0F) << 4) | (q_lo & 0x0F);
                int8_t byte = static_cast<int8_t>(
                    packed >= 128 ? packed - 256 : packed);
                // sX layout: sX[buf][qn_local][byte_in_group],
                //   byte_in_group = qcol_quad*16 + (k/2).
                sX[buf][qn_local][qcol_quad * 16 + (k >> 1)] =
                    static_cast<uint8_t>(byte);
            }
        }

        // 4-way reduce across qcol_quad for per-(token, group) sum.
        thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 1);
        thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, 2);

        if (qcol_quad == 0 && n_active) {
            s_sum_X[buf][qn_local] = thread_sum;
        }
        __syncthreads();
    };

    // ---- Main K-loop with double-buffered W + sX ----
    //
    // Simpler than legacy: sync dequant of the quantization+W load per
    // group.  cp.async + double-buffer overlap is a P0.4 optimisation.
    //
    // Fold math: identical to legacy kernel's fold_dense:
    //   corrected = int_acc - zero[m,g] * sum_X[n,g]
    //   y_fp     += corrected * scale_u4[m,g]
    // scale_x is multiplied at the epilogue (once, per-n).

    for (int g = 0; g < n_groups; ++g) {
        const int buf = g & 1;

        // Load W for group g.
        issue_w_dense_load(g, buf);
        // Quantize X for group g into sX[buf] and s_sum_X[buf][...].
        quant_one_group(g, buf);

        // Run MMA (reset y_int to 0 first — it's a fresh per-group acc).
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im)
          #pragma unroll
          for (int in = 0; in < kNsubPerCta; ++in)
            #pragma unroll
            for (int r = 0; r < 4; ++r) y_int[im][in][r] = 0;

        run_mma_pass_kbm128<kBn>(sW, sX, warp_id, lane, buf, y_int);

        // Fold per-group dequant into y_fp.
        //   Thread layout of MMA output C (m16n8 int32, 4 regs/thread):
        //     row_local = (lane >> 2) + 8 * (r >> 1)
        //     col_local = (lane & 3) * 2 + (r & 1)
        //   r ∈ {0,1,2,3} covers 4 int32 accumulators per sub-tile.
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im) {
            const int m_sub_base = warp_id * 32 + im * 16;
            #pragma unroll
            for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                const int n_sub_base = in_sub * 8;
                #pragma unroll
                for (int r = 0; r < 4; ++r) {
                    int row_local = (lane >> 2) + ((r >> 1) ? 8 : 0);
                    int col_local = (lane & 3) * 2 + (r & 1);
                    int m_g = m_tile + m_sub_base + row_local;
                    int n_l = n_sub_base + col_local;

                    float z = 0.0f, s = 0.0f, sumxn = 0.0f;
                    if (m_g < d_out) {
                        z = __half2float(zero_u4 [(int64_t)m_g * stride_zu_m
                                                + (int64_t)g   * stride_zu_g]);
                        s = __half2float(scale_u4[(int64_t)m_g * stride_su_m
                                                + (int64_t)g   * stride_su_g]);
                    }
                    if (n_l < kBn) {
                        sumxn = static_cast<float>(s_sum_X[buf][n_l]);
                    }
                    float corrected = static_cast<float>(y_int[im][in_sub][r])
                                    - z * sumxn;
                    y_fp[im][in_sub][r] += corrected * s;
                }
            }
        }
        __syncthreads();
    }

    // =================================================================
    // EPILOGUE: multiply by scale_x and write to Y
    // =================================================================
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im) {
        const int m_sub_base = warp_id * 32 + im * 16;
        #pragma unroll
        for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
            const int n_sub_base = in_sub * 8;
            #pragma unroll
            for (int rpair = 0; rpair < 2; ++rpair) {
                int r0 = rpair * 2;
                int r1 = r0 + 1;
                int row_local = (lane >> 2) + ((r0 >> 1) ? 8 : 0);
                int col_local = (lane & 3) * 2;
                int m_g = m_tile + m_sub_base + row_local;
                int n_l0 = n_sub_base + col_local;
                int n_l1 = n_l0 + 1;
                if (m_g >= d_out) continue;
                if (n_l0 >= kBn) continue;
                int n_g0 = n_tile + n_l0;
                if (n_g0 >= T) continue;

                float sxn0 = __half2float(s_scale_x[n_l0]);
                float sxn1 = (n_l1 < kBn) ? __half2float(s_scale_x[n_l1]) : 0.0f;
                float v0 = y_fp[im][in_sub][r0] * sxn0;
                float v1 = y_fp[im][in_sub][r1] * sxn1;

                int64_t off0 = (int64_t)m_g * stride_y_m
                             + (int64_t)n_g0 * stride_y_n;
                Y[off0] = __float2half(v0);
                if ((n_l1 < kBn) && (n_g0 + 1 < T)) {
                    Y[off0 + stride_y_n] = __float2half(v1);
                }
            }
        }
    }
}

// -------------------------------------------------------------------
// Host launcher
// -------------------------------------------------------------------
void launch(
    torch::Tensor X_fp16, torch::Tensor perm,
    torch::Tensor W_low, torch::Tensor W_high_blocks,
    torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
    torch::Tensor scale_u4, torch::Tensor zero_u4,
    torch::Tensor Y_total,
    int d_out, int d_in
) {
    TORCH_CHECK(X_fp16.dtype()   == torch::kHalf);
    TORCH_CHECK(perm.dtype()     == torch::kInt32);
    TORCH_CHECK(W_low.dtype()    == torch::kInt8);
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf);
    TORCH_CHECK(zero_u4.dtype()  == torch::kHalf);
    TORCH_CHECK(Y_total.dtype()  == torch::kHalf);
    TORCH_CHECK(d_in % BCOL == 0);
    TORCH_CHECK(X_fp16.dim() == 2);
    const int T = X_fp16.size(0);
    TORCH_CHECK(X_fp16.size(1) == d_in);
    TORCH_CHECK(T >= 2, "T must be >= 2 for this path (use fused_quant_gemv for T=1)");
    TORCH_CHECK(d_out % kBm == 0,
                "d_out must be % kBm == 0 in P0.2 (got d_out=", d_out, ")");

    // P0.2 dense-only: sparse residual not yet ported.  Fail loudly so
    // callers (bench harness) correctly fall back to legacy path when
    // hp_nnz > 0.
    TORCH_CHECK(hp_col_indices.numel() == 0,
                "P0.2 does not support hp_nnz > 0; use legacy path for sparse.");

    const int n_groups = d_in / BCOL;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto do_launch = [&](auto kBnC) {
        constexpr int kBn = decltype(kBnC)::value;
        dim3 block(kBm, 1, 1);
        dim3 grid((d_out + kBm - 1) / kBm, (T + kBn - 1) / kBn, 1);
        fused_quant_dense_mma_int4_kernel<kBn><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
            X_fp16.stride(0), X_fp16.stride(1),
            perm.data_ptr<int>(),
            reinterpret_cast<const uint8_t*>(W_low.data_ptr<int8_t>()),
            W_low.stride(0), W_low.stride(1),
            reinterpret_cast<const __half*>(scale_u4.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(zero_u4.data_ptr<at::Half>()),
            scale_u4.stride(0), scale_u4.stride(1),
            zero_u4.stride(0),  zero_u4.stride(1),
            reinterpret_cast<__half*>(Y_total.data_ptr<at::Half>()),
            Y_total.stride(0), Y_total.stride(1),
            d_out, d_in, T, n_groups
        );
    };

    // P0.2: start with kBn=32 (covers T=32/64/128 well, matches
    // common legacy dispatch).  kBn=64 is a P0.4 option.
    do_launch(std::integral_constant<int, 32>{});

    C10_CUDA_CHECK(cudaGetLastError());

    // Silence unused (sparse tensors — kept on the ABI for future P0.3).
    (void)W_high_blocks; (void)hp_row_offsets;
}

}  // namespace fused_quant_dense_sparse_mma_int4
}  // namespace hkust_v9

