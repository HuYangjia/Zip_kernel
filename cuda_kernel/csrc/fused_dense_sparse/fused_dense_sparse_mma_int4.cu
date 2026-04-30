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
#include <cstdlib>
#include <type_traits>

namespace hkust_v9 {
namespace fused_dense_sparse_mma_int4 {

// Stage I: Split-K reduce kernel.
// Sums split_k fp32 partial buffers and writes fp16 to Y.
// Y_partial: [split_k, d_out, T] fp32
// Y:         [d_out, T] fp16
__global__ void split_k_reduce_kernel(
    const float* __restrict__ Y_partial,
    __half* __restrict__ Y,
    int d_out, int T, int split_k
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_out * T) return;
    float acc = 0.0f;
    const int stride = d_out * T;
    for (int sk = 0; sk < split_k; ++sk) {
        acc += Y_partial[sk * stride + idx];
    }
    Y[idx] = __float2half(acc);
}

// Round 38: keep the bank-friendly 32-group row stride in shared memory,
// but allow n_groups up to 64 by reloading scale/zero in 32-group windows.
// This specifically targets Qwen3-14B hidden=5120 shapes (n_groups=40)
// without reintroducing the regression seen from a monolithic 40-group
// static cache.
constexpr int kGrpBuf = 32;
constexpr int kMaxWindowedGroups = 64;

// Stage C.1b — XOR swizzle for sW / sX to mitigate bank conflicts on
// ldmatrix A/B loads.
//
// Layout per row: 64 bytes (= kBk/2 int4 bytes), stride = 64B unchanged.
// Each ldmatrix.x4 expects 8 base addresses (one per lane in a group of
// 8), one 16-byte fragment per address.  Each base address must be
// **16-byte aligned** (PTX ldmatrix requirement).  Without swizzle, the
// 8 addresses lie at row_i * 64 (+col), landing on 2 distinct bank sets
// → 16-way conflict.
//
// Swizzle constraint: the XOR mask may only toggle bits [5:4] (the bits
// above bit 3), because toggling bit 3 would violate ldmatrix's 16-byte
// alignment.  That gives us only 1 usable bit (bit 4, since bit 5 is
// already used by the row stride of 64 bytes).  The best we can do with
// a 64B row stride is 2-way conflict reduction (16-way → 4-way).
//
// XOR = (row & 1) * 16   — toggles bit 4 only.
//   row 0: offset 0,   bank 0
//   row 1: offset 80,  bank 20
//   row 2: offset 128, bank 0
//   row 3: offset 208, bank 20
//   ... 2 distinct bank sets across 8 rows → 4-way conflict (was 16-way).
//
// Write alignment: the swizzle preserves 16-byte alignment because
// ((row & 1) * 16) has bit 3 = 0.  So uint4 writes remain valid —
// no need to degrade to uint2 granularity.
__device__ __forceinline__ int swizzle_row_col(int row, int col_bytes) {
    return col_bytes ^ ((row & 1) << 4);
}

// Round 41-P1: kBm templated (default = BROW = 128).
//   kBm=64 is opt-in via a narrow gate in the host launcher aimed at the
//   wave-starvation regime (T in [16,64] && d_out<=2048 && hp==0).  This
//   mirrors dense_gemm R40-B but for the fused kernel's dense branch.
//   Sparse branch requires BROW=128 for BSR block packing, so the opt-in
//   is STRICTLY gated on hp_row_offsets having no sparse blocks; if any
//   block is present we unconditionally use kBm=128.
// Stage C (r61) — kUseLdmatrix: replace manual A/B smem loads with
//   ldmatrix.sync.aligned.m8n8.x{4,2}.shared.b16 PTX instructions.
//   Reduces instruction count on the MMA A/B load path (32 manual loads
//   per ks-step → 2 ldmatrix instructions per ks-step), and allows HW to
//   coalesce smem accesses more efficiently.  Without XOR swizzle the
//   4-way bank conflict is not fully eliminated, but ldmatrix already
//   reduces the effective conflict because the HW combines the 8 lane
//   base addresses into a single wavefront request.
template <int kBn, bool kUseGroupCache, int kBm = BROW, bool kUseCpAsync = false,
          bool kUseLdmatrix = false>
__global__ void
// Stage D (REVERTED r55) — `__launch_bounds__(kBm, 3)` hint tried but
// rejected: mixed result on sweep (2048x2048 +2%, 1024x1024 +4%,
// 512x512 +11%, but 4096x4096 regressed -5%). The hint forces NVCC
// to assume 3 blocks/SM, which hurts register-hungry shapes with
// kBn=32 kBm=128 (167 regs * 128 = 21.3K, barely fits 3 blocks).
// Leaving the attribute *disabled* by commenting it out; see
// VALIDATION_LOG r55 for details.
// __launch_bounds__(kBm, 3)
fused_dense_sparse_mma_int4_kernel(
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
    float* __restrict__ Y_partial,   // Stage I: Split-K fp32 partial buffer (nullptr if split_k==1)
    int d_out, int d_in, int T,
    int n_groups,
    int split_k,                     // Stage I: number of K-splits (1 = no split)
    int64_t stride_w_m,   int64_t stride_w_k,
    int64_t stride_x_n,   int64_t stride_x_k,
    int64_t stride_su_m,  int64_t stride_su_g,
    int64_t stride_zu_m,  int64_t stride_zu_g,
    int64_t stride_sx_n,  int64_t stride_sx_g,
    int64_t stride_wb_blk, int64_t stride_wb_r, int64_t stride_wb_k,
    int64_t stride_y_m,   int64_t stride_y_n
) {
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
    const int n_cta_m = (d_out + kBm - 1) / kBm;

    // Stage I: Split-K. blockIdx.z is the split index (0..split_k-1).
    // Each CTA processes groups [g_start, g_end).
    const int split_k_idx = blockIdx.z;
    const int ng_per_split = (n_groups + split_k - 1) / split_k;
    const int g_start = split_k_idx * ng_per_split;
    const int g_end   = min(g_start + ng_per_split, n_groups);
    // When split_k==1, g_start=0, g_end=n_groups (no change).

    // R42-P1: when kBm<BROW two consecutive CTAs cover one BSR row.
    //   Each CTA consumes only the upper-half (br&1==0) or lower-half
    //   (br&1==1) of a 128-row BSR block.  For kBm==BROW both compile
    //   down to identity so there is zero cost on the legacy path.
    constexpr int kBsrPerCta = BROW / kBm;  // 1 when kBm==128, 2 when kBm==64
    const int bsr_br = br / kBsrPerCta;
    const int half_row_off = (br & (kBsrPerCta - 1)) * kBm;  // 0 or 64

    // Stage G (r58) REVERTED — sW/sX smem bank-conflict padding.
    // Stage H (r59) REVERTED — cp_async_ca_4 + 4-byte padding.
    //
    // Both attempts failed:
    //   r58: 36-byte stride not 16-byte aligned → cp.async.cg misaligned trap
    //   r59: cp.async.ca 4-byte granularity causes parity failures (rel>0.4)
    //        for ng>=16 shapes. Root cause: cp.async.ca.shared.global with
    //        4-byte size requires the global src to be 4-byte aligned AND
    //        the shared dst to be 4-byte aligned. While both conditions are
    //        met, the parity failure suggests a subtle issue with the
    //        cp.async.ca L1 cache coherence model vs cp.async.cg.
    //
    // Resolution: revert to 32-byte stride. The 4-way bank conflict on
    // sW/sX MMA loads remains a known limitation.
    //
    // Correct fix path: Stage C (ldmatrix for A/B loads). ldmatrix uses
    // a different smem addressing model that allows XOR-swizzled layouts,
    // which can be made conflict-free without alignment constraints.
    __shared__ alignas(16) uint8_t sW[2][kBm][bytes_per_group];
    __shared__ alignas(16) uint8_t sX[2][kBn][bytes_per_group];
    // Stage F (r57) — smem bank-conflict fix for scale/zero cache.
    //
    // s_scale_u4[kBm][kGrpBuf] is __half (2 bytes/element).  With
    // kGrpBuf=32 each row is 64 bytes = 16 bank slots (4 bytes/bank).
    // Rows 0 and 2 map to the same bank set (stride 64 = 16 banks, and
    // 32 banks total → period 2), causing a 4-way bank conflict on every
    // scale/zero read in the MMA inner loop.
    //
    // Fix: pad each row by 1 fp16 (2 bytes) so the row stride becomes
    // 33 fp16 = 66 bytes.  66/4 = 16.5 → not a multiple of 32, so
    // consecutive rows land on consecutive banks (no conflict).
    //
    // smem overhead: +1 fp16 × kBm × 2 arrays = +512 bytes total.
    // This is negligible vs the 17-34 KB already used.
    static constexpr int kScalePad = kUseGroupCache ? 1 : 0;
    __shared__ alignas(16) __half s_scale_u4[kBm][kUseGroupCache ? kGrpBuf + kScalePad : 1];
    __shared__ alignas(16) __half s_zero_u4 [kBm][kUseGroupCache ? kGrpBuf + kScalePad : 1];
    __shared__ __half s_scale_x[kBn];
    __shared__ __half s_scale_block[kBm];               // per BSR block
    __shared__ int s_sum_X[2][kBn];

    // Round 39: keep the original <=32 cache path always on, but only
    // extend the windowed 33..64-group cache to modest-M grids.  On very
    // wide outputs (for example Qwen3-14B gate_up_proj, n_cta_m=272) the
    // extra per-CTA window prefetch was not amortised and regressed T>=128.
    // q/o/kv-like shapes with n_cta_m<=40 still benefit.
    // Round 40: split cached vs no-cache into separate template variants.
    //   This lets gate_up/down no-cache shapes drop the otherwise dead
    //   16KB scale/zero shared-memory buffers and can improve occupancy.
    const bool cache_sz = kUseGroupCache &&
                          ((n_groups <= kGrpBuf) ||
                           (n_groups <= kMaxWindowedGroups && n_cta_m <= 64));

    auto issue_sz_window_load = [&](int g_base) {
        const int remaining = n_groups - g_base;
        const int g_count = (remaining < kGrpBuf) ? remaining : kGrpBuf;
        for (int idx = tid; idx < kBm * g_count; idx += kBm) {
            int m_local = idx / g_count;
            int g_local = idx - m_local * g_count;
            int g = g_base + g_local;
            int m = m_tile + m_local;
            if (m < d_out) {
                s_scale_u4[m_local][g_local] = scale_u4[(int64_t)m * stride_su_m
                                                      + (int64_t)g * stride_su_g];
                s_zero_u4 [m_local][g_local] = zero_u4 [(int64_t)m * stride_zu_m
                                                      + (int64_t)g * stride_zu_g];
            } else {
                s_scale_u4[m_local][g_local] = __half(0);
                s_zero_u4 [m_local][g_local] = __half(0);
            }
        }
    };

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

    auto issue_w_dense_load = [&](int g, int buf) {
        int m = m_tile + tid;
        uint8_t* dst = &sW[buf][tid][0];
        if (m < d_out) {
            const uint8_t* src = W_low + (int64_t)m * stride_w_m
                                       + (int64_t)(g * bytes_per_group) * stride_w_k;
            if constexpr (kUseLdmatrix) {
                // Stage C.1b: write 8× uint2 (8B) with XOR swizzle.
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint2 v = *reinterpret_cast<const uint2*>(src + i * 8);
                    int phys = swizzle_row_col(tid, i * 8);
                    *reinterpret_cast<uint2*>(dst + phys) = v;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint4 v = *reinterpret_cast<const uint4*>(src + i * 16);
                    *reinterpret_cast<uint4*>(dst + i * 16) = v;
                }
            }
        } else {
            if constexpr (kUseLdmatrix) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int phys = swizzle_row_col(tid, i * 8);
                    *reinterpret_cast<uint2*>(dst + phys) = make_uint2(0, 0);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *reinterpret_cast<uint4*>(dst + i * 16) = make_uint4(0, 0, 0, 0);
                }
            }
        }
    };

    // Stage A2: cp.async variant of issue_w_dense_load.
    // Each thread issues 4 × 16-byte cp.async for its row.
    // Out-of-bounds rows are zero-filled synchronously (cp.async with
    // pred=false does NOT zero-fill; it leaves dst unchanged).
    auto issue_w_dense_load_async = [&](int g, int buf) {
        int m = m_tile + tid;
        uint8_t* dst = &sW[buf][tid][0];
        if (m < d_out) {
            const uint8_t* src = W_low + (int64_t)m * stride_w_m
                                       + (int64_t)(g * bytes_per_group) * stride_w_k;
            if constexpr (kUseLdmatrix) {
                // Stage C.1b: swizzled cp.async.  We issue two 16-byte
                // chunks per 32-byte aligned pair (swap for odd swizzle
                // parity).  Since the 8-byte word swizzle toggles bit 3,
                // a 16-byte-aligned write only works when (row&7)∈{0,2,
                // 4,6} — but cp.async requires 16B src/dst alignment.
                // Workaround: do 8× 8-byte synchronous writes instead.
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint2 v = *reinterpret_cast<const uint2*>(src + i * 8);
                    int phys = swizzle_row_col(tid, i * 8);
                    *reinterpret_cast<uint2*>(dst + phys) = v;
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    cp_async_cg_16(dst + i * 16, src + i * 16);
                }
            }
        } else {
            // Synchronous zero-fill for out-of-bounds rows.
            if constexpr (kUseLdmatrix) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int phys = swizzle_row_col(tid, i * 8);
                    *reinterpret_cast<uint2*>(dst + phys) = make_uint2(0, 0);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    *reinterpret_cast<uint4*>(dst + i * 16) = make_uint4(0, 0, 0, 0);
                }
            }
        }
    };

    auto issue_x_load = [&](int g_or_bc, int buf) {
        if constexpr (kUseLdmatrix) {
            // Stage C.1b: iterate at 8-byte (uint2) granularity so the
            // XOR swizzle can toggle bit 3 of the column offset.
            const int total_octs = kBn * 8;  // 64 bytes / 8 bytes per oct
            for (int q = tid; q < total_octs; q += kBm) {
                int row = q >> 3;
                int oct = q & 7;
                int n = n_tile + row;
                uint2 v;
                if (n < T) {
                    int64_t off = (int64_t)n * stride_x_n
                                + (int64_t)(g_or_bc * bytes_per_group + oct * 8) * stride_x_k;
                    v = *reinterpret_cast<const uint2*>(X + off);
                } else {
                    v = make_uint2(0, 0);
                }
                int phys = swizzle_row_col(row, oct * 8);
                *reinterpret_cast<uint2*>(&sX[buf][row][phys]) = v;
            }
        } else {
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
        }
    };

    // Stage A2: cp.async variant of issue_x_load.
    // Out-of-bounds rows are zero-filled synchronously (same reason as W).
    auto issue_x_load_async = [&](int g_or_bc, int buf) {
        if constexpr (kUseLdmatrix) {
            // Stage C.1b: swizzled path uses synchronous uint2 writes
            // (cp.async can't toggle bit 3 cleanly at 8B granularity).
            const int total_octs = kBn * 8;
            for (int q = tid; q < total_octs; q += kBm) {
                int row = q >> 3;
                int oct = q & 7;
                int n = n_tile + row;
                uint2 v;
                if (n < T) {
                    int64_t off = (int64_t)n * stride_x_n
                                + (int64_t)(g_or_bc * bytes_per_group + oct * 8) * stride_x_k;
                    v = *reinterpret_cast<const uint2*>(X + off);
                } else {
                    v = make_uint2(0, 0);
                }
                int phys = swizzle_row_col(row, oct * 8);
                *reinterpret_cast<uint2*>(&sX[buf][row][phys]) = v;
            }
            return;
        }
        const int total_quads = kBn * 4;
        for (int q = tid; q < total_quads; q += kBm) {
            int row = q >> 2;
            int quad = q & 3;
            int n = n_tile + row;
            uint8_t* dst = &sX[buf][row][quad * 16];
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(g_or_bc * bytes_per_group + quad * 16) * stride_x_k;
                cp_async_cg_16(dst, X + off);
            } else {
                *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
            }
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
        // R42-P1: when kBm<BROW only load the upper or lower 64 rows
        //   of the 128-row BSR block; offset by (br&1)*64 rows.
        const uint8_t* src = W_high_blocks
                           + (int64_t)block_idx * stride_wb_blk
                           + (int64_t)(half_row_off + tid) * stride_wb_r;
        uint8_t* dst = &sW[buf][tid][0];
        if constexpr (kUseLdmatrix) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint2 v = *reinterpret_cast<const uint2*>(src + i * 8);
                int phys = swizzle_row_col(tid, i * 8);
                *reinterpret_cast<uint2*>(dst + phys) = v;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint4 v = *reinterpret_cast<const uint4*>(src + i * 16);
                *reinterpret_cast<uint4*>(dst + i * 16) = v;
            }
        }
    };

    // Stage A2.5: cp.async variant of issue_w_sparse_load.
    // W_high_blocks rows are always in-bounds (block_idx is a valid BSR block),
    // so no OOB guard needed here.
    auto issue_w_sparse_load_async = [&](int block_idx, int buf) {
        const uint8_t* src = W_high_blocks
                           + (int64_t)block_idx * stride_wb_blk
                           + (int64_t)(half_row_off + tid) * stride_wb_r;
        uint8_t* dst = &sW[buf][tid][0];
        if constexpr (kUseLdmatrix) {
            // Stage C.1b: swizzled synchronous uint2 writes.
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint2 v = *reinterpret_cast<const uint2*>(src + i * 8);
                int phys = swizzle_row_col(tid, i * 8);
                *reinterpret_cast<uint2*>(dst + phys) = v;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                cp_async_cg_16(dst + i * 16, src + i * 16);
            }
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
                if constexpr (kUseLdmatrix) {
                    // Stage C.1a+b: use ldmatrix.x4 for A operand with XOR swizzle.
                    //
                    // MMA m16n8k64.s4 A operand layout (per warp):
                    //   a0 = A[msub_base + 0..7,  ks*32..ks*32+15]  (tile 0)
                    //   a1 = A[msub_base + 8..15, ks*32..ks*32+15]  (tile 1)
                    //   a2 = A[msub_base + 0..7,  ks*32+16..ks*32+31] (tile 2)
                    //   a3 = A[msub_base + 8..15, ks*32+16..ks*32+31] (tile 3)
                    //
                    // ldmatrix.x4 lane→tile mapping:
                    //   lane in [ 0, 8):  tile 0, provides base for row msub_base+l
                    //   lane in [ 8,16):  tile 1, provides base for row msub_base+8+l
                    //   lane in [16,24):  tile 2, provides base for row msub_base+l   (col+16)
                    //   lane in [24,32):  tile 3, provides base for row msub_base+8+l (col+16)
                    //
                    // The returned 4 registers per lane hold (tile0, tile1, tile2, tile3)
                    // at position (lane % 8) row, (lane / 8) column-pair — matching MMA.
                    //
                    // Stage C.1b: apply XOR swizzle to eliminate bank conflicts.
                    //   phys_col = logical_col XOR ((row & 7) * 8)
                    //   — preserves 16-byte alignment required by ldmatrix because
                    //     logical_col ∈ {0,16,32,48} has bit 3 = 0 and XOR only
                    //     toggles bits [5:3].
                    int lane_in_tile = lane & 7;           // 0..7
                    int tile_idx     = lane >> 3;          // 0..3
                    int row_off      = (tile_idx & 1) * 8; // tiles 1,3 → row+8
                    int col_off      = (tile_idx & 2) * 8; // tiles 2,3 → col+16
                    int row = msub_base + row_off + lane_in_tile;
                    int col = kpb_base + col_off;
                    // Clamp OOB rows to a safe in-range address (row 0) and
                    //  rely on issue_*_load's zero-fill for the semantic zero.
                    if (row >= kBm) row = 0;
                    int phys_col = swizzle_row_col(row, col);
                    uint32_t r0, r1, r2, r3;
                    ldmatrix_x4_b16(&sW[buf][row][phys_col], r0, r1, r2, r3);
                    a_regs[im][0] = r0;
                    a_regs[im][1] = r1;
                    a_regs[im][2] = r2;
                    a_regs[im][3] = r3;
                } else {
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
            }

            uint32_t b_regs[kNsubPerCta][2];
            #pragma unroll
            for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                if constexpr (kUseLdmatrix) {
                    // Stage C.1a+b: ldmatrix.x2 for B operand with XOR swizzle.
                    //
                    // MMA m16n8k64.s4 B operand layout (per warp):
                    //   b0 = B[in_sub*8 + 0..7, ks*32..ks*32+15]  (tile 0)
                    //   b1 = B[in_sub*8 + 0..7, ks*32+16..ks*32+31] (tile 1)
                    //
                    // ldmatrix.x2 lane→tile mapping:
                    //   lane in [ 0, 8):  tile 0, base for row in_sub*8+l
                    //   lane in [ 8,16):  tile 1, base for row in_sub*8+l (col+16)
                    //   lane in [16,32):  unused — HW still issues reads; give
                    //                     safe in-range address and discard.
                    int lane_in_tile = lane & 7;
                    int tile_idx     = (lane >> 3) & 1;    // 0 or 1 (ignore hi bit)
                    int col_off      = tile_idx * 16;
                    int n_row        = in_sub * 8 + lane_in_tile;
                    int col          = kpb_base + col_off;
                    // Clamp OOB rows (n_row >= kBn) and lane>=16 to row 0.
                    if (n_row >= kBn || lane >= 16) n_row = 0;
                    int phys_col = swizzle_row_col(n_row, col);
                    uint32_t r0, r1;
                    ldmatrix_x2_b16(&sX[buf][n_row][phys_col], r0, r1);
                    b_regs[in_sub][0] = r0;
                    b_regs[in_sub][1] = r1;
                } else {
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
    if constexpr (kUseGroupCache) {
        if (cache_sz) {
            // Stage I: start window at g_start (may be > 0 under Split-K).
            issue_sz_window_load(g_start - (g_start % kGrpBuf));
        }
    }
    // Stage A2: use cp.async for W/X loads so that group g+1's HBM fetch
    // overlaps with group g's MMA computation (2-stage async pipeline).
    // sum_X is small (kBn ints) and stays on the synchronous path.
    if constexpr (kUseCpAsync) {
        issue_w_dense_load_async(g_start, 0);
        issue_x_load_async(g_start, 0);
        issue_sum_X_load(g_start, 0);
        cp_async_commit();
        cp_async_wait_group<0>();   // wait for g=g_start before first MMA
        __syncthreads();
    } else {
        issue_w_dense_load(g_start, 0);
        issue_x_load(g_start, 0);
        issue_sum_X_load(g_start, 0);
        __syncthreads();
    }

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

    // Stage I: window base must align to kGrpBuf boundary that contains g_start.
    // (If kUseGroupCache is false, g_window_base is unused.)
    int g_window_base = kUseGroupCache ? (g_start - (g_start % kGrpBuf)) : g_start;
    for (int g = g_start; g < g_end; ++g) {
        if constexpr (kUseGroupCache) {
            if (cache_sz && g != g_start && (g % kGrpBuf) == 0) {
                issue_sz_window_load(g);
                __syncthreads();
                g_window_base = g;
            }
        }

        const int buf = g & 1;
        if (g + 1 < g_end) {
            if constexpr (kUseCpAsync) {
                // Stage A2: issue g+1 loads asynchronously so they overlap
                // with the MMA computation for group g below.
                issue_w_dense_load_async(g + 1, buf ^ 1);
                issue_x_load_async(g + 1, buf ^ 1);
                issue_sum_X_load(g + 1, buf ^ 1);   // sum_X stays sync (small)
                cp_async_commit();
            } else {
                issue_w_dense_load(g + 1, buf ^ 1);
                issue_x_load(g + 1, buf ^ 1);
                issue_sum_X_load(g + 1, buf ^ 1);
            }
        }

        const int g_cache = cache_sz ? (g - g_window_base) : g;

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
            if constexpr (kUseGroupCache) {
                if (cache_sz) {
                    if (mrow0 < kBm) {
                        v.z0 = __half2float(s_zero_u4 [mrow0][gg]);
                        v.s0 = __half2float(s_scale_u4[mrow0][gg]);
                    }
                    if (mrow1 < kBm) {
                        v.z1 = __half2float(s_zero_u4 [mrow1][gg]);
                        v.s1 = __half2float(s_scale_u4[mrow1][gg]);
                    }
                    return v;
                }
            }
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
            return v;
        };

        auto fold_dense = [&](int d_val, int m_global, int m_local, int n_local,
                              int gg, int im, int in_sub, int r, int bb, auto pr) {
            float z = (r >> 1) ? pr.z1 : pr.z0;
            float s = (r >> 1) ? pr.s1 : pr.s0;
            // R23 sxn cache + R27: sxn factored out of g-loop, applied
            //   once after both dense and sparse branches complete.
            float sumxn = sumxn_cache[in_sub][r & 1];  // R24: register cache
            float corrected = static_cast<float>(d_val) - z * sumxn;
            y_fp[im][in_sub][r] += corrected * s;  // R27: no sxn here
        };
        run_mma_pass(buf, fold_dense, prefetch_dense, g_cache);

        // Stage A2: wait for the g+1 cp.async loads (issued above) to
        // complete before the next iteration uses buf^1.
        if constexpr (kUseCpAsync) {
            if (g + 1 < n_groups) {
                cp_async_wait_group<0>();
            }
        }
        __syncthreads();
    }

    // SPARSE BRANCH
    // R41-P1: sparse branch was compile-time disabled for kBm<BROW
    //   (`if constexpr (kBm == BROW)`) because `hp_row_offsets[br]` was
    //   out of bounds when grid_M > BSR_nrow.
    // R42-P1: re-enabled via bsr_br = br/kBsrPerCta which maps multiple
    //   CTAs onto the same BSR row; each CTA loads only half the block
    //   (upper/lower 64 rows) via half_row_off in issue_w_sparse_load.
    //   The legacy kBm==BROW path has kBsrPerCta=1 and half_row_off=0
    //   so it compiles to identical machine code.
    //
    // r62 F2: under Split-K (split_k > 1), the sparse branch does NOT
    //   loop over groups — it iterates over BSR blocks indexed by bc
    //   (col index in K), so its contribution is group-independent
    //   and must be added exactly ONCE across all gridDim.z CTAs for
    //   a given (m_tile, n_tile).  Gate on split_k_idx == 0 so only
    //   the first K-split picks up the sparse contribution; the other
    //   splits skip it.  This is safe for kBm=64 too because the
    //   bsr_br/half_row_off remapping inside issue_w_sparse_load does
    //   not depend on split_k_idx.
    if (split_k_idx == 0) {
        const int blk_start = hp_row_offsets[bsr_br];
        const int blk_end   = hp_row_offsets[bsr_br + 1];

        if (blk_start < blk_end) {
            int bc0 = __ldg(&hp_col_indices[blk_start]);
            if constexpr (kUseCpAsync) {
                issue_w_sparse_load_async(blk_start, 0);
                issue_x_load_async(bc0, 0);
                cp_async_commit();
                cp_async_wait_group<0>();
            } else {
                issue_w_sparse_load(blk_start, 0);
                issue_x_load(bc0, 0);
            }
            __syncthreads();

            for (int block_idx = blk_start; block_idx < blk_end; ++block_idx) {
                const int bc = __ldg(&hp_col_indices[block_idx]);
                const int buf = (block_idx - blk_start) & 1;
                if (block_idx + 1 < blk_end) {
                    int bc_next = __ldg(&hp_col_indices[block_idx + 1]);
                    if constexpr (kUseCpAsync) {
                        // Stage A2.5: overlap next block's HBM load with MMA.
                        issue_w_sparse_load_async(block_idx + 1, buf ^ 1);
                        issue_x_load_async(bc_next, buf ^ 1);
                        cp_async_commit();
                    } else {
                        issue_w_sparse_load(block_idx + 1, buf ^ 1);
                        issue_x_load(bc_next, buf ^ 1);
                    }
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
                    // R27: no sxn here (applied after both branches).
                    y_fp[im][in_sub][r] += 16.0f * static_cast<float>(d_val) * s;
                };
                run_mma_pass(buf, fold_sparse, prefetch_sparse, bc);

                if constexpr (kUseCpAsync) {
                    if (block_idx + 1 < blk_end) {
                        cp_async_wait_group<0>();
                    }
                }
                __syncthreads();
            }
        }
    }

    // R27: apply sxn once after both dense and sparse branches.
    //   y = sxn_n * (dense_sum + sparse_sum).  Valid because neither
    //   fold touched sxn and distribution is associative over the +.
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im) {
        #pragma unroll
        for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                y_fp[im][in_sub][r] *= sxn_cache[in_sub][r & 1];
            }
        }
    }

    // Writeback.
    // Stage B.1 — vectorized writeback.
    // Stage I — Split-K: when split_k > 1, write fp32 partial sums to
    //   Y_partial[split_k_idx * d_out * T + m * T + n] instead of Y.
    //   The reduce kernel will sum the partials and write fp16 to Y.
    //   When split_k == 1, write directly to Y (fp16) as before.
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im) {
        int msub_base = warp_id * 32 + im * 16;
        #pragma unroll
        for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
            int nsub_base = in_sub * 8;
            #pragma unroll
            for (int rpair = 0; rpair < 2; ++rpair) {
                int r0 = rpair * 2;       // 0 or 2
                int r1 = r0 + 1;          // 1 or 3
                int row_local = (lane >> 2) + ((r0 >> 1) ? 8 : 0);
                int col_local = (lane & 3) * 2;
                int m_global = m_tile + msub_base + row_local;
                int n_local0 = nsub_base + col_local;
                int n_local1 = n_local0 + 1;
                if (m_global >= d_out) continue;
                if (n_local0 >= kBn) continue;
                int n_global0 = n_tile + n_local0;
                if (n_global0 >= T) continue;

                float v0 = y_fp[im][in_sub][r0];
                float v1 = y_fp[im][in_sub][r1];

                if (split_k > 1) {
                    // Stage I: write fp32 partial to Y_partial.
                    int64_t base = (int64_t)split_k_idx * d_out * T;
                    int64_t off0 = base + (int64_t)m_global * T + n_global0;
                    Y_partial[off0] = v0;
                    if ((n_local1 < kBn) && (n_global0 + 1 < T)) {
                        Y_partial[off0 + 1] = v1;
                    }
                } else {
                    int64_t y_off = (int64_t)m_global * stride_y_m
                                  + (int64_t)n_global0 * stride_y_n;
                    bool pair_ok =
                        (n_local1 < kBn) && (n_global0 + 1 < T)
                        && (stride_y_n == 1)
                        && ((n_global0 & 1) == 0)
                        && ((stride_y_m & 1) == 0);
                    if (pair_ok) {
                        __half2 packed = __floats2half2_rn(v0, v1);
                        *reinterpret_cast<__half2*>(&Y[y_off]) = packed;
                    } else {
                        // Scalar fallback for tile/row edges.
                        Y[y_off] = __float2half(v0);
                        if ((n_local1 < kBn) && (n_global0 + 1 < T)) {
                            Y[y_off + stride_y_n] = __float2half(v1);
                        }
                    }
                }
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

    // Round 41-P1: opt-in kBm=64 for the dense branch ONLY.
    //   Gate: strictly hp_blocks == 0 (no BSR sparse work; BROW=128
    //   packing assumption for sparse cannot be broken) AND the
    //   kBm=128 grid under-fills a wave AND T is in the mid-batch
    //   wave-starvation regime (T in [16, 64], d_out <= 2048).
    //   When this gate fires we halve kBm, doubling grid_M and
    //   recovering up to 2x wave occupancy.  Mirror of dense_gemm
    //   R40-B but carefully restricted so hp>0 shapes NEVER reach
    //   the kBm=64 path (sparse block indices would misalign).
    // R42-P1: hp_empty restriction LIFTED — the sparse branch now
    //   supports kBm<BROW via bsr_br/half_row_off remapping inside
    //   the kernel.  Gate now fires regardless of hp_nnz.
    const int64_t hp_nnz = hp_col_indices.numel();
    const int n_cta_m_at_128 = ceil_div(d_out, 128);
    const bool hp_empty = (hp_nnz == 0);
    // R43 gate — derived from bench_r43_gate_sweep_20260427_213921 heat-map
    // on RTX 4090, hp_ratio ∈ {0, 0.05}, d_in=4096.  speedup(64/128):
    //
    // hp=0.05:        d=1024  d=2048  d=3072  d=4096
    //   T=8           1.161   1.163   1.144   1.131    all ✓
    //   T=16          1.146   1.165   1.147   0.867    d<=3072 ✓
    //   T=32          1.156   1.177   1.052   0.788    d<=3072 ✓
    //   T=48          1.156   0.820   1.018   0.992    d=1024 only ✓
    //   T=64          1.176   0.820   1.036   0.994    d=1024 only ✓
    //   T=96          1.053   1.030   0.964   0.898    d<=2048 ≈
    //   T=128         0.808   1.039   0.966   0.900    avoid
    //
    // R44 update (bench_r43_gate_sweep_20260427_214956 with kBn
    // demote on kBm=64 & T in [32,96]):
    //
    // hp=0.05:        d=1024  d=2048  d=3072  d=4096
    //   T=8           1.158   1.171   1.147   1.130   all ✓
    //   T=16          1.145   1.165   1.147   0.862   d<=3072 ✓
    //   T=32          1.151   1.177   1.048   1.029   d<=4096 ~ (was ×)
    //   T=48          1.154  *1.069*  1.196   1.145   all ✓ (was × @ d=2048)
    //   T=64          1.172  *1.065*  1.194   1.018   all ✓ (was × @ d=2048)
    //   T=96          1.050   1.183   0.913   0.524   d<=2048 ✓  d>=3072 ×
    //   T=128         0.807   1.036   0.966   0.895   avoid
    //
    // R44 kBn demote fixes the d=2048 T in [48,64] "bad zone" and
    // simultaneously unlocks d=3072/4096 at T in [48,64].  A new
    // cliff appears at T=96 d_out>=3072 (0.524x at d=4096 is the
    // worst cell yet!) — gate must strictly exclude it.
    //
    // r61 Stage G fix (hp=0 dense probe, 2026-04-29 bottleneck_sweep):
    //   In the historical R44 hp=0.05 sweep, kBm=64 at T=32 d_out=4096
    //   scored 1.029x but rounded down to `d_out <= 3072`.  In the
    //   hp=0 dense LLM-inference regime used by bench_qwen3_shapes,
    //   (4096,4096,T∈{32,48,64,96}) shows kBm=64 ⊕ kBn=32 winning by
    //   +22%..+136% over the current DEF (kBm=128).  Widen the gate:
    //     T ∈ (16, 32] : d_out <= 4096  *and*  d_in <= 4096
    //         (T=16 and tall d_in=14336 prefer kBm=128 / kBn=8, so
    //          tighten rather than include them.)
    //     T == 96      : keep d_out <= 2048 (the 0.524x cliff is real).
    const bool r44_shape_ok =
        ( (T <= 8)   && (d_out <= 4096) )
     || ( (T <= 16)  && (d_out <= 3072) )
     || ( (T > 16 && T <= 32)  && (d_out <= 4096) && (d_in <= 4096) )
     || ( (T >= 48 && T <= 64)  && (d_out <= 4096) )
     || ( (T == 96)  && (d_out <= 2048) )
     // R52: T=128 with 512<=d_out<=2048 and d_in>=2048 benefits from kBm=64.
     //   d_out<512 or d_in<2048 (ng<16): overhead > benefit.
     //   d_out=4096 at T=128: 0.95x with kBm=64, excluded.
     || ( (T == 128) && (d_out >= 512) && (d_out <= 2048) && (d_in >= 2048) );    // R45: the `< 64` threshold was too strict.  Probe data for T=48
    // d=4096 (n_cta_m_at_128=32, ceil(T/32)=2 → product = 64) showed:
    //     kBm=64 kBn=8 : 40.94us  (best)
    //     R44 auto     : 47.25us  (took kBm=128 because 64 not <64)
    // i.e. auto was 15% slower than the clear-winner configuration.
    // Relaxing to `<= 64` lets the R44 gate catch T in {48, 64} at
    // d_out=4096 while still blocking T>=128 (which would be 32*4=128).
    const bool kbm64_gate_default =
        r44_shape_ok &&
        ((int64_t)n_cta_m_at_128 * ceil_div(T, 32) <= 64);

    // R42-P1 bench hook: HKUST_V9_FUSED_FORCE_KBM overrides the gate.
    //   "128"       : force kBm=128 (disable R41/R42/R43 opt-in).
    //   "64"        : force kBm=64 (ignore T/d_out gate).
    //   unset/other : use the default gate above.
    bool kbm64_gate = kbm64_gate_default;
    {
        const char* env = std::getenv("HKUST_V9_FUSED_FORCE_KBM");
        if (env != nullptr) {
            if (env[0] == '1' && env[1] == '2' && env[2] == '8') kbm64_gate = false;
            else if (env[0] == '6' && env[1] == '4')             kbm64_gate = true;
        }
    }
    (void)hp_empty;  // R42-P1: retained for debug/future gates.
    const int kbm_pick = kbm64_gate ? 64 : 128;
    const int n_cta_m = ceil_div(d_out, kbm_pick);
    // Stage F (r61) — occupancy-aware cache gate (F.1 experiment-driven).
    //
    // The empirical bw_cache_experiment.py sweep at r61 showed that for
    // ng >= 16 shapes, smem-hungry group-cache (17 KB of scale/zero)
    // caps SM occupancy at 2 CTAs/SM → HBM BW stuck at 22-30 %. Disabling
    // the cache lifts BW to 28-55 % on those shapes (1.05–1.25×).
    //
    // But for ng ≤ 8 the full scale/zero set fits in cache exactly once,
    // eliminating ng identical HBM trips per CTA; disabling cache there
    // regressed 5-9 %. We therefore tighten the gate:
    //   * ng ≤ 8            : always cache (pure reuse win).
    //   * ng ≤ kGrpBuf (32)  +  T ≤ 32    : cache (compute-heavy, low T).
    //   * everything else   : no cache  (occupancy wins over reuse).
    //
    // The env switch HKUST_V9_FUSED_FORCE_CACHE still overrides this for
    // debug / A-B benching.
    //
    // C.1 refinement (r64, 2026-04-30 — _phase3_phase4_work_plan §1 C.1):
    //   The T=128 sweep (logs/r64_path_c/c1_group_cache_sweep.json,
    //   25 shapes) revealed a gap: n_groups ∈ [33, 48] at T=128 benefits
    //   from cache by +4.3% .. +7.6% because the windowed-cache path
    //   (n_cta_m ≤ 64) already matches this regime but the host gate
    //   never allowed it to trigger.  Concrete wins:
    //     Qwen3-14B gate_up T=128 (5120→34816, n_g=40): +7.6%
    //     Qwen3-1.7B down     T=128 (6144→2048, n_g=48):  +5.7%
    //     Qwen3-14B q / o     T=128 (5120→5120, n_g=40): +4.3–4.4%
    //   Safety:  kernel-internal `cache_sz` (L213) re-checks
    //   `n_groups ≤ kMaxWindowedGroups && n_cta_m ≤ 64`, so enabling
    //   this host gate for n_groups > 48 or wide grids is a no-op
    //   at the execution level (smem is allocated but cache path
    //   stays off at runtime).  That means the host-gate change is
    //   safe — it only unlocks a previously-unreachable path, never
    //   forces cache onto a shape that disables it internally.
    bool use_group_cache =
        (n_groups <= 8) ||
        (n_groups <= kGrpBuf && T <= 32) ||
        // C.1 extension: T=128 moderate-ng shapes with small-ish grid
        (T == 128 && n_groups <= kMaxWindowedGroups && n_cta_m <= 64);
    {
        const char* env = std::getenv("HKUST_V9_FUSED_FORCE_CACHE");
        if (env != nullptr) {
            if (env[0] == '0') use_group_cache = false;
            else if (env[0] == '1') use_group_cache = true;
        }
    }

    // Stage I: forward-declare split_k and Y_partial_ptr so do_launch can
    // capture them. Values are computed after kbn_pick is known (below).
    int split_k = 1;
    float* Y_partial_ptr = nullptr;
    torch::Tensor Y_partial_tensor;  // keeps the buffer alive until after launch

    auto do_launch = [&](auto kBn_c, auto kCache_c, auto kBm_c, auto kCpAsync_c) {
        constexpr int kBn = decltype(kBn_c)::value;
        constexpr bool kUseGroupCache = decltype(kCache_c)::value;
        constexpr int kBmLocal = decltype(kBm_c)::value;
        constexpr bool kUseCpAsync = decltype(kCpAsync_c)::value;
        dim3 block(kBmLocal, 1, 1);
        dim3 grid(ceil_div(d_out, kBmLocal), ceil_div(T, kBn), split_k);
        // Stage C.1a: ldmatrix opt-in via HKUST_V9_LDMATRIX env var.
        //   0 (default) → legacy manual smem loads.
        //   1           → ldmatrix.x4 for A, ldmatrix.x2 for B.
        // Kept as a runtime switch during development so the parity/bench
        // harness can A/B compare with zero code changes to callers.
        static const int ldmatrix_env = []() {
            const char* e = std::getenv("HKUST_V9_LDMATRIX");
            return e ? std::atoi(e) : 0;
        }();
        if (ldmatrix_env) {
            fused_dense_sparse_mma_int4_kernel<kBn, kUseGroupCache, kBmLocal, kUseCpAsync, true><<<grid, block, 0, stream>>>(
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
                (split_k > 1) ? Y_partial_ptr : nullptr,
                d_out, d_in, T, n_groups, split_k,
                W_low.stride(0), W_low.stride(1),
                X_s4.stride(0), X_s4.stride(1),
                scale_u4.stride(0), scale_u4.stride(1),
                zero_u4.stride(0), zero_u4.stride(1),
                sum_X.stride(0), sum_X.stride(1),
                W_high_blocks.stride(0), W_high_blocks.stride(1), W_high_blocks.stride(2),
                Y_total.stride(0), Y_total.stride(1)
            );
            return;
        }
        fused_dense_sparse_mma_int4_kernel<kBn, kUseGroupCache, kBmLocal, kUseCpAsync, false><<<grid, block, 0, stream>>>(
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
            (split_k > 1) ? Y_partial_ptr : nullptr,
            d_out, d_in, T, n_groups, split_k,
            W_low.stride(0), W_low.stride(1),
            X_s4.stride(0), X_s4.stride(1),
            scale_u4.stride(0), scale_u4.stride(1),
            zero_u4.stride(0), zero_u4.stride(1),
            sum_X.stride(0), sum_X.stride(1),
            W_high_blocks.stride(0), W_high_blocks.stride(1), W_high_blocks.stride(2),
            Y_total.stride(0), Y_total.stride(1)
        );
    };

    // Round 25b: wave-aware kBn dispatch (mirror dense_gemm).
    //   Pick kBn=64 iff grid at kBn=64 fills at least 1 wave (128 CTAs on
    //   SM89).  Otherwise fall back to kBn=32 for better wave occupancy.
    // Round 32: for small T (<=32) extend the kBn=8 bucket when kBn=32
    //   would not even fill half a wave (SM89 = 128 SMs).  grid at
    //   kBn=8 is N times larger than at kBn=32 (N = 32/8 = 4 for T=32,
    //   2 for T=16), which turns a heavily under-populated launch into
    //   a full or near-full wave.  Rule: if current kBn choice gives
    //   < 64 CTAs (= 0.5 wave), try a smaller kBn and pick the one with
    //   grid >= 64 CTAs, preferring the largest kBn that meets the bar.
    // Round 35 rollback: keep the dense scale/zero cache at 32 groups.
    //   A 40-group cache regressed the wide Qwen3 gate_up shape at T>=16.
    // Round 36 rollback: do NOT force T<=16 onto kBn=8.  That helped
    //   some larger-T wide shapes only indirectly (via the cache rollback)
    //   but badly regressed gate_up_proj at T=16.  Keep the original
    //   wave-aware dispatch with the existing T<=8 fast rule.
    // Round 37 (reverted): forcing 9<=T<=32 onto kBn=32 did not help the
    //   target gate_up T=16 shape materially, but it regressed q/kv/o and
    //   down_proj at T=16.  Keep the original wave-aware rule below.
    // r62 F2 Step 2: compute split_k BEFORE kBn pick, so pick can be
    //   split-K-aware.  The split-K heuristic itself doesn't depend on
    //   kbn_pick (it uses ceil_div(T, 64) as a fixed estimator of how
    //   many N-tiles we'll end up with), so moving it up is safe.
    //
    //   Previous order (kBn pick before split-K) made the kBn-pick
    //   `waves_at` thresholds ignore that sk=N multiplies the effective
    //   grid by N, causing T=32 shapes to pick kBn=8 (over-fragmented)
    //   even when sk=4 already delivered the needed CTA count via the
    //   split-K axis.
    //
    // Split-K gate (moved from below):
    //
    // Note on small T: production T<=16 is routed to dedicated GEMV
    // kernels (fused_quant_gemv / fused_gemv_smallT), so the main
    // kernel's behaviour at T<=4 doesn't affect e2e speedup.  But for
    // synthetic benches calling the main kernel directly at T=1/2/4,
    // sk=4 over-splits: the reduce kernel launch cost (~7us) exceeds
    // the split-K compute savings at such small N.  Gate split-K to
    // T >= 8 to avoid reporting sweep regressions for those cases.
    //
    // r62 F2: split-K is now safe for hp>0 shapes too.  The sparse
    //   branch inside the kernel is gated on split_k_idx==0 so the
    //   sparse contribution is added exactly once across all K-splits.
    //   Without this fix, auto dispatcher was forced to sk=1 on all
    //   Qwen3 projections (hp_ratio=0.05), completely canceling the
    //   F2 dispatch improvements measured in the dense-only sweep.
    if (n_groups >= 16 && T >= 8) {
        const int grid_mn_at_kbn64 =
            n_cta_m_at_128 * ceil_div(T, 64);
        const int target_wave = 128;
        const int denom = (grid_mn_at_kbn64 > 1) ? grid_mn_at_kbn64 : 1;
        int want_sk = ceil_div(target_wave, denom);
        if      (want_sk >= 4) want_sk = 4;
        else if (want_sk >= 2) want_sk = 2;
        else                   want_sk = 1;
        if      (want_sk == 4 && n_groups % 4 == 0) split_k = 4;
        else if (want_sk >= 2 && n_groups % 2 == 0) split_k = 2;
        else                                        split_k = 1;
    }
    // Override for testing (read here too so pick() below sees the
    // override-adjusted split_k).
    {
        const char* env_sk = std::getenv("HKUST_V9_FUSED_FORCE_SPLITK");
        if (env_sk != nullptr) {
            int sk = atoi(env_sk);
            if (sk >= 1 && sk <= 32) split_k = sk;
        }
    }

    auto waves_at = [&](int kBn_c) {
        // r62 F2: account for split-K's multiplicative effect on the
        // effective grid.  With sk=N the kernel launches N times more
        // CTAs along the K-dimension, so the wave-occupancy thresholds
        // below must be divided by sk before comparison.  Equivalent
        // to multiplying the raw wave count by sk.
        return (int64_t)n_cta_m * ceil_div(T, kBn_c) * split_k;
    };
    auto pick = [&]() -> int {
        if (T <= 8) return 8;
        // Stage E (r56) — large-ng override: when n_groups is very deep
        // (>= 64) the K-loop inside each CTA dominates; shrinking kBn to
        // fit wave-aware thresholds backfires because it doubles the
        // launch count without meaningfully raising SM occupancy (the
        // kernel is already compute/BW bound). For n_groups >= 64 stick
        // with kBn=64 as long as the grid still has at least a quarter
        // wave (>= 32 CTAs).
        // Measured gain on RTX 4090 (r54 baseline):
        //   4096x14336x128 ng=112:   174.67 -> 140.43 us (1.24x)
        if (n_groups >= 64 && waves_at(64) >= 32) return 64;
        if (waves_at(64) >= 128) return 64;
        if (waves_at(32) >= 64)  return 32;
        return 8;
    };
    int kbn_pick = pick();
    // R44: when kBm=64, grid_M doubles so waves_at() thresholds fire
    //   earlier.  In particular at d_out=2048 T=48/64 with kBm=64:
    //     waves_at(32) = 32*2 = 64 → pick kBn=32
    //   but this allocates 32 output columns per CTA for a short T, so
    //   the second N-tile mostly tail-warps; waves_at(8) = 32*6..8
    //   would be far healthier.  Demote kBn one step when kBm=64 and T
    //   lands in the [32, 96] "awkward mid-T" band so kBn=8 is used.
    //   Guarded so T<=8 (already kBn=8) and T>=128 (kBn=64 healthy) are
    //   unaffected.
    //
    // r61 Stage G refinement: the demote was originally calibrated on
    // d_out=2048 where `kBm=64 kBn=8` cleared the "bad zone".  Fresh
    // dense-mode probe (4096,4096,T∈{48,64}) shows the opposite for
    // large d_out: kBn=32 is 1.8x–2.4x faster than kBn=8 there (the
    // column tail-warp cost is dominated by the extra waves_at(8)
    // launches when d_out is large).  Restrict the demote to the
    // originally-justified d_out <= 2048 band.
    if (kbm_pick == 64 && T >= 32 && T <= 96 && kbn_pick >= 32 && d_out <= 2048) {
        kbn_pick = 8;
    }
    // R44 bench hook: HKUST_V9_FUSED_FORCE_KBN overrides the pick.
    //   "8" / "32" / "64" force that value.  Any other / unset uses
    //   the auto pick above.
    {
        const char* env_n = std::getenv("HKUST_V9_FUSED_FORCE_KBN");
        if (env_n != nullptr) {
            if      (env_n[0] == '6' && env_n[1] == '4') kbn_pick = 64;
            else if (env_n[0] == '3' && env_n[1] == '2') kbn_pick = 32;
            else if (env_n[0] == '8' && env_n[1] == '\0') kbn_pick = 8;
        }
    }
    // Stage I (r60): The initial Split-K design included a "kbn override"
    //   that switched kbn=8 -> kbn=64 when Split-K was enabled.  The
    //   calibration bench showed this was counterproductive:
    //     1024x4096x128  kbn=64+sk=4: 19.63us
    //                    kbn=8 +sk=4: 18.10us  <- better
    //   Keeping the wave-based kbn pick and only toggling split_k gives
    //   the best of both worlds.  No override needed.

    // Stage A2 dispatcher: use cp.async when n_groups >= 16 (large-K shapes
    // where HBM load latency dominates and overlap benefit > overhead).
    // For n_groups <= 8 the synchronous path is faster (cp.async overhead
    // exceeds the latency hiding benefit for short K-loops).
    const bool use_cp_async = (n_groups >= 16);

    //
    // Stage I: split_k is computed ABOVE (before the kBn pick, so the
    //   wave-aware kBn pick can see the split-K multiplier).  The
    //   buffer allocation below just uses whatever value is currently
    //   in `split_k`.
    //
    // kSplitKMax is retained as documentation of the historical upper
    // bound (reduce-kernel cost dominates beyond 4, we cap at 4 above).
    constexpr int kSplitKMax = 8;
    (void)kSplitKMax;

    // Allocate Y_partial buffer for split_k > 1.
    // Y_partial_ptr and Y_partial_tensor were forward-declared above.
    if (split_k > 1) {
        Y_partial_tensor = torch::empty(
            {split_k, d_out, T},
            torch::TensorOptions().dtype(torch::kFloat32).device(Y_total.device())
        );
        Y_partial_ptr = Y_partial_tensor.data_ptr<float>();
    }

    auto launch_for_kbn = [&](auto kBn_c, auto kBm_c) {
        auto cp_true  = std::true_type{};
        auto cp_false = std::false_type{};
        if (use_group_cache) {
            if (use_cp_async) do_launch(kBn_c, std::true_type{},  kBm_c, cp_true);
            else              do_launch(kBn_c, std::true_type{},  kBm_c, cp_false);
        } else {
            if (use_cp_async) do_launch(kBn_c, std::false_type{}, kBm_c, cp_true);
            else              do_launch(kBn_c, std::false_type{}, kBm_c, cp_false);
        }
    };
    if (kbm_pick == 128) {
        auto kbm_c = std::integral_constant<int, 128>{};
        if      (kbn_pick == 64) launch_for_kbn(std::integral_constant<int, 64>{}, kbm_c);
        else if (kbn_pick == 32) launch_for_kbn(std::integral_constant<int, 32>{}, kbm_c);
        else                     launch_for_kbn(std::integral_constant<int, 8>{},  kbm_c);
    } else {
        auto kbm_c = std::integral_constant<int, 64>{};
        if      (kbn_pick == 64) launch_for_kbn(std::integral_constant<int, 64>{}, kbm_c);
        else if (kbn_pick == 32) launch_for_kbn(std::integral_constant<int, 32>{}, kbm_c);
        else                     launch_for_kbn(std::integral_constant<int, 8>{},  kbm_c);
    }

    // Stage I: reduce Y_partial -> Y (fp16) when split_k > 1.
    if (split_k > 1) {
        // Simple reduce: each thread handles one (m, n) element.
        // Grid: (d_out * T + 255) / 256 blocks of 256 threads.
        const int total_elems = d_out * T;
        const int reduce_threads = 256;
        const int reduce_blocks = (total_elems + reduce_threads - 1) / reduce_threads;
        split_k_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
            Y_partial_ptr,
            reinterpret_cast<__half*>(Y_total.data_ptr<at::Half>()),
            d_out, T, split_k
        );
    }

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace fused_dense_sparse_mma_int4
}  // namespace hkust_v9
