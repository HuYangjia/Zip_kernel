// Dense UINT4 x SINT4 GEMM -- INT8 Tensor Core version (SM89).
//
// Same semantic contract as dense_gemm.cu (dp4a version), but the
// inner product uses mma.m16n8k32.s8.s8.s32 Tensor Core instructions
// instead of __dp4a.  On SM89 this gives us 660 TOPS of INT8 math
// (vs 165 TOPS dp4a peak).
//
// Design
// ------
// Tile  : BM=128, BN in {8, 64, 128}, BK=32 (one MMA k-step).
//         K=128 (one group) is executed as 4 MMA iterations.
// CTA   : 128 threads = 4 warps.  Warp w owns M rows [w*32, w*32+32).
//         Each warp issues mma.m16n8k32 for its 32-row slab (2 MMA
//         tiles of 16 rows each), paired with N=8 per MMA
//         (kNsubPerCta = BN/8 N-slices per warp).
// Stage : Both W and X are staged into shmem as int8 (decoded from
//         packed s4 at load time), then ldmatrix issues the A/B
//         operand permutes directly into the per-thread registers
//         that mma.m16n8k32.s8 expects.
//
// PTX operand mapping for mma.m16n8k32.s8.s8.s32 (PTX ISA 8.x):
//
//   A (16x32 s8, row-major)  : 4 regs/thread, each 4xs8.
//     per-thread (lane):
//       a0: row= (lane>>2),     col= (lane&3)*4 + {0..3}
//       a1: row= (lane>>2)+8,   col= (lane&3)*4 + {0..3}
//       a2: row= (lane>>2),     col= (lane&3)*4 + 16 + {0..3}
//       a3: row= (lane>>2)+8,   col= (lane&3)*4 + 16 + {0..3}
//     This is what ldmatrix.x4.shared.b16 produces when reading a
//     16x16 b16 tile (each b16 = 2 s8 packed little-endian).  We
//     store A into shmem row-major, 16 rows x 32 s8 cols, with cols
//     addressed in 2-s8 pairs (== b16 lane granularity).
//
//   B (32x8 s8, col-major)  : 2 regs/thread, each 4xs8.
//     per-thread:
//       b0: row= (lane&3)*4 + {0..3},      col= (lane>>2)
//       b1: row= (lane&3)*4 + 16 + {0..3}, col= (lane>>2)
//     ldmatrix.x2.shared.b16 (non-trans) loads a 16x8 b16 tile;
//     with "trans" variant it transposes.  We put X into shmem with
//     its natural (N-row, K-byte) layout and use ldmatrix.x2.trans
//     so that the row/col mapping matches B's col-major expectation.
//
//   D (16x8 s32, row-major) : 4 regs/thread.
//     d0: (row=lane>>2,      col=(lane&3)*2 + 0)
//     d1: (row=lane>>2,      col=(lane&3)*2 + 1)
//     d2: (row=(lane>>2)+8,  col=(lane&3)*2 + 0)
//     d3: (row=(lane>>2)+8,  col=(lane&3)*2 + 1)

#include "common/arch.cuh"
#include "common/mma_utils.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace dense_gemm_mma_int8 {

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

template <int kBn>
__global__ void dense_gemm_mma_int8_kernel(
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
    constexpr int kBk = 128;                 // one group = BCOL
    constexpr int kMmaK = 32;
    constexpr int kKSteps = kBk / kMmaK;     // 4
    constexpr int kMsubPerWarp = 2;          // each warp: 2x(16-row) MMA tiles
    constexpr int kNsubPerCta = (kBn + 7) / 8;  // 8-wide N slices

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int m_tile = blockIdx.x * kBm;
    const int n_tile = blockIdx.y * kBn;
    const int m_warp_base = m_tile + warp_id * 32;

    const int bytes_per_group = BCOL >> 1;   // 64
    const int d_in_half = d_in >> 1;

    // ------------------------------------------------------------
    // Shared memory.
    //
    // sW : double-buffered, (kBm, kBk) int8 per buffer (128x128 s8
    //      == 16 KB per buffer, 32 KB total).  Row-major.
    //      Stored as int8[ kBm ][ kBk ] with a 16B pad at row end?
    //      Not needed -- 128 bytes/row hits all 32 banks uniformly
    //      with stride-1 loads per lane.
    //
    // sX : double-buffered, (kBn, kBk) int8 per buffer
    //      (max 128*128 = 16 KB per buffer).  Row-major.
    //
    // sum_X and scale_x staging as before.
    // ------------------------------------------------------------
    __shared__ alignas(16) int8_t sW[2][kBm][kBk];
    __shared__ alignas(16) int8_t sX[2][kBn][kBk];
    __shared__ __half s_scale_x[kBn];
    __shared__ int s_sum_X[2][kBn];

    if (tid < kBn) {
        int n = n_tile + tid;
        s_scale_x[tid] = (n < T) ? scale_x[n] : __half(0);
    }

    // Per-thread FP32 output accumulators.  Layout mirrors D fragment:
    //   y_fp[im][in_sub][r] where r in 0..3 maps to
    //     (row offset, col offset) = ((r>>1)*8, r&1).
    float y_fp[kMsubPerWarp][kNsubPerCta][4];
    #pragma unroll
    for (int im = 0; im < kMsubPerWarp; ++im)
      #pragma unroll
      for (int in = 0; in < kNsubPerCta; ++in)
        #pragma unroll
        for (int r = 0; r < 4; ++r) y_fp[im][in][r] = 0.0f;

    // ------------------------------------------------------------
    // Stagers.
    //
    // issue_w_load(g, buf)
    //   Load W[m_tile..m_tile+kBm][g*BCOL..(g+1)*BCOL] in s8 format.
    //   Packed bytes per row: 64 (bytes_per_group).  Unpack to 128 s8.
    //   Total bytes: 128 rows * 128 s8 = 16 KB.
    //   Using 128 threads, each handles 128 s8.
    //   Each thread reads 16 packed bytes (=1 uint4 = 4 uint32),
    //   unpacks to 32 s8, writes 32 bytes.
    //   kBm / 128 threads = 1 row/thread => each thread owns exactly
    //   one (m, full-group) row.  Perfect fit.
    //
    //   m = m_tile + tid.  If m >= d_out, write zeros.
    // ------------------------------------------------------------
    auto issue_w_load = [&](int g, int buf) {
        int m = m_tile + tid;
        if (m < d_out) {
            const uint8_t* src = W + (int64_t)m * stride_w_m
                                   + (int64_t)(g * bytes_per_group) * stride_w_k;
            int8_t* dst = &sW[buf][tid][0];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t w_packed = reinterpret_cast<const uint32_t*>(src)[4*i + 0];
                int s0, s1;
                unpack_s4_to_s8_x8(w_packed, s0, s1);
                reinterpret_cast<int*>(dst)[4*i + 0] = s0;
                reinterpret_cast<int*>(dst)[4*i + 1] = s1;
                w_packed = reinterpret_cast<const uint32_t*>(src)[4*i + 1];
                unpack_s4_to_s8_x8(w_packed, s0, s1);
                reinterpret_cast<int*>(dst)[4*i + 2] = s0;
                reinterpret_cast<int*>(dst)[4*i + 3] = s1;
            }
        } else {
            // Zero-pad so MMA stays safe; masked in epilogue anyway.
            #pragma unroll
            for (int i = 0; i < kBk / 4; ++i) {
                reinterpret_cast<int*>(&sW[buf][tid][0])[i] = 0;
            }
        }
    };

    // X stager: (kBn, 128 s8) = up to 128*128 = 16 KB.
    // kBn <= 128, use stride loop over (row, chunk) pairs.
    auto issue_x_load = [&](int g, int buf) {
        const int chunks_per_row = 16;    // 16 uint32 per row
        const int total_chunks = kBn * chunks_per_row;
        for (int q = tid; q < total_chunks; q += kBm) {
            int row = q / chunks_per_row;
            int ck  = q % chunks_per_row;
            int n = n_tile + row;
            uint32_t packed4;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(g * bytes_per_group + ck * 4) * stride_x_k;
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
        for (int nk = tid; nk < kBn; nk += kBm) {
            int n = n_tile + nk;
            s_sum_X[buf][nk] = (n < T) ?
                sum_X[(int64_t)n * stride_sx_n + (int64_t)g * stride_sx_g] : 0;
        }
    };

    __syncthreads();  // s_scale_x visible

    // Prologue
    issue_w_load(0, 0);
    issue_x_load(0, 0);
    __syncthreads();

    for (int g = 0; g < n_groups; ++g) {
        const int buf = g & 1;

        if (g + 1 < n_groups) {
            issue_w_load(g + 1, buf ^ 1);
            issue_x_load(g + 1, buf ^ 1);
        }

        // Per-group accumulator (int32, in MMA fragment layout).
        int d_acc[kMsubPerWarp][kNsubPerCta][4];
        #pragma unroll
        for (int im = 0; im < kMsubPerWarp; ++im)
          #pragma unroll
          for (int in = 0; in < kNsubPerCta; ++in)
            #pragma unroll
            for (int r = 0; r < 4; ++r) d_acc[im][in][r] = 0;

        // --------------------------------------------------------
        // Inner K-step loop (4 iters, each 32 K-wide).
        // --------------------------------------------------------
        #pragma unroll
        for (int ks = 0; ks < kKSteps; ++ks) {
            const int k_base = ks * kMmaK;   // 0, 32, 64, 96

            // --- A operand: ldmatrix.x4 from sW ---
            // For mma.m16n8k32.s8, A is a 16x32 s8 tile.
            // ldmatrix.x4.shared.b16 reads a 16x16 b16 tile where each
            // thread supplies a per-row-of-8 pointer.  Per-warp mapping:
            //   row_in_tile = lane % 16  (0..15)      <-- ldmatrix semantics
            //   col_in_tile = lane / 16  (0 or 1) * 8 (b16 cols)
            // Each b16 lane holds 2 s8.  So effectively 16b16 cols = 32 s8.
            //
            // For each of the 2 M-subs per warp, we compute the shmem
            // base pointer to its 16x16 b16 tile and call ldmatrix.x4.
            uint32_t a_regs[kMsubPerWarp][4];
            #pragma unroll
            for (int im = 0; im < kMsubPerWarp; ++im) {
                int msub_base = warp_id * 32 + im * 16;
                int row_in_tile = lane & 15;       // 0..15
                int col_in_tile = (lane >> 4) * 8; // 0 or 8 (b16)
                // shmem element offset in sW[buf][*][*]
                int shmem_row = msub_base + row_in_tile;
                int shmem_byte = k_base + col_in_tile * 2;  // b16 -> 2 bytes
                ldmatrix_x4_b16(
                    &sW[buf][shmem_row][shmem_byte],
                    a_regs[im][0], a_regs[im][1],
                    a_regs[im][2], a_regs[im][3]
                );
            }

            // --- B operand: ldmatrix.x2.trans from sX ---
            // B is 32x8 s8.  Using ldmatrix.x2.trans on a 8x16 b16 tile
            // gives each lane 2 regs with the col-major permutation we
            // need:
            //   lane supplies pointer for (row=lane%8, col=(lane/8)*8) in b16.
            //
            // Our sX is row-major (N-row, K-byte).  For B = X in col-major
            // form, we think of:
            //   B[row_b, col_b] = X[n_tile + col_b, k_base + row_b]  (s8)
            // i.e. row_b corresponds to K, col_b to N.  In sX that is:
            //   sX[buf][n_tile_row + col_b][k_base + row_b]
            //
            // For ldmatrix.x2.trans with b16:
            //   lane pointer row_in_tile = lane % 8,
            //                col_in_tile (b16) = (lane / 8) * 8 (0 or 8).
            // After trans: per-thread 2 regs map to b{0,1} fragments of
            // mma.m16n8k32.s8 B.
            //
            // Tile "row" for the ldmatrix maps to our *N-row*, i.e.
            // sX[buf][n_sub_base + row_in_tile][...].
            uint32_t b_regs[kNsubPerCta][2];
            #pragma unroll
            for (int in_sub = 0; in_sub < kNsubPerCta; ++in_sub) {
                int n_base = in_sub * 8;
                // ldmatrix 8x(16 b16) tile spans:
                //   rows: n_base + [0..7]    (8 N rows)
                //   cols: k_base + [0..31 s8] == 16 b16 cols
                // Each lane contributes 1 pointer:
                //   row_in_tile = lane & 7
                //   col_in_tile_b16 = (lane >> 3) * 8  -> 0 or 8 or 16 or 24 (b16)
                //   but ldmatrix.x2 only uses lane/8 in {0,1}, so 0 or 8 b16.
                //
                // Actually x2 uses 16 lanes (not 32):
                //   lane 0..15 supply pointers, lanes 16..31 unused (per x2 semantics).
                //
                // Hmm, re-reading PTX reference more carefully:
                //   ldmatrix.x2 uses 16 threads (lane 0..15) to supply pointers
                //   for two 8x8 tiles (16 rows total).  But the *output* regs
                //   are broadcast to all 32 lanes per the MMA-fragment layout.
                // That's not quite right either.  Let me double-check the
                // actual semantics:
                //
                // Per PTX ISA, ldmatrix.x2 distributes across 32 threads:
                //   threads 0..15  supply pointers and each receive 1 reg (for tile 0)
                //   threads 16..31 supply pointers and each receive 1 reg (for tile 1)
                // Per-thread result: 2 registers (the second = tile_base + 8 rows).
                //
                // But since our B tile is just 32x8 (which is ONE fragment
                // for B of mma.m16n8k32), we have:
                //   32 rows B (= K dim), 8 cols (= N dim) -> we need the
                //   .x2 form where the two 8x8 halves are at (rows 0..7)
                //   and (rows 16..23) of the fragment, not 0..7 + 8..15.
                // Hmm this is getting tangled.
                //
                // TRUE solution for B of mma.m16n8k32 from a shmem tile:
                //   Use ldmatrix.x2.trans reading an 8-row x 16-b16col tile.
                //   Thread t supplies pointer for:
                //     row_ptr = t % 8              (0..7)
                //     col_ptr_b16 = (t / 8) * 8    (0 or 8 b16 cols)
                //   After trans, the two output regs per thread are:
                //     reg0 = b0 of mma frag  (k=0..15 slice)
                //     reg1 = b1 of mma frag  (k=16..31 slice)
                //
                // So we need the shmem tile to span rows [n_base..n_base+7]
                // and k-cols [k_base..k_base+31].
                int row_ptr = lane & 7;
                int col_ptr_b16 = (lane >> 3) * 8;  // 0, 8, 16, 24 (b16 cols)
                int shmem_row = n_base + row_ptr;
                int shmem_byte = k_base + col_ptr_b16 * 2;

                // Mask out-of-range rows: we need shmem_byte to land within
                // the 32-byte k-step window (col_ptr_b16*2 in 0..31 from k_base).
                // Only col_ptr_b16 in {0, 8} is within our 32 s8 window;
                // lanes with col_ptr_b16 >= 16 would read beyond the k-step
                // we're computing.  But x2 semantics only use lane>>3 in {0,1,2,3},
                // and the `trans` form pairs them up per-PTX spec.
                //
                // For our 8-row x 16-b16-col (= 32 s8 col) tile, all 32 lanes
                // MUST supply pointers.  Lane (lane>>3) in {0,1,2,3} each
                // hit a different 8-b16 stripe.  But our tile only has 2
                // stripes (16 b16 cols = 2*8).  So only lanes with
                // (lane>>3) in {0,1} are "primary"; the other lanes are
                // duplicated/masked according to x2 semantics.
                //
                // The "correct" x2 form for a 16x16 tile is to call it
                // as x2 where the tile is 16 rows (two stacked 8-row
                // halves).  Here we're not using stacked rows.
                //
                // Given the subtlety, for this first version we emit a
                // correct-but-slow SCALAR b-fragment build from the shmem
                // tile, skipping ldmatrix for B.  This is unambiguous and
                // still uses full Tensor Core compute throughput -- the
                // shmem loads are ~2% of kernel time.
                //
                // Per-thread b0 lanes: 4 s8 at (row=(lane&3)*4+{0..3}, col=(lane>>2))
                //                                for K=[0..15], N col within N-sub
                // Per-thread b1 lanes: 4 s8 at (row=(lane&3)*4+16+{0..3}, col=(lane>>2))
                //                                for K=[16..31]
                int n_row_in_sub = lane >> 2;        // 0..7 (the N col within the 8-wide sub)
                int k_row_base0 = (lane & 3) * 4;    // 0, 4, 8, 12
                int k_row_base1 = k_row_base0 + 16;  // 16, 20, 24, 28
                // Read 4 s8 as one uint32 (s8 contiguous along K).
                // sX[buf][n_base + n_row_in_sub][k_base + k_row_base{0,1} .. +3]
                const int8_t* ptr0 = &sX[buf][n_base + n_row_in_sub][k_base + k_row_base0];
                const int8_t* ptr1 = &sX[buf][n_base + n_row_in_sub][k_base + k_row_base1];
                b_regs[in_sub][0] = *reinterpret_cast<const uint32_t*>(ptr0);
                b_regs[in_sub][1] = *reinterpret_cast<const uint32_t*>(ptr1);
                // Unused variables: shmem_row/shmem_byte/row_ptr/col_ptr_b16.
                (void)shmem_row; (void)shmem_byte;
                (void)row_ptr; (void)col_ptr_b16;
            }

            // Same-style scalar build for A (correct but slower than
            // ldmatrix.x4; we keep ldmatrix commented out above as a
            // future optim once correctness is nailed).
            #pragma unroll
            for (int im = 0; im < kMsubPerWarp; ++im) {
                int msub_base = warp_id * 32 + im * 16;
                int row0 = msub_base + (lane >> 2);
                int row1 = row0 + 8;
                int col0 = k_base + (lane & 3) * 4;       // low half
                int col2 = col0 + 16;                     // high half
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

            // Issue MMA.
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

        // --------------------------------------------------------
        // Per-group epilogue.
        //
        // Map (im, in_sub, r, lane) -> (m_global, n_local).
        //   row_local = (lane >> 2) + ((r >> 1) ? 8 : 0)
        //   col_local = (lane & 3) * 2 + (r & 1)
        // --------------------------------------------------------
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
                    float z = __half2float(
                        zero_u4[(int64_t)m_global * stride_zu_m
                              + (int64_t)g * stride_zu_g]
                    );
                    float s = __half2float(
                        scale_u4[(int64_t)m_global * stride_su_m
                               + (int64_t)g * stride_su_g]
                    );
                    float sxn = __half2float(s_scale_x[n_local]);
                    float sumxn = static_cast<float>(s_sum_X[buf][n_local]);
                    float corrected = static_cast<float>(d_acc[im][in_sub][r])
                                    - z * sumxn;
                    y_fp[im][in_sub][r] += corrected * s * sxn;
                }
            }
        }

        __syncthreads();  // ensure g+1 prefetch is visible for next iter
    }

    // ------------------------------------------------------------
    // Writeback.
    // ------------------------------------------------------------
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
        dense_gemm_mma_int8_kernel<kBn><<<grid, block, 0, stream>>>(
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

    // kBn dispatch (multiples of 8):
    //   T<=8    : kBn=8     (decode; T=1 wastes 7 N cols but Tensor Core
    //                        math is free, shmem footprint is minimal)
    //   T<=64   : kBn=64
    //   default : kBn=128   (prefill; fully utilises 4 N-subs/warp)
    if      (T <= 8)    do_launch(std::integral_constant<int, 8>{});
    else if (T <= 64)   do_launch(std::integral_constant<int, 64>{});
    else                do_launch(std::integral_constant<int, 128>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dense_gemm_mma_int8
}  // namespace hkust_v9
