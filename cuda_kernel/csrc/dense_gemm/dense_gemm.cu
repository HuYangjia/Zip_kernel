// Dense UINT4 x SINT4 GEMM (CUDA, SM89).
//
// Drop-in replacement for
// ``kernel.triton_kernel.dense_u4s4_gemm.dense_gemm_kernel``.
//
// Contract (bit-accurate to the Triton reference within FP32 accumulator
// rounding; the final FP16 cast can differ from Triton by <= 1 ULP on
// pathological rounding ties, which we accept):
//   Y[m, n] = sum over groups g of
//               (W_low[m, g*BCOL:(g+1)*BCOL] . X_s4[n, g*BCOL:(g+1)*BCOL]
//                - zero_u4[m, g] * sum_X[n, g])
//             * scale_u4[m, g] * scale_x[n]
// All dot products are SINT4 x SINT4 -> INT32 (the UINT4 -> SINT4 offset
// is folded into zero_u4 offline; see pack_utils.pack_v9_weights).
// Output is (d_out, T) row-major FP16.
//
// Strategy
// --------
// This implementation uses SIMT __dp4a instructions rather than the
// m16n8k32.s8.s8.s32 MMA.  Rationale (see research notes, Phase 2
// scope):
//   - decode (T <= 64) is the dominant path.  At T=1 the TC tile is
//     128x16x32 and only 1/16 of N is useful, so ~93% of MMA FLOPS are
//     wasted.  SIMT dp4a with BN=1..16 keeps 100% utilization.
//   - prefill (T >= 256) the Triton kernel already achieves respectable
//     TC throughput; we cede 5-15% peak to the Triton path in that
//     regime (policy.py routes T >= 1024 to Triton) and take the
//     correctness/launch-overhead win at moderate sizes.
//
// Tile parameters
// ---------------
// - BM = 128 (match BROW; dense has no block-row dependence but we
//            keep BM=BROW so the same constants cover sparse / fused).
// - BN in {16, 128} selected at launch time by T.
// - BK = 128 (= BCOL, one full group per K iteration).  This matches
//   the Triton kernel's per-group epilogue: after one K iter we have
//   a complete group contribution and can fold (zero*sum_X, scale).

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <type_traits>

namespace hkust_v9 {
namespace dense_gemm {

// ---------------------------------------------------------------------------
// Unpack helpers: load 16 packed bytes (==32 SINT4 values) into 8 int32
// lanes of 4-int8-per-lane, ready for __dp4a.
//
// packed[i] bit layout:   bits[7:4] = col (2i+1)   (high nibble)
//                         bits[3:0] = col (2i  )   (low  nibble)
// After unpack, 4 consecutive output int8s lanes hold cols 2i..2i+3 of K.
// __dp4a accumulates 4 int8*int8 products per instruction, so a 32-wide
// K-stripe needs 8 dp4a invocations per (M, N) entry.
// ---------------------------------------------------------------------------

// Sign-extend a 4-bit value in the low nibble of a byte into int32.
__device__ __forceinline__ int8_t s4_lo(uint8_t b) {
    int v = b & 0x0F;
    return static_cast<int8_t>(v - ((v & 0x08) << 1));  // branch-free sign ext
}
__device__ __forceinline__ int8_t s4_hi(uint8_t b) {
    int v = (b >> 4) & 0x0F;
    return static_cast<int8_t>(v - ((v & 0x08) << 1));
}

// Unpack 4 packed bytes (== 8 SINT4 values) into 2 packed int32 lanes
// (each holding 4 int8s in little-endian byte order, ready for dp4a).
// byte[0] = cols (0,1), byte[1] = cols (2,3), ...
// dp4a wants a.int8[0]*b.int8[0] + a.int8[1]*b.int8[1] + ... so we pack
// cols 0..3 into one int32 and cols 4..7 into another.
__device__ __forceinline__ void unpack_4bytes_to_2int32(
    uint32_t packed4,     // 4 bytes -> 8 SINT4 values (cols 0..7)
    int& out0,            // cols 0..3 as int8 lanes
    int& out1             // cols 4..7 as int8 lanes
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
// Device helpers
// ---------------------------------------------------------------------------

// Compute dot product over one BCOL=128-wide group between W_low row m
// and X_s4 row n.  Both are laid out as 64 packed int8 bytes (128 SINT4
// values).  Returns int32 accumulator.
__device__ __forceinline__ int dot_group_128(
    const uint8_t* __restrict__ w_row,     // 64 bytes
    const uint8_t* __restrict__ x_row      // 64 bytes
) {
    int acc = 0;
    // 64 bytes = 16 uint32 = 16 iterations of 4-byte-unpack-and-dp4a-x2.
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint32_t wp = reinterpret_cast<const uint32_t*>(w_row)[i];
        uint32_t xp = reinterpret_cast<const uint32_t*>(x_row)[i];
        int w0, w1, x0, x1;
        unpack_4bytes_to_2int32(wp, w0, w1);
        unpack_4bytes_to_2int32(xp, x0, x1);
        acc = __dp4a(w0, x0, acc);
        acc = __dp4a(w1, x1, acc);
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
//
// Thread layout
// -------------
// - Block = 128 threads (4 warps).  One thread per row of the output
//   (BM dim); BM is pinned to 128 so every thread owns exactly one M.
// - Each thread computes kBn output entries Y[m, n0 .. n0+kBn-1] for
//   its m, accumulating in per-thread registers.
// - X rows for the kBn columns of this tile are staged to shmem once
//   per K iteration so 128 threads reuse them.  W rows are read
//   directly from HBM via __ldg; each W byte is consumed exactly once
//   per tile (no M reuse to justify shmem), and we rely on L1 for any
//   accidental reuse across neighbouring blocks.
//
// Grid
// ----
// - grid.x = ceil_div(d_out, 128)
// - grid.y = ceil_div(T, kBn)
//
// Note: kBn is a template parameter, not a runtime value, so the
// per-thread accumulator array and the shmem footprint are both
// compile-time-sized.  The launcher instantiates a small set of kBn
// values (1, 8, 16, 64, 128) covering the T range.

template <int kBn>
__global__ void dense_gemm_kernel(
    const uint8_t* __restrict__ W,         // (d_out, d_in/2) byte
    const uint8_t* __restrict__ X,         // (T, d_in/2) byte
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
    const int tid = threadIdx.x;               // 0..127
    const int m = blockIdx.x * kBm + tid;
    const bool m_active = m < d_out;

    // Base N index of this tile's columns.
    const int n_base = blockIdx.y * kBn;

    const int bytes_per_group = BCOL >> 1;     // 64
    const int d_in_half = d_in >> 1;

    // -----------------------------------------------------------------
    // Hoist per-N data: scale_x[n], and active mask.
    // Each thread needs *all kBn* scale_x values to form the epilogue,
    // so we stage them in shmem (kBn is up to 128 fp16 = 256 bytes).
    // -----------------------------------------------------------------
    __shared__ __half s_scale_x[kBn];
    if (tid < kBn) {
        int n = n_base + tid;
        // scale_x is a 1D contiguous tensor of shape (T,); its
        // stride is hard-coded to 1 (enforced by the launcher).  We
        // do *not* reuse stride_sx_n here: that stride belongs to
        // the 2D sum_X tensor and is n_groups in general, which
        // would read out-of-bounds for every tid >= 1.
        s_scale_x[tid] = (n < T) ? scale_x[n] : __half(0);
    }

    // Per-thread accumulator array: kBn FP32 slots for Y[m, n_base..].
    float y_acc[kBn];
    #pragma unroll
    for (int k = 0; k < kBn; ++k) y_acc[k] = 0.0f;

    // -----------------------------------------------------------------
    // Shmem: DOUBLE-BUFFERED X tile so we can overlap the next group's
    // global load with the current group's compute (Round 6 optim).
    // Two (kBn, 64) banks; at kBn<=4 this is at most 512 bytes total,
    // negligible.  cp.async writes directly to shmem without occupying
    // registers or stalling the compute pipeline.
    //
    // IMPORTANT: cp.async.cg requires the shmem destination to be
    // 16-byte aligned.  Shared memory arrays default to element-
    // alignment only, so we force alignas(16) explicitly.
    // -----------------------------------------------------------------
    __shared__ alignas(16) uint8_t sX[2][kBn][64];

    // Also stage the per-group sum_X[n] for all kBn columns.  Double-
    // buffered too so it stays in lock-step with sX.
    __shared__ int s_sum_X[2][kBn];

    __syncthreads();  // ensure s_scale_x write from tid<kBn is visible

    // -----------------------------------------------------------------
    // Helper lambda: issue all cp.async loads for group g into buffer
    // sX[buf].  Each row (64 bytes) is 4 uint4 == 4 cp.async 16-byte
    // transactions.  Thread t handles (row, quad) = (t >> 2, t & 3)
    // for t in [0 .. kBn*4).  Remaining threads idle (kBn*4 <= 16 for
    // our supported kBn, so only a fraction of the 128-thread block
    // participates -- but cp.async is async so this is fine).
    // -----------------------------------------------------------------
    auto issue_x_load = [&](int g, int buf) {
        const int total_quads = kBn * 4;
        if (tid < total_quads) {
            int row = tid >> 2;
            int quad = tid & 3;
            int n = n_base + row;
            if (n < T) {
                int64_t off = (int64_t)n * stride_x_n
                            + (int64_t)(g * bytes_per_group + quad * 16) * stride_x_k;
                cp_async_cg_16(
                    &sX[buf][row][quad * 16],
                    X + off
                );
            } else {
                // Tail row past T: zero-fill via plain shmem writes.
                // cp.async can't zero for out-of-bounds, and we rely on
                // acc_n's epilogue mask to skip these anyway, but we
                // still clear to avoid reading uninitialised bytes
                // (which would be UB under compute sanitisers).
                uint4 zero = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sX[buf][row][quad * 16]) = zero;
            }
        }
        // Also prime sum_X for this group (one scalar per N row).
        if (tid < kBn) {
            int n = n_base + tid;
            s_sum_X[buf][tid] = (n < T) ?
                sum_X[(int64_t)n * stride_sx_n + (int64_t)g * stride_sx_g] : 0;
        }
    };

    // ---- Prologue: kick off group 0 ----------------------------------
    issue_x_load(0, 0);
    cp_async_commit();

    for (int g = 0; g < n_groups; ++g) {
        const int buf = g & 1;

        // Kick off g+1 into the other buffer (if any) before waiting on g.
        if (g + 1 < n_groups) {
            issue_x_load(g + 1, buf ^ 1);
            cp_async_commit();
            cp_async_wait_group<1>();   // group g is ready, g+1 in flight
        } else {
            cp_async_wait_group<0>();   // last group: drain everything
        }
        __syncthreads();

        if (m_active) {
            // Load W row slice for this (m, g) once, reuse across kBn N entries.
            // 64 bytes = 4 uint4 = 16 uint32.  Using 128-bit (uint4) loads
            // instead of 32-bit halves the number of memory transactions
            // issued, reducing stalls against the L2/HBM pipeline (Round 4
            // optimisation).  SM89 coalesces these across 32 warp lanes
            // to 4 cache-line-sized transactions per warp per group.
            uint32_t w_words[16];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int64_t off_w = (int64_t)m * stride_w_m
                              + (int64_t)(g * bytes_per_group + i * 16) * stride_w_k;
                // uint4 == 4 x uint32, 16-byte aligned when m * stride_w_m
                // is 16-byte aligned.  W is packed uint8 with row stride
                // == d_in_half, so the address is always 16-byte aligned
                // as long as d_in_half is a multiple of 16 (== d_in is
                // a multiple of 32, which we enforce since BCOL=128 divides
                // d_in).
                uint4 v = __ldg(
                    reinterpret_cast<const uint4*>(W + off_w)
                );
                w_words[4*i    ] = v.x;
                w_words[4*i + 1] = v.y;
                w_words[4*i + 2] = v.z;
                w_words[4*i + 3] = v.w;
            }

            // Pre-unpack W row into 32 int32 lanes (holding 128 SINT4 as
            // 4-per-lane dp4a-ready words).  This keeps the inner N loop
            // to just 8 dp4a issues per N column.
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

            // ----------------------------------------------------------
            // ILP-friendly inner loop (iter-Round 2 optimisation).
            //
            // Previous layout iterated N on the outside and K on the
            // inside:
            //
            //     for nk in 0..kBn:
            //         acc = 0
            //         for i in 0..16:
            //             acc = dp4a(w[2i  ], x[2i  ], acc)
            //             acc = dp4a(w[2i+1], x[2i+1], acc)
            //
            // The 32 dp4a instructions per nk formed a single WAW/RAW
            // dependency chain on ``acc``, so the scheduler could issue
            // only ~1 dp4a per 4-6 cycles (dp4a latency) per thread.
            // Unrolling the N axis didn't help because nvcc wouldn't
            // predict that acc_nk are independent (the chain was hidden
            // inside the unroll).
            //
            // New layout: swap loops so K is outside, N inside.  Each
            // i-iteration issues kBn *independent* dp4a chains (one
            // per N accumulator), giving nvcc a wide pool of ILP-able
            // instructions.  Since the compile-time kBn is up to ~64
            // and the scheduler only needs ~8 in flight to cover dp4a
            // latency, this halves or better the K-loop's wall time.
            //
            // Staged the X words to registers first (one per-thread
            // register per (nk, i/2)) so shmem bank-conflict risk is
            // eliminated.
            // ----------------------------------------------------------

            // Per-N accumulators, already zero-inited as part of y_acc's
            // per-iteration lifetime; we reconstruct them each group
            // because we fold zero*sum_X per-group.
            int acc_n[kBn];
            #pragma unroll
            for (int nk = 0; nk < kBn; ++nk) acc_n[nk] = 0;

            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                // Stage the i-th word of each N row into registers.
                int x0_n[kBn], x1_n[kBn];
                #pragma unroll
                for (int nk = 0; nk < kBn; ++nk) {
                    uint32_t xp = reinterpret_cast<const uint32_t*>(&sX[buf][nk][0])[i];
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

            // Epilogue: apply per-group (zero, scale) correction.
            #pragma unroll
            for (int nk = 0; nk < kBn; ++nk) {
                int n = n_base + nk;
                if (n >= T) break;
                float corrected = static_cast<float>(acc_n[nk])
                                - zero_g * static_cast<float>(s_sum_X[buf][nk]);
                float sxn = __half2float(s_scale_x[nk]);
                y_acc[nk] += corrected * scale_g * sxn;
            }
        }

        __syncthreads();
    }

    // Write back Y[m, n_base..n_base+kBn).
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
    // 32-bit packed W / X loads require contiguous last dim.  Triton
    // caller always passes .contiguous() but we check explicitly.
    TORCH_CHECK(W_low.stride(1) == 1, "W_low must be K-contiguous");
    TORCH_CHECK(X_s4.stride(1) == 1, "X_s4 must be K-contiguous");
    // Kernel hard-codes scale_x stride=1 (see s_scale_x staging).
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
        dense_gemm_kernel<kBn><<<grid, block, 0, stream>>>(
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

    // kBn dispatch table (iter-Round 3, calibrated against
    // bench_20260424_131009).  Round 2's loop swap introduced kBn
    // registers for acc_n + 2*kBn registers for x0_n/x1_n; at kBn>=8
    // that spills to local memory and wall time explodes.  We now
    // keep kBn <= 4 always and move N-parallelism onto the grid (which
    // SM89 has plenty of headroom for: 64 warps/SM * 2 blocks/SM /
    // (d_out/128) CTAs available).  The empirical optimum is:
    //    T=1  -> kBn=1 (fewest regs, highest occupancy)
    //    T<=8 -> kBn=2 (still well within budget, doubles grid)
    //    else -> kBn=4 (caps spill; grid still covers the SM)
    if      (T <= 1)   do_launch(std::integral_constant<int, 1>{});
    else if (T <= 16)  do_launch(std::integral_constant<int, 2>{});
    else               do_launch(std::integral_constant<int, 4>{});

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace dense_gemm
}  // namespace hkust_v9
