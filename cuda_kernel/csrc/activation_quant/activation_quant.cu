// Fused per-token SINT4 activation quantization (CUDA, SM89).
//
// Drop-in replacement for
// ``kernel.triton_kernel.activation_quant.quantize_activation_kernel``.
//
// Contract (must be bit-exact with the Triton reference):
//   scale_fp32 = max(|X[t, perm[:]]|) / 7
//   scale_fp16 = fp16(scale_fp32)              // single fp16 rounding
//   scale_math = fp32(scale_fp16)              // back to fp32 for math
//   q = clamp(rint(x / scale_math), -8, 7)     // IEEE round-half-to-even
//   sum_X[t, g] = sum_{k in group_g} q[t, k]
//   X_s4[t, j]  = (q[t, 2j+1] << 4) | (q[t, 2j] & 0x0F)  // LE pack
//
// Design notes (SM89-specific):
// - One CTA handles ``kBt`` tokens in parallel.  kBt is a template
//   parameter so the host can pick small (decode) or medium (prefill)
//   variants without autotune dispatch cost.
// - Pass 1 (max-abs) and Pass 2 (quant+pack+sum) share one CTA-local
//   reduction shim; we do *not* split kernels because the work per
//   token is small enough that a second kernel launch dominates.
// - Per-group reduction in Pass 2 uses warp shuffles for the BCOL=128
//   case (4 warps * 32 lanes = 128 lanes == one group).  For other
//   BCOL we fall back to shared-memory reduction.
// - Gather over ``perm`` is done with ``__ldg`` (read-only cache).
//   For decode T=1 a warp loads 32 fp16 elements per iteration via
//   32 independent 16-bit gathers -- HBM bound regardless of
//   coalescing, so no pre-permute is needed.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace cg = cooperative_groups;

namespace hkust_v9 {
namespace activation_quant {

// Tunable tile parameters.  Only kBt varies with T; kNLanesPerGroup is
// locked to BCOL (== 128) because the sum-per-group contract ties the
// lane partition to the group boundary.
static_assert(BCOL == 128, "activation_quant kernel assumes BCOL == 128");
constexpr int kLanesPerGroup = BCOL;        // 128 lanes == 4 warps work one group

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Warp-level max-abs reduction over ``kWarpSize`` fp32 values.
__device__ __forceinline__ float warp_max_abs(float v) {
    v = fabsf(v);
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

// Warp-level sum reduction (int32).
__device__ __forceinline__ int warp_sum(int v) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// IEEE round-half-to-even for fp32 -> int32, matching Triton's
// ``tl.libdevice.rint`` + cast.  Bit-exact match requires reproducing
// Triton's fp32 division behaviour -- Triton lowers ``x / s`` to PTX
// ``div.approx.f32`` (single-precision, ~0.5 ULP via Newton-Raphson),
// which is NOT what nvcc emits for plain C++ ``x / s`` (that path
// picks ``div.rn.f32``, a fully-rounded IEEE div).  The two can
// disagree by 1 ULP on the dividend, which shifts results across
// ``rint``'s half-integer boundary and changes ~5-20 packed nibbles
// per 256K quantized values.  We therefore use ``__fdividef`` to
// force the approx-div path and get bytes-identical packed output.
__device__ __forceinline__ int quantize_one(float x, float scale_safe,
                                             bool scale_is_zero) {
    if (scale_is_zero) return 0;
    float q = rintf(__fdividef(x, scale_safe));
    q = fmaxf(fminf(q, 7.0f), -8.0f);
    return static_cast<int>(q);
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
//
// Grid  : (ceil_div(T, kBt), 1, 1)
// Block : (kLanesPerGroup, kBt, 1)    // 128 x kBt threads
//
// Each CTA processes exactly ``kBt`` tokens.  Along the second axis the
// threadIdx.y selects the token within the CTA; threadIdx.x tiles the
// d_in dimension with stride kLanesPerGroup.
//
// Pass 1 (max-abs):  each (y,x) lane walks X[t, perm[x::kLanesPerGroup]]
//   and computes its local max-abs.  A 128-wide reduction along x gives
//   the per-token max-abs, stored to shared memory.
//
// Pass 2 (quant/pack/sum):  walks d_in in strides of kLanesPerGroup,
//   where one 128-wide stride is exactly one group.  Per-group sum
//   uses the same 128-wide reduction.  Packing writes one byte per
//   two adjacent lanes (even lane = low nibble, odd lane = high
//   nibble), coordinated via a warp-level shuffle within each warp.

template <int kBt>
__global__ void activation_quant_kernel(
    const __half* __restrict__ X,         // (T, D) fp16
    const int* __restrict__ perm,         // (D,) int32
    int8_t* __restrict__ X_s4,            // (T, D/2) int8, LE packed
    __half* __restrict__ scale_x,         // (T,) fp16
    int* __restrict__ sum_X,              // (T, D/BCOL) int32
    int T, int D,
    int64_t stride_xt,                    // X stride along T (elements)
    int64_t stride_xd,                    // X stride along D (elements)
    int64_t stride_qt,                    // X_s4 stride along T (elements, int8 == 1 byte)
    int64_t stride_qd,                    // X_s4 stride along D/2 (elements, int8 == 1 byte)
    int64_t stride_st,                    // sum_X stride along T (elements)
    int64_t stride_sg                     // sum_X stride along groups (elements)
) {
    static_assert(kLanesPerGroup == 128, "kernel hard-codes 128-wide lane group");

    const int lane_x = threadIdx.x;            // 0..127
    const int lane_y = threadIdx.y;            // 0..kBt-1
    const int warp_id_x = lane_x / kWarpSize;  // 0..3
    const int lane_in_warp = lane_x & (kWarpSize - 1);

    const int t_base = blockIdx.x * kBt;
    const int t = t_base + lane_y;
    const bool t_active = t < T;

    // Shared-memory scratch: per-token scale / flags plus the 4-warp
    // reduction staging area.  Sized conservatively for kBt<=64.
    __shared__ float s_max_abs[kBt];
    __shared__ float s_scale_math[kBt];        // fp32 scale post-fp16-rounding
    __shared__ unsigned char s_scale_zero[kBt];

    // Staging slots for the 128-wide reduction (4 warp-partials per token).
    __shared__ float s_warp_part[kBt][4];
    __shared__ int   s_warp_part_int[kBt][4];

    // -----------------------------------------------------------------
    // Pass 1: max-abs per token over permuted columns.
    // -----------------------------------------------------------------
    float local_max = 0.0f;
    // Stride across D in units of kLanesPerGroup.
    for (int d = lane_x; d < D; d += kLanesPerGroup) {
        int pidx = __ldg(perm + d);
        __half h = __half(0);
        if (t_active) {
            h = __ldg(X + t * stride_xt + pidx * stride_xd);
        }
        float v = __half2float(h);
        local_max = fmaxf(local_max, fabsf(v));
    }
    // 32-wide warp reduction, then 4-warp reduction via shared memory.
    float warp_max = warp_max_abs(local_max);
    if (lane_in_warp == 0) {
        s_warp_part[lane_y][warp_id_x] = warp_max;
    }
    __syncthreads();

    // Lane 0..3 in warp 0 finalize: pick max of 4 warp partials.
    float token_max = 0.0f;
    if (warp_id_x == 0 && lane_in_warp < 4) {
        token_max = s_warp_part[lane_y][lane_in_warp];
    }
    // Reduce across the first 4 lanes of warp 0.  We cannot use
    // ``__shfl_xor_sync`` with a partial mask cleanly; do it in two
    // sequential shuffles restricted to lanes 0..3.
    if (warp_id_x == 0) {
        float o2 = __shfl_xor_sync(0xffffffff, token_max, 2);
        if (lane_in_warp < 4) token_max = fmaxf(token_max, o2);
        float o1 = __shfl_xor_sync(0xffffffff, token_max, 1);
        if (lane_in_warp < 4) token_max = fmaxf(token_max, o1);
    }

    if (warp_id_x == 0 && lane_in_warp == 0) {
        // Apply the Triton contract:
        //   scale_fp32 = max / 7
        //   scale_fp16 = fp16(scale_fp32)
        //   scale_math = fp32(scale_fp16)
        float scale_fp32 = token_max / 7.0f;
        __half scale_h = __float2half(scale_fp32);
        float scale_math = __half2float(scale_h);
        bool is_zero = !(scale_math > 0.0f);
        s_max_abs[lane_y] = token_max;
        s_scale_math[lane_y] = is_zero ? 1.0f : scale_math;
        s_scale_zero[lane_y] = is_zero ? 1 : 0;
        if (t_active) {
            scale_x[t] = scale_h;
        }
    }
    __syncthreads();

    // -----------------------------------------------------------------
    // Pass 2: quantize, pack LE, reduce per-group sum.
    // -----------------------------------------------------------------
    const float scale_math = s_scale_math[lane_y];
    const bool  is_zero = s_scale_zero[lane_y] != 0;

    const int n_groups = D / BCOL;
    for (int g = 0; g < n_groups; ++g) {
        const int d = g * BCOL + lane_x;
        int pidx = __ldg(perm + d);
        __half h = __half(0);
        if (t_active) {
            h = __ldg(X + t * stride_xt + pidx * stride_xd);
        }
        float x = __half2float(h);
        int q = quantize_one(x, scale_math, is_zero);

        // Per-group sum: 128-wide reduction over lane_x.
        int wsum = warp_sum(q);
        if (lane_in_warp == 0) {
            s_warp_part_int[lane_y][warp_id_x] = wsum;
        }
        __syncthreads();
        if (warp_id_x == 0 && lane_in_warp == 0) {
            int total = s_warp_part_int[lane_y][0] + s_warp_part_int[lane_y][1]
                      + s_warp_part_int[lane_y][2] + s_warp_part_int[lane_y][3];
            if (t_active) {
                sum_X[t * stride_st + g * stride_sg] = total;
            }
        }

        // Pack LE: even lane contributes low nibble, odd lane contributes
        // high nibble of the same byte.  Use a single-step warp shuffle
        // to fetch the neighbour's quantized value.
        int q_neighbour = __shfl_xor_sync(0xffffffff, q, 1);
        if (t_active && (lane_x & 1) == 0) {
            int low = q & 0x0F;
            int high = q_neighbour & 0x0F;
            int packed = (high << 4) | low;
            // Byte offset = (d / 2) since d is even here.
            int64_t byte_off = (int64_t)t * stride_qt
                             + (int64_t)(d >> 1) * stride_qd;
            X_s4[byte_off] = static_cast<int8_t>(
                packed >= 128 ? packed - 256 : packed
            );
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Single-pass kernel (Round 15 optimisation, T <= 8)
// ---------------------------------------------------------------------------
//
// Rationale: the 2-pass kernel above gathers X[perm[.]] twice (once for
// max-abs, once for quant+pack), which doubles HBM traffic on the
// gather path.  For T=1 decode on 4k->4k this dominates runtime.
//
// The single-pass variant caches the gathered X into shared memory
// during Pass 1 and reuses it in Pass 2, cutting gather traffic by 2x.
//
// Shape: block = (kLanesPerGroup=128, kBt), grid = (ceil_div(T, kBt),).
// Shmem: kBt * D fp16 values + small staging area.  D can be up to
//   ~11008 for the shapes we care about, so per-kBt footprint is
//   11008*2 = ~22 KB.  With kBt=1, total shmem ~22 KB (well under the
//   SM89 per-CTA limit).  For safety we gate on ``D <= kMaxShmemD`` at
//   launch time and fall back to the 2-pass kernel otherwise.

constexpr int kMaxShmemDPerToken = 16384;   // 32 KB per token (fp16)

template <int kBt>
__global__ void activation_quant_kernel_sp(
    const __half* __restrict__ X,
    const int* __restrict__ perm,
    int8_t* __restrict__ X_s4,
    __half* __restrict__ scale_x,
    int* __restrict__ sum_X,
    int T, int D,
    int64_t stride_xt, int64_t stride_xd,
    int64_t stride_qt, int64_t stride_qd,
    int64_t stride_st, int64_t stride_sg
) {
    static_assert(kLanesPerGroup == 128, "sp kernel hard-codes 128-wide lane group");

    extern __shared__ unsigned char s_raw[];
    // Layout: s_x[kBt][D] fp16,  s_warp_part[kBt][4] float,
    //         s_warp_part_int[kBt][4] int, s_scale_math[kBt] float,
    //         s_scale_zero[kBt] uchar.
    __half* s_x = reinterpret_cast<__half*>(s_raw);
    float*  s_warp_part     = reinterpret_cast<float*>(s_x + kBt * D);
    int*    s_warp_part_int = reinterpret_cast<int*>(s_warp_part + kBt * 4);
    float*  s_scale_math    = reinterpret_cast<float*>(s_warp_part_int + kBt * 4);
    unsigned char* s_scale_zero = reinterpret_cast<unsigned char*>(s_scale_math + kBt);

    const int lane_x = threadIdx.x;
    const int lane_y = threadIdx.y;
    const int warp_id_x = lane_x / kWarpSize;
    const int lane_in_warp = lane_x & (kWarpSize - 1);

    const int t_base = blockIdx.x * kBt;
    const int t = t_base + lane_y;
    const bool t_active = t < T;

    // ------------------------------------------------------------------
    // Pass 1: gather X[t, perm[d]] into shmem AND track local max-abs.
    // ------------------------------------------------------------------
    float local_max = 0.0f;
    for (int d = lane_x; d < D; d += kLanesPerGroup) {
        int pidx = __ldg(perm + d);
        __half h = __half(0);
        if (t_active) {
            h = __ldg(X + (int64_t)t * stride_xt + (int64_t)pidx * stride_xd);
        }
        s_x[lane_y * D + d] = h;
        float v = __half2float(h);
        local_max = fmaxf(local_max, fabsf(v));
    }

    float warp_max = warp_max_abs(local_max);
    if (lane_in_warp == 0) {
        s_warp_part[lane_y * 4 + warp_id_x] = warp_max;
    }
    __syncthreads();

    // Lane 0 of warp 0 finalises per token and writes scale.
    if (warp_id_x == 0 && lane_in_warp == 0) {
        float m0 = s_warp_part[lane_y * 4 + 0];
        float m1 = s_warp_part[lane_y * 4 + 1];
        float m2 = s_warp_part[lane_y * 4 + 2];
        float m3 = s_warp_part[lane_y * 4 + 3];
        float token_max = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));

        float scale_fp32 = token_max / 7.0f;
        __half scale_h = __float2half(scale_fp32);
        float scale_math = __half2float(scale_h);
        bool is_zero = !(scale_math > 0.0f);
        s_scale_math[lane_y] = is_zero ? 1.0f : scale_math;
        s_scale_zero[lane_y] = is_zero ? 1 : 0;
        if (t_active) {
            scale_x[t] = scale_h;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Pass 2: read X from shmem, quantize, pack, per-group sum.
    // ------------------------------------------------------------------
    const float scale_math = s_scale_math[lane_y];
    const bool  is_zero = s_scale_zero[lane_y] != 0;
    const int n_groups = D / BCOL;

    for (int g = 0; g < n_groups; ++g) {
        const int d = g * BCOL + lane_x;
        __half h = s_x[lane_y * D + d];
        float x = __half2float(h);
        int q = quantize_one(x, scale_math, is_zero);

        // Per-group sum: 128-wide reduction (4 warps).
        int wsum = warp_sum(q);
        if (lane_in_warp == 0) {
            s_warp_part_int[lane_y * 4 + warp_id_x] = wsum;
        }
        __syncthreads();
        if (warp_id_x == 0 && lane_in_warp == 0) {
            int total = s_warp_part_int[lane_y * 4 + 0]
                      + s_warp_part_int[lane_y * 4 + 1]
                      + s_warp_part_int[lane_y * 4 + 2]
                      + s_warp_part_int[lane_y * 4 + 3];
            if (t_active) {
                sum_X[(int64_t)t * stride_st + (int64_t)g * stride_sg] = total;
            }
        }

        // Pack LE: even lane low nibble, odd lane high nibble.
        int q_neighbour = __shfl_xor_sync(0xffffffff, q, 1);
        if (t_active && (lane_x & 1) == 0) {
            int low = q & 0x0F;
            int high = q_neighbour & 0x0F;
            int packed = (high << 4) | low;
            int64_t byte_off = (int64_t)t * stride_qt
                             + (int64_t)(d >> 1) * stride_qd;
            X_s4[byte_off] = static_cast<int8_t>(
                packed >= 128 ? packed - 256 : packed
            );
        }
        // No global sync required between groups: s_x is read-only in
        // Pass 2 and s_warp_part_int is overwritten fresh next iter.
        // But we do need lane-0 writers to finish before the next
        // iter reuses the slot, so a lightweight syncthreads is kept.
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Multi-CTA kernels (Round 33 optimisation for small T)
// ---------------------------------------------------------------------------
//
// Rationale: the single-CTA sp/2p kernels bottleneck on a single SM's
//   HBM bandwidth for T in {1, 2, 4}.  Wall time ~20 us for D=4096 is
//   dominated by the gather latency chain, NOT compute.  For T=1 only
//   1 CTA is launched; 127 SMs idle.
//
// Solution: split quant into two phases, each fully parallelised across
//   groups (grid.x) so all SMs participate in the HBM read.
//
//   Phase A (max): grid=(n_groups, T), block=(128,1).  Each CTA scans
//     its own group's 128 gathered elements, does warp-max-abs, then
//     atomicMax-into-int the per-token running max (fp32 bits, safe
//     because max is non-negative and we flush-denormals-to-zero).
//
//   Phase B (pack): grid=(n_groups, T), block=(128,1).  Each CTA
//     re-loads its group of X[t, perm[...]], reads per_token_max[t],
//     recomputes scale via the bit-exact (/7 -> fp16 -> fp32) chain,
//     quantises, packs, writes sum_X[t, g] once (single int32 store,
//     no cross-warp reduce because lane-parallel warp_sum is per-warp
//     and we use 4 warps summing via a mini shmem[4] reduce -- but
//     kept minimal with just ONE local __syncwarp + a single sync_0).
//
//   Contract with scale_x:
//     scale_fp32  = max / 7
//     scale_fp16  = fp16(scale_fp32)     (write to scale_x[t])
//     scale_math  = fp32(scale_fp16)     (used for division)
//   This matches the sp/2p kernels exactly.
//
//   per_token_max[t] lives in a small workspace tensor allocated by the
//   host-side launcher.  We use int32 + float_as_int to reuse
//   atomicMax(int*, int) which is lock-free on SM89.
//
// Bit-exactness: identical scale chain, identical quantize_one, and
// atomicMax on positive-fp32-bits is deterministic (commutative on
// max).  sum_X output differs from sp by at most the order of int
// additions, but int32 add is associative, so results are bit-equal.

__global__ void act_quant_phase_a_max(
    const __half* __restrict__ X,
    const int* __restrict__ perm,
    int* __restrict__ per_token_max_bits,     // fp32 bits, int32
    int T, int D,
    int64_t stride_xt, int64_t stride_xd
) {
    const int g = blockIdx.x;
    const int t = blockIdx.y;
    const int lane_x = threadIdx.x;
    const int lane_in_warp = lane_x & (kWarpSize - 1);
    const int warp_id_x = lane_x / kWarpSize;

    const int d = g * BCOL + lane_x;
    int pidx = __ldg(perm + d);
    __half h = __ldg(X + (int64_t)t * stride_xt + (int64_t)pidx * stride_xd);
    float v = fabsf(__half2float(h));

    // 128-wide max: warp-wise first then 4-warp combine via shmem.
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    __shared__ float s_part[4];
    if (lane_in_warp == 0) s_part[warp_id_x] = v;
    __syncthreads();
    if (warp_id_x == 0 && lane_in_warp < 4) {
        float m = s_part[lane_in_warp];
        float o2 = __shfl_xor_sync(0x0000000f, m, 2);
        m = fmaxf(m, o2);
        float o1 = __shfl_xor_sync(0x0000000f, m, 1);
        m = fmaxf(m, o1);
        if (lane_in_warp == 0) {
            // max is non-negative fp32 -> int32 bit-pattern preserves order.
            int m_bits = __float_as_int(m);
            atomicMax(&per_token_max_bits[t], m_bits);
        }
    }
}

__global__ void act_quant_phase_b_pack(
    const __half* __restrict__ X,
    const int* __restrict__ perm,
    const int* __restrict__ per_token_max_bits,
    int8_t* __restrict__ X_s4,
    __half* __restrict__ scale_x,
    int* __restrict__ sum_X,
    int T, int D,
    int64_t stride_xt, int64_t stride_xd,
    int64_t stride_qt, int64_t stride_qd,
    int64_t stride_st, int64_t stride_sg
) {
    const int g = blockIdx.x;
    const int t = blockIdx.y;
    const int lane_x = threadIdx.x;
    const int lane_in_warp = lane_x & (kWarpSize - 1);
    const int warp_id_x = lane_x / kWarpSize;

    // Recompute scale using bit-exact chain.
    float token_max = __int_as_float(per_token_max_bits[t]);
    float scale_fp32 = token_max / 7.0f;
    __half scale_h = __float2half(scale_fp32);
    float scale_math = __half2float(scale_h);
    bool is_zero = !(scale_math > 0.0f);
    float scale_safe = is_zero ? 1.0f : scale_math;

    // lane 0 of warp 0 also writes scale_x[t] (idempotent across groups;
    // every CTA of this token writes the same value).  Gate to only the
    // first group CTA to avoid redundant HBM traffic.
    if (g == 0 && lane_x == 0) {
        scale_x[t] = scale_h;
    }

    const int d = g * BCOL + lane_x;
    int pidx = __ldg(perm + d);
    __half h = __ldg(X + (int64_t)t * stride_xt + (int64_t)pidx * stride_xd);
    float x = __half2float(h);
    int q = quantize_one(x, scale_safe, is_zero);

    // Per-group sum: 128-wide reduction.
    int wsum = q;
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        wsum += __shfl_xor_sync(0xffffffff, wsum, offset);
    }
    __shared__ int s_part_int[4];
    if (lane_in_warp == 0) s_part_int[warp_id_x] = wsum;
    __syncthreads();
    if (lane_x == 0) {
        int total = s_part_int[0] + s_part_int[1] + s_part_int[2] + s_part_int[3];
        sum_X[(int64_t)t * stride_st + (int64_t)g * stride_sg] = total;
    }

    // Pack LE: even lane low nibble, odd lane high nibble.
    int q_neighbour = __shfl_xor_sync(0xffffffff, q, 1);
    if ((lane_x & 1) == 0) {
        int low = q & 0x0F;
        int high = q_neighbour & 0x0F;
        int packed = (high << 4) | low;
        int64_t byte_off = (int64_t)t * stride_qt
                         + (int64_t)(d >> 1) * stride_qd;
        X_s4[byte_off] = static_cast<int8_t>(
            packed >= 128 ? packed - 256 : packed
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-CTA v2 kernels (r62 P0 optimisation for all T)
// ---------------------------------------------------------------------------
//
// Rationale: the sp path uses grid=ceil_div(T, 4), so for T=1 only 1 CTA
//   runs and 127 SMs idle.  Measured wall time is ~9.5 us for T=1 D=4096
//   but HBM roofline is ~0.03 us -> 300x slowdown driven by SM underuse.
//
// R33's original mp split used atomicMax on a single per-token max slot,
//   which serialised the n_groups CTAs and canceled the SM-spread gain.
//
// This v2 split avoids the atomic entirely:
//   Phase A: grid=(n_groups, T, 1), block=(128,1,1).  Each CTA computes
//            max_abs over its own 128 elements and writes
//            workspace[t, g] (fp32, non-atomic, unique slot).
//   Phase B: grid=(n_groups, T, 1), block=(128,1,1).  Each CTA reduces
//            workspace[t, :] across g via a strided loop (ng <= 128 on
//            all shapes we care about), recomputes the scale chain
//            locally, gathers its own group, quantises, packs, writes
//            sum_X[t, g] (exactly once per CTA).
//
// Shmem: only 32 B per CTA (warp reduction staging), so occupancy is
//   bound purely by register pressure.  Kernels are <30 regs each, so
//   at least 16 CTAs per SM are resident -> full HBM pipeline.
//
// Bit-exactness: identical scale chain (max/7 -> fp16 -> fp32), identical
//   quantize_one, identical sum reduction order.  The only difference
//   from sp is HOW token_max is aggregated (atomicMax vs reduce loop),
//   which on non-negative fp32 produces the same bit pattern.

__global__ void act_quant_phase_a_max_v2(
    const __half* __restrict__ X,
    const int* __restrict__ perm,
    float* __restrict__ workspace,            // (T, n_groups) fp32
    int T, int D,
    int64_t stride_xt, int64_t stride_xd,
    int64_t stride_wt, int64_t stride_wg
) {
    const int g = blockIdx.x;
    const int t = blockIdx.y;
    const int lane_x = threadIdx.x;
    const int lane_in_warp = lane_x & (kWarpSize - 1);
    const int warp_id_x = lane_x / kWarpSize;

    const int d = g * BCOL + lane_x;
    int pidx = __ldg(perm + d);
    __half h = __ldg(X + (int64_t)t * stride_xt + (int64_t)pidx * stride_xd);
    float v = fabsf(__half2float(h));

    // 128-wide max: warp reduce then 4-warp combine via shmem.
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    __shared__ float s_part[4];
    if (lane_in_warp == 0) s_part[warp_id_x] = v;
    __syncthreads();
    if (lane_x == 0) {
        float m = fmaxf(fmaxf(s_part[0], s_part[1]),
                        fmaxf(s_part[2], s_part[3]));
        workspace[(int64_t)t * stride_wt + (int64_t)g * stride_wg] = m;
    }
}

__global__ void act_quant_phase_b_pack_v2(
    const __half* __restrict__ X,
    const int* __restrict__ perm,
    const float* __restrict__ workspace,      // (T, n_groups) fp32
    int8_t* __restrict__ X_s4,
    __half* __restrict__ scale_x,
    int* __restrict__ sum_X,
    int T, int D, int n_groups,
    int64_t stride_xt, int64_t stride_xd,
    int64_t stride_qt, int64_t stride_qd,
    int64_t stride_st, int64_t stride_sg,
    int64_t stride_wt, int64_t stride_wg
) {
    const int g = blockIdx.x;
    const int t = blockIdx.y;
    const int lane_x = threadIdx.x;
    const int lane_in_warp = lane_x & (kWarpSize - 1);
    const int warp_id_x = lane_x / kWarpSize;

    // Reduce workspace[t, :] -> token_max, spread across 128 lanes.
    // Each lane handles lanes_per_group strides through n_groups, then
    // 128-wide reduction finalises.
    float local_max = 0.0f;
    for (int gi = lane_x; gi < n_groups; gi += kLanesPerGroup) {
        float v = workspace[(int64_t)t * stride_wt + (int64_t)gi * stride_wg];
        local_max = fmaxf(local_max, v);
    }
    // warp reduce
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }
    __shared__ float s_tokmax[4];
    if (lane_in_warp == 0) s_tokmax[warp_id_x] = local_max;
    __syncthreads();
    float token_max = fmaxf(fmaxf(s_tokmax[0], s_tokmax[1]),
                            fmaxf(s_tokmax[2], s_tokmax[3]));

    // Bit-exact scale chain.
    float scale_fp32 = token_max / 7.0f;
    __half scale_h = __float2half(scale_fp32);
    float scale_math = __half2float(scale_h);
    bool is_zero = !(scale_math > 0.0f);
    float scale_safe = is_zero ? 1.0f : scale_math;

    // Only the g=0 CTA writes scale_x[t] (all CTAs compute the same
    // value, but we must avoid redundant HBM writes).
    if (g == 0 && lane_x == 0) {
        scale_x[t] = scale_h;
    }

    // Gather and quantise the 128 elements for this group.
    const int d = g * BCOL + lane_x;
    int pidx = __ldg(perm + d);
    __half h = __ldg(X + (int64_t)t * stride_xt + (int64_t)pidx * stride_xd);
    float x = __half2float(h);
    int q = quantize_one(x, scale_safe, is_zero);

    // Per-group sum: 128-wide reduction.
    int wsum = q;
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        wsum += __shfl_xor_sync(0xffffffff, wsum, offset);
    }
    __shared__ int s_part_int[4];
    if (lane_in_warp == 0) s_part_int[warp_id_x] = wsum;
    __syncthreads();
    if (lane_x == 0) {
        int total = s_part_int[0] + s_part_int[1] + s_part_int[2] + s_part_int[3];
        sum_X[(int64_t)t * stride_st + (int64_t)g * stride_sg] = total;
    }

    // Pack LE: even lane low nibble, odd lane high nibble.
    int q_neighbour = __shfl_xor_sync(0xffffffff, q, 1);
    if ((lane_x & 1) == 0) {
        int low = q & 0x0F;
        int high = q_neighbour & 0x0F;
        int packed = (high << 4) | low;
        int64_t byte_off = (int64_t)t * stride_qt
                         + (int64_t)(d >> 1) * stride_qd;
        X_s4[byte_off] = static_cast<int8_t>(
            packed >= 128 ? packed - 256 : packed
        );
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher
// ---------------------------------------------------------------------------

void launch(torch::Tensor X_fp16, torch::Tensor perm,
            torch::Tensor X_s4, torch::Tensor scale_x,
            torch::Tensor sum_X,
            int T, int D, int bcol) {
    TORCH_CHECK(bcol == BCOL,
                "CUDA activation_quant only supports bcol == 128 (got ",
                bcol, ")");
    TORCH_CHECK(D % BCOL == 0,
                "D (", D, ") must be divisible by BCOL (", BCOL, ")");
    TORCH_CHECK(X_fp16.scalar_type() == torch::kHalf, "X must be fp16");
    TORCH_CHECK(perm.scalar_type() == torch::kInt32, "perm must be int32");
    TORCH_CHECK(X_s4.scalar_type() == torch::kInt8, "X_s4 must be int8");
    TORCH_CHECK(scale_x.scalar_type() == torch::kHalf, "scale_x must be fp16");
    TORCH_CHECK(sum_X.scalar_type() == torch::kInt32, "sum_X must be int32");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Decide between single-pass (low T, D fits in shmem) and 2-pass
    // (fallback for large T or unusually large D).
    //
    // Per-CTA dynamic shmem limit on SM89 is 48 KB without opt-in.
    // sp shmem = kBt * D * 2 bytes; with D=4096, kBt=4 needs 32 KB
    // which fits, but kBt=8 needs 64 KB.
    //
    // Round 20: extend sp to T > 4 by routing to kBt=4 sp kernel with
    //   a larger grid.  Each CTA still only holds kBt*D bytes of shmem
    //   (<=48 KB), while multiple CTAs handle T/kBt tokens in parallel.
    //   The SP path halves HBM gather traffic vs 2-pass, which is the
    //   dominant cost at T=8..64 (wall time ~19 us is mostly gather).
    //
    // Round 21 fix: the total sp shmem is
    //     kBt * D * sizeof(half) + kBt * 4 * sizeof(float)
    //         + kBt * 4 * sizeof(int) + kBt * sizeof(float)
    //         + kBt * sizeof(unsigned char)   (rounded up to 16 bytes)
    //   so `kBt * D * 2` alone is NOT the full budget -- the staging
    //   area costs up to ~100 bytes per kBt plus alignment padding.
    //   Qwen3-1.7B down_proj (D=6144) triggered this: 4*6144*2 = 49152
    //   bytes alone already fit, but the total shmem was 49300 bytes,
    //   which exceeds SM89's default 48 KiB dynamic-shmem cap and made
    //   the launch fail with cudaErrorInvalidArgument.
    //
    //   Subtract a conservative 1 KiB safety margin for the staging
    //   area + alignment so the gate matches the real launch budget.
    constexpr size_t kSpSafetyMargin = 1024u;    // room for staging/align
    const size_t sp_budget_bytes = 48u * 1024u - kSpSafetyMargin;
    // Pick largest kBt (1, 2, 4) such that kBt * D * 2 fits in shmem.
    int sp_kBt = 0;
    if ((size_t)4 * D * 2u <= sp_budget_bytes)      sp_kBt = 4;
    else if ((size_t)2 * D * 2u <= sp_budget_bytes) sp_kBt = 2;
    else if ((size_t)1 * D * 2u <= sp_budget_bytes) sp_kBt = 1;
    const bool sp_ok = (sp_kBt > 0);

    auto dispatch_2p = [&](auto kBtConst) {
        constexpr int kBt = decltype(kBtConst)::value;
        dim3 block(kLanesPerGroup, kBt, 1);
        dim3 grid(ceil_div(T, kBt), 1, 1);
        activation_quant_kernel<kBt><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
            perm.data_ptr<int>(),
            X_s4.data_ptr<int8_t>(),
            reinterpret_cast<__half*>(scale_x.data_ptr<at::Half>()),
            sum_X.data_ptr<int>(),
            T, D,
            X_fp16.stride(0), X_fp16.stride(1),
            X_s4.stride(0),    X_s4.stride(1),
            sum_X.stride(0),   sum_X.stride(1)
        );
    };

    auto dispatch_sp = [&](auto kBtConst) {
        constexpr int kBt = decltype(kBtConst)::value;
        dim3 block(kLanesPerGroup, kBt, 1);
        dim3 grid(ceil_div(T, kBt), 1, 1);
        size_t shmem = kBt * (size_t)D * sizeof(__half)
                     + kBt * 4 * sizeof(float)
                     + kBt * 4 * sizeof(int)
                     + kBt * sizeof(float)
                     + kBt * sizeof(unsigned char);
        // Round up to 16-byte alignment; CUDA requires.
        shmem = (shmem + 15) & ~size_t(15);
        activation_quant_kernel_sp<kBt><<<grid, block, shmem, stream>>>(
            reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
            perm.data_ptr<int>(),
            X_s4.data_ptr<int8_t>(),
            reinterpret_cast<__half*>(scale_x.data_ptr<at::Half>()),
            sum_X.data_ptr<int>(),
            T, D,
            X_fp16.stride(0), X_fp16.stride(1),
            X_s4.stride(0),    X_s4.stride(1),
            sum_X.stride(0),   sum_X.stride(1)
        );
    };

    auto dispatch_mp2 = [&]() {
        // Phase A/B: grid=(n_groups, T, 1), block=(128, 1, 1).
        //   Each CTA handles one (token, group).
        //   Workspace: (T, n_groups) fp32, allocated on the current device.
        const int n_groups = D / BCOL;
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(X_fp16.device());
        auto workspace = torch::empty({T, n_groups}, opts);

        dim3 block(kLanesPerGroup, 1, 1);
        dim3 grid(n_groups, T, 1);

        act_quant_phase_a_max_v2<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
            perm.data_ptr<int>(),
            workspace.data_ptr<float>(),
            T, D,
            X_fp16.stride(0), X_fp16.stride(1),
            workspace.stride(0), workspace.stride(1)
        );
        act_quant_phase_b_pack_v2<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(X_fp16.data_ptr<at::Half>()),
            perm.data_ptr<int>(),
            workspace.data_ptr<float>(),
            X_s4.data_ptr<int8_t>(),
            reinterpret_cast<__half*>(scale_x.data_ptr<at::Half>()),
            sum_X.data_ptr<int>(),
            T, D, n_groups,
            X_fp16.stride(0), X_fp16.stride(1),
            X_s4.stride(0),    X_s4.stride(1),
            sum_X.stride(0),   sum_X.stride(1),
            workspace.stride(0), workspace.stride(1)
        );
    };

    // r62 P0: mp2 dispatch gate.  Enabled by default for T <= 256 where
    //   SM under-utilisation by the sp path dominates measured latency.
    //   Above T=256 the sp kernel already saturates both gather BW and
    //   SM count (grid ~= 64 CTAs for T=256 kBt=4), so mp2 loses to sp
    //   due to an extra pass over X.
    //
    //   The gate is overridable at runtime via HKUST_V9_ACTQUANT_PATH:
    //       "mp2"     -> always mp2
    //       "sp"      -> always sp (legacy)
    //       "auto"    -> heuristic (default)
    const char* path_env = std::getenv("HKUST_V9_ACTQUANT_PATH");
    std::string path = path_env ? std::string(path_env) : std::string("auto");
    bool use_mp2;
    if (path == "mp2") {
        use_mp2 = true;
    } else if (path == "sp") {
        use_mp2 = false;
    } else {
        // Heuristic: mp2 is a win when sp's SM count < ~32 (i.e. T/4 < 32).
        // On RTX 4090 (128 SMs) this means T <= 128.  Bump to 256 for
        // conservative rollout; will retune after bench data.
        use_mp2 = sp_ok && (T <= 256);
    }

    if (use_mp2) {
        dispatch_mp2();
    } else if (sp_ok) {
        // Round 20: use sp for all T.  sp_kBt already chosen to fit shmem.
        if      (sp_kBt == 1) dispatch_sp(std::integral_constant<int, 1>{});
        else if (sp_kBt == 2) dispatch_sp(std::integral_constant<int, 2>{});
        else                  dispatch_sp(std::integral_constant<int, 4>{});
    } else {
        // Shmem too small for any sp kBt (unusual: D > ~12 K fp16).
        if      (T <= 1) dispatch_2p(std::integral_constant<int, 1>{});
        else if (T <= 4) dispatch_2p(std::integral_constant<int, 4>{});
        else             dispatch_2p(std::integral_constant<int, 8>{});
    }

    C10_CUDA_CHECK(cudaGetLastError());
}

}  // namespace activation_quant
}  // namespace hkust_v9
