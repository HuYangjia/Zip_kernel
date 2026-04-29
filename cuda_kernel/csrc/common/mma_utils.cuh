// MMA helpers for SM89 (RTX 4090).
//
// Wraps three low-level primitives that the INT8 / INT4 MMA GEMM
// kernels share:
//
//   1. s4 -> s8 unpacking (two variants depending on target MMA).
//   2. ldmatrix.sync.aligned.m8n8.x{2,4}.shared.b16   (shmem -> A/B regs).
//   3. mma.sync.aligned.m16n8k32.s8.s8.s32            (INT8 Tensor Core).
//   4. mma.sync.aligned.m16n8k64.s4.s4.s32            (INT4 Tensor Core,
//                                                     SM89 legacy but
//                                                     still accelerated).
//
// Operand layouts follow the PTX ISA reference:
//
//   m16n8k32.s8  A = 16x32 int8 (4 regs/thread; thread t holds:
//                    a0 = row t//4 + {0,8},      col 4*(t%4) + {0..3}
//                    a1 = row t//4 + {0,8} + 16, col 4*(t%4) + {0..3}
//                    i.e. 16 int8 values per thread).
//                B = 32x8 int8 (2 regs/thread; t holds col t//4 + {0,4..4},
//                               row 4*(t%4) + {0..3} twice (k=0..15,16..31)).
//                C = 16x8 int32 accumulator (4 regs/thread).
//
//   m16n8k64.s4  A = 16x64 s4 (4 regs/thread, same layout as s8 but s4).
//                B = 64x8  s4 (2 regs/thread).
//                C = 16x8 int32 accumulator (4 regs/thread).
//
// Both instructions treat one 32-bit register as "4 s8 lanes" or
// "8 s4 lanes" respectively; ldmatrix.x{2,4} conveniently produces
// exactly that shape from a (8, 8) / (8, 16) shmem tile with b16
// element size.  We therefore always *stage* the operand tiles into
// shmem in a b16-granularity layout (2 s8 per b16 or 4 s4 per b16) and
// let ldmatrix do the cross-lane permute for us.

#pragma once

#include "common/arch.cuh"

#include <cstdint>

#if !defined(__CUDACC__)
#error "mma_utils.cuh must be included from a .cu compilation unit"
#endif

namespace hkust_v9 {

// ---------------------------------------------------------------------------
// s4 unpack helpers (four-element and eight-element variants).
// ---------------------------------------------------------------------------

// Sign-extend a 4-bit nibble stored in the low 4 bits of a byte.
__device__ __forceinline__ int8_t s4_lo(uint8_t b) {
    int v = b & 0x0F;
    return static_cast<int8_t>(v - ((v & 0x08) << 1));
}
__device__ __forceinline__ int8_t s4_hi(uint8_t b) {
    int v = (b >> 4) & 0x0F;
    return static_cast<int8_t>(v - ((v & 0x08) << 1));
}

// Unpack 4 packed s4 bytes (== 8 s4 values, little-endian) into two
// int32 words each holding 4 s8 lanes.  Output layout:
//   out0 = [c3 c2 c1 c0]    (low byte = col 0)
//   out1 = [c7 c6 c5 c4]
// Matches exactly what mma.m16n8k32.s8 expects per-register.
__device__ __forceinline__ void unpack_s4_to_s8_x8(
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
// ldmatrix wrappers.
//
// ``ptr`` is a *shared memory* address of the 8x8 (x2 / x4) tile in
// b16-granularity element size (16 bits per element).  For MMA.s8 the
// callers pack 2 s8 values per b16 lane; for MMA.s4 they pack 4 s4
// values per b16 lane.
//
// The x4 variant returns 4 registers per thread (four 8x8 tiles), x2
// returns 2 registers.  Caller chooses based on the operand dimension
// (A row-major m16k32 needs x4; B column-major k32n8 needs x2).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void ldmatrix_x4_b16(
    const void* shmem_ptr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3
) {
    uint32_t u = static_cast<uint32_t>(__cvta_generic_to_shared(shmem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(u)
    );
}

__device__ __forceinline__ void ldmatrix_x2_b16(
    const void* shmem_ptr,
    uint32_t& r0, uint32_t& r1
) {
    uint32_t u = static_cast<uint32_t>(__cvta_generic_to_shared(shmem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(u)
    );
}

__device__ __forceinline__ void ldmatrix_x1_b16(
    const void* shmem_ptr,
    uint32_t& r0
) {
    uint32_t u = static_cast<uint32_t>(__cvta_generic_to_shared(shmem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 "
        "{%0}, [%1];\n"
        : "=r"(r0)
        : "r"(u)
    );
}

// Trans variant (needed when B is row-major in shmem but MMA wants column-
// major operand layout, or vice versa; kept here for completeness).
__device__ __forceinline__ void ldmatrix_x2_trans_b16(
    const void* shmem_ptr,
    uint32_t& r0, uint32_t& r1
) {
    uint32_t u = static_cast<uint32_t>(__cvta_generic_to_shared(shmem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(u)
    );
}

__device__ __forceinline__ void ldmatrix_x4_trans_b16(
    const void* shmem_ptr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3
) {
    uint32_t u = static_cast<uint32_t>(__cvta_generic_to_shared(shmem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(u)
    );
}

// ---------------------------------------------------------------------------
// MMA instructions.
//
//   mma_m16n8k32_s8s8s32  : 16x8x32 INT8 MMA (SM80+; SM89 native rate).
//   mma_m16n8k64_s4s4s32  : 16x8x64 INT4 MMA (SM80+, deprecated in SM90
//                           but fully functional on SM89 Ada).
//
// Signatures:
//   A = 4 x uint32  (s8 lanes for INT8, s4 lanes for INT4)
//   B = 2 x uint32
//   C = D = 4 x int32
//
// ``D`` and ``C`` must be distinct registers per PTX manual; we write
// C += A*B by passing C on both sides (compiler will reuse if safe).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k32_s8s8s32(
    int& d0, int& d1, int& d2, int& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int c0, int c1, int c2, int c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
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

__device__ __forceinline__ void mma_m16n8k64_s4s4s32(
    int& d0, int& d1, int& d2, int& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int c0, int c1, int c2, int c3
) {
    // SM89 keeps m16n8k64.s4 as an accelerated (Tensor Core) path even
    // though it is deprecated in newer PTX.  ptxas will emit a warning
    // that we silence at the NVCC flag level (we don't add -Wno-deprecated
    // globally; the warning is harmless for one kernel).
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
// cp.async predicated helper (SM80+).
//
// arch.cuh already provides cp_async_cg_16, cp_async_commit, and
// cp_async_wait_group<N>.  This file adds only the predicated variant
// that zero-fills the destination when pred==false (needed for boundary
// rows/columns in the dense branch of the INT4 kernel).
// ---------------------------------------------------------------------------

// Copy 16 bytes with a predicate; zero-fills dst when pred==false.
__device__ __forceinline__ void cp_async_cg_16_pred(
    void* dst, const void* src, bool pred
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  cp.async.cg.shared.global [%0], [%1], 16, p;\n"
        "}\n"
        :
        : "r"(smem_addr), "l"(src), "r"((int)pred)
    );
}

}  // namespace hkust_v9
