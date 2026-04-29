// V9 CUDA kernel common definitions (RTX 4090 / SM89).
//
// This header gates all arch-specific code.  We target SM89 exclusively
// in Phase 1; adding SM90/SM100 later means adding alternative
// implementations behind ``#if __CUDA_ARCH__ >= 900`` without touching
// the call sites.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if !defined(__CUDACC__)
#error "arch.cuh must be included from a .cu compilation unit"
#endif

// ---------------------------------------------------------------------------
// SM capability probe.  We deliberately fail the build on anything < 80
// because our cp.async / ldmatrix usage is SM80+.  The runtime machine
// arch check (cudaDeviceProp.major*10 + minor == 89) is done on the
// host side in bindings.cc.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "V9 CUDA kernels require SM80 or newer (cp.async, ldmatrix)"
#endif

namespace hkust_v9 {

// Common block-shape constants, must match BROW/BCOL in pack_utils.py.
constexpr int BROW = 128;
constexpr int BCOL = 128;

// Warp size (fixed to 32 on all currently shipping NVIDIA GPUs).
constexpr int kWarpSize = 32;

// Helper: integer ceil-div.
template <typename T>
__host__ __device__ constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// Helper: round up to the next multiple of ``b``.
template <typename T>
__host__ __device__ constexpr T round_up(T a, T b) {
    return ceil_div(a, b) * b;
}

// CUDA error checking for host-side launches.  We prefer ``TORCH_CHECK``
// in bindings.cc, but raw kernels sometimes want a bare assertion.
#define HKUST_V9_CUDA_CHECK(expr)                                            \
    do {                                                                     \
        cudaError_t _err = (expr);                                           \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                  \
                    cudaGetErrorName(_err), __FILE__, __LINE__,              \
                    cudaGetErrorString(_err));                               \
            abort();                                                         \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// cp.async helpers (SM80+).  Used by dense/sparse/fused kernels to
// prefetch the next group's X tile into shared memory while the current
// group is still computing.  Double-buffering lets us overlap global
// memory latency with dp4a math.
//
// (We always target SM89 per the arch guard at the top of this header,
// so these helpers are unconditionally declared.  Using them from host
// code would fail to link, but that never happens in practice.)
// ---------------------------------------------------------------------------

// Convert a generic shared-memory pointer to its 32-bit shared-state
// address (required by the cp.async PTX operand).
__device__ __forceinline__ uint32_t shmem_ptr_to_uint32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Issue one 16-byte cp.async from global to shared.
__device__ __forceinline__ void cp_async_cg_16(
    void* dst_shmem, const void* src_gmem
) {
    uint32_t dst_u32 = shmem_ptr_to_uint32(dst_shmem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst_u32), "l"(src_gmem)
    );
}

// Issue one 4-byte cp.async from global to shared.
// Requires only 4-byte alignment (vs 16-byte for cp_async_cg_16).
// Use when the destination address is not 16-byte aligned (e.g. when
// smem rows are padded to a non-16-byte-multiple stride for bank-conflict
// avoidance).
__device__ __forceinline__ void cp_async_ca_4(
    void* dst_shmem, const void* src_gmem
) {
    uint32_t dst_u32 = shmem_ptr_to_uint32(dst_shmem);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(dst_u32), "l"(src_gmem)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

}  // namespace hkust_v9
