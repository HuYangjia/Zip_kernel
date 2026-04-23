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

}  // namespace hkust_v9
