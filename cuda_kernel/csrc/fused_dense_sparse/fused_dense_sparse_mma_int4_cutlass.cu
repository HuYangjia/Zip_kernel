// Copyright (c) 2026 HKUST R50 — CUTLASS INT4 W4A4 fused Dense+Sparse launcher.
//
// L3.6 of plan r50_cutlass_int4. This translation unit:
//   1. Pulls the entire `cutlass_helpers/*.hpp` tree so the SM80 INT4
//      `Int4Gemm` template is instantiated at build time (closing G1).
//   2. Exposes `hkust_v9::fused_dense_sparse_mma_int4_cutlass::launch`
//      with **the exact same signature** as the hand-rolled
//      `fused_dense_sparse_mma_int4::launch` in
//      `fused_dense_sparse_mma_int4.cu`, so the L3.7 runtime dispatch
//      can swap between them based on `HKUST_V9_USE_CUTLASS`.
//   3. Keeps the function body as a deliberate `TORCH_CHECK(false,...)`
//      during L3 — L3 is the **compile-only gate**; bitwise-correct
//      execution is deferred to L4+ (visitor tree wiring, pack adapter,
//      sparse fallback strategy).
//
// Why this split?
//   * L3.6 unblocks L3.7 (dispatcher) and L3.8 (top-level .cu) even
//     though `LinearCombinationDequantizeW4A4` is still a forward-decl.
//   * The `TORCH_CHECK(false,...)` makes any accidental runtime use
//     loud and safe — the dispatcher must explicitly gate on the env
//     variable, never fall through silently.
//   * If the user sets `HKUST_V9_USE_CUTLASS=1` before L4 is done,
//     they will get a clear torch runtime error with actionable text,
//     not UB.
//
// Build gate: this TU requires CUTLASS v2.11 headers and an SM80+
// target; under SM<80 the include tree is #if-guarded out to keep
// the ABI intact on unsupported arches.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdlib>

// -------------------------------------------------------------------------
// L3.6 compile-only gate: we need the entire CUTLASS template tree to
// instantiate cleanly so nvcc/build-system wiring is proven, but the
// production `LinearCombinationDequantizeW4A4` visitor is still a
// forward-decl (its full definition is L3.6's OTHER deliverable in the
// subsequent iteration — visitor-tree wiring).
//
// Solution: this TU is the ONLY compilation unit allowed to set the
// smoke macro, which makes `Int4Gemm` use the canonical
// `LinearCombinationClamp` epilogue. The alias is kept internal to this
// TU via an anonymous namespace below, so the rest of the build is
// unaffected and this macro does not leak into `cutlass_helpers` headers
// when they are included by `fused_dense_sparse.cu` / dispatch sites
// in a future iteration.
//
// Remove this macro and the `#include` of the helpers from this TU as
// part of L3.6's visitor-tree wiring.
#define HKUST_R50_CUTLASS_SMOKE_ONLY 1

// Only pull the CUTLASS helpers when we actually have an SM80+ target
// in the compile. Note: __CUDA_ARCH__ is only defined in device passes;
// during the host pass we still want the types available so the
// signature compiles. Hence we include unconditionally — the template
// instantiation itself (via `sizeof(Int4Gemm)`) is what triggers the
// architecture-dependent body.
#include "../cutlass_helpers/int4_weight_layout.hpp"
#include "../cutlass_helpers/int4_mma_builder.hpp"

namespace hkust_v9 {
namespace fused_dense_sparse_mma_int4_cutlass {

namespace {

// Force instantiation of the (smoke-mode) `Int4Gemm` type so nvcc
// materialises the full template tree (mainloop + threadblock + the
// canonical clamp epilogue). The production epilogue — custom visitor
// with `LinearCombinationDequantizeW4A4` — is L3.6's second-pass
// deliverable and replaces this probe when it lands.
struct InstantiationProbe {
    static constexpr std::size_t gemm_bytes = sizeof(hkust_r50::cutlass_int4::Int4Gemm);
};

}  // namespace

// -------------------------------------------------------------------------
// Public launcher — ABI-compatible with
// hkust_v9::fused_dense_sparse_mma_int4::launch
// -------------------------------------------------------------------------

void launch(
    torch::Tensor W_low, torch::Tensor W_high_blocks,
    torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
    torch::Tensor X_s4,
    torch::Tensor scale_u4, torch::Tensor zero_u4,
    torch::Tensor sum_X, torch::Tensor scale_x,
    torch::Tensor Y_total,
    int d_out, int d_in
) {
    // Silence -Wunused warnings while keeping the full signature visible.
    (void)W_low; (void)W_high_blocks;
    (void)hp_row_offsets; (void)hp_col_indices;
    (void)X_s4;
    (void)scale_u4; (void)zero_u4;
    (void)sum_X; (void)scale_x;
    (void)Y_total;
    (void)d_out; (void)d_in;
    (void)InstantiationProbe::gemm_bytes;  // force compile-time probe.

    TORCH_CHECK(false,
        "[r50_cutlass_int4] CUTLASS fused Dense+Sparse launcher is a "
        "compile-only stub at L3.6. Do not set HKUST_V9_USE_CUTLASS=1 "
        "until L4 (visitor tree + pack adapter) lands. See "
        ".codebuddy/plan/r50_cutlass_int4/task-item.md.");
}

}  // namespace fused_dense_sparse_mma_int4_cutlass
}  // namespace hkust_v9
