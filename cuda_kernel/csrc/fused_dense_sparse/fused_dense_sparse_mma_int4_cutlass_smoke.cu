// Copyright (c) 2026 HKUST R50 — minimal CUTLASS 2.11 INT4 smoke unit.
//
// Goal of this file: force full template instantiation of
// `cutlass::gemm::device::Gemm<int4b_t, ..., arch::Sm80, ...>` in two
// configurations derived from r50_cutlass_int4/layout_contract.md §2:
//
//   (S1) Official SM80 default tile  <128,256,128> — 100% validated by
//        `extern/cutlass/test/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sm80.cu`.
//        Used as a known-good CUTLASS 2.11 configuration to smoke-test the
//        compile toolchain (nvcc 12.8, sm_89, CUTLASS 2.11.0 header-only).
//
//   (S2) Contract §2.2 override   <128,128,128> — the ThreadblockShape we
//        actually want for the R50 W4A4 kernel (tile_k == BCOL == 128).
//        If this variant fails to compile, the failure surfaces here in
//        isolation and we re-negotiate the contract before any launcher
//        or epilogue work in L3.4/L3.5.
//
// This file intentionally does NOT launch the kernel. A host entry
// `launch_cutlass_smoke()` returns the `sizeof` both Gemm objects so
// the optimiser cannot DCE the template. This keeps the smoke
// self-contained (no host tensors, no pybind binding) and immune to
// runtime GPU state.
//
// Once the smoke build passes, L3.3–L3.5 flesh out:
//   - `cutlass_helpers/int4_mma_builder.hpp` — templated GemmBuilder
//     that returns the `device::Gemm` type selected by D.4 invariants
//   - `cutlass_helpers/int4_epilogue_dequant.hpp` — the W4A4 dequant
//     epilogue visitor (replaces LinearCombinationClamp used here)
//   - `fused_dense_sparse_mma_int4_cutlass.cu` — the real launcher

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"

#include <cstddef>

namespace hkust_r50 {
namespace cutlass_smoke {

// ---------------------------------------------------------------------------
// Common type aliases (match layout_contract.md §2.3 / §2.4)
// ---------------------------------------------------------------------------
using ElementA = cutlass::int4b_t;               // T1 weight, reinterpret
using ElementB = cutlass::int4b_t;               // T2 activation, reinterpret
using ElementC = int32_t;                        // T3 C_s32 accumulator output (for smoke; real epilogue -> half_t, L3.5)
using ElementAccumulator = int32_t;
using ElementCompute = int32_t;

using LayoutA = cutlass::layout::RowMajor;       // contract §2.4 (D.2)
using LayoutB = cutlass::layout::ColumnMajor;    // contract §2.4 (D.3)
using LayoutC = cutlass::layout::RowMajor;       // contract §2.4

// Smoke epilogue — placeholder only; the real W4A4 dequant visitor
// (LinearCombinationDequantizeW4A4 from contract §3) is L3.5.
template <int kElementsPerAccess>
using SmokeEpilogue = cutlass::epilogue::thread::LinearCombinationClamp<
    ElementC, kElementsPerAccess, ElementAccumulator, ElementCompute>;

using ThreadblockSwizzle =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// ---------------------------------------------------------------------------
// (S1) Known-good SM80 default (tile 128x256x128, warp 64x64x128, inst 16x8x64, 3-stage)
// ---------------------------------------------------------------------------
using GemmS1 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    SmokeEpilogue<128 / cutlass::sizeof_bits<ElementC>::value>,
    ThreadblockSwizzle,
    /* Stages = */ 3>;

// ---------------------------------------------------------------------------
// (S2) Contract §2.2 override (tile 128x128x128 — tile_k == BCOL == 128)
// ---------------------------------------------------------------------------
using GemmS2 = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 128>,
    cutlass::gemm::GemmShape<64, 64, 128>,
    cutlass::gemm::GemmShape<16, 8, 64>,
    SmokeEpilogue<128 / cutlass::sizeof_bits<ElementC>::value>,
    ThreadblockSwizzle,
    /* Stages = */ 3>;

}  // namespace cutlass_smoke
}  // namespace hkust_r50

// ---------------------------------------------------------------------------
// (S3) L3.3-L3.5 helpers integration — pulls in the cutlass_helpers/*.hpp
//      tree in smoke-only mode (no custom dequant visitor, falls back to
//      LinearCombinationClamp via the HKUST_R50_CUTLASS_SMOKE_ONLY macro).
//      Proves that the header-only alias chain instantiates cleanly.
// ---------------------------------------------------------------------------
#define HKUST_R50_CUTLASS_SMOKE_ONLY 1
#include "../cutlass_helpers/int4_mma_builder.hpp"

namespace hkust_r50 {
namespace cutlass_smoke {

using GemmS3 = hkust_r50::cutlass_int4::Int4Gemm;

static_assert(GemmS3::kStages == 3,
              "L3.4 contract: Stages must be 3 (B3 fix).");
static_assert(hkust_r50::cutlass_int4::ThreadblockShape::kK == 128,
              "L3.4 contract: tile_k == BCOL == 128.");

}  // namespace cutlass_smoke
}  // namespace hkust_r50

// composite sentinel (sum of sizes) so DCE cannot elide the templates.
// Called from a Python ctypes/pybind dispatch path in L3.9 if needed;
// otherwise simply linking this object file is sufficient to confirm
// template expansion succeeded.
// ---------------------------------------------------------------------------
extern "C" std::size_t hkust_r50_cutlass_smoke_probe() {
    using namespace hkust_r50::cutlass_smoke;
    // sizeof is a compile-time probe; the *real* instantiation happens
    // in the default-ctor of the inner MmaBase when these types are
    // used in device code. We therefore also construct a trivial
    // host-side Arguments to exercise the argument-packing path.
    typename GemmS1::Arguments args_s1{
        {1, 1, 128},  // problem_size = {M, N, K}
        {/*A*/ nullptr, 0},
        {/*B*/ nullptr, 0},
        {/*C*/ nullptr, 0},
        {/*D*/ nullptr, 0},
        {/*alpha*/ ElementCompute(1), /*beta*/ ElementCompute(0)}};
    typename GemmS2::Arguments args_s2{
        {1, 1, 128},
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
        {ElementCompute(1), ElementCompute(0)}};
    (void)args_s1;
    (void)args_s2;
    // S3 — ensure the cutlass_helpers template tree also instantiates.
    using GemmS3Local = hkust_r50::cutlass_smoke::GemmS3;
    typename GemmS3Local::Arguments args_s3{
        {1, 1, 128},
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
        {nullptr, 0},
        // S3 epilogue (LinearCombinationClamp, see int4_mma_builder.hpp
        // smoke-only branch) takes (alpha, beta) with ElementCompute=float.
        {1.f, 0.f}};
    (void)args_s3;
    return sizeof(GemmS1) + sizeof(GemmS2) + sizeof(GemmS3Local);
}
