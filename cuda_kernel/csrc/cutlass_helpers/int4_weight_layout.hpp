// Copyright (c) 2026 HKUST R50 — CUTLASS INT4 W4A4 layout aliases.
//
// L3.3 of plan r50_cutlass_int4, consumer of layout_contract.md §1+§2.3+§2.4.
//
// This header defines ONLY type aliases — no runtime state, no function
// bodies. Its purpose is to centralize the choices the rest of the
// cutlass_helpers/ tree depends on (weight / activation / output element
// and layout types), so that a breaking change to the contract shows up
// as exactly one edit here.
//
// INVARIANTS (must match layout_contract.md; do NOT renegotiate silently):
//   I-L1  no repack: W_low_packed streams in as RowMajor int4b_t
//   I-L2  atom = SM80_16x8x64_S32S4S4S32_TN (pinned in int4_mma_builder.hpp)
//   I-L3  ThreadblockShape<128,128,128> (pinned in int4_mma_builder.hpp)
//   I-L4  epilogue dequant formula (pinned in int4_epilogue_dequant.hpp)
//
// This file is header-only and safe to include from both .cu and .cpp.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/gemm.h"

namespace hkust_r50 {
namespace cutlass_int4 {

// ---------------------------------------------------------------------------
// 1. CUTLASS element types (frozen by layout_contract.md §2.3)
// ---------------------------------------------------------------------------

/// Weight operand element — signed 4-bit, packed 2-per-byte.
/// Upstream storage is `int8 (d_out, d_in/2)`, reinterpreted here.
using ElementW = cutlass::int4b_t;

/// Activation operand element — signed 4-bit, packed 2-per-byte.
/// Upstream storage is `int8 (T, d_in/2)`, reinterpreted here.
using ElementX = cutlass::int4b_t;

/// Accumulator element — mandated by the SM80 S32S4S4S32 atom.
using ElementAcc = int32_t;

/// Epilogue compute element — layout_contract.md §2.3 picks `float` for
/// numerical safety of the dequant chain before the final `half` cast.
using ElementCompute = float;

/// Final user-visible output element — `half_t` for compatibility with
/// the torch FP16 caller path.
using ElementY = cutlass::half_t;

/// Epilogue auxiliary element (scale_u4, zero_u4, scale_x all fp16).
using ElementAux = cutlass::half_t;

/// `sum_X` is int32 per layout_contract.md §1 T7.
using ElementSumX = int32_t;

// ---------------------------------------------------------------------------
// 2. CUTLASS layouts (frozen by layout_contract.md §2.4 + D.6)
// ---------------------------------------------------------------------------

/// Weight layout in CUTLASS's `(M, K)` = `(d_out, d_in)` view.
/// Matches T1 — no repack.
using LayoutW = cutlass::layout::RowMajor;

/// Activation layout in CUTLASS's `(K, N)` = `(d_in, T)` view.
/// Our `X_s4` is actually row-major in `(T, d_in)` — which is
/// column-major in `(d_in, T)`. See layout_contract.md D.3.
using LayoutX = cutlass::layout::ColumnMajor;

/// Output layout in CUTLASS's `(M, N)` = `(d_out, T)` view.
/// Column-major `(d_out, T)` is physically identical to row-major
/// `Y_half (T, d_out)` — see layout_contract.md D.6 (2026-04-29).
using LayoutY = cutlass::layout::ColumnMajor;

// ---------------------------------------------------------------------------
// 3. Alignment constants (frozen by layout_contract.md §2.2)
// ---------------------------------------------------------------------------

/// Elements per aligned vector load/store for A operand (int4b_t).
/// 128 bits / 4 bits = 32 elements. CUTLASS expresses alignment in
/// *element* counts, so the compile-time value is 32, not 128.
static constexpr int kAlignmentW = 32;

/// Elements per aligned vector load/store for B operand (int4b_t). Same as A.
static constexpr int kAlignmentX = 32;

/// Elements per epilogue store vector (half_t). 128 bits / 16 bits = 8.
/// layout_contract.md §3 requires 8-wide store to kill sub-bottleneck B1.
static constexpr int kElementsPerAccessEpilogue = 8;

// ---------------------------------------------------------------------------
// 4. TensorRef aliases — what the launcher / tests actually hand to CUTLASS
// ---------------------------------------------------------------------------

using TensorRefW = cutlass::TensorRef<ElementW, LayoutW>;
using TensorRefX = cutlass::TensorRef<ElementX, LayoutX>;
using TensorRefY = cutlass::TensorRef<ElementY, LayoutY>;

using TensorRefScaleU4 = cutlass::TensorRef<ElementAux,    cutlass::layout::RowMajor>;
using TensorRefZeroU4  = cutlass::TensorRef<ElementAux,    cutlass::layout::RowMajor>;
using TensorRefSumX    = cutlass::TensorRef<ElementSumX,   cutlass::layout::PackedVectorLayout>;
using TensorRefScaleX  = cutlass::TensorRef<ElementAux,    cutlass::layout::PackedVectorLayout>;

// ---------------------------------------------------------------------------
// 5. Utility: construct GemmCoord from physical problem shape
// ---------------------------------------------------------------------------
//
// CUTLASS computes `(M, N) = (d_out, T)` = A * B with K = d_in.
// This helper centralises the mapping so launcher code never makes
// the off-by-swap error described in layout_contract.md D.3.

CUTLASS_HOST_DEVICE
inline cutlass::gemm::GemmCoord make_problem_size(int d_out, int d_in, int T) {
    return cutlass::gemm::GemmCoord{/*M=*/d_out, /*N=*/T, /*K=*/d_in};
}

}  // namespace cutlass_int4
}  // namespace hkust_r50
