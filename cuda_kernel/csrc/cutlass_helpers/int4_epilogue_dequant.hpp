// Copyright (c) 2026 HKUST R50 — CUTLASS INT4 W4A4 dequant epilogue.
//
// L3.5 of plan r50_cutlass_int4. Consumer of layout_contract.md §3
// (epilogue math), §D.4 (group-K / tile-K equality), §1 T5..T8 (aux
// tensor inventory).
//
// ---------------------------------------------------------------------------
// STATUS & SCOPE
// ---------------------------------------------------------------------------
//
// This header is the **contract stub** for the real W4A4 dequant epilogue.
// The production functor `LinearCombinationDequantizeW4A4` is NOT yet
// instantiated here — its implementation is deferred to L3.6 (kernel
// launcher), because expressing the per-CTA `zero_u4[m,g]` /
// `scale_u4[m,g]` lookup requires an `EpilogueVisitor` tree instantiation
// that is interleaved with the kernel-side threadblock tile iterator.
// Splitting this across L3.5 (contract) and L3.6 (implementation) keeps
// L4 (Python pack adapter) un-blocked on this file.
//
// For smoke-only compilation (i.e. proving that the INT4 mainloop
// template tree instantiates cleanly without a visitor), define
// `HKUST_R50_CUTLASS_SMOKE_ONLY` before including `int4_mma_builder.hpp`;
// that macro causes the builder to substitute CUTLASS's canonical
// `LinearCombinationClamp<half_t, 8, int32, float>` — numerically wrong
// but structurally identical for compile-phase gating.
//
// ---------------------------------------------------------------------------
// FROZEN CONTRACT (layout_contract.md §3 — do NOT renegotiate silently)
// ---------------------------------------------------------------------------
//
// Per-element math computed by the final functor:
//
//     g    = k_cta_offset / GroupK              // one group per CTA-K slab
//     acc  = acc_s32[m, t]                      // int32 accumulator
//     zero = int32(zero_u4[m, g])               // fp16 → int32 widening
//     sumX = int32(sum_X[t])
//     scl  = float(scale_u4[m, g]) * float(scale_x[t])
//     y    = half_t( (float(acc) - float(zero) * float(sumX)) * scl )
//
// Invariants preserved from the upstream hand-rolled kernel:
//   * `zero_u4` was pre-subtracted by 8 inside `pack_v9_weights`
//     (pack_utils.py:207), so the formula above is the final
//     algebraically-reduced form — NO `-8` correction here.
//   * Final FP16 cast point matches the existing kernel; bitwise
//     equality is not required (see invariant I3 in design.md).
//   * Tile-K alignment: `GroupK` must equal the ThreadblockShape::kK
//     of `int4_mma_builder.hpp` (both 128). Enforced by static_assert
//     in `int4_mma_builder.hpp`.
//
// Vectorisation: `ElementsPerAccess == 8` → 128-bit store per thread
// → ST.G.E.128 in SASS. This is the fix for sub-bottleneck B1.
//
// ---------------------------------------------------------------------------
// AUX TENSOR POINTERS CARRIED BY Params
// ---------------------------------------------------------------------------
//
// The final `Params` struct (to be defined in L3.6) carries:
//   * `ElementAux const* ptr_scale_u4`   — shape (d_out, n_groups) fp16
//   * `ElementAux const* ptr_zero_u4`    — shape (d_out, n_groups) fp16
//   * `ElementSumX const* ptr_sum_X`     — shape (T,)               int32
//   * `ElementAux const* ptr_scale_x`    — shape (T,)               fp16
//   * `int n_groups`                     — == d_in / 128
//   * `int ld_scale`                     — n_groups (row stride of scale_u4)
//
// The visitor tree picks up `(m, t)` from the `OutputTileIterator`
// coord and `g` from the CTA's K-tile index (accessible via CUTLASS
// kernel Params::grid_tiled_shape and split_k).
//
// ---------------------------------------------------------------------------

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

#include "int4_weight_layout.hpp"

namespace hkust_r50 {
namespace cutlass_int4 {

// ---------------------------------------------------------------------------
// Forward declaration — full definition in L3.6 (kernel/fused_dense_sparse/
// fused_dense_sparse_mma_int4_cutlass_epilogue.hpp or merged into the
// launcher .cu)
// ---------------------------------------------------------------------------

template <
    int    ElementsPerAccess,
    typename ElementAccumulator_,
    typename ElementCompute_,
    typename ElementOutput_,
    int    GroupK
>
struct LinearCombinationDequantizeW4A4;

#if !defined(HKUST_R50_CUTLASS_SMOKE_ONLY)

// In production mode we deliberately leave the primary template
// undefined. Any translation unit that pulls in `int4_mma_builder.hpp`
// WITHOUT `HKUST_R50_CUTLASS_SMOKE_ONLY` expects L3.6 to have supplied
// the specialisation in the launcher TU. If it has not, a clear link /
// instantiation error surfaces instead of silent UB.
//
// Intentionally do not trigger `#error` here: header-only consumers
// (e.g. layout calculator unit tests) that never instantiate the
// template must still compile.

#endif  // !HKUST_R50_CUTLASS_SMOKE_ONLY

// ---------------------------------------------------------------------------
// Helper: per-thread dequant functor — the "pure math" inner-loop body.
//
// This IS available today (CPU-compilable) and is reused verbatim by the
// L3.6 visitor. Extracting it now gives L3.0 acceptance §8.1 (CPU-only
// unit test of the math) a concrete target: compare against the Python
// reference in `layout_calculator.py`.
// ---------------------------------------------------------------------------

struct DequantKernelMath {

    /// Compute a single output element from an int32 accumulator and
    /// the four aux scalars. Matches layout_contract.md §3 exactly.
    CUTLASS_HOST_DEVICE
    static cutlass::half_t apply(
        int32_t acc_s32,
        cutlass::half_t zero_u4,   // (d_out, g) slot, already -8 applied
        cutlass::half_t scale_u4,  // (d_out, g) slot
        int32_t         sum_X,     // (t,) slot
        cutlass::half_t scale_x    // (t,) slot
    ) {
        float f_acc  = static_cast<float>(acc_s32);
        float f_zero = static_cast<float>(zero_u4);
        float f_sum  = static_cast<float>(sum_X);
        float f_scl  = static_cast<float>(scale_u4) * static_cast<float>(scale_x);
        float f_y    = (f_acc - f_zero * f_sum) * f_scl;
        return static_cast<cutlass::half_t>(f_y);
    }
};

// ---------------------------------------------------------------------------
// Compile-time sanity: kGroupK must match ThreadblockShape::kK (=128) so
// exactly one (scale,zero) pair applies to the whole CTA-K slab. We
// cannot static_assert here (no ThreadblockShape visible in this TU),
// so the assertion lives in `int4_mma_builder.hpp`.
// ---------------------------------------------------------------------------

}  // namespace cutlass_int4
}  // namespace hkust_r50
