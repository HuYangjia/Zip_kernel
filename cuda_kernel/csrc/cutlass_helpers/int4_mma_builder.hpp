// Copyright (c) 2026 HKUST R50 — CUTLASS INT4 W4A4 Gemm type factory.
//
// L3.4 of plan r50_cutlass_int4. Consumer of:
//   - layout_contract.md §2 (tile schedule)
//   - layout_contract.md §D.1 / §D.5 (ArchTag=Sm80, v2.11 device::Gemm)
//   - layout_contract.md §D.6 (LayoutC=ColumnMajor)
//   - int4_weight_layout.hpp (element / layout aliases)
//   - int4_epilogue_dequant.hpp (LinearCombinationDequantizeW4A4)
//
// The type `Int4Gemm` below is the single Gemm instantiation the runtime
// launcher calls. `.cu` translation units should use this alias only;
// altering the tile schedule means editing this header and re-signing
// the layout contract (per invariant I-L3).
//
// Evidence that the (ElementA=int4b_t RowMajor, ElementB=int4b_t ColumnMajor,
// ArchTag=Sm80, TileShape<128,128,128>, WarpShape<64,64,128>, InstShape<16,8,64>)
// combination is a compilable CUTLASS 2.11 specialisation:
//   tests/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sm80.cu:253
// (officially-validated case with GemmShape<128,128,128>).
//
// Header-only. Safe to include from both .cu and .cpp.

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

#include "int4_weight_layout.hpp"
#include "int4_epilogue_dequant.hpp"

namespace hkust_r50 {
namespace cutlass_int4 {

// ---------------------------------------------------------------------------
// 1. Tile schedule (frozen by layout_contract.md §2.2 + I-L3)
// ---------------------------------------------------------------------------
//
// Tile choice rationale:
//   - ThreadblockShape<128,128,128>: matches `tile_k == BCOL == 128`
//     so exactly one scale / zero load is needed per CTA-K iteration
//     (layout_contract.md §D.4).
//   - WarpShape<64,64,128>: 4 warps per CTA = 2 M-tiles × 2 N-tiles × 1 K-tile,
//     each warp does 4 × (16×8) MMA atoms along M,N and 2 atoms along K.
//     Canonical for CUTLASS 2.x Sm80 MmaMultistage.
//   - InstructionShape<16,8,64>: non-negotiable (SM80_16x8x64_S32S4S4S32_TN
//     atom, invariant I-L2).
//   - Stages=3: 3-stage cp.async; addresses B3 (pipeline starvation).
//     3×(128×128 int4 A + 128×128 int4 B) = 48KB shmem, well under
//     Ada 100KB budget.

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 128>;
using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 128>;
using InstructionShape = cutlass::gemm::GemmShape< 16,   8,  64>;
static constexpr int kStages = 3;

using OperatorClass = cutlass::arch::OpClassTensorOp;
using ArchTag       = cutlass::arch::Sm80;  // see §D.1; compile target still sm_89

using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;

// ---------------------------------------------------------------------------
// 2. Epilogue binding (instantiated from int4_epilogue_dequant.hpp)
// ---------------------------------------------------------------------------
//
// `LinearCombinationDequantizeW4A4<...>` is our custom visitor tree that
// implements the dequant formula (see layout_contract.md §3). The 2nd
// template arg is `ElementsPerAccess = 8` which gives the 128-bit vector
// store required to kill sub-bottleneck B1.
//
// NOTE: for L3.5 smoke compilation we allow the visitor to fall back to
// the canonical `LinearCombinationClamp<half_t, 8, int32, float>` when
// the macro `HKUST_R50_CUTLASS_SMOKE_ONLY` is defined — this lets the
// Mac-side `clang -fsyntax-only` path close before the full visitor is
// implemented. The production path always uses the real functor.

#if defined(HKUST_R50_CUTLASS_SMOKE_ONLY)
  #include "cutlass/epilogue/thread/linear_combination_clamp.h"
  // In smoke mode ElementY is downgraded to int32 (see
  // int4_weight_layout.hpp for rationale). ElementsPerAccess is
  // therefore 128/32 = 4, not 8 — hard-coding 8 would trigger
  // DefaultIteratorsTensorOp assertion failures.
  static constexpr int kEpaSmoke =
      128 / cutlass::sizeof_bits<ElementY>::value;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
      ElementY,                           // ElementOutput (int32 in smoke)
      kEpaSmoke,                          // ElementsPerAccess = 4 in smoke
      ElementAcc,                         // ElementAccumulator = int32
      ElementCompute                      // ElementCompute = float
  >;
#else
  using EpilogueOutputOp = LinearCombinationDequantizeW4A4<
      kElementsPerAccessEpilogue,         // ElementsPerAccess = 8
      ElementAcc,                         // ElementAccumulator = int32
      ElementCompute,                     // ElementCompute = float
      ElementY,                           // ElementOutput = half_t
      /*GroupK=*/128                      // must == tile_k (§D.4)
  >;
#endif

// ---------------------------------------------------------------------------
// 3. The Gemm alias (the single entry point for L3.6 launcher)
// ---------------------------------------------------------------------------

using Int4Gemm = cutlass::gemm::device::Gemm<
    /*ElementA            */ ElementW,
    /*LayoutA             */ LayoutW,
    /*ElementB            */ ElementX,
    /*LayoutB             */ LayoutX,
    /*ElementC            */ ElementY,
    /*LayoutC             */ LayoutY,
    /*ElementAccumulator_ */ ElementAcc,
    /*OperatorClass_      */ OperatorClass,
    /*ArchTag_            */ ArchTag,
    /*ThreadblockShape_   */ ThreadblockShape,
    /*WarpShape_          */ WarpShape,
    /*InstructionShape_   */ InstructionShape,
    /*EpilogueOutputOp_   */ EpilogueOutputOp,
    /*ThreadblockSwizzle_ */ ThreadblockSwizzle,
    /*Stages              */ kStages,
    /*AlignmentA          */ kAlignmentW,
    /*AlignmentB          */ kAlignmentX,
    /*SplitKSerial        */ false
>;

// ---------------------------------------------------------------------------
// 4. Static contract checks (compile-time errors if something drifts)
// ---------------------------------------------------------------------------

static_assert(ThreadblockShape::kK == 128,
              "tile_k must equal BCOL=128 per layout_contract.md §D.4");
static_assert(InstructionShape::kM == 16 && InstructionShape::kN == 8 &&
              InstructionShape::kK == 64,
              "InstructionShape must be <16,8,64> per I-L2 (atom 16x8x64).");
static_assert(kStages == 3,
              "Stages must be 3 to address sub-bottleneck B3.");

}  // namespace cutlass_int4
}  // namespace hkust_r50
