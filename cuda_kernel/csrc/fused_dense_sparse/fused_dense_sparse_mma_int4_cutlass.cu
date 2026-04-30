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
//
// =========================================================================
// Stage A1 (dense, bit-exact for any n_groups) — 2026-04-29
// =========================================================================
//
// To bit-match the legacy per-group dequant semantics without building
// a custom CUTLASS epilogue visitor, we drive `n_groups` separate
// Int4Gemm calls, each consuming a `(BCOL=128)`-wide K-slice of
// (W_low, X_s4).  Each call writes its int32 output to one slice of a
// 3D workspace `(n_groups, d_out, T)`, then a single memory-bound
// `cutlass_dequant::launch_dequant` kernel combines them with the
// per-(m,g) scale/zero metadata into the final fp16 `Y_total`.
//
// Trade-off vs a fused visitor epilogue:
//   +  No CUTLASS epilogue customisation; uses stock `device::Gemm` with
//      `LinearCombinationClamp`.  Correct by construction for any
//      tile_k / stage count.
//   +  Dequant kernel is trivially parallel and memory-bound, staying
//      under 20-30% of Stage B wall time based on Stage B measurements.
//   -  `n_groups` GEMM launches per forward (vs 1).  Each launch is
//      ~5us launch overhead on RTX 4090, which costs up to n_groups*5us
//      extra for small shapes.  Amortised away for shapes with M*T*K
//      >= few GMAC; the T=128 tc_underutil cluster comfortably fits.
//   -  `n_groups` separate int32 HBM writes instead of one (in principle
//      addressable by split-k-serial, but that couples to epilogue work
//      we're avoiding).

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

// Forward declaration — defined in csrc/fused_dense_sparse/cutlass_dequant.cu.
namespace cutlass_dequant {
    void launch_dequant(
        torch::Tensor acc_int32,
        torch::Tensor scale_u4, torch::Tensor zero_u4,
        torch::Tensor sum_X,    torch::Tensor scale_x,
        torch::Tensor Y_total,
        int d_out, int T, int n_groups
    );
    void launch_dequant_accum(
        torch::Tensor acc_int32_g,
        torch::Tensor scale_u4, torch::Tensor zero_u4,
        torch::Tensor sum_X,
        torch::Tensor Y_fp32,
        int d_out, int T, int n_groups, int g
    );
    void launch_finalize_fp32_to_fp16(
        torch::Tensor Y_fp32, torch::Tensor scale_x, torch::Tensor Y_total,
        int d_out, int T
    );
}

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
    // Unused at Stage A1 (sparse path + visitor-tree pass):
    (void)W_high_blocks;
    (void)hp_row_offsets; (void)hp_col_indices;
    (void)InstantiationProbe::gemm_bytes;

    // -----------------------------------------------------------------
    // Input contract checks (must match layout_contract.md §1 T1/T2/T3).
    // -----------------------------------------------------------------
    TORCH_CHECK(W_low.is_cuda() && X_s4.is_cuda() && Y_total.is_cuda(),
                "[r50_cutlass_int4] all tensors must reside on CUDA.");
    TORCH_CHECK(W_low.dtype() == torch::kInt8 && X_s4.dtype() == torch::kInt8,
                "[r50_cutlass_int4] W_low / X_s4 must be int8 (packed int4).");
    TORCH_CHECK(Y_total.dtype() == torch::kHalf,
                "[r50_cutlass_int4] Y_total must be fp16.");
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf &&
                zero_u4.dtype()  == torch::kHalf &&
                scale_x.dtype()  == torch::kHalf,
                "[r50_cutlass_int4] scale_u4/zero_u4/scale_x must be fp16.");
    TORCH_CHECK(sum_X.dtype() == torch::kInt32,
                "[r50_cutlass_int4] sum_X must be int32.");

    const int T = static_cast<int>(X_s4.size(0));
    TORCH_CHECK(X_s4.size(1) == d_in / 2,
                "[r50_cutlass_int4] X_s4.size(1) must equal d_in/2.");
    TORCH_CHECK(W_low.size(0) == d_out && W_low.size(1) == d_in / 2,
                "[r50_cutlass_int4] W_low shape must be (d_out, d_in/2).");
    TORCH_CHECK(Y_total.size(0) == d_out && Y_total.size(1) == T,
                "[r50_cutlass_int4] Y_total shape must be (d_out, T) "
                "(torch row-major). Got (", Y_total.size(0), ",",
                Y_total.size(1), "), expected (", d_out, ",", T, ").");
    TORCH_CHECK(X_s4.stride(1) == 1 && W_low.stride(1) == 1,
                "[r50_cutlass_int4] inner-dim stride must be 1.");

    // F4.1: relax constraints so CUTLASS path can cover real LLM shapes
    // instead of only T%128==0 synthetic cases.
    //   * d_out must still be %128 (ThreadblockShape::kM).
    //   * T is unconstrained for the GEMM itself -- CUTLASS handles
    //     ragged N-tile via partial tiles. The dequant kernel writes
    //     one fp16 per (m,t) so any T>=1 is fine there too.
    //   * d_in must be %BCOL=128 (group structure of scale/zero).
    //   * Sparse (hp) path is deliberately NOT supported in F4 yet;
    //     when the caller passes a non-empty hp_col_indices we fail
    //     loudly so the dispatcher can fall back to the legacy kernel
    //     instead of silently dropping the high-precision contribution.
    constexpr int BCOL = 128;
    const int n_groups = d_in / BCOL;
    TORCH_CHECK(d_in % BCOL == 0,
                "[r50_cutlass_int4] d_in must be multiple of BCOL=128.");
    TORCH_CHECK(d_out % 128 == 0,
                "[r50_cutlass_int4] d_out must be a multiple of 128 "
                "(ThreadblockShape::kM); got d_out=", d_out, ".");
    TORCH_CHECK(T >= 1,
                "[r50_cutlass_int4] T must be >=1; got T=", T, ".");
    TORCH_CHECK(hp_col_indices.numel() == 0,
                "[r50_cutlass_int4] sparse high-precision path is not "
                "implemented on the CUTLASS backend yet; dispatcher "
                "should have routed hp>0 cases to the legacy kernel. "
                "Got hp_col_indices.numel()=", hp_col_indices.numel(), ".");
    TORCH_CHECK(scale_u4.size(0) == d_out && scale_u4.size(1) == n_groups,
                "[r50_cutlass_int4] scale_u4 shape must be (d_out, n_groups).");
    TORCH_CHECK(zero_u4.size(0)  == d_out && zero_u4.size(1)  == n_groups,
                "[r50_cutlass_int4] zero_u4 shape must be (d_out, n_groups).");
    TORCH_CHECK(sum_X.size(0) == T && sum_X.size(1) == n_groups,
                "[r50_cutlass_int4] sum_X shape must be (T, n_groups).");
    TORCH_CHECK(scale_x.size(0) == T,
                "[r50_cutlass_int4] scale_x shape must be (T,).");

    using Int4Gemm = hkust_r50::cutlass_int4::Int4Gemm;

    // -----------------------------------------------------------------
    // Stage A3 workspace layout:
    //   acc_int32_2d : (d_out, T) int32 — re-used across groups
    //   y_fp32       : (d_out, T) fp32  — running dequant accumulator
    //
    // Replaces the Stage A1.5 3D (n_groups, d_out, T) int32 workspace.
    // For gate_up_proj T=512 (d_out=24576, ng=32) this cuts the workspace
    // from 1.5 GB to ~100 MB (48 MB fp32 accumulator + 48 MB int32 acc)
    // and keeps both buffers L2-resident for all d_out*T <= ~4M elements,
    // eliminating the HBM round-trips that made Stage A1.5 8x slower
    // than the hand-rolled fused kernel at T=512.
    // -----------------------------------------------------------------
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(Y_total.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(Y_total.device());
    auto acc_int32_2d = torch::empty({d_out, T}, opts_i32);
    auto y_fp32        = torch::empty({d_out, T}, opts_f32);

    auto* c_base = acc_int32_2d.data_ptr<int32_t>();

    // LayoutA (RowMajor, M=d_out, K=BCOL): leading dim = d_in (row stride of W_low).
    // LayoutB (ColumnMajor, K=BCOL, N=T):   leading dim = d_in (row stride of X_s4).
    // LayoutC (RowMajor, M=d_out, N=T):     leading dim = T.
    const int ld_w = d_in;
    const int ld_x = d_in;
    const int ld_c = T;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    cutlass::gemm::GemmCoord problem{
        /*M=*/d_out, /*N=*/T, /*K=*/BCOL
    };

    auto* w_ptr = reinterpret_cast<cutlass::int4b_t*>(W_low.data_ptr<int8_t>());
    auto* x_ptr = reinterpret_cast<cutlass::int4b_t*>(X_s4.data_ptr<int8_t>());

    Int4Gemm gemm_op;

    // -----------------------------------------------------------------
    // Streamed GEMM + dequant loop.
    //   for each group g in [0, n_groups):
    //       1. Int4Gemm on the g-th BCOL-wide K-slice -> acc_int32_2d
    //       2. dequant_accum_kernel consumes acc_int32_2d and adds into
    //          y_fp32 (overwrites on g==0, adds otherwise)
    //   then: finalize kernel scales by scale_x[t] and casts to fp16.
    //
    // Per-group A/B pointers: each iteration advances the int4 operand
    // pointers by BCOL nibbles (= BCOL/2 bytes, but pointer arithmetic
    // on cutlass::int4b_t already handles the half-byte stride).
    // -----------------------------------------------------------------
    // Reusable contiguous contiguous tensors for the dequant helpers.
    auto scale_u4_c = scale_u4.contiguous();
    auto zero_u4_c  = zero_u4.contiguous();
    auto sum_X_c    = sum_X.contiguous();
    auto scale_x_c  = scale_x.contiguous();

    for (int g = 0; g < n_groups; ++g) {
        // Pointer arithmetic on int4b_t is bit-granular, so advancing by
        // BCOL elements is exactly one (128-element) group in bytes=64.
        auto* w_g = w_ptr + static_cast<int64_t>(g) * BCOL;
        auto* x_g = x_ptr + static_cast<int64_t>(g) * BCOL;

        typename Int4Gemm::Arguments args{
            problem,
            {w_g, typename Int4Gemm::LayoutA{ld_w}},
            {x_g, typename Int4Gemm::LayoutB{ld_x}},
            {c_base, typename Int4Gemm::LayoutC{ld_c}},
            {c_base, typename Int4Gemm::LayoutC{ld_c}},
            {/*alpha=*/1, /*beta=*/0}
        };

        cutlass::Status s = gemm_op.can_implement(args);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "[r50_cutlass_int4] A3 can_implement failed at g=", g,
                    ": ", cutlass::cutlassGetStatusString(s),
                    " (problem M=", d_out, " N=", T, " K=", BCOL, ").");
        s = gemm_op.initialize(args);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "[r50_cutlass_int4] A3 initialize failed at g=", g,
                    ": ", cutlass::cutlassGetStatusString(s));
        s = gemm_op(stream);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "[r50_cutlass_int4] A3 run failed at g=", g,
                    ": ", cutlass::cutlassGetStatusString(s));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        cutlass_dequant::launch_dequant_accum(
            acc_int32_2d, scale_u4_c, zero_u4_c, sum_X_c,
            y_fp32, d_out, T, n_groups, g
        );
    }

    cutlass_dequant::launch_finalize_fp32_to_fp16(
        y_fp32, scale_x_c, Y_total, d_out, T
    );
}

}  // namespace fused_dense_sparse_mma_int4_cutlass
}  // namespace hkust_v9
