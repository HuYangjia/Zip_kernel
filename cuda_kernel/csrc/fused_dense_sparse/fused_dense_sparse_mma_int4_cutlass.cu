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

    // Stage A1 alignment. M=d_out must be %128 to fit
    // ThreadblockShape<128,128,128>. K-per-slice is BCOL=128 (the V9
    // group size); N=T can be anything >=1 because CUTLASS handles
    // ragged N via partial tiles, but for now the tc_underutil
    // cluster always has T==128 so we keep the check strict.
    constexpr int BCOL = 128;
    const int n_groups = d_in / BCOL;
    TORCH_CHECK(d_in % BCOL == 0,
                "[r50_cutlass_int4] d_in must be multiple of BCOL=128.");
    TORCH_CHECK(d_out % 128 == 0 && T % 128 == 0,
                "[r50_cutlass_int4] d_out and T must each be multiples "
                "of 128 (d_out=", d_out, ", T=", T, ").");
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
    // Allocate int32 workspace: (n_groups, d_out, T) row-major.
    // Contiguous by construction; group g lives at offset g*d_out*T.
    // -----------------------------------------------------------------
    auto workspace_int32 = torch::empty(
        {n_groups, d_out, T},
        torch::TensorOptions().dtype(torch::kInt32).device(Y_total.device())
    );

    auto* c_base = workspace_int32.data_ptr<int32_t>();

    // LayoutA (RowMajor, M=d_out, K=BCOL): leading dim = d_in (row stride of W_low).
    // LayoutB (ColumnMajor, K=BCOL, N=T):   leading dim = d_in (row stride of X_s4).
    // LayoutC (RowMajor, M=d_out, N=T):     leading dim = T.
    const int ld_w = d_in;   // bytes/2-worth of int4 per row of W_low
    const int ld_x = d_in;
    const int ld_c = T;

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    cutlass::gemm::GemmCoord problem{
        /*M=*/d_out, /*N=*/T, /*K=*/BCOL
    };

    for (int g = 0; g < n_groups; ++g) {
        // Slice g covers K-range [g*BCOL, (g+1)*BCOL) of both operands.
        // CAUTION: `int4b_t` has sizeof==1 byte (it holds a uint8_t
        // storage field), so `int4b_t* + BCOL` would step BCOL BYTES
        // (= 2*BCOL int4 elements), which is wrong.  Do the arithmetic
        // on byte pointers and reinterpret back.
        const int byte_off = (g * BCOL) / 2;   // int4 elements -> bytes
        auto* w_g = reinterpret_cast<cutlass::int4b_t*>(
            W_low.data_ptr<int8_t>() + byte_off);
        auto* x_g = reinterpret_cast<cutlass::int4b_t*>(
            X_s4.data_ptr<int8_t>()  + byte_off);
        auto* c_g = c_base + static_cast<int64_t>(g) * d_out * T;

        typename Int4Gemm::Arguments args{
            problem,
            {w_g, typename Int4Gemm::LayoutA{ld_w}},
            {x_g, typename Int4Gemm::LayoutB{ld_x}},
            {c_g, typename Int4Gemm::LayoutC{ld_c}},
            {c_g, typename Int4Gemm::LayoutC{ld_c}},
            {/*alpha=*/1, /*beta=*/0}
        };

        Int4Gemm gemm_op;
        cutlass::Status s = gemm_op.can_implement(args);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "[r50_cutlass_int4] can_implement failed at group ", g,
                    ": ", cutlass::cutlassGetStatusString(s),
                    " (problem M=", d_out, " N=", T, " K=", BCOL, ").");
        s = gemm_op.initialize(args);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "[r50_cutlass_int4] initialize failed at group ", g,
                    ": ", cutlass::cutlassGetStatusString(s));
        s = gemm_op(stream);
        TORCH_CHECK(s == cutlass::Status::kSuccess,
                    "[r50_cutlass_int4] run failed at group ", g, ": ",
                    cutlass::cutlassGetStatusString(s));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // -----------------------------------------------------------------
    // Dequant: (n_groups, d_out, T) int32 + metadata -> (d_out, T) fp16
    // Declared in csrc/fused_dense_sparse/cutlass_dequant.cu.
    // -----------------------------------------------------------------
    cutlass_dequant::launch_dequant(
        workspace_int32,
        scale_u4.contiguous(),
        zero_u4.contiguous(),
        sum_X.contiguous(),
        scale_x.contiguous(),
        Y_total,
        d_out, T, n_groups
    );
}

}  // namespace fused_dense_sparse_mma_int4_cutlass
}  // namespace hkust_v9
