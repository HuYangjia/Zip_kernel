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
// Stage B (diagnostic int32-only, 2026-04-29)
// =========================================================================
//
// Before we wire the full dequant epilogue (stage A1), we first prove
// that the CUTLASS INT4 GEMM really does saturate the Ada Tensor Core
// MMA pipeline. To keep the change minimal and not pollute the final
// signature, this launcher now:
//
//   1. Allocates a scratch `int32 workspace` of shape (d_out, T) with
//      ColumnMajor layout (≡ row-major (T, d_out)).
//   2. Calls the smoke-mode `Int4Gemm` (ElementY=int32,
//      EpilogueOutputOp=LinearCombinationClamp) to fill that workspace.
//   3. Zeros `Y_total` (fp16) so any accidental downstream consumer
//      gets deterministic, well-formed data.
//
// This is *diagnostic*: the caller is NOT supposed to trust `Y_total`
// while stage B is active. The intent is to read `nsys` and SASS on
// the resulting kernel to confirm:
//
//   (a) A launch happens at all (no silent fallthrough).
//   (b) SASS contains `mma.m16n8k64.s4.s4.s32` at ≥ 99% MAC share.
//   (c) Warp-scheduler stall reasons differ from the legacy kernel.
//
// Stage A1 (next iteration) replaces the workspace + zeroing with a
// second tiny CUDA kernel that computes
//     y_fp16[m,t] = (acc_s32[m,t] - zero_u4[m,g] * sum_X[t])
//                   * scale_u4[m,g] * scale_x[t]
// reading the int32 workspace from HBM (one extra round-trip, accepted
// as tradeoff per `.codebuddy/plan/r50_cutlass_int4/` path-A1 decision).

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
    // Stage B diagnostic launcher — see file-header Stage B notes.
    //
    // Unused at Stage B (wired in A1 / visitor-tree second pass):
    (void)W_high_blocks;
    (void)hp_row_offsets; (void)hp_col_indices;
    (void)scale_u4; (void)zero_u4;
    (void)sum_X; (void)scale_x;
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

    // Alignment: CUTLASS int4 GEMM needs 32-element (128-bit) alignment
    // on both operands, i.e. d_in % 32 == 0. Also guard the tile size
    // `ThreadblockShape<128,128,128>` assumption (M=d_out, N=T, K=d_in
    // must each be divisible by the corresponding tile dim).
    TORCH_CHECK(d_in % 128 == 0 && d_out % 128 == 0 && T % 128 == 0,
                "[r50_cutlass_int4] M/N/K must each be divisible by 128 "
                "(d_out=", d_out, ", T=", T, ", d_in=", d_in,
                "). Stage B smoke currently requires tile-aligned shapes; "
                "ragged-T handling lands with Stage A1.");

    using Int4Gemm     = hkust_r50::cutlass_int4::Int4Gemm;
    using ElementY_cuT = hkust_r50::cutlass_int4::ElementY;  // int32 in smoke

    // -----------------------------------------------------------------
    // Allocate int32 workspace for C (MxN = d_out x T, RowMajor).
    // Matches production Y_total layout: `torch.empty((d_out, T), ...)`
    // is physically `(d_out, T) row-major`, so stride(M) = N = T.
    // -----------------------------------------------------------------
    auto workspace_int32 = torch::empty(
        {d_out, T},
        torch::TensorOptions().dtype(torch::kInt32).device(Y_total.device())
    );

    // Build TensorRefs. CUTLASS expects raw pointers in *element* units.
    // For int4b_t the physical pointer is int8, cast via reinterpret.
    auto* w_ptr = reinterpret_cast<cutlass::int4b_t*>(
        W_low.data_ptr<int8_t>());
    auto* x_ptr = reinterpret_cast<cutlass::int4b_t*>(
        X_s4.data_ptr<int8_t>());
    auto* c_ptr = workspace_int32.data_ptr<int32_t>();

    // LayoutA (RowMajor, M=d_out, K=d_in): leading dim = K = d_in.
    // LayoutB (ColumnMajor, K=d_in, N=T):   leading dim = K = d_in.
    //   (X_s4 is torch row-major (T, d_in) with stride(0) = d_in;
    //    reinterpreted as column-major (d_in, T) the leading dim
    //    is still d_in — exactly X_s4.stride(0).)
    // LayoutC (RowMajor, M=d_out, N=T):     leading dim = N = T.
    //   (workspace_int32 is torch row-major (d_out, T) with stride(0)
    //    = T; matches RowMajor leading dim exactly.)

    cutlass::gemm::GemmCoord problem =
        hkust_r50::cutlass_int4::make_problem_size(d_out, d_in, T);

    typename Int4Gemm::Arguments args{
        problem,
        /*ref_A=*/ {w_ptr, typename Int4Gemm::LayoutA{d_in}},
        /*ref_B=*/ {x_ptr, typename Int4Gemm::LayoutB{d_in}},
        /*ref_C=*/ {c_ptr, typename Int4Gemm::LayoutC{T}},
        /*ref_D=*/ {c_ptr, typename Int4Gemm::LayoutC{T}},
        /*epilogue=*/ {/*alpha=*/1, /*beta=*/0}
    };

    Int4Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "[r50_cutlass_int4] Int4Gemm.can_implement failed: ",
                cutlass::cutlassGetStatusString(status),
                " (problem M=", d_out, " N=", T, " K=", d_in, ").");

    status = gemm_op.initialize(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "[r50_cutlass_int4] Int4Gemm.initialize failed: ",
                cutlass::cutlassGetStatusString(status));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    status = gemm_op(stream);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "[r50_cutlass_int4] Int4Gemm.run failed: ",
                cutlass::cutlassGetStatusString(status));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Stage B contract: Y_total is NOT trusted; zero it for safety.
    // A1 will replace this with the dequant kernel that reads
    // workspace_int32 and writes a correct fp16 Y_total.
    Y_total.zero_();

    // `workspace_int32` drops out of scope here; torch's caching
    // allocator will recycle the buffer on the next call.
}

}  // namespace fused_dense_sparse_mma_int4_cutlass
}  // namespace hkust_v9
