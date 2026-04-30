// Fused Activation Quantization + Dense+Sparse GEMM (T>1, SM89).
//
// P0 (plan docs/P0_QUANT_FUSION_SPIKE.md): folds activation_quant into
// the prologue of fused_dense_sparse_mma_int4.  Removes the ~16us
// launch-overhead floor that dominates small/mid-shape kernels on
// RTX 4090 (2 × 7us cudaLaunchKernel + HBM round-trip of X_s4/sum_X).
//
// Relationship to siblings:
//   - fused_quant_gemv.cu      : T=1 dp4a path, already has quant fused.
//   - fused_dense_sparse_mma_int4.cu : T>1 legacy MMA path, quant is a
//                                      separate launch (activation_quant.cu).
//   - THIS FILE                : T>1 MMA path WITH quant fused into prologue.
//
// Design (see docs/P0_QUANT_FUSION_SPIKE.md §3-§6 for rationale):
//   - CTA layout mirrors the legacy kernel: blockDim = (kBm, 1, 1),
//     gridDim = (ceil(d_out/kBm), ceil(T/kBn), split_k).
//   - Prologue Phase 1 (max-abs scan over all D cols per assigned token):
//       All kBm threads cooperatively scan X[t_global, perm[0..D)] for
//       each of kBn tokens this CTA owns; warp-tree reduce → sScaleX[kBn].
//   - Prologue Phase 2 merged INTO main K-loop:
//       For each group g, fill sX[buf^1][...] by reading fp16 X directly
//       and quantizing on the fly (instead of cp.async from HBM X_s4).
//       Also compute sum_X[n_local, g] and store to s_sum_X[buf^1].
//   - MMA mainloop body unchanged bit-for-bit (reuses run_mma_pass).
//
// Status (P0.1): SKELETON — host launcher is a TORCH_CHECK(false) stub;
// kernel is empty.  Purpose: prove the build system accepts the new TU
// and the pybind binding routes correctly.  Kernel body lands in P0.2.

#include "common/arch.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace hkust_v9 {
namespace fused_quant_dense_sparse_mma_int4 {

// -------------------------------------------------------------------
// Host launcher (stub — P0.1)
// -------------------------------------------------------------------
//
// ABI contract (will be frozen in P0.2):
//   Inputs:
//     X_fp16           : (T, D) fp16  — raw activations, NOT quantized
//     perm             : (D,)   int32 — column permutation for quant
//     W_low            : (d_out, d_in/2) int8 — UINT4 packed
//     W_high_blocks    : BSR sparse residual (same as legacy)
//     hp_row_offsets   : BSR row offsets
//     hp_col_indices   : BSR col indices
//     scale_u4         : (d_out, n_groups) fp16
//     zero_u4          : (d_out, n_groups) fp16
//   Output:
//     Y_total          : (d_out, T) fp16
//   Scalars:
//     d_out, d_in      : matrix dimensions (d_in must be % BCOL == 0)
//
// Note: scale_x and sum_X are produced INTERNALLY (never written to HBM).
//       X_s4 likewise is ephemeral (lives only in smem).
// -------------------------------------------------------------------

void launch(
    torch::Tensor X_fp16, torch::Tensor perm,
    torch::Tensor W_low, torch::Tensor W_high_blocks,
    torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
    torch::Tensor scale_u4, torch::Tensor zero_u4,
    torch::Tensor Y_total,
    int d_out, int d_in
) {
    // P0.1 stub: intentionally unimplemented.  Callers must fall back
    // to the legacy (activation_quant + fused_dense_sparse) pipeline.
    // The stub still validates dtypes so any accidentally-routed call
    // fails fast and informatively.
    TORCH_CHECK(X_fp16.dtype() == torch::kHalf, "X_fp16 must be fp16");
    TORCH_CHECK(perm.dtype() == torch::kInt32, "perm must be int32");
    TORCH_CHECK(W_low.dtype() == torch::kInt8, "W_low must be int8");
    TORCH_CHECK(scale_u4.dtype() == torch::kHalf, "scale_u4 must be fp16");
    TORCH_CHECK(zero_u4.dtype()  == torch::kHalf, "zero_u4 must be fp16");
    TORCH_CHECK(Y_total.dtype()  == torch::kHalf, "Y_total must be fp16");
    TORCH_CHECK(d_in % BCOL == 0);
    TORCH_CHECK(false,
                "fused_quant_dense_sparse_mma_int4::launch is not implemented "
                "in P0.1 (skeleton only).  Use activation_quant_cuda + "
                "fused_dense_sparse_cuda_int4 as the production path.");
    // Silence unused-parameter warnings.
    (void)X_fp16; (void)perm; (void)W_low; (void)W_high_blocks;
    (void)hp_row_offsets; (void)hp_col_indices;
    (void)scale_u4; (void)zero_u4; (void)Y_total;
    (void)d_out; (void)d_in;
}

}  // namespace fused_quant_dense_sparse_mma_int4
}  // namespace hkust_v9

