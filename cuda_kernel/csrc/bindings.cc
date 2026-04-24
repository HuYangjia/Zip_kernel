// Python bindings for the V9 CUDA kernel suite.
//
// The kernel set uses Tensor Core MMA via mma.m16n8k64.s4.s4.s32
// (INT4 Tensor Core on SM89).  INT8 MMA and dp4a paths are archived
// (see kernel/cuda_kernel/VALIDATION_LOG.md, Round 11/12).

#include <torch/extension.h>

namespace hkust_v9 {
namespace activation_quant {
void launch(torch::Tensor X_fp16, torch::Tensor perm,
            torch::Tensor X_s4, torch::Tensor scale_x,
            torch::Tensor sum_X,
            int T, int D, int bcol);
}  // namespace activation_quant

namespace dense_gemm_mma_int4 {
void launch(torch::Tensor W_low, torch::Tensor X_s4,
            torch::Tensor scale_u4, torch::Tensor zero_u4,
            torch::Tensor sum_X, torch::Tensor scale_x,
            torch::Tensor Y_low);
}

namespace dense_gemv_decode {
void launch(torch::Tensor W_low, torch::Tensor X_s4,
            torch::Tensor scale_u4, torch::Tensor zero_u4,
            torch::Tensor sum_X, torch::Tensor scale_x,
            torch::Tensor Y_low);
}

namespace sparse_gemm_mma_int4 {
void launch(torch::Tensor W_high_blocks,
            torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
            torch::Tensor X_s4, torch::Tensor scale_u4, torch::Tensor scale_x,
            torch::Tensor Y_high,
            int d_out, int d_in);
}

namespace fused_dense_sparse_mma_int4 {
void launch(torch::Tensor W_low, torch::Tensor W_high_blocks,
            torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
            torch::Tensor X_s4,
            torch::Tensor scale_u4, torch::Tensor zero_u4,
            torch::Tensor sum_X, torch::Tensor scale_x,
            torch::Tensor Y_total,
            int d_out, int d_in);
}

namespace fused_gemv_decode {
void launch(torch::Tensor W_low, torch::Tensor W_high_blocks,
            torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
            torch::Tensor X_s4,
            torch::Tensor scale_u4, torch::Tensor zero_u4,
            torch::Tensor sum_X, torch::Tensor scale_x,
            torch::Tensor Y_total,
            int d_out, int d_in);
}

namespace fused_quant_gemv {
void launch(torch::Tensor X_fp16, torch::Tensor perm,
            torch::Tensor W_low, torch::Tensor W_high_blocks,
            torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
            torch::Tensor scale_u4, torch::Tensor zero_u4,
            torch::Tensor Y_total,
            int d_out, int d_in);
}
}  // namespace hkust_v9

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V9 CUDA kernel suite for SM89 (RTX 4090) -- INT4 Tensor Core MMA";

    m.def(
        "activation_quant_launch",
        &hkust_v9::activation_quant::launch,
        "Fused per-token SINT4 activation quantization (CUDA)",
        py::arg("X_fp16"), py::arg("perm"),
        py::arg("X_s4"), py::arg("scale_x"), py::arg("sum_X"),
        py::arg("T"), py::arg("D"), py::arg("bcol")
    );

    m.def(
        "dense_gemm_mma_int4_launch",
        &hkust_v9::dense_gemm_mma_int4::launch,
        "Dense UINT4 x SINT4 GEMM via mma.m16n8k64.s4 (CUDA)",
        py::arg("W_low"), py::arg("X_s4"),
        py::arg("scale_u4"), py::arg("zero_u4"),
        py::arg("sum_X"), py::arg("scale_x"),
        py::arg("Y_low")
    );

    m.def(
        "dense_gemv_decode_launch",
        &hkust_v9::dense_gemv_decode::launch,
        "Dense UINT4 x SINT4 GEMV (T=1 decode specialisation, dp4a)",
        py::arg("W_low"), py::arg("X_s4"),
        py::arg("scale_u4"), py::arg("zero_u4"),
        py::arg("sum_X"), py::arg("scale_x"),
        py::arg("Y_low")
    );

    m.def(
        "sparse_gemm_mma_int4_launch",
        &hkust_v9::sparse_gemm_mma_int4::launch,
        "Block-sparse SINT4 x SINT4 GEMM via mma.m16n8k64.s4 (CUDA)",
        py::arg("W_high_blocks"),
        py::arg("hp_row_offsets"), py::arg("hp_col_indices"),
        py::arg("X_s4"), py::arg("scale_u4"), py::arg("scale_x"),
        py::arg("Y_high"),
        py::arg("d_out"), py::arg("d_in")
    );

    m.def(
        "fused_dense_sparse_mma_int4_launch",
        &hkust_v9::fused_dense_sparse_mma_int4::launch,
        "Fused dense + sparse GEMM via mma.m16n8k64.s4 (CUDA)",
        py::arg("W_low"), py::arg("W_high_blocks"),
        py::arg("hp_row_offsets"), py::arg("hp_col_indices"),
        py::arg("X_s4"),
        py::arg("scale_u4"), py::arg("zero_u4"),
        py::arg("sum_X"), py::arg("scale_x"),
        py::arg("Y_total"),
        py::arg("d_out"), py::arg("d_in")
    );

    m.def(
        "fused_gemv_decode_launch",
        &hkust_v9::fused_gemv_decode::launch,
        "Fused dense + sparse GEMV (T=1 decode specialisation, dp4a)",
        py::arg("W_low"), py::arg("W_high_blocks"),
        py::arg("hp_row_offsets"), py::arg("hp_col_indices"),
        py::arg("X_s4"),
        py::arg("scale_u4"), py::arg("zero_u4"),
        py::arg("sum_X"), py::arg("scale_x"),
        py::arg("Y_total"),
        py::arg("d_out"), py::arg("d_in")
    );

    m.def(
        "fused_quant_gemv_launch",
        &hkust_v9::fused_quant_gemv::launch,
        "Fused activation quant + dense+sparse GEMV (T=1, single kernel)",
        py::arg("X_fp16"), py::arg("perm"),
        py::arg("W_low"), py::arg("W_high_blocks"),
        py::arg("hp_row_offsets"), py::arg("hp_col_indices"),
        py::arg("scale_u4"), py::arg("zero_u4"),
        py::arg("Y_total"),
        py::arg("d_out"), py::arg("d_in")
    );
}
