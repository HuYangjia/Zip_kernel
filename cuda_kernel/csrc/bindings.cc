// Python bindings for the V9 CUDA kernel suite.
//
// All four sub-kernels are now implemented and bound.  The
// corresponding Python wrappers in ``kernel.cuda_kernel.ops`` provide
// the idiomatic PyTorch-level entry points; this file only exposes
// the raw host-side launchers.

#include <torch/extension.h>

namespace hkust_v9 {
namespace activation_quant {
void launch(torch::Tensor X_fp16, torch::Tensor perm,
            torch::Tensor X_s4, torch::Tensor scale_x,
            torch::Tensor sum_X,
            int T, int D, int bcol);
}  // namespace activation_quant

namespace dense_gemm {
void launch(torch::Tensor W_low, torch::Tensor X_s4,
            torch::Tensor scale_u4, torch::Tensor zero_u4,
            torch::Tensor sum_X, torch::Tensor scale_x,
            torch::Tensor Y_low);
}  // namespace dense_gemm

namespace sparse_gemm {
void launch(torch::Tensor W_high_blocks,
            torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
            torch::Tensor X_s4, torch::Tensor scale_u4, torch::Tensor scale_x,
            torch::Tensor Y_high,
            int d_out, int d_in);
}  // namespace sparse_gemm

namespace fused_dense_sparse {
void launch(torch::Tensor W_low, torch::Tensor W_high_blocks,
            torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
            torch::Tensor X_s4,
            torch::Tensor scale_u4, torch::Tensor zero_u4,
            torch::Tensor sum_X, torch::Tensor scale_x,
            torch::Tensor Y_total,
            int d_out, int d_in);
}  // namespace fused_dense_sparse
}  // namespace hkust_v9

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V9 CUDA kernel suite for SM89 (RTX 4090)";

    m.def(
        "activation_quant_launch",
        &hkust_v9::activation_quant::launch,
        "Fused per-token SINT4 activation quantization (CUDA)",
        py::arg("X_fp16"), py::arg("perm"),
        py::arg("X_s4"), py::arg("scale_x"), py::arg("sum_X"),
        py::arg("T"), py::arg("D"), py::arg("bcol")
    );

    m.def(
        "dense_gemm_launch",
        &hkust_v9::dense_gemm::launch,
        "Dense UINT4 x SINT4 GEMM (CUDA)",
        py::arg("W_low"), py::arg("X_s4"),
        py::arg("scale_u4"), py::arg("zero_u4"),
        py::arg("sum_X"), py::arg("scale_x"),
        py::arg("Y_low")
    );

    m.def(
        "sparse_gemm_launch",
        &hkust_v9::sparse_gemm::launch,
        "Block-sparse SINT4 x SINT4 GEMM (CUDA)",
        py::arg("W_high_blocks"),
        py::arg("hp_row_offsets"), py::arg("hp_col_indices"),
        py::arg("X_s4"), py::arg("scale_u4"), py::arg("scale_x"),
        py::arg("Y_high"),
        py::arg("d_out"), py::arg("d_in")
    );

    m.def(
        "fused_dense_sparse_launch",
        &hkust_v9::fused_dense_sparse::launch,
        "Fused dense + sparse GEMM (CUDA)",
        py::arg("W_low"), py::arg("W_high_blocks"),
        py::arg("hp_row_offsets"), py::arg("hp_col_indices"),
        py::arg("X_s4"),
        py::arg("scale_u4"), py::arg("zero_u4"),
        py::arg("sum_X"), py::arg("scale_x"),
        py::arg("Y_total"),
        py::arg("d_out"), py::arg("d_in")
    );
}
