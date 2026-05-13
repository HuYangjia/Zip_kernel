// Python bindings for the *naive* CUDA kernel suite (reference baseline
// for the optimised W4A4 pipeline in the sibling `csrc/` tree).
//
// The naive path exposes exactly four kernels — no dispatcher, no
// fusion, no per-T specialisation:
//
//   1. activation_quant_naive  (per-token SINT4 quant + pack + sum_X)
//   2. dense_gemm_naive        (UINT4 x SINT4 tiled GEMM, INT4 Tensor Core)
//   3. sparse_gemm_naive       (SINT4 x SINT4 BSR tiled GEMM, INT4 Tensor Core)
//   4. reduce_sum_naive        (element-wise Y_total = Y_low + Y_high)
//
// ABI note: each launcher's signature matches the corresponding
// optimised kernel launcher so the Python wrapper (`ops_naive.py`) can
// keep using the same argument plumbing as `ops.py`.

#include <torch/extension.h>

namespace hkust_v9_naive {

namespace activation_quant {
void launch(torch::Tensor X_fp16, torch::Tensor perm,
            torch::Tensor X_s4, torch::Tensor scale_x,
            torch::Tensor sum_X,
            int T, int D, int bcol);
}

namespace dense_gemm {
void launch(torch::Tensor W_low, torch::Tensor X_s4,
            torch::Tensor scale_u4, torch::Tensor zero_u4,
            torch::Tensor sum_X, torch::Tensor scale_x,
            torch::Tensor Y_low);
}

namespace sparse_gemm {
void launch(torch::Tensor W_high_blocks,
            torch::Tensor hp_row_offsets, torch::Tensor hp_col_indices,
            torch::Tensor X_s4, torch::Tensor scale_u4, torch::Tensor scale_x,
            torch::Tensor Y_high,
            int d_out, int d_in);
}

namespace reduce_sum {
void launch(torch::Tensor Y_low, torch::Tensor Y_high, torch::Tensor Y_total);
}

}  // namespace hkust_v9_naive

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Naive W4A4 CUDA kernel suite (reference baseline)";

    m.def("activation_quant_naive_launch",
          &hkust_v9_naive::activation_quant::launch,
          "Naive per-token SINT4 activation quant (CUDA)",
          py::arg("X_fp16"), py::arg("perm"),
          py::arg("X_s4"), py::arg("scale_x"), py::arg("sum_X"),
          py::arg("T"), py::arg("D"), py::arg("bcol"));

    m.def("dense_gemm_naive_launch",
          &hkust_v9_naive::dense_gemm::launch,
          "Naive UINT4 x SINT4 tiled GEMM (INT4 Tensor Core, single-buffer)",
          py::arg("W_low"), py::arg("X_s4"),
          py::arg("scale_u4"), py::arg("zero_u4"),
          py::arg("sum_X"), py::arg("scale_x"),
          py::arg("Y_low"));

    m.def("sparse_gemm_naive_launch",
          &hkust_v9_naive::sparse_gemm::launch,
          "Naive SINT4 x SINT4 BSR tiled GEMM (INT4 Tensor Core, single-buffer)",
          py::arg("W_high_blocks"),
          py::arg("hp_row_offsets"), py::arg("hp_col_indices"),
          py::arg("X_s4"), py::arg("scale_u4"), py::arg("scale_x"),
          py::arg("Y_high"),
          py::arg("d_out"), py::arg("d_in"));

    m.def("reduce_sum_naive_launch",
          &hkust_v9_naive::reduce_sum::launch,
          "Naive element-wise add: Y_total = Y_low + Y_high",
          py::arg("Y_low"), py::arg("Y_high"), py::arg("Y_total"));
}
