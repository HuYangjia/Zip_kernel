// Naive element-wise add: Y_total = Y_low + Y_high
//
// Textbook baseline for the fused dense+sparse epilogue add that the
// optimised kernel in ``csrc/fused_dense_sparse/...`` performs in-register.
// Here we keep it as a separate kernel so the 4-step naive pipeline
// exposes the cost of the reduction as an isolated kernel launch.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>

namespace hkust_v9_naive {
namespace reduce_sum {

static constexpr int kBlockThreads = 256;

__global__ void reduce_sum_naive_kernel(
    const __half* __restrict__ Y_low,     // (d_out, T)
    const __half* __restrict__ Y_high,    // (d_out, T)
    __half*       __restrict__ Y_total,   // (d_out, T)
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        float a = __half2float(Y_low[idx]);
        float b = __half2float(Y_high[idx]);
        Y_total[idx] = __float2half(a + b);
    }
}

void launch(torch::Tensor Y_low, torch::Tensor Y_high, torch::Tensor Y_total) {
    TORCH_CHECK(Y_low.dtype()   == torch::kHalf);
    TORCH_CHECK(Y_high.dtype()  == torch::kHalf);
    TORCH_CHECK(Y_total.dtype() == torch::kHalf);
    TORCH_CHECK(Y_low.sizes()  == Y_high.sizes());
    TORCH_CHECK(Y_low.sizes()  == Y_total.sizes());
    TORCH_CHECK(Y_low.is_contiguous() && Y_high.is_contiguous()
                && Y_total.is_contiguous());

    int64_t numel = Y_low.numel();
    int blocks = static_cast<int>(
        std::min<int64_t>((numel + kBlockThreads - 1) / kBlockThreads,
                          (int64_t)65535));
    if (blocks < 1) blocks = 1;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    reduce_sum_naive_kernel<<<blocks, kBlockThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(Y_low.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(Y_high.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(Y_total.data_ptr<at::Half>()),
        numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace reduce_sum
}  // namespace hkust_v9_naive
