"""V9 True Quant inference Triton kernel suite.

Modules
-------
- pack_utils          : offline weight packing utilities (bit-split, BSR layout, 4-bit pack)
- activation_quant    : fused per-token SINT4 activation quantization kernel
- dense_u4s4_gemm     : Kernel (1) - dense UINT4 x SINT4 GEMM (handled as SINT4 x SINT4)
- sparse_s4s4_gemm    : Kernel (2) - 2D block-sparse SINT4 x SINT4 GEMM
- fused_dense_sparse_gemm : Kernel (1+2 fused) - dense + sparse into one pass
                        (used on prefill + hp>0 to save launch + HBM pass)
- v9_linear           : end-to-end V9 Linear forward wrapper
"""
