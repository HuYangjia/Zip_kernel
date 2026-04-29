"""Device-agnostic reference implementations for W4A4 V9 kernels.

These modules reproduce the *mathematical contract* of the production
CUDA / Triton kernels using pure torch operations.  They run on CPU
(and GPU, unchanged), serve as:

1. Golden references for parity tests (R50 Step 2 verification plan,
   decision D5=E2').
2. Executable documentation of what the kernels must compute; any
   drift between kernel and reference is either a kernel bug or a
   contract change that must be reflected here.

Modules
-------
``fp16_reference``
    ``fp16_dense_reference`` reproduces the dense W4A4 accumulator
    (W_low_packed × X_s4 with groupwise dequant); ``fp16_sparse_reference``
    reproduces the BSR S8×S4 sparse accumulator; ``fp16_fused_reference``
    combines them with the kernel's fused reduction (``Y_low + 16·Y_high``).

``quant_reference``
    Pure-torch implementations of ``quantize_activation_s4`` and
    ``pack_v9_weights`` round-trip helpers, usable as ground truth for
    activation-quant kernel parity.

Tolerances
----------
All references carry intermediate sums in FP32 and cast to FP16 once
at the end, matching the kernel contract.  ``rel_err < 5e-3`` is the
project-wide acceptance bound (see
``kernel/triton_kernel/tests/test_fused_dense_sparse.py`` for precedent).
"""
