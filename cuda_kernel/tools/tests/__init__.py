"""Tests for cuda_kernel.tools (pure-Python, CPU-only).

This sub-package lives *outside* ``kernel.cuda_kernel.tests/`` on
purpose: that directory transitively imports ``kernel.cuda_kernel``,
which eagerly JIT-builds the CUDA extension.  Tests in here must
stay importable on any host (Mac, CPU-only CI) without ninja / nvcc.
"""
