"""Baseline integrations for the r79 replacement bench.

This package holds *external* W4A4 baselines used to compare against the
project's own CUDA / Triton kernels.  Right now there is one baseline:

  * ``atom_punica`` — the Atom paper's RTX-4090-tuned INT4 kernels, accessed
    through the e2e Punica integration that ships with the public Atom repo
    (``other_baseline/atom/e2e/punica-atom``).

Design rules (per discussion 2026-05-06):
  1. **Per-kernel timing only** — we do *not* run Atom's full PyTorch fake-
     quant decoder layer (it would mix PyTorch eager overhead in).  Instead
     we time each of the 4 fused GEMMs (qkv / o / gate_up / down) at the
     Qwen3 shapes from ``configs/qwen3_shapes.py`` and rebuild a *layer*
     timing by reusing the BF16 non-GEMM residual that
     ``bench_bf16_per_op.py`` already produced.

  2. **End-to-end-of-replaced-region semantics** — the GEMM cost we report
     is exactly what the Atom kernel takes from quantized input tuple to
     fp16 output (dequant is fused inside the kernel).  The *quantization*
     step itself is part of the preceding op (RMSNorm / activate / reorder),
     which we time separately so the baseline layer sum is the *real* cost
     and not the GEMM-only optimistic number.

  3. **Atom keeper=128 default** — we run Atom in the configuration the
     Atom paper reports (W4A4 + 128 outlier channels in INT8).  This is
     the harder bar to beat and matches what reviewers will compare us to.

  4. **Fail loud, fail early** — every wrapper checks at construction time
     that the punica-atom Python package is importable and that the
     kernels were compiled for the current device's compute capability.
     If you run this on a non-4090 host it will say so immediately rather
     than producing meaningless numbers.

The actual W4A4 kernel does *not* need to be importable on the workstation
where you write code (e.g. macOS) — the wrappers raise a clear
``BaselineUnavailable`` error and the bench scripts skip the run.  This
keeps the source tree portable.
"""

from __future__ import annotations


class BaselineUnavailable(RuntimeError):
    """Raised when a baseline backend cannot be initialised.

    Distinct from generic RuntimeError so the bench driver can decide
    whether to skip the point or abort the whole sweep.
    """


__all__ = ["BaselineUnavailable"]
