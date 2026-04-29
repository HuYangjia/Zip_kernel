"""CPU-only helpers for the cuda_kernel extension.

Currently exports :mod:`layout_calculator` (R50 L3.0). The module is
importable on any host; it never pulls in torch or the compiled
extension.
"""

from . import layout_calculator  # noqa: F401

__all__ = ["layout_calculator"]
