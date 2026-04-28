"""Lightweight NVTX range shim for HKUST V9 kernel profiling.

Usage
-----
Set ``HKUST_V9_PROFILE=1`` before running to activate NVTX ranges.
When the env var is absent or not "1", all calls are no-ops and the
``torch.cuda.nvtx`` module is never imported — zero overhead on the
production hot path.

Example::

    from kernel.tools.profile.nvtx_shim import range_push, range_pop, nvtx_range

    range_push("my_section")
    try:
        do_work()
    finally:
        range_pop()

    # Or use the context manager:
    with nvtx_range("my_section"):
        do_work()
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

_ENABLED: bool = os.environ.get("HKUST_V9_PROFILE", "0") == "1"

# Lazily resolved nvtx handle — only imported when profiling is active.
_nvtx = None


def _get_nvtx():
    global _nvtx
    if _nvtx is None:
        import torch.cuda.nvtx as _m  # noqa: WPS433 — intentional lazy import
        _nvtx = _m
    return _nvtx


def range_push(msg: str) -> None:
    """Push an NVTX range.  No-op when profiling is disabled."""
    if _ENABLED:
        _get_nvtx().range_push(msg)


def range_pop() -> None:
    """Pop the current NVTX range.  No-op when profiling is disabled."""
    if _ENABLED:
        _get_nvtx().range_pop()


@contextmanager
def nvtx_range(msg: str) -> Generator[None, None, None]:
    """Context manager that wraps a block in an NVTX range.

    Always uses try/finally so the pop is guaranteed even on exceptions.
    When profiling is disabled this is a zero-cost pass-through.
    """
    if _ENABLED:
        _get_nvtx().range_push(msg)
        try:
            yield
        finally:
            _get_nvtx().range_pop()
    else:
        yield


__all__ = ["range_push", "range_pop", "nvtx_range", "_ENABLED"]
