"""CUDA Graph wrapper for V9 Linear decode-regime inference.

Motivation
----------
The V9 decode path launches at least 3 Triton kernels (quant, dense,
optionally sparse) plus the final combine/transpose.  Each launch costs
~30-50us of host-side overhead (CUDA driver + PyTorch framework);
combined with the actual kernel work (~10-70us depending on shape),
the launch tax dominates for decode shapes ``T <= 16`` --
it sits at 50-75% of wall time in the sweep data.

CUDA Graph replay replaces all those launches with a single
``cudaGraphLaunch`` (~5us), reclaiming almost the full overhead.
Measured savings on RTX 4090 (see ``bench_decode_launch_overhead.py``):

    T=1, 14336x4096, hp=0    : 192us -> 98us   (-49%)
    T=1, 14336x4096, hp=0.05 : 304us -> 122us  (-60%)
    T=1,  4096x4096, hp=0    : 192us -> 84us   (-56%)

Usage
-----
    from kernel.triton_kernel.v9_linear_graph import V9LinearCudaGraph

    packed_W = pack_v9_weights(...)
    graph_fn = V9LinearCudaGraph(packed_W)     # weight is captured-by-ref

    # First call of a new (T, d_in, d_out) shape triggers a one-time
    # warmup + capture (a few ms).  Subsequent calls of the same shape
    # are graph.replay() + H2D input copy.
    y = graph_fn(x)         # x: (T, d_in) fp16 on CUDA

Constraints
-----------
- Only supports shapes with ``T <= DECODE_T_THRESHOLD`` (128 by default).
  Larger shapes bypass the graph (the prefill regime is not launch-bound).
- Weight tensors ``W`` are captured by reference; do NOT mutate them in
  place after creating a graph for a given shape, or the graph will
  observe stale values.  If you need to update weights, build a new
  ``V9LinearCudaGraph`` (cheap after warmup of each shape).
- Input tensor contents are copied into a static input buffer on each
  call; this ``copy_`` is NOT captured and costs ~1-2us.  The output
  tensor is returned as a clone of the static output buffer for
  isolation (so the caller can keep the returned tensor alive while the
  graph is replayed again).

Safety
------
The graph is captured on a side CUDA stream to avoid interference with
the default compute stream.  After capture, ``replay()`` uses whichever
stream is current at call time (``torch.cuda.current_stream()``), which
matches PyTorch's normal eager execution semantics.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from .v9_linear import (
    DECODE_T_THRESHOLD,
    V9WeightContainer,
    _v9_forward_decode,
    v9_linear_forward,
)


# (T, d_in, d_out, dtype) is the graph-cache key.  hp_ratio is implicit
# in the captured weight (the shapes it induces are constant across
# weight-preserving calls).
_GraphKey = Tuple[int, int, int, torch.dtype]


class V9LinearCudaGraph:
    """Graph-captured V9 linear forward for decode shapes.

    A separate ``cudaGraph`` is captured per ``(T, d_in, d_out, dtype)``
    key.  The weight container is bound at construction time; do not
    mutate its tensors after first capture.
    """

    def __init__(self, W: V9WeightContainer, warmup_iters: int = 3):
        self.W = W
        self.d_in = int(W.d_in)
        self.d_out = int(W.d_out)
        self.warmup_iters = int(warmup_iters)

        # Per-shape graph cache.  Holds (graph, static_input, static_output).
        self._graphs: Dict[
            _GraphKey, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]
        ] = {}

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _capture(self, T: int, dtype: torch.dtype, device: torch.device):
        """Capture a new graph for the given decode shape."""
        # Build static input buffer.  Content is irrelevant (graph
        # captures kernel launches, not data).
        static_X = torch.empty((T, self.d_in), dtype=dtype, device=device)
        static_X.normal_()

        # Warmup so any autotune / first-launch compile happens outside
        # the capture region (otherwise the capture may stall).
        for _ in range(self.warmup_iters):
            _ = _v9_forward_decode(
                static_X, self.W, T=T, d_out=self.d_out, d_in=self.d_in
            )
        torch.cuda.synchronize()

        # Capture on a side stream per pytorch docs.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_Y = _v9_forward_decode(
                    static_X, self.W, T=T, d_out=self.d_out, d_in=self.d_in
                )
        torch.cuda.current_stream().wait_stream(s)

        return g, static_X, static_Y

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def __call__(self, X_fp16: torch.Tensor) -> torch.Tensor:
        """Run V9 linear forward; dispatches between graph and eager."""
        assert X_fp16.is_cuda and X_fp16.dtype == torch.float16

        original_shape = X_fp16.shape
        if X_fp16.shape[-1] != self.d_in:
            raise ValueError(
                f"X last dim ({X_fp16.shape[-1]}) != d_in ({self.d_in})"
            )
        X_2d = X_fp16.reshape(-1, self.d_in)
        T = int(X_2d.shape[0])

        # Prefill shapes are not launch-bound; let them go eager.
        if T > DECODE_T_THRESHOLD:
            return v9_linear_forward(X_fp16, self.W)

        key: _GraphKey = (T, self.d_in, self.d_out, X_fp16.dtype)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(T, X_fp16.dtype, X_fp16.device)
            self._graphs[key] = entry
        g, static_X, static_Y = entry

        # Copy caller's input into the static buffer, replay, clone out.
        static_X.copy_(X_2d, non_blocking=True)
        g.replay()
        out_shape = original_shape[:-1] + (self.d_out,)
        return static_Y.clone().reshape(out_shape)


__all__ = ["V9LinearCudaGraph"]
