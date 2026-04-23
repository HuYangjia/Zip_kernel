"""Correctness and speedup tests for V9LinearCudaGraph."""
from __future__ import annotations

import time

import pytest
import torch

from kernel.triton_kernel.benchmarks.sweep_v9 import _build_pack
from kernel.triton_kernel.v9_linear import v9_linear_forward_decode
from kernel.triton_kernel.v9_linear_graph import V9LinearCudaGraph


def _has_cuda():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    "T,d_out,d_in,hp_ratio",
    [
        (1,  4096, 4096, 0.0),
        (1,  4096, 4096, 0.05),
        (1, 14336, 4096, 0.0),
        (1, 14336, 4096, 0.05),
        (16, 4096, 4096, 0.0),
        (16, 4096, 4096, 0.10),
        (64, 4096, 4096, 0.0),
    ],
)
def test_graph_matches_eager(T, d_out, d_in, hp_ratio):
    """Graph replay must bit-exactly match eager decode path.

    We compare to the eager decode path (``_v9_forward_decode``), not to
    FP16 cuBLAS -- V9 is an approximation by construction, so the goal
    is that graph-capture introduces zero extra error.
    """
    torch.manual_seed(0)
    W = _build_pack(d_out, d_in, hp_ratio)
    X = torch.randn(T, d_in, device="cuda", dtype=torch.float16)

    y_eager = v9_linear_forward_decode(X, W)

    graph_fn = V9LinearCudaGraph(W)
    y_graph = graph_fn(X)

    # Graph replay runs exactly the same kernels with the same inputs,
    # so we expect bitwise equality after static_X is populated from X.
    assert y_eager.shape == y_graph.shape
    # Float tolerance is a no-op here but defensive.
    torch.testing.assert_close(
        y_graph, y_eager, atol=1e-5, rtol=0, msg="graph replay diverges from eager"
    )


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_graph_replay_is_strictly_faster_than_eager():
    """End-to-end: a second replay should be faster than eager by
    a clear margin on a launch-bound shape.

    We use T=1, d_out=14336, d_in=4096 (top-wins-in-sweep case) which
    the microbench shows gets -49% from CUDA Graph.  We use a relaxed
    threshold of -20% so the test remains stable across noisy
    environments.
    """
    torch.manual_seed(0)
    T, d_out, d_in, hp = 1, 14336, 4096, 0.05
    W = _build_pack(d_out, d_in, hp)
    X = torch.randn(T, d_in, device="cuda", dtype=torch.float16)

    graph_fn = V9LinearCudaGraph(W)
    # Warmup eager and graph paths.
    for _ in range(50):
        v9_linear_forward_decode(X, W)
        graph_fn(X)
    torch.cuda.synchronize()

    def timer(fn, iters=300):
        # Three windows, take min-of-means per team micro-timer convention.
        means = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            means.append((time.perf_counter() - t0) / iters)
        return min(means)

    t_eager = timer(lambda: v9_linear_forward_decode(X, W))
    t_graph = timer(lambda: graph_fn(X))

    # Expect graph to beat eager by >= 20% on this decode shape.
    assert t_graph < t_eager * 0.80, (
        f"CUDA Graph didn't beat eager by >=20%: "
        f"eager={t_eager*1e6:.1f}us, graph={t_graph*1e6:.1f}us"
    )


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_graph_cache_reuse():
    """Second call with the same shape hits the cache without re-capture."""
    torch.manual_seed(0)
    W = _build_pack(4096, 4096, 0.0)
    X1 = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
    X2 = torch.randn(1, 4096, device="cuda", dtype=torch.float16)

    graph_fn = V9LinearCudaGraph(W)
    _ = graph_fn(X1)
    n_before = len(graph_fn._graphs)
    _ = graph_fn(X2)
    n_after = len(graph_fn._graphs)
    assert n_before == 1 and n_after == 1, "same-shape calls should share graph"

    # Different T -> new graph entry.
    X3 = torch.randn(16, 4096, device="cuda", dtype=torch.float16)
    _ = graph_fn(X3)
    assert len(graph_fn._graphs) == 2


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_graph_prefill_falls_through_to_eager():
    """T > DECODE_T_THRESHOLD should bypass graph and go eager."""
    torch.manual_seed(0)
    W = _build_pack(4096, 4096, 0.0)
    # T > 128 triggers prefill path.
    X = torch.randn(256, 4096, device="cuda", dtype=torch.float16)
    graph_fn = V9LinearCudaGraph(W)
    y = graph_fn(X)
    assert y.shape == (256, 4096)
    # No graph should have been captured.
    assert len(graph_fn._graphs) == 0
