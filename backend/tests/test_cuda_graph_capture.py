"""CUDA-only integration tests for ``kernel.backend.graph_cache`` (D2).

Scope (D2):
  - Capture succeeds for a minimal decode shape.
  - Replay numerically matches eager (max abs diff <= 1e-3 fp16).
  - Cache-hit path runs replay, not a second capture.
  - Prewarm captures multiple shapes in one call.
  - Re-entrancy guard rejects nested capture cleanly.
  - Capture-failure sentinel demotes the shape to eager on subsequent calls.

These tests require a CUDA device and the V9 CUDA backend.  They are
skipped on machines without CUDA so the module can still be imported
in CPU-only test environments (D1 skeleton tests).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required for graph-cache capture tests",
                allow_module_level=True)

from kernel.backend import graph_cache as gc_mod
from kernel.backend import v9_linear_forward
# Reuse the canonical Phase-1 shape builder so we test exactly the
# same (X, W) distribution as every other bench.
from kernel.tools.profile._phase1_shapes import build_shape_inputs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A small, cheap shape that exercises the launch_sparse cluster
# (T in {1,2,4,8}, hp_blocks > 0).  Picked from the 17 audit shapes;
# any tag starting with "T1_" in Qwen3-0.6B works.  We fall back to
# a hand-built shape if the canonical catalogue doesn't carry it.
_SMALL_CLUSTER_TAG = "decode_T1_q_2048_2048"


@pytest.fixture(scope="module")
def small_inputs():
    """Return (X, W) for a small launch-tax-sensitive shape."""
    try:
        b = build_shape_inputs(_SMALL_CLUSTER_TAG)
    except Exception:
        pytest.skip(f"shape catalogue missing tag {_SMALL_CLUSTER_TAG}")
    return b.X, b.W


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    prev = gc_mod.get_cuda_graph_policy()
    gc_mod.clear_cuda_graph_cache()
    # Fresh counters per test.
    gc_mod._cache = gc_mod._GraphCache()
    gc_mod.set_cuda_graph_policy("off")
    yield
    gc_mod.clear_cuda_graph_cache()
    gc_mod.set_cuda_graph_policy(prev)
    gc_mod._cache = gc_mod._GraphCache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_capture_succeeds_and_stats_reflect(small_inputs):
    X, W = small_inputs
    gc_mod.set_cuda_graph_policy("force")
    y = gc_mod.v9_linear_forward_cuda_graph(X, W)
    assert y.is_cuda
    assert y.dtype == X.dtype
    stats = gc_mod.cuda_graph_cache_stats()
    assert stats["n_entries"] == 1, stats
    assert stats["n_capture_fail"] == 0, stats


def test_replay_matches_eager_within_fp16_tolerance(small_inputs):
    X, W = small_inputs
    # Baseline.
    y_eager = v9_linear_forward(X, W)
    # Graph path.
    gc_mod.set_cuda_graph_policy("force")
    y_graph = gc_mod.v9_linear_forward_cuda_graph(X, W)
    # Shapes must match exactly.
    assert y_eager.shape == y_graph.shape
    # fp16 numerics: we accept up to 1e-3 max-abs, matching the policy
    # layer's existing parity gate.
    diff = (y_eager.float() - y_graph.float()).abs().max().item()
    assert diff <= 1e-3, f"graph vs eager max_abs_diff={diff:.2e}"


def test_second_call_replays_does_not_recapture(small_inputs):
    X, W = small_inputs
    gc_mod.set_cuda_graph_policy("force")
    _ = gc_mod.v9_linear_forward_cuda_graph(X, W)     # captures
    stats_after_capture = gc_mod.cuda_graph_cache_stats()
    _ = gc_mod.v9_linear_forward_cuda_graph(X, W)     # should replay
    stats_after_replay = gc_mod.cuda_graph_cache_stats()
    # No new entries, no new capture failures.
    assert stats_after_replay["n_entries"] == stats_after_capture["n_entries"]
    assert stats_after_replay["n_capture_fail"] == stats_after_capture["n_capture_fail"]
    # One additional cache hit.
    assert stats_after_replay["n_hit"] == stats_after_capture["n_hit"] + 1


def test_different_shape_triggers_second_capture(small_inputs):
    """Two distinct shapes should yield two cache entries, each with
    its own graph."""
    X1, W = small_inputs
    # Build a second X with a different T but the same W by concatenating.
    # This stays on the same launch_sparse cluster (T=2).
    if X1.shape[0] >= 2:
        pytest.skip("small_inputs already has T>=2; can't build distinct T=2 fixture")
    X2 = torch.cat([X1, X1], dim=0).contiguous()
    gc_mod.set_cuda_graph_policy("force")
    _ = gc_mod.v9_linear_forward_cuda_graph(X1, W)
    _ = gc_mod.v9_linear_forward_cuda_graph(X2, W)
    stats = gc_mod.cuda_graph_cache_stats()
    assert stats["n_entries"] == 2, stats


def test_prewarm_captures_multiple_shapes(small_inputs):
    X1, W = small_inputs
    X2 = torch.cat([X1, X1], dim=0).contiguous()   # T=2
    X4 = torch.cat([X2, X2], dim=0).contiguous()   # T=4
    gc_mod.set_cuda_graph_policy("force")
    res = gc_mod.prewarm_cuda_graph_cache([(X1, W), (X2, W), (X4, W)])
    assert res["attempted"] == 3, res
    assert res["captured"] == 3, res
    assert res["skipped"] == 0, res
    assert gc_mod.cuda_graph_cache_stats()["n_entries"] == 3


def test_policy_off_does_not_capture(small_inputs):
    X, W = small_inputs
    # Policy stays at default "off".
    y_eager = v9_linear_forward(X, W)
    y_off = gc_mod.v9_linear_forward_cuda_graph(X, W)
    # Numerically identical (same kernel, same input), no cache use.
    assert torch.equal(y_eager, y_off)
    assert gc_mod.cuda_graph_cache_stats()["n_entries"] == 0


def test_capture_failure_installs_sentinel(monkeypatch, small_inputs):
    """If capture raises, we install a sentinel and subsequent calls
    must fall through to eager without re-attempting capture."""
    X, W = small_inputs
    gc_mod.set_cuda_graph_policy("force")

    # Monkey-patch torch.cuda.graph to always raise RuntimeError.
    class _PoisonCtx:
        def __init__(self, *a, **kw):
            pass
        def __enter__(self):
            raise RuntimeError("synthetic capture failure")
        def __exit__(self, *a, **kw):
            return False

    monkeypatch.setattr(torch.cuda, "graph", _PoisonCtx)

    _ = gc_mod.v9_linear_forward_cuda_graph(X, W)   # should not raise
    stats = gc_mod.cuda_graph_cache_stats()
    assert stats["n_capture_fail"] == 1, stats
    assert stats["n_entries"] == 1, stats   # sentinel installed
    # Second call on same shape: must NOT re-attempt capture.
    _ = gc_mod.v9_linear_forward_cuda_graph(X, W)
    stats2 = gc_mod.cuda_graph_cache_stats()
    assert stats2["n_capture_fail"] == 1, stats2   # still 1


def test_nested_capture_refused(monkeypatch, small_inputs):
    """If already inside a capture region, the cache must refuse
    rather than nest (which would produce a confusing CUDA error)."""
    X, W = small_inputs
    gc_mod.set_cuda_graph_policy("force")
    # Simulate being inside an outer capture region.  The cache uses
    # ``torch.cuda.is_current_stream_capturing`` to detect this.
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True
    )
    _ = gc_mod.v9_linear_forward_cuda_graph(X, W)
    stats = gc_mod.cuda_graph_cache_stats()
    assert stats["n_capture_fail"] == 1, stats
    # Sentinel installed so future calls short-circuit.
    assert stats["n_entries"] == 1, stats
