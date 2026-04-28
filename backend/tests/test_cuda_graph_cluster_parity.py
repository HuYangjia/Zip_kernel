"""D3 cluster-wide parity test: graph replay vs eager across all 17
launch_sparse audit shapes.

This file complements ``test_cuda_graph_capture.py`` (which runs a
fast subset on a single shape).  D3 is the full correctness gate for
the 17 shapes we intend to graph-accelerate in production.

Pass criterion (per shape):
  max_abs_diff(y_eager, y_graph).cast(fp32) <= 1e-3

These tests are CUDA-only and marked ``slow`` (they each capture a
real graph; capture costs ~20-80ms per shape plus warm-up).  Skip
cleanly when CUDA is absent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required for cluster-wide parity",
                allow_module_level=True)

from kernel.backend import graph_cache as gc_mod
from kernel.backend import v9_linear_forward
from kernel.tools.profile._phase1_shapes import (
    PHASE_ALL_SHAPES,
    build_shape_inputs,
)
# Registers the 17 audit tags into PHASE1_SHAPES_BY_TAG as a side-effect.
from kernel.tools.profile.audit_launch_tax import (
    AUDIT_SHAPES,
    _register_audit_shapes,
)


# One-time side effect before any test: inject audit tags.
_register_audit_shapes()


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    prev = gc_mod.get_cuda_graph_policy()
    gc_mod.clear_cuda_graph_cache()
    gc_mod._cache = gc_mod._GraphCache()
    gc_mod.set_cuda_graph_policy("off")
    yield
    gc_mod.clear_cuda_graph_cache()
    gc_mod.set_cuda_graph_policy(prev)
    gc_mod._cache = gc_mod._GraphCache()


@pytest.mark.parametrize(
    "tag",
    [a.tag for a in AUDIT_SHAPES],
    ids=[a.tag for a in AUDIT_SHAPES],
)
def test_graph_parity_for_audit_shape(tag: str):
    """For every launch_sparse cluster member, graph replay must match
    eager within fp16 tolerance (1e-3)."""
    bundle = build_shape_inputs(tag)
    X, W = bundle.X, bundle.W

    # Baseline.
    y_eager = v9_linear_forward(X, W)

    # Graph path.
    gc_mod.set_cuda_graph_policy("force")
    y_graph = gc_mod.v9_linear_forward_cuda_graph(X, W)

    assert y_eager.shape == y_graph.shape, tag
    diff = (y_eager.float() - y_graph.float()).abs().max().item()
    assert diff <= 1e-3, (
        f"{tag}: graph vs eager max_abs_diff={diff:.2e} > 1e-3"
    )


def test_cluster_capture_success_rate():
    """All 17 audit shapes must capture successfully (no sentinels)."""
    gc_mod.set_cuda_graph_policy("force")
    captured = 0
    failed_tags: list[str] = []
    for a in AUDIT_SHAPES:
        bundle = build_shape_inputs(a.tag)
        _ = gc_mod.v9_linear_forward_cuda_graph(bundle.X, bundle.W)
        stats = gc_mod.cuda_graph_cache_stats()
        # A capture failure would bump this counter; a successful capture
        # leaves it at 0 for the lifetime of this run.
        if stats["n_capture_fail"] > len(failed_tags):
            failed_tags.append(a.tag)
        else:
            captured += 1
    assert captured == len(AUDIT_SHAPES), (
        f"{len(failed_tags)}/{len(AUDIT_SHAPES)} failed: {failed_tags}"
    )


def test_outside_bucket_T_falls_back_to_eager():
    """Auto policy must bypass the graph path for ``T`` values outside
    ``_AUTO_T_BUCKETS`` (covers T3.2 integration contract)."""
    # T=3 is not in {1,2,4,8,16,32,64,128}.  Build a hand-shaped bundle.
    # Pick a shape whose kernel we know captures cleanly, then splice X.
    bundle = build_shape_inputs(AUDIT_SHAPES[0].tag)
    X = bundle.X[:1]  # T=1 for the shape catalogue
    # Manually tile to T=3 for the eligibility probe; W is unchanged.
    X3 = torch.cat([X, X, X], dim=0).contiguous()
    gc_mod.set_cuda_graph_policy("auto")
    out = gc_mod.v9_linear_forward_cuda_graph(X3, bundle.W)
    assert out.shape[0] == 3
    stats = gc_mod.cuda_graph_cache_stats()
    assert stats["n_entries"] == 0, (
        f"T=3 must not capture under auto policy; got stats={stats}"
    )


def test_bucketed_T_captures_once_per_shape():
    """For T in the bucket list, exactly one capture per (X-shape, W)
    pair, one replay on the second call (covers T3.3 integration)."""
    bundle = build_shape_inputs(AUDIT_SHAPES[0].tag)
    X, W = bundle.X, bundle.W
    gc_mod.set_cuda_graph_policy("auto")
    # Call 1: capture.
    _ = gc_mod.v9_linear_forward_cuda_graph(X, W)
    s1 = gc_mod.cuda_graph_cache_stats()
    # Call 2: replay.
    _ = gc_mod.v9_linear_forward_cuda_graph(X, W)
    s2 = gc_mod.cuda_graph_cache_stats()
    assert s2["n_entries"] == s1["n_entries"] == 1
    assert s2["n_hit"] == s1["n_hit"] + 1
    assert s2["n_capture_fail"] == s1["n_capture_fail"] == 0


# ---------------------------------------------------------------------------
# Gate 1 (design §1): parity across the 8 Phase-1/Phase-2 representative
# shapes, combined with the 17 audit shapes = 25 shapes total.  For
# representatives with T > DECODE_T_THRESHOLD (prefill regime), the graph
# path is required by contract to fall back to eager; parity in that case
# is trivially exact (same function call, same output).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tag",
    [s.tag for s in PHASE_ALL_SHAPES],
    ids=[s.tag for s in PHASE_ALL_SHAPES],
)
def test_graph_parity_for_representative_shape(tag: str):
    """Design §1 Gate 1: eager vs graph_cuda parity on each of the
    8 Phase-1/Phase-2 representative shapes (fp16 tol 1e-3).

    Decode-regime shapes (T <= 128) go through the real capture+replay
    path under ``force`` policy; prefill-regime shapes (T > 128) fall
    back to eager by design, so the diff is structurally zero.
    """
    bundle = build_shape_inputs(tag)
    X, W = bundle.X, bundle.W
    T = int(X.shape[0])

    y_eager = v9_linear_forward(X, W)
    gc_mod.set_cuda_graph_policy("force")
    y_graph = gc_mod.v9_linear_forward_cuda_graph(X, W)

    assert y_eager.shape == y_graph.shape, tag
    diff = (y_eager.float() - y_graph.float()).abs().max().item()
    assert diff <= 1e-3, (
        f"{tag} (T={T}): graph vs eager max_abs_diff={diff:.2e} > 1e-3"
    )

    # Sanity: prefill shapes MUST have taken the eager fallback (no new
    # cache entry), decode shapes MUST have captured.
    stats = gc_mod.cuda_graph_cache_stats()
    if T > 128:
        assert stats["n_entries"] == 0, (
            f"{tag} (T={T}): prefill shape unexpectedly captured "
            f"(stats={stats})"
        )
    else:
        assert stats["n_entries"] >= 1, (
            f"{tag} (T={T}): decode shape failed to capture "
            f"(stats={stats})"
        )
