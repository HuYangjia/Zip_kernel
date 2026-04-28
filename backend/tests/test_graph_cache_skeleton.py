"""Unit tests for the D1 skeleton of ``kernel.backend.graph_cache``.

Scope (D1):
  - cache key construction: correctness-critical fields must all
    differentiate keys (shape, stride, dtype, device, W identity).
  - LRU behaviour: exceeding capacity evicts the oldest entry.
  - Weak-ref cleanup: dropping the W container frees its entry.
  - Policy plumbing: set_cuda_graph_policy accepts/rejects values;
    eligibility respects T buckets, n_hp_blocks, and DECODE_T_THRESHOLD.
  - Stats surface: counters monotonically increase on hit/miss/evict.

Out of scope (D2):
  - Actual CUDA Graph capture & replay.  The capture pathway in
    ``v9_linear_forward_cuda_graph`` falls through to eager today, so
    we only test that the pre-capture plumbing never misclassifies a
    correct call.

These tests DO NOT require a CUDA device (run-anywhere).  D2 will add
a GPU-only sibling file ``test_cuda_graph_capture.py`` guarded by a
``pytest.importorskip`` style gate.
"""

from __future__ import annotations

import gc
import sys
import types
from dataclasses import dataclass
from typing import Tuple

import pytest

# ---------------------------------------------------------------------------
# Lightweight stubs so the tests run on machines without CUDA.
#
# ``kernel.backend.graph_cache`` imports:
#   - torch (needed for Tensor types and device; we use real torch since
#     the CPU build is always available)
#   - ``V9WeightContainer`` from ``kernel.triton_kernel.pack_utils``
#     (purely a dataclass; safe to import)
#   - ``DECODE_T_THRESHOLD`` from ``kernel.triton_kernel.v9_linear``
#     (a plain constant; safe to import)
# None of these require a CUDA build, so we can use the module as-is.
# ---------------------------------------------------------------------------

import torch

from kernel.backend import graph_cache as gc_mod


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@dataclass
class _FakeWeight:
    """Minimal stand-in for V9WeightContainer for non-capture tests.

    We only need ``n_hp_blocks``, ``d_in``, ``d_out`` accessors.  The
    cache uses ``id(W)`` as the identity, not any container fields, so
    field contents do not influence key hashing.
    """
    d_in: int = 2048
    d_out: int = 4096
    n_hp_blocks: int = 3          # >0 keeps us on the launch_sparse cluster
    # Additional attributes touched by code paths we exercise:
    W_low_packed: object = None


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Restore graph_cache module state between tests."""
    # Snapshot + reset.
    prev_policy = gc_mod.get_cuda_graph_policy()
    gc_mod.clear_cuda_graph_cache()
    gc_mod.set_cuda_graph_policy("off")
    # Reset stats counters via a clean cache instance.
    gc_mod._cache = gc_mod._GraphCache()
    yield
    # Restore for any downstream suites.
    gc_mod.clear_cuda_graph_cache()
    gc_mod.set_cuda_graph_policy(prev_policy)
    gc_mod._cache = gc_mod._GraphCache()


def _make_cpu_tensor(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Build a 'CUDA-looking' tensor for key tests.

    The cache's key helper requires ``X.is_cuda`` to be True; for
    non-CUDA machines we satisfy this by using the meta device, which
    advertises ``is_cuda=False`` but whose key-relevant attributes
    (shape, stride, dtype, device.index) behave identically.
    """
    t = torch.empty(shape, dtype=dtype, device="cpu")
    return t


def _make_fake_cuda_tensor(
    shape: Tuple[int, ...],
    *,
    dtype: torch.dtype = torch.float16,
    stride: Tuple[int, ...] = None,
    device_index: int = 0,
) -> torch.Tensor:
    """Fabricate a tensor that looks CUDA-resident to the cache key.

    We don't actually need device memory; the cache only reads
    ``is_cuda``, ``shape``, ``dtype``, ``stride``, ``device.index``,
    ``is_contiguous``.  We intercept those with a thin wrapper.
    """
    base = torch.empty(shape, dtype=dtype, device="cpu")
    if stride is not None:
        base = base.as_strided(shape, stride)
    return _FakeCudaTensor(base, device_index=device_index)


class _FakeCudaTensor:
    """Duck-typed tensor proxy with ``is_cuda == True``."""

    def __init__(self, wrapped: torch.Tensor, device_index: int = 0):
        self._w = wrapped
        self._device_index = device_index

    @property
    def is_cuda(self) -> bool:
        return True

    @property
    def shape(self):
        return self._w.shape

    @property
    def dtype(self):
        return self._w.dtype

    @property
    def device(self):
        # Pretend this is cuda:<index>
        return torch.device(f"cuda:{self._device_index}")

    def stride(self):
        return self._w.stride()

    def is_contiguous(self) -> bool:
        return self._w.is_contiguous()

    def dim(self) -> int:
        return self._w.dim()


# ---------------------------------------------------------------------------
# Cache key tests (design doc §2.2: every correctness-sensitive field
# must affect the key)
# ---------------------------------------------------------------------------

def test_key_identical_inputs_produce_identical_key():
    W = _FakeWeight()
    X1 = _make_fake_cuda_tensor((1, 2048))
    X2 = _make_fake_cuda_tensor((1, 2048))
    # Note: X1 and X2 are *different* Python objects but carry the same
    # shape/dtype/stride.  The key must be identical (same bucket).
    k1 = gc_mod._build_cache_key(X1, W)
    k2 = gc_mod._build_cache_key(X2, W)
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_key_differs_on_shape():
    W = _FakeWeight()
    X_t1 = _make_fake_cuda_tensor((1, 2048))
    X_t8 = _make_fake_cuda_tensor((8, 2048))
    assert gc_mod._build_cache_key(X_t1, W) != gc_mod._build_cache_key(X_t8, W)


def test_key_differs_on_dtype():
    W = _FakeWeight()
    Xf16 = _make_fake_cuda_tensor((1, 2048), dtype=torch.float16)
    Xf32 = _make_fake_cuda_tensor((1, 2048), dtype=torch.float32)
    assert gc_mod._build_cache_key(Xf16, W) != gc_mod._build_cache_key(Xf32, W)


def test_key_differs_on_stride():
    """Two shape-identical tensors with different strides must NOT
    share a cache entry (CUDA Graph replays pointer arithmetic; a
    non-contiguous view reads different memory).  See design §2.2."""
    W = _FakeWeight()
    # Contiguous (1, 2048) -> stride (2048, 1)
    contig = torch.empty((1, 2048), dtype=torch.float16, device="cpu")
    # A broadcast-via-expand yields shape (1, 2048) but stride (0, 0).
    one_scalar = torch.empty((1, 1), dtype=torch.float16, device="cpu")
    expanded = one_scalar.expand(1, 2048)
    X_contig = _FakeCudaTensor(contig, device_index=0)
    X_expanded = _FakeCudaTensor(expanded, device_index=0)
    k_c = gc_mod._build_cache_key(X_contig, W)
    k_e = gc_mod._build_cache_key(X_expanded, W)
    assert k_c.x_shape == k_e.x_shape
    assert k_c.x_stride != k_e.x_stride
    assert k_c != k_e


def test_key_differs_on_device_index():
    W = _FakeWeight()
    X_d0 = _make_fake_cuda_tensor((1, 2048), device_index=0)
    X_d1 = _make_fake_cuda_tensor((1, 2048), device_index=1)
    assert gc_mod._build_cache_key(X_d0, W) != gc_mod._build_cache_key(X_d1, W)


def test_key_differs_on_weight_identity():
    W1 = _FakeWeight()
    W2 = _FakeWeight()  # different Python object, same field values
    X = _make_fake_cuda_tensor((1, 2048))
    assert gc_mod._build_cache_key(X, W1) != gc_mod._build_cache_key(X, W2)


def test_key_rejects_non_cuda_tensor():
    W = _FakeWeight()
    X_cpu = _make_cpu_tensor((1, 2048))  # is_cuda == False
    with pytest.raises(ValueError, match="CUDA"):
        gc_mod._build_cache_key(X_cpu, W)


# ---------------------------------------------------------------------------
# LRU tests
# ---------------------------------------------------------------------------

def test_lru_evicts_oldest_when_full():
    cache = gc_mod._GraphCache(max_entries=2)
    weights = [_FakeWeight() for _ in range(3)]
    X = _make_fake_cuda_tensor((1, 2048))

    entries = []
    for W in weights:
        k = gc_mod._build_cache_key(X, W)
        e = gc_mod._CacheEntry(
            key=k, T=1, d_in=2048, d_out=4096, n_hp_blocks=3,
        )
        cache.put(k, e, weight_ref_target=W)
        entries.append((k, W, e))

    # After inserting 3 into a cap-2 cache, the first must be evicted.
    assert cache.get(entries[0][0]) is None        # evicted
    assert cache.get(entries[1][0]) is not None     # survived
    assert cache.get(entries[2][0]) is not None     # newest

    stats = cache.stats()
    assert stats["n_evict"] == 1
    assert stats["n_entries"] == 2


def test_lru_promotes_on_get():
    """Accessing an entry should promote it, so a subsequent insert
    evicts the *other* (now-oldest) entry instead."""
    cache = gc_mod._GraphCache(max_entries=2)
    W_a, W_b, W_c = _FakeWeight(), _FakeWeight(), _FakeWeight()
    X = _make_fake_cuda_tensor((1, 2048))

    def _insert(W):
        k = gc_mod._build_cache_key(X, W)
        cache.put(
            k,
            gc_mod._CacheEntry(
                key=k, T=1, d_in=2048, d_out=4096, n_hp_blocks=3
            ),
            weight_ref_target=W,
        )
        return k

    k_a = _insert(W_a)
    k_b = _insert(W_b)
    # Promote A by reading it.
    assert cache.get(k_a) is not None
    # Now insert C; B should be evicted (it's the oldest after the promote).
    k_c = _insert(W_c)
    assert cache.get(k_a) is not None
    assert cache.get(k_b) is None
    assert cache.get(k_c) is not None


def test_weakref_cleanup_releases_entry():
    """Dropping the owning W must cause the entry to be removed."""
    cache = gc_mod._GraphCache(max_entries=8)
    X = _make_fake_cuda_tensor((1, 2048))

    W = _FakeWeight()
    k = gc_mod._build_cache_key(X, W)
    cache.put(
        k,
        gc_mod._CacheEntry(key=k, T=1, d_in=2048, d_out=4096, n_hp_blocks=3),
        weight_ref_target=W,
    )
    assert cache.get(k) is not None

    # Drop the strong ref and force finalizer run.
    del W
    gc.collect()
    # weakref.finalize callbacks run deterministically after GC on
    # CPython; re-querying should now miss.
    assert cache.get(k) is None


# ---------------------------------------------------------------------------
# Policy plumbing
# ---------------------------------------------------------------------------

def test_set_cuda_graph_policy_round_trip():
    assert gc_mod.get_cuda_graph_policy() == "off"
    prev = gc_mod.set_cuda_graph_policy("auto")
    assert prev == "off"
    assert gc_mod.get_cuda_graph_policy() == "auto"
    prev = gc_mod.set_cuda_graph_policy("force")
    assert prev == "auto"
    assert gc_mod.get_cuda_graph_policy() == "force"
    gc_mod.set_cuda_graph_policy("off")


def test_set_cuda_graph_policy_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown CUDA Graph policy"):
        gc_mod.set_cuda_graph_policy("turbo")


@pytest.mark.parametrize(
    "policy, T, n_hp, expected",
    [
        # policy=off  -> never eligible
        ("off",   1,    3,   False),
        ("off",   8,    3,   False),
        # policy=auto -> T in buckets AND hp>0
        ("auto",  1,    3,   True),
        ("auto",  8,    3,   True),
        ("auto",  128,  3,   True),
        ("auto",  5,    3,   False),   # not in bucket list
        ("auto",  256,  3,   False),   # T too large
        ("auto",  1,    0,   False),   # hp==0 -> not our cluster
        # policy=force -> any T <= DECODE_T_THRESHOLD
        ("force", 1,    0,   True),
        ("force", 5,    0,   True),    # non-bucket T still OK under force
        ("force", 128,  0,   True),
    ],
)
def test_eligibility_matrix(policy, T, n_hp, expected):
    prev = gc_mod.set_cuda_graph_policy(policy)
    try:
        W = _FakeWeight(n_hp_blocks=n_hp)
        X = _make_fake_cuda_tensor((T, 2048))
        assert gc_mod._is_eligible(X, W, T) is expected
    finally:
        gc_mod.set_cuda_graph_policy(prev)


def test_eligibility_rejects_prefill_t():
    """Safety belt: even under 'force' we must not attempt graph capture
    for T > DECODE_T_THRESHOLD (prefill path hits cuBLAS / W4A16)."""
    prev = gc_mod.set_cuda_graph_policy("force")
    try:
        W = _FakeWeight(n_hp_blocks=3)
        T_big = gc_mod.DECODE_T_THRESHOLD + 1
        X = _make_fake_cuda_tensor((T_big, 2048))
        assert gc_mod._is_eligible(X, W, T_big) is False
    finally:
        gc_mod.set_cuda_graph_policy(prev)


# ---------------------------------------------------------------------------
# Public entry behaviour (D1: always falls through to eager)
# ---------------------------------------------------------------------------

def test_entry_point_with_off_policy_bypasses_cache(monkeypatch):
    """With policy='off', the entry point must not even build a key;
    it delegates straight to ``v9_linear_forward``."""
    from kernel.backend import dispatcher

    called = {}

    def _fake_forward(X, W):
        called["n"] = called.get("n", 0) + 1
        return X  # identity for the test

    monkeypatch.setattr(dispatcher, "v9_linear_forward", _fake_forward)

    W = _FakeWeight()
    X = _make_fake_cuda_tensor((1, 2048))
    # Verify we didn't touch the cache.
    initial_stats = gc_mod.cuda_graph_cache_stats()
    out = gc_mod.v9_linear_forward_cuda_graph(X, W)
    # out is whatever _fake_forward returned; we don't care about value.
    assert called["n"] == 1
    post_stats = gc_mod.cuda_graph_cache_stats()
    # With 'off' we should have done zero cache lookups.
    assert post_stats["n_hit"] == initial_stats["n_hit"]
    assert post_stats["n_miss"] == initial_stats["n_miss"]


def test_stats_snapshot_shape():
    stats = gc_mod.cuda_graph_cache_stats()
    expected_keys = {
        "n_entries", "max_entries",
        "n_hit", "n_miss", "n_evict", "n_capture_fail",
        "policy",
    }
    assert expected_keys.issubset(stats.keys())


def test_clear_cuda_graph_cache_is_idempotent():
    gc_mod.clear_cuda_graph_cache()
    gc_mod.clear_cuda_graph_cache()
    assert gc_mod.cuda_graph_cache_stats()["n_entries"] == 0
