"""CUDA Graph cache for ``v9_linear_forward`` (R49 Step 1).

This module adds an opt-in, shape-bucketed CUDA Graph replay path for
the V9 Linear forward pipeline.  The goal is to eliminate the 50-90%
Python-side launch tax observed on decode shapes (T<=8, see
``cuda_kernel/logs/phase2_microscope/audit_launch_tax/launch_tax_audit.md``).

Design notes are in ``.codebuddy/plan/r49_cuda_graph/design.md`` and
should be read before editing this file.  TL;DR of the contract:

* **Non-invasive.**  ``v9_linear_forward`` itself is NOT modified.
  Callers opt in explicitly by calling ``v9_linear_forward_cuda_graph``
  or by flipping the global policy via ``set_cuda_graph_policy``.
* **Correctness-first.**  On any capture failure or cache miss we
  transparently delegate to the eager path; the graph path must never
  produce different-amplitude output.
* **Lifetime-safe.**  Cache entries own GPU memory (static input
  buffer, graph-private intermediates, static output buffer) and are
  tied via weak reference to the originating weight container so that
  dropping the weights also frees the graph.

Implemented (D1 + D2):

* public API surface (``v9_linear_forward_cuda_graph``,
  ``set_cuda_graph_policy``, ``prewarm_cuda_graph_cache``,
  ``cuda_graph_cache_stats``, ``clear_cuda_graph_cache``)
* shape-bucket cache key construction
* LRU + weak-ref-driven eviction
* policy plumbing (``'off'`` / ``'auto'`` / ``'force'``)
* CUDA Graph capture on a side stream (3 warm-up iters outside the
  capture region, shared ``graph_pool_handle`` across entries so the
  allocator can re-use memory) and ``replay`` with a lightweight
  ``x_static.copy_(X)`` + ``y_static.clone()`` input/output bounce.

Capture failures (any ``Exception`` thrown inside ``torch.cuda.graph``)
are caught, counted in ``cuda_graph_cache_stats()['n_capture_fail']``,
and the shape is permanently demoted to eager for the life of the
weight container (we insert a sentinel entry so subsequent hits skip
re-attempting capture).
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from kernel.triton_kernel.pack_utils import V9WeightContainer
from kernel.triton_kernel.v9_linear import DECODE_T_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants (tune points are all here; see design doc §6)
# ---------------------------------------------------------------------------

#: Maximum number of cached (shape, weight-identity) entries.  When
#: exceeded, LRU eviction releases GPU memory held by the least-recently-
#: used entry, and that shape falls back to eager on its next call.
#: See design doc §7 Q3 for sizing rationale (Qwen3-4B: 32 entries max;
#: cap provides 2x headroom for mixed-model serving).
_GRAPH_CACHE_MAX_ENTRIES: int = 64

#: Shape-bucket list for ``'auto'`` policy eligibility.  Only ``T``
#: values in this set trigger graph-path attempts; everything else
#: falls through to eager regardless of policy (design doc §2.5).
_AUTO_T_BUCKETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)

#: Policy state (process-global).  Legal values: 'off' | 'auto' | 'force'.
#: - 'off'   : graph path disabled; entry point is a thin delegator.
#: - 'auto'  : graph path enabled iff T in _AUTO_T_BUCKETS AND W.n_hp_blocks > 0.
#: - 'force' : graph path enabled for any T <= DECODE_T_THRESHOLD.
#: Default is 'off' so that merely importing ``kernel.backend`` never
#: changes observable behaviour.
_POLICY_LEGAL: Tuple[str, ...] = ("off", "auto", "force")
_policy: str = "off"

#: Re-entrancy guard.  CUDA Graph capture cannot nest; attempting to
#: capture while already inside a capture region raises a confusing
#: cudnn error.  We detect this with ``torch.cuda.is_current_stream_
#: capturing()`` at capture time and refuse cleanly.  A re-entrant
#: lock additionally serialises capture across threads so that two
#: concurrent miss-path calls don't both try to allocate pool memory
#: simultaneously (the shared pool handle is single-producer-safe).
_capture_lock = threading.RLock()

# ---------------------------------------------------------------------------
# Cache entry (D1 stub; D2 will populate `graph`/`x_static`/`y_static`)
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    """Metadata and GPU-side buffers for one captured graph.

    D1 version holds only the metadata needed to exercise the key/
    LRU/weak-ref machinery; ``graph``, ``x_static``, ``y_static`` are
    populated by D2's capture routine.
    """
    # Identity of the (shape, W) bucket this entry serves.
    key: "_CacheKey"
    # Shape metadata, cached for fast eligibility re-check (the cache
    # key already encodes shape, but we keep a human-readable copy for
    # logs and stats).
    T: int
    d_in: int
    d_out: int
    n_hp_blocks: int
    # Filled by D2:
    graph: Optional["torch.cuda.CUDAGraph"] = None
    x_static: Optional[torch.Tensor] = None
    y_static: Optional[torch.Tensor] = None
    # Bookkeeping
    capture_us: float = 0.0          # wall-clock spent during capture
    replay_count: int = 0            # how many times replayed
    # Finalizer hook (set by cache.put); calling it releases the graph.
    _finalizer: Optional[weakref.finalize] = None

    def release(self) -> None:
        """Release GPU-side resources held by this entry.

        Called on LRU eviction or when the originating W is garbage-
        collected.  Safe to call multiple times.
        """
        # Drop strong refs; the CUDAGraph destructor + the caching
        # allocator handle the actual GPU free.
        self.graph = None
        self.x_static = None
        self.y_static = None
        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer.detach()

# ---------------------------------------------------------------------------
# Cache key — narrow, correctness-safe (see design doc §2.2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _CacheKey:
    """Hashable key identifying one graph bucket.

    Correctness-sensitive fields (must be in key):
      - ``w_id``       : weight-container identity (pointer-addressed)
      - ``x_shape``    : full shape tuple, NOT just ``T`` (catches BS>1 misuse)
      - ``x_dtype``    : dtype string
      - ``x_stride``   : CUDA Graph captures pointer arithmetic; two
                         views with same shape but different strides
                         read different memory.  Must differentiate.
      - ``x_contiguous``: fast-path flag; technically implied by stride
                         but kept explicit to match allocator behaviour.
      - ``device_index``: multi-GPU safety.
    """
    w_id: int
    x_shape: Tuple[int, ...]
    x_dtype: str
    x_stride: Tuple[int, ...]
    x_contiguous: bool
    device_index: int


def _build_cache_key(X: torch.Tensor, W: V9WeightContainer) -> _CacheKey:
    """Build a cache key for an ``(X, W)`` pair.

    ``O(1)`` in both time and allocations; measured ~0.5us on 4090.

    We key on ``id(W)`` rather than on ``W.W_low_packed.data_ptr()``
    because the latter can alias if CUDA's caching allocator reuses
    freed pointers across weight containers (observed in stress tests).
    ``id()`` is stable for the lifetime of the Python object, which is
    exactly the lifetime over which the cache entry is valid.
    """
    if not X.is_cuda:
        raise ValueError(
            "CUDA Graph cache requires X on CUDA device; got "
            f"device={X.device}"
        )
    return _CacheKey(
        w_id=id(W),
        x_shape=tuple(X.shape),
        x_dtype=str(X.dtype),
        x_stride=tuple(X.stride()),
        x_contiguous=X.is_contiguous(),
        device_index=X.device.index if X.device.index is not None else 0,
    )

# ---------------------------------------------------------------------------
# Cache container — LRU + weak-ref eviction
# ---------------------------------------------------------------------------

class _GraphCache:
    """Process-global LRU cache of CUDA Graph entries.

    Thread-safety: this class holds a single ``threading.RLock`` around
    all mutation.  Replay-only reads after an initial ``get()`` are
    lock-free (the returned entry is not mutated under the lock), but
    the ``get``/``put``/``evict`` path is serialized.  Capture itself
    additionally takes the module-level ``_capture_lock`` to refuse
    nested captures.
    """

    def __init__(self, max_entries: int = _GRAPH_CACHE_MAX_ENTRIES) -> None:
        self._entries: "OrderedDict[_CacheKey, _CacheEntry]" = OrderedDict()
        self._max = int(max_entries)
        self._mutex = threading.RLock()
        # Counters (public via cuda_graph_cache_stats)
        self._n_hit = 0
        self._n_miss = 0
        self._n_evict = 0
        self._n_capture_fail = 0

    # -- mutators -----------------------------------------------------

    def get(self, key: _CacheKey) -> Optional[_CacheEntry]:
        """Return entry for ``key`` and bump its LRU position."""
        with self._mutex:
            entry = self._entries.get(key)
            if entry is None:
                self._n_miss += 1
                return None
            # LRU bump
            self._entries.move_to_end(key, last=True)
            self._n_hit += 1
            return entry

    def put(
        self,
        key: _CacheKey,
        entry: _CacheEntry,
        weight_ref_target: Any,
    ) -> None:
        """Install ``entry`` under ``key``.

        ``weight_ref_target`` is an object whose lifetime gates the
        entry: when it is garbage-collected, the entry is auto-released.
        In practice this is the originating ``V9WeightContainer``.
        """
        with self._mutex:
            # Install weak-ref finalizer that evicts this key when W dies.
            # ``weakref.finalize`` captures positional args by reference
            # so we pass self + key, not the entry (which we want freed).
            entry._finalizer = weakref.finalize(
                weight_ref_target, _on_weight_dead, self, key
            )
            self._entries[key] = entry
            self._entries.move_to_end(key, last=True)
            while len(self._entries) > self._max:
                old_key, old_entry = self._entries.popitem(last=False)
                old_entry.release()
                self._n_evict += 1
                logger.debug(
                    "cuda_graph_cache: LRU evicted key T=%d d_in=%d d_out=%d",
                    old_entry.T, old_entry.d_in, old_entry.d_out,
                )

    def drop(self, key: _CacheKey) -> None:
        """Explicitly drop an entry (used by the weakref finalizer)."""
        with self._mutex:
            entry = self._entries.pop(key, None)
            if entry is not None:
                entry.release()

    def clear(self) -> None:
        with self._mutex:
            for entry in self._entries.values():
                entry.release()
            self._entries.clear()

    def register_capture_failure(self) -> None:
        with self._mutex:
            self._n_capture_fail += 1

    # -- read-only ----------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._mutex:
            return {
                "n_entries": len(self._entries),
                "max_entries": self._max,
                "n_hit": self._n_hit,
                "n_miss": self._n_miss,
                "n_evict": self._n_evict,
                "n_capture_fail": self._n_capture_fail,
                "policy": _policy,
            }

    def __len__(self) -> int:
        with self._mutex:
            return len(self._entries)


def _on_weight_dead(cache: _GraphCache, key: _CacheKey) -> None:
    """Finalizer target: called when the weight container dies."""
    try:
        cache.drop(key)
    except Exception:
        # Finalizers must never raise.  Pure best-effort cleanup.
        pass


# Module-global singleton.  Swapped out only by ``clear_cuda_graph_cache``.
_cache = _GraphCache()

#: Lazily-initialised shared memory pool handle for all captured graphs.
#: Sharing a pool lets CUDA's graph-aware allocator re-use memory across
#: entries that don't temporally overlap; without this each graph would
#: reserve its own peak memory footprint.  Initialised on first capture
#: because ``graph_pool_handle`` requires a CUDA context.
_shared_pool_handle: Optional[Any] = None
_shared_pool_mutex = threading.Lock()


def _get_shared_pool_handle() -> Any:
    """Return (creating if needed) the process-wide graph memory pool."""
    global _shared_pool_handle
    with _shared_pool_mutex:
        if _shared_pool_handle is None:
            _shared_pool_handle = torch.cuda.graph_pool_handle()
        return _shared_pool_handle

# ---------------------------------------------------------------------------
# Eligibility (design doc §3)
# ---------------------------------------------------------------------------

def _is_eligible(X: torch.Tensor, W: V9WeightContainer, T: int) -> bool:
    """Return True iff current policy allows using the graph path.

    Correctness-safe: returning False is ALWAYS valid (degrades to eager).
    Returning True only means "try the graph path; on capture failure
    still degrade gracefully".
    """
    if _policy == "off":
        return False
    if T > DECODE_T_THRESHOLD:
        # Prefill path calls W4A16 / cuBLAS which is not capture-friendly.
        return False
    if _policy == "force":
        return True
    # policy == 'auto'
    if T not in _AUTO_T_BUCKETS:
        return False
    if W.n_hp_blocks <= 0:
        # launch_sparse cluster is exclusively hp>0 shapes; hp==0 has a
        # different (smaller) launch tax profile and is not Step 1's scope.
        return False
    return True

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def v9_linear_forward_cuda_graph(
    X_fp16: torch.Tensor, W: V9WeightContainer
) -> torch.Tensor:
    """Graph-replay entry for the V9 Linear forward pipeline.

    Behaviour-compatible with :func:`kernel.backend.v9_linear_forward`.
    When the active policy and input shape make the current call
    eligible for graph replay, we consult (or populate) the cache.
    Any failure — policy off, ineligible shape, capture error — falls
    through to eager.

    D1 NOTE: the capture/replay implementation lands in D2.  Today we
    only exercise the key/eligibility/stats machinery; all eligible
    calls log a single debug line and delegate to eager.
    """
    # Late import: avoids circular dep with dispatcher.py (which imports
    # V9WeightContainer at module-init time).
    from kernel.backend.dispatcher import v9_linear_forward

    # Fast path: non-CUDA tensor or eager-only policy.  Delegate immediately.
    if not X_fp16.is_cuda or _policy == "off":
        return v9_linear_forward(X_fp16, W)

    # Derive T the same way the dispatcher does.
    if X_fp16.dim() == 2:
        T = int(X_fp16.shape[0])
    elif X_fp16.dim() == 3:
        T = int(X_fp16.shape[0] * X_fp16.shape[1])
    else:
        # Defensive: unusual ranks go straight to eager (their contract
        # check happens inside v9_linear_forward).
        return v9_linear_forward(X_fp16, W)

    if not _is_eligible(X_fp16, W, T):
        return v9_linear_forward(X_fp16, W)

    # Eligible.  Look up the cache.
    key = _build_cache_key(X_fp16, W)
    entry = _cache.get(key)
    if entry is not None:
        if entry.graph is None:
            # Sentinel for a previously-failed capture — demoted to eager.
            return v9_linear_forward(X_fp16, W)
        return _replay(entry, X_fp16)

    # Cache miss.  Attempt capture; on any failure install a sentinel
    # so we don't re-attempt capture every call.
    new_entry = _capture_entry(X_fp16, W, T, key)
    _cache.put(key, new_entry, weight_ref_target=W)
    if new_entry.graph is None:
        # Capture failed (sentinel stored).  Run eager for this call.
        return v9_linear_forward(X_fp16, W)
    return _replay(new_entry, X_fp16)


def _capture_entry(
    X_live: torch.Tensor,
    W: V9WeightContainer,
    T: int,
    key: _CacheKey,
) -> _CacheEntry:
    """Capture a CUDA Graph for ``(X_live.shape, W)`` and return the entry.

    On success the returned entry carries ``graph``/``x_static``/
    ``y_static`` bound to the shared memory pool.  On any capture
    failure (shape-incompatible kernel, stream error, CUDA driver OOM,
    ...) we return a *sentinel* entry (``graph=None``) so the caller
    can install it in the cache and future hits short-circuit to eager
    without re-attempting capture.

    Notes on correctness:

    * ``x_static`` is allocated on the same device/dtype/stride
      layout as ``X_live``.  Non-contiguous ``X_live`` is captured
      against a contiguous ``x_static`` because the underlying
      ``v9_linear_forward`` already tolerates a contiguous 2D input;
      we simply reshape/contiguify on replay-side.  The eligibility
      gate guarantees this path is only taken for decode shapes where
      the copy cost is negligible (<2us for T<=128).

    * We fill ``x_static`` with ``randn`` before warm-up so any NaN-
      branchy kernel code observes valid inputs (zeros could happen
      to alias pointer reads on the scale epilogue fast-path).

    * Warm-up iterations live on a side stream so that any autotune
      cache misses / lazy CUDA module loads settle *before* entering
      the capture region (otherwise ``torch.cuda.graph`` will capture
      those side-effects too and break on replay).
    """
    from kernel.backend.dispatcher import v9_linear_forward

    # -- prepare static buffers ---------------------------------------
    # Always use contiguous 2D shape (T, d_in) for capture; dispatcher
    # accepts either 2D or 3D but the underlying kernel only needs 2D,
    # and staying contiguous keeps strides deterministic across replays.
    d_in = int(W.d_in)
    d_out = int(W.d_out)
    device = X_live.device
    dtype = X_live.dtype
    entry = _CacheEntry(
        key=key, T=T, d_in=d_in, d_out=d_out,
        n_hp_blocks=int(W.n_hp_blocks),
    )

    # Refuse nested capture on two axes:
    #   (a) the current CUDA stream is already capturing (the user's
    #       framework wrapped us inside its own ``torch.cuda.graph``).
    #       Detect this first; we cannot begin a nested capture.
    #   (b) another host thread is mid-capture.  The shared memory
    #       pool is single-producer-safe; serialise with the lock and
    #       refuse (rather than block) if it's held.
    if torch.cuda.is_current_stream_capturing():
        logger.warning(
            "cuda_graph_cache: refusing capture while parent stream is "
            "already capturing (T=%d d_in=%d d_out=%d)",
            T, d_in, d_out,
        )
        _cache.register_capture_failure()
        return entry  # sentinel (graph=None)
    if not _capture_lock.acquire(blocking=False):
        logger.warning(
            "cuda_graph_cache: refusing nested capture for T=%d d_in=%d d_out=%d",
            T, d_in, d_out,
        )
        _cache.register_capture_failure()
        return entry  # sentinel (graph=None)

    t0 = time.perf_counter()
    try:
        x_static = torch.empty((T, d_in), dtype=dtype, device=device)
        x_static.normal_()

        # Warm-up on side stream — resolves autotune / first-launch
        # module loads before the capture region starts.
        side_stream = torch.cuda.Stream(device=device)
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                _ = v9_linear_forward(x_static, W)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        # Capture.
        g = torch.cuda.CUDAGraph()
        pool = _get_shared_pool_handle()
        with torch.cuda.graph(g, pool=pool):
            y_static = v9_linear_forward(x_static, W)

        entry.graph = g
        entry.x_static = x_static
        entry.y_static = y_static
        entry.capture_us = (time.perf_counter() - t0) * 1e6
        logger.info(
            "cuda_graph_cache: captured T=%d d_in=%d d_out=%d hp=%d in %.1fus",
            T, d_in, d_out, entry.n_hp_blocks, entry.capture_us,
        )
    except Exception as exc:
        # Capture failed.  Store sentinel; counter-increment; let caller
        # fall through to eager.
        _cache.register_capture_failure()
        logger.warning(
            "cuda_graph_cache: capture failed for T=%d d_in=%d d_out=%d "
            "(%s: %s); demoting shape to eager",
            T, d_in, d_out, type(exc).__name__, exc,
        )
        # entry stays with graph=None — that's the sentinel state.
    finally:
        _capture_lock.release()
    return entry


def _replay(entry: _CacheEntry, X: torch.Tensor) -> torch.Tensor:
    """Replay ``entry.graph`` with live ``X`` copied into ``entry.x_static``.

    Contract:
      * ``X`` must have the same shape/dtype/device as the captured
        ``entry.x_static`` (the cache-key machinery guarantees this).
      * Returns a **clone** of ``entry.y_static`` so callers can hold
        the result past subsequent replays.
    """
    assert entry.graph is not None
    assert entry.x_static is not None
    assert entry.y_static is not None

    x_static = entry.x_static
    # Reshape X into the captured 2D layout.  The cache key already
    # pins the full shape, so ``reshape`` is always valid here and
    # costs nothing when X is already 2D contiguous.
    if X.shape != x_static.shape:
        X_view = X.reshape(x_static.shape)
    else:
        X_view = X
    # Copy live data into the static buffer.  copy_ is NOT captured;
    # it's a plain async memcpy on the current stream.
    x_static.copy_(X_view, non_blocking=True)
    entry.graph.replay()
    entry.replay_count += 1
    # Clone output so the caller holds an independent tensor; without
    # this, a subsequent replay would overwrite ``y_static`` in-place.
    y_out = entry.y_static.clone()
    # Restore the caller's batch layout.
    # The captured static_Y is (T, d_out); if the live X was 3D
    # (B, S, d_in) we need (B, S, d_out).  The cache key pinned the
    # live shape, so we recover it from there.
    live_shape = entry.key.x_shape
    if len(live_shape) == 3:
        b, s, _ = live_shape
        y_out = y_out.reshape(b, s, entry.d_out)
    return y_out


def set_cuda_graph_policy(policy: str) -> str:
    """Set the process-global CUDA Graph policy.

    Returns the previous policy string.  Legal values: ``'off'``,
    ``'auto'``, ``'force'``.  See module-level docstring or design
    doc §3 for semantics.
    """
    global _policy
    if policy not in _POLICY_LEGAL:
        raise ValueError(
            f"Unknown CUDA Graph policy {policy!r}; "
            f"legal values are {_POLICY_LEGAL}"
        )
    previous = _policy
    _policy = policy
    logger.info("cuda_graph_cache: policy %r -> %r", previous, policy)
    return previous


def get_cuda_graph_policy() -> str:
    """Return the currently active policy."""
    return _policy


def prewarm_cuda_graph_cache(
    entries: List[Tuple[torch.Tensor, V9WeightContainer]],
) -> Dict[str, int]:
    """Pre-capture graphs for the given ``(X, W)`` pairs.

    Useful at server startup to avoid the ~20ms first-call cost per
    shape from bleeding into user-visible latency.

    Any entry that cannot be captured (ineligible shape OR capture
    failure) is silently skipped; this function never raises.  The
    returned dict has three keys:
      - ``attempted``: number of ``(X, W)`` pairs that passed the
        eligibility gate and entered the capture path.
      - ``captured`` : of those, how many produced a usable entry.
      - ``skipped``  : number of pairs that failed eligibility or had
        unusual ranks and never entered capture.
    """
    attempted = 0
    captured = 0
    skipped = 0
    for X, W in entries:
        if X.dim() == 2:
            T = int(X.shape[0])
        elif X.dim() == 3:
            T = int(X.shape[0] * X.shape[1])
        else:
            skipped += 1
            continue
        if not X.is_cuda or not _is_eligible(X, W, T):
            skipped += 1
            continue
        attempted += 1
        key = _build_cache_key(X, W)
        # Don't re-capture if already present.
        if _cache.get(key) is not None:
            captured += 1
            continue
        new_entry = _capture_entry(X, W, T, key)
        _cache.put(key, new_entry, weight_ref_target=W)
        if new_entry.graph is not None:
            captured += 1
    return {"attempted": attempted, "captured": captured, "skipped": skipped}


def cuda_graph_cache_stats() -> Dict[str, Any]:
    """Return a snapshot of cache counters for diagnostics."""
    return _cache.stats()


def clear_cuda_graph_cache() -> None:
    """Release all cached graphs (tests and interactive use)."""
    _cache.clear()


__all__ = [
    "v9_linear_forward_cuda_graph",
    "set_cuda_graph_policy",
    "get_cuda_graph_policy",
    "prewarm_cuda_graph_cache",
    "cuda_graph_cache_stats",
    "clear_cuda_graph_cache",
]
