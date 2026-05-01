# P0 Integration Attempt — Negative Result (2026-05-01)

## TL;DR

**P0.2 kernel is correct (10/10 parity) but SLOWER than legacy
two-step on every tested shape by 0-10%.**  Default dispatcher
gate has been set to **disabled**; `fused_dense_sparse_e2e_cuda`
falls back to legacy.  Env `HKUST_V9_P0_MODE=1` available for
P0.4 development.

## What was done

1. Added `fused_dense_sparse_e2e_cuda(X_fp16, perm, ...)` entry
   point to [ops.py](../ops.py): accepts raw fp16 X, dispatches
   among (P0 fused-quant MMA | legacy two-step | future GEMV
   specialisation).
2. Kept `fused_dense_sparse_cuda` ABI (X_s4 in) unchanged so no
   existing caller breaks.
3. Wired P0 through `fused_quant_dense_sparse_cuda_int4` which
   binds the existing `fused_quant_dense_sparse_mma_int4.cu`
   kernel (has been in-tree, parity-verified since 2026-04-29).
4. Verified 10/10 parity on
   [tests/parity_fused_quant.py](../tests/parity_fused_quant.py)
   — max rel error ≤ 0.015 (1-2 fp16 ulp).

## Why it didn't help — A/B data

[tests/p0_integration_bench.py](../tests/p0_integration_bench.py)
compares `legacy two-step (activation_quant + fused_dense_sparse)`
vs `P0 fused_dense_sparse_e2e_cuda(P0 on)` with the standard
interleaved protocol (warmup=500, outer=10, inner=200, 4 trials
median).  Excerpt:

| shape | legacy us | P0 us | Δ |
|---|---:|---:|---:|
| 0.6B q_proj T=32 (1024→2048) | 29.78 | 32.74 | **+9.96%** regress |
| 1.7B q_proj T=32 | 34.27 | 37.23 | +8.65% regress |
| 8B q_proj T=32 (4096→4096) | 34.21 | 37.02 | +8.21% regress |
| 8B gu T=32 (4096→24576) | 64.80 | 64.84 | +0.07% neutral |
| 4B q_proj T=128 (2560→4096) | 34.60 | 37.42 | +8.16% regress |
| 8B q_proj T=128 | 44.48 | 44.49 | +0.03% neutral |
| 8B gu T=128 (4096→24576) | 142.44 | 142.47 | +0.03% neutral |
| 14B gu T=128 (5120→34816) | 363.07 | 362.93 | -0.04% neutral |

- **Zero shapes win by ≥5%.**
- **Small shapes (n_groups ≤ 32, cuda_us ≈ 30-40us) regress +8-10%.**
- **Mid/large shapes (bigger grid, more n_groups) are neutral.**

## Root cause

P0.2 kernel was designed as a **correctness-first stub**.  It
omits the three optimisations that make the legacy mainloop
fast:

| feature | legacy (r67) | P0.2 | impact on small shapes |
|---|---|---|---|
| `cp.async` double-buffer | yes | **no** | HBM latency exposed inside K-loop |
| group-cache (F.1) | yes (conditional) | **no** | per-group scale/zero re-fetched from HBM |
| split-K | yes (F2) | **no** | wave under-fill on small grids |

The ~16us `activation_quant_cuda` launch floor that P0 was
meant to eliminate is indeed eliminated (verified by nsys in
a prior diagnostic) — but the less-optimised mainloop adds
~18us of inefficiency on small-n_groups shapes, giving net
+8%.  On large shapes the mainloop inefficiency is amortised
away and P0 lands close-to-neutral, **but still not faster**
because activation_quant as fraction of total time is tiny
(<5% on T=128 gu shapes).

## Decision

**Keep P0.2 in-tree; keep dispatcher disabled by default.**

- `fused_dense_sparse_e2e_cuda` exists as the forward-looking
  entry point, so future callers wire up now.
- `HKUST_V9_P0_MODE=1` overrides the gate for P0.4 development.
- Production dispatcher stays on legacy two-step (r67).

## Future work — P0.4 roadmap

Only if someone wants to pursue this further:

1. **Add group-cache to P0 kernel** (r61 Stage F's design —
   cache up to 32 groups of scale_u4/zero_u4 in smem at start,
   reuse across groups).  This is ~2 days of work; expected to
   close the small-shape regression.
2. **Add cp.async double-buffer to P0 K-loop** (Stage A2 design).
   Another ~1 day; expected to make large-shape P0 net-neutral
   to +3% (small saving from launch floor).
3. **Port sparse branch** (blocked on parity discipline).  ~1 day.

With all three, P0.4 should give **net +3-8% on T=32 small
shapes and +0-3% on T=128 mid shapes**.  That's the real upper
bound of the activation_quant launch floor saving.  Not a huge
prize, but real.

**Is P0.4 worth the 4 day investment?**  Given Phase C/C.5
have already delivered the bulk of achievable gains (r67
median 1.033× on full bench, +7.8% on 4 target shapes from C.5),
P0.4's expected +3-8% on a handful of shapes is marginal.
Recommend deferring until there's a clear user ask.
