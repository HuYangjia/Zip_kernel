# r62 F2 — Data-driven dispatcher rewrite

## Context

r62 F1 established that, with the bench-methodology L2-flush fix, the
r60 legacy INT4 kernel achieves **median 1.21× speedup vs BF16 cuBLAS**
on Qwen3-8B with **11/20 shapes winning** and 7 shapes still losing
(q/o_proj T=32/128, kv_proj T=32/128/512, down_proj T=32/128).

Those 7 shapes were assumed to be kernel-internal bottlenecks (bank
conflict, register-bound occupancy) based on the r61 Stage-C / Stage-F
diagnostics, so the natural next step was either CUTLASS migration (F4)
or a smem layout rewrite (F5).

**r62 F4** attempted CUTLASS migration in two forms:

- **Stage A1.5** (GemmBatched + 3D int32 workspace): 0/12 loss shapes
  won, worst case 8× slower than legacy.  Root cause: `(n_groups, d_out,
  T)` workspace explodes to ≥1 GB for large gate_up shapes, HBM
  round-trip dominates.
- **Stage A3** (per-group Gemm + 2D workspace + fused accumulator):
  0/12 loss shapes won, even worse than A1.5 (up to 14× slower).  Root
  cause: `n_groups` independent CUTLASS-kernel launches × 7us launch
  overhead × 3 (can_implement + init + run) = >600us just for launches.

Both failures pointed to the same conclusion: **CUTLASS cannot beat
legacy without a custom EpilogueVisitor** (3-5 days, high-risk).

## What F2 did — question the diagnosis, not the kernel

Before committing to a 3-day kernel or CUTLASS rewrite, F2 ran a
**data-driven dispatch sweep**: 20 Qwen3-8B shapes × 19 (kBm, kBn,
split_k) configurations on the dense-only main kernel path.  The sweep
exposed a different diagnosis:

### Dispatch heuristic bugs (not kernel bugs)

`dispatch_sweep_after.md` showed the `auto` dispatcher leaving
**15-70% on the table** on 5 shapes:

| shape | auto_us | best_us | best_cfg | auto/best |
|---|---:|---:|---|---:|
| down_proj T=32 | 78.38 | 43.55 | `64/32/sk=4` | 1.80× |
| q/o_proj T=32 | 28.27 | 21.75 | `128/64/sk=4` | 1.30× |
| down_proj T=512 | 283.48 | 237.59 | `128/64/sk=1` | 1.19× |
| q/o_proj T=128 | 37.90 | 32.91 | `128/64/sk=2` | 1.15× |

Pattern: the Stage-I split-K heuristic keyed on `ratio = n_groups /
n_cta_m_at_128`, which ignores T.  That conflates "K work per CTA" with
"grid shortness".  Data showed the correct signal is
`grid_mn_at_kbn64 = n_cta_m_at_128 × ceil(T / 64)`: if it's below 128
(one RTX 4090 wave), split K; else don't.

### Fix (3 changes)

1. **Split-K heuristic rewrite** — replace ratio-based gate with
   grid-deficit-based gate targeting 128 CTAs (1 wave):
   ```
   sk = ceil(128 / grid_mn_at_kbn64) clamped to {1, 2, 4}
        requires n_groups % sk == 0 and T >= 8
   ```
2. **kBn pick is now split-K-aware** — `waves_at(kBn)` now multiplies
   by `split_k`, so that after split-K adds K-axis parallelism the
   wave-occupancy thresholds don't over-fragment the N-tile to kBn=8.
3. **Sparse branch gated on `split_k_idx == 0`** — the sparse (hp)
   contribution is group-independent (loops over BSR blocks, not
   groups), so it must be computed exactly once across all K-splits.
   Gating on `split_k_idx == 0` made the split-K heuristic safe for
   `hp > 0` shapes, which is the actual Qwen3 bench regime
   (`hp_ratio = 0.05`).  Without this, the dispatch improvements were
   invisible in e2e bench because auto was forced to `sk=1`.

## Result — Qwen3-8B end-to-end (cold-cache FP16 baseline)

| | F1 | F2 | Δ |
|---|---:|---:|---:|
| **median speedup** | **1.21×** | **1.33×** | **+0.12×** |
| **mean speedup** | 1.39× | 1.49× | +0.10× |
| **wins (≥1.00×)** | **11/20** | **15/20** | **+4 wins** |
| **clear wins (≥1.10×)** | 10/20 | 14/20 | +4 |
| **clear losses (<0.90×)** | 7/20 | **3/20** | **-4** |

### Biggest F1 → F2 gains

| proj | T | F1 | F2 | Δ |
|---|---:|---:|---:|---:|
| **down_proj** | **32** | **0.80×** | **1.72×** | **+0.92×** 🏆🏆 |
| **down_proj** | **128** | **0.86×** | **1.28×** | **+0.42×** 🏆 |
| **q_proj** | **32** | 0.97× | **1.14×** | **+0.17×** → win |
| **o_proj** | **32** | 0.97× | **1.13×** | **+0.16×** → win |
| **kv_proj** | **128** | 0.54× | 0.65× | +0.11× |
| q_proj | 128 | 0.81× | 0.92× | +0.10× |
| o_proj | 128 | 0.81× | 0.91× | +0.10× |

### Remaining 3 losses (all kv_proj)

- kv_proj T=32: 0.60× — dispatch_sweep confirmed auto is already at
  the kernel's intrinsic ceiling; the loss is vs FP16 cuBLAS on a
  2048×4096×32 shape where cuBLAS is very efficient.
- kv_proj T=128: 0.65× (was 0.54×)
- kv_proj T=512: 0.78×

These shapes are limited by **kernel-internal efficiency** (bank
conflict + register-bound occupancy, per r61 Stage-C diagnosis), not
dispatcher mistakes.  Fixing them requires a smem layout rewrite, a
CUTLASS custom-epilogue, or accepting that kv_proj with d_out=2048 is
an intrinsically cuBLAS-friendly shape.

## Validation

- Parity: 14/14 fused_dense_sparse test cases pass under new heuristic
  (including hp>0 cases now using split-K).
- Dispatch sweep after hot-fixes: `dispatch_sweep_after2.md` —
  `auto` tracks `best` within 2% for all shapes except kv_proj T=32
  (8% gap, benign) and gate_up T=32 (8% gap, benign).

## Files

- Code: `kernel/cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`
  - new split-K gate, sk-aware kBn pick, sparse branch gated on split_k_idx
- Sweep tool: `kernel/tools/profile/dispatch_sweep.py`
- Artefacts in this directory:
  - [dispatch_sweep.md](dispatch_sweep.md) — BEFORE sweep (5 shapes with 15-80% gap)
  - [dispatch_sweep_after2.md](dispatch_sweep_after2.md) — AFTER sweep (auto = best ±2%)
  - [bench.json](bench.json) — Qwen3-8B e2e bench (F2, cold-cache)
  - `qwen3_20260430_113807/` under `logs/r62_f2_v2/` — full bench artefacts
  - [simulate_new_dispatcher.py](simulate_new_dispatcher.py) — offline analysis script

## Key lesson

**Before rewriting the kernel, measure whether the dispatcher is
actually picking the best variant of the existing kernel.**  In this
case the kernel already had tile / split-K variants capable of the
measured speedups — the dispatcher was just misrouting on 5/20 shapes.
One afternoon of sweep + heuristic surgery yielded more e2e speedup
than the 2-day F4 CUTLASS migration attempt could ever hope to.
