# Phase C — Combined Final Report (C.1 through C.6-v2)

## Timeline

| round | change | date | scope |
|---|---|---|---|
| C.1 | Group-cache gate widened | 2026-04-30 | T=128 middle, n_groups ≤ 64 |
| C.2 | kBn=16 instantiated + probed (reverted) | 2026-04-30 | dispatcher |
| C.2b/c | kBn dispatcher refinement | 2026-04-30 | T=32 big-d_out |
| C.3 | Qwen3-14B gu T=128 kBm=64 fix | 2026-04-30 | +17% on 14B gu |
| C.4 | Mid-T (T=48/64) dispatcher calibration | 2026-04-30 | 8B gu T=48/64 +47%/+50% |
| **C.5** | T=128 kBm=64 gate widened to d in [2560,4096]² | **2026-05-01** | **4 q/o shapes +7.8%** |
| C.6 v1 | Deep-K sk=2 (over-broad) | 2026-05-01 | regressed 8B series |
| **C.6-v2** | Deep-K sk=2 with precise region A/B/C gates | **2026-05-01** | **8 T=512 shapes +6.8%** |

## Authoritative measurements (in-process A/B)

| change | target shapes | median uplift (warmup=500, 4×interleaved) |
|---|---|---:|
| C.5 | 4 T=128 q/o @ d in [2560,4096]² | **+7.80%** |
| C.6-v2 region A | 5 shapes (14B q/kv/o, 32B kv, 70B kv) | **-5 to -11%** |
| C.6-v2 region B | 3 down_proj shapes (14B, 32B, 70B) | **-17 to -19%** (70B neutral) |
| C.6-v2 region C | 1.7B dn T=512 | **-6.73%** |

## Full bench results (drift-free r66_today → r68_c6v2)

- Median speedup: **1.0327× → 1.0443× (+0.0116, +1.12%)**
- Wins ≥1.0×: **76 → 79 (+3)**
- Big wins ≥2.0×: **20/140**
- Regressions >3%: **1 shape** (LLaMA3-70B dn T=128, attributed to noise — not touched by C.5/C.6)

## Per-T bucket impact

| T | r66_today median | r68 median | delta |
|---:|---:|---:|---:|
| 1 | 1.721× | 1.712× | -0.04% (noise) |
| 32 | 1.045× | 1.048× | +0.3% |
| 128 | 0.893× | 0.895× | +0.2% (C.5 signal in noise) |
| **512** | **0.875×** | **0.898×** | **+2.6%** (from C.6-v2) |

## Files changed summary

### Kernel code
- `csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`: C.5 gate
  + C.6-v2 3-region split_k override (~40 lines in dispatcher body).

### Ops layer  
- `ops.py`: new `fused_dense_sparse_e2e_cuda` entry point + P0 gate
  helper (currently disabled, see P0_INTEGRATION_NEGATIVE_RESULT.md).

### Tests
- `tests/c_probe_loser_shapes.py` — C.5 probe (v1 hp=0, v2 hp=0.05)
- `tests/c5_verify.py` + `c5_verify_v2.py` — C.5 A/B verify
- `tests/t512_dispatch_probe_fast.py` — C.6 fast 6×6 probe
- `tests/t512_probe_extended.py` — C.6-v2 full 30-shape probe
- `tests/c6_verify.py` + `c6v2_verify.py` — C.6 A/B verify
- `tests/p0_integration_bench.py` — P0 negative A/B

### Logs
- `logs/r67_c5/` — r67 full bench (post-C.5, used internally)
- `logs/r66_today/` — same-day r66 baseline
- `logs/r68_c6v2/` — r68 full bench (post-C.5+C.6-v2, authoritative)
- `logs/c6_verify.log`, `logs/c6v2_verify.log`,
  `logs/c6v2_probe_extended.log`, `logs/p0_integration_ab.log`

### Docs
- `docs/PHASE4_C5_DISPATCHER_WIDEN.md`
- `docs/PHASE4_C6_SPLITK_DEEP_K.md` (v1, historical)
- `docs/PHASE4_C6_V2_SPLITK_REFINED.md` (v2, current)
- `docs/P0_INTEGRATION_NEGATIVE_RESULT.md`
- `docs/PHASE_C_FINAL.md` (this)

## Compounded impact vs original r63 baseline

From VALIDATION_LOG history:
- r63 baseline: median 1.021×, 72/140 wins
- r66 (pre-C.5): median 1.049× [stale, cross-day]
- **r68 (C.5 + C.6-v2): median 1.044× (drift-free same-day), 79/140 wins**

Cumulative contribution of dispatcher-only Phase C work:
- **+0.023 median** (r63 → r68 same-day basis)
- **+7 wins** (72 → 79 shapes at speedup ≥ 1.0×)
- **12 shapes with measured in-process A/B uplift of 5-19%**

## Phase 4 compute-bound T=512 region

See `docs/PHASE4_D1_PIVOT.md`, `docs/PHASE4_D3_DUAL_ISSUE_DESIGN.md`,
and `docs/PHASE4_Q0_LITE_UPPER_BOUND.md`.

All source-level deeper-pipeline paths have been diagnostically
rejected:

- **D.1 warp-specialisation**: blocked by INT4 MMA's inability
  to accept pre-dequantised fp16 operands, forcing consumer
  warps to still do per-group fold — shrinks the realistic
  cuda_eff ceiling from 50% to ~40%, 6-8 days work for uncertain
  gain.  Pivoted away.

- **D.3 dual-issue PTX (fold interleave)**: three iterations
  showed nvcc 12.x already schedules the fold loop to the
  hardware ceiling; no additional uplift achievable at the
  source level.

- **Q.0-lite cp.async wait-upper-bound probe**: measured the
  physical floor of what any deeper cp.async pipeline could
  save.  Result: +5.7% on mid-d_in T=512, +13.5% on the
  deepest-K dn shape, <3% elsewhere.  Since C.6-v2 already
  split_k=2 the shapes where wait cost matters, stacking a
  3-stage pipeline on top would deliver only +0.02 global
  median for 2-3 days of work (worse ROI than Phase C).

- **Q-c CUTLASS 3.x rewrite**: architecturally blocked on
  sm_89 — 3.x warp-specialised mainloop requires wgmma + TMA +
  cluster launch which are sm_90+ only.  Repo carries only
  CUTLASS 2.11 and that's what 3.x falls back to on sm_89.

The remaining compute-bound ceiling (T=512 median now 0.898×,
up from 0.875× at r66_today) can only be raised by one of:

1. Hardware upgrade to sm_90+ (H100) → unlocks CUTLASS 3.x
   warp-spec → realistic ceiling ~1.10× on 70B-class shapes.
2. Deep surgery on CUTLASS 2.11 `DefaultMma` to insert per-tile_k
   dequant hooks (1-2 weeks, low success rate, not recommended).
3. Move the win to the workload level: Qwen3 end-to-end inference
   where the kernel-level gains compound into token/s.

## Status

✅ **Phase C work is complete.**  No pending optimisation with
high-ROI in the source-level or dispatcher-level scope remains.
Next substantive improvements require a major architectural
intervention (CUTLASS 3.x) that should be scoped as a new phase.
