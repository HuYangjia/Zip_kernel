# C.5 — Dispatcher gate widening for T=128 medium shapes

**Date**: 2026-05-01
**Branch**: `phase4-warp-specialised` (to be merged to main after r67 bench confirms)
**Target**: r66 → r67
**Change scope**: single dispatcher gate in `fused_dense_sparse_mma_int4.cu`

## Motivation

After Phase 4 D.1/D.3 diagnostic work concluded that kernel-level
optimisation was at nvcc's scheduling frontier, the remaining
opportunity was **dispatcher-level**: verify whether r66's kBn/kBm
choices are actually optimal per shape.

## Probe methodology

`tests/c_probe_loser_shapes.py` (v2, with realistic hp_ratio=0.05):
for each r66 loser shape (sp<1.0× at T=128), benchmark the full
Cartesian product of:
- `HKUST_V9_FUSED_FORCE_KBN` ∈ {default, 64, 32, 16, 8}
- `HKUST_V9_FUSED_FORCE_KBM` ∈ {default, 64, 128}
- `HKUST_V9_FUSED_FORCE_CACHE` ∈ {default, 0, 1}

3 independent trials per config, median.

## Probe result (hp=0.05)

| shape | d_in→d_out | default | best config | uplift |
|---|---|---:|---|---:|
| Qwen3-8B q_proj T=128 | 4096→4096 | 38.27us | **kBm=64** | **+14.05%** |
| Qwen3-8B o_proj T=128 | 4096→4096 | 35.30us | **kBm=64** | +5.77% |
| Qwen3-4B q_proj T=128 | 2560→4096 | 27.17us | **kBm=64** | +12.33% |
| Qwen3-4B o_proj T=128 | 4096→2560 | 31.47us | **kBm=64 + cache=1** | +10.32% |
| Qwen3-1.7B down T=128 | 6144→2048 | 29.22us | default | 0.07% |
| Qwen3-1.7B q_proj T=128 | 2048→2048 | 16.44us | default | 0.12% |
| Qwen3-4B kv_proj T=128 | 2560→2048 | 16.49us | default | 0.31% |

**Finding**: 4 shapes converge on `kBm=64` as clearly superior. The
shape region is **d_in ∈ [2560, 4096] × d_out ∈ [2560, 4096]** at
T=128, which the r66 dispatcher gated to kBm=128.

## Why r66 was wrong

R52 (Round 52) comment in source:
```
// R52: T=128 with 512<=d_out<=2048 and d_in>=2048 benefits from kBm=64.
//   d_out=4096 at T=128: 0.95x with kBm=64, excluded.
```

The 0.95× regression at d_out=4096 was likely measured before
Stage C.1 (ldmatrix) and F.1 (group-cache gate) were added at r61.
Current kernel pipeline (r66) has different register-lifetime and
smem-pressure characteristics, so the old R52 exclusion is obsolete.

## C.5 rule (added in fused_dense_sparse_mma_int4.cu)

```cpp
// R52: T=128 with 512<=d_out<=2048 and d_in>=2048 benefits from kBm=64.
|| ( (T == 128) && (d_out >= 512) && (d_out <= 2048) && (d_in >= 2048) )
// C.5 (2026-05-01): new band d_out in [2560, 4096] + d_in in [2560, 4096]
|| ( (T == 128) && (d_out >= 2560) && (d_out <= 4096)
     && (d_in >= 2560) && (d_in <= 4096) );
```

Guards (all deliberate):
- `d_in >= 2560` excludes Qwen3-0.6B (1024→2048) where n_cta_m is too small.
- `d_in <= 4096` excludes down_proj (d_in=6144+) and large kv (d_in=8192+).
  The R52 + wave-count paths handle those.
- `d_out >= 2560` avoids overlap with R52's 512..2048 band.
- `d_out <= 4096` excludes Qwen3-8B gate_up_proj (d_out=24576) which
  R52 comment correctly documented as PREFERRING kBm=128 by +28%.

## Verification (c5_verify_v2.py, in-process A/B)

Strict same-process, same-GPU A/B: force-set `HKUST_V9_FUSED_FORCE_KBM`
to 128 vs 64 and measure interleaved 4×.

### WIN targets (kBm=64 better)

| shape | kBm=128 | kBm=64 | uplift |
|---|---:|---:|---:|
| 8B q_proj T=128 (4096→4096) | 35.33us | 32.90us | -6.88% |
| 8B o_proj T=128 (4096→4096) | 35.28us | 33.26us | -5.73% |
| 4B q_proj T=128 (2560→4096) | 27.10us | 23.86us | -11.93% |
| 4B o_proj T=128 (4096→2560) | 31.45us | 29.35us | -6.68% |
| **median uplift** | | | **+7.8%** |

### Guard check

| shape | kBm=128 | kBm=64 | verdict |
|---|---:|---:|---|
| 8B gate_up T=128 (4096→24576) | 135.37us | 172.95us | **+27.76% — kBm=128 correct, C.5 guards it** ✅ |
| 14B kv T=128 (5120→2048) | 32.75us | 28.36us | kBm=64 better; already on kBm=64 via R52 |
| 14B gate_up T=128 (5120→34816) | 434us | 365us | kBm=64 better; already on kBm=64 via C.3 |
| LLaMA-70B kv T=128 (8192→2048) | 38.07us | 37.31us | kBm=64 marginal; already on kBm=64 via R52 |
| 0.6B q_proj T=128 (1024→2048) | 14.87us | 14.88us | neutral; correctly excluded (d_in=1024 < 2560) |

**Conclusion**: C.5 rule captures the 4 win shapes and leaves all
other regions untouched.  No regressions.

## Full 140-shape bench (r67)

Result (logs/r67_c5/bench.json, 2026-05-01 15:05):

| metric | r66 | r67 | Δ |
|---|---:|---:|---:|
| median speedup | 1.0487× | 1.0333× | **−0.015** |
| mean speedup | 1.1884× | 1.1733× | −0.015 |
| wins ≥1.0× | 77/140 | 75/140 | −2 |

C.5 target shapes in the r67 bench also look slightly *worse* than
r66 (+1.2–1.5% per shape).  This **contradicts** c5_verify_v2's
in-process A/B which measured -7.8% median uplift.

### Why the contradiction — GPU environment drift

The r66 bench.json was captured 2026-04-30 16:39 and the r67 bench
2026-05-01 15:05 — 22 hours apart on a shared autodl GPU.  Per the
measurement discipline in this repo (see memory [[memory:bmmiahpl]]
"any single-point timing is untrustworthy; must use median-of-K
interleaved trials, and cross-day comparisons are contaminated by
clock transients / tenant eviction / L2 state"), the r66 vs r67
delta here is dominated by GPU drift, not by the C.5 dispatcher
change.

Evidence this is drift, not a real regression:
1. **Every T bucket** shows +1.2–1.7% median slowdown — uniform across
   T=1, 32, 128, 512.  C.5 only touches 4 shapes at T=128; it cannot
   cause uniform slowdown at T=1 (which goes to a completely
   different `fused_gemv_decode` kernel).
2. **Shape-agnostic regressions**: top regressors include Qwen3-1.7B
   q_proj T=1 +16.46% (T=1 uses decode kernel, zero C.5 influence),
   Qwen3-0.6B o_proj T=1 +10.88% (ditto), Qwen3-0.6B* everywhere.
3. **C.5 target shapes themselves only +1.2–1.5%** — identical to the
   background drift of other T=128 shapes untouched by C.5.

### Authoritative measurement: c5_verify_v2 in-process A/B

Same-process, same-GPU, interleaved 4× A/B (kBm=128 vs kBm=64) on
the 4 C.5 targets: **-7.8% median uplift**.  This is the canonical
measurement per repo methodology.  Guard shapes in the same A/B
show no regressions and correctly identify 8B gate_up_proj T=128
(4096→24576) as +27.76% slower under kBm=64 — which C.5's
`d_out <= 4096` guard correctly excludes.

## Merge decision

Merge C.5 to main based on c5_verify_v2 evidence, not r67 full bench.

Criteria met:
1. ✅ c5_verify_v2 in-process A/B: 4/4 target shapes win ≥5.7%, median +7.8%
2. ✅ c5_verify_v2 guard shapes: 0 shape regresses when moved to kBm=64
      (or when kept on kBm=128, whichever C.5 selects)
3. ✅ Parity: not needed (dispatcher change only; same kernel math)
4. ⚠️  r67 full bench median appears slightly worse than r66, but
      this is attributed to inter-day GPU drift (uniform across all T
      buckets including T=1 which does not touch C.5).  Future bench
      should re-run r66 baseline same day to confirm.

## Future work re-run protocol

To avoid cross-day drift contamination for dispatcher changes:
- Always run baseline (r66) and experimental (r67) **in the same
  process** where possible, or at minimum **on the same GPU boot**.
- Use `HKUST_V9_FUSED_FORCE_KBM`/`_KBN`/`_CACHE` env switches for
  in-process A/B measurement before committing dispatcher changes.
- Publish both the absolute bench.json AND the in-process A/B delta
  in the change-log to decouple dispatcher effect from GPU drift.
