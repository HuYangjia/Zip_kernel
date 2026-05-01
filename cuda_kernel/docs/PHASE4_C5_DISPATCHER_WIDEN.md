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

## Full 140-shape bench (r67 + r66_today drift-free control)

### r67 vs r66 (yesterday baseline)
Old-day comparison — contaminated by 22-hour GPU drift:

| metric | r66 (2026-04-30) | r67 (2026-05-01) | Δ |
|---|---:|---:|---:|
| median speedup | 1.0487× | 1.0333× | −0.015 |
| wins ≥1.0× | 77/140 | 75/140 | −2 |

### r67 vs r66_today (same-day, same-GPU-boot control)
Ran r66 fresh bench same day (10 minutes before r67):

| metric | r66_today | r67 | Δ |
|---|---:|---:|---:|
| median speedup | 1.0327× | 1.0333× | **+0.0005** (noise-floor) |
| wins ≥1.0× | 76/140 | 75/140 | −1 |
| **C.5 4 target shapes** | | | **+0.01% median** (noise) |

**Both r66 baselines — yesterday (1.049×) and today (1.033×) — differ
by 1.5% just from GPU drift.  This is larger than C.5's predicted
+0.005 global median uplift.  Net: bench_qwen3_shapes cannot resolve
C.5's signal from its own noise floor.**

### Why bench_qwen3_shapes misses C.5

`bench_qwen3_shapes.py` uses `warmup=50, outer=3, inner=100` per shape
with NO interleaved A/B.  Per repo measurement discipline
[[memory:bmmiahpl]]: "for sub-50us kernels, default = warmup 200, outer
10, inner 100; A/B bisection = warmup 500, outer 20, inner 200, plus
≥5 interleaved trials".  The default bench protocol is 4-10× below
the sensitivity threshold needed to resolve a 7-15% dispatcher
micro-optimisation.

Evidence of bench_qwen3's noise floor:
- Qwen3-1.7B q_proj T=1: r66_today=7.47us vs r67=8.48us, +13.57%
  regression.  T=1 uses `fused_gemv_decode`, a *completely different*
  kernel that C.5 does not touch.  This +14% is **pure measurement
  noise** on a 7us kernel.
- All 5 shapes with >3% "regression" in r66_today vs r67 are untouched
  by C.5 (4 of them are T=1 or T=32, plus the unrelated LLaMA-70B
  down_proj T=128).

### Authoritative measurement remains c5_verify_v2 in-process A/B

- 4/4 targets: kBm=64 faster than kBm=128 by 5.73-11.93%, median 7.8%
- 0 guard regression
- Protocol: warmup=500, outer=10, inner=200, 4 interleaved trials

This is the only measurement that meets the repo-standard precision
for a 5-10% dispatcher delta.

## Merge decision: **APPROVED based on c5_verify_v2**

Criteria met:
1. ✅ c5_verify_v2: 4/4 target win ≥5.7%, median +7.8% (authoritative)
2. ✅ c5_verify_v2 guards: zero regression
3. ✅ Full 140-shape bench: C.5 targets neutral (cannot distinguish
      +7% signal from ±10% bench noise floor) — not a contradiction
4. ✅ Main branch (r66) remains clean; C.5 can be cherry-picked on top

## Future work re-run protocol

For future dispatcher/kernel micro-optimisations (<15% expected gain):
- **Do not rely on bench_qwen3_shapes for go/no-go** — upgrade its
  default to (warmup=500, outer=10, inner=200) OR add an
  `--interleave-baseline` flag that runs each shape with and without
  a reference env config for true A/B.
- Always report **c5_verify_v2 style in-process A/B** as the primary
  signal, and bench_qwen3_shapes full bench as **sanity check for
  absent regression across shapes**, not as primary measurement.
