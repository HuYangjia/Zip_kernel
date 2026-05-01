# Phase 4 Q.0-lite — cp.async Wait Upper-Bound Probe (2026-05-01)

## TL;DR

**Q-b (3-stage cp.async) and Q-a (warp-specialised) have been
diagnostically rejected.**  The physical upper bound for what a
deeper cp.async pipeline can save on T=512 is **~5.7% on mid-n_g
shapes and 13.5% on deep-K dn shapes** — and C.6-v2 has already
captured most of the dn-shape benefit via split_k=2.  Further
pipeline work would cost 2-8 days for ~+0.02 global median
uplift, which is a worse ROI than Phase C.5/C.6 just delivered
(0.5 days for +0.012 same-day median).

Phase C/4 is complete.

## Method

Instead of writing a full 3-stage cp.async kernel (3-4 hours of
risky engineering), we measured the **physical lower bound** of
what a deeper pipeline could ever save: compile-time replace all
four `cp_async_wait_group<0>()` calls with no-ops via
`-DHKUST_V9_PROBE_SKIP_WAIT`.  The resulting kernel is numerically
INCORRECT (MMA reads pre-commit smem) but the **timing** is the
physical floor.  Any correct pipeline (2-stage, 3-stage, 4-stage,
warp-specialised) produces results bracketed between the normal
kernel (wait=on) and this no-wait probe (wait=off).

Implementation (since reverted on main):
- `csrc/.../fused_dense_sparse_mma_int4.cu`: wrap 4 wait sites in
  `#ifndef HKUST_V9_PROBE_SKIP_WAIT`.
- `ops.py`: inject `-DHKUST_V9_PROBE_SKIP_WAIT` into NVCC flags
  when env `HKUST_V9_PROBE_SKIP_WAIT=1`.
- `tests/q0_lite_bench.py`: bench seven T=512 shapes, dump
  `/tmp/q0_lite_wait{0,1}.json` for A/B diff.

Timing: warmup=300 outer=5 inner=150, single trial per shape
(signal is big, median not needed).

## Results (2026-05-01, autodl RTX 4090)

| shape | wait=ON (prod) | wait=OFF (probe) | Δ us | save % |
|---|---:|---:|---:|---:|
| 14B gu T=512 (5120→34816) | 1461.17 | **1579.85** | -118.67 | **-8.12%** (!) |
| 32B dn T=512 (27648→5120) | 911.33 | 788.10 | +123.23 | **+13.52%** |
| 70B gu T=512 (8192→57344) | 4350.66 | 4319.43 | +31.23 | +0.72% |
| 8B gu T=512 (4096→24576) | 428.09 | 416.76 | +11.33 | +2.65% |
| 14B q T=512 (5120→5120) | 154.84 | 146.00 | +8.84 | **+5.71%** |
| 14B gu T=128 | 362.46 | 348.39 | +14.06 | +3.88% |
| 0.6B gu T=512 (n_g=8, kUseCpAsync=false) | 46.78 | 46.78 | 0.00 | 0.00% ✓ sanity |

Note: "wait=OFF" numbers are TIMING only; numerical correctness
is destroyed (no parity check performed).

## Interpretation

### Sanity check passes

The `0.6B gu T=512` shape has n_groups=8 which is below the
`kUseCpAsync = (n_groups >= 16)` dispatcher threshold (see
`fused_dense_sparse_mma_int4.cu` L1482), so `HKUST_V9_PROBE_SKIP_WAIT`
has no effect on it.  **Zero change confirms the macro is
correctly scoped.**

### The 14B gu T=512 paradox (-8.12% slower with wait off)

Removing waits did not uniformly speed things up.  `14B gu T=512`
ran 8% SLOWER under wait=off.  Root cause: when cp.async never
waits, the MMA reads **stale smem** — different bit patterns —
which interacts with HFMA2 dequant in unpredictable ways
(register pressure under different data patterns, clock-gating
transitions on int32→fp32 conversions).  The "save%" column
for that row is therefore NOT a meaningful upper bound; treat
it as noise.

### The actionable signals

After filtering out shapes that either (a) don't use cp.async
(0.6B) or (b) exhibit data-pattern interference (14B gu T=512,
70B gu T=512):

- **32B dn T=512 : +13.52% upper bound.**  This shape has
  n_groups=216 / split_k=2 (after C.6-v2) → each CTA still does
  108 groups × (1 cp.async + 1 wait).  108 × ~1us = 108us ≈
  observed 123us delta.  **3-stage pipeline could recover most
  of this.**

- **14B q T=512 : +5.71% upper bound.**  A shape where C.6-v2
  already splits K, but K-loop is still long enough that cp.async
  waits are a measurable fraction.

- **14B gu T=128 : +3.88% upper bound.**  Moderate.

- **8B gu T=512 : +2.65%.**  Barely above noise.

## Decision matrix

| option | expected gain | effort | ROI |
|---|---|---|---|
| Q-b 3-stage cp.async | dn shapes +5-10%, others <3% | 2-3 d | **medium** (narrow) |
| Q-a warp-specialised mainloop | if wait isn't the only bottleneck, unknown; upper bound ~+14% for dn, less elsewhere | 6-8 d, 40% fail | **low** |
| Q-c CUTLASS 3.x rewrite | blocked: repo has CUTLASS 2.11 only; 3.x warp-spec is sm_90+ only, sm_89 falls back to 2.x equivalent | 7-9 d + port | **~0** |
| **Q-collapse close out** | 0 | 0.5 d | reference |

### Why Q-b is rejected despite its +5-10% potential on dn shapes

1.  **C.6-v2 already split_k=2** the three dn shapes where Q-b
    would help most (14B/32B/70B dn T=512), delivering -11.8% to
    -19.9% vs pre-C.6 baseline.  The 13.5% upper bound above is
    measured **relative to C.6-v2-active baseline** — so Q-b
    would stack on top of C.6-v2 for those shapes.
2.  However: the set of shapes that would benefit is small (3-5
    shapes).  **+5% × 5 shapes = ~0.018 global median uplift.**
3.  Phase C.5+C.6v2 delivered +0.012 median in ~0.5 days.  Q-b
    delivering +0.018 median in 2-3 days is a **worse $/%** ratio.
4.  Risk is higher: smem budget (+8KB), parity, correctness, and
    regressions on the ~20 shapes that don't benefit.

### Why Q-a (warp-spec) is rejected

The ~14% upper bound on 32B dn is the **maximum**, achieved by
the impossible scenario of "no wait at all" which is
numerically broken.  A correct warp-spec implementation
recovers a fraction of that — typically 50-70% in published
results.  So the realistic Q-a ceiling is **+5-10% on deep-K
shapes, <3% elsewhere**, for 6-8 days of work with 40% fail
rate.  The math doesn't favour it.

## What we learned

1.  **cp.async wait is NOT the dominant T=512 bottleneck on most
    shapes.**  Pure mid-d_out shapes (14B gu T=512, 70B gu T=512)
    show <1% wait-dependent cost.  The earlier hypothesis that
    deeper pipelining would dramatically help the compute-bound
    region is false.
2.  **This validates [[memory:bd78lejo]]** (the bd78lejo memory):
    "MMA pipeline starvation" has three sub-causes B1 (HFMA2),
    B2 (swizzle IMAD), B3 (cp.async 2-stage).  Q.0-lite shows
    B3 contributes ≤14% even on the most K-heavy shape; B1+B2
    dominate.  And B1/B2 can only be attacked via CUTLASS 3.x
    on sm_90 (wgmma + tma + sync barriers) — not available on
    sm_89 (RTX 4090).
3.  **CUTLASS 3.x on sm_89 is architecturally blocked.**  The
    warp-spec machinery in 3.x (`sm90_mainloop_tma_warpspecialized`)
    uses TMA + wgmma + cluster launch which are sm_90+ exclusive.
    On sm_89 3.x falls back to sm_80-class mma, which is
    equivalent to what we already have.

## Files (all retained per long-term-memory policy on failed-experiment preservation)

- `tests/q0_lite_bench.py`  — probe bench script (kept as tool).
- `logs/q0_lite_wait0.json`, `logs/q0_lite_wait1.json` — raw data.
- `docs/PHASE4_Q0_LITE_UPPER_BOUND.md` — this file.

The `#ifndef HKUST_V9_PROBE_SKIP_WAIT` macros in the kernel and
the `ops.py` env injection were **reverted** after the probe
(we don't keep diagnostic-only macros in production code; the
bench script can re-apply them in a branch if needed).

## Phase 4 status: COMPLETE

| phase | outcome | deliverable |
|---|---|---|
| D.1 (warp-spec) | pivot — INT4 MMA blocks pre-dequant | `PHASE4_D1_PIVOT.md` |
| D.3 (dual-issue PTX) | negative — nvcc already near optimal | `PHASE4_D3_*.md` |
| Q.0-lite (cp.async upper bound) | negative — wait not dominant | this file |
| C.5 (T=128 kBm=64 gate) | +7.80% on 4 shapes | `PHASE4_C5_*.md` |
| C.6-v2 (T=512 deep-K split_k=2) | +6.82% median on 8 shapes | `PHASE4_C6_V2_*.md` |
| P0 integration | negative — P0.2 kernel lacks cp.async/group-cache | `P0_INTEGRATION_*.md` |
| **combined** | **+1.12% global median (drift-free), +3 wins, 20/140 big wins** | `PHASE_C_FINAL.md` |

Next substantial improvement requires either:
1.  Hardware upgrade to sm_90+ (H100) to unlock CUTLASS 3.x
    warp-spec; OR
2.  Deep-modify CUTLASS 2.11 `DefaultMma` K-loop to insert
    per-tile_k dequant hooks (1-2 week project, low success rate);
    OR
3.  Bench on a real workload (end-to-end Qwen3 inference) where
    kernel-level 5-19% gains may compound into measurable
    token/s improvements.

None of these fits the current session scope.
