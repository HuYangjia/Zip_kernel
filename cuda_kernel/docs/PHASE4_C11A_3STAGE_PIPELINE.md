> **Status**: **DESIGN — implementation starts 2026-05-02 16:00**
> **Target shapes**: Qwen2.5-32B gu / LLaMA3-70B gu, T ∈ {512, 1024, 2048}, currently at speedup 0.45–0.55×
> **Success criterion**: ≥ +10% on the 6 loser shapes, ≤ +1% regression on the other 134 shapes.

# Phase 4 — C.11-A: 3-stage cp.async pipeline rewrite

## 1. Motivation

The r72 C.10 / C.10-v2 dispatcher sweeps (see VALIDATION_LOG) conclusively
showed that on the 32B/70B gu T≥512 cluster, every dispatcher knob
(`split_k`, `kBn`, `kBm`) produces **zero** wall-clock change — 12-cell
sweep variance was <0.04%. The kernel has hit a **per-MMA compute wall**
at ~4082us on 32B gu T=512 (speedup 0.47×).

Root-cause candidates (per [[memory:bd78lejo]]):
- **B1** — serial HFMA2 dequant between MMAs
- **B2** — IMAD swizzle arithmetic
- **B3** — **only 2-stage `cp.async` double-buffering**, CTA calls
  `cp_async_wait_group<0>` after each group

**C.11-A targets B3**. Q.0-lite had previously estimated 3-stage
pipeline at ≈ +2% global ROI (rejected at r68), but that evaluation was
**global-averaged**. The current 6 loser shapes form a tight cluster
where 3-stage uplift can be concentrated — per-shape gain is expected
at +10–15%.

## 2. Current 2-stage pipeline (baseline)

DENSE branch, `fused_dense_sparse_mma_int4.cu` L181–L780:

```
__shared__ uint8_t sW[2][kBm][bytes_per_group];  // 2-stage ring
__shared__ uint8_t sX[2][kBn][bytes_per_group];

// PROLOGUE
issue(g_start, buf=0); commit(); wait<0>; sync;

// MAINLOOP (n_groups iterations)
for g in [g_start, g_end):
    buf = g & 1
    if g+1 < g_end:
        issue(g+1, buf^1); commit()     # fills other slot
    run_mma_pass(buf)                    # ~300 fp16 + 64 IMMA per thread
    if g+1 < n_groups:
        wait<0>                          # BLOCKS until all in-flight done
    sync()
```

Problem: `wait<0>` blocks until **all** in-flight `cp.async` groups complete.
With only 1 group in flight (the g+1 issue), the wait devolves into a
serial barrier — MMA issue pipeline drains every iteration.

## 3. Proposed 3-stage pipeline (C.11-A)

```
__shared__ uint8_t sW[3][kBm][bytes_per_group];
__shared__ uint8_t sX[3][kBn][bytes_per_group];

// PROLOGUE (2 prefetches)
issue(g_start,   buf=0); commit()
if g_start+1 < g_end: issue(g_start+1, buf=1); commit()
wait_group<1>    # wait until only 1 in-flight left  (g_start ready)
sync()

// MAINLOOP
for g in [g_start, g_end):
    buf = g % 3
    if g+2 < g_end:
        issue(g+2, (g+2)%3); commit()    # prefetch 2 ahead
    run_mma_pass(buf)                     # g ready, (g+1) in flight
    if g+1 < g_end:
        wait_group<1>                     # keep ≤1 in flight
    else:
        wait_group<0>                     # drain
    sync()

// epilogue: handled by loop guard above (no separate tail)
```

Invariant: at MMA time, buffer `buf=g%3` is **ready**, buffer `(g+1)%3`
is **in-flight** (will be drained next iter), buffer `(g+2)%3` has been
**just issued** to HBM.

## 4. Shared-memory budget

| component | kBm=128 | kBm=64 |
|---|---|---|
| sW[3][kBm][64] | 24576 B | 12288 B |
| sX[3][kBn=64][64] | 12288 B | 12288 B |
| s_sum_X[3][kBn] if multi-stage | 768 B | 768 B |
| s_scale_u4[kBm][kGrpBuf] | 2048 B | 1024 B |
| s_zero_u4[kBm][kGrpBuf]  | 2048 B | 1024 B |
| s_scale_x[kBn] | 128 B | 128 B |
| s_scale_block[kBm] | 256 B | 128 B |
| **total** | **~42 KB** | **~27 KB** |

sm_89 limits:
- Static smem per CTA: **48 KB** (default, no opt-in needed)
- Dynamic smem per CTA: up to 100 KB with `cudaFuncSetAttribute`

**42 KB < 48 KB**, so we stay on static smem path — no opt-in cost.
**Occupancy**: at 48 KB per CTA, sm_89 allows 2 CTAs/SM (100KB total).
Current 2-stage (~28 KB) allows 3 CTAs/SM. **Potential occupancy drop
from 3 → 2 CTAs/SM on kBm=128 path** — must measure; if this hurts more
than 3-stage helps, we revert.

## 5. Implementation plan

### Phase 1 — code skeleton

- [ ] Add template parameter `int kStages = 2` to the kernel entry point
  (currently `fused_dense_sparse_mma_int4_kernel<...>`).
- [ ] Change `__shared__ alignas(16) uint8_t sW[2][...]` → `sW[kStages][...]`.
- [ ] Update all `sW[buf]` / `sX[buf]` accesses to use `buf = g % kStages`
  (they currently use `buf = g & 1`).
- [ ] Change prologue to issue `kStages-1` groups.
- [ ] Change mainloop to `wait_group<kStages-2>` instead of `wait_group<0>`.

### Phase 2 — dispatcher gate

- [ ] In dispatcher `do_launch`, add branch: if `kBm==128 && kBn in {32,64}`
  AND (`HKUST_V9_FUSED_3STAGE=1` OR auto-enabled) → instantiate with
  `kStages=3`.
- [ ] Env override `HKUST_V9_FUSED_FORCE_STAGES ∈ {2,3}` for probe.
- [ ] Auto-enable rule (tentative): `d_out > 30000 && T >= 512` (same
  cluster as C.8.1a).

### Phase 3 — correctness

- [ ] Parity test on 32B gu T=512 (single shape, existing parity script).
- [ ] Parity test on 70B gu T=1024.

### Phase 4 — perf A/B

- [ ] 6 loser shapes probe with `FORCE_STAGES=2` vs `FORCE_STAGES=3`.
- [ ] Full 140-shape regression (ensure no shape loses >1%).

## 6. Risks and fallback

| Risk | Mitigation |
|---|---|
| occupancy drop 3 → 2 CTAs/SM | measure kBm=128 path; if global regression, gate by `kBm==128 && T≥512` only |
| smem overflow if kBn=128 case exists | kBn=128 path doesn't exist in dispatcher (kBn ∈ {8,16,32,64}) |
| prologue boundary bugs | static_assert `g_end - g_start >= 2` before 3-stage; fall back to 2-stage otherwise |
| correctness: wrong `wait<N>` count | add debug-mode `__syncwarp` after each stage during dev |

## 7. Files touched

- `csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu` (mainloop, +120/-40 lines est.)
- `ops.py` (JIT extra compile flags if any — should be none)
- `benchmarks/c11a_stages_probe.py` (new, A/B 2-stage vs 3-stage on 6 loser shapes)

## 8. Landing checklist

- [ ] parity ok on 3 representative shapes (32B gu T=512, 70B gu T=1024, 8B q T=128 as smoke test)
- [ ] probe shows ≥ +10% on ≥ 4 of 6 loser shapes
- [ ] full 140-shape bench shows median speedup unchanged or better, no shape regresses >1%
- [ ] r73 tag, VALIDATION_LOG entry, `logs/r73_c11a/` archive

## 9. If C.11-A fails (contingency)

- **C.11-B** — LOP3/PRMT dequant fast-path (targets B1)
- **C.11-C** — int8 scale/zero pre-cast (targets B1 small)
- **C.11-D** — full warp-specialisation rewrite (last resort, 6-8 days)
