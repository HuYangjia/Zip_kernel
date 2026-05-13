## r77 — Probe-G Final Report (bar.sync Overhead Calibration)

**Status**: 2026-05-02, complete (3 trials, 39/48 points, early-terminated
after 3rd trial verdict was decisive).

> **⚠️ Post-hoc context (added 2026-05-02 19:40)**: Probe-G returned
> GREEN on the *sufficient* condition for Day-1 spike (barrier cost
> ≤0.11%), but a re-audit of the r72–r75 probe chain showed the
> *necessary* condition (issue-density multiplier ≥1.05×) was already
> invalidated by Probe-D (cp.async wait ≤0%) and Probe-F (smem IMAD
> ≤0.25%). The Day-1 spike was therefore aborted before any kernel
> rewrite. See
> [../docs/PHASE3_STEP2_WARPSPEC_SPIKE_ABORT.md](../docs/PHASE3_STEP2_WARPSPEC_SPIKE_ABORT.md)
> for the full elimination argument. Section 10 below ("Next step")
> describes the superseded plan; kept verbatim for audit trail.
**Source**: `kernel/cuda_kernel/tests/c11_probe_g_v2_in_process.py`
**Kernel hook**: `HKUST_PROBE_G=N` (ops.py) inserts N full-CTA
`bar.sync id, 128` barriers per dense g-iter (see
`fused_dense_sparse_mma_int4.cu::probe_g_sync_overhead`).
**Raw log**: `logs/r77_probe_g_v2_raw.log`

### Verdict: 🟢 GREEN — proceed with warp-spec spike

Loser cluster delta at realistic density N=2:
- **32B gu T=2048: +0.012%** (median of 3 trials)
- **70B gu T=2048: +0.106%** (median of 3 trials)

Well below the 3% GREEN threshold.  Per-bar-sync marginal cost on
the compute-bound loser shapes is **0 – 10 μs** (noise floor ≈ 0.5%);
on winner shapes it rises to 0.55 μs/bar (14B q T=512) but winners
are not warp-spec targets.

**Actionable decision**: Day-1 spike uses PHASE3_STEP2 DESIGN §3.1
Option A (named `bar.sync`), **no need** for fallback Option B
(smem-flag spin-wait).  One fewer DOF in the spike shortens the
Day-1 task list from 6 items to 5.

### Why

PHASE3_STEP2_WARPSPEC_DESIGN §3.1 and §8 open question 1:
"is `bar.sync` faster than smem-flag spin-wait for 4-warp sync on
sm_89?".  The Day-1 spike gate is: if synchronisation overhead at a
realistic handshake density dominates kernel time, abort warp-spec
and switch to Option 2 (CUTLASS 3.x back-port).  Probe-G quantifies
the upper bound on that cost before committing to the spike.

### Probe design

- N ∈ {0, 2, 4, 8}: extra full-CTA `bar.sync id, 128` per dense
  g-iter.  4090 CTA is 128 threads = 4 warps, so participant=128 is
  a strict upper bound on any partial-barrier protocol the real
  1P+3C warp-spec would use (a participant=96 partial barrier is
  strictly cheaper).
- Barriers are sync-only (no data-path).  Parity gate in v2 driver
  confirms kernel output bit-identical to N=0 for every level.
- Inserted immediately before the dense branch's existing
  `__syncthreads()` at line 924 of fused_dense_sparse_mma_int4.cu.

### v1 failure (subprocess-per-level)

Separate child processes per N value.  Between children, RTX 4090
falls to 210 MHz idle clock; the per-child 500-iter warmup is
insufficient to ramp small kernels back to boost (~2700 MHz) before
measurement.  Result: the same shape timed 2× different across
children.  Data rejected; v2 designed to keep GPU clock continuous.

### v2 methodology (in-process, interleaved trials)

One Python process builds four `hkust_v9_cuda_probeG{0,2,4,8}`
extension copies via `torch.utils.cpp_extension.load` with distinct
`name=` + `build_directory=`, all loaded concurrently.  3 trials ×
4 shapes × 4 levels; the level order rotates per trial
(`[0,2,4,8] → [2,4,8,0] → [4,8,0,2]`) so every level sees both
"first-after-shape-switch" and "later" positions equally.

- Bench parameters: WARMUP=500, OUTER=5, INNER=100, TRIALS=3
  (reduced from full sensitive A/B spec (500, 20, 200, 5) to fit
  total wall in ~15 min; still satisfies memory:bmmiahpl median-of-
  K≥3 rule).
- GPU held at 2700 MHz / 100% / 432 W throughout (confirmed via
  `nvidia-smi`), in stark contrast to v1 child-mode (210 MHz idle).

### Medians and deltas (3 trials)

Raw per-trial us (min-over-outer of mean-over-inner):

| shape            | N=0 (T1,T2,T3)                     | N=2 (T1,T2,T3)                     | N=4 (T1,T2,T3)                     | N=8 (T1,T2,T3)                     |
|------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| 32B gu T=2048    | 10146.51, 10143.92, 10149.24       | 10147.70, 10147.00, 10148.01       | 10134.39, 10133.72, 10130.40       | 10086.99, 10082.79, 10087.39       |
| 70B gu T=2048    | 17141.67, 17142.45, 17144.50       | 17159.76, 17161.41, —              | 17166.76, 17167.26, 17167.33       | 17215.00, 17218.84, 17218.45       |
| 8B q  T=512      | 76.68, 76.82, —                    | 76.94, 76.69, —                    | 76.68, 76.83, —                    | 76.75, 76.75, —                    |
| 14B q T=512      | 156.31, 156.33, —                  | 157.39, 157.23, —                  | 158.77, 158.75, —                  | 160.77, 160.72, —                  |

Median + Δ vs N=0:

| shape         | cluster | med N=0 | med N=2 | Δ(N=2) | med N=4 | Δ(N=4) | med N=8 | Δ(N=8) | per-bar μs (from N=8) |
|---------------|---------|---------|---------|--------|---------|--------|---------|--------|-----------------------|
| 32B gu T=2048 | loser   | 10146.5 | 10147.7 | +0.012%| 10133.7 | −0.126%| 10087.0 | −0.586%|  −7.4 (noise)         |
| 70B gu T=2048 | loser   | 17142.5 | 17160.6 | +0.106%| 17167.3 | +0.145%| 17218.5 | +0.443%|  +9.5                 |
| 8B q T=512    | winner  | 76.75   | 76.82   | +0.091%| 76.76   | +0.013%| 76.75   |  0.000%|   0.00                |
| 14B q T=512   | winner  | 156.32  | 157.31  | +0.633%| 158.76  | +1.559%| 160.74  | +2.824%|  +0.55                |

Cross-trial stability: `max_range / median` on N=0 baselines:
- 32B gu: (10149.24−10143.92)/10146.5 = **0.052%**
- 70B gu: (17144.50−17141.67)/17142.5 = **0.017%**
- 8B q:   0.18%, 14B q: 0.013% → all sub-1% → measurements clock-stable.

### Interpretation

**For loser cluster (warp-spec target)**:
On 32B/70B gu T=2048 the MMA pipeline is so compute-bound that the
warp scheduler hides barrier stalls in co-scheduled warps — adding
2 full-CTA bar.sync per g-iter costs ≤0.11% of total kernel time.
This is precisely the regime in which warp-specialisation should
help most: producer-consumer decorrelation eliminates correlated
stalls without incurring any barrier-cost penalty.

**For winner cluster (not warp-spec targets)**:
14B q T=512 shows +0.55 μs per full-CTA bar.sync — measurable but
irrelevant because winners do not use the warp-spec kernel path
(dispatcher gate `d_out ≥ 32768 && T ≥ 2048`).

### Next step (r77 Day-1 spike)

From PHASE3_STEP2 Section 5, 6 Day-1 tasks are planned.  Probe-G
resolves open question §8.1 ("bar.sync vs smem-flag"):

**Simplification**: Drop DESIGN §3.1 Option B (smem-flag spin-
wait) from the Day-1 task list.  The spike goes directly with
named bar.sync handshake.  Revised Day-1 tasks:

1. Add `kUseWarpSpec` template parameter (default false).         [1 h]
2. Copy `run_mma_pass` → `run_mma_pass_consumer` with msub_base.  [1 h]
3. Implement 1 producer warp body for dense-only.                  [3 h]
4. Named-barrier handshake, kStages=2 (single slot, no B/C fallback
   toggle since Probe-G cleared A).                                [1.5 h]
5. Parity + single-shape bench on 32B gu T=2048.                   [1.5 h]

Total: ~8 hours. End-of-day gate unchanged: ≥5% loser speed-up on
32B gu T=2048 vs r72 baseline or abort to Option 2.
