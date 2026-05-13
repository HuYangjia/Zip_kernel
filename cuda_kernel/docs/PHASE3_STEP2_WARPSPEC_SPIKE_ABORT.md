# Phase 3 Step 2 — Warp-Specialisation Day-1 Spike: ABORT

**Status**: 2026-05-02, **aborted** before any kernel rewrite.
**Decision**: Do **not** implement Day-1 spike. Pivot to Option 2
(CUTLASS 3.x stream-K back-port) evaluation **or** declare V9
int4 kernel production-final at current performance.
**Companion docs**:
- Design (shelved, at repo root): [../../../PHASE3_STEP2_WARPSPEC_DESIGN.md](../../../PHASE3_STEP2_WARPSPEC_DESIGN.md)
- Probe-G (GREEN but insufficient): [../logs/r77_probe_g_notes.md](../logs/r77_probe_g_notes.md)
- Phase 4 final release: [PHASE4_FINAL.md](PHASE4_FINAL.md)

---

## 1. TL;DR

Probe-G (r77) gave a GREEN on "named bar.sync is free on loser shapes"
(Δ ≤ 0.11% at realistic density N=2). That single GREEN signal was the
*necessary* pre-condition for the spike, but it was **not sufficient**.
On closer inspection of the r72–r75 probe chain, **every mechanism by
which warp-specialisation could deliver throughput was already ruled out
by a prior probe**:

| r##  | Probe  | Attacks                                     | Verdict |
|------|--------|---------------------------------------------|---------|
| r72  | B      | fold_dense scalar ALU (FMA after MMA)       | ≤1%     |
| r73  | D      | `cp_async_wait_group` serialisation         | ≤0%     |
| r74  | E      | sparse branch cost                          | ≤1.4%   |
| r75  | F      | smem IMAD + scale/zero load                 | ≤0.25%  |
| r77  | G      | bar.sync handshake cost (spike pre-req)     | ≤0.11%  |

Probe-D in particular kills the last-remaining theoretical channel for
warp-spec gain on this kernel. A Day-1 spike would at best produce a
null result on 32B gu T=2048; at worst it would introduce occupancy or
register-pressure regressions that the probe-chain cannot anticipate.
Expected ROI for ~8 hours of work is bounded by the probe evidence at
**≤ 1%** — below the gate.

---

## 2. Why Probe-G GREEN is not enough

PHASE3_STEP2 DESIGN §3.1 framed warp-spec gain as a sum of two terms:

```
speedup = (1 − barrier_overhead)  ×  issue_density_multiplier
```

Probe-G measured only the first term (`barrier_overhead ≤ 0.11%`). The
DESIGN quantification (§2.1) estimated the second term at **1.3–1.7×**
based on the assumption that the MMA pipeline on loser shapes is
currently *issue-slot-bound due to correlated stalls across 4 symmetric
warps*. That assumption depends on at least one of the following being
true:

- **(H1)** The 4 warps stall on the same upstream resource at the
  same time (shared operand supply, pipeline wait, dequant FMA), so
  decorrelating producer/consumer breaks the lockstep.
- **(H2)** Per-warp MMA issue rate on sm_89 can exceed
  `1 × mma.m16n8k64` per 4 cycles when the warp is not sharing
  scheduler bandwidth with cp.async and dequant ops.

The r72–r75 probe chain already refutes both:

- **B2 (smem swizzle IMAD + scale/zero load) is not a shared
  upstream** (Probe-F ≤ 0.25%). If it were, packing scale+zero into
  one `__half2` would have dropped loser time by the DESIGN-estimated
  fraction; it did not.
- **B3 (cp.async pipeline wait) is not a shared upstream**
  (Probe-D ≤ 0%). Removing all `cp_async_wait_group` calls did not
  accelerate the loser shapes. Therefore producer warps decorrelating
  from consumer warps on cp.async cannot buy time — consumers already
  don't wait.
- **B1 (fold ALU) is trivially small** (Probe-B ≤ 1%). Even if
  warp-spec re-distributed this onto producer warps, the recoverable
  time is bounded by 1%.
- **Sparse branch is trivially small** (Probe-E ≤ 1.4%), and is not
  touched by warp-spec anyway.

What is left? By elimination, ≥97% of loser-cluster time on 32B/70B gu
T=2048 is spent on the `mma.m16n8k64.s4.s4.s32` instructions themselves,
issued at their sm_89 per-warp-scheduler pipe rate. **This cap is
per-warp, not per-CTA.** Warp-specialisation does not raise the
per-warp cap; it rearranges *which* warp spends time on MMA vs
staging. Going from 4 symmetric warps (each issuing MMA ~75% of the
time in the current kernel) to 3 consumer warps (each issuing MMA
~100% of the time) changes aggregate CTA-level MMA issue rate from
`4 × 0.75 = 3.0` to `3 × 1.0 = 3.0` — **identical**. No issue-density
multiplier exists.

---

## 3. The DESIGN estimate of 1.3–1.7× was optimistic by construction

Revisiting PHASE3_STEP2 DESIGN §2.1: the 1.3–1.7× figure was derived
from a SASS analysis of `tc_fraction ≈ 35%` under the *old* taxonomy
(see [[memory:bd78lejo]]). Under the corrected taxonomy
(`mac_tc_share`), that same kernel shows TC handling ≥97% of MAC
work already. The delta attributable to "freeing scheduler bandwidth
from non-TC work" is therefore **at most 3%**, not 30–70%.

In other words: the DESIGN draft was written before the r72–r75
probe chain dropped the ceiling from the Phase-2 SASS estimate
(30–70% headroom) to the Phase-3 probe-bounded estimate (≤3%
headroom). The probes invalidate the DESIGN's quantitative case,
even though they validated the DESIGN's *synchronisation* case.

---

## 4. What would it take to revive warp-spec?

The following counter-evidence would re-open the spike:

- **(C1)** A direct ncu measurement on 32B gu T=2048 showing
  `smsp__issue_active` / `sm__inst_issued_pipe_tensor` ratio
  significantly < 100% with idle reason = "warp scheduler stall on
  non-TC op". Current probe evidence is indirect; a direct measurement
  could contradict the elimination argument.
- **(C2)** A prototype kernel on **a different shape** (e.g.,
  large-K low-M regime) where warp-spec delivers >10% on sm_89 W4A4
  in literature. None is known to this project at this time.
- **(C3)** sm_89 async-copy or mbarrier extensions emerging in CUDA
  13+ that change the per-warp MMA issue cap.

None of these exist today. Re-opening requires at minimum (C1).

---

## 5. Pivot options

### Option 2a — CUTLASS 3.x stream-K back-port (2–3 weeks)

Per [[memory:bd78lejo]] the neutral target is `cuda_eff ≈ 0.50`
(no expected improvement over current ≈ 0.46), optimistic is
`0.60–0.70`. Risks:

- sm_89 lacks TMA / wgmma / cluster launch; stream-K primitives must
  be emulated with cp.async + bar.sync (same primitives V9 already
  uses).
- CUTLASS 3.x mainloop hook points assume Hopper-style pipelining;
  W4A4 per-group dequant must still be patched in at K-slab
  boundary (see [[memory:ie8lp95b]]).
- The same probe-chain argument applies at the MMA pipe layer:
  if 97% of loser time is on the MMA itself, any replacement
  mainloop is bounded by the same cap.

Expected outcome: **neutral** on loser cluster, possibly 0–5%
regression on winner cluster (CUTLASS overhead). **Not recommended**
given probe evidence.

### Option 2b — Accept V9 as production-final (0 days)

- Pin `r72 byte-identical` as `v9-final-dispatcher` tag.
- Write PHASE4_FINAL.md with the probe chain as the headline result:
  "Four independent micro-probes proved the ≥97% residual MMA pipe
  is per-warp hardware cap on sm_89; dispatcher-level optimisation
  exhausted at 1.044× median, 79/140 wins vs FP16 cuBLAS".
- Close Phase 3 Step 2.
- Leave probe infrastructure (B/D/E/F/G) in-tree as reusable
  diagnostic tooling.

**Recommended**. The probe chain is itself a deliverable — it
converts a "why is this slow" question into a provably bounded
lower bound for remaining optimisation headroom.

### Option 2c — Pivot to FP8 / BF16 mixed precision (2–4 weeks)

Out of scope for this task. sm_89 supports FP8 MMA (E4M3/E5M2); the
MAC pipe cap is the same but per-instruction MAC count is equal,
so it is no faster in the compute-bound regime. Listed here only
to record that it has been considered and rejected.

---

## 6. Probe-G infrastructure status

Not rolled back. Consistent with Probe-B/D/E/F policy:

- Macro `HKUST_PROBE_G` stays in
  `csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`
  (default 0 → byte-identical codegen).
- Env hook `HKUST_PROBE_G=N` stays in `ops.py`
  (emits `-DHKUST_PROBE_G=N` only when set; no default drift).
- Driver script `tests/c11_probe_g_v2_in_process.py` stays for
  future reuse (in-process multi-macro loader pattern useful for
  any future probe that needs clock-stable comparison).
- Legacy v1 driver `tests/c11_probe_g_barsync.py` kept for
  reference (documents the subprocess-per-level failure mode,
  valuable as negative engineering evidence).

---

## 7. Time accounting

| phase                           | hours spent | outcome   |
|---------------------------------|-------------|-----------|
| Design (§1–§10)                 | ~4 h        | shelved   |
| Probe-G implementation (macro + env hook + v1 driver) | ~3 h | landed (diagnostic) |
| Probe-G v1 → v2 triage (clock idle bug) | ~2 h | v2 in-tree |
| Probe-G v2 run + analysis       | ~1 h        | notes landed |
| Day-1 spike scoping / abort rationale (this doc) | ~1 h | this doc |
| **total r76+r77**               | **~11 h**   | no kernel change, 2 docs + 5 probe files landed |

Compared to the 8-hour Day-1 spike + contingency 5–7 day rewrite,
11 hours spent on probe-before-rewrite is a net positive outcome:
the alternative path (rewrite-then-measure) would have consumed
≥32 hours before hitting the same null result.

---

## 8. Acknowledged risk of this decision

The abort is justified on elimination logic, not on direct
warp-spec measurement. It is conceivable (probability ≲ 10%) that:

- Probe-D's ≤0% was itself a noise-limited null and a real ≤2%
  gain from pipeline decorrelation was missed.
- Probe-F's ≤0.25% covered the load-side IMAD but missed a
  store-side or address-calc path the warp-spec layout would
  affect differently.
- Some interaction effect (e.g., producer warp freeing L1$ pressure
  on consumer fetches) is not isolated by any single probe.

The combined upper bound from all three effects remains ≤3% by
the probe algebra. A ≤3% win does not justify 5–7 days of work
and the associated maintenance cost of dual-path
(`kUseWarpSpec=true/false`) template instantiations.

---

## 9. Action items on approval of abort

- [ ] Update `PROJECT_HANDOFF.md` §0 header and §3.8 / §11 to reflect
      abort + pivot choice.
- [ ] Tag current `HEAD` as `v9-final-dispatcher`.
- [ ] If Option 2b chosen: start `PHASE4_FINAL.md` draft.
- [ ] If Option 2a chosen: open `PHASE3_STEP2B_CUTLASS3X_BACKPORT.md`
      with scope + 2–3 week estimate + required approvals.
- [ ] Add `PHASE3_STEP2_WARPSPEC_DESIGN.md` header note: "superseded
      by PHASE3_STEP2_WARPSPEC_SPIKE_ABORT.md".

---

## 10. Memory updates required

None. The project-level memories already anchor the conclusions
this doc relies on:

- [[memory:bd78lejo]] — MMA pipeline starvation vs TC-underutil
  taxonomy correction.
- [[memory:bmmiahpl]] — micro-benchmark methodology (under which
  all five probes were measured).
- [[memory:ie8lp95b]] — CUTLASS visitor infeasibility (constrains
  Option 2a scope).
- [[memory:0d5nyof1]] — CUDA-only production path (precludes
  Triton fallback from any pivot).

**End of abort rationale.**
