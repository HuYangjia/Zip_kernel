#  `kernel/cuda_kernel/tests/` — Script Classification Index

**Freeze tag**: `v9-final-dispatcher` (2026-05-02)
**Authoritative summary**: [../docs/PHASE4_FINAL.md](../docs/PHASE4_FINAL.md)

This directory intentionally retains all exploratory, verification, and
one-shot probe scripts produced during Phases 2 / 3 / C / 4. Per repo
policy (failed-experiment retention), scripts are **not deleted** after
a negative / superseded verdict — they remain on disk as evidence, with
their disposition recorded below.

---

## 1. ACTIVE — Probe-chain drivers (retained as diagnostic tooling)

These six scripts are the **reusable methodology deliverable** of
Phase 4. Each is self-contained, prints a GREEN / NEGATIVE verdict,
and can be re-run after any kernel change to re-validate the
MMA-pipeline-starvation ≤ 3% residual bound (see
[../docs/PHASE4_FINAL.md §3](../docs/PHASE4_FINAL.md)).

| Script | Probe | Attacks | Verdict |
|---|---|---|---|
| `c11_probe_b_fold_skip.py` | B | fold FMA scalar ALU (post-MMA dequant) | ≤ 1.0% |
| `c11_probe_d_cpasync_waitoff.py` | D | `cp.async` wait-group serialisation | ≤ 0.0% |
| `c11_probe_e_sparse_off.py` | E | entire sparse branch (ldmatrix + BSR) | ≤ 1.4% |
| `c11_probe_f_smem_pack.py` | F | smem swizzle IMAD + scale/zero load | ≤ 0.25% |
| `c11_probe_g_barsync.py` | G (v1, subprocess) | `bar.sync` overhead at warp-spec density | retained as negative-engineering reference |
| `c11_probe_g_v2_in_process.py` | G (v2, in-process) | same, clock-stable pattern | ≤ 0.11% — **reusable template** |

Run the whole chain to reproduce the 97% MMA-issue-residency bound:

```bash
python tests/c11_probe_b_fold_skip.py
python tests/c11_probe_d_cpasync_waitoff.py
python tests/c11_probe_e_sparse_off.py
python tests/c11_probe_f_smem_pack.py
python tests/c11_probe_g_v2_in_process.py
```

---

## 2. ACTIVE — Parity / benchmark infrastructure

Shared utilities. Keep these up to date with any kernel ABI change.

| Script | Purpose |
|---|---|
| `__init__.py` | test package init |
| `test_parity.py` | numerical parity (CUDA kernel vs reference dequant-matmul) |
| `test_splitk_parity.py` | split-k variant parity check |
| `parity_fused_quant.py` | fused-quant path parity harness |
| `perf_fused_quant.py` | fused-quant perf micro-bench |
| `bench_methodology_audit.py` | timing-protocol audit ([[memory:bmmiahpl]] compliance check) |
| `bench_smallT_revisit.py` | small-T (T∈{1,8}) perf re-audit harness |

---

## 3. ARCHIVED — Phase C dispatcher sweep scripts (C.1 → C.8)

These scripts drove the r64–r68 dispatcher widening work. All
findings folded into `csrc/fused_dense_sparse/fused_quant_dense_sparse_mma_int4.cu::main_dispatcher`.
Summaries in [../docs/PHASE_C_FINAL.md](../docs/PHASE_C_FINAL.md).

| Script | Phase | Summary doc |
|---|---|---|
| `c1_group_cache_sweep.py` | C.1 group-cache T=128 gate widen | PHASE_C_FINAL.md |
| `c2_kbn_sweep.py`, `c2_kbn_sweep_subprocess.py`, `c2_kbn_sweep_trials.py` | C.2 kBn sweep (negative result) | PHASE_C_FINAL.md |
| `c3_gate_up_diagnose.py`, `c3_sanity_check.py` | C.3 Qwen3-14B gu T=128 kBm=64 fix | PHASE_C_FINAL.md |
| `c4_mid_T_sweep.py`, `c4_verify.py` | C.4 mid-T (T=48/64) calibration | PHASE_C_FINAL.md |
| `c5_verify.py`, `c5_verify_v2.py` | C.5 T=128 kBm=64 gate widen | ../docs/PHASE4_C5_DISPATCHER_WIDEN.md |
| `c6_verify.py`, `c6v2_verify.py` | C.6 / C.6-v2 deep-K split_k=2 | ../docs/PHASE4_C6_V2_SPLITK_REFINED.md |
| `c7_validate.py` | C.7 14B gu rescue | ../docs/PHASE4_C7_14B_GU_RESCUE.md |
| `c8_ab_verify.py`, `c8_quick_verify.py` | C.8 loser-shape opt | ../docs/PHASE4_C8_LOSER_SHAPES_OPTIMIZATION.md |

Status: **frozen; kept as reproducibility evidence**. Do not modify
without a new experiment ID.

---

## 4. ARCHIVED — One-shot probes and exploratory scripts

Kept on disk per failed-experiment retention policy.

| Script | Experiment | Outcome |
|---|---|---|
| `kv_p0_viability.py`, `kv_splitk_probe.py` | KV-cache / split-k viability for decode T=1 | shelved (see KV notes) |
| `p0_integration_bench.py` | P0 end-to-end dispatcher helper | **negative** — see [../docs/P0_INTEGRATION_NEGATIVE_RESULT.md](../docs/P0_INTEGRATION_NEGATIVE_RESULT.md) |
| `phase_r_beta_scan.py`, `phase_r_beta_slim.py`, `phase_r_probe.py` | Phase-R β scan (dispatcher tuning knob) | folded into dispatcher |
| `q0_lite_bench.py` | Q0-lite upper-bound measurement | see [../docs/PHASE4_Q0_LITE_UPPER_BOUND.md](../docs/PHASE4_Q0_LITE_UPPER_BOUND.md) |
| `t512_dispatch_probe.py`, `t512_dispatch_probe_fast.py`, `t512_probe_extended.py` | T=512 dispatcher probe | folded into C.6-v2 |
| `trace_kernel_dispatch.py` | ad-hoc kernel-dispatch trace | retained as debugging pattern |

Status: **archived-in-place**; correspond to negative / superseded
experiments. Do not re-run in CI; do not modify.

---

## 5. Output directories

| Directory | Producer | Notes |
|---|---|---|
| `sweep_out/` | c1/c2/c4 sweep scripts | JSON dumps; kept as reference data |

---

## 6. Conventions

- **Do not delete** scripts here, even after a negative verdict —
  move them to §3 / §4 in this index and update their disposition.
- **New probe scripts** must follow the Probe-chain pattern
  (default-off compile macro + single-hypothesis kill + strict
  in-process timing protocol, [[memory:bmmiahpl]]).
- **Timing-sensitive scripts** must import the shared timer from
  `kernel/cuda_kernel/tools/profile/_phase1_shapes.py::time_forward_us`
  (skeleton: `min-over-outer of mean-over-inner`).

---

## 7. Entry points

- For the **kernel**: see
  [../csrc/fused_dense_sparse/fused_quant_dense_sparse_mma_int4.cu](../csrc/fused_dense_sparse/fused_quant_dense_sparse_mma_int4.cu).
- For the **final benchmark numbers**: see
  [../docs/PHASE4_FINAL.md](../docs/PHASE4_FINAL.md).
- For the **probe-chain methodology**: see
  [../docs/PHASE4_FINAL.md §3](../docs/PHASE4_FINAL.md) and the
  `c11_probe_*` scripts in §1 above.
- For **project-level recovery from freeze**:
  [../../../PIN_v9-final-dispatcher.txt](../../../PIN_v9-final-dispatcher.txt)
  + [../../../PROJECT_HANDOFF.md](../../../PROJECT_HANDOFF.md).
