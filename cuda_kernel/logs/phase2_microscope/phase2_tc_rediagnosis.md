---
title: Phase 2 Rediagnosis — `tc_underutil` is MMA Pipeline Starvation, Not TC-Off
date: 2026-04-28
supersedes_sections:
  - file: phase2_kernel_microscope_report.md
    section: "Global SASS finding"
  - file: phase2_kernel_microscope_report.md
    section: "Per-shape evidence / Primary bottleneck: tc_underutil"
audience: Phase 3 roadmap consumers (cluster_all_shapes.py / phase3_render_roadmap.py)
---

# TL;DR

The Phase-2 verdict **`tc_underutil`** was correct in its numeric SASS
signal but **wrong in its root-cause narrative**. The label is retained
for downstream compatibility, but its *meaning* is hereby re-stated:

| Old (wrong) meaning | New (correct) meaning |
|---|---|
| "Tensor Core not emitted — kernel runs on CUDA core FMA" | "Tensor Core **is** emitted, but the MMA pipeline starves because 50% of issue slots are spent on epilogue FMA, shared-memory swizzle address math (IMAD), and 2-stage async-copy staging." |
| 42/42 kernels have TC-fraction < 2% → no TC usage | 42/42 kernels have **IMMA issue-slot share < 2%**, but **MAC-weighted TC share = 99.7%**. Each `mma.m16n8k64.s4.s4.s32` does 8192 MACs; 64 IMMAs deliver 524288 MACs while 1408 CUDA-FMA deliver ≤ 2816 MACs. |

The roadmap's proposed fix (**CUTLASS 3.x INT4 rewrite**) is still the
right direction, but the expected efficiency win shrinks from
*"0 → has TC"* to *"TC that starves → TC that streams"*.

# Decisive evidence (SASS truth, 2026-04-28)

Source: `_sass/sass_profile.json` re-analysed with MAC-weighted share.

## Top-5 largest kernels (by instruction count)

| kernel (abbreviated) | total_insts | IMMA | CUDA_FMA | issue-slot TC% | **MAC-weighted TC share** |
|---|---:|---:|---:|---:|---:|
| `fused_dense_sparse_mma_int4<64,1,128>` | 5000 | 64 | 1408 | 1.28% | **99.46%** |
| `fused_dense_sparse_mma_int4<64,0,128>` | 5000 | 64 | 1408 | 1.28% | **99.46%** |
| `fused_dense_sparse_mma_int4<..32..>` | 3872 | 32 | — | 0.83% | ≥ 99% |
| `dense_gemm_mma_int4<64,...>` | 3616 | 32 | — | 0.88% | ≥ 99% |
| `sparse_gemm_mma_int4<64>` | 2488 | 32 | — | 1.29% | ≥ 99% |

Peak-throughput per instruction on sm_89:
- `mma.m16n8k64.s4.s4.s32` → **8192 MAC/inst**
- scalar `HFMA2` / `IMAD` → 2 / 1 MAC/inst
- ratio ≈ **4000 – 8000x per-inst MAC throughput**

So a count-weighted SASS histogram **cannot** tell you whether TC is
idle: one IMMA looks the same as one IMAD in a bar-chart, but it does
thousands of times more work.

## Why cuda_eff is still only 13 – 39 % if TC dominates MACs

Pipelines, not arithmetic. The 5000-instruction body breaks down as:

```
IMMA (tensor pipeline)        64  issues   ~16 cycle/issue → ≈  1024 cycle busy
CUDA_FMA (ALU pipeline)     1408  issues   ~1 cycle/issue  → ≈  1408 cycle busy
INT_ALU (address / mask)    1052  issues   ~1 cycle/issue  → ≈  1052 cycle busy
LDS / STS (shmem)            178  issues   ~4 cycle/issue  → ≈   712 cycle busy
LDG / STG                    116  issues   ~20+ cycle lat  → variable (hidden)
SYNC / BAR                   215  issues   ~waits          → tail-dominated
```

Tensor pipeline occupancy ≈ 1024 / (1024 + 1408 + 1052 + 712) ≈ **24 %**.

That matches the observed cuda_eff of 13 – 39 % far better than the old
"kernel runs on CUDA core FMA" story (which would predict << 10 %).

## The three real sub-bottlenecks inside `tc_underutil`

| Sub-bottleneck | Evidence | Rough share of wasted slots | Fix direction |
|---|---|---:|---|
| **B1. Epilogue dequant runs in-kernel, serialised per-MMA** | `CUDA_FMA = 1408 (28 % of insts)` dominated by `HFMA2` computing `y = (acc_s32 → fp16) * scale_w - scale_x * sum_x * zero_w`. One chain per output, register-blocked with the MMA producer. | 30 – 40 % | Vectorise epilogue (pack 4× HFMA2 per output row), or schedule via CUTLASS `Collective` epilogue operator; split-epilogue stage so it can overlap the *next* MMA tile. |
| **B2. Shared-memory swizzle address math on the critical path** | `INT_ALU = 1052 (21 %)` + `IMAD.MOV.U32` hidden inside CUDA_FMA bucket; these are the LDS/STS index generators. Today they run in-line between MMAs. | 20 – 30 % | Prefer `cp.async.ca.shared.global` that writes swizzled addresses directly without register-bounce, or precompute a stride table (one IMAD per tile, not per load). |
| **B3. Only 2-stage async copy; MMA waits on weight load** | `LDG = 52`, `STS = 98` — indicates double-buffering but not more. sm_89 supports 3 – 4 stage `cp.async` pipelines and warp-specialised producer/consumer that CUTLASS 3.x `CollectiveBuilder` emits for free. | 20 – 30 % | 3 – 4 stage pipeline; producer warp(s) drive LDG → STS, consumer warp(s) drive LDS → MMA with half-warp staggering so the tensor pipeline is never starved. |
| **B4. Micro-granularity MMA pack** (residual) | `IMMA = 64`, tile chosen for `m16n8k64` is fine; marginal residual after B1-B3 handled. | 5 – 15 % | Tile-shape sweep; last-mile, not first-mile. |

Sum of B1+B2+B3 ≈ **70 – 100 %** of the 76 % non-tensor-pipeline slot
budget. That is *exactly* the surface area CUTLASS 3.x `CollectiveBuilder`
exists to shrink, which is why the Phase-3 plan (Step 2 → CUTLASS
rewrite) is not invalidated by this rediagnosis.

# Impact on Phase-3 roadmap

| Step | Before | After | Delta |
|---|---|---|---|
| Step 1 (CUDA Graph, 16 `launch_sparse` shapes) | ROI 2.30, 3-4 days | **unchanged** | 0 |
| Step 2 (CUTLASS 3.x INT4 rewrite, 84 `tc_underutil` shapes) | Narrative: "0 → TC"; target `cuda_eff` = 0.60 | Narrative: "TC that starves → TC that streams"; target `cuda_eff` = 0.50 (pessimistic) / 0.60 (neutral) / 0.70 (optimistic with full warp-specialised pipeline) | Same technique, re-calibrated target band |
| Sub-bottleneck microbench (new) | N/A | **Optional 0.5-day** epilogue-strip A/B to quantify B1's share before committing to CUTLASS | +0.5d |

The **Step 1 → Step 2** ordering is preserved because Step 1 is
independently justified (Python-side launch tax, orthogonal to the
tensor pipeline story) and Step 1 also cleans the measurement floor on
which Step 2's improvement is evaluated.

# Action items (atomic, closed with PR-style check-offs)

- [x] **R1.** Rewrite this rediagnosis doc and link it from
  `phase2_kernel_microscope_report.md` as the canonical source.
- [x] **R2.** Extend `sass_analyze.py` to emit `mac_tc_share` per-kernel
  (MAC-weighted) and keep `tc_fraction` as "issue-slot share" with a
  doc-comment clarifying the distinction.
- [x] **R3.** Update the `verdicts()` logic: keep the label
  `tc_underutil` (stable taxonomy for downstream tooling) but gate it
  on `mac_tc_share < 0.5` *instead of* `tc_fraction < 0.05`. Under the
  new metric no W4A4 MMA kernel will trigger it for the wrong reason;
  it will trigger only if a kernel really lacks MMA (e.g. the
  `activation_quant` and `gemv_decode` variants, which is correct).
- [x] **R4.** Update human-readable labels in
  `phase3_render_roadmap.py::_bottleneck_label`: `tc_underutil` →
  `"MMA pipeline starvation"` (the text the user sees). The taxonomy
  key stays `tc_underutil` to preserve CSV/JSON compatibility.
- [x] **R5.** Update `phase3_render_roadmap.py::CLUSTER_PLANS[
  "tc_underutil"]` narrative & expected_eff_after (0.60 → 0.50/0.60/0.70
  band with note).
- [x] **R6.** Patch `phase2_kernel_microscope_report.md` with a visible
  "Rediagnosis 2026-04-28" banner pointing here.
- [x] **R7.** Re-run `phase3_render_roadmap.py` to regenerate the
  roadmap document with corrected narrative; commit outputs under a
  fresh timestamp.
- [x] **R8.** Three-way sync (local → autodl → repo) after all the above.
- [x] **R9.** Update long-term memory: remove "TC% < 2% → TC not emitted"
  shared-truth; replace with the MAC-weighted framing.

# Provenance

- Raw JSON: `cuda_kernel/logs/phase2_microscope/_sass/sass_profile.json`
- Reanalysis script (ephemeral, not checked in):
  - `/tmp/sass_truth.py` (issue-slot bucket counts)
  - `/tmp/sass_mac.py`   (MAC-weighted share)
- This narrative was reconstructed from the SASS counters *only* —
  it contains **no** timing-based claims and is therefore immune to the
  warmup-budget fragility that invalidated earlier epilogue/x_zero
  verdicts.
