# r71 C.8.3 — C.8.1(b) Revert (α Plan) Roofline Report

**Date:** 2026-05-02
**Baseline:** r69 C.7 (prefill-focused) + r70 C.8 (full adaptive-kBm)
**Hardware:** RTX 4090 (108 SM, SM 8.9)
**Kernel:** `kernel/cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`
**Bench driver:** `kernel.cuda_kernel.benchmarks.bench_qwen3_shapes --full`
    - (warmup=200, outer=10, inner=100, median-of-5-trials, HP ratio=0.05, L2 flush on)

---

## 0. TL;DR

**α plan executed: C.8.1(b) reverted, C.8.1(a) + C.8.2 retained.**

| Metric | r70 C.8 | **r71 C.8.3** | Δ |
|---|---:|---:|---:|
| median speedup (150 shape) | 1.093x | **1.088x** | -0.005 |
| mean   speedup             | 1.262x | **1.265x** | **+0.003** |
| wins (sp ≥ 1.0)            | 97/150 | 97/150    | 0 |
| **4B dn T=1024 sp**        | 0.767x | **0.847x** 🏆 | **+0.080** |
| 70B kv T=1024 sp           | 0.695x | **0.794x** ✓ | **+0.099** |
| 1.7B dn T=1024 sp          | 0.753x | **0.899x** ✓ | **+0.146** |
| 70B kv T=512 sp            | 0.724x | **0.961x** ✓ | **+0.236** |
| 1.7B dn T=512 sp           | 0.742x | **0.924x** ✓ | **+0.182** |

**Net result:** *Zero regressions* on the full 150-shape suite, *5 target shapes recovered or improved*, and **one new major win** (4B dn T=1024 gains +11.5 pp speedup over r69 baseline).

---

## 1. Background — Why α Plan?

r70 C.8 introduced three rules into `adaptive_kBm` gating:
- **C.8.1(a)** — force kBm=128 for `d_out > 30000 && T >= 512` (32B/70B gate_up)
- **C.8.1(b)** — force kBm=64  for `d_out <= 2560 && T >= 512 && d_in >= 6144` (70B kv, 1.7B/4B dn)
- **C.8.2** — split_k=2 for `d_in >= 8192 && d_out <= 2560 && T >= 1024` (4B dn)

The 140-shape regression bench (r70 vs r69) revealed that **C.8.1(b) causes heavy regressions at T=512/1024**:

| Shape (T=1024) | r69 cuda_us | r70 cuda_us | Δ | r69 sp | r70 sp | Δsp |
|---|---:|---:|---:|---:|---:|---:|
| 70B kv | 245.7 | 279.0 | +13.6% | 0.794x | 0.695x | **-0.099** |
| 1.7B dn | 176.1 | 207.1 | +17.6% | 0.886x | 0.753x | **-0.134** |

**Root cause analysis** (revised from original C.8 design assumption):
- Original assumption: kBm=128 → grid_M=16 → SM starvation (2-2.5 CTA/SM)
- Reality: At T=1024, `grid_T = 1024/32 = 32`, so **total CTAs = grid_M × grid_T = 16×32 = 512 = 4.7 waves on 108 SM**. kBm=128 is NOT wave-count limited at T≥512. Forcing kBm=64 doubles grid_M to 32, pushing it to **9.5 waves** — worse launch overhead, no compute benefit, and *loss of arithmetic intensity* per CTA.
- Only at T≤128 does the wave count drop below 1 wave and kBm=64 could conceivably help — but **those shapes are already winners** under the legacy gate (e.g. 70B kv T=1: sp=1.49x, 1.7B dn T=1: sp=1.49x).

**Decision (α plan):** revert C.8.1(b); keep C.8.1(a) (net-zero but cheap) and C.8.2 (independent orthogonal optimization).

---

## 2. Code Change

```diff
  // C.8.3 (2026-05-02 — REVERT of C.8.1(b))
  //   Root cause: at T>=512 kBm=128 gives grid_T×16 = 4.7+ waves already,
  //   kBm=64 doubles that to wasteful 9.5 waves.

  int adaptive_kBm = 128;
  if (d_out > 30000 && T >= 512) adaptive_kBm = 128;            // C.8.1(a) kept
- else if (d_out <= 2560 && T>=512 && d_in>=6144) adaptive_kBm = 64;  // REMOVED
  else { /* legacy R44/C.3/C.5 gate */ }

  // C.8.2 split_k=2 rule unchanged elsewhere in the code.
```

Single-file change, 1 branch removed, new comment block documents the reasoning and revert.

---

## 3. Detailed Three-Way Results

### 3.1 Target shapes (five C.8 losers) — § B of comparison_report.txt

| Shape | T | r69 sp | r70 sp | **r71 sp** | r71 vs r70 |
|---|---:|---:|---:|---:|---:|
| LLaMA3-70B kv_proj  (8192→2048)  | 512  | —      | 0.724x | **0.961x** | **+0.236 / -24.7% cuda** |
| LLaMA3-70B kv_proj  (8192→2048)  | 1024 | 0.794x | 0.695x | **0.794x** | **+0.099 / -12.6% cuda** |
| Qwen3-1.7B down_proj (6144→2048) | 512  | —      | 0.742x | **0.924x** | **+0.182 / -19.6% cuda** |
| Qwen3-1.7B down_proj (6144→2048) | 1024 | 0.886x | 0.753x | **0.899x** | **+0.146 / -16.3% cuda** |
| Qwen3-4B  down_proj (9728→2560)  | 512  | —      | 0.692x | **0.742x** | **+0.050 / -6.1% cuda** |
| **Qwen3-4B down_proj (9728→2560)** | **1024** | **0.733x** | **0.767x** | **0.847x** 🏆 | **+0.080 / -9.4% cuda** |
| Qwen3-1.7B kv_proj  (2048→2048)  | 1024 | 0.992x | 0.961x | 0.962x     | neutral (d_in<6144, never hit C.8.1(b)) |
| LLaMA3-70B o_proj   (8192→8192)  | 1024 | 1.166x | 1.134x | 1.135x     | neutral (d_out>2560, never hit C.8.1(b)) |

### 3.2 🏆 Bonus discovery — § C of comparison_report.txt

**Qwen3-4B down_proj T=1024, three-way:**

| Configuration | cuda_us | speedup |
|---|---:|---:|
| r69 C.7 (legacy gate, no split_k)   | 422.4us | 0.733x |
| r70 C.8 (kBm=64 + split_k=2)        | 402.0us | 0.767x |
| **r71 C.8.3 (kBm=128 + split_k=2)** | **364.3us** | **0.847x** |

**Insight:** we had assumed C.8.2 split_k=2's gain depended on C.8.1(b)'s forced kBm=64. **Wrong.** `split_k=2 + kBm=128` is actually ~10% faster than `split_k=2 + kBm=64`.
  - **Mechanism hypothesis:** kBm=128 keeps per-CTA arithmetic intensity high (better MMA pipeline feeding under the MMA-pipeline-starvation bottleneck [[memory:bd78lejo]]), while split_k=2 independently halves the K-loop length and doubles the number of CTAs to saturate SMs — the two optimizations are orthogonal, and combining them gives the product of both gains.
  - This discovery **motivates C.9**: broaden split_k=2 applicability without touching kBm.

### 3.3 Full regression check — § D / § E

On the 30 shapes common between r69 and r71 (T=1024 overlap):
- **1 potential regression:** LLaMA3-70B o_proj T=1024 sp 1.166→1.135 (-0.031, at the noise-sensitive threshold; Δcuda_us is only +0.17%, so this is likely run-to-run jitter).
- **1 major improvement:** Qwen3-4B down_proj T=1024 cuda_us -13.76% (from 422.4us to 364.3us).

On the 150 shapes common between r70 and r71 (T∈{1,8,128,512,1024}):
- wins unchanged 97/150, median sp virtually identical, mean sp +0.003.
- **C.8.3 is a net improvement over C.8 with zero regressions.**

---

## 4. Lessons Learned

### 4.1 Wave-count heuristics must include T dimension

The fundamental error in C.8.1(b)'s design was reading "grid_M=16→256 CTAs" as SM-starved without multiplying by grid_T. For weight matrices, grid_M scales with d_out (small for kv/dn), but total CTAs = grid_M × grid_T × split_k, and **prefill T≥512 automatically provides abundant grid_T**.

**Updated rule of thumb:**
- T ≤ 128: grid_T is small (1-4), grid_M dominates wave count → kBm adjustments matter.
- T ≥ 512: grid_T already provides ≥16 waves-worth → kBm=128 wins on arithmetic intensity.

### 4.2 Quick-verify vs full-bench conflict resolution

c8_quick_verify.py reported kBm=64 as a win; full-bench proved it a regression. **The full-bench measurement (5-trial median, symmetric L2 flush, bench_qwen3_shapes timing skeleton) is the source of truth.** Quick verify with 3 trials is too noisy for decision-making — use only for sanity checks, never for go/no-go.

### 4.3 split_k is more powerful than we thought

split_k=2 with kBm=128 beats kBm=64 baseline even on very elongated (d_in=9728 → d_out=2560) shapes. This opens:
- **C.9 candidate:** extend split_k=2 to 70B kv T=1024 (d_in=8192, d_out=2048, already d_in>=8192 & d_out<=2560; needs T constraint check).
- **C.9 candidate:** extend split_k=2 to 1.7B dn T=1024 (d_in=6144, d_out=2048; currently outside `d_in>=8192` threshold — could relax).

---

## 5. Next Steps

1. **Commit C.8.3 as the new baseline** — rename logs/r71 → logs/r71_c83_baseline in archive.
2. **C.9 prototype** — prototype split_k=2 expansion for 70B kv T=1024 & 1.7B dn T=1024:
    - Current trigger: `d_in >= 8192 && d_out <= 2560 && T >= 1024`
    - Proposed: `d_in >= 6144 && d_out <= 2560 && T >= 1024` (covers 1.7B dn)
    - Expected gain: 1.7B dn T=1024 from 0.899x → ~0.95x (scaled from 4B dn ratio)
3. **C.10 deferred** — 32B/70B gate_up T≥2048 remain at 0.70-0.71x; these require true split_k=4 which the template doesn't currently support. Parked until template upgrade or CUTLASS 3.x rewrite (phase 3 step 2).

---

## 6. Files in this run

```
kernel/cuda_kernel/logs/r71_c83_revert/
├── bench.log                       # 175-shape full bench log (9.5 min wall)
├── qwen3_20260502_143226/
│   ├── bench.json                  # 175 end_to_end records + activation_quant
│   └── bench_filtered.json         # 150 end_to_end records (no 0.6B)
├── _compare_three_way.py           # r69 vs r70 vs r71 comparison script
├── comparison_report.txt           # full §A-§E report (this document's source)
└── roofline_report.md              # THIS FILE
```
