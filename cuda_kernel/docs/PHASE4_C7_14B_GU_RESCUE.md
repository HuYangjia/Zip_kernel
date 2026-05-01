# C.7 Dispatcher Rule — Qwen3-14B gate_up Family Rescue

**Status**: Shipped 2026-05-01.  Validated on 10 shapes (4 target + 6 guard).

## TL;DR

Added a fourth region (D) to the `split_k=2` dispatcher rule, specifically
targeting the Qwen3-14B gate_up_proj shape family (d_in=5120, d_out=34816)
at T >= 2048.  This rescues **4 loser shapes from 0.76-0.91× → 1.04-1.17×**
without touching any other shape's behaviour.

## Problem — what C.6 v2 missed

When C.6 v2 shipped (2026-05-01) it carved out three `split_k=2` regions
covering 14B/32B kv, deep-K dn, and narrow-d_out kv/dn.  Its code
comment explicitly called out a **known miss**:

> Qwen3-14B gu T=512 (5120→34816, n_g=40) would have given -19.75%
> but lies in the gap between regions A (d_out too big) and B (d_in
> too small).  Adding it would require a hard-coded special case;
> not worth the fragility given the shape is an 8-shape-only loser.

At the time C.6 v2's author judged the 8-shape-only loser as
"not worth the fragility" because the optimisation work was focused
on T=512.  The r68 prefill survey (Ts={1024, 2048, 4096, 8192})
showed that this "miss" compounds across every prefill T:

| Shape           | r68 speedup vs fp16 |
|---|---:|
| 14B gu T=512    | 0.90× |
| 14B gu T=1024   | 0.78× |
| 14B gu T=2048   | 0.76× |
| 14B gu T=4096   | 0.77× |
| 14B gu T=8192   | 0.83× |

14B is a production-critical model size in LLM serving.  These five
shapes span the entire prefill-T regime; leaving them at 0.77-0.90×
blocks the "all W4A4 projections >= 1.0× of FP16" quality bar.

## Root cause of the original loss

Qwen3-14B gu at T=2048 has:
  - d_in=5120 → n_groups=40 (moderate depth)
  - d_out=34816 → n_cta_m=272 (very wide)
  - grid = 272 × 32 = 8704 CTAs (compute-bound, cuda_eff=22%)

The kernel mainloop is MMA-starved (B1 HFMA2 dequantise dominates the
warp scheduler, per `phase2_tc_rediagnosis.md`).  Forcing `split_k=2`
halves the per-CTA K-loop length, reducing the B1 serial chain and
letting the MMA pipeline re-saturate.  This is bounded: at larger
d_out (32B gu, 55296) or different d_in (8B gu, 4096), the grid is
either too saturated or the K-loop too short to benefit.

## Design of region D

After a 6-knob × 20-shape smart-pruned β-scan
(`tests/phase_r_beta_slim.py`), only **one knob** produced loser
wins without winner regressions: plain `splitk=2`, applied narrowly.

The β-scan data:

| Shape            | default sk=1 | sk=2     | Δ        | Winner/Loser |
|---|---:|---:|---:|---|
| 14B gu T=2048    | 5862us       | **4298** | **-26.7%** | loser → 1.04× ✓ |
| 14B gu T=4096    | 11613us      | **8557** | **-26.3%** | loser → 1.07× ✓ |
| 14B gu T=8192    | 23089us      | **16997**| **-26.4%** | loser → 1.17× ✓ |
| 32B gu T=2048    | 10180us      | 10627    | +4.4%      | loser, excluded (d_out=55296 > 44000) |
| 32B gu T=4096    | 20303us      | 21102    | +3.9%      | loser, excluded |
| 70B gu T=2048    | 17097us      | 19184    | +12.2%     | loser, excluded (d_in=8192) |
| **8B gu T=2048** | 1688us       | 2509     | **+48.7%** | **winner**, MUST exclude (d_in=4096) |
| 8B gu T=4096     | 3425us       | 4924     | +43.8%     | winner, excluded |
| 4B q T=2048      | 206us        | 312      | +51.7%     | winner, excluded (d_in=2560) |

**The rule must therefore be tight on three dimensions**:
- `d_out >= 32768` — excludes 14B q/kv/o (d_out ≤ 5120)
- `d_out <= 44000` — excludes 32B gu (d_out=55296) and 70B gu (d_out=57344)
- `d_in == 5120` — excludes 8B gu (d_in=4096) and 70B gu (d_in=8192)

## Code

File: `cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`

```cpp
if (split_k == 1 && T >= 256 && n_groups >= 32 && (n_groups % 2) == 0) {
    const bool c6_region_A = (d_out <= 5120 && d_in >= 5120 && d_in <= 8192);
    const bool c6_region_B = (d_in >= 16384);
    const bool c6_region_C = (d_out <= 2048 && d_in >= 6144);
    const bool c7_region_D =                                          // NEW
        (d_in == 5120 && d_out >= 32768 && d_out <= 44000);          // NEW
    if (c6_region_A || c6_region_B || c6_region_C || c7_region_D) {
        split_k = 2;
    }
}
```

## A/B validation

`tests/c7_validate.py` strictly measures (warmup=500, outer=10,
inner=200, per `[[memory:bmmiahpl]]`) with default / force-sk=1 / force-sk=2
env overrides and with L2-flush-calibrated FP16 baseline.

### TARGET shapes — all 4 rescued ✓

| Shape            | FP16 (us) | default | force sk=1 | force sk=2 | default_sp | sk1_sp  | sk2_sp  |
|---|---:|---:|---:|---:|---:|---:|---:|
| 14B gu T=512     | 1288      | 1192    | 1414       | 1192       | **1.081×** | 0.911×  | 1.081×  |
| 14B gu T=2048    | 4456      | 4280    | 5865       | 4331       | **1.041×** | 0.760×  | 1.029×  |
| 14B gu T=4096    | 9142      | 8538    | 11380      | 8551       | **1.071×** | 0.803×  | 1.069×  |
| 14B gu T=8192    | 19953     | 16997   | 23188      | 17009      | **1.174×** | 0.861×  | 1.173×  |

For every target, `default ≈ force sk=2`, confirming C.7 is firing.
T=512 is a bonus — β-scan missed it, but C.7 gate `T>=256` catches it.

### GUARD shapes — none disturbed ✓

| Shape                     | default | sk=1  | sk=2  | Behaviour              |
|---|---:|---:|---:|---|
| 32B gu T=2048 (excl.)     | 10102   | 10106 | 10640 | default = sk=1 ✓       |
| 70B gu T=2048 (excl.)     | 17145   | 17145 | 19420 | default = sk=1 ✓       |
| **8B gu T=2048** (excl.)  | 1711    | 1718  | 2527  | default = sk=1 ✓ (winner safe!) |
| **4B q T=2048** (excl.)   | 209     | 211   | 313   | default = sk=1 ✓ (winner safe!) |
| 14B q T=2048 (C.6v2)      | 666     | 499   | 668   | default = sk=2 (C.6v2 region A, pre-existing) |
| 32B dn T=2048 (C.6v2)     | 3543    | 3491  | 3547  | sk=1 ≈ sk=2 (C.6v2 region B, neutral) |

## Expected impact on full bench (r69 pending)

| Shape           | r68 sp | expected r69 sp | delta |
|---|---:|---:|---:|
| 14B gu T=512    | 0.90× | ~1.08×          | **+0.18** |
| 14B gu T=1024   | 0.78× | ~1.03×          | **+0.25** |
| 14B gu T=2048   | 0.76× | ~1.04×          | **+0.28** |
| 14B gu T=4096   | 0.77× | ~1.07×          | **+0.30** |
| 14B gu T=8192   | 0.83× | ~1.17×          | **+0.34** |
| All other shapes| unchanged | unchanged    | 0         |

Absolute time savings (per forward pass of a single 14B gu layer):
- T=2048: ~1700us/call saved
- T=4096: ~3400us/call saved
- T=8192: ~6900us/call saved

Across Qwen3-14B's 40 layers and full prefill, this is a notable
serving-latency improvement on the most popular production model size.

## Orthogonal finding — bench methodology audit

During C.7 validation we discovered that `bench_qwen3_shapes`'s FP16
L2-flush calibration under-subtracts the flush cost (`warmup=50`
isn't enough to cold-cache the 96MB flush tensor itself), causing
FP16 baseline to measure **6-7% artificially low** on tight-kernel
shapes like kv_proj.  See `tests/bench_methodology_audit.py` for the
raw data.  Impact: shapes reported at 0.93-0.97× in bench_qwen3
are actually 1.00-1.04× in strict probes.  Not fixed in this commit
(out of scope) but documented for future bench tool hardening.

## Known limitation (intentional)

C.7 does NOT help:
- 32B gu T >= 2048 (0.72× → 0.83× via sk=4, but still < 1.0×)
- 70B gu T >= 2048 (0.70× floor, all knobs useless)
- 70B kv T >= 1024 (0.91× floor, default already optimal)

These three shape families have cuda_eff stuck at 20-35% due to
sm_89-level MMA pipeline starvation and cannot be fixed by
dispatcher-level tuning alone.  They need kernel-level surgery
(warp-specialisation or epilogue register-reuse), which is out of
scope for C.7.

## Files changed

- `cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`
  (+18 lines of comment + 2-line region D condition)
- `cuda_kernel/tests/phase_r_beta_slim.py` (new — β-scan driver)
- `cuda_kernel/tests/c7_validate.py` (new — A/B validator)
- `cuda_kernel/tests/bench_methodology_audit.py` (new — orthogonal finding)
- `cuda_kernel/docs/PHASE4_C7_14B_GU_RESCUE.md` (this file)
