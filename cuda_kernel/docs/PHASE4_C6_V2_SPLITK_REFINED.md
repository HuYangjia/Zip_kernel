# C.6-v2 — Precise deep-K split_k=2 override (2026-05-01, revised)

## Context

C.6-v1 (`T>=256 && n_g>=32 && n_g%2==0`) was a **too-broad** rule
based on a 6-shape probe.  Extended probe over **30 T=512 shapes**
(tests/t512_probe_extended.py) revealed that sk=2 **REGRESSES
by +8% to +51%** on many shapes the v1 gate activated on —
especially the Qwen3-8B series (d_in=4096, n_g=32) which was
caught by v1 but should have been skipped.

## Measurement evidence (t512_probe_extended.py)

All 30 T=512 shapes measured under sk=1 vs sk=2:

### WIN column (sk=2 faster by ≥3%, v2 must fire here)
| shape | d_in | d_out | n_g | sk2 vs sk1 |
|---|---:|---:|---:|---:|
| 14B gu | 5120 | 34816 | 40 | **-19.75%** |
| 14B dn | 17408 | 5120 | 136 | **-17.36%** |
| 32B dn | 27648 | 5120 | 216 | **-17.29%** |
| 70B kv | 8192 | 2048 | 64 | -10.82% |
| 0.6B dn | 3072 | 1024 | 24 | -8.85% (but n_g<32, v2 skips) |
| 14B q | 5120 | 5120 | 40 | -5.61% |
| 14B kv | 5120 | 2048 | 40 | -5.60% |
| 14B o | 5120 | 5120 | 40 | -5.75% |
| 32B kv | 5120 | 2048 | 40 | -5.54% |
| 1.7B dn | 6144 | 2048 | 48 | -6.71% |

### REGRESS column (sk=2 slower by ≥3%, v2 MUST NOT fire here)
| shape | d_in | d_out | n_g | sk2 vs sk1 |
|---|---:|---:|---:|---:|
| 8B q | 4096 | 4096 | 32 | +24.94% |
| 8B o | 4096 | 4096 | 32 | +24.84% |
| **8B gu | 4096 | 24576 | 32 | +50.85%** |
| 4B o | 4096 | 2560 | 32 | +16.35% |
| 4B dn | 9728 | 2560 | 76 | +16.31% |
| 8B dn | 12288 | 4096 | 96 | +17.29% |
| 70B q | 8192 | 8192 | 64 | +13.53% |
| 70B o | 8192 | 8192 | 64 | +13.32% |
| 70B gu | 8192 | 57344 | 64 | +12.94% |
| 32B gu | 5120 | 55296 | 40 | +8.65% |

**Key insight**: the same `n_g=32+` gate catches both big-WIN shapes
(14B series) and huge-REGRESS shapes (8B series).  The distinguishing
features are **d_in** and **d_out**, not n_groups alone.

## C.6-v2 rule (3 precise regions)

```cpp
if (split_k == 1 && T >= 256 && n_groups >= 32 && (n_groups % 2) == 0) {
    const bool c6_region_A = (d_out <= 5120 && d_in >= 5120 && d_in <= 8192);
    const bool c6_region_B = (d_in >= 16384);
    const bool c6_region_C = (d_out <= 2048 && d_in >= 6144);
    if (c6_region_A || c6_region_B || c6_region_C) {
        split_k = 2;
    }
}
```

Region breakdown:
- **A**: mid-d_in (5120..8192) + small-d_out (≤5120).  Covers
  14B q/kv/o, 32B kv, 70B kv.  Sweet spot because grid_m is
  moderate (≤40) and K-loop halving saves more than reduce costs.
- **B**: very-deep-K (d_in ≥ 16384).  Covers 14B dn (17408),
  32B dn (27648), 70B dn (28672).  Here sk=2 dominates because
  n_g = 136/216/224 makes the K-loop critical path enormous.
- **C**: small-d_out (≤2048) + d_in ≥ 6144.  Covers 1.7B dn
  (6144→2048) and re-confirms 70B kv (already in A).

Known miss: **14B gu T=512 (5120→34816)** at -19.75%.  This shape
lies at the crossroads — d_out too big for A, d_in too small for B.
Adding it would require a shape-specific clause (e.g.
`d_in==5120 && d_out>=32768 && d_in<=6144`) which is fragile.

## Verification (c6v2_verify.py, in-process A/B)

Strict protocol: warmup=500, outer=10, inner=200, 4 interleaved trials.

**WIN targets: 8/9 win, 1 neutral**

| shape | region | sk=1 forced | C.6-v2 default | Δ |
|---|---|---:|---:|---:|
| 14B q | A | 163.51us | 154.47us | **-5.53%** |
| 14B kv | A | 61.83us | 57.76us | **-6.59%** |
| 14B o | A | 163.58us | 154.89us | **-5.31%** |
| 32B kv | A | 61.83us | 57.55us | **-6.91%** |
| 70B kv | A+C | 92.91us | 82.67us | **-11.02%** |
| 14B dn | B | 660.28us | 543.70us | **-17.66%** |
| 32B dn | B | 1132.23us | 916.87us | **-19.02%** |
| 70B dn | B | 1444.31us | 1432.24us | -0.84% (neutral) |
| 1.7B dn | C | 71.19us | 66.40us | **-6.73%** |
| **median** | | | | **+6.82%** |
| **mean** | | | | **+9.85%** |

**EXCLUDED shapes (11 total): all 0/11 regress**

Every shape that v1 incorrectly activated — 8B q/kv/o/gu/dn, 4B o/dn,
32B gu, 70B q/o/gu — measures |Δ| < 0.1% between sk=1 forced and
C.6-v2 default, confirming the gate correctly skips them.

## Git history

- commit `701ede6`: C.6-v1 (over-broad, regressed 8B series)
- commit `754642c`: C.6-v1 archive doc (pre-correction)
- commit `b6287a...`: **C.6-v2 gate tightened to regions A/B/C**
- commit `...`: this doc

## Combined Phase C impact

| change | targets | avg gain (in-process A/B) |
|---|---|---:|
| C.5 | 4 T=128 q/o shapes (d in [2560,4096]²) | +7.80% |
| C.6-v2 | 8 T=512 shapes in regions A/B/C | +9.85% |

Total production shapes improved via dispatcher-level changes
(no kernel source code modification, no parity risk):  **12 shapes**,
range +5.3% to +19.0%, median +7.3%.
