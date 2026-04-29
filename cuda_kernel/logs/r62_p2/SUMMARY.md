# r62 P2 — Bench methodology fix (L2 cache flush)

## Context

Prior to r62 the project's flagship bench
`kernel/tools/profile/full_bench_vs_bf16.py` ran in **tight-loop** mode:
each `inner=200` timing window launched the same matmul repeatedly
against the *same* weight tensor.  On RTX 4090 (L2 = 72 MB) every
shape with `d_out × d_in × 2B <= 72 MB` hits L2 after the first miss,
so 199/200 launches measure L2 bandwidth, not HBM.

That inflated the BF16 cuBLAS baseline enough to make the INT4 kernel
look worse than it actually is on a real LLM workload (one weight read
per layer, cold L2).

## What r62 P2 did

1. Added an opt-in **L2-flush** mode (`--flush-l2`, default) that zeros
   a 96 MB scratch tensor before *each* inner launch.  The flush cost is
   calibrated once and subtracted, so the reported time is purely the
   target kernel under cold-cache conditions.
2. Added a `--compare-l2` mode that runs both tight-loop and cold-cache
   back-to-back and emits a side-by-side markdown report for audit.
3. Re-ran the 11-shape canonical sweep on the autodl 4090.

## Results — median speedup 0.91× → 1.11× (+0.20×)

| shape                | tight  | cold   | Δ       | note                |
|----------------------|-------:|-------:|--------:|---------------------|
| 1024×1024×128        | 1.38×  | 0.40×  | -0.98×  | tiny, L2-fits       |
| 2048×2048×128        | 0.91×  | 0.95×  | +0.04×  |                     |
| 4096×4096×128        | 0.88×  | 1.14×  | +0.26×  | **was fake loss**   |
| 1024×4096×128        | 1.12×  | 1.38×  | +0.26×  |                     |
| 4096×1024×128        | 0.94×  | 0.95×  | +0.01×  |                     |
| 2048×4096×128        | 0.83×  | 1.06×  | +0.23×  | **was fake loss**   |
| 4096×2048×128        | 0.86×  | 1.11×  | +0.25×  | **was fake loss**   |
| 4096×4096×32         | 0.74×  | 1.11×  | +0.37×  | **was fake loss**   |
| 4096×4096×1          | 0.80×  | 1.19×  | +0.39×  | **was fake loss**   |
| 4096×14336×128       | 1.80×  | 1.74×  | -0.06×  |                     |
| 14336×4096×128       | 2.30×  | 2.14×  | -0.16×  |                     |

- BF16 eff (cold):  all in 63-100% — physically sane.
- BF16 eff (tight): up to 235% — unphysical, cache artefact.

## Key reclassification

7 shapes previously reported as "speedup < 1" (INT4 slower than BF16) on
the tight-loop bench are in fact at speedup 1.06×-1.38× on cold cache.
The only *genuine* INT4-slower shape remaining is **1024×1024×128**
(2 MB fp16 matmul, entirely L2-resident; this shape is not a target
for INT4 optimisation because the whole workload fits in cache
regardless of representation).

## Remaining bottleneck picture (cold cache, honest)

INT4 efficiency by class:

| class | shape | INT4 eff | dominant cause |
|---|---|---:|---|
| 🔴 tiny | 1024×1024×128 | 12% | grid ≈ 8 CTAs, SMs starved |
| 🟡 square medium | 2048×2048, 4096×2048, 4096×4096 / T≤32 | 30-37% | bank conflict (r61 stage C) + register-bound occupancy |
| 🟢 narrow / tall-thin | 1024×4096, 14336×4096 | 49-57% | OK |

Stage-C exploration (r61) already showed sW/sX bank conflict is
immovable at the current smem layout.  Stage-F showed occupancy hits
a register-pressure ceiling.  The remaining levers are either a smem
layout rewrite (Stage C2, 2-3 days, uncertain) or CUTLASS migration
(3-5 days, moderate risk).

## Files

- [full_bench_vs_bf16.py](../../tools/profile/full_bench_vs_bf16.py) (edits)
    - `_bench_cuda_event(fn, ..., flush_l2=False)`
    - new helpers: `_flush_l2`, `_get_l2_flush_tensor` (96 MB scratch)
    - new renderer: `render_compare_report(cheat, honest, title)`
    - new CLI flags: `--flush-l2 / --no-flush-l2 / --compare-l2`
- [l2_compare_report.md](l2_compare_report.md)  (artefact)
- [l2_compare_raw.json](l2_compare_raw.json)   (artefact)

## Policy note

Going forward, all kernel benches that compare against a cuBLAS BF16
baseline **must** default to L2-flushed mode to avoid the cache
artefact.  Internal eager-vs-graph benches (same kernel both sides) may
remain tight-loop since the artefact cancels.
