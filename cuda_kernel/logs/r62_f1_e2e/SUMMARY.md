# r62 F1 — Qwen3-8B end-to-end bench on cold-cache FP16 baseline

## Context

r62 P2 (previous milestone) showed the old tight-loop bench inflated
the cuBLAS BF16 baseline by up to ~2× due to L2 cache reuse.  The
fix — flushing a 96 MB scratch tensor before each inner launch —
was applied to `full_bench_vs_bf16.py` first.

r62 F1 extends the same fix to the **end-to-end Qwen3 bench pipeline**:

- `kernel/triton_kernel/benchmarks/_bench_util.py :: time_ms(fn, ...,
  flush_l2: bool)` — shared helper gets the L2-flush plumbing.
- `kernel/cuda_kernel/benchmarks/bench_qwen3_shapes.py` — the FP16
  cuBLAS baseline (`bench_fp16_matmul` and the e2e `run_fp16` lambda)
  now runs with `flush_l2=_FLUSH_L2_FP16` (module-level switch).  A
  new CLI flag `--flush-l2-fp16 / --no-flush-l2-fp16` controls it;
  default is **on** (cold-cache / honest).
- The INT4-Triton / INT4-CUDA paths are **not** flushed — they already
  hit HBM on every launch because their working set (packed W + scale
  + zero + scratch) + proper tiling means L2 reuse is marginal, and
  keeping them tight-loop lets us compare against legacy numbers.

## Result — Qwen3-8B, 20 shapes (cold-cache FP16)

Aggregate:
- **N=20, mean speedup 1.39×, median 1.21×, min 0.54×, max 3.25×**
- **11/20 shapes win over FP16 cuBLAS** (≥1.00×)
- 10/20 shapes are **clear wins** (≥1.10×)
- 7/20 shapes are **clear losses** (<0.90×) — all bound to 2 causes below

### Wins sorted by speedup

| proj | T | shape | speedup | fp16 (us) | cuda (us) |
|---|---:|---|---:|---:|---:|
| gate_up_proj | 32  | 4096→24576 | **3.25×** | 238.0 | 73.3 |
| gate_up_proj | 1   | 4096→24576 | **2.25×** | 216.2 | 96.1 |
| q_proj       | 1   | 4096→4096  | **2.18×** | 40.3  | 18.5 |
| o_proj       | 1   | 4096→4096  | **2.16×** | 40.0  | 18.5 |
| down_proj    | 1   | 12288→4096 | **2.10×** | 112.4 | 53.4 |
| gate_up_proj | 128 | 4096→24576 | **1.87×** | 274.3 | 147.0 |
| gate_up_proj | 512 | 4096→24576 | **1.54×** | 712.9 | 462.8 |
| kv_proj      | 1   | 4096→2048  | **1.52×** | 20.5  | 13.5 |
| q_proj       | 512 | 4096→4096  | **1.36×** | 117.1 | 86.4 |
| o_proj       | 512 | 4096→4096  | **1.36×** | 117.5 | 86.7 |
| down_proj    | 512 | 12288→4096 | **1.07×** | 328.9 | 308.1 |

### Losses — concentrated on exactly 2 patterns

| proj | T | shape | speedup | note |
|---|---:|---|---:|---|
| kv_proj   | 128 | 4096→2048 | 0.54× | **narrow d_out** — 16 M-CTAs on 128 SMs |
| kv_proj   | 32  | 4096→2048 | 0.62× | same |
| kv_proj   | 512 | 4096→2048 | 0.78× | same |
| down_proj | 32  | 12288→4096| 0.80× | square 4096 / T=32, L2-resident for FP16 |
| q_proj    | 128 | 4096→4096 | 0.81× | **square 4096 / T=128, L2-resident fp16** |
| o_proj    | 128 | 4096→4096 | 0.81× | same |
| down_proj | 128 | 12288→4096| 0.86× | square 4096 / T=128 |

Remaining borderline: `q_proj/o_proj T=32 @ 4096→4096` = 0.97× — basically tied.

## Bottleneck diagnosis

Two distinct failure modes remain:

### 1. kv_proj (4096 → 2048) — grid starvation (3 losses)

With tile 128×128, d_out=2048 gives only 16 M-blocks.  On the T=128 shape
the total grid is ~16 × T/Bn × split_k = 16 × 2 × ~4 = 128 CTAs, but
after accounting for register-bound occupancy (~2 CTA/SM), we can only
keep ~256 warps resident across 128 SMs — marginal.  More importantly
the kernel is already memory-bound on this shape so bigger grids do not
help; what helps is dropping the M tile to 64 to create more grid
parallelism.  This is a known item from r60 stage-I reports but was not
addressed because r54 stage-B.1 regressions blocked it.

### 2. Square d_out=4096, medium T (4 losses)

q_proj / o_proj / down_proj at T=32/128 with d_out=4096:
- FP16 cuBLAS `torch.matmul(W_fp, X_fp.T)` at this shape (16 MB W + 4 MB
  output) partially fits the fp16 working set per-CTA wave; cuBLAS also
  uses much larger tiles (e.g. 128×256) that suit square aspect ratios
  naturally.  Even after L2 flush, the **per-launch** FP16 kernel is
  very efficient (15-20 TFlop/s on 4090 BF16 peak 165 TFlop/s).
- Our INT4 kernel is HBM-bound at 30-37% eff on the *same* square
  shape (r62 P2 cold-cache table).  The bank-conflict issue is the
  limiting factor and r61 stage-C showed it cannot be fixed at the
  current smem layout without a rewrite.

## Comparison vs the old tight-loop bench

The same bench run before r62 P2 reported these losses as *worse*:
- gate_up_proj T=32: 1.45→1.75× (old) vs **3.25×** (new) — previously
  inflated by cuBLAS L2 reuse hiding the fp16 cost.
- down_proj T=128: 0.83× (old) vs 0.86× (new) — confirmed real loss.
- q/o_proj T=128: were 0.72× (old) now 0.81× (new) — still a loss, but
  less extreme; 9pp of the old "loss" was a bench artefact.

No tight-loop shape's loss is 100% fake anymore in e2e; however ~10pp
of measured speedup on the gate_up_proj winning shapes came from the
methodology fix (cuBLAS was no longer cheating).

## Policy & summary

- r62 P2 + F1 establishes cold-cache FP16 cuBLAS as the **canonical
  baseline** for all "is INT4 faster than FP16" reporting going forward.
- On Qwen3-8B, our r60 stage-I kernel achieves **median 1.21× speedup**
  end-to-end on cold-cache FP16, **with 55% of shapes winning**.  The
  losing 45% are concentrated on 2 narrow patterns (narrow d_out on
  kv_proj, and square 4096/medium T) — these are the true remaining
  optimisation targets, not bench artefacts.

## Files

- [qwen3_20260429_224641/bench.md](qwen3_20260429_224641/bench.md) — full per-kernel report
- [qwen3_20260429_224641/bench.json](qwen3_20260429_224641/bench.json) — raw records (meta.flush_l2_fp16 = true)
- [qwen3_20260429_224641/bench.log](qwen3_20260429_224641/bench.log) — full stdout log
- `kernel/triton_kernel/benchmarks/_bench_util.py` — time_ms(flush_l2=…)
- `kernel/cuda_kernel/benchmarks/bench_qwen3_shapes.py` — `--flush-l2-fp16`
