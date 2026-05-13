# Naive W4A4 CUDA baseline

This tree implements the **textbook-level** version of the 4-step W4A4
pipeline used as a comparison baseline against the multi-iteration
optimised kernels in [`../csrc/`](../csrc/).

It is intentionally kept **isolated**: it has its own `csrc_naive/`
source tree, its own torch extension (`hkust_v9_cuda_naive`), its own
Python wrapper ([`ops_naive.py`](../ops_naive.py)), and its own bench
scripts under [`kernel/bench/scripts/`](../../bench/scripts/).
Nothing here shadows or re-registers any symbol of the optimised path —
both backends load into the same Python process side-by-side without
conflict, which is exactly what the parity test relies on.

## Four kernels — one per source file

| #  | File                          | Kernel symbol                       | What it does                                                              |
|----|-------------------------------|-------------------------------------|---------------------------------------------------------------------------|
| K1 | `activation_quant_naive.cu`   | `activation_quant_naive_launch`     | Per-token SINT4 quant + 4-bit LE pack + per-group int32 sum               |
| K2 | `dense_gemm_naive.cu`         | `dense_gemm_naive_launch`           | UINT4 × SINT4 dense GEMM (`mma.m16n8k64.s4.s4.s32`) with per-group fold |
| K3 | `sparse_gemm_naive.cu`        | `sparse_gemm_naive_launch`          | SINT4 × SINT4 BSR GEMM (`mma.m16n8k64.s4.s4.s32`) with per-block scale  |
| K4 | `reduce_sum_naive.cu`         | `reduce_sum_naive_launch`           | Element-wise `Y_total = Y_low + Y_high`                                   |

The Python wrapper (`ops_naive.py`) composes K1–K4 into a
single "one projection call" that is directly comparable to the
optimised `activation_quant_cuda + fused_dense_sparse_cuda_int4`
two-kernel call used in production.

## "Naive" scope (Level L1 — Tensor-Core, no fusion/pipelining)

Present:
- **Tensor-Core `mma.m16n8k64.s4.s4.s32`** for both dense and sparse GEMM
  (fixed tile: `kBm=128, kBn=32, kBk=128`; 128 threads = 4 warps)
- Shared-memory tile caching (W tile + X tile loaded once per K group)
- Per-group fp32 fold (scale / zero·sum_X) inline in the K loop
- MMA operand registers assembled by direct lane-indexed 32-bit shmem
  reads (no `ldmatrix` — keeps the code one file one purpose)

Intentionally **absent** (what makes it a valid reference baseline):
- `cp.async` prefetch / double-buffered shmem / group cache
- Scale / zero / sum_X / sxn prefetch into shmem
- Kernel fusion (quant + GEMM + add are still 4 separate kernel launches,
  each writing its own DRAM buffer)
- Per-T dispatcher (kBn/kBm are fixed), T=1 GEMV specialisation, warp
  specialisation, split-K, CUTLASS, P0 fused quant

Rationale: by keeping the Tensor Core active, the L1 → optimised gap
isolates the cost of **fusion + pipelining + dispatch**, not the cost
of "not using Tensor Core at all".  Expected consequence: ~2-10× slower
than the optimised path on RTX 4090, i.e. the same order of magnitude
that cross-project CUTLASS-vs-tuned-CUTLASS benchmarks report.

## Parity contract

Both backends must produce the same Y_total within:
- `atol = 1e-2` OR
- `rtol = 5e-2`  (≈ 1% relative; well below the W4A4 quant error itself)

Intermediates (X_s4, scale_x, sum_X) are also sanity-compared and
reported separately; fp16 ulp differences at `rint` half-integer
boundaries are tolerated there (tolerance ≤2 counts on sum_X, ≤1e-3 on
scale_x).

Run:

```bash
python -m kernel.cuda_kernel.tests.parity_naive_vs_optimised
```

## Bench usage

Full sweep (24 triples = 3 models × 2 phases × 4 batch sizes, every
triple in a fresh subprocess):

```bash
bash kernel/bench/scripts/run_bench_w4a4_naive.sh naive_v1
# → kernel/bench/logs/w4a4_naive_isolated_naive_v1/
```

Subset (e.g. only Qwen3-4B decode for quick iteration):

```bash
bash kernel/bench/scripts/run_bench_w4a4_naive.sh smoke \
    -- --models Qwen3-4B --phases decode --batch-sizes 4 8
```

Side-by-side comparison report — read these two files together:
- `logs/w4a4_fused_ops_isolated_<LABEL>/bench_w4a4_summary.md`       (optimised / "iteration")
- `logs/w4a4_naive_isolated_<LABEL>/bench_w4a4_naive_summary.md`     (naive)

Both tables share the schema `(model, phase, bs, op, T, d_in, d_out,
median_us, ...)` so a row-wise diff gives per-op speedup directly.

## Provenance / why this baseline is trustworthy

- **Same tensor layouts.**  W_low UINT4 packing, X_s4 SINT4 LE pack,
  per-group scale/zero/sum_X, BSR (128-row block × BCOL/2-byte columns),
  perm — all identical to the optimised path.  The parity test feeds
  the *same* random inputs to both and checks `Y_total` bit-for-bit.
- **Same sparsity target.**  5% block density is fixed across every
  (model, phase, bs) triple, matching the agreed comparison setup.
- **Same timing protocol.**  Uses `kernel.bench.layer.timing.measure`
  (min-of-outer of mean-of-inner, median over ≥5 trials), per
  [[memory:bmmiahpl]].  Adaptive preset thresholds are re-tuned for
  naive's slower per-call time (HEAVY_MS=20, MED_MS=3 vs 2/0.5 on the
  optimised side).

## Files added by this baseline

```
kernel/cuda_kernel/
  csrc_naive/
    activation_quant_naive.cu
    dense_gemm_naive.cu
    sparse_gemm_naive.cu
    reduce_sum_naive.cu
    bindings_naive.cc
  ops_naive.py
  tests/parity_naive_vs_optimised.py

kernel/bench/
  layer/qwen3_w4a4_ops_naive.py
  scripts/bench_w4a4_naive_fused_ops.py
  scripts/bench_w4a4_naive_isolated_driver.py
  scripts/run_bench_w4a4_naive.sh
```

None of the files in [`../csrc/`](../csrc/), `ops.py`,
`qwen3_w4a4_ops.py`, or the existing bench drivers are modified.
