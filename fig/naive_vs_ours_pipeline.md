# Naive vs Ours — Where the Kernel Pipeline Actually Changed

> Purpose: a paper-ready narrative of the structural differences between the
> naive 4-kernel baseline and our fused single-kernel pipeline for W4A4
> sparse-augmented GEMM on RTX 4090, including which operations got
> **split**, which got **hoisted earlier into bubbles**, and which got
> **eliminated entirely**.
>
> Sources:
> - Naive tree: [`kernel/cuda_kernel/csrc_naive/`](../cuda_kernel/csrc_naive/)
> - Ours:       [`kernel/cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`](../cuda_kernel/csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu)

---

## 1 · Top-level topology (4 kernels → 1 persistent-CTA pipeline)

### Naive (`csrc_naive/`)
Four independent kernels, each launched separately, each round-tripping
through HBM; stages are fully serialized:

```
quant → HBM(X_s4, sum_X, scale_x)
      ↓  launch barrier
dense_gemm  → HBM(Y_low)
      ↓  launch barrier
sparse_gemm → HBM(Y_high)
      ↓  launch barrier
reduce_sum  → HBM(Y_total)     ← Y_total = Y_low + Y_high
```

Four concrete pieces of evidence:
- `csrc_naive/activation_quant_naive.cu` writes `X_s4 / scale_x / sum_X`
  all the way back to HBM.
- `csrc_naive/dense_gemm_naive.cu` writes `Y` (i.e. `Y_low`) to HBM in its
  epilogue, and re-reads `scale_u4 / zero_u4 / sum_X` **from HBM for every
  K-group** (see `__half2float(zero_u4[...])` and
  `static_cast<float>(sum_X[...])` in the g-loop fold).
- `csrc_naive/sparse_gemm_naive.cu` symmetrically writes `Y_high` back to
  HBM.
- `csrc_naive/reduce_sum_naive.cu` then performs a separate, standalone
  `Y_low + Y_high` kernel — a full HBM round-trip for nothing but an
  fp16 add.

### Ours (`csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu`)
A single kernel. Inside one CTA we run dense → sparse → epilogue in order,
and **`Y_low`/`Y_high` never materialize in HBM**. The dense-branch line

```cpp
y_fp[im][in_sub][r] += corrected * s;      // R27
```

and the sparse-branch line

```cpp
y_fp[im][in_sub][r] += 16.0f * static_cast<float>(d_val) * s;
```

both accumulate into the **same** FP32 register tile `y_fp[...]`. The
final scale-x multiply and writeback produce a single output tensor
`Y_total`.

> **This is the concrete meaning of "kernel fusion" here: four HBM
> handshakes collapse into one HBM read (W/X) plus one HBM write
> (`Y_total`). `reduce_add` is not "faster" — it is _eliminated_, leaving
> a single `+=` in registers.**

---

## 2 · Three-fold mainloop pipelining (entirely absent in naive)

### 2.1 kStages=2 `cp.async` ring buffer — HBM↔SMEM overlapped with MMA

Naive: strictly serial `LDG → __syncthreads → MMA → __syncthreads`, with
single-buffered `sW[128][64]`, `sX[32][64]`.

Ours (line 275 onward in `fused_dense_sparse_mma_int4.cu`):

```cpp
__shared__ alignas(16) uint8_t sW[kStages][kBm][bytes_per_group];
__shared__ alignas(16) uint8_t sX[kStages][kBn][bytes_per_group];
// ...
// Prologue: issue kStages-1 cp.async groups (g_start .. g_start+kStages-2)
// Mainloop iteration g:
//   1) cp.async group g_ahead = g + (kStages-1) into buf_ahead
//   2) cp_async_wait_group<kStages-2> — wait until only the oldest
//      in-flight group remains
//   3) ldmatrix + MMA on buf = g & 1
```

**Effect:** MMA of group *g* runs **concurrently with the HBM→SMEM
transfer of group g+1**. On the figure this is the "HBM load" lane and
the "Tensor Core" lane offset by one tile and overlapping.

### 2.2 Per-m-row prefetch — dequant coefficients hoisted into MMA bubbles

Naive reads `zero_u4[m,g]`, `scale_u4[m,g]`, `sum_X[n,g]` in the
**epilogue** — i.e. only _after_ the MMAs finish. During those HBM reads
the Tensor Cores sit idle (classic pipeline bubble).

Ours issues a per-m-row prefetch inside `run_mma_pass` (lines ~594–738),
so the coefficients are fetched **before** the MMAs of that row subtile,
and the fold is applied in-register immediately afterwards:

```cpp
auto run_mma_pass = [&](int buf, auto fold_fn, auto prefetch_fn, int g_or_bc) {
    // For each m-row subtile:
    //   1) prefetch_fn(mrow0, mrow1, g_or_bc)
    //        — pulls (z0, s0, z1, s1) from HBM into registers
    //          _before_ issuing this row's MMAs
    //   2) ldmatrix + MMA
    //   3) fold_fn: in-register  acc - z * sum_X  then  * s
};
```

`sum_X[kStages][kBn]` is additionally preloaded into shared memory
(lines 311 / 532), so the epilogue no longer needs a per-output-tile HBM
trip for it.

**Effect:** dequant `z * sum_X` and `* scale` stop being "another phase
after MMA" and instead get **interleaved with Tensor-Core issue**,
consuming warp-scheduler math-issue slots that would otherwise be idle.

### 2.3 `scale_x` promoted to CTA prologue + one fp16→fp32 pre-conversion

Naive: `scale_x[n_global]` is fetched from HBM **per output element** in
the final writeback (both the dense and sparse naive kernels contain
that pattern), plus one `__half2float` conversion per element.

Ours (lines 309 / 359 / 822–836):

```cpp
__shared__ __half s_scale_x[kBn];                 // one HBM→SMEM, prologue
s_scale_x[tid] = (n < T) ? scale_x[n] : __half(0);
// ...
// Each thread converts only the n_local values it actually needs, once:
float sxn_cache[kNsubPerCta][2];
sxn_cache[in_sub][cc] = __half2float(s_scale_x[n_local]);
```

**Effect:** `scale_x` goes from "one HBM read + one `hft2f` per output
element" to "one HBM read per CTA + a constant number of `hft2f` per
thread". On the figure this is the small `scale_x` chore being **slid
forward into the prologue's HBM-wait bubble** — which is precisely the
"what got filled into the bubble" answer.

### 2.4 `ldmatrix` + XOR swizzle replaces lane-indexed 32-bit scalar reads

Naive assembles `a_regs / b_regs` via
`*reinterpret_cast<const uint32_t*>(&sW[row][col])` — four scalar 32-bit
loads per thread per MMA, with **no `ldmatrix`**.

Ours, when `kUseLdmatrix = true`, uses
`ldmatrix.sync.aligned.m8n8.x4.shared.b16` (lines 645 / 685) together
with an XOR address swizzle (lines 140–185) — hardware handles the
warp-level distribution and cuts shared-memory bank conflicts.

**Effect:** two `ldmatrix.x4` replace eight 32-bit scalar loads, reducing
both issue pressure and bank-conflict stalls. This is the precondition
that allows the Tensor-Core lane to sit flush against the `cp.async`
lane in the space–time diagram.

---

## 3 · Three paper-ready claims

Turn §1–§2 into three sentences that can be dropped into the paper:

1. **Fusion removes the reduce-add stage and two intermediate HBM
   writes + one intermediate HBM read.** `Y_low` and `Y_high` are not
   "added faster" — they never exist in HBM; they are simply two partial
   sums accumulated into the same FP32 register tile. In the space–time
   diagram the `reduce_add` stage is *deleted*, not shortened.
2. **The mainloop uses `cp.async` double-buffering to overlap HBM
   transfers with Tensor-Core MMAs.** Naive runs "load a tile, stall,
   compute a tile, stall"; ours computes tile *g* while tile *g+1* is
   streaming in. The synchronous bubbles are consumed by `cp.async`.
3. **Dequantization is sliced into the MMA pass and interleaved with
   Tensor-Core issue, instead of being a separate post-MMA phase.**
   Per-m-row `(z, s)` prefetch, plus promoting `sum_X` / `scale_x` into
   shared memory with a single fp16→fp32 pre-conversion, splits the
   original "dequant fold" stage in half: coefficient fetch overlaps
   with Tensor-Core issue, and `acc − z·sum_X` overlaps with the
   Tensor-Core wait slot.

---

## 4 · Suggested three annotation points for the figure

| Marker | Naive (baseline) | Ours (this kernel) |
|---|---|---|
| 🅐 **Stage elimination** — `reduce_add` no longer exists | Explicit 4-th kernel; one full HBM read + write | `+=` on the `y_fp` register tile; no stage at all |
| 🅑 **Stage overlap** — `cp.async` double-buffer | `LDG → sync → MMA → sync`, strictly serial | `cp.async(g+1)` runs in parallel with `MMA(g)` |
| 🅒 **Stage hoisting** — dequant coefficients pulled forward | Coefficients read from HBM *after* MMA | `(z, s)` prefetch + `sum_X`/`scale_x` in SMEM, filling TC issue slots |

On the existing `pipeline_spacetime.tex` figure, the light-yellow
overlap band (`hlC`) is the natural place to drop ①②③ circled markers
so reviewers immediately see where to look.

---

## 5 · Likely reviewer pushback (and pre-emptive answers)

- **Q: Why is quant itself slightly slower in ours (22.79 μs) than in
  naive (19.96 μs)?**
  A: Ours emits `scale_x` / `sum_X` in a layout the downstream fused
  kernel can load directly from SMEM, avoiding `stride_sx_n /
  stride_sx_g` mis-alignment in the mainloop. Those ~2.8 μs are a
  one-time prepayment and are recovered (with interest) by the fused
  stage (`Fused 21.73 μs ≪ Dense 41.75 + Sparse 23.12`). The reported
  **2.09×** end-to-end speed-up is the net accounting.

- **Q: `Fused (dense+sparse) = 21.73 μs` is faster than `Dense (41.75
  μs)` alone — is the dense part itself being optimized, not just
  fused?**
  A: Yes. `cp.async` + `ldmatrix` + per-m-row prefetch act on the dense
  mainloop simultaneously, so even before counting the elimination of
  `reduce_add`, the fused kernel's dense phase is already faster than
  the naive dense kernel. The paper should state this honestly as a
  **joint** benefit of fusion and intra-stage pipelining, not attribute
  it to either alone — the two are inseparable in our design.

- **Q: Why `kStages=2` and not `kStages=3`?**
  A: `static_assert(kStages >= 2 && kStages <= 3)` leaves 3 as a valid
  knob, but the dispatcher pins to 2 by default (see the `C.11-A`
  post-mortem comment). At our target shapes (kBm=128, T < 512), the
  register-pressure vs stages Pareto point sits at 2; `kStages=3`
  regressed. Paper note: "`kStages=3` was evaluated but regressed at our
  target shapes due to register pressure; see ablation Table X."

---

## 6 · Recommended figure caption (drop-in)

> **Figure (a). Latency breakdown of the W4A4 sparse-augmented GEMM on
> RTX 4090.** The naive baseline launches four serialized kernels that
> round-trip through HBM. Our kernel (i) *fuses* the dense, sparse, and
> reduce-add stages into a single persistent-CTA pipeline whose
> `Y_low / Y_high` live only in FP32 registers, (ii) overlaps HBM→SMEM
> transfers with Tensor-Core MMAs via a `cp.async` 2-stage ring buffer,
> and (iii) hoists the per-group `(z, s, sum_X)` dequantization
> coefficients into SMEM and interleaves their fold with MMA issue,
> collapsing **93.01 μs → 44.52 μs (2.09×)**.

