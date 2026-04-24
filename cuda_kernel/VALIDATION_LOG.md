## `cuda_kernel` Validation Log

RTX 4090 / SM89 / torch 2.8.0+cu126 / triton 3.4.0.  One-host
experiment journal: every run, decision, and delta lives here so we
can resume after a context switch without re-deriving history.

All timestamps in UTC+8 (server `autodl`).

---

## Run 2026-04-24 11:52: first JIT + first parity attempt

### Build

- Installed `ninja==1.13.0` in conda env `zip` (was missing).
- `HKUST_V9_CUDA_VERBOSE=1 python -c "from kernel.cuda_kernel import ops"`
  compiled all 5 TU (bindings + 4 kernels) in **118.8 s** against
  `arch=compute_89,code=sm_89`.  All 4 Python wrappers report `OK`.
- Register / shared-memory footprint per kernel (ptxas summary):

  | Kernel              | widest kBn/kBt | regs | smem (B) | spill |
  |---------------------|---------------:|-----:|---------:|------:|
  | activation_quant    | kBt=32         |  36  |   1184   | 0     |
  | dense_gemm          | kBn=64         | 128  |   4480   | 0     |
  | sparse_gemm         | kBn=64         | 128  |   4224   | 0     |
  | fused_dense_sparse  | kBn=64         | 238  |   4480   | 0     |

  No spills anywhere.  `fused` at 238 regs caps occupancy at ~1 block
  per SM; flagged as future optimisation target but not blocking.

### First parity run

Command:

```bash
python -m pytest kernel/cuda_kernel/tests/test_parity.py -x -rA --tb=short
```

Result:

```
test_activation_quant_parity[identity-1-4096]       PASSED
test_activation_quant_parity[identity-16-4096]      FAILED
  CUDA error: invalid configuration argument
```

### Root cause

`activation_quant.cu::launch` dispatched `kBt=16` for `T in (4, 16]`
and `kBt=32` for larger T.  Block dimension is
`(kLanesPerGroup=128, kBt, 1)` → `128 * 16 = 2048` threads, which
exceeds the **1024 threads/block hardware limit on SM89**.  The
kernel compiled (ptxas only validates register/smem budgets) but the
launch was rejected at runtime.

### Fix

`activation_quant.cu`: cap `kBt` at 8 (= 1024 threads/block max).
Prefill path has no user for large-T anyway -- `policy._auto_policy`
already routes `T >= 256` to Triton.

```diff
-    else if (T <= 16)  dispatch(std::integral_constant<int, 16>{});
-    else if (T <= 64)  dispatch(std::integral_constant<int, 32>{});
-    else               dispatch(std::integral_constant<int, 32>{});
+    else               dispatch(std::integral_constant<int, 8>{});
```

### Cross-check: other kernels

`dense_gemm`, `sparse_gemm`, `fused_dense_sparse` all use
`dim3 block(128, 1, 1)` -- 128 threads/block, well under the limit.
No change needed.

### Deliverables

- [x] code patch committed
- [ ] rerun parity (pending)
- [ ] record full parity pass / fail matrix
- [ ] benchmark vs Triton

---

## Run 2026-04-24 12:00: parity round 2 (3 parity bugs fixed)

### Second parity attempt reproduced

```
35 tests collected
activation_quant: 14 tests -- 4 passed (T<=8) / 10 failed (T>=16)
dense_gemm:       7  tests -- 5 passed / 2 failed (T=16, T=64)
sparse_gemm:      5  tests -- all 5 passed
fused:            4  tests -- all 4 passed
end_to_end:       4  tests -- all 4 NameError
dispatcher:       1  test  -- passed
```

### Bug #2: `dense_gemm` / `fused_dense_sparse` scale_x stride bug

**Root cause (reconstructed from failure on T=16, M=4096):**
The `s_scale_x` shared-memory staging loop used `stride_sx_n` as the
index multiplier for `scale_x`:

```cpp
s_scale_x[tid] = scale_x[(int64_t)n * stride_sx_n];
```

But `stride_sx_n` was actually `sum_X.stride(0)` (== `n_groups`) in
the launcher -- `scale_x` is a 1D tensor and its stride is always 1.
For `tid >= 1` this read `scale_x[n_groups]`, `scale_x[2*n_groups]`,
... which is out-of-bounds and returned garbage.

**Why T=1 passed:** at T=1 `kBn=1`, so only `tid=0` participates and
`0 * stride = 0` — it incidentally hit the first element correctly.

**Fix:** `scale_x` is 1D contiguous; enforce `stride(0) == 1` in the
launcher and just index `scale_x[n]` directly.

### Bug #3: `test_end_to_end_parity` NameError

**Root cause:** copy-paste from `test_activation_quant_parity` left
`assert torch.equal(scale_c, scale_t)` and
`assert torch.equal(sum_c, sum_t)` in the end-to-end test, but the
end-to-end test only computes a single `Y` output — those names don't
exist.

**Fix:** delete the stray asserts.

### Bug #4: `activation_quant` T>=16 bytes-diff (5..18 per 256K bytes)

**Symptom:** `scale_x` bit-exact match, `sum_X` slightly off,
`X_s4` differs in ~5-18 positions.  All diffs land on half-integer
`x/s` values (e.g. 0.43359375 / 0.2890625 == 1.5 in IEEE fp32 math).

**Root cause (subtle, GPU-specific):**

- Triton lowers `x / s` to PTX `div.approx.f32` (Newton-Raphson,
  ~0.5 ULP error) — Triton's rendered IR uses `llvm.nvvm.div.approx.f`.
- nvcc for plain C++ `x / s` emits PTX `div.rn.f32` (fully rounded
  IEEE div) by default.

For the 1.5-boundary case, approx-div returned 0x3fbfffff
(1.4999998807907104) while rn-div returned 0x3fc00000 (1.5 exact).
`rintf` (round-to-nearest-even) then mapped them to 1 vs 2.

**Fix:** use `__fdividef(x, s)` in the CUDA kernel to force the
approx-div path, matching Triton's PTX byte-for-byte.

```diff
-    float q = rintf(x / scale_safe);
+    float q = rintf(__fdividef(x, scale_safe));
```

### Re-run result

```
35 passed in 22.54s
```

Every shape, every permutation, every sparsity ratio — bit-exact with
Triton.  Tolerance for fp16 GEMM output retained (1 ULP cast rounding),
never triggered.

### Deliverables

- [x] 3 parity bugs diagnosed + fixed
- [x] full parity pass (35/35)
- [x] commits pushed to `server` + `origin` (eda1de6)
- [ ] benchmark vs Triton (up next)

---

## Run 2026-04-24 12:25: first CUDA-vs-Triton benchmark

### Setup

Harness: `kernel/cuda_kernel/benchmarks/bench_cuda_vs_triton.py`
  - min-of-means, 10 outer × 50 inner, 10 warmup
  - dual-sink logging (INFO stdout + DEBUG log file)
  - output under `/root/logs/cuda_kernel/bench_20260424_122507.{json,md,log}`

Shapes covered (T × d_in × d_out):
  decode   : T ∈ {1, 8, 16}, d ∈ {4k, 11k}
  batch    : T ∈ {64, 128}, d = 4k
  prefill  : T ∈ {512, 1024}, d = 4k

### Results

`activation_quant` — CUDA wins uniformly:

| shape             | triton | cuda  | speedup |
|-------------------|-------:|------:|--------:|
| dec_T1_4k_4k      |  44us  | 15us  | **2.98x** |
| dec_T1_11k_4k     | 113us  | 24us  | **4.70x** |
| bat_T128_4k_4k    |  55us  | 18us  | **3.09x** |
| pre_T1024_4k_4k   |  63us  | 19us  | **3.30x** |

`dense_gemm` — CUDA only wins at T=1:

| shape             | triton | cuda   | speedup |
|-------------------|-------:|-------:|--------:|
| dec_T1_4k_4k      |  69us  |  59us  | 1.17x    |
| dec_T1_11k_4k     |  69us  |  59us  | 1.17x    |
| dec_T1_4k→11k     | 134us  | 180us  | **0.75x** (lose) |
| dec_T8_4k_4k      |  69us  | 260us  | 0.27x    |
| dec_T16_4k_4k     |  69us  | 903us  | **0.08x** |
| bat_T64_4k_4k     |  69us  | 1812us | **0.04x** |
| pre_T1024_4k_4k   | 284us  | 5197us | **0.05x** |

`sparse_gemm` — CUDA only wins at T=1:

| shape             | triton | cuda   | speedup |
|-------------------|-------:|-------:|--------:|
| dec_T1_4k_4k      |  68us  |  18us  | **3.73x** |
| dec_T16_4k_4k     |  68us  | 108us  | 0.63x    |
| bat_T128_4k_4k    |  67us  | 482us  | 0.14x    |

`fused_dense_sparse` — CUDA narrowly wins at T=1 (small d_out):

| shape             | triton | cuda   | speedup |
|-------------------|-------:|-------:|--------:|
| dec_T1_4k_4k      |  80us  |  65us  | 1.25x    |
| dec_T1_11k_4k     |  81us  |  64us  | 1.26x    |
| dec_T1_4k→11k     | 179us  | 188us  | 0.95x    |
| dec_T16_4k_4k     | 121us  | 995us  | 0.12x    |

**End-to-end `v9_linear_forward`** (chains quant + fused):

| shape             | triton | cuda   | speedup |
|-------------------|-------:|-------:|--------:|
| dec_T1_4k_4k      | 231us  |  82us  | **2.82x** ✨ |
| dec_T1_4k_11k     | 229us  |  83us  | **2.75x** |
| dec_T1_11k_4k     | 265us  | 223us  | 1.19x    |
| dec_T8+           | ~237us | 325us+ | 0.73x and worse |

### Analysis

**Where CUDA wins (T=1 decode):** launch overhead dominates Triton's
total latency.  Triton has to autotune-dispatch, initialise a grid of
1-element programs, and pay for tl.dot constexpr expansion.  The
CUDA kernels are single-CTA-per-row with fixed tile shape and skip
all of that.

**Where CUDA loses (T ≥ 8):** our SIMT dp4a inner loop iterates
`kBn` output columns *sequentially inside one thread*.  At kBn=8 this
produces a 4096-long dp4a chain with no ILP -- nvcc can't unroll far
enough to hide latency, and we have no MMA TC throughput to fall back
on.  Triton's m16n8k16 MMA tile keeps the tensor cores fed.

The **5-50x CUDA loss** at T ≥ 16 is alarming at face value, but
doesn't hurt the product because the policy routes those shapes to
Triton automatically.

### Policy calibration

Updated `kernel/backend/policy.py::_auto_policy` to the measured
table:

- `activation_quant`  → CUDA always
- `dense_gemm`        → CUDA iff (T==1 and d_out<=d_in) else Triton
- `sparse_gemm`       → CUDA iff T==1 else Triton
- `fused_dense_sparse`→ CUDA iff (T==1 and d_out<=d_in) else Triton

Net product effect: **end-to-end decode (T=1) gets ≈2.8x** over the
pure-Triton baseline; prefill (T≥8) is unchanged.

### Deliverables

- [x] bench harness + JSON/MD output
- [x] full 45-row result table captured
- [x] policy re-calibrated to real measurements
- [ ] Phase 5 (future): MMA PTX rewrite of dense/sparse to flip
      the T≥8 loss; persistent-BSR queue for sparse.

---

## Iteration log (automated self-optimization)

Summary: across 5 rounds I pushed end-to-end T=1 from 2.82x to 3.17x,
T=8 from 0.73x (net loss) to 2.21x, and T=16 from 0.50x to 1.67x.
Each round changes ONE dimension at a time and is recorded below.

### Round 2 (13:06) -- K-outside/N-inside loop swap  [REGRESSION]

- Motivation: in the N-outside/K-inside layout, each thread ran a
  32-deep serial dp4a chain per (M,N) entry.  Swapping exposes kBn
  independent chains per i-iteration -> ILP.
- Change: swap loops in dense_gemm.cu, fused (dense + sparse
  halves), sparse_gemm.cu.
- Result: T=1 decode up 1.17x -> 1.37x.  But T>=8 **got worse**: T=8
  went from 0.27x to 0.08x, T=16 from 0.08x to 0.05x.
- Diagnosis: The new layout stores ``acc_n[kBn]``, ``x0_n[kBn]``,
  ``x1_n[kBn]`` as register arrays.  At kBn>=8 these exceed the
  budget and spill ``w_dp4a[32]`` to local memory; spill cost >> ILP
  gain.  Confirmed with a probe: slicing X along T and looping
  ``dense_gemm_cuda`` per row (which forces kBn=1) is 2.3x faster
  than the dispatched kBn=8 path at T=8.
- Decision: keep the loop swap (T=1 win is real) but cap kBn.
- Commit: `524cc21`.

### Round 3 (13:19) -- cap kBn<=4 to eliminate spill  [BIG WIN]

- Change: dispatch table becomes
  ```
  T=1   -> kBn=1
  T<=8  -> kBn=2
  else  -> kBn=4
  ```
- Rationale: N-parallelism moves onto the grid (grid.y = T/kBn) instead
  of the thread.  4090 has 128 SMs; grid of up to 32 (M) * 4 (N) = 128
  CTAs saturates one wave.  Each thread's reg footprint drops to
  ~40 baseline + 4*3 = 52 regs, well under the 64-reg budget for 2
  blocks/SM occupancy.
- Result: `bench_20260424_132022`.  Dramatic wins across the board:

  | path | Round 2 | Round 3 | Δ |
  |---|---:|---:|---:|
  | dense T=8        | 0.08x | **1.11x** | ~14x |
  | dense T=16       | 0.05x | 0.59x | ~12x |
  | sparse T=16      | 0.42x | **3.68x** | ~9x |
  | sparse T=64      | 0.15x | **1.59x** | ~11x |
  | sparse T=128     | 0.10x | **1.16x** | ~12x |
  | fused T=8        | 0.09x | **1.05x** | ~12x |
  | end-to-end T=8   | 0.22x | **2.30x** | ~10x |
  | end-to-end T=16  | 0.14x | **1.47x** | ~10x |

- Diagnosis lock-in: confirms Round 2's regression was purely
  register-spill-driven.  The ILP benefit was real but was being
  completely masked.  Once spill is removed the ILP gain is still
  exposed and compounds.
- Commit: `97002cf`.

### Round 4 (13:29) -- 128-bit __ldg + policy widen  [MARGINAL]

- Kernel: replace the 16x `__ldg(uint32_t*)` per W row with 4x
  `__ldg(uint4*)`.  Halves LD instruction count, safe because
  `d_in_half = d_in/2` is always a multiple of 16 (since BCOL=128
  divides d_in).
- Policy: extend CUDA coverage to T<=8 for dense/fused, T<=128 for
  sparse.
- Result: `bench_20260424_132910`.  Kernel changes are within noise
  (+/- 5%); policy changes are the real win (new end-to-end wins at
  T=8 and T=16 are now dispatched to CUDA instead of falling back).
- Diagnosis: W loads are not the bandwidth bottleneck -- we are
  compute-bound on the dp4a chain.  128-bit loads hurt nothing but
  don't help either.  The dominant cost is still the ~32 dp4a/group
  per thread.
- Commit: `f449c3e`.

### Round 5 (13:33) -- T<=16 also kBn=2  [SMALL WIN]

- Motivation: T=16 measured at 108us in Round 4 (kBn=4) vs T=8 at
  68us.  kBn=4 is already borderline-spilling.
- Change: shift the threshold so T<=16 uses kBn=2, T>16 stays at
  kBn=4.
- Result: `bench_20260424_133330`.
  - dense T=16: 108us -> 101us (0.64x -> 0.71x)
  - fused T=16: 123us -> 110us (0.66x -> 0.76x)
  - end-to-end T=16: 150us -> 141us (1.56x -> **1.67x**)
- Diagnosis: confirms kBn=4 is still mildly spill-pressured.  But the
  improvement is small because T=16 with kBn=2 needs grid.y=8 and
  4090 fills exactly 2 waves (32 M-CTAs * 8 N-CTAs = 256 total, 2
  blocks/SM * 128 SM = 256) -- edge effects in the tail wave
  dominate the remaining latency.  Further shrinking kBn won't help.
- Commit: `5c444cf`.

### Round 6 (13:52) -- cp.async double-buffered X for dense_gemm  [BIG WIN T=1]

- Motivation: T=1 dense was stuck at 1.17-1.37x (~63us wall time).
  Profiling suggested the 32-group K-loop was serialising HBM load
  and dp4a; specifically, each group's sX staging was a ~60-cycle
  latency bubble before dp4a could start consuming it.
- Change: introduce cp.async helpers in common/arch.cuh
  (cp_async_cg_16, cp_async_commit, cp_async_wait_group<N>).
  dense_gemm now allocates two alignas(16) shmem banks sX[2][kBn][64]
  and s_sum_X[2][kBn].  Prologue kicks off group 0; per-iter body
  issues g+1's load, commits, waits on group 1 stack, syncs, and
  computes on bank (g&1).  Last iteration waits<0> (drain).
- Gotchas resolved:
  1. Removed the `#if __CUDA_ARCH__ >= 800` guard around the helpers
     in arch.cuh -- with it on, nvcc's host-pass elided the definitions
     and the subsequent device pass reported them as undefined.  The
     header already #errors for arch<800 at the top so it's safe to
     declare them unconditionally.
  2. Added `alignas(16)` to the sX shmem array; without it CUDA
     raised "misaligned address" at runtime because cp.async.cg
     requires 16-byte aligned shmem destination.
- Result: `bench_20260424_134230`.
  - dense T=1 4k/4k:  61us -> **48us** (1.17x -> **1.46x**), +25%
  - dense T=1 4k/11k: 63us -> **48us** (1.08x -> **1.43x**), +32%
  - dense T=1 11k/4k: 179us -> 129us (0.74x -> **1.04x**), flipped
  - T>=8: small noise-level changes, still within budget
- End-to-end T=1: 73us -> 70us (3.17x -> **3.26x**)
- Commits: `dafb642`, `6c05ee6`, `d7d1aa9`.

### Round 6b (13:58) -- extend cp.async to fused dense branch

- Same double-buffer template applied to the dense half of
  fused_dense_sparse.cu; sparse half (BSR loop) left single-buffered
  for now.  sX is now alignas(16) sX[2][kBn][64] and sparse uses
  sX[0].
- Result:
  - fused T=1 4k/4k:  68us -> **52us** (1.19x -> **1.55x**), +25%
  - fused T=1 4k/11k: 75us -> **52us** (1.07x -> **1.54x**), +42%
  - end-to-end T=1: unchanged at 70us (already dominated by dense)
- Commit: `dadede5`.

### Round 7 (14:02) -- cp.async for sparse_gemm BSR loop

- BSR indirection makes prefetch non-trivial: we need bc_next =
  __ldg(&hp_col_indices[block_idx+1]) before we can issue the next
  cp.async.  Luckily hp_col_indices is small and fully cached in L1.
- Result mixed:
  - sparse T=1 4k/4k:   18us -> **17.7us** (3.78x -> **3.89x**)
  - sparse T=1 4k/11k:  18us -> **17.5us** (3.83x -> **3.96x**)
  - sparse T=8 4k/4k:   18us -> **17.7us** (3.84x -> **3.95x**)
  - sparse T=16 4k/4k:  18us -> **17.6us** (3.76x -> **3.93x**)
  - sparse T=64 4k/4k:  42us -> 82us (1.59x -> 0.84x) REGRESSION
  - sparse T=128 4k/4k: 60us -> 128us (1.16x -> 0.54x) REGRESSION
- Diagnosis: at T>=64 the kBn=4 path already has per-thread register
  pressure (acc_n[4]+x0_n[4]+x1_n[4]+w_dp4a[32]=44 regs), and the
  added register pressure from the cp.async bc_next prefetch path
  + double-buffered addressing tipped us over the spill threshold.
  Rather than re-engineer, narrow policy: sparse -> cuda only for T<=16.
- Commit: `ab5f5de`, policy tweak `f9d5a9`.

### Where we are (as of Round 7)

Final bench (iter-Round 7, `bench_20260424_140158`):

| kernel              | shape        | Triton  | CUDA    | speedup |
|---------------------|--------------|--------:|--------:|--------:|
| dense_gemm          | T=1 4k/4k    | 69us    | 48us    | **1.46x** |
| dense_gemm          | T=1 4k/11k   | 69us    | 48us    | **1.43x** |
| dense_gemm          | T=1 11k/4k   | 135us   | 129us   | 1.04x     |
| dense_gemm          | T=8 4k/4k    | 69us    | 66us    | 1.04x     |
| sparse_gemm         | T=1 4k/4k    | 69us    | 17.7us  | **3.89x** |
| sparse_gemm         | T=8 4k/4k    | 70us    | 17.7us  | **3.95x** |
| sparse_gemm         | T=16 4k/4k   | 69us    | 17.6us  | **3.93x** |
| fused_dense_sparse  | T=1 4k/4k    | 81us    | 52us    | **1.55x** |
| fused_dense_sparse  | T=1 4k/11k   | 81us    | 52us    | **1.54x** |
| fused_dense_sparse  | T=1 11k/4k   | 161us   | 135us   | 1.19x     |
| fused_dense_sparse  | T=8 4k/4k    | 80us    | 77us    | 1.04x     |
| end-to-end v9_linear| T=1 4k/4k    | 229us   | 69us    | **3.31x** |
| end-to-end v9_linear| T=1 4k/11k   | 229us   | 70us    | **3.26x** |
| end-to-end v9_linear| T=1 11k/4k   | 266us   | 170us   | **1.56x** |
| end-to-end v9_linear| T=8 4k/4k    | 240us   | 105us   | **2.29x** |
| end-to-end v9_linear| T=16 4k/4k   | 239us   | 142us   | **1.68x** |

Final policy:
  - activation_quant   -> cuda always
  - dense_gemm         -> cuda iff T<=8 AND d_out<=d_in
  - sparse_gemm        -> cuda iff T<=16
  - fused_dense_sparse -> cuda iff T<=8 AND d_out<=d_in

### What's left on the table (Phase 5 backlog)

- dense_gemm T>=32: still loses badly on CUDA; unavoidable without
  tensor-core MMA (``m16n8k32.s8.s8.s32``) which we did not write in
  this phase.  Likely 4-8x speedup potential for prefill if done.
- sparse_gemm T>=512: similar; the BSR persistent queue idea in the
  comments would help, independently of MMA.
- activation_quant: already 3-4.7x and memory-bound.  No headroom.

### Files of record

- iteration bench artefacts:
  `logs/cuda_kernel/bench_2026042[4_*].{json,md,log}`
- final policy: `kernel/backend/policy.py::_auto_policy`
- final kernels: `kernel/cuda_kernel/csrc/**`
