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
