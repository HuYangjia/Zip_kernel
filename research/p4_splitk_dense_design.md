# P4 — Split-K Dense GEMM for Decode (routes to unlocking SM occupancy)

**Author**: kernel-dev · **Date**: 2026-04-23 · **Status**: design, pre-implementation

---

## 0. One-line goal

> **For `T ≤ 32` decode shapes, run dense UINT4 × SINT4 GEMM with a K-axis split so the grid is no longer `d_out/BM` programs (≤ 32 programs) but `(d_out/BM) × SPLIT_K` programs (~128+), letting all 128 SMs of the RTX 4090 work in parallel.**

Expected win: **1.5×–2.0×** on the decode-hot shapes where grid size is the direct bottleneck.
Target shapes: `T ∈ {1, 4, 16}`, `d_out ∈ {4096, 11008, 14336}`, `d_in ∈ {4096, 11008, 14336}`.

---

## 1. Why P4 now: the evidence

### 1.1 Step 1 (P3) exit data recap

From `bench_dense_to_out.py` (2026-04-23, RTX 4090, min-of-means):

| T | d_out | d_in | plain_us | fused_us | vs_plain |
|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 60.56 | 59.13 | **1.02×** 🟡 |
| 1 | 4096 | 11008 | 135.73 | 121.42 | 1.12× |
| 1 | 11008 | 4096 | 61.60 | 59.55 | **1.03×** 🟡 |
| 1 | 14336 | 4096 | 61.92 | 60.39 | **1.03×** 🟡 |
| 1 | 28672 | 4096 | 78.37 | 75.20 | 1.04× |
| 4 | 4096 | 4096 | 70.88 | 58.57 | 1.21× |
| 4 | 14336 | 4096 | 70.82 | 59.60 | 1.19× |
| 16 | 4096 | 4096 | 71.45 | 59.90 | 1.19× |
| 16 | 14336 | 4096 | 70.11 | 59.94 | 1.17× |
| 16 | 28672 | 4096 | 86.62 | 81.64 | 1.06× |

**Key takeaway**: epilogue-fusion already lands. But `T=1, d_in=4096` tier only gains **3–4%** because:
- plain kernel is already at **~59 µs** ≈ effectively back-to-back launch + compute floor
- further time is spent *inside* the GEMM kernel, not in surrounding passes

### 1.2 The SM-occupancy story (the actual bottleneck)

Grid formula for `dense_gemm_kernel` (see `dense_u4s4_gemm.py`):

```
grid = (cdiv(d_out, BM), cdiv(T, BN))
```

Decode regime autotune picks `BM ∈ {64, 128}` and `BN=16` (the only small-BN config), so:

| shape | best BM | grid = (d_out/BM, T/BN) | # programs | SM util on 4090 (128 SM) |
|---|---|---|---|---|
| T=1, d_out=4096 | 128 | (32, 1) | **32** | **25%** |
| T=1, d_out=11008 | 128 | (86, 1) | 86 | 67% |
| T=1, d_out=14336 | 128 | (112, 1) | 112 | **88%** |
| T=1, d_out=28672 | 128 | (224, 1) | 224 | 175% (2 waves) |
| T=4, d_out=4096 | 128 | (32, 1) | 32 | 25% |
| T=16, d_out=4096 | 128 | (32, 1) | 32 | 25% |

**The red rows above are exactly the rows stuck at 1.02× in Step 1.**

The worst case `T=1, d_out=4096, d_in=4096`:
- 32 programs → 1 wave of 32 CTAs on 32 SMs → **96 SMs idle the entire kernel**
- Even 100% boost clock cannot hide this waste; compute must wait on HBM latency

### 1.3 HBM bandwidth sanity check

For `T=1, d_out=4096, d_in=4096`:
- Weight W4 read: `4096 × 2048 × 1 B = 8 MiB` (single pass)
- Activation X: `1 × 2048 × 1 B = 2 KiB` (L1-resident after first load)
- Output Y: `4096 × 1 × 2 B = 8 KiB`
- Total HBM: ~8 MiB per kernel call

At the observed **59 µs**, effective BW = `8 / (59e-6) = 135 GB/s` = **~13% of 1008 GB/s roof**.

**This confirms it's not HBM-bound. It's SM-idle-bound.** Split-K is the structural fix.

---

## 2. Design: Split-K for UINT4×SINT4 with per-group scales

### 2.1 The challenge: per-group dequantization

Standard Split-K is trivial for plain FP16 GEMMs because `C = Σ A_k × B_k` is linear and each split accumulates in FP32 atomics, then reduces.

Our kernel is **not** `Σ W_low × X_s4`. The epilogue per K-group is:

```
Y_group[m, n] = scale_x[n] * (
    (dot(W_low[m, g*BCOL:(g+1)*BCOL], X_s4[n, g*BCOL:(g+1)*BCOL])
     - zero_u4[m, g] * sum_X[n, g])
    * scale_u4[m, g]
)
Y[m, n] = Σ_g Y_group[m, n]
```

The per-group `(dot - zero*sum) * scale_u4` happens **inside** the K-loop and produces a **group-dequantized FP32 partial** that is then summed across groups.

**Good news**: because the group-epilogue is *also* linear in `(dot - zero*sum)`, and the `Σ_g` outer sum is plain FP32 addition, we can split the outer `Σ_g` across K-blocks **and** accumulate in FP32 safely. The `scale_x[n]` is shape `(T,)` so it's a scalar per output column — we keep it outside the K-loop (apply once at the reduce step).

### 2.2 Two-kernel plan (preferred)

**Kernel A — `dense_gemm_u4_s4_splitk_kernel`**

```
grid = (cdiv(d_out, BM), cdiv(T, BN), SPLIT_K)

# Each program handles K_SLICE = cdiv(n_groups, SPLIT_K) groups
for g_local in range(K_SLICE):
    g = split_id * K_SLICE + g_local
    if g >= n_groups: break
    # identical inner loop to existing kernel, 1 K-iter = 1 group
    ... dequant with scale_u4[m, g], zero_u4[m, g], sum_X[n, g] ...
    y_partial_acc += y_group        # FP32

# Write partial to shared 3D buffer:
Y_partial[split_id, m, n] = y_partial_acc        # FP32
```

**Kernel B — `_splitk_reduce_kernel`**

```
grid = (cdiv(d_out, BM_R), cdiv(T, BN_R))

y = 0.0
for split_id in range(SPLIT_K):
    y += Y_partial[split_id, m, n]
y_out[m, n] = (y * scale_x[n]).to(fp16)         # (T, d_out) output
```

**Variant for `_to_out` path**: kernel B writes transposed, directly into `(T, d_out)` (analogous to Step 1).

### 2.3 Single-kernel atomic variant (backup)

Instead of Kernel B, use `tl.atomic_add` on an FP32 output buffer and then a lightweight `scale_x` apply + fp16 cast. This saves one kernel launch but:
- atomic add FP32 on Hopper/Ada is decently fast (~cache line throughput) but still contentious
- the `scale_x[n]` apply still needs a second kernel (or fused into atomic epilogue with lock)

**Decision**: start with the **two-kernel plan** (clearer semantics, easier to autotune, negligible launch overhead at T=1: ~5 µs extra on top of a kernel we're trying to make faster by 20-40 µs). Revisit atomic variant only if the reduce kernel >10% of total.

### 2.4 Choosing `SPLIT_K`

| `d_in` | `n_groups = d_in/128` | candidate SPLIT_K | grid for T=1, d_out=4096, BM=128 |
|---|---|---|---|
| 4096 | 32 | 2, 4, 8 | 64 / 128 / 256 programs |
| 11008 | 86 | 2, 4, 8 | 64 / 128 / 256 programs |
| 14336 | 112 | 2, 4, 8 | 64 / 128 / 256 programs |

**Constraint**: each split must cover ≥1 group, and preferably ≥2 groups to amortize fixed per-program startup cost. With `d_in=4096, n_groups=32`, `SPLIT_K=8` → 4 groups per split, still healthy.

**Autotune grid for SPLIT_K**: expose `{2, 4, 8}` as a tunable constant and let Triton pick per shape.

### 2.5 Memory cost of `Y_partial`

`Y_partial: (SPLIT_K, d_out, T) fp32`

For worst case `SPLIT_K=8, d_out=28672, T=16`: `8 × 28672 × 16 × 4 B = 14 MiB`. Fits in L2 (72 MiB on 4090) and also fits HBM write bandwidth. Negligible.

For decode typical `SPLIT_K=4, d_out=4096, T=1`: `4 × 4096 × 1 × 4 = 64 KiB`. Stays in L2.

### 2.6 Data layout for the reduce pass

`Y_partial[split_id, m, n]` — split_id is the outermost axis so each split's writes are contiguous FP32 blobs (good write coalescing per program).

Reduce kernel reads `SPLIT_K` FP32 values from stride `d_out*T*4` apart per `(m, n)`. For SPLIT_K=8 that's 8 strided FP32 loads per element — fine with L2 hits.

---

## 3. Bit-exactness and numerical considerations

The existing kernel accumulates group contributions as:

```
y_f32 += (dot_i32 - zero_fp * sum_fp) * scale_fp
```

Split-K reorders the `Σ_g` sum. FP32 addition is **not associative**, so the output will differ from the non-split kernel by up to `n_groups × ULP(final_value)`. For typical magnitudes in LLM activations (~1e-1 to 1e1), this is `~1e-5` absolute.

**Validation plan**: reuse `test_dense_gemm_to_out.py` test harness but relax the assert to `atol=1e-3, rtol=1e-3` (same tolerance as `test_fused_dense_sparse`). The existing bit-exact test becomes a strict sanity test against a non-split reference.

---

## 4. Implementation plan (4 sub-steps, each with go/no-go)

### Step 4.1 — Standalone split-K kernel + correctness test (half-day)
- New file `triton_kernel/dense_gemm_splitk.py`
- Kernel A + Kernel B (straight write, `(d_out, T)` output for now — to match plain dense)
- Autotune: `SPLIT_K ∈ {1, 2, 4, 8}` × existing BM/BN grid
- `SPLIT_K=1` must be **bit-exact** with plain dense (this is our canary)
- Test: `test_dense_gemm_splitk.py`, all 25+ shapes pass with `atol=1e-3`

**Go/no-go**: Does any `SPLIT_K>1` configuration **actually get autotuned-in** on decode shapes? If Triton heuristic always picks `SPLIT_K=1`, autotune grid is wrong — stop and fix.

### Step 4.2 — Decode-shape microbench (half-day)
- `bench_dense_splitk.py`: same shape list as `bench_dense_to_out.py`, 3-way comparison (plain / Step1-fused / Step4-splitk)
- Measure per-shape `SPLIT_K` chosen by autotune (print it)
- Expected: `T=1, d_in=4096, d_out=4096` drops from 59 µs → ~35–40 µs (1.5×–1.7×)

**Go/no-go**: If the worst-case shape doesn't gain at least **1.3×** over Step 1, the SM-occupancy thesis is wrong — pivot to P2 (pure-CUDA GEMV, path 2 in ROI table) instead.

### Step 4.3 — Fuse split-K with transpose-to-out (half-day)
- Kernel B writes directly into `(T, d_out)` layout (transposed reduce)
- Replaces Step 1's `dense_gemm_u4_s4_to_out` for the hp=0 decode path
- Bit-exactness not required (Step 4.1 already proved); atol=1e-3 enough

**Go/no-go**: Transposed-reduce kernel within 2 µs of non-transposed-reduce? If not, either keep a separate transpose pass or use Triton's `tl.atomic_add` reduce.

### Step 4.4 — Wire into `v9_linear.py` decode hp=0 path
- Replace the `dense_gemm_u4_s4_to_out` call in `_v9_forward_decode` with the split-K version
- Re-run full `test_end2end.py` + `test_prefill_decode_dispatch.py` + `test_v9_linear_graph.py`
- Re-run `sweep_v9.py`, confirm:
  - decode shapes improved ≥ 1.3× vs prior sweep
  - **no** shape (including prefill) regressed > 3% (prefill hp=0 is protected since we only touch decode)

**Go/no-go**: Commit + tag `p4-step4-splitk` if all above pass; else hold code behind a feature flag.

---

## 5. What we are explicitly NOT doing in P4

| Not doing | Why |
|---|---|
| Split-K on prefill path | Prefill already has grid ≥ 256 programs; SM-full, adding split-K only costs HBM atomics |
| Split-K with sparse (hp>0) | Sparse kernel is a separate pipeline; we keep it out of scope. Revisit in P5 |
| Atomic-add variant | Two-kernel is simpler; revisit only if reduce > 10% of total |
| Change `BK ≠ BCOL_K` | Per-group dequant locks `BK=128`; this is a hard invariant from `pack_utils.py` |
| CUDA Graph integration | Already available via `V9LinearCudaGraph` wrapper; works unchanged because split-K kernel still has static shapes |

---

## 6. Risk analysis

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Triton autotune never picks `SPLIT_K>1` | med | blocks whole P4 | Force-prune configs to SPLIT_K>1 for small-grid shapes; if still lost, fall back to manual `SPLIT_K` selection on T |
| Reduce kernel eats the win | low | cuts win by half | Fuse reduce with transpose (Step 4.3); worst case add it back as optional flag |
| FP32 partial buffer allocation overhead | very low | ~1-2 µs | Pool it via `torch.empty` outside hot loop once `(SPLIT_K, d_out, T)` shape is known |
| Bit-exact tests fail on edge cases | low | cosmetic | Explicitly downgrade tolerance and document why |
| Split-K loses on `d_out=28672, T=16` (already 2-wave) | med on that shape | regression on 1 shape | Autotune `SPLIT_K=1` is always in the config list → falls back to plain path for this shape |

---

## 7. Success criteria (exit test)

After Step 4.4 is merged, rerun `sweep_v9.py` and compare against `sweep_20260423_144232.csv`:

| tier | current vs FP16 | P4 target |
|---|---|---|
| decode hp=0 (14 shapes, T≤16) | 0.65× | **≥ 0.85×** |
| decode hp>0 (42 shapes, T≤16) | 0.47× | ≥ 0.55× (partial, sparse path unchanged) |
| prefill any hp | 0.66–0.94× | **no regression > 3%** |

Headline number to chase: `T=1, d_out=4096, d_in=4096, hp=0` — this is the highest-frequency shape in real LLM serving; going from **~135 µs** v9_total → **~95 µs** would push vs-FP16 from 0.65× to ~0.92×.

---

## 8. File manifest (what Step 4.1–4.4 will add)

```
triton_kernel/
├── dense_gemm_splitk.py                    # new (Step 4.1, 4.3)
├── tests/
│   ├── test_dense_gemm_splitk.py           # new (Step 4.1)
│   └── test_end2end.py                     # existing, re-run in Step 4.4
└── benchmarks/
    └── bench_dense_splitk.py               # new (Step 4.2)

research/
└── p4_splitk_dense_design.md               # this file
```

No existing file is modified in Steps 4.1–4.3. Step 4.4 touches `v9_linear.py` in the decode hp=0 branch only (the same 5-line block we just added for P3).

---

## 9. Appendix — sanity math for the headline shape

`T=1, d_out=4096, d_in=4096, hp=0`:

| quantity | plain | splitK=4 | splitK=8 |
|---|---|---|---|
| grid programs | 32 | 128 | 256 |
| SMs utilised | 32 | 128 | 128 (2 waves) |
| SM util | 25% | **100%** | 100% |
| HBM reads (W4) | 8 MiB | 8 MiB (same, split reads partition) | 8 MiB |
| HBM writes (Y_partial FP32) | 8 KiB (fp16) | 64 KiB | 128 KiB |
| reduce kernel HBM read | 0 | 64 KiB | 128 KiB |
| reduce kernel HBM write | 0 | 8 KiB (fp16) | 8 KiB |
| Total HBM | 8.01 MiB | 8.14 MiB (+1.6%) | 8.27 MiB (+3.4%) |

HBM overhead is negligible. The win comes from **4× more programs finishing the same HBM fetch in parallel**, not from doing less work.
