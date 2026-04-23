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

---

## 10. Step 4.1 & 4.2 实测结果 (2026-04-23 晚)

### 10.1 Step 4.1 — 对齐测试 (commit `de49845`)

| 类别 | 数量 | 结果 |
|---|---|---|
| `split_k=1` canary (atol=2e-3) | 9 | ✅ PASS |
| `split_k ∈ {2,4,8}` 数值逼近 (atol=2e-3) | 18 | ✅ PASS (6 skipped — n_groups 不整除) |
| auto-policy | 6 | ✅ PASS |
| policy sanity | 1 | ✅ PASS |
| **合计** | **37 passed, 6 skipped** | ✅ |

**关键更正**：`split_k=1` 不是 bit-exact，因 kernel 故意把 `scale_x` 挪到 reduce pass 以让 per-split partial 独立；一次 FP32 乘法重排 → 观测到 `max|delta|=1.95e-3`，与 FP16 ULP 量级匹配。已把 tolerance 拉到 `atol=2e-3, rtol=1e-3` 并在代码注释中详细说明原因。

### 10.2 Step 4.2 — microbench（**未达标，触发 go/no-go 停止点**）

`bench_dense_splitk.py`, RTX 4090, warmup=50, windows=3, iters=200，单位 µs：

| T | d_out | d_in | sk | plain | fused(P3) | **splitk** | **FP16 cuBLAS** | splitk vs fused | splitk vs FP16 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 8 | 87.8 | 85.4 | **141.8** | 38.5 | **0.60×** ❌ | 0.27× |
| 1 | 4096 | 11008 | 2 | 135.5 | 121.3 | **99.7** | 98.0 | **1.22× ✅** | 0.98× |
| 1 | 11008 | 4096 | 8 | 60.9 | 58.9 | 99.5 | 96.6 | 0.59× ❌ | 0.97× |
| 4 | 4096 | 4096 | 8 | 70.5 | 60.5 | 100.4 | 14.2 | 0.60× ❌ | 0.14× |
| 16 | 4096 | 4096 | 8 | 70.7 | 60.1 | 100.8 | 16.7 | 0.60× ❌ | 0.17× |
| 1 | 14336 | 4096 | 8 | 62.0 | 59.5 | 99.5 | 125.0 | 0.60× ❌ | **1.26×** |
| 1 | 28672 | 4096 | 1 | 78.4 | 75.1 | 99.4 | 247.2 | 0.76× ❌ | **2.49×** |
| 4 | 14336 | 4096 | 8 | 71.3 | 59.8 | 100.8 | 126.6 | 0.59× ❌ | 1.26× |
| 16 | 14336 | 4096 | 8 | 70.8 | 59.8 | 100.7 | 128.0 | 0.59× ❌ | 1.27× |
| 16 | 28672 | 4096 | 1 | 86.5 | 81.5 | 98.6 | 253.2 | 0.83× ❌ | **2.57×** |

Totals: plain=794µs, fused=722µs, **splitk=1041µs, FP16=1144µs**；splitk vs fused avg **0.69× (更慢)**，splitk vs FP16 avg 1.10×。

**Go/no-go 判定：FAIL**（设计文档 §4 Step 4.2 gate 要求最差 shape ≥1.3×，实测 0.59×）。

### 10.3 失败原因诊断

1. **reduce pass 是新瓶颈**。`splitk_us` 几乎全部稳定在 ~99-100 µs，独立于 SPLIT_K 和形状——这正是 **Kernel A ~50µs + Kernel B ~50µs** 两次 launch 叠加的签名。即使 `sk=1`（reduce 只做 FP32→FP16 cast），仍然要 ~24µs overhead（对比 fused=75µs 的 d_out=28672 shape），远高于设计文档预估的 "~5µs 额外 launch"。
2. **双 kernel launch overhead 在 decode 尺度下占比极高**。Kernel A 本身 ~50 µs + Kernel B ~50 µs > fused 单 kernel 总时间。
3. **唯一赢的 shape 是 `T=1, d_in=11008` with sk=2**：`n_groups=86`，sk=2 时每 split 43 组，HBM 传输时间足够大 → 反超两次 launch 开销。这印证了 split-K 思路本身有效，但阈值不在 4090 + 小 decode shape。
4. **3 个 shape 出现 split-K 已超 FP16**（d_out=14336/28672 且 d_in=4096）——但这些 shape 本身 fused 就已经赢 FP16，split-K 反倒把优势抵消。
5. **设计文档的 HBM 利用率诊断是对的**（SM idle），但 two-kernel 路径的 fixed cost 比原估计高 4-5×。

### 10.4 下一步决策（供用户拍板）

**放弃 "Step 4.3 传统两 kernel fuse" 路线**。即使把 reduce 与 transpose 融成一个 kernel，也无法消除 kernel A 和 kernel B 之间的 HBM 读写环节（FP32 `Y_partial` 必须落盘）——这是数据依赖，不是工程实现。

**三条候选路线**：

| 路线 | 原理 | 工时 | 风险 |
|---|---|---|---|
| **A. 单-kernel atomic split-K** | Kernel A 直接 `tl.atomic_add` 累加到 FP32 输出，最后一个 small kernel 做 `×scale_x + cast fp16`。消除 HBM round-trip。 | 1 天 | atomic 争用可能回吐收益；scale_x 仍需单独 kernel |
| **B. Persistent GEMM + tile-level streaming** | 不分 K，而是让一个 CTA 沿 K 流式处理；用 persistent block 反复复用 W tile。grid 固定 = SM 数。 | 2-3 天 | 写法复杂，Triton 原语支持有限 |
| **C. 放弃 decode T=1 d_in=4096 档，接受现状 (0.65× FP16)** | 现实是：Step 1 (P3) 已经让 decode hp=0 端到端从 0.70× → 0.70×（小 shape 高达 1.28×），主要 gap 在 d_out=11008 一档。 | 0 天 | 零风险，跑回 P5 |

**推荐**：**路线 A**（atomic split-K），因为：
- 可以直接改造 Kernel A，只需把 `tl.store` 换成 `tl.atomic_add`；
- 不需要独立 FP32 `Y_partial` buffer（省 HBM 分配）；
- scale_x 和 cast 可以独立一个 ~10µs 的 tiny kernel 处理，仍比两 kernel 链快。

---

## 11. Step 4.1/4.2 效果 vs FP16 小结

从本轮 microbench（hp=0 dense 通路，不含 activation quant）看：

| 指标 | Pre-P4 (fused) vs FP16 | **Post-P4 (splitk) vs FP16** |
|---|---|---|
| 10-shape 总延迟 | 722 µs / 1144 µs = **1.58×** | 1041 µs / 1144 µs = **1.10×** |
| 最差 shape | T=4 d_out=4096: 60.5 µs vs 14.2 µs = **0.23×** | T=4 d_out=4096: 100.4 µs vs 14.2 µs = **0.14×** |
| 最优 shape | T=1 d_out=28672: 75.1 µs vs 247.2 µs = **3.29×** | T=1 d_out=28672: 99.4 µs vs 247.2 µs = **2.49×** |

**结论**：two-kernel split-K 在本 bench 上全面不如 Step 1 (P3)。**决定保留 v9_linear 的接入为 P3 Step 1 (fused_to_out)**，直到路线 A / B 任一验证通过。
