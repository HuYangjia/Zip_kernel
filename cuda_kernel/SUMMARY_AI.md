## `cuda_kernel` Optimization Summary (AI Technical Reference)

> W4A4 quantized GEMM / Sparse-GEMM / Fused kernel optimization log
> Hardware: RTX 4090 (SM89, Ada Lovelace) / torch 2.8.0+cu126 / triton 3.4.0
> Baseline: cuBLAS FP16 `torch.matmul`
> Period: 2026-04-24 (Round 8 -> Round 18)

This document is the AI-facing contract for the optimization state:
precise shapes, kernel launch parameters, dispatch rules, numerical
contracts.  All English, all exact.  For narrative / rationale see
[`SUMMARY_HUMAN.md`](./SUMMARY_HUMAN.md).

---

### 0. Update addendum — R19 → R47 (2026-04-28)

> The rest of this document (sections 1-8) documents the **R8-R18**
> snapshot verbatim.  The state below is a drop-in delta; read it
> first, then treat sections 1-8 as historical background.

**Extended period**: 2026-04-24 (R8) → **2026-04-28 (R47)**.

**Authoritative bench**: `logs/qwen3_iter_round10/bench.json` for E2E
(625 records, 125 shapes), `logs/qwen3_iter_round11_v3/bench.json` for
`dense_gemm` sub-kernel (RTX 4090, harness from memory [[bmmiahpl]]).

**Rounds landed since R18** (see [`VALIDATION_LOG.md`](./VALIDATION_LOG.md) for full detail per round):

| Round | Scope                                          | Verdict      |
|------:|------------------------------------------------|--------------|
| R19   | ldmatrix for MMA A-operand                     | REJECTED     |
| R20-30| (epilogue refactors, wave-aware `kBn` dispatch, dense `kGrpBuf=128` opt-in, etc. — superset documented in VALIDATION_LOG R20-R30) | mixed; R20/R21/R22/R27/R31/R32 kept |
| R31   | dense `kGrpBuf=128` opt-in for `d_in > 4096`   | ACCEPTED     |
| R32   | dispatcher extension `T ≤ 32 && d_out ≤ d_in → kBn=8` | ACCEPTED |
| R33   | multi-CTA activation_quant split (T=1..4)      | REJECTED     |
| R34   | Split-K along group axis for dense_gemm        | REJECTED     |
| **R35** | **decode `kMaxGroups` 128 → 160** (unlocks 14B `down_proj` T=1) | **ACCEPTED** |
| R36   | fused `kGrpBuf` 32 → 40                        | REJECTED     |
| R37   | fused `pick()` force `kBn=8` for T≤16 wide shapes | REJECTED     |
| R38   | shared `robust_kernel_time` harness + per-run `--out-root` | ACCEPTED |
| R39   | HEAD-state re-baseline + bottleneck re-lock    | BASELINE SNAPSHOT |
| **R40-B** | `dense_gemm_mma_int4` `kBm=64` opt-in (dense-only; `T∈[16,64] && d_out≤2048`) | **ACCEPTED** |
| R41-P1| `fused_dense_sparse_mma_int4` `kBm` template (hp=0 only, infra for R42) | infra-only |
| **R42-P1** | fused `kBm=64` opt-in for hp>0 (BSR remap: 2 CTAs/row) | **ACCEPTED**; +14..+34% on 7 hit shapes |
| **R43** | fused (T, d_out) matrix gate (9 extra hit shapes, +5..+17%) | **ACCEPTED** |
| **R44** | fused kBn demote for kBm=64 & T∈[32,96] + gate expansion | **ACCEPTED**; unlocks d=2048 T∈[48,64] & d=3072/4096 T∈[48,64] |
| **R45** | fused gate wave-threshold off-by-one fix (`< 64` → `<= 64`) | **ACCEPTED**; unlocks T∈[48,64] × d_out=4096 (+15% on T=48 d=4096, +4.4% on production bat_T64_4k_4k) |
| **R46** | dispatcher `_forward_decode` switches hp>0 path to single `fused_dense_sparse` kernel | **ACCEPTED 🔥🔥**; E2E **-14% to -40%** across every decode/batch shape (biggest single round since R19) |
| **R47** | `backend/policy.py::_auto_policy` recalibrated to R46 evidence (every kernel → CUDA on every T) | **ACCEPTED**; `auto/cuda = 1.000x` on 8/9 shapes, `auto/triton = 1.77x..4.14x` — every production call now hits the CUDA fast path (previously all T≥8 mid-narrow shapes fell back to Triton) |

**Delta to sections 2-3 (current HEAD dispatch table)**:

1. `ops.py::_DECODE_MAX_GROUPS = 160` (was 128).  This is the **only**
   runtime-configurable dispatch constant that changed in 2026-04-27.
2. Three decode kernels carry `constexpr int kMaxGroups = 160` at
   source level (was 128): `dense_gemv_decode.cu`,
   `fused_gemv_decode.cu`, `fused_quant_gemv.cu`.
3. `dense_gemm_mma_int4.cu`: **dispatcher** bucket extended
   `T ≤ 32 && d_out ≤ d_in → kBn=8` (R32).  `kGrpBuf ∈ {32, 128}`
   opt-in (R31): `n_groups ≤ 32 → 32`; `n_groups ≤ 128 → 128`
   (dynamic shmem opt-in).  **R40-B**: `kBm ∈ {128, 64}` now a
   template parameter (was `constexpr 128`); `kBm=64` opt-in under
   tight gate (next bullet).
3a. **R40-B dense_gemm kBm dispatch gate**:
    ```
    kBm64_enabled =
      (T >= 16 && T <= 64)
      && (d_out <= 2048)
      && (ceil(d_out/128) * ceil(T/32) < 64)
    ```
4. **R42-P1 / R43 / R44 fused_dense_sparse kBm dispatch gate**
   (applies to `fused_dense_sparse_cuda_int4`; mirrors dense but
   now supports hp>0 via BSR 2-CTA remap):
    ```
    r44_shape_ok =
         (T <= 8  && d_out <= 4096)
      || (T <= 32 && d_out <= 3072)
      || (T in [48, 64] && d_out <= 4096)
      || (T == 96 && d_out <= 2048)
    kbm64_gate =
         r44_shape_ok
      && (ceil(d_out/128) * ceil(T/32) <= 64)   // R45: was `< 64`
    ```
    When enabled, grid.x doubles (`ceil(d_out/64)`), `blockDim.x`
    halves to 64 (2 warps).  Sparse branch uses
    `bsr_br = br / 2; half_row_off = (br & 1) * 64` remap so two
    CTAs share one BSR row block.  6 new template instances
    (`kBn ∈ {8, 32, 64}` × `kBm ∈ {64, 128}` × fused-specific).
4a. **R44 kBn demote rule** (inside `launch_for_kbn()` for fused
    kernel only): after the wave-health auto-pick, if
    `kbm_pick == 64 && T ∈ [32, 96] && kbn_pick >= 32`, force
    `kbn_pick = 8`.  This unwinds the artifact where
    `n_cta_m * ceil(T/32) = 64` threshold flips `kBn` one step too
    wide at kBm=64.
5. Env overrides (debug only):
    - `HKUST_V9_FUSED_FORCE_KBM ∈ {"64","128"}` force fused kBm
    - `HKUST_V9_FUSED_FORCE_KBN ∈ {"8","32","64"}` force fused kBn
    Both take precedence over the auto gate; unset defaults use gate.
6. `fused_dense_sparse_mma_int4.cu`: dispatcher mirrors dense but
   extended as (4) above.  `kGrpBuf = 32` fixed (R36's 40-bump
   rejected, re-confirmed by R43/R44 sweeps).
7. Split-K sources **removed** from `dense_gemm_mma_int4.cu` (R34).
8. R33's `act_quant_phase_a_max` / `_b_pack` sources remain on disk
   but are unwired from `ops.py`.

**Authoritative R42-R44 wins (hp=0.05, RTX 4090, d_in=4096, auto-gate vs forced kBm=128)**:

| Shape                     | R42  | R43  | R44  | Notes                          |
|---------------------------|------|------|------|--------------------------------|
| d=1024 T=8                | .    | +16% | +16% | new in R43 (T<=8 all d_out)    |
| d=4096 T=8                | .    | +13% | +13% | new in R43                     |
| d=1024 T=64               | .    | +17% | +17% | new in R43 (d<=1024 T<=96)     |
| d=3072 T=16               | .    | +15% | +15% | new in R43 (T<=32 d<=3072)     |
| d=2048 T=48               | ×(0.82) | × | **+7%** | new in R44 (kBn demote fixes) |
| d=2048 T=64               | ×(0.82) | × | **+7%** | new in R44 (kBn demote fixes) |
| d=3072 T=48               | ≈    | +2% | **+20%** | R44 kBn demote unlocks     |
| d=3072 T=64               | ≈    | +4% | **+19%** | R44 kBn demote unlocks     |
| d=4096 T=48               | ≈    | ≈   | **+15%** | R44 kBn demote unlocks     |
| d=2048 T=96               | .    | ≈   | **+18%** | R44 new                    |

**Authoritative pure-CUDA / pure-Triton numbers (memory [[0d5nyof1]])**:

Qwen3-14B `down_proj [17408→5120]`, T=1 — the flagship R35 win:

```
fp16    e2e   189.2 us
triton  e2e   427.7 us
cuda    e2e    86.9 us       cuda/fp16 = 2.18x,  cuda/triton = 4.92x
```

125-shape aggregate (median / p95-worst against Triton):

| T     | cuda/triton median | cuda/triton p95 worst | cuda/fp16 median |
|-------|--------------------|------------------------|------------------|
| 1     | 6.5x               | 13.5x                  | 1.85x            |
| 16    | 2.6x               | 5.3x                   | 0.33x            |
| 128   | 2.1x               | 5.1x                   | 0.47x            |
| 512   | 1.9x               | 2.9x                   | 0.85x            |
| 2048  | 1.8x               | 1.9x                   | 1.15x            |

CUDA beats Triton on **every one** of 125 shapes.  CUDA beats FP16
on all T=1 shapes and ~60 % of T=2048 shapes.

**Live bottleneck list (supersedes section 6)**:

1. `fused_dense_sparse` wave starvation at `T ∈ [16, 128], d_out ≤ 4096`
   — next lever is **kBm=64** (gated), *not* Split-K (R34 disproved
   K-axis split) and *not* forced `kBn=8` (R37 disproved wider-shape
   kBn shrink).
2. `activation_quant` 14 us launch floor at `T ∈ [1, 512]` — next
   lever is **fuse quant INTO fused_dense_sparse's prologue**; the
   multi-CTA two-kernel approach (R33) is dead.
3. `gate_up_proj` on 14B at T=2048 — 0.77 x FP16, true compute-bound
   INT4 on SM89; levers are epilogue FP32 op repack or register-
   spill reduction only.

---


- **BCOL = 128**: single quantization group spans 128 s4 columns.
  `d_in % BCOL == 0` required.
- **W4A4**: W stored as packed s4 (`(d_out, d_in/2) uint8`, `pack_s4_le`);
  X quantized per-row per-group (`(T, d_in/2) uint8`, scale `(T,) fp16`,
  sum `(T, n_groups) int32`).
- **Weight scale/zero format**: fp16 per-row per-group
  `scale_u4, zero_u4: (d_out, n_groups)`.
- **Accumulator**: int32 from MMA / dp4a, promoted to fp32 for the
  `(acc - zero * sum_X) * s_u4 * s_x` epilogue, stored back as fp16.
- **Parity tolerance**: abs<=1e-3 vs Triton reference on fp16 scale
  magnitudes in {0.001, 0.05} range.

---

### 2. Current kernel inventory (R18)

| binary symbol | source | role | dispatch predicate |
|---|---|---|---|
| `activation_quant_cuda` | `activation_quant/activation_quant.cu` | standalone quant | always |
| `dense_gemv_cuda_decode` | `dense_gemm/dense_gemv_decode.cu` | T=1 dense GEMV (dp4a) | `T == 1` |
| `dense_gemm_cuda_int4` | `dense_gemm/dense_gemm_mma_int4.cu` | T>=2 dense MMA | `T >= 2` |
| `sparse_gemm_cuda_int4` | `sparse_gemm/sparse_gemm_mma_int4.cu` | all-T sparse MMA | always (sparse) |
| `fused_gemv_cuda_decode` | `fused_dense_sparse/fused_gemv_decode.cu` | T=1 fused GEMV | (internal) |
| `fused_quant_gemv_cuda_decode` | `fused_dense_sparse/fused_quant_gemv.cu` | T=1 quant+GEMV fused | `T == 1` (e2e) |
| `fused_dense_sparse_cuda_int4` | `fused_dense_sparse/fused_dense_sparse_mma_int4.cu` | T>=2 fused MMA | `T >= 2` |

Archived (present on disk, excluded from build):
- `dense_gemm_mma_int8.cu`, `sparse_gemm_mma_int8.cu`, `fused_dense_sparse_mma_int8.cu`
  -- R11 experiment concluded INT4 MMA is ~1.9x faster than INT8 MMA on SM89.
- `fused_gemv_smallT.cu` -- R16 experiment concluded dp4a GEMV loses to
  MMA at T=2..16.

---

### 3. Launch parameter contract (MMA kernels)

All three MMA kernels (`dense_gemm_mma_int4`, `sparse_gemm_mma_int4`,
`fused_dense_sparse_mma_int4`) share:

```
constexpr int kBm       = 128;         // M-tile per CTA
constexpr int kBk       = 128;         // one group per K-tile (kBk == BCOL)
constexpr int kMmaK     = 64;          // s4 MMA K dimension
constexpr int kKSteps   = kBk / kMmaK; // = 2
constexpr int kMsubPerWarp = 2;        // 2 * m16n8k64 per warp in M
dim3 block(128, 1, 1);                 // 4 warps / CTA
dim3 grid(ceil_div(d_out, kBm), ceil_div(T, kBn), 1);
```

`kBn` is the only free template parameter.  R18 dispatch table:

| kernel | predicate | kBn |
|---|---|---:|
| dense_gemm_mma_int4, fused_dense_sparse_mma_int4 | `T <= 8` | 8 |
|                                                  | `T <= 128` | **32** (R18 bucket) |
|                                                  | otherwise | 64 |
| sparse_gemm_mma_int4 | `T <= 8` | 8 |
|                      | `T <= 96` | 32 (R17 bucket; R18 extension regressed sparse) |
|                      | otherwise | 64 |

Dispatch performed at `launch()` C++ level via
`std::integral_constant<int, KBn>` tag-dispatch; all three kBn values
are instantiated at compile time (see the bottom of each `.cu` file).

---

### 4. ptxas footprint (R12 post-cap; unchanged through R18)

| kernel | kBn | regs | smem | spill |
|---|---:|---:|---:|---:|
| dense_int4 | 64 | 166 | 41 KB | 0 |
| dense_int4 | 32 | 165 | 37 KB | 0 |
| dense_int4 | 8 | 72 | 34 KB | 0 |
| sparse_int4 | 64 | 170 | 25 KB | 0 |
| sparse_int4 | 32 | 130 | 21 KB | 0 |
| sparse_int4 | 8 | 60 | 17 KB | 0 |
| fused_int4 | 64 | 170 | 42 KB | 0 |
| fused_int4 | 32 | 166 | 37 KB | 0 |
| fused_int4 | 8 | 80 | 34 KB | 0 |

Zero register spill across the board.  All variants fit within the
64 KB shared memory budget per SM89 SM.

---

### 5. Benchmark table (bench_20260424_183142.md)

End-to-end v9_linear, auto-dispatch, cuBLAS FP16 baseline:

| shape | FP16 us | CUDA us | ratio |
|---|---:|---:|---:|
| dec_T1_4k_4k    | 16.42 |  19.80 | 0.83x |
| dec_T1_4k_11k   | 93.98 |  45.45 | **2.07x** |
| dec_T1_11k_4k   | 94.99 |  48.54 | **1.96x** |
| dec_T8_4k_4k    | 14.93 |  61.01 | 0.24x |
| dec_T16_4k_4k   | 16.22 |  81.82 | 0.20x |
| bat_T64_4k_4k   | 19.13 |  92.75 | 0.21x |
| bat_T128_4k_4k  | 33.10 |  92.26 | **0.36x** |
| pre_T512_4k_4k  | 110.10 | 156.55 | 0.70x |
| pre_T1024_4k_4k | 212.87 | 289.28 | 0.74x |

---

### 6. Next optimization directions (ranked by ROI)

Each directive includes: **target shape, expected gain, risk class,
implementation sketch**.

#### 6.1 [ATTEMPTED R19, NO GAIN] ldmatrix for MMA kernels -- target T>=512

- **Target shapes**: `pre_T512_4k_4k` (0.70x -> expected 0.95x-1.05x),
  `pre_T1024_4k_4k` (0.74x -> expected 1.0x-1.1x), and by transitivity
  all T>=128 kBn=64 path.
- **R19 result**: **negative**.  Replaced 8-scalar-uint32 A loads with
  2 `ldmatrix.x4.shared.b16` per `ks`.  Parity 10/10 passed; wall time
  regressed +3% at T=64/128 and +4% at T=1024, all other T unchanged.
- **Explanation**: the existing A load pattern
  `sW[msub + lane/4][kpb + (lane&3)*4]` is already maximally warp-
  coalesced into a single `LDS.32` per 4-byte word (lanes 0..3 of each
  quad read adjacent words in the same row).  ldmatrix would be an
  improvement only if the operand were staged in a banked layout that
  otherwise required bank-conflict resolution -- not our case.
- **Remaining sub-direction (untested)**: `ldmatrix.x2.trans.shared.b16`
  for B operand.  Current B-load is `kNsubPerCta` = 8 scalar reads per
  lane per ks (for kBn=64), and the shmem layout `sX[n_row][col]` is
  not optimally coalesced because `n_row = in_sub*8 + lane/4` means
  different in_sub values spread lanes across different n_row strides.
  Replacing with ldmatrix.x2.trans may yield a true reduction in LDS
  transactions.  Not attempted because the trans-variant address
  arithmetic is subtle and R19.A negative result lowered priority.

#### 6.2 [HIGH-ROI, MEDIUM-RISK] cp.async double-buffer for MMA kernels

- **Target shapes**: all T>=8 MMA shapes (T=8 0.25x, T=16 0.20x, T=64
  0.21x, T=128 0.36x).  Expected each +10-25% as the W/X HBM->shmem
  staging overlaps MMA compute.
- **Why it works here**: current kernels are single-buffered
  (R12 explicitly reverted the R6 cp.async pattern to fit shmem budget
  after shmem-caching scale_u4/zero_u4).  At MMA T=8 the MMA instruction
  throughput is unlocked but compute can't start until sW/sX staging
  completes for the current group -- 60-cycle latency bubble per group.
- **Implementation sketch**:
  1. Define `sW[2][kBm][BCOL/2]` and `sX[2][kBn][BCOL/2]` double-buffered
     shmem (add 8-16 KB -- still fits in 64 KB budget after dropping the
     32-group scale/zero cache to a 16-group cache, or splitting the
     staging loop across groups more carefully).
  2. Prologue: issue cp.async for group 0 into bank 0.
  3. Per-group body:
     - `cp_async_cg_16` group `g+1` into bank `(g+1)&1`
     - `cp_async_commit; cp_async_wait_group<1>; __syncthreads()`
     - Do MMA chain on bank `g&1`
- **Risk**: **medium-high**.  Main risk is shmem budget clash with the
  shmem-cached scale_u4/zero_u4; may need to make the scale cache
  predicated on `n_groups <= 16` instead of `<= 32`.  Also, register
  pressure may climb (the MMA frag registers + the cp.async
  bookkeeping).  Monitor ptxas output; if regs > 170 for kBn=64 and
  spill appears, pull back.

#### 6.3 [MEDIUM-ROI, LOW-RISK] quant fusion into MMA kernel for T=2..8

- **Target shapes**: e2e T=1..8 for non-square shapes that currently
  pay `activation_quant` as a separate kernel.  T=1 4k->4k already
  uses `fused_quant_gemv` (R15); T=2..8 does not.
- **Why it works here**: at T=2..8 the activation_quant kernel is
  still launch-overhead-bound (5us per launch on 4090).  Fusing into
  the MMA kernel removes one launch.
- **Implementation sketch**: new kernel
  `fused_quant_mma_smallT.cu` that mirrors R15's `fused_quant_gemv`
  but routes to MMA instead of dp4a in phase B.  Phase A (cooperative
  max-abs + quant+pack+sum) reuses R15c's design with all kBm warps.
- **Risk**: **low** (R15c already validated the quant-cooperation
  pattern; parity-test against R14 fused_gemv_decode on T=1 gave
  bit-exact match).
- **Expected gain**: e2e T=2..8 +5-10% (5us savings on ~80us total).

#### 6.4 [LOW-ROI] kBn=16 instantiation

- **Target shapes**: T=12..24 on dense/fused MMA path.
- **Analysis**: T=12 kBn=16 grid.y=1 vs T=12 kBn=32 grid.y=1;
  kNsubPerCta=2 vs 4, MMA instruction count halves.
  But T=12..24 are not in the e2e bench (we only have T=8, 16), and
  T=16 already on kBn=32 path takes 81us -- reducing MMA count by 2x
  would at best take it to ~60us (0.27x FP16), still dominated by
  shmem-LSU occupancy rather than Tensor Core throughput.
- **Expected gain**: 10-15% at T=16 (0.20x -> 0.23x).
- **Deferred reason**: does not move the overall FP16 ratio by a
  meaningful amount; does not unlock any new winning shape.

#### 6.5 [LOW-ROI] BSR-indexed cp.async for sparse_gemm

- **Target shapes**: sparse T=8..64 (currently 0.75-0.80x).
- **Background**: the R7 attempt regressed sparse T>=64 because the
  cp.async bc_next lookup added register pressure.  With post-R12
  register budget (170 regs at kBn=64) there is now ~80-reg headroom
  before hitting 255; this may be feasible again.
- **Risk**: **medium**.  Same register-budget risk that killed R7.

#### 6.6 [EXPLORATORY, HIGH-RISK] Hopper-style warp-specialized staging

- Producer warps run cp.async, consumer warps run MMA.
- Expected to close the T=8..64 gap entirely.
- **Deferred reason**: SM89 has no TMA, so warp specialization has to
  fake a producer-consumer queue in shmem.  This is a full kernel
  rewrite (~1000 LOC); risk class "R&D experiment".

---

### 7. Profiling gap (known-unknown)

All optimization rounds from R8 onwards operated **without ncu /
nsight access** on the AutoDL execution host.  The diagnostic
methodology substituted was:
- T-sweep benchmarks isolating kernel wall-time as a function of T
- ptxas output for register / shmem / spill accounting
- `cuobjdump --dump-sass` for dp4a-vs-MMA instruction counting

If the next person has ncu access, **the single most useful
measurement is `smsp__inst_executed_pipe_tensor_core.avg.pct_of_peak_sustained_elapsed`**
on the `dense_T=512` shape.  A reading below 30% would confirm that
ldmatrix+cp.async (6.1, 6.2 above) is the correct next move; a
reading above 70% would indicate we are already near Tensor Core
peak and the gap to FP16 is purely architectural (FP16 uses
scheduler-friendly patterns that s4 MMA cannot).

---

### 8. File-level churn map (R8 -> R18)

| path | state | notes |
|---|---|---|
| `csrc/common/mma_utils.cuh` | NEW (R11) | ldmatrix + mma.s4/s8 PTX wrappers |
| `csrc/common/arch.cuh` | MOD (R6, R11) | cp.async helpers |
| `csrc/dense_gemm/dense_gemm_mma_int4.cu` | NEW (R11), MOD (R12, R17, R18) | main dense kernel |
| `csrc/dense_gemm/dense_gemv_decode.cu` | NEW (R13) | T=1 dp4a GEMV |
| `csrc/dense_gemm/dense_gemm_mma_int8.cu` | NEW (R11), UNUSED (R12) | archived |
| `csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu` | NEW (R11), MOD (R12, R17, R18) | |
| `csrc/fused_dense_sparse/fused_gemv_decode.cu` | NEW (R14) | |
| `csrc/fused_dense_sparse/fused_quant_gemv.cu` | NEW (R15) | e2e T=1 default |
| `csrc/fused_dense_sparse/fused_gemv_smallT.cu` | NEW (R16), NOT DISPATCHED | failed experiment |
| `csrc/fused_dense_sparse/fused_dense_sparse_mma_int8.cu` | NEW (R11), UNUSED (R12) | archived |
| `csrc/sparse_gemm/sparse_gemm_mma_int4.cu` | NEW (R11), MOD (R12, R17) | |
| `csrc/sparse_gemm/sparse_gemm_mma_int8.cu` | NEW (R11), UNUSED (R12) | archived |
| `csrc/bindings.cc` | MOD | 6 symbol exposures |
| `ops.py` | MOD | auto-dispatch by T |
| `benchmarks/bench_kernels.py` | NEW (R8) | fp16 baseline harness |
| `tests/test_parity.py` | MOD | all kernels parameterized |

End of document.
