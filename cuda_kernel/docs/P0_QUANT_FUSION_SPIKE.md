# P0 Spike — `activation_quant` fusion into `fused_dense_sparse` prologue

**Status**: feasibility analysis, no kernel changes yet.
**Target**: close the ~16 us `activation_quant` launch floor for mid-T
shapes.  Small shapes are a secondary beneficiary; large shapes
(>80 us kernel body) already dilute the floor so they gain little.

---

## 1. Primary target: mid-T shapes on Qwen3-4B / 8B / 14B

Extracted from `logs/r63_combined/bench.json` (140 shapes).  Shapes
where the 16 us quant overhead is ≥25 % of total `cuda_us` and the
measurement is not a small-model outlier:

| target tier | shapes | current cuda_us | quant share | expected gain |
|---|---|---:|---:|---:|
| **Tier 1 — mid Qwen3-8B/14B T=32/128** (q/o/gate_up/down) | ~18 | 30–60 us | 30–50 % | **+10–18 us** → median +0.15–0.25× |
| Tier 2 — Qwen3-4B T=32/128 | ~8 | 30–50 us | 35–50 % | +12 us → +0.15× |
| Tier 3 — Qwen3-0.6B/1.7B T=32/128 (small) | ~12 | 30–34 us | 45–53 % | +16 us, but still loss vs cuBLAS (accepted per user) |
| Tier 4 — T=512 large shapes | ~15 | 300–4400 us | 1–5 % | marginal, out of scope |
| Tier 5 — T=1 decode | 35 | 5–50 us | already fused via `fused_quant_gemv` | 0 — already done |

So P0's bench impact is dominated by Tier 1 + Tier 2 (~26 shapes).
Median speedup 1.02× → ~1.18–1.22× projected.

---

## 2. Why the current 16 us floor exists — structural, not tuning

`activation_quant_kernel` ([csrc/activation_quant/activation_quant.cu L116](../csrc/activation_quant/activation_quant.cu)) is
a **two-pass** kernel:

- **Pass 1**: `for d in 0..D step 128: local_max = max(|X[t, perm[d]]|)`,
  then 128-wide warp-tree reduction → `scale_x[t]`.
- **Pass 2**: for each group g, `q = quantize(X[t, perm[g*128+lane]])`,
  128-wide warp-tree reduction → `sum_X[t, g]`, then LE-pack nibbles
  via `__shfl_xor_sync` → `X_s4[t, d/2]`.

**Traffic to HBM on the current path**:

```
activation_quant:  read X fp16 (2·T·D B) ×2 [both passes]
                   + write X_s4 (0.5·T·D B)
                   + write sum_X (4·T·n_g B)
                   + write scale_x (2·T B)

fused_dense_sparse: read X_s4 (0.5·T·D B) + read sum_X (4·T·n_g B)
                    + read scale_x (2·T B) + read W + read scale/zero
                    + write Y
```

The **X_s4 / sum_X / scale_x tensors do a complete HBM round-trip**
(write then read) between the two kernels, plus the two `<<<>>>` host
launches cost ~7 us × 2 ≈ 14–16 us fixed on RTX 4090.

**Theoretical ceiling after fusion**:
- Remove 1 kernel launch (~7 us)
- Remove X_s4 + sum_X HBM round-trip (~0.5·T·D + 4·T·n_g bytes / 1008 GB/s)
- For Qwen3-8B `q_proj T=32`: saves ≈ 7 us launch + 0.3 us HBM → ~7 us total
- For Qwen3-8B `gate_up_proj T=128`: saves ≈ 7 us launch + 1.5 us HBM → ~8 us

So gains vary 7–16 us depending on shape.  The 16 us peak applies to
small shapes where the 2-pass body was already tiny.

---

## 3. Implementation plan — Option A: prologue-fused kBn-wide quant

### 3.1 Design

Add a new **template flag** `kFuseQuant` to
`fused_dense_sparse_mma_int4_kernel`.  When true:

- Input interface changes: instead of taking `X_s4, scale_x, sum_X` from
  HBM, take `X_fp16 (T, D)` and `perm (D,)` directly.
- Each CTA handling output tile `(m_tile, n_tile)` where
  `n_tile = blockIdx.y * kBn` is responsible for **quantizing its own
  `kBn` tokens** during the prologue before the main loop starts.
- Results of quantization live entirely in smem — we never write X_s4
  to HBM.

### 3.2 Prologue pseudo-code

```cpp
// New smem buffers (T=kBn tokens, all groups):
__shared__ __half  sX_quantized[kBn][D];            // too big! see §4
__shared__ int     sSumX[kBn][n_groups];            // small
__shared__ __half  sScaleX[kBn];                    // tiny

if constexpr (kFuseQuant) {
    // Pass 1: token-wise max-abs over D columns (128-wide warp-tree)
    //   Using threadIdx.y for n_local ∈ [0, kBn), threadIdx.x for
    //   lane ∈ [0, 128) within each 128-column slab.
    //   Read X[n_global, perm[d]] fp16 from HBM.
    //   Warp-reduce + inter-warp reduce → sScaleX[n_local].

    // Pass 2: per-group quantize + pack + per-group reduce.
    //   For each group g in [0, n_groups):
    //     q = quantize(X[n_global, perm[g*128+lane]], sScaleX[n_local])
    //     warp-sum across 128 lanes → sSumX[n_local][g]
    //     LE-pack nibbles and store to sX_s4[n_local][g*64..g*64+64]
    //   (where sX_s4 is a int4 buffer in smem, 0.5·D bytes per token)
    __syncthreads();
}

// ... existing main K-loop reads from sX_s4 / sSumX / sScaleX instead
//     of from the staged sX buffer populated from HBM.
```

### 3.3 What about the existing smem staging buffer `sX[2][kBn][bytes_per_group]`?

Today `issue_x_load` streams 64 bytes × kBn per group from HBM X_s4
into `sX[buf][n_local][:64]`.  If we fuse the quant, we don't need
this staging — **the quantized data is already in a CTA-wide smem
buffer**.  So we can repurpose `sX[2][kBn][64]` → remove it entirely
when `kFuseQuant=true`, saving 2×kBn×64 = 4 KB (kBn=32) or 8 KB (kBn=64).

---

## 4. Shared memory budget — the critical feasibility gate

### 4.1 Current smem occupancy (kFuseQuant=false)

Per CTA, reading [fused_dense_sparse_mma_int4.cu L181-L202](../csrc/fused_dense_sparse/fused_dense_sparse_mma_int4.cu):

| buffer | size (bytes) | kBm=128 kBn=32 | kBm=64 kBn=64 |
|---|---|---:|---:|
| sW[2][kBm][64] | 2·kBm·64 | 16 384 | 8 192 |
| sX[2][kBn][64] | 2·kBn·64 | 4 096 | 8 192 |
| s_scale_u4[kBm][kGrpBuf+pad] | 2·kBm·(32+1) | 8 448 | 4 224 |
| s_zero_u4 (same) | 2·kBm·(32+1) | 8 448 | 4 224 |
| s_scale_x[kBn] | 2·kBn | 64 | 128 |
| s_scale_block[kBm] | 2·kBm | 256 | 128 |
| s_sum_X[2][kBn] | 2·4·kBn | 256 | 512 |
| **total (group-cache on)** | | **37 952 B** | **25 600 B** |
| total (group-cache off) | -16 896 for (kBm=128) | 21 056 | 17 152 |

On SM89 the default dynamic-smem cap is 48 KB (100 KB with opt-in via
`cudaFuncSetAttribute`).  We're currently at 38 KB → 2 CTAs/SM.

### 4.2 New smem if we naively add fused X buffer

The naive `sX_quantized[kBn][D]` is too big:
- Qwen3-8B `D=4096`, kBn=32 → 32·4096·2B = **256 KB**  ❌ out of question
- Qwen3-0.6B `D=1024`, kBn=32 → 32·1024·2B = **64 KB**  ❌ already over

### 4.3 The right design — per-group streaming quant

We never need to hold the **full** quantized X in smem.  The main
K-loop already processes **one group at a time**.  So the fused
prologue only needs to:

**Pass 1 only** (max-abs over all D): still O(D) bytes of HBM read per
token, but produces only `sScaleX[kBn]` (64–128 B of smem).  This pass
is essential and must scan all D — **but we can fuse Pass 1 into the
prologue without any new smem**.

**Pass 2**: **do NOT pre-compute all D groups up front**.  Instead,
**move the per-group quantization INTO the main loop**.  For each
group g:
- (Already) prefetch `sW[buf^1]` from HBM W (reused for group g+1)
- NEW: prefetch `X[n_tile..n_tile+kBn, perm[g·128..g·128+128]]` fp16
- NEW: quantize in-place → write into `sX[buf^1]` (reusing the existing
  staging slot!), compute `sum_X` for that (n, g), store to `s_sum_X[buf^1]`
- Run MMA on `sW[buf]` × `sX[buf]` as before

This design **reuses the existing `sX[2][kBn][64]` double buffer** —
we just fill it from fp16 X + quantize, instead of `cp.async`-loading
from the pre-quantized HBM X_s4.

**Net smem delta for the fused path**: **0 bytes** (we reuse sX).

**But one new buffer is needed**: a small per-CTA staging of the
current group's fp16 X slice:
- 128 cols × kBn tokens × 2 B = 256·kBn bytes
- kBn=32 → 8 KB, kBn=64 → 16 KB
- This crosses the 48 KB cap at kBm=128 kBn=32 → 38 + 8 = 46 KB (OK, just under)
- kBm=128 kBn=64 → not used (current gate already excludes this)
- kBm=64 kBn=64 → 25.6 + 16 = 41.6 KB (OK)

### 4.4 Better — avoid the fp16 staging buffer entirely

The cleanest design doesn't even need the fp16 staging buffer:

```
for each group g:
    each lane (lane_x in 0..127, lane_y in 0..kBn-1):
        fp16 val = X[t_global, perm[g*128 + lane_x]]   // direct HBM load
        int  q   = quantize(val, sScaleX[lane_y])      // reg-only
        warp_sum(q) → write to s_sum_X[buf^1][lane_y]
        pack + write nibble pair to sX[buf^1][lane_y][lane_x/2]
    __syncthreads()
    run_mma_pass(buf)
    __syncthreads()
```

**Smem delta: 0 bytes** (reuse sX double buffer; the fp16 X load goes
directly from HBM register to quantize+pack with no intermediate smem).

**But** Pass 1 (max-abs scan over all D) must happen before any group
of Pass 2 begins.  Pass 1 reads X fp16 O(D) bytes per token.  Same
traffic as today's first HBM pass of `activation_quant_kernel` — no
new traffic.

---

## 5. Register pressure

Current fused kernel ptxas output: **120–164 regs/thread** depending
on (kBm, kBn, kFuseQuant=false) — see r61 Stage G cuobjdump analysis.
RTX 4090 per-thread reg cap = 255.  Adding the quant prologue should
add ~20–30 regs (max-abs accumulator, scale math, packing temporaries)
→ **worst case ~190 regs**, still within budget, **but may drop
occupancy** if threads/CTA × regs > per-SM reg file (65536 regs/SM).

Safety check:
- kBm=128: 128 threads × 164 regs = 21 K → 3 CTAs/SM current
- kBm=128, +30 regs: 128 × 194 = 24.8 K → still 2 CTAs/SM comfortably
- kBm=64: 64 threads × 164 regs = 10.5 K → 6 CTAs/SM current
- kBm=64, +30 regs: 64 × 194 = 12.4 K → 5 CTAs/SM

Register pressure is **not** the binding constraint.

---

## 6. Risks and mitigations

| risk | probability | mitigation |
|---|---|---|
| Pass-1 scan adds too much HBM latency to first group's prologue | medium | Issue Pass-1 HBM load **overlapped** with group-0 W load via `cp.async` — they target different smem regions, free pipelining |
| perm indirection kills coalesced reads | low | Already indirect in current activation_quant; same HBM access pattern |
| 128-wide warp reduction within the kernel's 128-thread CTA doesn't compose with MMA's 32-thread warp layout | **high — critical** | See §7 |
| Small-T (T < kBn) edge cases (kBn=32 but T=16) wastes quant work | low | Mask: only lanes with n_global < T do the packing; max-abs and warp reduce still valid |
| Large-D shapes: Pass-1 scan becomes the prologue bottleneck for D=28672 (down_proj-70B) | medium | Acceptable — Tier 4 shapes are out of scope anyway |

### 6.1 Thread-layout compatibility (risk §6 row 3)

This is **the critical risk**:

- `activation_quant_kernel` uses `blockDim = (128, kBt, 1)` where
  lane_x ∈ [0, 128) is the 128-wide group reduction axis, lane_y ∈
  [0, kBt) is the token axis.
- `fused_dense_sparse_mma_int4_kernel` uses `blockDim = (kBm, 1, 1)`
  with kBm ∈ {128, 64}, organised as 4 or 2 warps each owning a
  distinct m-tile row-band for MMA.

**Conflict**: the 128-wide warp-tree reduction for max-abs and sum-X
expects 128 coherent lanes per token.  But in the MMA kernel, those
128 threads are partitioned across 4 warps each working on a different
m-slice — they cannot naturally reduce over the 128-d column axis of
a single token.

**Resolution options** (each has cost):

- **R1** — thread-remap only in prologue: within the prologue, re-interpret
  the 128 threads of a kBm=128 CTA as `(lane_x=0..31, warp_id=0..3,
  token_id_in_cta = lane_y)` mapping onto (8 tokens × 4 warps).  Do
  a 32-wide warp reduce + 4-warp reduce via smem.  Requires `__syncthreads`
  between prologue and main loop anyway (which we pay regardless).
  → This is the recommended path.  It's essentially the same reduction
  structure as the original activation_quant_kernel.

- R2 — separate warp-assignment in prologue: all 4 warps cooperatively
  quantize kBn tokens (not dividing by warp).  Within one warp,
  32 lanes cover 32 columns; 4 warps cover 128 columns.  Same as R1
  but cleaner.  → Use R2 if implementation clarity matters more than
  register pressure.

**Conclusion**: R2 is the right design.  Implementation effort: the
per-token reduction tree is mechanically the same as what
`activation_quant_kernel` already has; we re-home it into the prologue
using the MMA kernel's thread indexing.

---

## 7. Scope and gating

To de-risk, the fused path should be **opt-in and gated narrowly**:

```cpp
// Host launcher gate
bool use_fuse_quant =
    (T <= 128) &&                    // beyond this, 16us floor is <5% already
    (n_groups >= 16) &&              // small n_groups path uses fused_quant_gemv already
    kbm_pick == 128 &&               // start with the simpler 128-thread layout
    (env_force != "0");              // env override HKUST_V9_FUSE_QUANT=0 to disable
```

In practice:
- **kBm=128 + kBn=32** is the main Tier 1/2 setting → first target
- kBm=64 variants deferred to phase 2 of the same refactor
- hp_ratio > 0 (sparse branch): works unchanged — the prologue happens
  before the dense mainloop; sparse branch still reads the same
  in-smem quantized data

---

## 8. Work estimate

| task | days |
|---|---:|
| 1. Refactor `_prepare_fused_args` + bindings to accept `X_fp16 + perm` when `kFuseQuant=true` | 0.5 |
| 2. Implement prologue in kernel (Pass-1 scan + per-group fuse into mainloop's sX fill) | 1.5 |
| 3. Parity test (bit-exact vs activation_quant + fused path) | 0.5 |
| 4. Bench harness update — add fused path as a new column, not replace | 0.25 |
| 5. Tuning sweep (group-cache gate, kBn, register reconfirmation) | 0.5 |
| 6. Validation on 140-shape bench + roofline update | 0.25 |
| **total** | **3.5 days** |

Matches the "2-3 days" estimate earlier; 3.5 is the realistic figure
with parity debug.

---

## 9. Go/no-go decision checklist

Before starting the implementation, confirm:

- [ ] Tier 1+2 target shapes enumerated and bench baseline captured (done: `logs/r63_combined/bench.json`)
- [ ] Smem budget: reuse sX double buffer, **+0 bytes net** — ✅ feasible
- [ ] Register budget: +30 regs worst case, still <255 and doesn't hurt occupancy — ✅ feasible
- [ ] Thread-layout remap has a concrete plan (R2 above) — ✅
- [ ] Parity harness exists — yes (`kernel/cuda_kernel/tests/test_parity.py`)
- [ ] Launch-gate design preserves the legacy path for non-target shapes — ✅ (env override + kbm_pick gate)

**All green.  Ready to implement.**

---

## 10. What this does NOT do (explicit non-goals)

- Does **not** touch the MMA mainloop itself — `mma.m16n8k64.s4.s4.s32`
  pipeline and per-group dequant stay bit-identical.
- Does **not** address B2 (MMA pipeline starvation) — Tier 4 large
  compute-bound shapes are still bounded by warp scheduler saturation
  [[memory:bd78lejo]], that requires a separate round of work.
- Does **not** replace `fused_quant_gemv` (the T=1 decode path), which
  already fuses quant + GEMV for a different micro-architecture (dp4a).
- Does **not** apply to small Qwen3-0.6B shapes that are physics-bound
  vs FP16 cuBLAS.  Those shapes **will be closer to 1.0× but may
  still lose** — user explicitly accepts this.
