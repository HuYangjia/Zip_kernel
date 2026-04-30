# D.1 Warp-Specialised Kernel — Design Spec

**Author**: Phase 4 work stream
**Date**: 2026-04-30
**Target**: Branch `phase4-warp-specialised`, fallback = r66 main
**Time budget**: 12 days hard cap

---

## 0. Context & goal

### What we're attacking
After Path C (r66), T ≥ 128 compute-bound shapes sit at **cuda_eff ≈ 30%**
against their INT4 roofline.  Root cause per [[memory:bd78lejo]]:

- **B1**: ~76% of warp scheduler time is spent on serial HFMA2 dequant
  (per-group scale/zero fold) between consecutive MMA instructions.
- **B2**: shared-memory swizzle IMAD address arithmetic.
- **B3**: only 2-stage `cp.async` double buffering.

The existing kernel uses **1 warp role** — every warp does (load → dequant → MMA
→ epilogue).  Warp scheduler starves the TC pipeline because the same warp
must interleave all four duties.

### Goal
**Split warp roles** so that different warps work on different pipeline stages
in parallel:

| warp role | duty | target issue mix |
|---|---|---|
| **P (producer)** | `cp.async` W + X from HBM → smem | MEM |
| **D (dequant)** | int4 → fp16 HFMA2 into a **pre-dequantised smem buffer** | FP |
| **C (consumer)** | `ldmatrix` + `mma.sync.s4.s4.s32` → accumulator → epilogue | TC |

Because RTX 4090 (Ada sm_89) has 4 warp schedulers per SM, and a warp-specialised
CTA lets each scheduler pick the **instruction stream it was idle on**, we can
sustain higher TC throughput: the Consumer warp issues `mma.sync` every cycle
while Producer/Dequant warps hide their latency on the other schedulers.

### Success criterion
On Qwen3-8B T=512 gate_up (d_in=4096, d_out=24576) and Qwen3-14B T=512:
- cuda_eff: 30% → **≥ 45%**
- speedup: 0.87× → **≥ 1.10×**
- Parity 10/10 bit-exact (no tolerance).

---

## 1. Baseline kernel topology (r66)

From `fused_dense_sparse_mma_int4.cu`:

- **Block shape**: `blockDim = kBm` threads (kBm=128 → 128 threads = 4 warps; kBm=64 → 64 threads = 2 warps)
- **MMA pattern**: `mma.m16n8k64.s4.s4.s32`
  - K dim = 64 int4 elements (= 32 bytes per warp per iter)
  - M sub-tile = 16, warp owns `kMsubPerWarp = kBm / 4 / 16` rows? let me verify …
- **Smem budget (kBm=128, kBn=32, cache on)**:
  - `sW[2][128][64]` = 16 KB
  - `sX[2][32][64]`  = 4 KB
  - `s_scale_u4[128][34]` = 8.5 KB
  - `s_zero_u4 [128][34]` = 8.5 KB
  - `s_scale_x[32]` = 64 B
  - `s_scale_block[128]` = 256 B
  - `s_sum_X[2][32]` = 256 B
  - **Total ≈ 37.6 KB**  (fits in the default 48 KB smem)
- **Registers**: 167 per thread at kBn=32 kBm=128 → 128 × 167 = 21.4 KB regs
  → 3 blocks per SM possible with occupancy = 0.75.
- **cp.async**: 2-stage ping-pong on sW/sX.

---

## 2. D.1 design proposal

### 2.1 CTA shape (new)

| param | value | rationale |
|---|---|---|
| threads per CTA | **256 (8 warps)** | bigger CTA lets us split roles |
| P warps | 2 | need ~2 warps to saturate 4-stage cp.async on W (biggest transfer) |
| D warps | 2 | HFMA2 dequant is the dominant non-MMA work |
| C warps | 4 | drive `mma.sync` — same count as current kernel's MMA warps |
| kBm | **128** (fixed for D.1 first cut) | matches r66's main compute path |
| kBn | **64** (fixed) | T=128/512 target — kBn=64 is dispatcher's default for large T |
| kBk | 64 | same as r66 (one MMA group = 64 int4 K) |

### 2.2 Smem layout

Need a **fp16 scratch buffer** for pre-dequantised W (so consumers can `ldmatrix` directly without doing dequant themselves):

| buffer | shape | size | stages |
|---|---|---:|---:|
| `sW_i4`  (int4 raw) | `[stages_W][kBm][kBk/2]` | 128×32 = 4 KB × stages | **4** |
| `sX_i4`  (int4 raw) | `[stages_X][kBn][kBk/2]` | 64×32  = 2 KB × stages | **4** |
| `sW_fp16` (dequantised) | `[stages_D][kBm][kBk]`   | 128×128 = 16 KB × stages | **2** |
| `s_scale_u4` | `[kBm][kGrpBuf]` | 8 KB | 1 |
| `s_zero_u4`  | `[kBm][kGrpBuf]` | 8 KB | 1 |
| `s_scale_x`  | `[kBn]` | 128 B | 1 |
| `s_sum_X`    | `[stages_X][kBn]` | 512 B | 4 |

**Budget**:
- sW_i4 = 4 × 4 KB = 16 KB
- sX_i4 = 4 × 2 KB = 8 KB
- sW_fp16 = 2 × 16 KB = **32 KB** ← biggest new cost
- scales/zeros = 16 KB
- smaller = 1 KB
- **Total ≈ 73 KB**

RTX 4090 per-SM smem caps:
- default: 48 KB (too small)
- opt-in: **100 KB** via `cudaFuncSetAttribute(...PreferredSharedMemoryCarveout...)` — **this is what we need**

✅ 73 KB < 100 KB, **smem fits with carveout**.

**Occupancy**: 1 block per SM at 73 KB (100 KB / 73 KB = 1.37).  Lost occupancy is offset by warp-specialisation making **1 block fully utilise** all 4 schedulers (vs r66's 3 blocks each fighting for schedulers).

### 2.3 Synchronisation

Use `barrier.sync` with **named barrier IDs**:
- `bar.sync 0, 256` — full CTA barrier (stage rollover)
- `bar.sync 1, 128` — P→D handshake (P.new_stage_ready / D.old_stage_consumed)
- `bar.sync 2, 192` — D→C handshake (D.fp16_buffer_ready / C.consumed)

Alternative: **`mbarrier`** (Hopper/Ada) for arrive/wait with per-stage count.
For D.1 v1, use **named barriers** (simpler, proven on Ada).

### 2.4 Pipeline schedule

```
stage i:  | P loads W[i+3],X[i+3] | D dequants W_i4[i+1] -> W_fp16[i%2] | C does MMA on W_fp16[(i-1)%2] |
```

Steady state:
- **P**: always 3 stages ahead of D (deep cp.async pipeline hides HBM latency)
- **D**: always 1 stage ahead of C
- **C**: issues mma.sync back-to-back from W_fp16[prev] + X directly from sX_i4

### 2.5 Consumer design (crucial)

Consumer warps must **NOT** do dequant — that's the whole point. They:
1. `ldmatrix.sync.aligned.m8n8.x4` from `sW_fp16[d_stage]` → frag_a (8 fp16 regs per MMA)
2. `ldmatrix` from `sX_i4[x_stage]` → frag_b (4 int8 regs = 8 int4)
3. Issue `mma.m16n8k64.s4.s4.s32` — **but wait, frag_a is fp16 now, not s4**!

**⚠️ Critical issue**: `mma.m16n8k64.s4.s4.s32` takes **int4** operands, not fp16.  If D dequantises W to fp16, then C needs to use `mma.m16n8k16.f16.f16.f32` instead, which:
- has K=16 (not K=64) → **4× more MMA instructions**
- uses different MMA template entirely
- **gives up the INT4 TC throughput advantage**

This breaks the whole design.

### 2.6 Design revision — fix 2.5

Two options:

**Option α**: Keep int4 MMA, **D does scale/zero pre-fold into sX instead of sW**.
- At fragment load: `sW` stays raw int4 (same as r66), `sX` becomes **pre-scaled fp16** (rather, pre-scaled int8 via fold trick? no, X is already int4 quantised)
- Hmm, but the dequant fold in r66 is: `y_fp = scale_x * (scale_u4 * d_acc - scale_u4 * zero * sum_X)` applied **after** MMA.  D warps can pre-compute `scale_u4 * zero * sum_X` for each (n, g) pair into a smem buffer, offloading that from C.
- **Limited benefit**: only moves ~30% of the dequant work off C.

**Option β**: Keep r66 MMA path, move **HFMA2 chain to a dedicated D warp** that operates on `d_acc` (int32 accumulator) via smem spill.
- C dumps int32 accumulator to smem → D picks it up, does HFMA2 fold → writes fp16 back to smem → C does epilogue store
- **This breaks register-resident accumulator**, adds smem round-trips.  Likely slower.

**Option γ (RECOMMENDED)**: **keep r66's in-warp MMA+dequant**, specialise only the **cp.async producer**.
- P warp group does all `cp.async` issue for W and X, does `cp.async.commit_group`.
- C warps (6 of 8) do MMA + dequant as in r66, but **never issue cp.async** — they just `cp.async.wait_group` and go.
- This gets rid of the `cp.async` IMAD overhead in the hot loop (B2) and lets C warps focus on MMA issue + HFMA2.
- **Deeper pipeline**: 4-stage cp.async (B3 also fixed).

### 2.7 Revised plan (Option γ)

| warp role | count | duty |
|---|---:|---|
| **P (producer)** | 2 warps (64 threads) | `cp.async` W + X + scales; `cp.async.commit_group` every stage |
| **C (consumer)** | 6 warps (192 threads) | `ldmatrix` + `mma.sync.s4.s4.s32` + **in-warp HFMA2 dequant fold** (as r66) + epilogue |

**CTA = 256 threads = 8 warps**.  `blockDim = 256`.

Benefits:
- **B3 solved**: 4-stage cp.async (only producer issues, so no contention).
- **B2 reduced**: consumer warps don't execute cp.async IMAD path.
- **B1 half-mitigated**: consumer has 6 warps worth of MMA issue bandwidth, which gives HFMA2 more slots to hide. But the fundamental HFMA2 serialisation per MMA is still there — **only a true dequant-offload (Option α/β) could fully solve B1**, and we've shown those aren't tractable for int4 MMA.

**Expected eff uplift**: 30% → **~40%** (not the original ~50% target).
**Expected speedup**: 0.87× → **~1.05×** on Qwen3-8B T=512.

### 2.8 Is Option γ worth the 12-day budget?

Honest answer: **no, γ alone is roughly equivalent to D.2 (just deeper pipeline)** in terms of outcome, and could be done in 4-5 days not 12.

But γ **sets up** Option α follow-on:
- Once P is separated, we have a clean 2/6 split.
- A further pass can try pushing some **scale/zero lookup** work into P's spare capacity (P has HBM latency bubbles, can run FP ops in the meantime).

---

## 3. Revised strategy after design review

### 3.1 My recommendation

Given the 2.5 discovery (int4 MMA blocks full dequant offload), **I propose a smaller D.1 scope**:

**D.1a (4-5 days)**: Implement **Option γ** — producer/consumer split with 4-stage cp.async.
- Expected: eff 30 → 40%, Qwen3-8B T=512 speedup 0.87 → 1.05.
- Failure mode: if 4-stage cp.async doesn't help enough, fall back to r66.

**D.1b (3-4 days, conditional on D.1a success)**: Try **Option α lite** — P warps pre-compute `scale_u4 * zero * sum_X` into smem.
- Expected additional: eff 40 → 45%, 1.05 → 1.10.

**Total 7-9 days**, less than the 12-day cap.  If either stage fails parity, merge what's stable, fall back on the rest.

### 3.2 What I give up

- The "warp-specialised rewrite" framed in the 8-10-day Path D.1 estimate assumed a full pre-dequant-to-fp16 architecture.  Section 2.5 showed that's incompatible with s4 MMA without giving up TC throughput.
- Realistic Path D.1 ceiling on Ada for s4 MMA = **~45% eff**, not 50-60%.  The 50-60% figure in Phase 4 work plan assumed fp16/bf16 MMA where pre-dequant works.

### 3.3 Open decision points (need your input)

Before writing code, please confirm:

**Q1**: Accept the **revised 45% eff ceiling** (vs original 50-60%)?
- If yes → proceed with Option γ + α lite (D.1a + D.1b).
- If no → I should investigate whether **switching to fp16 MMA path** is feasible (would lose INT4 throughput advantage, but enable full dequant offload).

**Q2**: Do D.1a first and only commit to D.1b if D.1a succeeds?
- Yes (recommended): safer, ~7-9 day total.
- No: batch both upfront, higher risk.

**Q3**: Are 6 consumer warps OK (vs r66's 4 MMA warps)?
- Implication: blockDim 128 → 256.  We've verified smem/reg fits.
- Implication: the dispatcher must choose this kernel only when there's enough work (d_out × d_in × T ≥ some threshold) — small shapes still use r66 path.

---

## 4. Phased milestones (assuming γ+α lite path)

| MS | days | content | parity gate |
|---|---:|---|---|
| MS0 | 0.5 | this design doc; sign-off | — |
| MS1 | 1.5 | add 256-thread CTA variant + 4-stage cp.async on a separate kernel template; all 8 warps still do full work (no role split yet) | 10/10 |
| MS2 | 1.5 | introduce P/C role split via warp-id conditional; P does cp.async issues, C does everything else | 10/10 |
| MS3 | 1.0 | bench on 30-shape (T=128/512 × 3 models) vs r66 | — |
| MS4 | 2.0 | D.1b: add scale/zero/sum_X pre-fold in P's spare cycles | 10/10 |
| MS5 | 1.0 | final 30-shape bench + 140-shape full validation | — |
| MS6 | 0.5 | merge decision + fallback docs | — |
| **total** | **7 days** | | |

### Hard cap discipline
If any MS blows its time by 2×, **pause + evaluate**.  If MS2 or later fails bit-exact parity for 2 consecutive fix attempts, **retreat to last passing MS** and cap deliverable there.

---

## 5. Risk register

| risk | mitigation |
|---|---|
| **named barrier deadlock** | start with a 1-warp-role (all 8 warps do same thing) version first; introduce roles gradually |
| **smem exceeds 100 KB** | we're at 73 KB with D.1b; if α lite adds too much, drop it |
| **register count blows up to 255 (spill)** | D.1 uses kBn=64 only (not 32); test with `-Xptxas -v` at each MS |
| **parity regression** | MS1 & MS2 are no-op equivalents — same math, different threading.  Any parity break == bug, fix immediately |
| **dispatcher thrash** | new kernel is opt-in via `HKUST_V9_WARP_SPEC` env; final dispatcher gate narrowed to T ≥ 128 only |
| **bench noise masking real regression** | always compare against r66 on same machine same day; use 7-trial interleaved for suspect shapes |

---

## 6. Fallback plan

If at any hard-cap break the final state is *worse than r66*:
1. `git checkout main` (r66 is pristine there)
2. `git branch -D phase4-warp-specialised` (nuke the branch)
3. Archive design doc + what-didn't-work notes in `docs/PHASE4_D1_POST_MORTEM.md`
4. Move on to Path D.3 (dual-issue PTX) or D final-report.

---

## 7. What I need from you

Please reply with **one of**:
- "**proceed γ+α**" — I start MS1 immediately on this branch.
- "**γ only**" — skip D.1b; stop after MS3.
- "**full rewrite fp16**" — abandon the int4-MMA path, switch to fp16-MMA with full dequant offload (means 14-day effort, 40% failure rate).
- "**rethink**" — with specific pushback.
