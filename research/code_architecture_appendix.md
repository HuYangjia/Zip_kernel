# V9 Kernel 技术附录（AI / Machine Reference）

> **目的**：给"要动 kernel"的人一本精确的 contract 参考书。
> **风格**：精确、可执行、机器可读；不提供直觉解释（解释在 [`code_architecture.md`](./code_architecture.md)）。
> **适用**：读代码 / 写 PR / 给 AI agent 喂 prompt。

## 目录

**Part 1 — 规范与契约**
- [§A. Pipeline 精确数据流图](#a-pipeline-精确数据流图)
- [§B. 数据结构契约](#b-数据结构契约)
- [§C. 当前 Autotune 网格快照](#c-当前-autotune-网格快照commit-参考main)
- [§D. 测试与基准运行手册](#d-测试与基准运行手册)
- [§E. Commit / PR 规范](#e-commit--pr-规范)

**Part 2 — 诊断与推导**
- [§F. 常见雷区](#f-常见雷区已踩过的)
- [§G. Roofline 推导](#g-推导为什么-densefp16-的-roofline-在-10x-附近)
- [§H. 文件 index](#h-文件-index精确行号参考用)

**Part 3 — 深度优化 Playbook**（对应主文"下次改它时要看的"四大方向）
- [§J. Dequant 在 Triton IR / SASS 上发生了什么](#j-dequant-在-triton-ir--sass-上发生了什么) ← PTX `prmt.b32`
- [§K. 4090 Shared Memory Budget 推导](#k-4090-shared-memory-budget-推导) ← autotune 为何止步 256×256
- [§L. W4A16 Fallback 阈值调参方法](#l-w4a16-fallback-阈值调参方法) ← 跨卡迁移时必读
- [§M. CUDA Graph 接入 Decode Path 的具体步骤](#m-cuda-graph-接入-decode-path-的具体步骤)

**Part 4 — 参考**
- [§N. Glossary](#n-glossary)

---

## §A. Pipeline 精确数据流图

```
INPUT                                  OUTPUT
─────                                  ──────
X_fp16          : (T, d_in) fp16        Y_fp16 : (T, d_out) fp16
V9WeightContainer (static)

STAGE 1  activation_quant_kernel
  in   : X_fp16 (T, d_in)
  out  : X_s4_packed (T, d_in//2) int8   -- bit[0:4]=col[2i], bit[4:8]=col[2i+1]
         scale_x     (T,) fp16           -- row-wise symmetric SINT4 scale
         sum_X       (T, n_groups) int32 -- group sum of q_s4 (for zero-point fix)

STAGE 2  dense_gemm_kernel
  in   : W_low_packed, scale_u4, zero_u4 (from V9WeightContainer)
         X_s4_packed, scale_x, sum_X
  out  : Y_low (d_out, T) fp16           -- NOTE: (d_out, T) layout for coalesced stores

STAGE 3  sparse_gemm_kernel  [SKIPPED if n_hp_blocks == 0]
  in   : W_high_blocks_packed, hp_row_offsets, hp_col_indices,
         scale_u4 (reused), X_s4_packed, scale_x, sum_X
  out  : Y_high (d_out, T) fp16          -- same layout as Y_low

STAGE 4  _combine_transpose  (Triton kernel OR torch fallback)
  in   : Y_low, Y_high (d_out, T) fp16
  out  : Y_fp16 (T, d_out) fp16          -- Y = (Y_low + 16*Y_high).T.contiguous()
  dispatch:
     T * d_out <= 4*1024*1024  ->  torch native
     T * d_out >  4*1024*1024  ->  _combine_transpose_kernel

PREFILL W4A16 FALLBACK  [when T >= 1024 or (T>=512 and d_in*d_out<=4096^2) and hp=0]
  Replace STAGE 2+3+4 with:
    W_fp16 = dequant_u4_to_fp16_kernel(W_low_packed, scale_u4, zero_u4)
    Y_fp16 = torch.nn.functional.linear(X_fp16[:, perm], W_fp16)
```

---

## §B. 数据结构契约

### B.1 `V9WeightContainer`（`pack_utils.py`）

```python
@dataclass
class V9WeightContainer:
    # --- dense low-precision (always present) ---
    W_low_packed        : torch.Tensor  # (d_out, d_in // 2)           int8  row-major
    scale_u4            : torch.Tensor  # (d_out, n_groups)             fp16
    zero_u4             : torch.Tensor  # (d_out, n_groups)             fp16  (pre-subtracted 8)

    # --- sparse high-precision (may have n_hp_blocks == 0) ---
    W_high_blocks_packed: torch.Tensor  # (n_hp_blocks, 128, 64)        int8  BSR value blocks
    hp_row_offsets      : torch.Tensor  # (d_out // 128 + 1,)           int32 BSR indptr
    hp_col_indices      : torch.Tensor  # (n_hp_blocks,)                int32 BSR indices

    # --- permutation (GPTQ act-order) ---
    perm                : torch.Tensor  # (d_in,)                       int32 permutation of K axis

    # --- metadata ---
    d_out               : int
    d_in                : int
    block_shape         : Tuple[int, int] = (128, 128)   # (block_M, block_K)
    group_size          : int = 128                       # n_groups = d_in / group_size
```

**Invariants**（代码中未必全 assert，但必须满足）：
- `d_in % 128 == 0` 且 `d_out % 128 == 0`
- `W_low_packed.shape == (d_out, d_in // 2)`，`dtype == torch.int8`
- `scale_u4.shape == (d_out, d_in // 128)`
- `zero_u4` 存的是 `-(z - 8) * scale`（已预减 8，给 SINT4 MMA 用）
- BSR：`hp_row_offsets[-1] == n_hp_blocks`；`hp_col_indices ∈ [0, d_in//128)`
- `perm` 是 `[0, d_in)` 的一个排列（整数 permutation）
- 所有 tensor 在同一 GPU 上，`.is_contiguous() == True`

### B.2 Packed 4-bit Layout

**W_low_packed**（uint4 → int8 容器）：
```
W_low_packed[m, k_byte] = (W_u4[m, 2*k_byte+1] << 4) | W_u4[m, 2*k_byte] & 0xF
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^
                                high nibble                   low nibble
```
其中 `W_u4` 取值范围 `[0, 15]`（UINT4）。MMA 时先减 `zero`（里面已经预减 8 = 将 UINT4 → SINT4 range `[-8, 7]`），再和 SINT4 X 做 `tl.dot`。

**X_s4_packed**（sint4 → int8 容器）：
```
X_s4_packed[t, k_byte] = (X_s4[t, 2*k_byte+1] << 4) | X_s4[t, 2*k_byte] & 0xF
```
`X_s4` 取值范围 `[-7, 7]`（对称，避开 `-8`）。

**W_high_blocks_packed**（BSR block value）：
```
W_high_blocks_packed[b, i, j] = (high_nibble << 4) | low_nibble    # 同上
  shape: (n_hp_blocks, 128, 64)     # 128 = block_M, 64 = block_K//2
```

### B.3 Kernel 参数 schema

**`activation_quant_kernel`**：
```python
quantize_activation_kernel(
    X_ptr,              # (T, d_in) fp16
    X_s4_ptr,           # (T, d_in//2) int8    out
    scale_x_ptr,        # (T,) fp16            out
    sum_X_ptr,          # (T, d_in//BCOL) int32 out
    T, d_in,            # int
    stride_x_t, stride_x_d,
    stride_xs4_t, stride_xs4_d,
    BCOL: tl.constexpr = 128,   # group size == 128
    BT:   tl.constexpr,
    BD:   tl.constexpr,
)
# grid = (cdiv(T, BT), cdiv(d_in, BD))
```

**`dense_gemm_kernel`**：
```python
dense_gemm_kernel(
    W_low_ptr,          # (d_out, d_in//2) int8
    X_s4_ptr,           # (T, d_in//2) int8
    scale_w_ptr,        # (d_out, n_groups) fp16
    zero_w_ptr,         # (d_out, n_groups) fp16   pre-subtracted
    scale_x_ptr,        # (T,) fp16
    sum_X_ptr,          # (T, n_groups) int32
    Y_low_ptr,          # (d_out, T) fp16          out
    d_out, T, d_in,     # int
    stride_w_m, stride_w_k,
    stride_x_t, stride_x_k,
    stride_sw_m, stride_sw_g,
    stride_zw_m, stride_zw_g,
    stride_sumx_t, stride_sumx_g,
    stride_y_m, stride_y_t,
    BCOL: tl.constexpr = 128,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,  # BK must == BCOL == 128
    GROUP_SIZE_M: tl.constexpr,
)
# grid = (cdiv(d_out, BM) * cdiv(T, BN),)   with GROUP_SIZE_M swizzle
```

**`sparse_gemm_kernel`**：
```python
sparse_gemm_kernel(
    W_high_blocks_ptr,  # (n_hp_blocks, 128, 64) int8
    hp_row_offsets_ptr, # (d_out//128 + 1,) int32
    hp_col_indices_ptr, # (n_hp_blocks,) int32
    scale_w_ptr, X_s4_ptr, scale_x_ptr, sum_X_ptr,
    Y_high_ptr,         # (d_out, T) fp16          out
    d_out, T, d_in,
    strides...,
    BLOCK_M: tl.constexpr = 128,     # fixed: matches block_shape
    BLOCK_K: tl.constexpr = 128,
    BM: tl.constexpr, BN: tl.constexpr,
)
# grid = (cdiv(d_out, BM) * cdiv(T, BN),)
```

**`_dequant_u4_to_fp16_kernel`**：
```python
_dequant_u4_to_fp16_kernel(
    W_low_ptr,          # (d_out, d_in//2) int8
    scale_ptr,          # (d_out, n_groups) fp16
    zero_ptr,           # (d_out, n_groups) fp16   pre-subtracted
    W_fp16_ptr,         # (d_out, d_in) fp16       out
    d_out, d_in,
    stride_w_m, stride_w_k,
    stride_s_m, stride_s_g,
    stride_z_m, stride_z_g,
    stride_out_m, stride_out_k,
    BCOL_K: tl.constexpr = 128,
    BM: tl.constexpr, BK: tl.constexpr,
)
# grid = (cdiv(d_out, BM), cdiv(d_in, BK))
```

**`_combine_transpose_kernel`**：
```python
_combine_transpose_kernel(
    Y_low_ptr,          # (d_out, T) fp16
    Y_high_ptr,         # (d_out, T) fp16     may be null if HAS_HIGH=False
    Y_out_ptr,          # (T, d_out) fp16     out
    T, d_out,
    stride_low_m, stride_low_t,
    stride_high_m, stride_high_t,
    stride_out_t, stride_out_m,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_HIGH: tl.constexpr,
)
# grid = (cdiv(T, BT), cdiv(d_out, BD))
# Y_out[t, m] = Y_low[m, t] + (16 * Y_high[m, t] if HAS_HIGH else 0)
```

---

## §C. 当前 Autotune 网格快照（commit 参考：main）

### C.1 `dense_gemm_kernel`（15 configs, `key=["d_out","d_in","T"]`）

```python
# Decode tier (T<=16): tiny BN
Config(BM=64,  BN=16,  BK=128, GROUP_SIZE_M=1,  warps=2, stages=3)
Config(BM=128, BN=16,  BK=128, GROUP_SIZE_M=1,  warps=4, stages=3)
Config(BM=128, BN=32,  BK=128, GROUP_SIZE_M=4,  warps=4, stages=3)

# Small/mid tier (T=32..512)
Config(BM=64,  BN=64,  BK=128, GROUP_SIZE_M=8,  warps=4, stages=3)
Config(BM=128, BN=64,  BK=128, GROUP_SIZE_M=8,  warps=4, stages=3)
Config(BM=64,  BN=128, BK=128, GROUP_SIZE_M=8,  warps=4, stages=3)

# Prefill tier (T>=2K)
Config(BM=128, BN=128, BK=128, GROUP_SIZE_M=8,  warps=4, stages=3)
Config(BM=128, BN=128, BK=128, GROUP_SIZE_M=8,  warps=8, stages=4)
Config(BM=256, BN=128, BK=128, GROUP_SIZE_M=8,  warps=8, stages=3)
Config(BM=128, BN=256, BK=128, GROUP_SIZE_M=8,  warps=8, stages=3)

# Phase B-1 experimental (big TC tiles)
Config(BM=256, BN=256, BK=128, GROUP_SIZE_M=8,  warps=8, stages=3)
Config(BM=256, BN=128, BK=128, GROUP_SIZE_M=8,  warps=8, stages=4)
Config(BM=128, BN=256, BK=128, GROUP_SIZE_M=8,  warps=8, stages=4)
Config(BM=128, BN=128, BK=128, GROUP_SIZE_M=8,  warps=4, stages=5)
Config(BM=128, BN=128, BK=128, GROUP_SIZE_M=16, warps=8, stages=4)
```

**CONSTRAINT**: `BK must == BCOL == 128`（组内反量化假设）。
**SHARED MEM BUDGET**（4090, 164KB）：`BM*BK + BN*BK` of int8 + `BM*BN` of fp32 acc + `BM*n_groups` scales × stages。`BM=256, BN=256, stages=3` 已逼近上限。

### C.2 `activation_quant_kernel`（11 configs, `key=["T","d_in"]`）

```python
# Decode tier (T<=16)
Config(BT=16,  BD=256,  warps=2, stages=2)
Config(BT=16,  BD=512,  warps=2, stages=3)
Config(BT=32,  BD=256,  warps=2, stages=2)
Config(BT=32,  BD=512,  warps=4, stages=3)

# Small/mid
Config(BT=64,  BD=512,  warps=4, stages=2)
Config(BT=64,  BD=1024, warps=4, stages=3)
Config(BT=128, BD=512,  warps=4, stages=2)
Config(BT=128, BD=1024, warps=8, stages=2)

# Prefill
Config(BT=64,  BD=2048, warps=8, stages=2)
Config(BT=64,  BD=2048, warps=8, stages=3)
Config(BT=128, BD=2048, warps=8, stages=3)
```

### C.3 `sparse_gemm_kernel`（3 configs — 欠 tuning）

```python
Config(BM=128, BN=128, warps=4, stages=2)
Config(BM=128, BN=128, warps=8, stages=3)
Config(BM=64,  BN=128, warps=4, stages=2)
```
**改进点**：缺 `BN=16/32` 的 decode config；缺 `warps=2` 选项；缺 stages=4。

### C.4 `_dequant_u4_to_fp16_kernel`（6 configs）

```python
Config(BM=64,  BK=256, warps=4, stages=3)
Config(BM=64,  BK=512, warps=4, stages=3)
Config(BM=128, BK=256, warps=4, stages=3)
Config(BM=128, BK=512, warps=8, stages=3)
Config(BM=32,  BK=512, warps=4, stages=3)
Config(BM=256, BK=256, warps=8, stages=3)
```
**CONSTRAINT**: `BK % BCOL_K == 0`，即 `BK % 128 == 0`。

### C.5 `_combine_transpose_kernel`（5 configs）

```python
Config(BT=32,  BD=256, warps=4)
Config(BT=64,  BD=128, warps=4)
Config(BT=32,  BD=512, warps=8)
Config(BT=64,  BD=256, warps=8)
Config(BT=128, BD=128, warps=8)
```

### C.6 关键路径常量

```python
# v9_linear.py
DECODE_T_THRESHOLD      = 128       # dispatcher: T <= -> decode; T > -> prefill
SMALL_SURFACE           = 4 * 1024 * 1024   # 4M fp16 elements = 8 MiB

# W4A16 fallback trigger (_v9_forward_prefill)
use_w4a16 = (
    W.n_hp_blocks == 0
    and (T >= 1024 or (T >= 512 and d_out * d_in <= 4096 * 4096))
)

# Fixed dimensions
BCOL          = 128     # quantization group size (K-axis tile)
BLOCK_M_HP    = 128     # BSR block M
BLOCK_K_HP    = 128     # BSR block K
```

---

## §D. 测试与基准运行手册

### D.1 单元测试

| 文件 | 覆盖 | 运行命令 |
|---|---|---|
| `tests/test_activation.py` | quant kernel 精度 | `pytest tests/test_activation.py -v` |
| `tests/test_dense.py` | dense kernel 精度 vs fakequant | `pytest tests/test_dense.py -v` |
| `tests/test_sparse.py` | sparse kernel 精度 | `pytest tests/test_sparse.py -v` |
| `tests/test_pack_utils.py` | BSR 打包正确性 | `pytest tests/test_pack_utils.py -v` |
| `tests/test_end2end.py` | 完整 forward vs fakequant | `pytest tests/test_end2end.py -v` |
| `tests/test_prefill_decode_dispatch.py` | dispatcher 一致性 | `pytest tests/test_prefill_decode_dispatch.py -v` |
| `tests/test_w4a16_fallback.py` | W4A16 fallback 等价 | `pytest tests/test_w4a16_fallback.py -v` |
| **全部** | — | `pytest triton_kernel/tests/ -q` |

**合格线**：`rtol=1e-3, atol=5e-3`（fakequant 参考实现），`median_rel_err < 2e-3` 典型。

### D.2 性能基准

| 脚本 | 用途 | 耗时 | 输出 |
|---|---|---|---|
| `bench_dense.py` | 单 kernel vs cuBLAS FP16 | 1-2min | `results/bench_*.{csv,md}` |
| `bench_sparse.py` | sparse kernel 扫 hp_ratio | 1min | 同上 |
| `bench_linear.py` | 端到端对比 | 2min | 同上 |
| `bench_dispatcher_overhead.py` | dispatcher 开销 | 30s | stdout |
| `bench_phase_b1_compare.py` | Dense autotune A/B | 3min | `phase_b1_*.{csv,md}` |
| `sweep_v9.py` | **Golden**: 7 shapes × 6 batches × 4 hp × 3 metrics | 5-10min | `sweep_<ts>.{csv,md,log}` |
| `diag_fp16_variance.py` | 测量噪声诊断 | 30s | `diag_fp16_variance.csv` |

### D.3 Profiling

```bash
# Nsight Systems (time line + NVTX ranges + kernel occupancy)
bash triton_kernel/benchmarks/run_nsys_sweep.sh
python triton_kernel/benchmarks/summarize_nsys.py
# outputs: results/nsys_<ts>_*.csv + nsys_summary_<ts>.md

# Nsight Compute (per-kernel roofline / memory / compute)
ncu --set full --kernel-name regex:dense_gemm_kernel \
    --target-processes all \
    python -m triton_kernel.benchmarks.bench_dense
```

### D.4 结果归因工具

```bash
# Bottleneck breakdown per bucket (bs_tier × hp_tier)
python -m research.tools.analyze_sweep_bottleneck \
    triton_kernel/benchmarks/results/sweep_<ts>.csv

# Shows:
#  - stage time share (quant/dense/sparse/combine as % of v9_total)
#  - Amdahl upper-bound per stage
#  - dense/fp16 ratio by bucket
#  - cases closest to beating FP16
```

### D.5 微基准规范（强制） [[memory:bmmiahpl]]

```python
from triton_kernel.benchmarks._bench_util import bench_kernel

# Contract:
# - warmup >= 50 iterations
# - measurement: 3 windows of >= 100 iterations each
# - return: min-of-means (NOT single-pass mean)
# - DO NOT use nsys/ncu as primary timer (CUPTI hook biases micro-kernels)
ms = bench_kernel(fn, warmup=50, windows=3, iters=100)
```

---

## §E. Commit / PR 规范

### E.1 Commit message 格式

```
<type>(<scope>): <summary, imperative, <=72 chars>

<body explaining WHY and WHAT>

Perf impact:
  - bs=...:  before X ms -> after Y ms (Z%)
  - sweep geomean speedup vs FP16: A -> B

Tests:
  - pytest tests/  : N passed
  - sweep_v9.py    : no regression on M shapes

Refs: #issue or optimization_report.md §x.y
```

### E.2 type / scope

- `type`: `feat` / `perf` / `fix` / `refactor` / `docs` / `test` / `bench`
- `scope`: `dense` / `sparse` / `quant` / `dequant` / `dispatcher` / `pack` / `bench` / `infra`

### E.3 改动必跑清单（PR 前）

```bash
# 1. 精度回归
pytest triton_kernel/tests/ -q

# 2. 性能基准
python -m triton_kernel.benchmarks.sweep_v9

# 3. 瓶颈归因（可选，看是否解锁新形态）
python -m research.tools.analyze_sweep_bottleneck \
    triton_kernel/benchmarks/results/sweep_<latest>.csv

# 4. 写入 optimization_report.md 新 section
```

---

## §F. 常见雷区（已踩过的）

| 现象 | 根因 | 正确处理 |
|---|---|---|
| FP16 baseline `bs=16` 比 `bs=1` 还慢 | warmup 不足 / 测量窗口太少 | 见 §D.5 规范 |
| Triton kernel 比 torch native 还慢 | 微 kernel 有 55μs 固定 launch；surface < 4M 时 torch 的 copy kernel 赢 | `SMALL_SURFACE` fallback |
| dense/fp16 停在 1.27x 降不下去 | TC 占用率 < 30%，4-bit unpack 喂不饱 | Dequant PTX `prmt.b32` + Split-K |
| combine kernel 自己跑得快，端到端没收益 | stage 4 是"尾巴"，bottleneck 在 dense | 用 §D.4 Amdahl 分析确认 |
| 修改 `W_low_packed` 布局后 pytest 过了但 e2e 错 | `pack_utils.pack_v9_weights` 没同步更新 | 改 layout 必跑 `test_pack_utils` + `test_end2end` |
| W4A16 fallback 在小 T 反而更慢 | 反量化 + cuBLAS 的 overhead > 自家 kernel | 保持 T>=1024 的阈值；fine tune 时跑 `bench_phase_b1_compare` |
| sparse kernel 在 hp=0 仍被调用占时间 | early-skip 判断漏了 | `v9_linear.py` 里确保 `if W.n_hp_blocks > 0:` guard |
| autotune 第一次 runtime 几秒 | 所有 config 编译 + 实测一次 | 生产前预热所有 (d_out, d_in, T) 组合 |

---

## §G. 推导：为什么 dense/fp16 的 roofline 在 1.0x 附近

**假设**（RTX 4090 spec）：
- Peak FP16 Tensor Core: 330 TFLOPS
- Peak INT4 Tensor Core: 1320 TOPS
- HBM BW: 1008 GB/s
- L2: 72 MB, L1/SMEM: 128KB per SM

**Compute-bound regime**（大 T）：
- 理论 4-bit GEMM 比 FP16 快 4x（`1320/330`）
- 实际 dense/fp16 = 1.27x（TC 占用率 ~30%）→ 有 **~3x headroom** 是真的空间
- 主要损失：(1) dequant 软件 unpack 占 MMA pipeline bubble；(2) K-axis 不够长 (d_in=4096/11008) 无法喂满 stages=3

**Memory-bound regime**（小 T = decode）：
- FP16 每次 matvec 读 `2 * d_in * d_out` bytes
- W4A4 每次 matvec 读 `d_in * d_out / 2` bytes（权重） + `d_in/2` bytes（激活, 可忽略）
- 理论带宽比 4:1，**decode speedup 上限 ≈ 4x** 对纯 memory-bound 场景
- 实测 `bs=1, d_out=28672` 达到 0.73x —— 这不是带宽问题（已到 73% peak），是 **quant + launch + dequant scalar 开销**吃掉了份额

→ **结论**：
- Prefill 优化 ROI = 打磨 TC 占用率（Split-K, PTX dequant, epilogue fusion）
- Decode 优化 ROI = 消灭 overhead（CUDA Graph, kernel fusion, 小 tile）

---

## §H. 文件 index（精确行号参考用）

**`v9_linear.py`**（~570 lines）：
- L45–54: `_combine_transpose_kernel` autotune configs
- L58–116: `_combine_transpose_kernel` kernel body
- L125–175: `_combine_transpose` launcher with `SMALL_SURFACE` fallback
- L204: `DECODE_T_THRESHOLD = 128`
- L210–248: `_v9_forward_decode`
- L251–330: `_v9_forward_prefill`（含 W4A16 fallback）
- L340–395: `v9_linear_forward`（dispatcher）
- L400–440: `v9_linear_forward_decode` / `v9_linear_forward_prefill`（显式 API）
- L480–535: `v9_linear_fakequant`（参考实现）

**`dense_u4s4_gemm.py`**（~400 lines）：
- L61–90: autotune configs (15 entries)
- L100–280: `dense_gemm_kernel` body
- L300–400: `dense_gemm_forward` launcher

**`sparse_s4s4_gemm.py`**（~260 lines）：
- L42–48: autotune configs (3 entries)
- L60–200: `sparse_gemm_kernel` body
- L210–260: launcher

**`activation_quant.py`**（~350 lines）：
- L27–54: autotune configs (11 entries)
- L70–250: kernel body
- L280–350: launcher + tests helper

**`dequant_w4_to_fp16.py`**（~200 lines）：
- L49–59: autotune configs (6 entries)
- L70–150: kernel body
- L160–200: launcher

**`pack_utils.py`**（~500 lines）：
- L50–100: `V9WeightContainer` dataclass
- L150–250: `pack_v9_weights` (GPTQ → container)
- L300–400: BSR builder
- L420–480: unpack helpers (for test ref)

---

## §J. Dequant 在 Triton IR / SASS 上发生了什么

本节解释"为什么 prefill dense/fp16 = 1.27x"的根因（TC pipeline bubble），并给出 PTX `prmt.b32` 优化方案的精确入口。

### J.1 当前软件 unpack（`dense_u4s4_gemm.py` L31–43）

```python
@triton.jit
def _unpack_packed_s4_rowmajor(packed, BM: tl.constexpr, BK: tl.constexpr):
    # packed : (BM, BK // 2) int8   -- each byte holds (high << 4) | (low & 0x0F)
    low  = packed & 0x0F                  # (BM, BK//2) int8, UINT4 value
    high = (packed >> 4) & 0x0F           # (BM, BK//2) int8, UINT4 value
    # sign-extend to SINT4 by (x ^ 0x8) - 8  —— implicitly done by MMA via `zero` pre-subtraction
    # interleave: [low[0], high[0], low[1], high[1], ...] along K axis
    stacked = tl.join(low, high)          # (BM, BK//2, 2) int8
    return tl.reshape(stacked, (BM, BK))  # (BM, BK) int8
```

**Triton IR 翻译**（`triton-dejavu` 可以 dump；或直接 `TRITON_CACHE_DIR=/tmp/cache python ...`，去 `/tmp/cache/*/` 翻）：

```mlir
// 简化示意；BM=128, BK=128, 4-stage pipelined load
%packed = tt.load %w_ptr : tensor<128x64xi8>         // (BM, BK_half) int8, 8KB
%low    = arith.andi %packed, %c15 : tensor<128x64xi8>
%shifted= arith.shrui %packed, %c4  : tensor<128x64xi8>
%high   = arith.andi %shifted, %c15 : tensor<128x64xi8>
%joined = tt.join %low, %high : tensor<128x64x2xi8>
%reshp  = tt.reshape %joined : tensor<128x128xi8>    // 16KB (expanded 2x)
...
%acc    = tt.dot %reshp, %x_reshp, %acc_in : ... i32  // hmma.16816.s4.s4
```

**SASS 层发生的事**（4090/SM89；用 `cuobjdump --dump-sass` 看 triton `.so` 内的 cubin 可确认）：

| 步骤 | SASS 指令 | 耗 cycles（近似） | bubble 代价 |
|---|---|---|---|
| load packed 8KB to SMEM | `LDGSTS.E.128` ×16（async） | 覆盖在 pipeline 里 | 0 |
| SMEM → RF load | `LDS.64` ×N | ~10 × per quad | 0（小）|
| `& 0x0F` | `LOP3.LUT` | 1 | **0 但占 FMA port** |
| `>> 4` | `SHR.U32` | 1 | 0 |
| `tl.join` + reshape | Triton 层映射为若干 `PRMT` / `MOV` | **每 4 元素 1 个 PRMT** | **占 FMA/ALU port, 挤占 MMA issue window** |
| `tl.dot (s4 × s4 → s32)` | `HMMA.16816.S4.S4` | 16 cycle | — |

**痛点**：SM89 上每个 SM 每 cycle 可以发 1 条 TC 指令 OR 1 条 ALU 指令，但不能都发。当前每发 1 条 HMMA 前面要发 ~8 条 ALU（`AND`/`SHR`/`PRMT`/`MOV` 去展开 4-bit→8-bit），**MMA 实际占空比 ≈ 1/9 ≈ 11%**，这就是 TC 占用率 <30% 的根因。

### J.2 PTX `prmt.b32` 优化方案

`prmt.b32` 是 **Permute Bytes** 指令，一条指令做 4-byte 任意字节级重排。用它 **一次性从 1 个 packed uint32（8 个 u4）展开成 2 个 uint32（各含 4 个 u8）**：

```
// 原始 32-bit: [h7 l7 h6 l6 h5 l5 h4 l4 h3 l3 h2 l2 h1 l1 h0 l0]  (nibble-packed LE)
// 目标 lo_u32: [0 l3 0 l2 0 l1 0 l0]   (sign-extended to s8)
// 目标 hi_u32: [0 h3 0 h2 0 h1 0 h0]
```

等价于**两条 `prmt.b32` + 两条 `lop3.b32`**（vs 当前"每 byte 1 条 `& 0x0F`、1 条 `>> 4`、1 条 `PRMT` 去 join"，共 ~6 条）。

**预期收益**（AWQ / Marlin 实测参考）：
- 相同 TC tile 下 HMMA issue-rate 从 ~11% → ~25%
- dense/fp16 从 1.27x → **0.90–1.00x**（即追平 cuBLAS FP16）

### J.3 Triton 里怎么写：`tl.inline_asm_elementwise`

Triton ≥ 2.2 提供 `tl.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack)`：

```python
@triton.jit
def _unpack_packed_s4_ptx(packed, BM: tl.constexpr, BK: tl.constexpr):
    # packed : (BM, BK // 2) int8 viewed as packed u32 tiles of 4 bytes each
    # Treat as u32 lane by grouping 4 bytes along K axis (requires BK_half % 4 == 0)
    BK_HALF: tl.constexpr = BK // 2
    tl.static_assert(BK_HALF % 4 == 0, "BK_half must be multiple of 4 for u32 lanes")

    # Reinterpret as u32 tiles: shape (BM, BK_HALF//4)
    packed_u32 = tl.reshape(packed, (BM, BK_HALF // 4, 4))
    packed_u32 = tl.view(packed_u32, (BM, BK_HALF // 4), dtype=tl.uint32)

    # Each u32 lane holds 8 nibbles.  Extract low 4 nibbles (bits 0,8,16,24 → byte lanes)
    # via prmt.b32 with per-lane selector.  prmt's imm byte selector:
    #   lo: 0x5140  → byte0=byte[0]&0x0F, byte1=byte[1]&0x0F, ...  [not quite]
    # In practice the standard AWQ recipe is:
    #   asm = """{
    #     .reg .b32 tmp;
    #     lop3.b32 tmp, $1, 0x0F0F0F0F, 0, 0xC0;   // mask low nibbles
    #     prmt.b32 $0, tmp, tmp, 0x0000;           // broadcast? (simplified)
    #   }"""
    # — see Marlin dequant for the production-grade sequence.

    lo_u32, hi_u32 = tl.inline_asm_elementwise(
        asm="""
        lop3.b32 $0, $2, 0x0F0F0F0F, 0x00000000, 0xC0;    // lo  = x & 0x0F0F0F0F
        prmt.b32 $1, $2, $2, 0x7362;                       // hi  = bytes [3>>4, 2>>4, 1>>4, 0>>4]
        """,
        constraints="=r,=r,r",
        args=[packed_u32],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    # Now reinterpret lo/hi back to (BM, BK_HALF) int8 each, then interleave along K.
    low_i8  = tl.view(lo_u32, (BM, BK_HALF), dtype=tl.int8)
    high_i8 = tl.view(hi_u32, (BM, BK_HALF), dtype=tl.int8)
    stacked = tl.join(low_i8, high_i8)                     # (BM, BK_HALF, 2)
    return tl.reshape(stacked, (BM, BK))
```

> ⚠️ 上面 `prmt` 的 selector `0x7362` 是**示意性伪 selector**。
> 生产级 byte selector 需要针对"每 byte 的高 4 位→到下一个 byte 的低 4 位"精确计算；参考 Marlin 和 AWQ 的 `dequant.h`（GitHub: `NVIDIA/TensorRT-LLM` / `mit-han-lab/llm-awq`）。

### J.4 验证步骤（改完 kernel 必做）

```bash
# 1. 精度对齐
pytest triton_kernel/tests/test_dense.py -v

# 2. 反汇编确认 HMMA 前面 ALU 指令数下降
python - <<EOF
import torch, triton
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_kernel
# force compile + dump
import os; os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_ptx"
# ... compile-only launch ...
EOF
find /tmp/triton_cache_ptx -name "*.cubin" -exec cuobjdump --dump-sass {} \; \
  | grep -E "HMMA|LOP3|PRMT|SHR" | head -50

# 3. Tensor Core 利用率（必看）
ncu --section SpeedOfLight \
    --kernel-name regex:dense_gemm_kernel \
    python -m triton_kernel.benchmarks.bench_dense
# 关注：sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
# 目标：从 ~30% → >50%

# 4. 端到端 sweep 对比
python -m triton_kernel.benchmarks.sweep_v9
python -m research.tools.analyze_sweep_bottleneck results/sweep_<ts>.csv
# 目标：prefill dense/fp16 从 1.27x → <1.10x
```

---

## §K. 4090 Shared Memory Budget 推导

本节说明当前 autotune 为什么止步于 `BM=256, BN=256, stages=3`，以及在哪个维度扩展最安全。

### K.1 RTX 4090 (SM89) SMEM 规格

| 项 | 值 | 备注 |
|---|---|---|
| 每 SM SMEM 物理容量 | **128 KB** | L1$ + shared 共享 |
| 可切成 shared 的上限 | **100 KB** per kernel | 驱动预留 ~28 KB for L1/system |
| 每 CTA 最大 SMEM | **99 KB**（动态）或 48 KB（静态） | Triton 走动态 |
| 每 SM CTA 数上限 | 16 | SMEM 足够时 |
| 每 SM warp 上限 | 48 | = 32 threads × 48 = 1536 threads |

Triton 编译时实际预留：`shmem_usage = max(load_tile_bytes × stages, mma_operand_bytes × (stages-1)) + acc_bytes + misc`。

### K.2 Dense kernel 的 SMEM 账单

对每个 config：`BM × BN × BK × stages`

**Load tiles（async pipeline）**：
- W tile (packed s4): `BM × BK/2` bytes × `stages`
- X tile (packed s4): `BN × BK/2` bytes × `stages`
- Scale / zero per group: `BM × (BK/BCOL) × 2` bytes × stages（fp16）
- sum_X: `BN × (BK/BCOL) × 4` bytes × stages（int32）

**Accumulator**（常驻 RF，不占 SMEM）：`BM × BN × 4` bytes int32

**实例计算（当前最大 config `BM=256, BN=256, BK=128, stages=3`）**：
```
W_tile   : 256 × 64 × 3  = 49152 B = 48.0 KB
X_tile   : 256 × 64 × 3  = 49152 B = 48.0 KB
scale    : 256 × 1 × 2 × 3 = 1536 B  (×2 for W and X sides)
zero     : 256 × 1 × 2 × 3 = 1536 B
sum_X    : 256 × 1 × 4 × 3 = 3072 B
-------------------------------------------------
TOTAL    : ~99.5 KB  ★ 已逼近 100KB hard limit
```

→ **`BM=256, BN=256, stages=3` 已经是 4090 物理上限**。`stages=4` 时会溢出，Triton 会自动退化成 `stages=3` 或编译失败。

### K.3 能再扩的方向（安全边界）

**方向 A：提升 stages 而非 tile 尺寸**（最佳性价比）
```
BM=128, BN=128, BK=128, stages=4
  W+X : 2 × 128 × 64 × 4 = 64 KB
  其他 : ~10 KB
  total ~74 KB  ✅ 安全
```

**方向 B：不对称 tile（prefill 形状更常见 d_out > T）**
```
BM=256, BN=128, BK=128, stages=3
  W : 256 × 64 × 3 = 48 KB
  X : 128 × 64 × 3 = 24 KB
  total ~76 KB  ✅ 最佳
```

**方向 C：扩 BK（需要改约束！）**
- `BK` 强绑 `BCOL=128`（组内反量化），**不能改**
- 若想 `BK=256`，需要在 kernel 内做**两次 scale 广播 + 两次 sum_X accumulation**，改动大

### K.4 验证命令

```bash
# 查看编译后 kernel 的实际 SMEM 使用
python - <<EOF
import torch
from kernel.triton_kernel.dense_u4s4_gemm import dense_gemm_kernel
# trigger compilation for a shape
... (小 warmup) ...
# read .shared_memory attribute
import triton
for cfg in dense_gemm_kernel.cache:
    print(cfg, "->", dense_gemm_kernel.cache[cfg].shared_memory)
EOF

# 或通过 ncu
ncu --section LaunchStats --kernel-name regex:dense_gemm_kernel <cmd>
# 关注：launch__shared_mem_per_block.max
```

### K.5 Occupancy 联动

SMEM 用满不是唯一考量，还要看每 SM 能同时住几个 CTA：

| config | SMEM/CTA | 每 SM CTA 上限 | warp 上限 | **实际 occupancy** |
|---|---|---|---|---|
| 128×128 st=3 | ~40 KB | 100/40 = 2 | 2 × (128×128/1024)×2 warps = 2×8=16 / 48 | **33%** |
| 128×128 st=4 | ~52 KB | 100/52 = 1 | 1×8 = 8 / 48 | **17%** |
| 256×256 st=3 | ~99 KB | 100/99 = 1 | 1×16 = 16 / 48 | **33%** |
| 256×128 st=3 | ~76 KB | 100/76 = 1 | 1×12 = 12 / 48 | **25%** |

→ **在 TC-bound 场景，占用率 >25% 就够用**（HMMA 本身吃不满），所以 `256×128 st=3` 是甜点：**tile 最大 + occupancy 充足**。

---

## §L. W4A16 Fallback 阈值调参方法

本节说明 `v9_linear.py` L295-302 的 `use_w4a16` 阈值是怎么得出来的，以及迁移到其他 GPU 时怎么重调。

### L.1 当前阈值

```python
# v9_linear.py  _v9_forward_prefill
use_w4a16 = (
    W.n_hp_blocks == 0                                   # 必须纯 dense（无 sparse 补偿）
    and (
        T >= 1024                                         # 大 T 时 W4A16 稳赢
        or (T >= 512 and d_out * d_in <= 4096 * 4096)    # 小权重 + 中 T 也可以走
    )
)
```

### L.2 背后的权衡模型

W4A16 path 成本：
```
t_w4a16 = t_dequant(W) + t_cuBLAS_hgemm(X @ W.T)
        ≈ (2 × d_out × d_in / BW_HBM) + (2 × T × d_out × d_in / FLOPS_TC_fp16 × η_fp16)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          memory-bound, 不摊薄            随 T 线性增长
```

W4A4 native path 成本：
```
t_w4a4  = t_quant(X) + t_dense_gemm(W, X)
        ≈ (2 × T × d_in / BW_HBM) + (T × d_out × d_in / FLOPS_TC_s4 × η_s4)
          ^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          随 T 线性                  TC_s4 更快，但 η_s4 低（当前 30%）
```

**交叉点**：`t_w4a16 == t_w4a4` 时的 T：
```
T_cross ≈ t_dequant / (t_fp16_gemm_per_T − t_s4_gemm_per_T)
        = (2 × d_out × d_in / BW) / (d_out × d_in × (1/TC_fp16·η_fp16 − 1/TC_s4·η_s4))
```

把当前数 (`BW=1008GB/s, TC_fp16=330T, η_fp16=0.85, TC_s4=1320T, η_s4=0.30`) 代入：
- 分子：`2 × d_out × d_in × 2B / 1008e9` = `4×d_out×d_in / 1.008e12` 秒
- 分母：`d_out × d_in × (1/280 − 1/396) / 1e12` 秒/token

消项：`T_cross ≈ 4 / ((1/280 − 1/396)) ≈ 4 / 0.00105 ≈ 3800 tokens`

但实测 T>=1024 就切 W4A16 已经赢——这是因为 **η_s4 在小 T 时更低**（tile 太大，尾块空转），并且 dequant kernel 比理论值快很多（HBM 利用率 60%）。

### L.3 调参 Playbook

#### Step 1：采集 A/B 数据
```bash
python -m triton_kernel.benchmarks.bench_phase_b1_compare \
    --shapes "4096x4096,4096x11008,11008x4096,14336x4096,28672x4096" \
    --bs_list "128,256,512,1024,2048,4096,8192" \
    --modes w4a4,w4a16
# 输出: results/phase_b1_<ts>.csv  含 mode, shape, bs, ms
```

#### Step 2：画交叉点曲线
```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results/phase_b1_<ts>.csv")
for (d_out, d_in), g in df.groupby(["d_out", "d_in"]):
    piv = g.pivot(index="bs", columns="mode", values="ms")
    plt.plot(piv.index, piv["w4a4"], label=f"{d_out}x{d_in} w4a4")
    plt.plot(piv.index, piv["w4a16"], "--", label=f"{d_out}x{d_in} w4a16")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("T (batch size)"); plt.ylabel("ms")
plt.legend(); plt.savefig("w4a4_vs_w4a16_crossover.png")
```

#### Step 3：决定 per-shape 阈值

从图上读出**每条曲线的交叉点 T\***，构造查表：
```python
# v9_linear.py 里可以做更精细的判据
CROSSOVER_TABLE = {
    # (d_out, d_in): T_cross
    (4096, 4096):  384,
    (4096, 11008): 512,
    (11008, 4096): 768,
    (14336, 4096): 1024,
    (28672, 4096): 1024,
    (8192, 14336): 512,
}
def should_use_w4a16(d_out, d_in, T, hp_blocks):
    if hp_blocks > 0: return False
    T_cross = CROSSOVER_TABLE.get((d_out, d_in), 1024)  # safe default
    return T >= T_cross
```

但当前代码选了**更简单的 2 条规则**（T>=1024 always；T>=512 if small shape），权衡点：
- ✅ 无需维护 per-shape 表
- ✅ 对新 shape 的 graceful degradation 是"继续跑 W4A4"（最坏情况仍可用）
- ❌ 错过 8% 的 fine-grain 收益

#### Step 4：迁移到新 GPU 时必跑

换卡（H100 / L40S / A100）时，`BW_HBM / FLOPS_TC` 都变：
- **H100**: HBM=3TB/s, TC_fp16=989T, TC_s4=3960T → T_cross 下降（dequant 变快）→ W4A16 覆盖更大 T 区间
- **A100**: HBM=1.55TB/s, TC_fp16=312T, TC_s4=1248T → 类似 4090
- **L40S**: HBM=864GB/s, TC_fp16=362T → 比 4090 略差

重跑步骤：
```bash
# 1. 确保 dense kernel 的 autotune 在新卡上已预热
python -c "from kernel.triton_kernel.benchmarks.sweep_v9 import main; main()"
# 2. 跑 A/B 对比
python -m triton_kernel.benchmarks.bench_phase_b1_compare --modes w4a4,w4a16
# 3. 更新 v9_linear.py 的 use_w4a16 阈值
# 4. pytest + sweep 回归
```

### L.4 阈值失效的症状

| 症状 | 说明 | 排查 |
|---|---|---|
| 端到端 prefill 比预期慢 20%+ | 可能走错路径 | 在 `_v9_forward_prefill` 里加 `logger.debug` 打 `use_w4a16` 值 |
| W4A16 在小 T 下反而更慢 | 阈值太低 / dequant kernel 没 autotune | 跑 `bench_phase_b1_compare`；看 dequant 是否命中最佳 config |
| 某些 shape speedup 停滞在 0.85x | 该 shape 处于交叉点附近，两路径都不理想 | 考虑 §J 的 PTX dequant 提升 W4A4 路径 |

---

## §M. CUDA Graph 接入 Decode Path 的具体步骤

本节给出**把整条 decode pipeline 录制成 CUDA Graph** 的可执行方案，用于摊薄 decode 下 44% 的 launch overhead。

### M.1 为什么 decode 最适合 CUDA Graph

Decode 场景（T ≤ 16，autoregressive）：
- **T 固定**：每步都是 T=1 或小固定值
- **shape 稳定**：d_out, d_in 整层不变
- **权重地址不变**：V9WeightContainer 是 long-lived
- **输入/输出 tensor 地址**可以用 `static buffers` 预分配

而 prefill 场景 T 随 seq_len 变，不适合 graph。

### M.2 5 阶段 launch 开销量化

当前 decode path 的 kernel launch 开销（用 `nsys` 实测，commit `21f446f` 前后）：

| Stage | kernel | 单次 launch | 占 decode 总时间 |
|---|---|---|---|
| quant | `quantize_activation_kernel` | ~5 μs | 15% |
| dense | `dense_gemm_kernel` | ~5 μs | 12% |
| sparse | `sparse_gemm_kernel` | ~5 μs（仅 hp>0） | 12% |
| combine | `_combine_transpose_kernel` 或 torch | ~3 μs | 7% |
| 各 stage 间 stream sync / metadata | — | ~5-10 μs | — |
| **合计** | — | **~25-35 μs** | **~40-50%** |

Graph replay 后每次总 overhead → **~5 μs**（只需一次 launch），**节省 20-30 μs**。

### M.3 接入步骤（完整代码）

在 `v9_linear.py` 末尾新增一个子模块 `V9DecodeGraphRunner`：

```python
# ---------------------------------------------------------------------------
# CUDA Graph runner for decode path
# ---------------------------------------------------------------------------
class V9DecodeGraphRunner:
    """
    Capture the decode forward path as a CUDA Graph for a fixed (T, d_in, d_out) shape.
    Replay amortizes ~25us of launch overhead per step down to ~5us.

    Usage:
        runner = V9DecodeGraphRunner(W, T=1, d_in=4096)
        runner.warmup()          # triggers triton autotune + graph capture
        Y = runner(X)            # fast path; X must match (T, d_in) and dtype

    Constraints:
        - X and Y are static buffers; caller must copy into runner.X_static
        - If T or hp_blocks changes, need a new runner
    """

    def __init__(self, W: V9WeightContainer, T: int, d_in: int):
        assert T <= DECODE_T_THRESHOLD, "graph runner only for decode"
        assert W.d_in == d_in
        self.W = W
        self.T = T
        self.d_in = d_in
        self.d_out = W.d_out

        # --- Static I/O buffers (owned by the runner) ---
        device = W.W_low_packed.device
        self.X_static = torch.empty((T, d_in), dtype=torch.float16, device=device)
        self.Y_static = torch.empty((T, W.d_out), dtype=torch.float16, device=device)

        self._graph = None
        self._captured = False

    def _run_once(self, X_in: torch.Tensor, Y_out: torch.Tensor):
        """One full decode forward, writing into Y_out."""
        # Exactly replicates _v9_forward_decode but onto static buffers
        X_s4, scale_x, sum_X = quantize_activation(X_in)
        Y_low = dense_gemm_forward(
            self.W.W_low_packed, X_s4, self.W.scale_u4, self.W.zero_u4,
            sum_X, scale_x, BCOL=128,
        )
        if self.W.n_hp_blocks > 0:
            Y_high = sparse_gemm_forward(
                self.W.W_high_blocks_packed, self.W.hp_row_offsets,
                self.W.hp_col_indices, self.W.scale_u4,
                X_s4, scale_x, sum_X, self.W.d_out, self.T, self.d_in,
            )
            _combine_transpose(Y_low, Y_high, out=Y_out)
        else:
            _combine_transpose(Y_low, None, out=Y_out)

    def warmup(self, n: int = 3):
        """Must call before capture.  Triggers autotune and PTX compilation."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n):
                self._run_once(self.X_static, self.Y_static)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

    def capture(self):
        """Record the graph."""
        self._graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(self._graph):
                self._run_once(self.X_static, self.Y_static)
        torch.cuda.current_stream().wait_stream(s)
        self._captured = True

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        assert self._captured, "call warmup() + capture() first"
        assert X.shape == self.X_static.shape and X.dtype == self.X_static.dtype
        self.X_static.copy_(X, non_blocking=True)
        self._graph.replay()
        # Return a clone so caller can hold it independently of static buffer
        return self.Y_static.clone()
```

### M.4 使用与测试

```python
# tests/test_cuda_graph.py
def test_cuda_graph_matches_eager():
    W = make_test_weights(d_out=4096, d_in=4096)
    X = torch.randn(1, 4096, dtype=torch.float16, device="cuda")

    # eager reference
    Y_eager = v9_linear_forward_decode(X, W)

    # graph
    runner = V9DecodeGraphRunner(W, T=1, d_in=4096)
    runner.warmup(n=3)
    runner.capture()
    Y_graph = runner(X)

    torch.testing.assert_close(Y_graph, Y_eager, rtol=1e-3, atol=5e-3)
```

### M.5 Bench 脚本

```python
# triton_kernel/benchmarks/bench_decode_cuda_graph.py
from kernel.triton_kernel.benchmarks._bench_util import bench_kernel
from kernel.triton_kernel.v9_linear import (
    v9_linear_forward_decode, V9DecodeGraphRunner,
)

def main():
    for (d_out, d_in) in [(4096,4096), (11008,4096), (14336,4096), (28672,4096)]:
        W = make_test_weights(d_out, d_in)
        X = torch.randn(1, d_in, dtype=torch.float16, device="cuda")

        # Eager
        t_eager = bench_kernel(
            lambda: v9_linear_forward_decode(X, W),
            warmup=50, windows=3, iters=100,
        )

        # Graph
        runner = V9DecodeGraphRunner(W, T=1, d_in=d_in)
        runner.warmup(); runner.capture()
        t_graph = bench_kernel(lambda: runner(X), warmup=50, windows=3, iters=100)

        logger.info(
            "d_out=%d d_in=%d | eager=%.4f ms | graph=%.4f ms | speedup=%.2fx",
            d_out, d_in, t_eager, t_graph, t_eager / t_graph,
        )
```

**预期结果**（基于 nsys 实测的 launch overhead 分解）：
- `4096×4096 bs=1`: eager 0.200 ms → graph 0.175 ms (**−12%**)
- `28672×4096 bs=1`: eager 0.350 ms → graph 0.325 ms (**−7%**)
- 收益取决于 dense kernel 本身占比；dense 越大（HBM-bound），graph 收益越小

### M.6 集成到 dispatcher

修改 `v9_linear_forward`：

```python
# Simple cache: {(shape_key) -> runner}
_DECODE_GRAPH_CACHE: Dict[Tuple[int,int,int,int], V9DecodeGraphRunner] = {}

def v9_linear_forward(X: torch.Tensor, W: V9WeightContainer,
                      use_graph: bool = False) -> torch.Tensor:
    T = X.numel() // W.d_in
    if T > DECODE_T_THRESHOLD:
        return _v9_forward_prefill(X, W)

    if use_graph:
        key = (T, W.d_in, W.d_out, W.n_hp_blocks)
        runner = _DECODE_GRAPH_CACHE.get(key)
        if runner is None:
            runner = V9DecodeGraphRunner(W, T=T, d_in=W.d_in)
            runner.warmup(); runner.capture()
            _DECODE_GRAPH_CACHE[key] = runner
        return runner(X)

    return _v9_forward_decode(X, W)
```

### M.7 常见坑

| 坑 | 原因 | 处理 |
|---|---|---|
| `RuntimeError: CUDA error: operation not permitted when stream is capturing` | autotune 在 capture 时触发编译 | warmup 次数不够，多跑几次 |
| Graph 精度和 eager 差很多 | static buffer 被外部写入污染 | runner 内部保持 buffer 私有；每次 `X_static.copy_(X)` |
| 改 hp_ratio 后 graph 崩 | control flow 变了（n_hp_blocks 分支） | 按 (n_hp_blocks==0) 分两个 runner |
| 内存占用爆涨 | 每个 (d_out, d_in, T) 一个 graph | 加 LRU cache，限制 size ≤ 32 |
| 第一次 call 很慢 | autotune + graph capture 合计 ~200ms | 模型启动时统一 warm-up 所有 shape |

### M.8 验证 & 监控

```bash
# 单测
pytest triton_kernel/tests/test_cuda_graph.py -v

# bench
python -m triton_kernel.benchmarks.bench_decode_cuda_graph

# nsys 确认 launch 从 5 kernel/iter 变成 1 graph/iter
nsys profile -o /tmp/graph_check \
    python -m triton_kernel.benchmarks.bench_decode_cuda_graph
nsys stats --report cuda_api_sum /tmp/graph_check.nsys-rep | head -20
# 期望：cudaLaunchKernel 次数下降，cudaGraphLaunch 出现
```

---

## §N. Glossary

| 术语 | 含义 |
|---|---|
| **BCOL** | 量化 group size (=128)；也是 K-tile size |
| **BSR** | Block Sparse Row format, 用 `(indptr, indices, values)` 存块稀疏矩阵 |
| **HP block** | High-Precision block, W_high 的 128×128 sparse block |
| **u4 / s4** | UINT4 (0..15) / SINT4 (-8..7)，容器均为 int8 |
| **W4A4** | 4-bit weight + 4-bit activation |
| **W4A16** | 4-bit weight + fp16 activation (fallback 用) |
| **TC** | Tensor Core |
| **HBM** | High-Bandwidth Memory, 4090 spec 1008 GB/s |
| **act-order / perm** | GPTQ 按 Hessian diag 重排 K 轴，恢复时 input 侧做 `X[:, perm]` |
| **swizzle / GROUP_SIZE_M** | program id 重排，把相邻 M-block 调度到一起提升 L2 复用 |
| **Split-K** | 把 K-loop 切多段用 atomic 累加，提升 prefill TC 占用 |
| **Epilogue** | GEMM kernel 的尾段（bias add / activation / 转置），理想在 SRAM 完成 |
