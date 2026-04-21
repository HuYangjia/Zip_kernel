# V9 Triton Kernel 开发前置约定

> **文档类型**：Triton 算子开发的前置提示（Prompt）与硬件/调度约定
> **目标硬件**：NVIDIA RTX 4090 (SM89, Ada Lovelace)
> **前置文档**：
> - [kernel_algorithm.md](./kernel_algorithm.md) — V9 True Quant 算法逻辑（**必读**）
> - [model.md](./model.md) — 量化方案与训练时逻辑
> **目的**：在进入 Triton 编码之前，固化硬件能力、调度策略、数据布局等关键决策，作为每个 kernel 开发时的共享背景约定
> **日期**：2026-04-20

---

## 0. 文档定位

本文档 **不重复** `kernel_algorithm.md` 中的算法推导，只固化开发 Triton 算子前必须敲定的工程决策，具体包括：

1. **硬件能力确认**：RTX 4090 的 4-bit MMA 支持情况
2. **Kernel ② 的调度策略**：按什么维度切 tile、如何消除 scatter atomic
3. **激活量化 kernel 的产出布局**：`X_s4 / scale_x / sum_X` 如何组织
4. **全局性能优先原则**：在若干设计选择上的取向

后续编写任何一个 Triton 算子（激活量化、Kernel ①、Kernel ②）时，**以本文档为前置上下文**，避免决策漂移。

---

## 1. 硬件能力确认：RTX 4090 原生支持 INT4 Tensor Core

### 1.1 官方依据

根据 NVIDIA 官方发布的 **Ada Lovelace GPU 架构白皮书**：
<https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf>

RTX 4090 (SM89, Ada Lovelace) 的第四代 Tensor Core 原生支持以下整数精度：

| 精度 | 是否原生支持 | 备注 |
|------|-------------|------|
| **INT8** | ✅ | 660 TOPS |
| **INT4** | ✅ | **1320 TOPS，本方案主力** |
| FP8 (E4M3/E5M2) | ✅ | 新引入 |
| FP16 / BF16 | ✅ | 标准精度 |

**结论**：本方案的两个 GEMM kernel（`UINT4 × SINT4` 和 `SINT4 × SINT4`）都可直接走**原生 INT4 Tensor Core**，无需走 INT8 模拟。

### 1.2 对应的 PTX/MMA 指令

- **主指令**：`mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`（纯 SINT4 × SINT4）
- **混合符号**：`mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32`（UINT4 × SINT4，Kernel ① 用）与 `s4.u4` 变体
- **累加类型**：INT32（K 维累加精度充足）
- **数据加载**：`ldmatrix.sync.aligned.m8n8.x4.b16` 配合 4-bit 寄存器装配

### 1.3 对 Triton 的映射

- Triton `tl.dot(a, b, acc, out_dtype=tl.int32)` 在 SM89 上，当 `a`、`b` 为 `tl.int8` 且实际值域落在 `[-8, 7]`（对 s4）或 `[0, 15]`（对 u4）时，可自动 lower 到 INT4 MMA（需要 Triton 版本支持；必要时退回 INT8 Tensor Core，性能仍然可观）
- **关键约束**：两个 kernel 都按"K 维 packed 4-bit"设计，单个寄存器元素宽度 4 bits，`BLOCK_K` 必须是 64 的倍数（匹配 `m16n8k64`）

### 1.4 Kernel ① 的 `u4 × s4` 实现方式（推荐）

Triton 当前版本对 `u4 × s4` 的原生混合符号 MMA 支持可能不完整。**推荐做法**：在加载 `W_low` 时做**离线偏移转换**到 SINT4：

```
W_low_s4 = W_low_u4 - 8         ∈ [-8, 7]
Y_u4_s4  = W_low_u4 @ X_s4
         = (W_low_s4 + 8) @ X_s4
         = W_low_s4 @ X_s4  +  8 × sum_X (按 group broadcast)
```

- **存储侧**：`W_low` 仍以 UINT4 打包（物理上 `[0, 15]`），或直接预减 8 存为 SINT4（由打包工具决定）
- **Kernel 侧**：统一用 `s4 × s4` MMA；epilogue 里把 `+ 8 × sum_X` 项作为已预计算的 broadcast 项修正
- **好处**：两个 kernel 的 MMA 指令**完全一致**（都是 `s4 × s4`），kernel 模板最大化复用

> 本项决策（统一走 `s4 × s4`）在全局生效，所有 Triton 算子以此为准。

---

## 2. 全局调度原则：以算子性能最优为第一目标

### 2.1 总原则

本项目中所有 Triton 算子的设计、tile 切分、pipeline、data layout 选型**以实测性能为单一目标**，具体表现为：

1. **优先最大化 Tensor Core 利用率**（INT4 MMA `m16n8k64` 吞吐 1320 TOPS）
2. **次优先最小化 HBM 读写**（权重 4-bit packed 减半 + 激活 4-bit 减半）
3. **允许稍微复杂的离线布局**（只要能换来运行时性能），离线打包工具承担复杂度
4. **epilogue 尽量融合**：反量化、zero 项修正、`+ 8 × sum_X` 项全部在 GEMM kernel 内完成，避免单独的 dequant kernel
5. **kernel 数量最小化**：端到端单次 Linear 只启动 **3 个 kernel**（激活量化 + Kernel ① + Kernel ②），不引入辅助 kernel

### 2.2 对每个 kernel 的性能评估口径

- **基准对比对象**：cuBLAS FP16 GEMM（`torch.nn.Linear` FP16 forward）
- **目标加速比**：端到端 Linear（Kernel ① + Kernel ② + 激活量化）相对 FP16 基线 **≥ 3×**（保守目标；理论 roofline 约 6×）
- **分 kernel 目标**：
  - 激活量化 kernel：带宽受限，接近 HBM 带宽上限（RTX 4090 为 1008 GB/s）
  - Kernel ①：计算受限，INT4 Tensor Core 利用率 ≥ 60%
  - Kernel ②：5% 稀疏下实际开销 ≤ 5% × Kernel ① 的延迟

### 2.3 不做的事情

- **不手写 PTX**：Triton 表达能力足够；仅在性能瓶颈确需时降级到 `inline asm` 的特定 `ldmatrix` / `mma`
- **不为了可移植性牺牲性能**：本项目只针对 SM89（RTX 4090），tile 尺寸、pipeline stages 等可以对 SM89 特化
- **不在线做位级拆分**：所有位级拆分、scale 覆盖都在离线打包工具中完成（见 `kernel_algorithm.md` §4.4）

---

## 3. Kernel ② 的调度策略：按输出行 tile 分桶，消除 scatter atomic

### 3.1 选型结论

**采用"按输出行 tile 调度"模型**（非"按 (br, bc) 独立块调度"）：

- **Grid 维度**：`grid = (ceil(d_out / BM), ceil(batch*seq / BN))`（与 Kernel ① 的 grid 完全同形）
- **每个 Triton program**：处理一个输出 tile `Y_high[r0:r1, j0:j1]`
  - 在 kernel 内部遍历**属于该输出行 `br = r0 / brow` 的所有高位块**（即 K 维稀疏 loop）
  - 直接在 shared memory / 累加寄存器中做累加，**kernel 结束时一次性写回**，无 scatter-add

### 3.2 为什么选这个方案

| 候选方案 | 优点 | 缺点 | 选择 |
|---------|------|------|------|
| (a) 按 `(br, bc)` 块独立调度 | 调度简单 | 需 scatter-add 或按 `br` barrier；atomic 在 INT32 上对性能很不友好 | ❌ |
| **(b) 按输出行 tile 调度** | **无 atomic、无 scatter；与 Kernel ① grid 同形；epilogue 统一** | 每个 program 需读取"自己这行的块索引列表"，需要额外的 row-indexed 元数据 | ✅ |
| (c) 按 K 维 tile 并行 + 规约 | 最大并行度 | 需要 split-K reduction kernel，引入额外 kernel | ❌（违反 §2.1 第 5 点）|

### 3.3 离线打包侧的必要支持

为了让 Kernel ② 能以"输出行 tile"为维度高效访问，离线打包必须额外生成 **BSR 风格的行索引元数据**（对 `kernel_algorithm.md` §4.3 的增强）：

```
hp_row_offsets    (nrow + 1,)                INT32   ← 类似 CSR 的 indptr
                                                       hp_row_offsets[br] .. hp_row_offsets[br+1] 是属于输出块行 br 的块范围
hp_col_indices    (n_hp_blocks,)             INT32   ← 排序后每块的 bc 坐标
W_high_blocks     (n_hp_blocks, brow, bcol)  SINT4   ← 按 br 升序排列后的块数据
```

**打包约束**：
1. `hp_block_indices` 按 `br` 升序排序，同 `br` 内按 `bc` 升序
2. 排序后同步 `W_high_blocks` 的第 0 维以保持对应
3. `hp_row_offsets[br+1] - hp_row_offsets[br]` 给出输出行 `br` 的高位块数量（典型值 0~ncol × 5%）

### 3.4 Kernel ② 伪代码

```python
@triton.jit
def kernel_high_sparse(
    W_high_blocks_ptr,      # (n_hp_blocks, brow, bcol/2) UINT8 packed s4
    hp_row_offsets_ptr,     # (nrow+1,) INT32
    hp_col_indices_ptr,     # (n_hp_blocks,) INT32
    X_s4_ptr,               # (batch*seq, d_in/2) UINT8 packed s4
    scale_u4_ptr,           # (d_out, n_groups) FP16
    scale_x_ptr,            # (batch*seq,) FP16
    Y_high_ptr,             # (d_out, batch*seq) FP16 ← 直接写回，无 atomic
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    BROW: tl.constexpr, BCOL: tl.constexpr,
    ...
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    br = (pid_m * BM) // BROW     # 本 tile 所属的 block_row

    # 从 BSR 读取本 br 的块范围
    start = tl.load(hp_row_offsets_ptr + br)
    end   = tl.load(hp_row_offsets_ptr + br + 1)

    acc = tl.zeros((BM, BN), dtype=tl.int32)

    # K 维稀疏 loop：只访问属于本 br 的块
    for block_idx in range(start, end):
        bc       = tl.load(hp_col_indices_ptr + block_idx)
        W_tile   = load_block_s4(W_high_blocks_ptr, block_idx, ...)    # (BROW, BCOL) s4
        X_tile   = load_X_s4(X_s4_ptr, bc * BCOL, pid_n * BN, ...)     # (BCOL, BN) s4
        acc     += tl.dot(W_tile, X_tile, out_dtype=tl.int32)

    # Epilogue: dequant + write（无 scatter，无 atomic）
    # 16 × scale_u4[i, bc] × scale_x[j] × acc
    # 注意 Kernel ② 的 epilogue 里 scale 依赖 bc，
    # 实际是 Σ_{block} scale_u4[i, bc] × acc_block，因此要在 loop 内按块做 partial dequant 累加
    Y_high_ptr_tile = Y_high_ptr + ...
    tl.store(Y_high_ptr_tile, result_fp16, mask=...)
```

**注意**：Kernel ② 的 epilogue 因为 scale 依赖于块索引 `bc`，**反量化必须在 K 维 loop 内部按块做**（而不是 loop 结束后统一 dequant）。这与 Kernel ① 的按 group 累加反量化结构类似。

### 3.5 Kernel ② scatter-add 问题彻底消失

由于每个 Triton program 独占一个输出 tile `Y_high[r0:r1, j0:j1]`，不同 program 之间**不共享写目标**，所以：

- ✅ 无 atomic
- ✅ 无 split-K reduction kernel
- ✅ 一次 `tl.store` 完成写回

这也是 `kernel_algorithm.md` §7.6（检查点 6）的工程落地。

---

## 4. 激活量化 kernel 的产出布局

### 4.1 融合为单个 kernel

激活量化负责在一次 kernel 调用内同时产出：

| 产出 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `X_s4` | `(batch*seq, d_in)` packed 4-bit | SINT4 | 按行 pack，2 元素/字节 |
| `scale_x` | `(batch*seq,)` | FP16 | per-token 对称 scale = max(|X[t,:]|) / 7 |
| `sum_X` | `(batch*seq, n_groups)` | INT32 | `Σ_{k ∈ group g} X_s4[t, k]`，给 UINT4 zero 项修正用 |

### 4.2 融合的理由（性能）

- 激活量化本身是**带宽受限**的（读 FP16 X，写 SINT4 + FP16 + INT32 元数据）
- 若拆成 3 个 kernel（量化 / 存 pack / 算 sum_X），需要 3 次 HBM 往返
- 融合为 1 个 kernel：一次读入 X，寄存器内算完所有产出，一次写回，**至少减少 2× 的 HBM 压力**

### 4.3 实现要点

- **Grid**：`grid = (ceil(batch*seq / BT),)`，每个 program 处理 `BT` 个 token
- **Row 内两次 pass**：
  1. Pass 1：读入 `X[t, :]`，求 `max(|X[t, :]|)` → `scale_x[t]`
  2. Pass 2：用 `scale_x[t]` 量化得到 `q ∈ [-8, 7]`；同时按 `BCOL` 分块累加 `sum_X`；pack 2 个 q 到 1 字节
- 若 `d_in` 不大（≤ 16K），可将 `X[t, :]` 缓存在 shared memory / 寄存器中完成两次 pass
- **BCOL 对齐**：`sum_X` 按 bcol 累加，bcol = 128 是当前设计默认值

### 4.4 `X_s4` 的 pack 约定

- **4-bit packing 方向**：**小端**，即 `byte[i] = (x[2i+1] << 4) | (x[2i] & 0x0F)`
- 物理上 `SINT4 ∈ [-8, 7]` 需要保持二进制补码：`x[i] & 0x0F` 对负数也正确（`-1 → 0xF, -8 → 0x8`）
- 与 `W_low` 的 pack 方向**必须一致**（都是小端），否则 MMA 装配不对齐

---

## 5. Kernel ① 的关键布局与策略

### 5.1 Grid 与 Tile

- **Grid**：`grid = (ceil(d_out / BM), ceil(batch*seq / BN))`
- **推荐起点 tile**：`BM=128, BN=128, BK=128`（与 `bcol=128` 对齐）
  - `BK = bcol`：每一轮 K 维 loop 恰好覆盖一个 group，group 累加/反量化天然按 K 迭代落位
- **后续可对特定形状（如 d_in=4096, d_out=11008）做 tile 扫参**

### 5.2 K 维 loop 结构（按 group 累加）

```
for g in range(n_groups):                     # K 维 loop
    W_tile = load_W_low_s4(...)               # (BM, BK) s4  (W_low - 8 如果用 s4×s4)
    X_tile = load_X_s4(...)                   # (BK, BN) s4
    acc_g  = tl.dot(W_tile, X_tile, out_dtype=tl.int32)

    # 按 group 立即 dequant + 累加到 FP16 主累加器
    # acc_g - zero_u4[i, g] × sum_X[j, g]     ← zero 项修正
    # × scale_u4[i, g] × scale_x[j]
    # （若用 s4×s4，需加 + 8 × sum_X[j, g] 修正 UINT4→SINT4 偏移）
    Y_acc += dequant_per_group(acc_g, scale_u4_g, zero_u4_g, sum_X_g, scale_x)
```

### 5.3 `scale_u4, zero_u4` 的访问

- `scale_u4, zero_u4` 形状 `(d_out, n_groups)`，对 tile `(BM, BK=bcol)` 每 K 迭代只需加载 `(BM,)` 个 scale + `(BM,)` 个 zero
- **推荐 layout**：`(d_out, n_groups)` row-major，K 迭代间按输出行常驻寄存器
- **预加载策略**：`scale_x[j:j+BN]` 作为 `(BN,)` 向量在 kernel 入口一次加载入寄存器/SMEM，所有 K 迭代共用

### 5.4 `sum_X` 的访问

- 形状 `(batch*seq, n_groups)`
- 每个 Kernel ① tile 按 `(BN,) × 1` 访问 —— `sum_X[j:j+BN, g]` 每 K 迭代加载一次

---

## 6. 开发顺序约定

按"依赖顺序 + 能独立验证"的原则：

| # | Kernel | 依赖 | 验证方式 |
|---|--------|------|---------|
| 1 | **激活量化 kernel** | 无 | 与 PyTorch `X.abs().max() / 7` + `round/clamp` 参考实现对拍 `X_s4, scale_x, sum_X` |
| 2 | **打包工具（Python）** | 无 | 对 FakeQuant 参考重构 `W_fp16_dequant` 应逐元素等于 GPTQ FakeQuant 输出；`scale_u4` 覆盖后满足 `kernel_algorithm.md` §7.3 所有断言 |
| 3 | **Kernel ①（稠密 UINT4×SINT4）** | #1, #2 | 在只有 UINT4 块（无 SINT8 块）的极端情况下，Kernel ① 单独输出应等于 FakeQuant 参考 |
| 4 | **Kernel ②（稀疏 SINT4×SINT4）** | #1, #2, #3 | 在只有 SINT8 块（100% 高精度）的极端情况下，Kernel ① + Kernel ② 合成应等于 FakeQuant 参考 |
| 5 | **端到端联调** | #1-#4 | 真实 5% 高精度块比例下，Linear 输出与 FakeQuant 参考**逐元素等价**（差异仅来自 FP 累加顺序，应在 1e-3 量级内） |

---

## 7. 代码规范与约定

### 7.1 路径规范

- **绝对路径解析**：所有脚本用 `pathlib.Path(__file__).resolve().parent` 基准解析数据路径，禁止依赖 CWD
- **Kernel 代码目录**：`kernel/triton/`（待创建），结构约定：
  ```
  kernel/
    research/           ← 本文档和算法分析
    triton/             ← Triton kernel 源码
      activation_quant.py    (#1)
      dense_u4s4_gemm.py     (#3 Kernel ①)
      sparse_s4s4_gemm.py    (#4 Kernel ②)
      pack_utils.py          (#2 打包工具)
      tests/
        test_activation.py
        test_dense.py
        test_sparse.py
        test_end2end.py
  ```

### 7.2 输出语言

- 代码中所有 `print` / `log` / `raise` / 图表 label / title / 注释 **强制使用英文**，避免终端编码问题
- 项目级文档（README / MD 文件如本文）可使用中文

### 7.3 数据类型 / 命名

- `tl.int8` 用作 4-bit 容器（2 个 4-bit pack 在 1 个 INT8 字节里）
- `tl.float16` 用作 scale
- `tl.int32` 用作 MMA 累加器
- 变量命名沿用 `kernel_algorithm.md`：`W_low, W_high_blocks, scale_u4, zero_u4, sum_X, scale_x` 等

---

## 8. 快速索引（后续 kernel 开发时的"提示"）

当开始某个 Triton 算子开发时，把本文档作为前置上下文喂给 AI，并补充以下触发点：

- **"要写激活量化 kernel"** → §4, §7.1
- **"要写 Kernel ① 稠密 UINT4×SINT4 GEMM"** → §1, §2, §5, §7
- **"要写 Kernel ② 稀疏 SINT4×SINT4 GEMM"** → §1, §2, §3, §7
- **"要写打包工具"** → `kernel_algorithm.md` §4.4 + 本文档 §3.3

---

## 9. 决策记录摘要

| 决策项 | 选择 | 理由 |
|--------|------|------|
| INT4 Tensor Core | ✅ 使用（SM89 原生支持） | Ada 白皮书确认 |
| MMA 指令类型 | 统一 `s4 × s4`（`mma.m16n8k64`） | 两个 kernel 硬件路径一致；UINT4 用 `−8` 偏移 + `sum_X` 修正 |
| Kernel ② 调度 | 按输出行 tile 分桶（BSR indptr） | 无 atomic、无 scatter、与 Kernel ① grid 同形 |
| 激活量化 | 单 kernel 融合产出 `X_s4/scale_x/sum_X` | 最小化 HBM 往返 |
| BK 大小 | 默认 `BK = bcol = 128` | 与 group 边界严格对齐 |
| 稀疏格式 | 扁平 BSR（`hp_row_offsets + hp_col_indices + W_high_blocks`） | `kernel_algorithm.md` §4.3 的增强 |
| 优化目标 | 实测性能 | 以 cuBLAS FP16 GEMM 为基线，目标 ≥ 3× 加速 |
| kernel 数量 | 仅 3 个（激活量化 + ① + ②） | 不引入辅助 reduce kernel |
