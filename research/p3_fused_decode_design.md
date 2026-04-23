# P3 — Fused Decode Kernel (Design)

**目标**：把 decode 路径（`T ≤ 16`, 覆盖 64 个 shape）从 "3~4 个独立 kernel launch + Python 调度" 压成 **1 个 Triton kernel**，让 quant/dense/sparse/combine 共享同一次 HBM 遍历，使 T=1 场景进一步从 graphed 后的 ~0.5~1.0× FP16 推到 **1.2~1.5× FP16**。

---

## 1. 当前 decode 路径画像（commit `425bbc1`, V9LinearCudaGraph replay 后）

| shape (T,d_out,d_in,hp) | graphed plain | ideal min (FP16) | gap | 主占 |
|---|---|---|---|---|
| 1, 4096, 4096, 0.00 | 83 µs | 16 µs | **5.2×** | 小方阵，全是 launch 余震 |
| 1, 4096, 4096, 0.05 | 103 µs | 16 µs | **6.4×** | sparse 还能看见 |
| 1, 14336, 4096, 0.00 | 98 µs | 98 µs | 1.00× ✅ | 已贴 HBM roof (1 row weight scan) |
| 1, 14336, 4096, 0.05 | 122 µs | 98 µs | 1.24× | sparse blocks 占 ~25 µs |
| 1, 28672, 4096, 0.00 | 118 µs | 115 µs | 1.03× ✅ | 已贴 HBM roof |
| 1, 28672, 4096, 0.05 | 141 µs | 115 µs | 1.23× | sparse 占 ~25 µs |
| 16, 14336, 4096, 0.05 | 132 µs | 96 µs | 1.37× | sparse 26 µs + 小 tile 不满 |

**两个问题**：

1. **小方阵 `4096×4096`** 即使 graphed 后仍 0.16× FP16（100 µs vs 16 µs）
   - 原因：我们有 4 次 kernel body（量化 20 µs + dense 35 µs + sparse 25 µs + combine 10 µs + graph 开销 ~5 µs），每个 body 的"真正工作"小到只用 1~3 SM，但不能共享 SMEM/寄存器 → 4 次 HBM 往返
   - Graph 只消除了 launch overhead，没消除 HBM 往返
2. **hp>0 时 sparse 固定加 ~25 µs**（和 d_out 几乎无关）
   - 原因：sparse kernel 独立启动，即便只有 ~5% blocks，它也要做一轮 grid scan + BSR 访问 + 回写 (d_out, T) 的 Y_high

## 2. 融合机会点

### 2.1 HBM 访问统计（T=1, d_out=4096, d_in=4096, hp=0.05）

| stage | 输入读 | 输出写 | 小计 |
|---|---|---|---|
| quant | X_fp16 (32 KB) | X_s4 (8 KB) + scale_x (2 B) + sum_X (128 B) | **40 KB r/w** |
| dense | W_low_packed (8 MB) + X_s4 (8 KB) + scale/zero/sum | Y_low (8 KB) | **8 MB r** |
| sparse | W_high_blocks (~400 KB, hp=5%) + X_s4 (8 KB 重读!) + bsr idx | Y_high (8 KB) | **~410 KB r + 8 KB w** |
| combine | Y_low (8 KB 重读!) + Y_high (8 KB 重读!) | Y_out (8 KB) | **24 KB r/w** |
| **合计** | — | — | ~8.4 MB |
| **fused 下限** | X_fp16 + W_low + W_high_blocks + bsr | Y_out | ~8.4 MB **不变（HBM 限）** |

**结论**：dense 阶段的 8 MB 权重读已经是 HBM 下限，单靠融合不能把总字节数再降。**但融合能消除**：
- 所有中间张量（X_s4/scale_x/sum_X/Y_low/Y_high）的**写+读一次**往返 ≈ 50 KB/call
- **3 次 kernel launch**（graph 后每次 ~2-3 µs）
- **每次 autotune dispatcher** 的 Python/host 开销

粗算 `4096×4096 bs=1 hp=0.05`：HBM 下限 = 8.4 MB / 1 TB/s = 8.4 µs；加上 ~8 µs launch 和 SM 填充，理论能冲到 **~20 µs**，对应 **~0.8× FP16** 的小方阵新基线。

### 2.2 大 d_out 是 HBM-bound，融合意义不同
对 `d_out ≥ 14336`（hp=0）已贴 HBM roof（plain=1.00~1.03× FP16），**融合的价值在 hp>0**：把 sparse 并进同一 kernel 的主循环 → 省掉那固定的 25 µs。

## 3. Kernel 设计

### 3.1 并行切分

```
grid = (cdiv(d_out, BM),)      # 单维 grid，每 program 处理一整条 (BM, T) 片
T <= 16 始终一次性读完              # 不切 N 轴
```

**理由**：
- T=1~16 时 BN 最小可取 16 → 单个 program 就能覆盖整个 N 轴，**不需要跨 program reduction**
- d_out 方向 grid 宽 = ceil(d_out/BM)，典型 BM=64~128 → 对 d_out=4096 = 32~64 programs，对 RTX 4090 (128 SM) 刚好填满 1 wave

### 3.2 单 program 内的伪代码

```python
@triton.jit
def fused_decode_kernel(
    # 输入
    X_fp16_ptr,     # (T, d_in)          原始输入
    perm_ptr,       # (d_in,) int32      act-order
    W_low_ptr,      # (d_out, d_in/2)    packed int8 U4
    scale_u4_ptr,   # (d_out, n_groups)  fp16
    zero_u4_ptr,    # (d_out, n_groups)  fp16 (已减 8)
    W_high_ptr,     # (nblk, BR, BC/2)   packed int8 S4  [optional]
    hp_row_offsets, # (nrow_blk+1,) i32
    hp_col_indices, # (nblk,) i32
    # 输出
    Y_out_ptr,      # (T, d_out) fp16
    # 标量
    T, d_out, d_in, N_GROUPS,
    # 常量
    BM: tl.constexpr,      # d_out 方向 tile，典型 64/128
    BN: tl.constexpr,      # T 方向 tile，固定 = 16（decode 上限）
    BCOL: tl.constexpr,    # group 宽度 = 128
    HAS_SPARSE: tl.constexpr,
    BR: tl.constexpr,      # sparse block row = 128
    BC: tl.constexpr,      # sparse block col = 128
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BM + tl.arange(0, BM)            # d_out 方向
    offs_n = tl.arange(0, BN)                         # T 方向
    mask_n = offs_n < T

    # ------------------------------------------------------------------
    # 阶段 A: 把 X 整行量化到 SMEM（T × d_in fp16 → T × d_in/2 s4 in SMEM）
    # ------------------------------------------------------------------
    # 对 decode，T <= 16 且 d_in <= 14336，T * d_in * 2B <= 460 KB，
    # 无法整体放 SMEM（4090 SMEM / SM = 228 KB）。
    # 折中：按 BCOL 分块量化，边量化边消费。
    # 对 T=1~16 的 scale_x 可以先做一次 max-abs pass：
    #   - 从 HBM 读一行 X (d_in 个 fp16)
    #   - 一次扫出 scale_x[n] 存 SMEM（16 个 fp16）
    # 同时累积 sum_X[n, g] 到 SMEM（16 × n_groups int32）
    # 阶段 A 结束后 SMEM 里有 scale_x, sum_X；X_s4 分段 regenerate
    # ------------------------------------------------------------------

    # A1: load X[0:T, 0:d_in] via permuted gather, compute per-token max-abs
    x_max = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, d_in, BCOL):
        col_ids = perm_ptr[k0 : k0+BCOL]          # (BCOL,) int32
        x_tile  = tl.load(X_fp16_ptr + offs_n[:, None]*d_in + col_ids[None, :],
                          mask=mask_n[:, None])   # (BN, BCOL) fp16
        x_max   = tl.maximum(x_max, tl.max(tl.abs(x_tile), axis=1))
    scale_x = (x_max / 7.0).to(tl.float16)        # 保留在寄存器

    # A2: 第二遍 load X, pack to s4, 同时累积 sum_X 并把 packed 块放 SMEM
    # （SMEM 只需 BN × d_in/2 字节，T=16,d_in=14336 → 112 KB, OK on 4090）
    # 可选：如果 SMEM 不够，改成 "分段保存，分段 dense"（内循环嵌套）

    # ------------------------------------------------------------------
    # 阶段 B: Dense GEMM，累加到寄存器 acc[BM, BN] fp32
    # ------------------------------------------------------------------
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(N_GROUPS):
        # 取本 group 的 X_s4（刚刚存 SMEM）和 W_low 切片 (BM, BCOL/2)
        w_tile = tl.load(W_low_ptr_for_group)     # (BM, BCOL/2) int8
        w_unpk = unpack_s4(w_tile)                 # (BM, BCOL) int8
        x_unpk = unpack_s4_from_smem(g)           # (BN, BCOL) int8
        acc   += tl.dot(w_unpk.to(f32), tl.trans(x_unpk).to(f32))
        # 合入 per-group 反量化
        scale_g = tl.load(scale_u4_ptr[offs_m, g])    # (BM,)
        zero_g  = tl.load(zero_u4_ptr[offs_m, g])     # (BM,)
        sum_g_x = tl.load(sum_X_smem[:, g])           # (BN,)
        # acc 此处还是 "raw MMA 积"，下面按公式修正：
        #   y_contrib = scale_u4 * scale_x * (acc_raw - zero_u4 * sum_x_s4)
        ...

    # ------------------------------------------------------------------
    # 阶段 C (可选, HAS_SPARSE)：追加高精度 block 贡献
    # ------------------------------------------------------------------
    if HAS_SPARSE:
        # 本 program 的 d_out 行号对应的 block row = pid_m*BM // BR
        # 遍历 hp_row_offsets[br] .. hp_row_offsets[br+1]
        br = pid_m * BM // BR
        for blk in range(hp_row_offsets[br], hp_row_offsets[br+1]):
            bc = hp_col_indices[blk]
            # 读取 W_high_blocks[blk] (BR, BC/2) int8 SINT4
            # 读取对应的 X_s4[:, bc*BC:(bc+1)*BC] 已在 SMEM
            # acc_high = 16 * scale_u4[:, bc] * scale_x * dot(w_high, x_s4)
            # 直接加到 acc
            ...

    # ------------------------------------------------------------------
    # 阶段 D: 转置写回 Y_out (T, d_out)
    # ------------------------------------------------------------------
    y_out = acc.to(tl.float16)                       # (BM, BN)
    y_out_t = tl.trans(y_out)                        # (BN, BM)
    ptrs = Y_out_ptr + offs_n[:, None] * d_out + offs_m[None, :]
    tl.store(ptrs, y_out_t, mask=mask_n[:, None])
```

### 3.3 SMEM 预算核算（RTX 4090, 228 KB/SM）

| 用途 | 计算 | 尺寸 @ T=16 d_in=14336 | 尺寸 @ T=16 d_in=4096 |
|---|---|---|---|
| X_s4 SMEM (整行) | T × d_in/2 B | 112 KB | 32 KB |
| scale_x | T × 2B | 32 B | 32 B |
| sum_X | T × n_groups × 4B | 16×112×4=7 KB | 16×32×4=2 KB |
| W_low tile (active) | BM × BCOL/2 | 64×64 = 4 KB | 同 |
| acc (寄存器) | BM×BN×4B | 64×16×4 = 4 KB | 同 |
| stages=2 流水 | W × 2 | 8 KB | 同 |
| **总计** | — | **~135 KB** | **~50 KB** |

**结论**：
- `d_in ≤ 11008`：单 program 舒适放下全部中间结果 → 可以走"整行量化 → 整行 dense"
- `d_in = 14336`：紧一点，需要 `BM = 64` 而非 128，或把 X_s4 分段存

### 3.4 对三种 hp 的策略

| hp_ratio | 路径 |
|---|---|
| hp == 0 | HAS_SPARSE=False，阶段 C 编译期删除；主路径 dense 已达 HBM roof，融合收益主要来自消除 X_s4/Y_low 的中间写回 |
| 0 < hp ≤ 5% | HAS_SPARSE=True，阶段 C 遍历 ~0~3 blocks/row；省掉独立 sparse kernel 的 25 µs |
| hp > 5% | 可能退化成 dense-like，直接走 fused 也可以；若回归再加阈值 |

## 4. 实施顺序（严格分阶段，每步都跑 pytest 再往下）

### Step 1 — 基础骨架 + hp=0 路径（1 天）
- 新文件 `triton_kernel/fused_decode_kernel.py`
- 只实现 **阶段 A + 阶段 B**，硬编码 HAS_SPARSE=False
- Python wrapper `fused_decode_forward(X_fp16, W)` 直接返回 (T, d_out) fp16
- 单元测试：bit-exact 对齐 `_v9_forward_decode` 在 hp=0 的输出
- Benchmark：`4096×4096 T=1 hp=0` vs graphed plain，目标 ≥1.3× 加速

### Step 2 — 加 sparse 追加分支（1 天）
- 阶段 C 接入，HAS_SPARSE tl.constexpr 编译期分支
- 测试：7 种 shape × hp∈{0, 0.05, 0.1, 0.2} × T∈{1,4,16}，对齐 `_v9_forward_decode`（数值差 < 1%）
- Benchmark：全 decode grid 28 shape，目标 ≥1.2× 平均加速 vs graphed plain

### Step 3 — 集成到 v9_linear（1 天）
- 在 `_v9_forward_decode` 里加阈值开关：`T ≤ 16 且 d_in 对齐` 走 fused，否则走原路径
- 默认不启用：加 env flag `V9_USE_FUSED_DECODE=1`
- 通过 `V9LinearCudaGraph` 组合测试：fused + graph 双开加速叠加验证

### Step 4 — 清理 + 默认启用（0.5 天）
- 如果全 grid 无回归 → 默认启用
- 把 `V9LinearCudaGraph` 的 warmup 改成同时 capture fused 路径

### Step 5 — 文档沉淀 + 推送（0.5 天）
- `optimization_report.md` 加 §2.11
- `code_architecture.md` Kernel 卡片扩充

**总工期 3-4 天**，每一步都有明确的 go/no-go 判据。

## 5. 验证矩阵

### 5.1 正确性
```python
# 所有已有 decode 测试必须继续通过
pytest kernel/triton_kernel/tests/test_v9_linear.py -k decode
# 新增 test_fused_decode_kernel.py：
#   - bit-exact 对齐整条 _v9_forward_decode 的 hp=0 输出
#   - fp16 误差 < 1e-3 对齐 hp>0 的输出
#   - T 扫过 [1, 2, 4, 8, 16]
#   - d_out × d_in 扫过 (4096,4096), (14336,4096), (28672,4096), (4096,11008)
```

### 5.2 性能
```python
# 复用 bench_decode_launch_overhead.py 的 (shape, hp) 矩阵
# 对比三种模式：
#   eager:    原来的 _v9_forward_decode
#   graphed:  V9LinearCudaGraph 封装
#   fused:    新 fused_decode + 可选 graph
# 接受标准：fused vs eager ≥1.5x；fused vs graphed ≥1.1x；无 case 回归
```

### 5.3 Nsys 结构
```
nsys profile --trace cuda,nvtx python bench_decode_launch_overhead.py --nvtx
# 期望：fused 路径时 GPU 上只能看到 1 个 kernel name，而非 4 个
```

## 6. 回退路径

每一步都保留 fallback：
1. Step 1 失败 → 不合入，保留 `V9LinearCudaGraph` 已有收益
2. Step 2 失败 → 只合入 hp=0 的 fused，hp>0 继续走原路径
3. Step 3/4 回归 → env flag 默认关闭，保留代码作为未来迭代基线

## 7. 已知风险 & 预案

| 风险 | 识别方法 | 预案 |
|---|---|---|
| SMEM 不够（d_in=14336） | 编译时 `ptxas info` / runtime `RuntimeError` | 拆成 "分 K-chunk 流式量化+dense"，用寄存器而非 SMEM 持 X_s4 |
| 量化 max-abs 两遍读 X 反而比独立 quant 慢 | 对比 Step 1 bench | 改成一遍读+SMEM staging |
| sparse 的 tl.load 地址依赖 hp_row_offsets 导致分支发散 | Nsys warp_stall | 在 Python 侧把 block 列表展平、用 int32 数组传入，kernel 用 for 循环 + masking |
| hp>0 时误差累积超过 fakequant 参考 | Step 2 测试失败 | 把累加切成两个 pass（先 dense，再 sparse），acc 保持 fp32 |

---

**下一步**：在服务器上启动 Step 1。首版目标是拿到 `4096×4096 T=1 hp=0` 从 83 µs 降到 **≤ 50 µs**（即 ≥1.6×），若不达标则停下来分析 SMEM/launch 占比再决定继续。

---

## 8. Step 1 设计修订（2026-04-23）

在读 `dense_u4s4_gemm.py` 主循环后，我把 Step 1 收窄为一个更保守的版本：

**不融合 quant**，只做 **"dense + transpose 合一"** kernel，名字 `dense_gemm_u4_s4_to_out`：

- 输入：和 `dense_gemm_u4_s4` 完全一致（`W_low_packed, X_s4, scale_u4, zero_u4, sum_X, scale_x`）
- 输出：直接写 `Y_out (T, d_out) fp16`（不是 `Y_low (d_out, T)`）
- hp=0 的 decode 路径调用者拿到的就是最终结果
- 算子主体完全继承 `dense_gemm_kernel` 的 MMA / 反量化循环；**仅**把 store 段改为"tl.trans(acc).to(fp16) 写到 (T, d_out)"

**这样做的理由**：

1. **bit-exact 对齐**：输出和现有 `dense → _combine_transpose(hp=None)` 路径的结果严格一致；测试用 `torch.equal` 即可
2. **省掉 Y_low 的 HBM 往返**：bs=1, d_out=4096 省 8 KB r/w；bs=1, d_out=28672 省 56 KB r/w
3. **省掉一次 kernel launch + autotune dispatch**（graph 后每次 2~3 µs，no-graph 时 5~8 µs）
4. **Store 方向改成沿 d_out 连续**（stride = 2B）依旧 coalesced，和 `_combine_transpose_kernel` 是同样的 swizzle 思路
5. **SMEM/寄存器占用不变**，autotune 配置可以直接继承 dense_gemm_kernel 的 configs，去掉 GROUP_SIZE_M 相关的一些

**去掉 quant 融合的理由**：
- `quantize_activation_s4` 在 T≤512 已经走 fast-path（固定 config，15 µs/call）——Python dispatcher 开销也被 fast-path 消掉了
- decode 路径又叠上 `V9LinearCudaGraph`，quant kernel launch 也只是 graph node，host 侧摊到 ~0
- 把 quant 压进 dense kernel 会让后者的寄存器 + SMEM 预算紧张，尤其 d_in=14336 边界场景容易 spill

**Step 1 性能目标**：
- 微基准（`bench_dense_to_out.py`）：`4096×4096 T=1` 从 **dense ~35 µs + combine ~10 µs = 45 µs** 压到 **~32 µs**（-29%）
- 微基准：`28672×4096 T=1` 从 **dense ~80 µs + combine ~10 µs = 90 µs** 压到 **~82 µs**（-9%, 接近 HBM roof 所以收益小）
- 端到端 decode (hp=0) 路径 graphed：平均再加 5~10% 提速
- 正确性：全 decode grid × T∈[1,2,4,8,16] bit-exact

Step 2/3/4 保持原设计（加 sparse / 集成 / 清理）。
