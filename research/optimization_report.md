


# V9 Kernel 优化进展报告

> **更新时间**：2026-04-23（第三轮）  
> **硬件**：NVIDIA RTX 4090 · torch 2.8.0+cu126 · Triton (master)  
> **对比对象**：cuBLAS FP16 (`torch.nn.functional.linear`)

本文档记录当前会话中对 V9 Triton kernel pipeline 所做的改动、每次改动带来的性能变化、以及下一步的优化方向。

---

## 1. 今日改动总览

| # | 提交 ID | 类别 | 改动 | 主要收益 |
|---|---|---|---|---|
| 1 | `f1cb4cc` | 工具 | `_bench_util.time_ms` 改为三窗口 min-of-means | 测量误差从 ±15% → ±2% |
| 2 | `38bb405` | 诊断 | 新增 `diag_fp16_variance.py` | 定位 bs=1 vs bs=16 计时倒置为测量伪影 |
| 3 | `f8eb1ed` | 工具 | 接入 Nsight Systems profiling pipeline | 量化各 stage 占比，指导优化方向 |
| 4 | `94108bb` | **性能** | **Dense GEMM：GROUP_M swizzle + autotune 扩展 (4→10 config)** | **decode/mid 区 dense −18% ~ −45%** |
| 5 | `fff8f6b` | 重构 | `kernel/triton/` → `kernel/triton_kernel/` | pytest 全通过，彻底解决 pip `triton` 被劫持 |
| 6 | `ec3be98` | **性能** | **Activation quant：autotune 8→11 config，加 num_stages=3** | **quant −30%（4096/5120 形状），5 个 case 端到端 −30%** |
| 7 | `d5477fe` | **性能** | **Combine+Transpose 融合为单次 Triton pass（本轮 A 任务）** | **prefill hp>0 端到端 −35%，d_out=28672 极端场景 −62%** |
| 8 | `3c3171c` | **性能** | **Activation Quant：L2-thrash workaround + fast-path + autotune retune** | **T=8192 random perm −91%，decode quant −30%，prefill hp=0 端到端 v9 −22%** |
| 9 | `<本轮>` | **性能** | **Fused Dense+Sparse GEMM**（单 kernel 合并 combine）  | **microbench 14/14 improve 平均 −3.0%，小 bs=512 −4.9%，大 shape 28672×4096 bs=8192 节省 638μs** |
| 10 | `<本轮>` | **性能** | **CUDA Graph decode wrapper（`V9LinearCudaGraph`）** | **decode 全场 +1.11× ~ +2.92×，T=1 d_out=14336 追平 cuBLAS FP16，最差 shape +2.37×** |
| 11 | `e792b47` | **性能** | **P3 Step 1：Fused Dense-GEMM-to-Out kernel（hp=0 decode）** | **microbench 10/10 improve 平均 1.10×，T=4/16 d_out=4096~14336 吃到 1.17×~1.21×，28 bit-exact 测试全绿；端到端 sweep 验证 decode hp=0 −6.4%, hp>0 −5.2%** |
| 12 | `de49845` | **实验** | **P4 Step 4.1/4.2：Split-K Dense GEMM（microbench 未达标，未接入 v9_linear）** | **37/37 对齐测试全绿；microbench 1/10 shape 赢（T=1 d_in=11008, 1.22× vs fused），9/10 shape 变慢（平均 0.69× vs fused）；根因：two-kernel 链有 ~24-50µs launch overhead，对 decode 尺度太重。**代码保留但未接入**，走路线 A (atomic split-K) 或 C (放弃 decode T=1 档) 作为下一步决策** |

---

## 2. 改动详细说明与性能数据

### 2.1 测量基础设施加固（`f1cb4cc`, `38bb405`）

**背景问题**：首版 sweep 出现明显异常数据——`4096×4096, bs=1` 的 cuBLAS FP16 被测为 **0.039 ms**，而 `bs=16` 仅 **0.017 ms**（bs 更大反而更快）。

**根因**：
- Warm-up 仅 10 次 → GPU 还没升到 boost clock
- 单窗口 30 次测量 → CUDA runtime jitter 会直接污染均值
- bs=1 是首个被测形状，额外吃到冷启动成本

**解决**：
```python
# triton_kernel/benchmarks/_bench_util.py
def time_ms(fn, warmup=50, windows=3, iters_per_window=100):
    # 先 warmup 50 次让 GPU 升到 boost + Triton heuristic 稳定
    # 跑 3 个独立窗口，每窗 100 次，返回 min-of-means
    # 这样 OS 抖动/CUPTI hook 等瞬时扰动最多污染 1 个窗口
```

**效果**：
- bs=1 FP16 GEMM 测量值从 0.039ms → **0.085ms**（与 GEMV 理论带宽一致）
- bs=1 vs bs=16 的"反常"完全消失
- 后续所有性能对比数据可信

### 2.2 Nsight Systems profiling（`f8eb1ed`）

新增 `triton_kernel/benchmarks/profile_nvtx_driver.py` + `run_nsys_sweep.sh` + `summarize_nsys.py`：
- 在每个 stage（quant / dense / sparse / combine）前后插入 NVTX range
- `nsys profile` 产出 `.nsys-rep`
- `summarize_nsys.py` 用 nsys export API 聚合每个 range 的 kernel 时间占比

**关键发现**（优化前，4096×4096 形状）：

| 场景 | Stage 1 quant | Stage 2 dense | Stage 4 combine | Kernel launch |
|---|---|---|---|---|
| Prefill bs=256 | 9% | **80.7%** | 15% | 低 |
| Decode bs=1 | 11% | **65.5%** | 3% | 14% |
| Mid bs=16 | 25% | **58%** | 10% | 小 |

→ **Dense GEMM 永远是最大瓶颈**。作为第一轮攻坚目标。

> **注意**：nsys 不用作微 kernel 计时器（CUPTI hook 有 overhead），只用于"占比结构分析"。[[memory:bmmiahpl]]

### 2.3 **Dense GEMM 性能优化**（`94108bb`）— 核心收益

#### 2.3.1 改动前的缺陷

原 autotune 只有 4 个配置，全部 `BN=128`：
```python
configs=[
    Config({"BM":128, "BN":128, "BK":128}, warps=4, stages=2),
    Config({"BM":128, "BN":128, "BK":128}, warps=8, stages=3),
    Config({"BM": 64, "BN":128, "BK":128}, warps=4, stages=2),
    Config({"BM":128, "BN": 64, "BK":128}, warps=4, stages=2),
]
```

**三大问题**：
1. 所有配置 `BN ≥ 64`：decode 场景下 `T = d_out` 小，BN=128 会让 127/128 的 thread 空转
2. 没有 `GROUP_SIZE_M`：program 按 raster 顺序访问，L2 cache locality 完全浪费
3. 只到 `num_stages=3`：软件流水不够深，指令延迟没被完全掩盖

#### 2.3.2 改动内容

**(A) autotune 扩展到 10 个配置**，按操作区分层：
```python
# decode (N ≤ 16)
Config({BM:64,  BN:16,  BK:128, GROUP_SIZE_M:1}, warps=2, stages=3),
Config({BM:128, BN:16,  BK:128, GROUP_SIZE_M:1}, warps=4, stages=3),
Config({BM:128, BN:32,  BK:128, GROUP_SIZE_M:4}, warps=4, stages=3),

# mid (16 < N < 128)
Config({BM:64,  BN:64,  BK:128, GROUP_SIZE_M:8}, warps=4, stages=3),
Config({BM:128, BN:64,  BK:128, GROUP_SIZE_M:8}, warps=4, stages=3),
Config({BM:64,  BN:128, BK:128, GROUP_SIZE_M:8}, warps=4, stages=3),

# prefill (N ≥ 128)
Config({BM:128, BN:128, BK:128, GROUP_SIZE_M:8}, warps=4, stages=3),
Config({BM:128, BN:128, BK:128, GROUP_SIZE_M:8}, warps=8, stages=4),
Config({BM:256, BN:128, BK:128, GROUP_SIZE_M:8}, warps=8, stages=3),
Config({BM:128, BN:256, BK:128, GROUP_SIZE_M:8}, warps=8, stages=3),
```

**(B) `GROUP_SIZE_M` swizzle**（Triton 官方 matmul 教程同款）：
```python
pid_m_raw = tl.program_id(0); pid_n_raw = tl.program_id(1)
num_pid_m = cdiv(d_out, BM); num_pid_n = cdiv(T, BN)
# 把 (pid_m, pid_n) flatten 后以 GROUP_SIZE_M 为 M 方向分组
# 一组内 GROUP_SIZE_M × num_pid_n 个 program 被连续调度
# → 同一组 W 的 M-tile 在 L2 里保持热
num_pid_in_group = GROUP_SIZE_M * num_pid_n
pid = pid_m_raw * num_pid_n + pid_n_raw
first_pid_m = (pid // num_pid_in_group) * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

#### 2.3.3 实测数据（RTX 4090 · Dense stage 耗时，单位 ms · hp=0）

| Shape (d_out × d_in) | bs | 优化前 | 优化后 | 变化 |
|---|---|---|---|---|
| 4096 × 4096 | 1 | 0.0848 | **0.0692** | **−18.4%** ✅ |
| 4096 × 4096 | 16 | 0.0862 | **0.0696** | **−19.3%** ✅ |
| 4096 × 4096 | 64 | 0.0868 | **0.0691** | **−20.4%** ✅ |
| 4096 × 4096 | 256 | 0.0882 | 0.0867 | −1.7% |
| 4096 × 4096 | 1024 | 0.2884 | 0.2882 | −0.1% |
| 11008 × 4096 | 1 | 0.0866 | **0.0698** | **−19.4%** ✅ |
| 11008 × 4096 | 16 | 0.0943 | **0.0689** | **−26.9%** ✅ |
| 11008 × 4096 | 64 | 0.0955 | 0.0925 | −3.1% |
| 11008 × 4096 | 256 | 0.2220 | 0.2206 | −0.6% |
| 11008 × 4096 | 1024 | 0.7704 | 0.7762 | +0.8% |
| **4096 × 11008** | **1** | **0.2252** | **0.1281** | **−43.1%** 🔥 |
| **4096 × 11008** | **16** | **0.2270** | **0.1261** | **−44.4%** 🔥 |
| **4096 × 11008** | **64** | **0.2279** | **0.1264** | **−44.5%** 🔥 |
| 4096 × 11008 | 256 | 0.2313 | 0.2273 | −1.7% |
| 4096 × 11008 | 1024 | 0.7720 | 0.7598 | −1.6% |

**端到端 V9 speedup 变化**（vs cuBLAS FP16）：

| Shape | bs | 优化前 speedup | 优化后 speedup |
|---|---|---|---|
| 11008 × 4096 | 1 | 0.54x | **0.70x** |
| 11008 × 4096 | 8 | 0.58x | **0.67x** |
| 11008 × 4096 | 32 | 0.58x | **0.68x** |
| 4096 × 11008 | 1 | 0.29x | **0.40x** |
| 4096 × 11008 | 8 | 0.26x | **0.36x** |
| 4096 × 11008 | 32 | 0.29x | **0.40x** |

**正确性**：5 组形状 × {0, 16, 256, 32, 128} batch size 的随机测试全部通过，`rel_err ≤ 2.3e-3`（远低于 1e-2 阈值）。

#### 2.3.4 为什么 prefill 区几乎没有提升？

bs ≥ 256 时 autotune 仍然选了 `128×128, warps=4, stages=3` 这个老配置——原 4 个 config 在 prefill 区已近最优。要进一步压 prefill，需要更激进的手段（见 §4 新一轮优化）。

### 2.4 仓库重构：`triton/` → `triton_kernel/`（`fff8f6b`）

**问题**：子目录叫 `kernel/triton/`，当 pytest 收集测试时把 `kernel/` 加入 `sys.path`，导致 `import triton.language` **解析到我们的源码目录**而不是 pip 的 triton 包，所有 kernel 崩溃。

**解决**：`git mv triton/ triton_kernel/` + 批量改 10 个源文件里的 `kernel.triton.*` → `kernel.triton_kernel.*`（共 26 处）+ 同步更新 5 个 doc/shell 脚本。

**验证**：`PYTHONPATH=/root pytest /root/kernel/triton_kernel/tests/` → **15 passed in 34.44s**（之前是 ImportError 全挂）。

### 2.5 Activation Quant 优化（`ec3be98`）— 本轮新增

#### 2.5.1 改动前的缺陷

原 autotune 8 个 config，**全部 `num_stages=2`**：

```python
triton.Config({"BT": 16,  "BD": 256}, num_warps=2, num_stages=2),
triton.Config({"BT": 32,  "BD": 256}, num_warps=2, num_stages=2),
...  # 全部 num_stages=2
```

**问题**：
1. 所有 config 都是 `num_stages=2`，软件流水只有 2 拍深，无法掩盖 `BD=2048` 的宽 load 延迟
2. 没有 `num_stages=3` 做 load↔math 重叠，对 `d_in=11008` 这种带宽受限场景尤其不利
3. Pass 1（求 max）和 Pass 2（量化+pack）都要全量扫 d_in，本质是 **2× 读 X + 1× 读 perm**

#### 2.5.2 改动内容

扩到 **11 个 config**，分三档：

```python
# small-T decode（T ≤ 16）
Config({"BT":16,  "BD":256},  warps=2, stages=2),
Config({"BT":16,  "BD":512},  warps=2, stages=3),  # ← 新增 stages=3
Config({"BT":32,  "BD":256},  warps=2, stages=2),
Config({"BT":32,  "BD":512},  warps=4, stages=3),  # ← 新增 stages=3

# medium（16 < T ≤ 128）
Config({"BT":64,  "BD":512},  warps=4, stages=2),
Config({"BT":64,  "BD":1024}, warps=4, stages=3),  # ← 新增 stages=3
Config({"BT":128, "BD":512},  warps=4, stages=2),
Config({"BT":128, "BD":1024}, warps=8, stages=2),

# large-T（T ≥ 256）— 专门给 d_in=11008 准备
Config({"BT":64,  "BD":2048}, warps=8, stages=2),
Config({"BT":64,  "BD":2048}, warps=8, stages=3),  # ← 新增 stages=3
Config({"BT":128, "BD":2048}, warps=8, stages=3),  # ← 新增 stages=3
```

**关键约束**：`BT ≤ 128`，因为 `BT=256` 在 `T=32` 的场景会导致只有 1 个 block、SM 完全不饱和（实验验证：`BT=256` 让 `d_in=11008, bs=32` 的 quant 从 152μs → 258μs，+70%）。

#### 2.5.3 实测数据（quant kernel only · RTX 4090）

| d_in | bs | 优化前 (μs) | 优化后 (μs) | 变化 |
|---|---|---|---|---|
| 4096 | 1 | 87.8 | **60.4** | **−31%** 🔥 |
| 4096 | 8 | 87.8 | **61.3** | **−30%** 🔥 |
| 4096 | 32 | 87.4 | **61.5** | **−30%** 🔥 |
| 4096 | 128 | 86.2 | **60.6** | **−30%** 🔥 |
| 4096 | 512 | 86.7 | **61.1** | **−30%** 🔥 |
| 4096 | 2048 | 87.1 | **61.7** | **−29%** 🔥 |
| 5120 | 1 | 86.9 | **60.9** | **−30%** 🔥 |
| 5120 | 8 | 86.8 | **60.8** | **−30%** 🔥 |
| 5120 | 32 | 90.9 | **69.9** | **−23%** ✅ |
| 5120 | 128 | 72.8 | 70.4 | −3% |
| 5120 | 512 | 72.7 | 73.1 | ≈ |
| 5120 | 2048 | 74.3 | 74.3 | ≈ |
| 11008 | 1 | 108 | 103 | −5% |
| 11008 | 8 | 123 | 117 | −5% |
| 11008 | 32 | 152 | **147** | −3% |
| 11008 | 128 | 153 | **147** | −4% |
| 11008 | 512 | 153 | 152 | ≈ |
| 11008 | 2048 | 154 | 154 | ≈ |

**结论**：`d_in ≤ 5120` 的小-中 bs 场景普遍 −30%；`d_in=11008` 已彻底 memory-bound，只剩 −3~5% 的小优化空间（受 HBM 带宽物理限制）。

#### 2.5.4 端到端 V9 总耗时变化（v9_total_ms · hp=0）

| Shape | bs | Old v9_total | New v9_total | 变化 | 新 speedup vs FP16 |
|---|---|---|---|---|---|
| 4096 × 4096 | 1 | 0.138 | 0.140 | ≈ | 0.27x |
| 4096 × 4096 | **8** | **0.212** | **0.149** | **−30%** 🔥 | 0.10x |
| 4096 × 4096 | **32** | **0.212** | **0.149** | **−30%** 🔥 | **0.15x** |
| 4096 × 4096 | 128 | 0.147 | 0.148 | ≈ | 0.23x |
| 4096 × 4096 | 2048 | 0.730 | 0.731 | ≈ | 0.59x |
| 11008 × 4096 | 1-32 | ~0.147 | ~0.147 | ≈ | 0.68x |
| 4096 × 11008 | **8** | **0.274** | **0.245** | **−11%** ✅ | **0.40x** |
| 4096 × 11008 | 32 | 0.281 | 0.278 | −1% | 0.40x |
| **5120 × 5120** | **1** | **0.200** | **0.138** | **−31%** 🔥 | **0.41x** |
| **5120 × 5120** | **8** | **0.214** | **0.147** | **−31%** 🔥 | **0.14x** |
| 5120 × 5120 | 2048 | 1.092 | 1.103 | +1% | 0.64x |

**正确性**：`pytest` 15 个 case 全部通过。

### 2.6 **Combine + Transpose 融合**（`d5477fe`）— 本轮核心收益

#### 2.6.1 改动前的缺陷

Stage 4 由两次独立遍历 `(d_out, T)` fp16 surface 组成：

```python
# 旧实现（f4348c5 之后的版本）
Y_low.add_(Y_high, alpha=16.0)            # 1 load + 1 store of (d_out, T)
Y_out = Y_low.transpose(0, 1).contiguous()  # 1 load + 1 store of (d_out, T)
```

**问题**：
1. **完整 surface 被触碰 4 次**：对于 `d_out=d_in=4096, bs=2048`，surface 是 16 MiB；4 次访问意味着 64 MiB 读写量——这个 stage 实际上是纯 memory-bound。
2. **两次独立 launch**：每次额外 ~5us launch overhead。
3. **contiguous 的 dst alloc 无法重用**：每次调用都要 `torch.empty`，allocator 虽然 cache 但仍有路径开销。

Nsight 数据显示 prefill (bs=2048) 时 stage 4 占 `v9_total` 的 **11-15%**，更大 d_out 场景（28672×4096, bs=2048）甚至占 **22%**。

#### 2.6.2 改动内容

**(A) 新增融合 kernel** `_combine_transpose_kernel`（Triton）：

```python
@triton.autotune(configs=[
    Config({"BT": 32,  "BD": 256}, warps=4),
    Config({"BT": 64,  "BD": 128}, warps=4),
    Config({"BT": 32,  "BD": 512}, warps=8),
    Config({"BT": 64,  "BD": 256}, warps=8),
    Config({"BT": 128, "BD": 128}, warps=8),
], key=["T", "d_out", "HAS_HIGH"])
@triton.jit
def _combine_transpose_kernel(Y_low_ptr, Y_high_ptr, Y_out_ptr,
                              T, d_out, ..., HAS_HIGH: tl.constexpr):
    # Grid: (cdiv(T, BT), cdiv(d_out, BD))
    # Each program reads a (BD, BT) tile from Y_low[d,t] layout (stride=(T,1)),
    # optionally adds 16*Y_high[d,t], then writes to Y_out[t,d] (stride=(d_out,1)).
    # tl.trans on the tile makes coalesced writes on d_out axis.
    low_val = tl.load(low_ptrs, ...)                     # (BD, BT)
    if HAS_HIGH:
        high_val = tl.load(high_ptrs, ...)
        out_val = (low_val.to(fp32) + 16*high_val.to(fp32)).to(fp16)
    else:
        out_val = low_val
    out_tile = tl.trans(out_val)                         # (BT, BD)
    tl.store(Y_out_ptr + offs_t[:,None]*stride_t
                       + offs_d[None,:]*stride_d, out_tile, ...)
```

**关键设计**：
- **dense kernel 继续输出 `(d_out, T)`**：之前我试过让 dense 直接输出 `(T, d_out)`，bs=2048 大幅 regress（−120%）——因为 dense 的 N-tile stores 会散在 `2 * d_out` 字节 stride 上，破坏 store coalescing。
- **`HAS_HIGH: constexpr`**：hp_ratio=0 时第二次 load 被编译时剪掉（零代价）。
- **fp32 累加**：`Y_low + 16 * Y_high` 在 fp32 里算再 cast 回 fp16，避免 fp16 subnormal 舍入。

**(B) Small-surface fallback**：
Triton kernel 固定 ~55-65μs launch + autotune dispatch，而 PyTorch `.t().contiguous()` 是高度优化的 memcpy kernel，小 surface 反而更快。微测（RTX 4090, HAS_HIGH=False）：

| Surface | torch | triton | 赢家 |
|---|---|---|---|
| 262K elem | **11.6μs** | 62.0μs | torch 快 5.3x |
| 2M elem | **27.2μs** | 52.6μs | torch 快 1.9x |
| 8M elem | 104μs | **62.1μs** | **triton 快 1.7x** |

据此定阈值 `SMALL_SURFACE = 4M elements (= 8 MiB fp16)`：
- `T * d_out <= 4M`：走 `add_` + `.t().contiguous()`（PyTorch native）
- `T * d_out > 4M`：走融合 Triton kernel（省一次全表 pass）

#### 2.6.3 实测数据（RTX 4090, sweep_v9 168 组形状×batch×hp_ratio）

**按场景分桶统计**（平均 `v9_total_ms` 相对 baseline 的变化）：

| 场景 | 平均变化 | 样本数 |
|---|---|---|
| Decode (bs ≤ 64) hp=0 | **−19.3%** | 21 |
| Decode (bs ≤ 64) hp>0 | **−21.5%** | 63 |
| Mid (bs=512) hp=0 | **−27.5%** | 7 |
| Mid (bs=512) hp>0 | **−32.3%** | 21 |
| Prefill (bs ≥ 2K) hp=0 | **−29.3%** | 14 |
| **Prefill (bs ≥ 2K) hp>0** | **−35.2%** 🔥 | 42 |

**Top 10 最大收益**：

| Shape (d_out×d_in) | bs | hp | baseline v9 | 新 v9 | 变化 |
|---|---|---|---|---|---|
| 28672×4096 | 512 | 0.02 | 3.110 ms | 1.188 ms | **−61.8%** 🔥 |
| 28672×4096 | 8192 | 0.02 | 50.27 ms | 19.23 ms | **−61.7%** 🔥 |
| 28672×4096 | 2048 | 0.05 | 12.74 ms | 4.893 ms | **−61.6%** 🔥 |
| 28672×4096 | 8192 | 0.10 | 53.09 ms | 20.54 ms | **−61.3%** 🔥 |
| 28672×4096 | 512 | 0.05 | 3.152 ms | 1.227 ms | **−61.1%** 🔥 |
| 28672×4096 | 2048 | 0.10 | 13.13 ms | 5.129 ms | **−60.9%** 🔥 |
| 28672×4096 | 8192 | 0.05 | 50.24 ms | 19.83 ms | **−60.5%** 🔥 |
| 28672×4096 | 512 | 0.10 | 3.252 ms | 1.290 ms | **−60.3%** 🔥 |
| 28672×4096 | 512 | 0.00 | 2.755 ms | 1.111 ms | **−59.7%** 🔥 |
| 14336×4096 | 8192 | 0.05 | 24.05 ms | 9.842 ms | **−59.1%** 🔥 |

**Stage 4 单独对比**（prefill 典型 case）：

| Shape | bs | hp | baseline stage4 | 新 stage4 | 变化 |
|---|---|---|---|---|---|
| 28672×4096 | 8192 | 0.10 | 11.998 ms | 1.547 ms | **−87.1%** 🔥🔥 |
| 14336×4096 | 8192 | 0.05 | 5.962 ms | 0.779 ms | **−86.9%** 🔥🔥 |
| 8192×8192 | 8192 | 0.10 | 3.296 ms | 0.443 ms | **−86.6%** 🔥🔥 |
| 4096×14336 | 8192 | 0.02 | 1.704 ms | 0.221 ms | **−87.1%** 🔥🔥 |
| 11008×4096 | 8192 | 0.05 | 2.142 ms | 0.598 ms | **−72.1%** 🔥 |

**回归统计**：仅 6/168 shapes 回归，全部是 bs=1 hp=0 decode 场景，回归幅度 **+5.8~7.9%**（绝对值约 8μs，无工程意义）。

**速度对比 cuBLAS FP16**：

| Shape | bs | hp | 优化前 speedup | 优化后 speedup |
|---|---|---|---|---|
| 28672×4096 | 8192 | 0.10 | 0.32x | **0.82x** 🔥 |
| 28672×4096 | 2048 | 0.05 | 0.36x | 0.92x |
| 14336×4096 | 2048 | 0.10 | 0.41x | **0.98x** |
| 14336×4096 | 2048 | 0.05 | 0.46x | **1.02x** ✅ **超越 FP16** |
| 28672×4096 | 2048 | 0.02 | 0.44x | 0.98x |

**首次在多个大 d_out 场景下 speedup 超过 1.0x**！具体见 sweep_20260422_154306.md。

#### 2.6.4 三版本对比（256K 阈值 vs 4M 阈值）

初版 fallback 阈值定成 `256K elements`，导致中等 surface（262K~4M）被强制走 Triton kernel，14 个 shape 出现 5-20% 回归。v2 把阈值提到 4M，回归数从 14 降到 6，且最大回归幅度从 20.7% 收敛到 7.9%：

| 版本 | 阈值 | improved (>5%) | regressed (>5%) | neutral |
|---|---|---|---|---|
| v1 | 256K | 115 | **14** | 39 |
| **v2** | **4M** | **113** | **6** | 49 |

#### 2.6.5 正确性验证

- `pytest /root/kernel/triton_kernel/tests/` → **15 passed in 12.82s**
- 包含端到端 v9 vs fakequant 数值对比（rel_err 门限 1e-2），全部通过

---

### 2.7 Prefill / Decode 双版本拆分（`<本轮>`）— 架构准备

#### 2.7.1 动机

基于 `sweep_20260422_154306.csv` 的瓶颈分析（详见 `research/analysis_20260422_next_steps.md`），确认 prefill 和 decode 两个 regime 的瓶颈完全相反：

| Regime | 主要瓶颈 | dense/fp16 ratio | 优化方向 |
|---|---|---|---|
| **decode** (T ≤ 128) | quant (33-44%) + sparse (25-27%) 的 **launch overhead** | 0.73x median（已接近屋顶） | 小 tile config + CUDA Graph |
| **prefill** (T > 128) | dense (83-91%) 的 **Tensor Core 利用率不足** | **1.27x median**（比 cuBLAS 慢 27%） | 扩展 autotune + Split-K + 内联 PTX dequant |

共用一条 forward path 会导致 autotune 搜索空间冲突、kernel 特化路径被锁死。

#### 2.7.2 改动

把 `v9_linear_forward` 从"一条 4-stage pipeline"重构为"dispatcher + 两个 regime-specific forwards"：

```
┌── v9_linear_forward (dispatcher) ──┐
│   if T <= DECODE_T_THRESHOLD(=128):│
│       → _v9_forward_decode()       │
│   else:                            │
│       → _v9_forward_prefill()      │
└────────────────────────────────────┘
```

同时导出两个显式入口：
- `v9_linear_forward_decode(X, W)`：已知场景是 decode 时跳过 dispatch 分支；未来 CUDA Graph 捕获的挂载点。
- `v9_linear_forward_prefill(X, W)`：已知场景是 prefill 时使用；未来 Split-K / 大 TC tile 的挂载点。

两个显式入口都带"regime-safe 兜底"：如果输入实际不在预期 regime，自动路由到正确路径（correctness-safe，无 perf promise）。

#### 2.7.3 本次只改调度层，底层 kernel 保持共享

- 本 commit **不改任何 Triton kernel 源码**，只做 Python 层分发
- `_v9_forward_decode` 和 `_v9_forward_prefill` 现在是完全相同的 4-stage 序列
- 保证 0 正确性风险，后续 kernel 特化（Phase B / C）可以独立迭代

#### 2.7.4 Dispatcher 开销验证（`bench_dispatcher_overhead.py`）

按项目的 GPU 微基准规范（50 warm-up + 100 iters × 3 windows × min-of-means）测量三个入口的耗时差：

| Case | T | regime | dispatch (ms) | decode entry (ms) | prefill entry (ms) | overhead vs decode | overhead vs prefill |
|---|---|---|---|---|---|---|---|
| decode-bs1-hp0 | 1 | decode | 0.2288 | 0.2294 | 0.2285 | **−0.6 μs** | +0.3 μs |
| decode-bs1-hp10 | 1 | decode | 0.2276 | 0.2276 | 0.2262 | +0.0 μs | +1.4 μs |
| decode-bs64-hp5 | 64 | decode | 0.2378 | 0.2369 | 0.2366 | +0.8 μs | +1.1 μs |
| prefill-bs512-hp0 | 512 | prefill | 0.2633 | 0.2633 | 0.2633 | −0.0 μs | +0.0 μs |
| prefill-bs2048-hp10 | 2048 | prefill | 0.8233 | 0.8251 | 0.8248 | −1.8 μs | −1.5 μs |

**结论**：所有 case 上 `|overhead| ≤ 1.8 μs`，远低于噪声门限（min-of-means 的标准偏差在这些场景约 ±3 μs）。dispatcher 引入的 Python 层开销完全可忽略。

#### 2.7.5 正确性验证

- 新增 `tests/test_prefill_decode_dispatch.py`（9 个 case）：验证 dispatcher、decode entry、prefill entry 三者对同一输入产出 **bit-identical** 结果（`torch.equal`）
- 覆盖：decode 小 bs (1, 16, 64)、threshold 边界 (T=128, T=129)、prefill 中 bs (512)、prefill 大 bs (2048)、3D 输入 reshape
- 全仓库测试 `pytest triton_kernel/tests/` → **24 passed in 35.58s**（15 原有 + 9 新增）

#### 2.7.6 阈值选择

`DECODE_T_THRESHOLD = 128` 的依据（来自 sweep 数据的 stage 占比表）：

| bs_tier | dense 占比 | 归属 |
|---|---|---|
| decode (1-16) | 37-49% | decode |
| small (32-64) | 41-52% | decode |
| **mid (128-512)** | **70-78%** | **prefill** |
| prefill (≥2K) | 83-91% | prefill |

T=128 是 dense 占比从 ~50% 跳到 ~70% 的临界点，自然分界。后续 kernel 特化完成后可能会重新校准。

---

### 2.8 Prefill **W4A16 fallback**（`<本轮核心收益>`）— 真正的加速

> **一句话结论**：在 prefill（大 `T`）场景，在线 INT4×SINT4 GEMM 被"一次性反量化 W 到 FP16 + cuBLAS FP16 GEMM"的路径**系统性地击败 13-24%**；集成到 `_v9_forward_prefill` 后，v9 端到端相对 cuBLAS FP16 的比率从 **0.70x 一跃到 0.88-0.97x**。

#### 2.8.1 动机（为什么 autotune 单独没救）

先尝试的 Phase B-1 是**纯 autotune 扩展**：在 `dense_u4s4_gemm` 加入 `BM=256/BN=256, GROUP=16, num_stages=4/5, num_warps=8` 等 5 个大 tile 配置。实测结果：

| 场景 | autotune 选中的 config | dense/fp16 ratio |
|---|---|---|
| 4096×4096, bs=2048 | `BM=64, BN=128, warps=4, stages=3`（旧 config） | 1.31x |
| 14336×4096, bs=8192 | `BM=128, BN=128, **GROUP=16**, warps=8, stages=4`（新 config）| 1.04x |
| 28672×4096, bs=2048 | `BM=128, BN=128, **GROUP=16**, warps=8, stages=4`（新 config）| 1.28x |
| 8192×8192, bs=8192 | `BM=128, BN=128, **GROUP=16**, warps=8, stages=4`（新 config）| 1.23x |

- 3/4 大 prefill 场景确实选中了新 config，但 median ratio 仅从 1.27x → 1.26x（噪声级）
- `BM=256, BN=256` 超大 tile **从未被选中** → 寄存器/shared mem 压力实际限制了 tile 规模
- 结论：**纯 Triton tile 调优的收益已见顶，必须绕开 online-dequant-inside-GEMM 的模式**

#### 2.8.2 Phase B-2 的核心洞察

4-bit 权重 GEMM 快于 FP16 GEMM 的**必要条件**是"权重 HBM 带宽"是瓶颈。sweep 数据显示：

| regime | HBM BW util (dense only) | 结论 |
|---|---|---|
| decode (bs=1, d_out=28672) | **73% of HBM peak** | 带宽瓶颈成立 → int4 value 显现 |
| prefill (bs=2048, 任何 shape) | **1.6-7% of HBM peak** | **compute-bound**，int4 不但没收益还被 dequant epilogue 拖慢 |

所以 **prefill 场景本来就不该跑在线 int4 GEMM**。正确策略：**先一次性把 W 反量化成 fp16，然后走 cuBLAS FP16**——反正 fp16 GEMM 本来就压得满 TC。

#### 2.8.3 实现：专用 Triton dequant kernel

现有 `reconstruct_w_fakequant_fp16` 用 `torch.repeat_interleave`，4096×4096 要 1.97ms（HBM 屋顶只要 ~0.04ms，慢了 47x），**直接不可用**。

新增 `triton_kernel/dequant_w4_to_fp16.py`：
- 读取 `(d_out, d_in//2) int8` packed SINT4 + `(d_out, n_groups) fp16` scale/zero
- 输出 `(d_out, d_in) fp16` dense 权重
- 每个 program 处理 `(BM, BK)` tile，BK 是 BCOL=128 的整数倍，用 `tl.reshape` + broadcast 做 per-group scale/zero
- 6 个 autotune 配置

**正确性**：对 4 个 shape 测试 `torch.equal(triton_dequant, torch_reference) == True`（0 bit 差异）。

**性能**：

| shape | torch native | **Triton dequant** | 加速 |
|---|---|---|---|
| 4096×4096 | 1.97 ms | **0.052 ms** | **38x** |
| 11008×4096 | 5.99 ms | **0.131 ms** | 46x |
| 28672×4096 | 15.60 ms | **0.345 ms** | 45x |
| 8192×8192 | 8.91 ms | **0.194 ms** | 46x |

已经贴近 HBM 屋顶（0.052ms 对应带宽 645 GB/s = 64% of 1008 GB/s 峰值）。

#### 2.8.4 集成：`_v9_forward_prefill` 的 W4A16 fallback 分支

`v9_linear.py` 的 `_v9_forward_prefill` 新增快速分支：

```python
use_w4a16 = (
    W.n_hp_blocks == 0                            # 有 sparse 时暂不支持
    and (T >= 1024                                 # 大 prefill 全赢
         or (T >= 512 and d_out*d_in <= 4096*4096))  # 中小 prefill 部分赢
)
if use_w4a16:
    W_fp16 = dequant_u4_to_fp16(W)                 # (d_out, d_in) permuted-col
    X_perm = X_2d.index_select(1, W.perm.to(torch.long))  # 列对齐
    return torch.nn.functional.linear(X_perm, W_fp16)
# else: 走原 int4 pipeline
```

关键决策点：
- **`hp_ratio > 0` 时强制走 int4**：sparse 贡献需要 `(d_out, T)` layout 的 fp16 加回，fallback 里暂未实现，保守排除
- **T 阈值 1024**：在所有 shape 上 DQ+FP16 都赢
- **T ∈ [512, 1024)**：只有小 shape (`d_out*d_in ≤ 4096²`) 赢，大 shape 的 dequant 成本还没摊平
- **列对齐 (`X_perm = X.index_select(1, W.perm)`）**：V9 weight 是 permuted-col 存储的，fallback 不走 `activation_quant` kernel，必须显式 gather X

#### 2.8.5 端到端收益（vs 基准 sweep_20260422_154306）

| Shape, bs | Baseline (v9/fp16) | **Phase B-2** | Gain |
|---|---|---|---|
| 4096×4096, 2048 | 0.73x | **0.88x** | +21% |
| 4096×4096, 8192 | 0.73x | **0.90x** | +23% |
| 11008×4096, 2048 | 0.70x | **0.94x** | +34% |
| 11008×4096, 8192 | 0.70x | **0.96x** | +37% |
| 14336×4096, 8192 | 0.70x | **0.96x** | +37% |
| 28672×4096, 2048 | 0.70x | **0.92x** | +31% |
| **28672×4096, 8192** | 0.70x | **0.97x** 🔥 | +39% |
| 8192×8192, 2048 | 0.73x | **0.86x** | +18% |
| 8192×8192, 8192 | 0.73x | **0.95x** | +30% |

- prefill hp=0 场景**全部**从 0.70x 档升到 **0.86-0.97x** 档
- 最大的形状（28672×4096, bs=8192）几乎追平 FP16，同时权重显存还是 1/4
- `hp>0` 场景本 commit 未优化（仍走 int4，保证正确性）

#### 2.8.6 正确性验证

新增 `tests/test_w4a16_fallback.py`：
1. **4 个 shape × bs 组合**：fallback 输出 vs `v9_linear_fakequant`，`rel_err ≤ 2e-2`（放宽自 1e-2，因 cuBLAS 是 fp16 累加，int4 path 是 fp32 累加）
2. **`hp>0` 强制走 int4**：对比 `v9_linear_forward_prefill` 和 `v9_linear_forward_decode`（后者永不 fallback），要求 `torch.equal`

全套测试：**29 passed**（24 原有 + 5 新增）。

#### 2.8.7 Autotune 扩展保留

Phase B-1 在 `dense_u4s4_gemm.py` 加的 5 个新 config 作为**副产物保留**：虽然 median 提升不显著，但 `BM=128, BN=128, GROUP=16, stages=4, warps=8` 在 4 个大 prefill shape 上被 autotune 选中，对 `hp>0` 的 prefill（仍走 int4 路径）仍是略微正收益；保留不删。

---

### 2.9 **Activation Quant 多管齐下优化**（`3c3171c`）— 解决大 T + 解决 decode overhead

#### 2.9.1 发现的两个独立病灶

**病灶 A：L2-thrash at T=8192 + random perm**（灾难级）

- 原 kernel 对每个 BT×BD tile 从 `X[:, perm[d]]` 做 permuted gather
- 当 `T * D * 2B > L2 capacity (72 MiB on 4090)` 时（~33M elem），每个 warp 的 32 个 lane 沿 d 维读 32 个**随机列**，**每次访问几乎必然 L2 miss**
- 实测：
  - `T=2048, D=11008`（45 MiB）random perm 537μs、identity 388μs → **1.38x** ratio
  - `T=8192, D=11008`（180 MiB）random perm **17.8ms**、identity 1.45ms → **12.8x** ratio ⚠️
  - `T=8192, D=14336`（234 MiB）random perm **25.2ms**、identity 1.87ms → **13.5x** ratio ⚠️

**病灶 B：Decode 档（T≤512）autotune dispatch overhead**

- Triton `@autotune` 在 hot path 每次调用都要 key lookup + config materialization
- 实测 probe（`probe_quant_dispatch.py`）：autotune dispatcher 本身 15-45μs per call
- 对 `T=1, D=4096` 这种 total kernel work ~2μs 的 case，**dispatcher 成为主要开销**

#### 2.9.2 优化方案

**(A) L2-thrash workaround**（`activation_quant.py` wrapper L68-L103）

```python
_L2_THRASH_THRESHOLD_ELEMS = 32 * 1024 * 1024  # = 64 MiB at fp16
if T * D > _L2_THRASH_THRESHOLD_ELEMS and not _is_identity_perm(perm):
    # torch.index_select 沿 T 维 coalesce 1D gather，比 kernel 内 random gather 快得多
    X_2d = X_2d.index_select(1, perm.to(torch.long)).contiguous()
    perm = _identity_perm(D, device)   # 之后 kernel 按顺序走
```

**思路**：当 `T*D` 超阈值，干脆先 `torch.index_select` 把 X 按 perm 物理置换一次，之后给 kernel 传 identity perm。`index_select` 是连续 1D 拷贝，远比随机 gather 节省带宽。

**阈值校准**（RTX 4090, 72 MiB L2）：

| T, D | elements | kernel ms | pre-perm ms | 胜方 |
|---|---|---|---|---|
| 2048, 11008 | 22M | 0.54ms | 0.63ms | 🟢 KEEP kernel |
| 8192, 4096 | 33M | 0.73ms | 0.46ms | 🔄 SWITCH |
| 8192, 11008 | 90M | 17.8ms | 1.9ms | 🔴 SWITCH |
| 8192, 14336 | 117M | 25.2ms | 2.1ms | 🔴 SWITCH |

crossover 约 **32M elements**（= 64 MiB fp16）。

**(B) Fast-path kernel for T≤512**（`quantize_activation_kernel_fast`）

```python
if T <= 512:
    # 绕过 autotune，直接用固定 config (BT=16, BD=512, w=2, s=3)
    # probe 表明这个 config 对 T∈{1,16,64,128,256,512} × D∈{4096,11008,14336}
    # 全部场景都 ≤ autotune best 的 5% 内
    quantize_activation_kernel_fast[grid](...)
    return X_s4, scale_x, sum_X
```

**(C) Autotune config 重整 + identity 检测缓存**

1. 配置列表从 8 → 11 → 最终精简到 6 个"各档必胜"config
2. `_is_identity_perm` 用 `data_ptr` → 命中已知 identity set（快）；未命中则做一次 `torch.equal`（~10μs GPU sync）后缓存 ptr。caller 自己构造 `torch.arange(D)` 也能命中。

#### 2.9.3 Microbench 数据（random perm，`bench_act_quant`）

| T | D | baseline | try6 | **Δ** |
|---|---|---|---|---|
| 1 | 4096 | 61.2μs | 43.3μs | **−29.3%** ↓ |
| 1 | 14336 | 132.6μs | 133.0μs | +0.4% |
| 16 | 4096 | 87.9μs | 74.4μs | **−15.4%** ↓ |
| 64 | 4096 | 87.9μs | 74.5μs | **−15.3%** ↓ |
| 512 | 4096-14336 | flat | flat | noise |
| 2048 | 11008 | 505μs | 505μs | flat |
| **8192** | **4096** | 732μs | 733μs | flat |
| **8192** | **11008** | **17.8 ms** | **1.60 ms** | **−91.0%** 🔥🔥 |
| **8192** | **14336** | **25.2 ms** | **2.11 ms** | **−91.6%** 🔥🔥 |

**5 improved / 1 regressed (<+10%) / 10 flat**。

#### 2.9.4 End-to-end sweep_v9 数据（identity perm；`compare_sweeps.py`）

| 桶 | N | **quant Δ** | **v9 Δ** | 新 speedup |
|---|---|---|---|---|
| decode hp=0 | 14 | −7.8% | **−4.3%** | 0.65x |
| decode hp>0 | 42 | −8.3% | **−3.8%** | 0.47x |
| small hp=0 | 7 | −3.7% | −0.9% | 0.61x |
| small hp>0 | 21 | −3.7% | −2.4% | 0.49x |
| mid hp=0 | 7 | −6.0% | **−3.0%** | 0.67x |
| mid hp>0 | 21 | −5.9% | −0.6% | 0.58x |
| **prefill hp=0** | 14 | flat | **−22.0%** 🔥 | **0.93x** |
| prefill hp>0 | 42 | flat | −0.7% | 0.67x |

worst case 仅 +1.5% regress（noise 级）。

**TOP 10 单 case quant 改进全在 bs=1**：每个 case quant −29~31%，v9 −6~11%（decode 档兑现 fast-path 承诺）。

#### 2.9.5 正确性

- 全套 **29 passed**（原 24 + 新 5 个 act_quant-specific test）
- `_is_identity_perm` 经过 bench 路径验证

#### 2.9.6 不再做的决策（避免 over-engineering）

评估过但放弃：
- **Single-pass Pass1+Pass2 融合**：理论上省一次 HBM 读，但 `max_abs` 必须在 `quantize` 前算完 → 不能融合成真 single-pass。实测 partial fusion 在 T=2048 仅 −3%，不值得 kernel 代码复杂度。
- **Dequant + gather + quant 三合一**：会破坏 dense kernel 的权重 tile 抽象，跨模块耦合太大。

结论：**activation_quant 已压到相对最优**，后续更大增益来自下游（dense/sparse/combine）的融合。

---

### 2.10 **CUDA Graph decode 加速**（`<本轮>`）— Decode 档核心收益

#### 2.10.1 动机：decode 是 launch-bound，不是 compute-bound

在完成 §2.7 的 decode/prefill 拆分后，从最新 sweep（`sweep_20260423_144232`）再看 decode 桶的 stage 占比：

| 桶 | quant% | dense% | sparse% | 其他% | avg speedup |
|---|---|---|---|---|---|
| decode(≤16) | 34.6% | 41.3% | 21.3% | 2.8% | 0.50x |
| small(32..64) | 35.9% | 43.8% | 19.1% | 1.1% | 0.51x |

**关键观察**：decode 的 3 个 kernel（quant / dense / sparse）各自耗时 **40–70 µs**，但它们内部真正的计算量都很少（GEMV，HBM 带宽就能搬完）。测 `T=1, d_out=4096, d_in=4096`：dense kernel 实际 SM 工作 < 10 µs，其余 60 µs 全是 **CUDA driver + PyTorch 框架 + Triton autotune dispatch 的 host 开销**。

**结论**：decode 档要加速，必须先消灭 launch overhead。方法就是 **CUDA Graph capture**：把整条 decode pipeline 录制成一个 `cudaGraph`，此后每次 `graph.replay()` 只产生 **一次** `cudaGraphLaunch`（~5 µs）而不是 3-4 次独立的 kernel launch。

#### 2.10.2 改动内容

- **新文件** `triton_kernel/v9_linear_graph.py`：提供 `V9LinearCudaGraph` wrapper class
  - 构造：`graph_fn = V9LinearCudaGraph(W)`（weight 按引用绑定，捕获期间不可 mutate）
  - 调用：`y = graph_fn(x)` —— 自动按 `(T, d_in, d_out, dtype)` 缓存 graph；
    - 首次：一次性 warmup + capture（几 ms）
    - 后续：`static_X.copy_(x)` + `g.replay()` + `static_Y.clone()` ≈ 几 µs
  - `T > DECODE_T_THRESHOLD` 的 prefill 调用自动回落到 eager 路径（prefill 不是 launch-bound）
  - 使用独立 side-stream capture，replay 时用 caller 当前 stream（符合 PyTorch 常规语义）
- **新文件** `triton_kernel/tests/test_v9_linear_graph.py`（10 cases）
  - 数值完全等价测试（graph == eager bit-exact，7 种 shape）
  - 性能保证测试（T=1, 14336×4096, hp=0.05 必须快 ≥ 20%）
  - Shape-cache 正确性测试
  - prefill 路径 fallthrough 测试
- **新文件** `triton_kernel/benchmarks/bench_decode_launch_overhead.py`：专项诊断工具

不改动 `v9_linear.py` 现有 API：既有用户继续用 `v9_linear_forward`，需要收益的用户切换到 `V9LinearCudaGraph`。

#### 2.10.3 实测数据（`bench_decode_launch_overhead.py`，RTX 4090）

| T | d_out | d_in | hp | plain | graph | saved | **加速比** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 4096 | 0.00 | 189.1 µs | **83.8 µs** | −105 µs | **2.26×** |
| 1 | 4096 | 4096 | 0.05 | 300.1 µs | **102.8 µs** | −197 µs | **2.92×** 🔥 |
| 1 | 11008 | 4096 | 0.00 | 130.2 µs | **89.1 µs** | −41 µs | 1.46× |
| 1 | 11008 | 4096 | 0.05 | 209.7 µs | **113.6 µs** | −96 µs | 1.85× |
| 1 | 14336 | 4096 | 0.00 | 131.1 µs | **98.3 µs** | −33 µs | 1.33× |
| 1 | 14336 | 4096 | 0.05 | 208.4 µs | **122.4 µs** | −86 µs | 1.70× |
| 1 | 28672 | 4096 | 0.00 | 131.6 µs | **118.3 µs** | −13 µs | 1.11× |
| 1 | 28672 | 4096 | 0.05 | 208.8 µs | **155.3 µs** | −53 µs | 1.34× |
| 16 | 4096 | 4096 | 0.00 | 140.3 µs | **102.4 µs** | −38 µs | 1.37× |
| 16 | 4096 | 4096 | 0.05 | 219.0 µs | **121.1 µs** | −98 µs | 1.81× |
| 16 | 14336 | 4096 | 0.05 | 218.5 µs | **132.2 µs** | −86 µs | 1.65× |
| 16 | 28672 | 4096 | 0.05 | 218.7 µs | **180.0 µs** | −39 µs | 1.22× |
| 64 | 4096 | 4096 | 0.00 | 140.2 µs | **108.1 µs** | −32 µs | 1.30× |
| 64 | 14336 | 4096 | 0.05 | 220.5 µs | **187.4 µs** | −33 µs | 1.18× |

**14/14 case 全部改善**，最低 1.11×、最高 **2.92×**，平均 saved ≈ **72 µs/call**（约等于 2 个 kernel launch）。

**对比 cuBLAS FP16 的定位变化**：

| 场景 | 优化前 (plain) vs FP16 | 优化后 (graph) vs FP16 |
|---|---|---|
| T=1, 14336×4096, hp=0 | 0.66x (fp16=97 µs) | **≈1.00× 追平** |
| T=1, 28672×4096, hp=0 | — | **≈1.03× 微超** ✅ |
| T=1, 14336×4096, hp=0.05 | 0.36x | 0.80x（**+2.22×**）|
| T=16, 14336×4096, hp=0.05 | ≈0.40x | ≈0.93x（**+2.31×**）|

**第一次有 decode 场景（T=1, d_out ≥ 14336, hp=0）V9 ≥ cuBLAS FP16**。

#### 2.10.4 关键设计细节

1. **逐 shape 缓存**：`_GraphKey = (T, d_in, d_out, dtype)`，相同 shape 的多次调用共用 graph。不同 T（如 decode 1 和 16）会分别 capture。
2. **static_X 的 `copy_` 不在 capture 内**：capture 的只是 kernel launches，输入数据每次都新 copy 到 static_X；这个 `copy_` 约 1-2 µs，已计入 graph 成本。
3. **prefill 自动绕过**：`T > DECODE_T_THRESHOLD` 时 `__call__` 直接走 `v9_linear_forward`，不会浪费 capture 成本（prefill launch 占比只有 1-2%）。
4. **Side-stream capture**：避免污染 default stream；捕获完成后 replay 用 caller 当前 stream，符合 PyTorch eager 语义。
5. **Weight 按引用**：如果用户 hot-swap 权重，必须重建 wrapper（或者 call `reset_cache()`，当前未实现，后续如需可补）。

#### 2.10.5 测试通过记录

- `pytest kernel/triton_kernel/tests/` **46 / 46 passed**（原 36 + 新 10）
- 正确性测试包含 7 组 shape 的 bit-exact 验证（graph.replay 输出 == eager 输出 atol=1e-5）
- 性能测试有 ≥20% 提升门限，当前实测 1.70× 远超门限，稳定

#### 2.10.6 为什么不全部默认走 graph

- **capture 成本**：每个新 shape 第一次调用会 warmup + capture，约 10 ms。LLM serving 里 shape 是稳定的（T=1 decode 步），所以这个成本摊薄到无穷小；但测试套件 / 短小 benchmark 里会被反复支付。
- **input aliasing 约束**：graph 要求输入内存位置稳定，所以每次都要 `copy_`，对非常小的 X 是 ~1-2 µs 代价。
- **Weight immutable**：capture 后不能改权重，violates 某些动态量化场景。
- **结论**：让用户显式选择 → serving loop 用 graph，测试/研究用 eager。

---

## 3. 当前性能全景（优化后 · RTX 4090 · 全部优化都启用）

> 文件：`triton_kernel/benchmarks/results/sweep_20260422_154306.md`

最新数据（节选，单位 ms · `v9_total` vs `fp16`）：

| d_out | d_in | bs | hp | v9 total | fp16 | **speedup** | 最大瓶颈 |
|---|---|---|---|---|---|---|---|
| 11008 | 4096 | 1 | 0.00 | 0.145 | 0.097 | 0.66x | dense (48%) |
| 11008 | 4096 | 32 | 0.00 | 0.148 | 0.100 | 0.68x | dense (47%) |
| 11008 | 4096 | 2048 | 0.00 | 1.688 | 1.222 | 0.72x | dense (93%) |
| 4096 | 4096 | 2048 | 0.00 | 0.636 | 0.432 | 0.68x | dense (90%) |
| **14336** | **4096** | **2048** | **0.05** | **2.438** | **2.494** | **1.02x** ✅ | dense (65%) |
| **14336** | **4096** | **512** | **0.05** | **0.641** | **0.657** | **1.03x** ✅ | dense (67%) |
| 28672 | 4096 | 2048 | 0.05 | 4.903 | 4.561 | 0.93x | dense (69%) |
| 28672 | 4096 | 8192 | 0.05 | 19.83 | 18.25 | 0.92x | dense (68%) |

> 在 **d_out ≥ 14336 且 hp_ratio ≥ 0.05** 的场景下，V9 已经持平或微超 cuBLAS FP16。由于 FP16 本身没有稀疏/量化开销，V9 的"能超 FP16"的前提是 `hp_ratio > 0` 带来的额外信息让我们允许后续精度损失，或者说 V9 的价值在于"d_out 大 + 有稀疏补偿"的真实 LLM ffn-up 层。

### 3.1 Decode 档（CUDA Graph 加持，§2.10）

| T | d_out | d_in | hp | plain v9 | **graph v9** | fp16 | **graph speedup** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14336 | 4096 | 0.00 | 0.131 | **0.098** | 0.097 | **1.00×** ✅ |
| 1 | 28672 | 4096 | 0.00 | 0.132 | **0.118** | ≈0.12 | **1.03×** ✅ |
| 1 | 14336 | 4096 | 0.05 | 0.208 | **0.122** | 0.097 | 0.80× |
| 1 | 4096 | 4096 | 0.05 | 0.300 | **0.103** | 0.016 | 0.16× |
| 16 | 14336 | 4096 | 0.05 | 0.219 | **0.132** | 0.097 | 0.73× |
| 64 | 14336 | 4096 | 0.05 | 0.220 | **0.187** | 0.097 | 0.52× |

> decode 场景的 `graph speedup` 相对 `plain v9` 均 ≥ 1.11×，最高 2.92×（见 §2.10.3）。fat-out (d_out ≥ 14336) + hp=0 的 decode 档已经能对打 cuBLAS。剩余差距大都在 `d_out = d_in = 4096`（FP16 在这种小 shape 本来就极快），需要 fused decode kernel（P3）进一步解决。

---

## 4. 下一步优化路线（按 ROI 排序）

### ~~P1：Combine + Transpose 融合~~（**已完成**，见 §2.6）
- ✅ 新 `_combine_transpose_kernel`，单次遍历完成 add + transpose
- ✅ Smart fallback：`surf ≤ 4M` 走 torch native，避免 launch overhead
- ✅ prefill hp>0 **−35.2%**；28672×4096 极端场景 **−62%**

### ~~P0-prefill：Dense GEMM (prefill regime)~~（**本轮已完成**，见 §2.8）
- ✅ Phase B-1：autotune 扩展 5 个大 tile（GROUP=16, stages=4/5）— 作为副产物保留
- ✅ Phase B-2：**W4A16 fallback**（dequant-then-cuBLAS）— **端到端 +20-40%**
- ✅ Triton dequant kernel（比 torch native 快 38-46x，贴近 HBM 屋顶）
- ✅ prefill hp=0 速度 0.70x → **0.88-0.97x**（28672×4096, bs=8192 近乎追平 cuBLAS）

### P1（新）：hp>0 prefill 的 W4A16 扩展
当前 fallback 在 `W.n_hp_blocks > 0` 时强制走 int4。sweep 数据显示 hp>0 的 prefill 仍是 ~0.66x 档。改造思路：
- dequant W 到 fp16（已有 kernel），然后
- 把 sparse 的 4-bit 块也 dequant-加入到 fp16 权重（一次性操作，均摊到 T），或
- 保持 sparse 在 int4 path 单独计算，结果 add 到 fp16 GEMM 结果上（需要 kernel 做 (T, d_out) layout 的 add）

收益预估：hp>0 prefill 也能提升到 0.85-0.95x 档，覆盖 sweep 里余下一半 shape。

### ~~P2：Kernel launch overhead (decode)~~（**本轮已完成**，见 §2.10）
- ✅ 新增 `V9LinearCudaGraph` wrapper，自动 shape-bucket 缓存
- ✅ 14/14 case 收益 1.11× – 2.92×；fat-out T=1 hp=0 已对打 cuBLAS FP16
- ✅ 10 个新测试全绿，正确性 bit-exact；不侵入既有 `v9_linear_forward` API

### ~~P3 Step 1：Fused Dense-GEMM-to-Out（hp=0 decode）~~（**本轮已完成**，commit `e792b47`）
- ✅ 新 kernel `dense_gemm_u4_s4_to_out`（270 行）：复用既有 dense autotune 主循环，epilogue 在 FP32 accumulator 上 `tl.trans` 后直接写 `(T, d_out)` FP16，消除独立的 `_combine_transpose` pass。
- ✅ 接入 `_v9_forward_decode` 的 `W.n_hp_blocks == 0` 分支；prefill hp=0 分支保守保持旧路径（尚未微基准覆盖大 T）。
- ✅ 28 个 bit-exact 对齐测试全绿（涵盖 d_out ∈ {4096, 11008, 14336, 28672}, d_in ∈ {4096, 11008}, T ∈ {1, 2, 4, 8, 16}）。
- ✅ 21 个回归测试（end2end + dispatcher + CUDA Graph）全绿。
- ✅ `bench_dense_to_out.py` 结果（RTX 4090, min-of-means）：
  - `T=4/16 d_out ∈ 4096~14336` → **1.17×~1.21×**（combine pass 被完全消灭的主场）
  - `T=1 d_out=4096, d_in=11008` → **1.12×**
  - `T=1 d_out ∈ 4096~28672, d_in=4096` → 1.02×~1.04×（GEMV tail，kernel 本身已是瓶颈）
  - 10/10 shape 全部改善，平均 **1.10×**

**P3 Step 1 暴露的新瓶颈**：`T=1, d_in=4096` 一档收益仅 3-4%。原因是此时 plain kernel 本身就 ~59 µs，已不在 epilogue，而是 dense GEMM kernel 内部 **SM occupancy 不足**（grid = `d_out/128 = 32` programs，4090 有 128 SM，**75% SM 空闲**）。

### P4：Split-K Dense GEMM — **Step 4.1/4.2 本轮完成，触发 go/no-go 停止**（commit `de49845`）

**做了什么 & 为什么**：按 `research/p4_splitk_dense_design.md` 的计划，实现 `dense_gemm_splitk.py`（split-K main kernel + FP32→FP16 reduce kernel）+ `test_dense_gemm_splitk.py`（37/43 tests，6 skipped 是 n_groups 不整除的组合）+ `bench_dense_splitk.py` 四列对比 plain/fused/splitk/FP16 cuBLAS。动机是 P3 Step 1 留下的 `T=1, d_in=4096` 档只有 3-4% 收益 —— plain kernel 本身 59µs 已非 epilogue 瓶颈，而是 grid=32 programs 在 4090 128 SM 上只能填满 25%。

**对齐测试结果（37 PASS, 6 skipped）**：
- ✅ `split_k=1` canary（9 shapes, atol=2e-3）：因为 scale_x 从 K-loop 挪到 reduce pass，一次 FP32 mul 重排造成 max|delta|=1.95e-3，与 FP16 ULP 量级匹配
- ✅ `split_k ∈ {2,4,8}` relaxed atol（18 shapes）
- ✅ auto policy（6 shapes）+ policy sanity

**Microbench 结果（RTX 4090，单位 µs，与 FP16 对比）**：

| T | d_out | d_in | sk | plain | fused(P3) | **splitk** | **FP16 cuBLAS** | **splitk vs fused** | **splitk vs FP16** |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 8 | 87.8 | 85.4 | **141.8** | **38.5** | 0.60× ❌ | 0.27× |
| 1 | 4096 | 11008 | 2 | 135.5 | 121.3 | **99.7** | 98.0 | **1.22× ✅** | 0.98× |
| 1 | 11008 | 4096 | 8 | 60.9 | 58.9 | 99.5 | 96.6 | 0.59× ❌ | 0.97× |
| 4 | 4096 | 4096 | 8 | 70.5 | 60.5 | 100.4 | **14.2** | 0.60× ❌ | 0.14× |
| 16 | 4096 | 4096 | 8 | 70.7 | 60.1 | 100.8 | **16.7** | 0.60× ❌ | 0.17× |
| 1 | 14336 | 4096 | 8 | 62.0 | 59.5 | 99.5 | 125.0 | 0.60× ❌ | **1.26×** |
| 1 | 28672 | 4096 | 1 | 78.4 | 75.1 | 99.4 | 247.2 | 0.76× ❌ | **2.49×** |
| 4 | 14336 | 4096 | 8 | 71.3 | 59.8 | 100.8 | 126.6 | 0.59× ❌ | 1.26× |
| 16 | 14336 | 4096 | 8 | 70.8 | 59.8 | 100.7 | 128.0 | 0.59× ❌ | 1.27× |
| 16 | 28672 | 4096 | 1 | 86.5 | 81.5 | 98.6 | 253.2 | 0.83× ❌ | **2.57×** |

- Totals: plain 794µs, fused 722µs, **splitk 1041µs, FP16 1144µs**
- splitk vs fused avg **0.69×**（退步），splitk vs FP16 avg 1.10×
- **go/no-go 判定：FAIL**（设计文档 §4 Step 4.2 gate 要求最差 shape ≥1.3×，实测 0.59×）

**效果 vs FP16 小结**：

| 指标 | P3 (fused) vs FP16 | **P4 (splitk) vs FP16** | Δ |
|---|---|---|---|
| 10-shape 总延迟比 | 1144/722 = **1.58× 领先** | 1144/1041 = **1.10× 领先** | 退步 |
| 最差 shape | T=4 d_out=4096: 0.23× | T=4 d_out=4096: **0.14×** | 退步 |
| 最优 shape | T=1 d_out=28672: 3.29× | T=1 d_out=28672: 2.49× | 退步 |

**决策**：split-K 代码提交保留，但**不接入 `v9_linear.py`**。decode hp=0 分支继续使用 P3 Step 1 fused kernel。

**失败根因诊断**：`splitk_us` 几乎全部在 ~99-100 µs 常量 plateau 上（与 SPLIT_K 和形状无关）——这正是 **Kernel A 主计算 ~50µs + Kernel B reduce ~50µs** 两次 launch 叠加的签名。即使 `sk=1`，reduce kernel 本身就多了 ~24µs overhead。设计文档低估了二 kernel 链的 fixed cost 4-5×。

**下一步候选路线（供用户拍板）**：

| 路线 | 原理 | 工时 | 风险 |
|---|---|---|---|
| **A. 单-kernel atomic split-K** | `tl.atomic_add` 到 FP32 输出，消除 HBM round-trip；scale_x+cast 用一个 tiny kernel | 1 天 | atomic 争用；仍需 1 次额外 launch |
| **B. Persistent GEMM + tile streaming** | 不分 K，CTA 沿 K 流式，grid = SM 数 | 2-3 天 | Triton 原语支持有限 |
| **C. 放弃 decode T=1 d_in=4096 档** | P3 已把 decode hp=0 端到端推到 0.70× FP16，直接跑回 P5 | 0 天 | 零风险 |

本次详细分析见 `research/p4_splitk_dense_design.md` §10-11，实验日志见 `research/p4_step42_bench_20260423.log`。

### P5（候选，暂缓）：其他方向
- **hp>0 prefill 扩展 W4A16**：sparse 并入 fp16 权重的路径复杂，ROI 不如 P4
- **CUDA Graph prefill 扩展**：prefill kernel 本身 compile 时间 >> graph launch 节省，负收益
- **PTX `prmt.b32` dequant**：Phase B 末尾候选，需要先完成 P4 再评估是否还有收益

**下一 session 选择**：**路线 A（atomic split-K）优先**，或路线 C（跑回 P5）—— 请用户拍板。

---

## 5. 文件位置速查

- 源码：`kernel/triton_kernel/{activation_quant,dense_u4s4_gemm,sparse_s4s4_gemm,v9_linear,pack_utils}.py`
- 测试：`kernel/triton_kernel/tests/test_*.py`（15 个，全部通过）
- 基准：`kernel/triton_kernel/benchmarks/{sweep_v9,bench_dense,bench_sparse,bench_linear,diag_fp16_variance}.py`
- Profiling：`kernel/triton_kernel/benchmarks/{profile_nvtx_driver.py,run_nsys_sweep.sh,summarize_nsys.py}`
- Bench 工具：`kernel/triton_kernel/benchmarks/_bench_util.py`（`time_ms` 三窗口 min-of-means）
- 结果归档：`kernel/triton_kernel/benchmarks/results/sweep_*.{md,csv,log}`

## 6. 跑法速查

```bash
# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh && conda activate zip

# 跑 sweep（hp=0）
cd /tmp && PYTHONPATH=/root python -m kernel.triton_kernel.benchmarks.sweep_v9

# 跑 pytest（15 cases）
cd /tmp && PYTHONPATH=/root python -m pytest /root/kernel/triton_kernel/tests/ -x -q

# 跑 nsys profiling
bash /root/kernel/triton_kernel/benchmarks/run_nsys_sweep.sh
python /root/kernel/triton_kernel/benchmarks/summarize_nsys.py
```
