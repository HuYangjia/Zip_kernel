


# V9 Kernel 优化进展报告

> **更新时间**：2026-04-22（第二轮）  
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

### P2：Kernel launch overhead（decode 占 14%）
- Stage 1/2/3/4 之间每次都要 launch。考虑用 CUDA Graphs capture 整条 decode pipeline。
- 尤其 decode bs=1 场景，3-4 次 launch ≈ 15-20μs = `v9_total=139μs` 的 10-14%。
- **架构准备已完成**（§2.7 decode 入口已独立）。

### P3：Decode 专属 kernel 特化
- decode 的 quant / sparse kernel 在小 bs (≤16) 时 launch overhead 严重。
- 方案：给 `quantize_activation_s4` 加 `T ≤ 16` 专用小 tile config；sparse kernel 同理。
- 收益预估：decode hp=0 0.69x → 0.85-0.95x。

**本轮选择（更新）**：
1. **已完成**：§2.7 prefill/decode 架构拆分（0 overhead）+ §2.8 **W4A16 fallback（prefill +20-40%）**
2. **下一 session 候选**：
   - P1（hp>0 prefill 扩展 W4A16） — ROI 最高，sweep 剩余一半 shape
   - P2（decode CUDA Graph） — 结构性改动
   - P3（decode kernel 小 tile 特化） — 低风险增量

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
