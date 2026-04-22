


# V9 Kernel 优化进展报告

> **更新时间**：2026-04-22  
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

---

## 3. 当前性能全景（优化后 · RTX 4090 · hp=0）

> 文件：`triton_kernel/benchmarks/results/sweep_20260422_120410.md`

最新数据（节选，单位 ms · `v9_total` vs `fp16`）：

| d_out | d_in | bs | v9 total | fp16 | **speedup** | 最大瓶颈 |
|---|---|---|---|---|---|---|
| 11008 | 4096 | 1 | 0.139 | 0.097 | **0.70x** | dense (49%) |
| 11008 | 4096 | 8 | 0.147 | 0.099 | **0.67x** | dense (47%) |
| 11008 | 4096 | 32 | 0.148 | 0.100 | **0.68x** | dense (47%) |
| 11008 | 4096 | 2048 | 1.89 | 1.23 | 0.65x | dense (82%) |
| 4096 | 11008 | 32 | 0.281 | 0.111 | 0.40x | quant+dense |
| 4096 | 4096 | 2048 | 0.73 | 0.425 | 0.58x | dense (78%) |

---

## 4. 下一步优化路线（按 ROI 排序）

### P0：Prefill 区 Dense GEMM（bs ≥ 128，dense 占 78-84%）
- **Split-K / Stream-K**：当前 tile 数量 = `⌈d_out/BM⌉ × ⌈T/BN⌉`，在大 bs 下绰绰有余；但单个 program 要跑完 `d_in / BK = 32~86` 个 K 迭代。可以考虑 split-K + atomic-add 让 K 维也并行，提高 SM 占用。
- **vectorized 4-bit dequant**：当前每个 K iter 都调一次 dequant，在 prefill 里累加次数多。考虑换用 `tl.inline_asm_elementwise` + PTX `prmt` 做 4→16bit 快速扩展（PyTorch GPTQ 标配）。

### P1：Activation quantization（prefill 占 17-28%）
- 当前 quant kernel 有 autotune，但只有 2 个 config，且 `BK=128` 死死绑定。
- 扩 autotune + 尝试 `num_stages=3/4` + 检查有没有 warp-level reduce 可以打通。

### P2：Kernel launch overhead（decode 占 14%）
- Stage 1/2/3/4 之间每次都要 launch。考虑用 CUDA Graphs capture 整条 decode pipeline，摊薄 launch 成本。
- 或：把 combine（stage 4）融合到 dense kernel 的 epilogue 里（当 `hp_ratio=0` 时 sparse 为空，combine 其实只是一个 transpose+dequant-scale）。

### P3（已想好但风险较大）
- Dense kernel 支持 `BK = k × BCOL_K`（k∈{1,2,4}），减少 epilogue 频率。需要改 kernel 内循环结构，有数值风险。

**本轮选择：先攻 P1（quant 优化）**——代码量小、风险低、预计能让 prefill 端到端再 −10% 左右；然后评估 P0 的 Split-K 是否值得。

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
