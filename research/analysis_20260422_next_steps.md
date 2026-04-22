# V9 Kernel 下一步优化方向分析 — 2026-04-22

- Sweep 数据源：`triton_kernel/benchmarks/results/sweep_20260422_154306.csv`（168 shapes）
- Commit baseline：`2b55f26`（Stage 4 combine+transpose 融合已完成）
- 硬件：RTX 4090（HBM 峰值 ≈ 1008 GB/s，FP16 TC 峰值 ≈ 330 TFLOPS）
- 分析工具：`kernel/research/tools/analyze_sweep_bottleneck.py`

---

## 1. 各 Stage 在 v9_total 中的占比（按场景分桶）

| bs_tier           | hp   |  N | quant  | dense  | sparse | comb  | v9 (ms) | fp16 (ms) | speed |
|-------------------|------|---:|-------:|-------:|-------:|------:|--------:|----------:|------:|
| decode (1–16)     | hp=0 | 14 | 43.8%  | 49.0%  |  0.0%  |  2.8% | 0.199   | 0.125     | 0.69x |
| decode (1–16)     | hp>0 | 42 | 33.0%  | 36.6%  | 26.7%  |  4.0% | 0.265   | 0.119     | 0.47x |
| small (32–64)     | hp=0 |  7 | 42.0%  | 52.0%  |  0.0%  |  5.9% | 0.228   | 0.138     | 0.62x |
| small (32–64)     | hp>0 | 21 | 33.2%  | 40.6%  | 24.6%  |  6.0% | 0.287   | 0.137     | 0.48x |
| mid (128–512)     | hp=0 |  7 | 17.9%  | 77.5%  |  0.0%  |  7.0% | 0.638   | 0.413     | 0.62x |
| mid (128–512)     | hp>0 | 21 | 15.9%  | 69.6%  | 12.9%  |  7.2% | 0.712   | 0.410     | 0.56x |
| **prefill (≥2K)** | hp=0 | 14 |  5.2%  | **91.4%** | 0.0% |  4.3% | 5.552   | 4.095     | 0.73x |
| **prefill (≥2K)** | hp>0 | 42 |  4.8%  | **83.4%** | 7.1% |  5.6% | 6.111   | 4.086     | 0.66x |

## 2. Amdahl 上界（若将某 stage 归零，可达的平均 speedup）

| bs_tier       | hp   | elim_q | **elim_d**   | elim_s | elim_c | current |
|---------------|------|-------:|-------------:|-------:|-------:|--------:|
| decode        | hp=0 | 1.21x  | **1.39x**    | –      | 0.71x  | 0.69x   |
| decode        | hp>0 | 0.70x  | 0.75x        | 0.65x  | 0.49x  | 0.47x   |
| small         | hp=0 | 1.03x  | **1.41x**    | –      | 0.66x  | 0.62x   |
| small         | hp>0 | 0.72x  | 0.87x        | 0.64x  | 0.52x  | 0.48x   |
| mid           | hp=0 | 0.76x  | **4.11x**    | –      | 0.67x  | 0.62x   |
| mid           | hp>0 | 0.66x  | **2.22x**    | 0.64x  | 0.60x  | 0.56x   |
| **prefill**   | hp=0 | 0.77x  | **9.03x** 🔥 | –      | 0.76x  | 0.73x   |
| **prefill**   | hp>0 | 0.69x  | **4.13x** 🔥 | 0.71x  | 0.70x  | 0.66x   |

结论：**prefill 以 dense 为主、decode 以 quant+sparse 为主**，两类场景的瓶颈完全不同，无法用一套 kernel 同时照顾。

## 3. dense_ms / fp16_ms 比值（核心信号）

| bs_tier | median | best  | worst |
|---------|-------:|------:|------:|
| decode  | 0.73x  | 0.32x | 4.24x |
| small   | 0.90x  | 0.53x | 3.56x |
| mid     | **1.21x** | 1.19x | 1.32x |
| prefill | **1.27x** | 1.06x | 1.35x |

- **prefill 的 dense 本身就比 cuBLAS FP16 慢 27%**——这是 prefill 无法超过 FP16 的主因。
- decode 在某些大 `d_out`（28672×4096, bs=1）反而比 cuBLAS **快 3x**（0.32x 比值 = 快 3.1x），这是 4-bit 权重节省带宽的胜利场景。

## 4. Dense 带宽利用率（hp=0）

| d_out | d_in | bs   | dense_ms | GB/s  | vs HBM peak | dense/fp16 |
|------:|-----:|-----:|---------:|------:|------------:|-----------:|
|  4096 | 4096 |    1 | 0.069    | 122   | 12.1%       | 1.80x      |
| 11008 | 4096 |    1 | 0.069    | 327   | 32.4%       | 0.71x      |
| 28672 | 4096 |    1 | 0.079    | **742** | **73.6%** | **0.32x**  |
|  4096 | 4096 |  512 | 0.145    |  72   |  7.2%       | 1.29x      |
| 11008 | 4096 |  512 | 0.428    |  58   |  5.7%       | 1.32x      |
| 28672 | 4096 |  512 |  1.029   |  59   |  5.9%       | 1.21x      |
|  4096 | 4096 | 2048 |  0.569   |  30   |  2.9%       | 1.32x      |
| 28672 | 4096 | 2048 |  4.148   |  16   |  1.6%       | 1.29x      |

- **Decode 场景是 memory-bound**：bs=1, d_out=28672 吃到 73% 带宽，接近屋顶，没什么可优化了（唯一可做的是 launch overhead）。
- **Prefill 场景是 compute-bound**：带宽利用率掉到 1.6-7%，但 dense_ms 依然比 cuBLAS 慢——说明 **Tensor Core 利用率严重不足**，cuBLAS 能塞满 TC，我们没塞满。

## 5. Sparse 成本（被低估）

| bs_tier | avg sparse_ms | avg share of v9 |
|---------|--------------:|----------------:|
| decode  | 0.068         | **26.7%**       |
| small   | 0.068         | 24.6%           |
| mid     | 0.081         | 12.9%           |
| prefill | 0.429         |  7.1%           |

- 小 bs 下 sparse 绝对成本 68μs 基本固定——**典型的 launch overhead + 大 tile autotune 不匹配**现象。
- prefill 下 sparse 的 429μs 是真的在做活（对应 hp=0.1, bs=8192, 行数多），优化空间有限。

---

## 6. 最终优化路线图（你已决定走此路线）

### Phase A：Kernel 按规模二分

把 `v9_linear_forward` 拆成：

- `v9_linear_prefill(X, W)`：`T ≥ 256` 或 `T × d_out ≥ 2M` 场景使用
- `v9_linear_decode(X, W)` ：`T ≤ 128` 场景使用
- 一个自动 dispatcher 根据 `X.shape[0]` 选择

**动因**：
- 两类场景的 kernel autotune config 空间完全不一样（prefill 需要大 BLOCK + TC-heavy；decode 需要小 BLOCK + 低 warps 降 launch）
- 共用一套 autotune 会导致 autotune key 爆炸、first-run 编译时间暴涨
- 分开还能用**完全不同的 pipeline 结构**（例如 decode 可以走 persistent kernel / CUDA Graph，prefill 可以走 Split-K）

### Phase B：Prefill 专项优化（target: dense/fp16 从 1.27x → 1.0x）

**P0-1 扩展 autotune 配置**：加入 `BM=128, BN=256, BK=32, num_stages=4, num_warps=8` 这种大 TC tile；引入 Split-K 处理 `d_in ≥ 8192` 场景。

**P0-2 Dequant 向量化**：用 `tl.inline_asm_elementwise` + PTX `prmt.b32` / `lop3.b32` 替代 Triton 软件 unpack，节省 dequant 路径 1.3-1.5x。

**P0-3 K-loop software pipelining**：显式 double-buffer producer/consumer，打穿 Triton 自动 pipeline 限制。

**预期**：prefill hp=0 speedup 0.73x → **0.95-1.05x**（至少追平 FP16，大形状反超）。

### Phase C：Decode 专项优化（target: speedup 0.47-0.69x → 0.75-0.95x）

**P1-1 Quant kernel 小 bs 专用 config**：`T ≤ 16` 跳过 autotune、走固定 `BT=16, BD=256, num_warps=2` 配置；或者提供 `activation_quant_decode` 专用函数。

**P1-2 Sparse kernel 小 bs 专用 config**：类似 P1-1，加小 tile 到 autotune 或单独提供 decode 版本。

**P1-3 Sparse + Combine 融合**：sparse 输出直接 atomic_add 到 `(T, d_out)` 的 `Y_out`，省掉一次 add pass。

**P1-4（终极）整条 decode pipeline 走 CUDA Graph**：capture + replay，把 4 个 stage 的 ~30μs launch overhead 压到 ~5μs。

**预期**：decode hp=0 speedup 0.69x → **0.85-0.95x**；decode hp>0 从 0.47x → **0.70-0.80x**。

---

## 7. 本次决策日志

- **本次决定**：采纳 Phase A（分 prefill/decode 两版本 + dispatcher）
- **下一 session 首要任务**：实现 Phase B P0-1（prefill 扩展 autotune）并验证
- **暂不做**：P1-4 CUDA Graph（依赖 Phase A 完成后的 decode 版本稳定再做）

## 8. 产物清单

- `kernel/research/tools/analyze_sweep_bottleneck.py`（分析工具，长期复用）
- `kernel/research/analysis_20260422_next_steps.md`（本文档）
- Baseline sweep：`triton_kernel/benchmarks/results/sweep_20260422_154306.{md,csv,log}`
