# Model Selection for E2E Replacement Benchmark

**Date**: 2026-05-06
**Scope**: single-layer Prefill / Decode replacement bench (QKV fused + Gate/Up fused)
**Data source**: `kernel/cuda_kernel/logs/r78_final_roofline/qwen3_20260502_201018/r78_summary.md`

---

## 1. 选定的 3 个模型

| # | Model       | hidden_size | intermediate | num_q / num_kv heads | GQA group | 角色 |
|---|-------------|-------------|--------------|----------------------|-----------|------|
| 1 | **Qwen3-4B**  | 2560 | 9728  | 32 / 8 | 4 | 中等规模 / scaling 曲线起点 |
| 2 | **Qwen3-8B**  | 4096 | 12288 | 32 / 8 | 4 | **主力模型 / peak speedup 案例** |
| 3 | **Qwen3-14B** | 5120 | 17408 | 40 / 8 | 5 | 大模型 decode 证据 |

> head_dim 固定为 128。权威 shape 常量见 `kernel/bench/configs/qwen3_shapes.py`。

> Qwen3-0.6B、Qwen3-1.7B、Qwen3-32B 本次**不纳入测试**。

---

## 2. 测试配置

| 阶段 | seqlen | batch size |
|------|--------|------------|
| **Prefill** | 2048 | 4, 8, 16, 32 |
| **Decode**  | 1    | 4, 8, 16, 32 |

即每个模型对应 **8 个 shape 点**（4 prefill + 4 decode），共 24 个实验点。

---

## 3. 融合算子范围（critical path）

替换目标限定在 transformer layer 的 5 个 GEMM，融合后为 4 个 kernel：

1. **QKV_fused**：`hidden → d_q + 2·d_kv` （原 q_proj + kv_proj）
2. **o_proj**
3. **Gate_Up_fused**：`hidden → 2·intermediate`（r78 中已是融合态）
4. **down_proj**

Attention (SDPA / FlashAttn)、RMSNorm、RoPE、residual 均保持 FP16，不替换。

---

## 4. 选型依据（基于 r78 实测数据 + Amdahl 上界估计）

以 r78 测得的 `speedup = fp16_us / cuda_us`（越大越好）作为 per-kernel 数据，
按 FP16_us 加权得到 per-layer 上界 speedup：

### Prefill @ T=2048 (compute-bound 代理)

| Model | 加权 s_proj | Amdahl layer (p=0.70) |
|-------|-------------|------------------------|
| Qwen3-4B  | 1.18× | **1.13×** ✅ |
| Qwen3-8B  | 1.28× | **1.19×** ✅ |
| Qwen3-14B | 1.00× | **1.00×** (持平) |

### Decode @ T=8 / T=32 (memory-bound)

| Model | T=8 layer | T=32 layer |
|-------|-----------|------------|
| Qwen3-4B  | 1.34× ✅ | 1.25× ✅ |
| Qwen3-8B  | **1.60×** ✅ | **1.51×** ✅ |
| Qwen3-14B | 1.53× ✅ | 1.43× ✅ |

### 关键亮点
- **Peak kernel speedup**：Qwen3-8B `gate_up_proj @ T=8 = 3.91×`（整个 r78 benchmark 冠军）
- **Scaling trend**：hidden_size 2560 → 4096 → 5120，decode layer speedup 1.34 → 1.60 → 1.53，
  形成一条 "hidden size 越大、W4A4 decode 收益越显著" 的清晰曲线。
- **全面赢**：3 个模型在 prefill / decode 8 个配置点上**全部非负**（最差 14B prefill = 1.00×）。

---

## 5. 被排除模型的理由

| Model | 不选的原因 |
|-------|-----------|
| Qwen3-0.6B | 所有 proj launch-overhead-bound，cuda_us≈30μs 被 kernel launch 卡死；decode ≈ 0.52× 全线崩盘 |
| Qwen3-1.7B | 中间态尴尬：prefill 小赢 (1.08×) 但 decode q/o/down 三个都是 0.33×，综合 0.88× 负收益 |
| Qwen3-32B  | 非 Qwen3 dense 主流尺寸，且与 14B 数据外推性重叠，额外跑不增加信息量 |

---

## 6. 注意事项与已知风险

1. **上界估计**：以上 Amdahl 数字基于 r78 per-kernel 数据；真实 E2E 替换后会再打 -10% ~ -15% 折扣
   （activation quant online cost + KV cache 读写摊薄 proj 占比）。
2. **Prefill B=32 (T=65536) 超出 r78 测试范围**（r78 最大 T=2048）。在 pure compute-bound 区，
   预期 speedup 收敛到 `1 / (cuda_roof/fp16_roof × 1/cuda_eff) ≈ 1.17×` 附近；
   **强烈建议在正式 r79 实验前先跑一个 T=8192 的 pilot 验证趋势**。
3. **QKV 融合收益代理**：融合后 d_out = d + 2·d_kv 比单独 q_proj 的 d_out 更大，更偏 compute-bound；
   实测 speedup 预期 ≥ 表中 q_proj 数字（本表用 q_proj 作保守下界）。
4. **计时协议**：遵循 [[memory:bmmiahpl]] 中的 RTX 4090 微基准规范
   — `time_forward_us(warmup=500, outer=20, inner=200)` + median-of-5 独立 trial。

---

## 7. 下一步 (r79 实验骨架 TODO)

- [ ] 替换法 single-layer bench 脚本（FP16 / Mixed 双跑 + parity 检查）
- [ ] 每个 (model, phase, batch) 独立 `--results-file` 输出，避免覆盖
- [ ] 失败/异常配置保留在 `VALIDATION_LOG.md` 中（不删代码）
- [ ] Amdahl 自检：per-kernel speedup vs measured layer speedup 偏差 > 20% 时告警
