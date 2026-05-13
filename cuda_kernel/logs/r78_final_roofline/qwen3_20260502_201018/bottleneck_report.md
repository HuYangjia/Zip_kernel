# r78 Bottleneck Attribution Report

**Source**: `cuda_kernel/logs/r78_final_roofline/qwen3_20260502_201018/bench.json`  
**GPU**: RTX 4090, ACHIEVABLE_FRACTION=0.85  
**Total shapes**: 245

## §1 是否达到瓶颈：按 cuda_efficiency 分桶

> `cuda_efficiency = cuda_roof_us / cuda_us`，衡量实测速度距离 W4A4 的物理上限有多近。

> - `at-roof` (≥80%)：已到瓶颈，继续优化 ROI 极低
> - `mid-gap` (50-80%)：kernel 层仍有改进空间
> - `large-gap` (<50%)：显著 under-utilize，要找根因

| status | N | 占比 | median cuda_eff | median speedup vs FP16 |
|:---|---:|---:|---:|---:|
| **at-roof** | 4 | 1.6% | 91.1% | 3.38× |
| **mid-gap** | 27 | 11.0% | 58.3% | 2.16× |
| **large-gap** | 214 | 87.3% | 32.9% | 1.00× |

## §2 理论瓶颈所在（roofline 侧）

> 在达到 roofline 的前提下，kernel 最终会被谁卡住？

> - `hbm-bound`：T≥2 的 gemm 阶段 mem 时间 > compute 时间（HBM 带宽是上限）
> - `tc-bound`：gemm 阶段 compute 时间 > mem 时间（INT4 TC 峰值是上限）
> - `quant+hbm/tc`：quant 阶段占 roofline ≥30%，额外串联开销显著
> - `fused-mem`：T=1 fused GEMV 纯内存受限
> - `fused-tc`：T=1 fused GEMV 计算受限（极少，仅超扁平 shape 出现）

| bound | N | 占比 | median cuda_eff | at-roof 命中 | median speedup vs FP16 |
|:---|---:|---:|---:|---:|---:|
| hbm-bound | 106 | 43.3% | 28.5% | 4/106 | 1.02× |
| tc-bound | 96 | 39.2% | 35.1% | 0/96 | 1.02× |
| fused-mem | 35 | 14.3% | 41.3% | 0/35 | 1.71× |
| quant+tc | 6 | 2.4% | 38.9% | 0/6 | 0.76× |
| quant+hbm | 2 | 0.8% | 8.8% | 0/2 | 0.27× |

## §3 实际瓶颈所在（gap 根因）

> 对于 `mid-gap + large-gap` 的 shape，到底是什么在拖慢 kernel？

> 经验规则：

> - `launch-bound`：T≤32 且 cuda_us 贴着 30-36us 地板（kernel launch/dispatcher 开销占主导）
> - `gemv-tail`：T=1 GEMV，尾 wave 不满 + 小 d_in 寄存器利用率差
> - `epilogue-fma`：tc-bound 但 eff<80%，INT4 TC 峰值达不到（per-group dequant FMA 在 CUDA-core 执行）
> - `mem-access-suboptimal`：hbm-bound 但 eff<80%（pack/stage/L2 利用未最优）

| gap_cause | N | median cuda_eff | median cuda_us | median speedup |
|:---|---:|---:|---:|---:|
| **epilogue-fma** | 102 | 35.1% | 234.4 us | 1.01× |
| **mem-access-suboptimal** | 73 | 31.9% | 54.9 us | 1.16× |
| **gemv-tail** | 35 | 41.3% | 25.5 us | 1.71× |
| **launch-bound** | 29 | 9.6% | 34.9 us | 0.43× |
| **other** | 2 | 8.8% | 35.0 us | 0.27× |

## §4 达到瓶颈的 shape 清单（`at-roof` 全量）

共 **4** 个 shape 达到 ≥80% W4A4 roofline：

| model | proj | T | shape | bound | cuda_us | cuda_roof_us | cuda_eff | speedup vs FP16 |
|:---|:---|---:|:---:|:---|---:|---:|---:|---:|
| Qwen3-8B | gate_up_proj | 8 | 4096→24576 | hbm-bound | 62.0 | 62.99 | 101.6% | 3.91× |
| Qwen3-8B | gate_up_proj | 32 | 4096→24576 | hbm-bound | 68.3 | 64.71 | 94.7% | 3.50× |
| Qwen3-4B | gate_up_proj | 8 | 2560→19456 | hbm-bound | 35.8 | 31.32 | 87.5% | 3.26× |
| Qwen3-4B | gate_up_proj | 32 | 2560→19456 | hbm-bound | 40.3 | 32.63 | 80.9% | 2.94× |

## §5 严重 under-utilize 清单（`large-gap` TOP-20）

共 **214** 个 shape cuda_eff < 50%，下面列最糟 20 个：

| model | proj | T | shape | bound | cause | cuda_us | cuda_roof | cuda_eff | speedup |
|:---|:---|---:|:---:|:---|:---|---:|---:|---:|---:|
| Qwen3-0.6B | o_proj | 8 | 2048→1024 | hbm-bound | launch-bound | 34.6 | 1.38 | 4.0% | 0.20× |
| Qwen3-0.6B | kv_proj | 8 | 1024→2048 | hbm-bound | launch-bound | 31.0 | 1.37 | 4.4% | 0.27× |
| Qwen3-0.6B | q_proj | 8 | 1024→2048 | hbm-bound | launch-bound | 30.9 | 1.37 | 4.4% | 0.26× |
| Qwen3-0.6B | o_proj | 32 | 2048→1024 | hbm-bound | launch-bound | 34.9 | 1.61 | 4.6% | 0.19× |
| Qwen3-0.6B | kv_proj | 32 | 1024→2048 | hbm-bound | launch-bound | 30.9 | 1.57 | 5.1% | 0.26× |
| Qwen3-0.6B | q_proj | 32 | 1024→2048 | hbm-bound | launch-bound | 30.6 | 1.57 | 5.1% | 0.27× |
| Qwen3-0.6B | down_proj | 8 | 3072→1024 | hbm-bound | launch-bound | 34.9 | 2.06 | 5.9% | 0.28× |
| Qwen3-0.6B | down_proj | 32 | 3072→1024 | hbm-bound | launch-bound | 35.1 | 2.37 | 6.8% | 0.26× |
| Qwen3-0.6B | o_proj | 128 | 2048→1024 | quant+hbm | other | 34.9 | 2.53 | 7.3% | 0.22× |
| Qwen3-0.6B | kv_proj | 128 | 1024→2048 | hbm-bound | mem-access-suboptimal | 30.9 | 2.38 | 7.7% | 0.22× |
| Qwen3-1.7B | q_proj | 8 | 2048→2048 | hbm-bound | launch-bound | 35.1 | 2.70 | 7.7% | 0.33× |
| Qwen3-1.7B | kv_proj | 8 | 2048→2048 | hbm-bound | launch-bound | 35.0 | 2.70 | 7.7% | 0.33× |
| Qwen3-0.6B | q_proj | 128 | 1024→2048 | hbm-bound | mem-access-suboptimal | 30.7 | 2.38 | 7.7% | 0.24× |
| Qwen3-1.7B | o_proj | 8 | 2048→2048 | hbm-bound | launch-bound | 34.8 | 2.70 | 7.7% | 0.33× |
| Qwen3-1.7B | o_proj | 32 | 2048→2048 | hbm-bound | launch-bound | 35.3 | 2.99 | 8.5% | 0.33× |
| Qwen3-1.7B | q_proj | 32 | 2048→2048 | hbm-bound | launch-bound | 35.0 | 2.99 | 8.5% | 0.33× |
| Qwen3-1.7B | kv_proj | 32 | 2048→2048 | hbm-bound | launch-bound | 34.8 | 2.99 | 8.6% | 0.33× |
| Qwen3-4B | kv_proj | 8 | 2560→2048 | hbm-bound | launch-bound | 34.9 | 3.36 | 9.6% | 0.43× |
| Qwen3-0.6B | down_proj | 128 | 3072→1024 | quant+hbm | other | 35.0 | 3.65 | 10.4% | 0.31× |
| Qwen3-4B | kv_proj | 32 | 2560→2048 | hbm-bound | launch-bound | 35.1 | 3.69 | 10.5% | 0.43× |

## §6 交叉表：bound × T（达 roofline 的命中率）

> 格子里写 `at-roof / total`（`at-roof` 数 / 该 bound+T 组合的总 shape 数）

| bound \ T | 1 | 8 | 32 | 128 | 512 | 1024 | 2048 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| fused-mem | 0/35 | — | — | — | — | — | — |
| hbm-bound | — | 2/35 | 2/35 | 0/33 | 0/3 | — | — |
| quant+hbm | — | — | — | 0/2 | — | — | — |
| quant+tc | — | — | — | — | 0/2 | 0/2 | 0/2 |
| tc-bound | — | — | — | — | 0/30 | 0/33 | 0/33 |

## §7 结论

1. **已达瓶颈（at-roof）**：4/245 (1.6%)，全部是 `gate_up_proj` 的大 shape（d_out 19k/24k 级）在 T=8/32 的 hbm-bound 路径，speedup 中位 3.38×，继续优化 ROI 极低。
2. **中等缺口（mid-gap）**：27/245 (11.0%)，一半是大模型 T=1 的 fused-mem GEMV（gemv-tail），另一半是 T=512/1024/2048 的 tc-bound shape——后者根因是 epilogue per-group dequant FMA 在 CUDA-core 上执行，INT4 TC 峰值无法达到。
3. **严重缺口（large-gap）**：214/245 (87.3%)，分两类：(i) T≤32 小 shape 的 `launch-bound`/`mem-access-suboptimal`（~100 shape，cuda_us 被 ~34us dispatcher 地板托住，roof 仅 1-10us）；(ii) 大 T tc-bound 的 `epilogue-fma`（~100 shape，eff 30-45%，roof 计入了 660 TOPS INT4 TC 峰值但实际达不到）。

**瓶颈优先级**：
- (A) 小 T launch 开销：把 30us 地板拍下来，能救约 80+ shape（收益最大）
- (B) 大 T epilogue FMA：CUTLASS int4 epilogue / dequant-in-register 能救 mid-gap 的 tc-bound 段
- (C) 大 shape gate_up_proj 已到瓶颈，无需再动