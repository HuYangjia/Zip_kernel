# r63 → r78 regression & improvement report

**r63_combined**: 140 shapes  
**r78_final_roofline**: 245 shapes  
**Common overlap**: 140 shapes  

## §1 FP16 baseline consistency

如果 cuBLAS 基线偏离 >5%，说明 GPU 频率 / 上下文变化，对比不可信。

| metric | median | mean | min | max |
|:---|---:|---:|---:|---:|
| fp16_us relative change | +0.4% | +0.8% | -5.2% | +9.1% |
| cuda_us relative change | +1.2% | +0.6% | -25.6% | +33.7% |
| **speedup Δ (r78 - r63)** | **-0.01×** | +0.01× | -0.25× | +0.50× |

✅ FP16 baseline drift (+0.4%) within ±5% — kernel comparison is reliable.

## §2 per-T improvement distribution

Positive Δ = r78 faster than r63 on that shape.

| T | N | median Δ | mean Δ | % improved |
|---:|---:|---:|---:|---:|
| 1 | 35 | -0.03× | -0.03× | 9% |
| 32 | 35 | -0.01× | +0.03× | 31% |
| 128 | 35 | -0.01× | +0.01× | 37% |
| 512 | 35 | +0.01× | +0.03× | 63% |

## §3 per-proj improvement distribution

| proj | N | median Δ | mean Δ | % improved |
|:---|---:|---:|---:|---:|
| down_proj | 28 | -0.01× | -0.00× | 32% |
| gate_up_proj | 28 | +0.00× | +0.06× | 50% |
| kv_proj | 28 | -0.01× | -0.01× | 29% |
| o_proj | 28 | -0.00× | +0.00× | 36% |
| q_proj | 28 | -0.01× | -0.01× | 29% |

## §4 per-model improvement distribution

| model | N | median Δ | mean Δ | % improved |
|:---|---:|---:|---:|---:|
| Qwen3-0.6B | 20 | -0.00× | -0.01× | 20% |
| Qwen3-1.7B | 20 | -0.01× | -0.01× | 20% |
| Qwen3-4B | 20 | -0.01× | +0.01× | 25% |
| Qwen3-8B | 20 | -0.02× | -0.00× | 30% |
| Qwen3-14B | 20 | +0.04× | +0.05× | 55% |
| Qwen2.5-32B | 20 | +0.02× | +0.03× | 60% |
| LLaMA3-70B | 20 | -0.01× | -0.00× | 35% |

## §5 TOP-10 shapes with biggest speedup improvement

| rank | model | proj | T | shape | r63 speedup | r78 speedup | Δ |
|---:|:---|:---|---:|:---:|---:|---:|---:|
| 1 | Qwen3-14B | gate_up_proj | 32 | 5120→34816 | 1.45× | 1.95× | **+0.50×** |
| 2 | Qwen3-4B | gate_up_proj | 32 | 2560→19456 | 2.52× | 2.94× | **+0.42×** |
| 3 | Qwen3-8B | gate_up_proj | 32 | 4096→24576 | 3.25× | 3.50× | **+0.25×** |
| 4 | Qwen3-14B | gate_up_proj | 512 | 5120→34816 | 0.77× | 1.01× | **+0.24×** |
| 5 | Qwen3-14B | gate_up_proj | 128 | 5120→34816 | 0.96× | 1.18× | **+0.22×** |
| 6 | Qwen2.5-32B | down_proj | 512 | 27648→5120 | 0.70× | 0.89× | **+0.19×** |
| 7 | Qwen2.5-32B | gate_up_proj | 32 | 5120→55296 | 1.76× | 1.88× | **+0.12×** |
| 8 | Qwen3-14B | down_proj | 512 | 17408→5120 | 0.80× | 0.90× | **+0.10×** |
| 9 | Qwen3-1.7B | down_proj | 128 | 6144→2048 | 0.73× | 0.80× | **+0.08×** |
| 10 | Qwen3-0.6B | o_proj | 1 | 2048→1024 | 0.88× | 0.96× | **+0.07×** |

## §6 TOP-10 shapes with biggest regression

| rank | model | proj | T | shape | r63 speedup | r78 speedup | Δ |
|---:|:---|:---|---:|:---:|---:|---:|---:|
| 1 | Qwen3-1.7B | gate_up_proj | 1 | 2048→12288 | 2.35× | 2.30× | -0.05× |
| 2 | Qwen3-4B | kv_proj | 1 | 2560→2048 | 1.42× | 1.36× | -0.06× |
| 3 | Qwen3-1.7B | kv_proj | 1 | 2048→2048 | 1.54× | 1.48× | -0.06× |
| 4 | Qwen2.5-32B | down_proj | 32 | 27648→5120 | 1.08× | 1.02× | -0.06× |
| 5 | LLaMA3-70B | q_proj | 1 | 8192→8192 | 2.23× | 2.16× | -0.07× |
| 6 | Qwen3-8B | kv_proj | 32 | 4096→2048 | 0.59× | 0.51× | -0.08× |
| 7 | Qwen3-14B | gate_up_proj | 1 | 5120→34816 | 2.07× | 1.98× | -0.09× |
| 8 | Qwen3-4B | down_proj | 512 | 9728→2560 | 0.84× | 0.74× | -0.10× |
| 9 | Qwen3-14B | kv_proj | 128 | 5120→2048 | 0.74× | 0.56× | -0.18× |
| 10 | Qwen3-0.6B | q_proj | 1 | 1024→2048 | 0.97× | 0.72× | -0.25× |
