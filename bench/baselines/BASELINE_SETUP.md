# Atom W4A4 baseline — autodl 服务器编译手册

> 目的：在 autodl RTX 4090 上把 `other_baseline/atom/e2e/punica-atom` 的 CUDA kernel
> 编译成可以 `import punica.ops` 的 Python 包，供
> `kernel/bench/baselines/atom_punica.py` 调用。
>
> **重要**：这个环境是**完全独立**的 conda env（不要污染 `zip` env）。
> 编译完后 `bench_baseline_w4a4_gemm.py` 在这个 env 里跑，bf16 bench 仍在 `zip` env。

---

## 0. 前置 — 仓库准备

确认 `autodl:/root/Zip_kernel/other_baseline/atom/` 已经存在（如果只 push 了部分目录，
需要先 `git clone --depth 1 https://github.com/efeslab/Atom.git` 到本地后 rsync）。

```bash
# 本地（如已 clone 过则跳过）
cd /Users/yangjiahu/Desktop/workspace/HKUST/other_baseline
ls atom/e2e/punica-atom/setup.py    # 应该存在

# 推到服务器
rsync -azv atom/ autodl:/root/Zip_kernel/other_baseline/atom/
```

---

## 1. 创建独立 conda env

```bash
ssh autodl
source /root/miniconda3/etc/profile.d/conda.sh

conda create -n atom_baseline python=3.10 -y
conda activate atom_baseline
```

---

## 2. 安装 torch + cuda toolchain

> ⚠️ **关键决策**：upstream 要求 `torch==2.0.1 + cu11`，但 autodl 的驱动是
> CUDA 12.6（由 4090 驱动决定，不可降）。我们用 **`torch==2.1.2 + cu118`**
> （pip 自带 cu118 runtime 就够，不依赖系统 cuda），实测 punica-atom 的
> kernel 用 `torch.utils.cpp_extension` 编译能通过。

```bash
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望: 2.1.2+cu118  11.8  True
```

如果 autodl 的内网下载慢：
```bash
source /etc/network_turbo
pip install ... # 下完后:
unset http_proxy && unset https_proxy
```

---

## 3. 安装编译时依赖

```bash
# ninja 让 torch_cpp_ext 编译并行
pip install ninja packaging

# punica-atom 运行时依赖（裁剪后的最小集合）
pip install \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    safetensors \
    sentencepiece \
    numpy \
    'cmake>=3.24'

# 我们 bench 自己用的
pip install pytest
```

如果 nvcc 缺失（`which nvcc` 无输出），追加：
```bash
conda install -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-libraries-dev=11.8 -y
```

---

## 4. 编译 punica-atom 的 C++ extension

```bash
cd /root/Zip_kernel/other_baseline/atom/e2e/punica-atom

# 关键：指定 SM 8.6（4090 是 SM 8.9 但 8.6 PTX 兼容运行；
# 如果想原生 8.9 则改成 "8.9"——但 punica-atom 的部分 ldmatrix 内联汇编
# 是按 8.6 写的，建议先用 8.6 试）。
env TORCH_CUDA_ARCH_LIST="8.6" \
    MAX_JOBS=4 \
    pip install --no-build-isolation -e .
```

预期编译时间 ~10-20 分钟，会在 stderr 看到大量 nvcc 输出。

### 4.1 常见编译错误及处置

| 报错 | 原因 | 处置 |
|---|---|---|
| `error: identifier "__hadd2" is undefined` | torch 2.1 + cuda 11.8 移除了部分 fp16 macros | 在 `setup.py::remove_unwanted_pytorch_nvcc_flags` 后追加 `torch_cpp_ext.COMMON_NVCC_FLAGS.append('-D__CUDA_NO_HALF2_OPERATORS__=0')` |
| `THCDeviceUtils.cuh: No such file` | upstream 引用了被 torch 2.0 移除的旧头 | 在 `csrc/` 下 `grep -rn THC` 把所有出现替换为 `c10/cuda/CUDAGuard.h` 或注释掉对应行 |
| `'flashinfer_decode_kvcache_int4' undeclared` | flashinfer 子模块没拉下来 | `cd punica/ops/csrc/flashinfer_adapter && ls flashinfer/`，若空则 `git submodule update --init --recursive` 或手动从 flashinfer master 取相同版本 |
| `nvcc fatal: Unsupported gpu architecture 'compute_86'` | nvcc < 11.1 | 检查 `nvcc --version`；用 step 3 的 conda 通道安装 11.8 |

如果实在卡住，把完整 stderr 贴给 AI 助手或保留到 `BASELINE_BUILD_LOG.md` 里以便后续 debug。

### 4.2 验证编译成功

```bash
python - <<'PY'
import torch
import punica.ops
print("punica.ops attrs:", [x for x in dir(punica.ops) if not x.startswith('_')])

# 试一次最小 GEMM 调用
from punica.ops import dense_layer_gemm_i4_fp16, scale_size
M, N, K = 128, 4096, 4096
GROUP = 128; KEEPER = 128
dev = "cuda"
a_norms = torch.zeros((M, (K-KEEPER)//2), dtype=torch.int8, device=dev)
a_keep  = torch.zeros((M, KEEPER),         dtype=torch.int8, device=dev)
a_ns    = torch.zeros((K//GROUP-1, scale_size(M)), dtype=torch.float16, device=dev)
a_ks    = torch.zeros((scale_size(M),),            dtype=torch.float16, device=dev)
w4      = torch.zeros((N, (K-KEEPER)//2), dtype=torch.uint8, device=dev)
w8      = torch.zeros((N, KEEPER),         dtype=torch.int8, device=dev)
w_ns    = torch.zeros((K//GROUP-1, scale_size(N)), dtype=torch.float16, device=dev)
w_ks    = torch.zeros((scale_size(N),),            dtype=torch.float16, device=dev)
y = dense_layer_gemm_i4_fp16(a_norms, w4, a_ns, w_ns, a_keep, w8, a_ks, w_ks)
print("OK shape", y.shape, "dtype", y.dtype)
PY
```

期望输出：`OK shape torch.Size([128, 4096]) dtype torch.float16`。

---

## 5. 跑 baseline bench

回到本项目根目录：
```bash
cd /root/Zip_kernel
# 注意此时 conda env 仍然是 atom_baseline；运行 baseline bench 用这个 env

# 5.1 smoke (light 协议，~3 分钟)
python -m kernel.bench.scripts.bench_baseline_w4a4_gemm \
    --models Qwen3-8B \
    --phases prefill decode \
    --batches 8 32 \
    --timing light \
    --out-dir kernel/bench/logs/atom_smoke_$(date +%Y%m%d_%H%M)

# 5.2 strict（~30 分钟，全部 24 个点）
python -m kernel.bench.scripts.bench_baseline_w4a4_gemm \
    --timing strict \
    --out-dir kernel/bench/logs/atom_$(date +%Y%m%d_%H%M)
```

**成功标志**：`atom_w4a4_gemm.json` 中所有 prefill 点 `status == "ok"`；
decode B=4/8 可能因 M%16 限制而 status="partial"——这是预期的，可以接受。

---

## 6. 合成 baseline 整层

合成步骤**不需要 punica.ops**，可以在 `zip` env 跑：

```bash
conda activate zip   # 切回 BF16 bench 同环境

python -m kernel.bench.scripts.compose_baseline_layer \
    --bf16-json  kernel/bench/logs/bf16_<TS>/bench_bf16_per_op.json \
    --atom-json  kernel/bench/logs/atom_<TS>/atom_w4a4_gemm.json \
    --out-dir    kernel/bench/logs/atom_<TS>
```

输出：
* `baseline_layer.json` — composed timings，每条记录都有 `baseline_layer_us`、`speedup_vs_bf16`
* `baseline_layer_summary.md` — 一眼可读的 markdown 表

---

## 7. 三方对比

```bash
python -m kernel.bench.scripts.compare_vs_ours \
    --bf16-json     kernel/bench/logs/bf16_<TS>/bench_bf16_per_op.json \
    --baseline-json kernel/bench/logs/atom_<TS>/baseline_layer.json \
    --ours-json     kernel/bench/logs/r79_<TS>/ours_layer.json \
    --out-dir       kernel/bench/logs/comparison_$(date +%Y%m%d_%H%M)
```

最终生成 `comparison_table.md`，三列 speedup（atom/bf16, ours/bf16, ours/atom）+ Amdahl ceiling + 自动 verdict。

---

## 附录 A：完全失败时的降级方案

如果 punica-atom 在 4090 + cu118 这套上始终编不过，可降级为：

1. 用 `bench_baseline_w4a4_gemm.py --gemm-only` —— 跳过 quant 部分，
   只测 4 个 INT4 GEMM 本身。这相当于一个**乐观上界 baseline**，
   论文里要明确说明 "Atom-baseline_GEMM-only assumption" 来限制结论强度。
2. 退到 ResQ paper 表格里的 reported per-shape 数（手抠）填到一个 `atom_w4a4_gemm.json`-shape 的手写 JSON。compose_baseline_layer 不关心数据来源是实测还是手填。

---

## 附录 B：清理

```bash
conda deactivate
# 不要删 env！下次 bench 还要用：
conda env list   # 应该看到 atom_baseline
```

---

## 附录 C：实际测试这个手册前的 sanity 检查

在动手编译前，先确认下面这条命令在 autodl 上跑通：

```bash
ssh autodl 'source /root/miniconda3/etc/profile.d/conda.sh && \
            ls -la /root/Zip_kernel/other_baseline/atom/e2e/punica-atom/setup.py && \
            ls /root/Zip_kernel/other_baseline/atom/e2e/punica-atom/punica/ops/csrc/'
```

如果路径不对，先用 `rsync` 把仓库推上去。
