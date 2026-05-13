# 服务器运行手册（AutoDL）

> 目的：在 AutoDL GPU 服务器上编译并运行 `Zip_kernel` 项目的 triton kernel 测试与性能基准。

## 0. 前置说明

- 服务器别名：`autodl`（可直接 `ssh autodl`）
- 项目路径：`/root/Zip_kernel`
- 软链接：`/root/kernel -> /root/Zip_kernel`（将外部文件夹名与 Python import 路径解耦；脚本统一 `from kernel.triton import ...`）
- GPU：NVIDIA RTX 4090（Ada，SM 8.9）
- Conda：`/root/miniconda3`，项目环境名 **`zip`**（Python 3.10）

## 1. 环境栈（一次性已安装）

| 组件 | 版本 |
|---|---|
| Python | 3.10.20 |
| torch | 2.8.0+cu126 |
| triton | 3.4.0（torch 自带） |
| torchvision | 0.23.0+cu126 |
| numpy | 2.2.6 |
| pytest | 9.0.3 |

若需重建：见文末"环境重建步骤"。

## 2. 跑 bench / pytest 前置（每次新开 shell 必做）

```bash
# 1) 进入服务器
ssh autodl

# 2) 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zip

# 3) 进入项目工作根目录（重要：以 /root 为根，才能 import kernel.triton）
cd /root

# 4) 快速自检
python -c "import torch, triton; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'triton', triton.__version__)"
```

> **非交互式 SSH 一行版**（适合脚本化执行）：
> ```bash
> ssh autodl 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate zip && cd /root && <your command>'
> ```

## 3. 运行 pytest 测试

```bash
# 运行所有 triton kernel 单元测试
cd /root && pytest Zip_kernel/triton/tests/ -v

# 单独跑某一个
cd /root && pytest Zip_kernel/triton/tests/test_end2end.py -v
```

## 4. 运行 benchmark（性能基准）

三个 bench 脚本均可直接执行，内部已用 `__file__` 动态解析 import 路径，无需预设 `PYTHONPATH`：

```bash
# Kernel(1) dense GEMM vs cuBLAS FP16
python /root/Zip_kernel/triton/benchmarks/bench_dense.py

# Kernel(2) sparse s4×s4 vs cuBLAS FP16
python /root/Zip_kernel/triton/benchmarks/bench_sparse.py

# 端到端 V9 linear
python /root/Zip_kernel/triton/benchmarks/bench_linear.py
```

## 5. 日常调试小贴士

- 清除 pyc 缓存（换了 Python 版本后必须清）：
  ```bash
  find /root/Zip_kernel -type d -name __pycache__ -exec rm -rf {} +
  ```
- 查看 GPU：`nvidia-smi`
- 长任务后台执行 + 实时看日志：
  ```bash
  nohup python xxx.py > /tmp/run.log 2>&1 &
  tail -f /tmp/run.log
  ```

## 6. 环境重建步骤（仅在服务器初始化后需要）

```bash
# a. 装 Miniconda
curl -fsSL -o /tmp/Miniconda3.sh \
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_25.5.1-1-Linux-x86_64.sh
bash /tmp/Miniconda3.sh -b -p /root/miniconda3

# b. 创建 zip 环境
/root/miniconda3/bin/conda create -n zip python=3.10 -y

# c. 装 torch 2.8.0 + cu126（走 PyTorch r2 CDN，速度远快于官方 CDN）
/root/miniconda3/envs/zip/bin/pip install \
  torch==2.8.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126

# d. 装其他依赖
/root/miniconda3/envs/zip/bin/pip install pytest numpy

# e. 拉仓库
git clone git@github.com:HuYangjia/Zip_kernel.git /root/Zip_kernel
ln -s /root/Zip_kernel /root/kernel
```

---

## 附录：AutoDL 学术加速代理（可选）

> 声明：限于学术使用 github 和 huggingface 网络速度慢的问题，以下为方便用户学术用途使用相关资源提供的加速代理，不承诺稳定性保证。此外如遭遇恶意攻击等，将随时停止该加速服务。

- 开启：
  ```bash
  source /etc/network_turbo
  ```
- 关闭：
  ```bash
  unset http_proxy && unset https_proxy
  ```
