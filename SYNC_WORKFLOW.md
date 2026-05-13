# 多端同步与归档流程

> 适用路径：本机 `/Users/yangjiahu/Desktop/workspace/HKUST/kernel`，AutoDL `/root/Zip_kernel`，GitHub `origin`。

## 1. 原则

- GitHub `origin/main` 作为最终归档源。
- 本机和 AutoDL 都先确认 `git status` 干净或明确知道改动来源，再同步。
- 代码、脚本、最终报告、关键 `md/json/csv/png/pdf` 结果可以入库。
- 可复现实验产物、缓存、profile 二进制和构建中间文件不要入库。

## 2. 本机检查

```bash
cd /Users/yangjiahu/Desktop/workspace/HKUST/kernel
git status --short --branch
git diff --stat
git ls-files --others --exclude-standard
```

## 3. AutoDL 检查

```bash
ssh autodl 'cd /root/Zip_kernel && git status --short --branch && git diff --stat && git ls-files --others --exclude-standard'
```

## 4. 从 AutoDL 拉回本机

推荐先 dry-run，确认不会误覆盖：

```bash
rsync -avzn --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.DS_Store' \
  --exclude 'bench/logs/' \
  --exclude 'triton_kernel/benchmarks/results/' \
  --exclude '*.nsys-rep' \
  --exclude '*.qdrep' \
  --exclude '*.ncu-rep' \
  --exclude '*.sqlite' \
  autodl:/root/Zip_kernel/ \
  /Users/yangjiahu/Desktop/workspace/HKUST/kernel/
```

确认无误后去掉 `-n` 执行：

```bash
rsync -avz --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.DS_Store' \
  --exclude 'bench/logs/' \
  --exclude 'triton_kernel/benchmarks/results/' \
  --exclude '*.nsys-rep' \
  --exclude '*.qdrep' \
  --exclude '*.ncu-rep' \
  --exclude '*.sqlite' \
  autodl:/root/Zip_kernel/ \
  /Users/yangjiahu/Desktop/workspace/HKUST/kernel/
```

## 5. 归档到 GitHub

```bash
cd /Users/yangjiahu/Desktop/workspace/HKUST/kernel
git status --short
git add -A
git status --short
git commit -m "Archive final kernel experiments"
git push origin main
```

## 6. 让 AutoDL 跟随 GitHub

```bash
ssh autodl 'cd /root/Zip_kernel && git pull --ff-only origin main'
```

如果服务器也有本地未提交改动，先在服务器上提交或拉回本机合并，不要直接 `reset --hard`。
