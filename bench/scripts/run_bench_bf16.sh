#!/usr/bin/env bash
# run_bench_bf16.sh — autodl 上一键跑 BF16 per-op bench
#
# 用法（在 autodl 仓库根目录下）：
#   # 1) 先冒烟 (~1-2 分钟)
#   bash kernel/bench/scripts/run_bench_bf16.sh smoke
#
#   # 2) 确认 VALIDATION_LOG 全绿后，跑正式 strict (~15-25 分钟)
#   bash kernel/bench/scripts/run_bench_bf16.sh strict
#
#   # 可选：传 subset，例如只跑 8B / decode / B=32
#   bash kernel/bench/scripts/run_bench_bf16.sh strict \
#        --models Qwen3-8B --phases decode --batches 32

set -euo pipefail

MODE="${1:-smoke}"; shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
case "$MODE" in
  smoke)
    TIMING="light"
    OUT="kernel/bench/logs/smoke_${TS}"
    ;;
  strict)
    TIMING="strict"
    OUT="kernel/bench/logs/bf16_${TS}"
    ;;
  *)
    echo "[fatal] unknown mode '$MODE' — use 'smoke' or 'strict'" >&2
    exit 2
    ;;
esac

echo "[bench] mode=$MODE timing=$TIMING out=$OUT"
echo "[bench] repo_root=$REPO_ROOT"
mkdir -p "$OUT"

# 锁频（可选；autodl 大多无权限，失败忽略）
( nvidia-smi -lgc 1395 >/dev/null 2>&1 && echo "[bench] locked SM clock to 1395 MHz" ) \
  || echo "[bench] (skip) cannot lock SM clock — may add slight noise"

# 记录 git 状态到 out-dir，便于追溯
{
  echo "# Bench run $TS"
  echo "- mode: $MODE"
  echo "- timing: $TIMING"
  echo "- git: $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
  echo "- host: $(hostname)"
  echo "- nvidia-smi:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
} > "$OUT/RUN_INFO.md"

python -m kernel.bench.scripts.bench_layer_per_op_bf16 \
  --timing "$TIMING" \
  --out-dir "$OUT" \
  "$@" 2>&1 | tee "$OUT/run.log"

echo
echo "[bench] done. artifacts:"
ls -la "$OUT"
echo
echo "[bench] quick peek at VALIDATION_LOG:"
head -n 40 "$OUT/VALIDATION_LOG.md" || true
