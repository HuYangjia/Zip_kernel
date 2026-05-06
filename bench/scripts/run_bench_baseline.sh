#!/usr/bin/env bash
# run_bench_baseline.sh — autodl 上一键跑 Atom W4A4 baseline bench
#
# 假设：
#   * conda env "atom_baseline" 已按 BASELINE_SETUP.md §1-§4 编译完成，
#     即 `python -c "import punica.ops"` 成功
#   * 你已经先跑过 BF16 bench (run_bench_bf16.sh strict)，
#     输出在 kernel/bench/logs/bf16_<TS>/bench_bf16_per_op.json
#
# 用法：
#   # 1) 冒烟（~3-5 分钟）
#   bash kernel/bench/scripts/run_bench_baseline.sh smoke
#
#   # 2) strict（~30 分钟）
#   bash kernel/bench/scripts/run_bench_baseline.sh strict
#
#   # 3) 给定具体的 bf16 json，直接合成（跳过 atom 实测）
#   bash kernel/bench/scripts/run_bench_baseline.sh compose \
#        kernel/bench/logs/bf16_xxx/bench_bf16_per_op.json \
#        kernel/bench/logs/atom_yyy/atom_w4a4_gemm.json

set -euo pipefail

MODE="${1:-smoke}"; shift || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

TS="$(date +%Y%m%d_%H%M%S)"

# Detect which conda env we're in.
if [[ "${CONDA_DEFAULT_ENV:-}" != "atom_baseline" && "$MODE" != "compose" ]]; then
  echo "[fatal] this script needs the 'atom_baseline' conda env activated."
  echo "        run: conda activate atom_baseline"
  exit 2
fi

case "$MODE" in
  smoke)
    OUT="kernel/bench/logs/atom_smoke_${TS}"
    mkdir -p "$OUT"
    echo "[bench] mode=smoke timing=light out=$OUT"
    python -m kernel.bench.scripts.bench_baseline_w4a4_gemm \
      --models Qwen3-8B \
      --phases prefill decode \
      --batches 8 32 \
      --timing light \
      --out-dir "$OUT" \
      "$@" 2>&1 | tee "$OUT/run.log"
    ;;

  strict)
    OUT="kernel/bench/logs/atom_${TS}"
    mkdir -p "$OUT"
    echo "[bench] mode=strict timing=strict out=$OUT"
    # Lock SM clock if possible (mirrors run_bench_bf16.sh).
    ( nvidia-smi -lgc 1395 >/dev/null 2>&1 && echo "[bench] locked SM clock to 1395 MHz" ) \
      || echo "[bench] (skip) cannot lock SM clock"

    {
      echo "# Atom baseline run $TS"
      echo "- mode: strict"
      echo "- env: atom_baseline"
      echo "- git: $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
      echo "- host: $(hostname)"
      echo "- nvidia-smi:"
      nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
    } > "$OUT/RUN_INFO.md"

    python -m kernel.bench.scripts.bench_baseline_w4a4_gemm \
      --timing strict \
      --out-dir "$OUT" \
      "$@" 2>&1 | tee "$OUT/run.log"
    ;;

  compose)
    BF16_JSON="${1:-}"
    ATOM_JSON="${2:-}"
    if [[ -z "$BF16_JSON" || -z "$ATOM_JSON" ]]; then
      echo "[fatal] usage: $0 compose <bf16.json> <atom.json>" >&2
      exit 2
    fi
    if [[ ! -f "$BF16_JSON" ]]; then
      echo "[fatal] not found: $BF16_JSON" >&2; exit 2
    fi
    if [[ ! -f "$ATOM_JSON" ]]; then
      echo "[fatal] not found: $ATOM_JSON" >&2; exit 2
    fi
    OUT="$(dirname "$ATOM_JSON")"
    echo "[compose] bf16=$BF16_JSON atom=$ATOM_JSON out=$OUT"
    python -m kernel.bench.scripts.compose_baseline_layer \
      --bf16-json  "$BF16_JSON" \
      --atom-json  "$ATOM_JSON" \
      --out-dir    "$OUT"
    ;;

  *)
    echo "[fatal] unknown mode '$MODE' — use 'smoke' / 'strict' / 'compose'" >&2
    exit 2
    ;;
esac

echo
echo "[bench] done. artifacts in: $OUT"
ls -la "$OUT"
