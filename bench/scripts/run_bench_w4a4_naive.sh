#!/usr/bin/env bash
# r80 W4A4 NAIVE-backend ISOLATED bench driver.
#
# Sibling of run_bench_w4a4_isolated.sh.  Drives the naive 4-kernel
# pipeline (activation_quant_naive → dense_gemm_naive →
# sparse_gemm_naive → reduce_sum_naive) across the full
# {Qwen3-4B,8B,14B} × {prefill,decode} × {4,8,16,32} sweep, with every
# triple in a fresh CUDA subprocess.
#
# Usage (from repo root, on the GPU box):
#   bash kernel/bench/scripts/run_bench_w4a4_naive.sh [LABEL] [-- extra args]
#
# Extra args forwarded verbatim to bench_w4a4_naive_isolated_driver.py
# (e.g. --resume, --models Qwen3-14B, --phases prefill).
#
# Default LABEL: timestamp.
# Output: kernel/bench/logs/w4a4_naive_isolated_${LABEL}/
#     ├── driver.log                              (this wrapper + driver)
#     ├── run.log                                 (tee of this wrapper)
#     ├── bench_w4a4_naive_per_op.json            (aggregated)
#     ├── bench_w4a4_naive_summary.md
#     ├── VALIDATION_LOG.md
#     └── per_triple/<model>__<phase>__bs<N>/
#             ├── bench_w4a4_naive_per_op.json    (single triple = 4 rows)
#             └── run.log

set -euo pipefail

LABEL="${1:-$(date +%Y%m%d_%H%M%S)}"
shift || true

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="${REPO_ROOT}/kernel/bench/logs/w4a4_naive_isolated_${LABEL}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

COMMIT="$(cd "${REPO_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo '?')"

# Naive backend has no dispatch gates; only pin the HBM frequency to
# suppress clock-transient jitter (see memory:bmmiahpl).
nvidia-smi --lock-gpu-clocks=base >/dev/null 2>&1 || true

{
    echo "=== W4A4 NAIVE bench — ISOLATED driver ==="
    echo "label:       ${LABEL}"
    echo "commit:      ${COMMIT}"
    echo "out-dir:     ${OUT_DIR}"
    echo "backend:     naive (csrc_naive)"
    echo "host:        $(hostname)"
    echo "date:        $(date -Iseconds)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
    echo "==========================================="

    cd "${REPO_ROOT}"
    python -u kernel/bench/scripts/bench_w4a4_naive_isolated_driver.py \
        --out-dir "${OUT_DIR}" \
        --commit "${COMMIT}" \
        "$@"

    echo "=== naive isolated bench done: $(date -Iseconds) ==="
} 2>&1 | tee "${LOG_FILE}"

nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
echo "[driver] outputs under ${OUT_DIR}"
