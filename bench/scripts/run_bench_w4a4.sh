#!/usr/bin/env bash
# r79 W4A4 fused-op kernel bench driver.
#
# Usage (from repo root, on the GPU box):
#   bash kernel/bench/scripts/run_bench_w4a4.sh [LABEL]
#
# LABEL (optional) is appended to the output dir, so multiple runs can
# coexist:  bench/logs/w4a4_fused_ops_${LABEL}/
# Default LABEL: timestamp.

set -euo pipefail

LABEL="${1:-$(date +%Y%m%d_%H%M%S)}"
shift || true  # drop the label so "$@" only carries python-side args
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="${REPO_ROOT}/kernel/bench/logs/w4a4_fused_ops_${LABEL}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

# Record provenance (best-effort; not fatal if not a git checkout).
COMMIT="$(cd "${REPO_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo '?')"

# Ensure the r79 production dispatch gate: P0.2 disabled, legacy_mma
# path taken — matches production default (see ops.py _p0_fused_quant_supported).
export HKUST_V9_P0_MODE="${HKUST_V9_P0_MODE:-0}"

# Lock GPU clocks if the user has permission (otherwise silently continue;
# min-of-outer + median-of-trials already absorbs boost jitter per
# bench/layer/timing.py contract).
nvidia-smi --lock-gpu-clocks=base >/dev/null 2>&1 || true

{
    echo "=== W4A4 fused-op bench ==="
    echo "label:       ${LABEL}"
    echo "commit:      ${COMMIT}"
    echo "out-dir:     ${OUT_DIR}"
    echo "HKUST_V9_P0_MODE=${HKUST_V9_P0_MODE}"
    echo "host:        $(hostname)"
    echo "date:        $(date -Iseconds)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
    echo "==========================="

    cd "${REPO_ROOT}"
    python -u kernel/bench/scripts/bench_w4a4_fused_ops.py \
        --out-dir "${OUT_DIR}" \
        --commit "${COMMIT}" \
        "$@"

    echo "=== bench done: $(date -Iseconds) ==="
} 2>&1 | tee "${LOG_FILE}"

nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
echo "[driver] outputs under ${OUT_DIR}"
