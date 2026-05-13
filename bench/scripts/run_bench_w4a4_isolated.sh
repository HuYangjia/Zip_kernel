#!/usr/bin/env bash
# r79 W4A4 fused-op ISOLATED bench driver.
#
# Same output shape as run_bench_w4a4.sh, but every (model, phase, bs)
# triple runs in a fresh Python subprocess.  This eliminates the
# cross-triple CUDA-workspace / allocator state that caused 14B
# prefill bs=16 gate_up_fused to crash in the single-process sweep.
#
# Usage (from repo root, on the GPU box):
#   bash kernel/bench/scripts/run_bench_w4a4_isolated.sh [LABEL] [-- extra args]
#
# Extra args are forwarded verbatim to bench_w4a4_isolated_driver.py
# (e.g. --resume, --models Qwen3-14B, --phases prefill, etc.).
#
# Default LABEL: timestamp.
# Output: kernel/bench/logs/w4a4_fused_ops_isolated_${LABEL}/
#     ├── driver.log                        (this wrapper + driver)
#     ├── run.log                           (tee of this wrapper)
#     ├── bench_w4a4_per_op.json            (aggregated from all triples)
#     ├── bench_w4a4_summary.md
#     ├── VALIDATION_LOG.md
#     └── per_triple/<model>__<phase>__bs<N>/
#             ├── bench_w4a4_per_op.json    (single triple = 4 rows)
#             └── run.log                   (that subprocess's tee)

set -euo pipefail

LABEL="${1:-$(date +%Y%m%d_%H%M%S)}"
shift || true  # drop the label so "$@" only carries driver-side args

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="${REPO_ROOT}/kernel/bench/logs/w4a4_fused_ops_isolated_${LABEL}"
LOG_FILE="${OUT_DIR}/run.log"

mkdir -p "${OUT_DIR}"

# Provenance (best-effort; not fatal if not a git checkout).
COMMIT="$(cd "${REPO_ROOT}" && git rev-parse --short HEAD 2>/dev/null || echo '?')"

# Match the single-process driver: P0.2 off, legacy_mma path.
export HKUST_V9_P0_MODE="${HKUST_V9_P0_MODE:-0}"

# Lock GPU clocks if permission allows; harmless otherwise.
nvidia-smi --lock-gpu-clocks=base >/dev/null 2>&1 || true

{
    echo "=== W4A4 fused-op bench — ISOLATED driver ==="
    echo "label:       ${LABEL}"
    echo "commit:      ${COMMIT}"
    echo "out-dir:     ${OUT_DIR}"
    echo "HKUST_V9_P0_MODE=${HKUST_V9_P0_MODE}"
    echo "host:        $(hostname)"
    echo "date:        $(date -Iseconds)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
    echo "=============================================="

    cd "${REPO_ROOT}"
    python -u kernel/bench/scripts/bench_w4a4_isolated_driver.py \
        --out-dir "${OUT_DIR}" \
        --commit "${COMMIT}" \
        "$@"

    echo "=== isolated bench done: $(date -Iseconds) ==="
} 2>&1 | tee "${LOG_FILE}"

nvidia-smi --reset-gpu-clocks >/dev/null 2>&1 || true
echo "[driver] outputs under ${OUT_DIR}"
