#!/usr/bin/env bash
# Validate the V9 CUDA kernel suite on an SM89 host (e.g. RTX 4090).
#
# Responsibilities (idempotent, safe to re-run):
#   1. Probe the GPU and refuse to run on non-SM89 cards.
#   2. Sanity-check that nvcc is on PATH and matches the torch build.
#   3. Prime the JIT build cache (first run compiles ~30 s; subsequent
#      runs reuse ~/.cache/hkust_v9_cuda).
#   4. Run the pytest parity suite with strict failure reporting.
#   5. Print a backend-status summary so the operator can confirm the
#      CUDA backend is actually in use.
#
# Conventions (per project preferences):
#   - All paths resolved relative to this script via BASH_SOURCE.
#   - All log output in English.
#   - Logs tee'd to a timestamped file under logs/.

set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd -- "${SCRIPT_DIR}/../../.." &> /dev/null && pwd )"

LOG_DIR="${REPO_ROOT}/logs/cuda_kernel_validate"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$( date +%Y%m%d_%H%M%S )"
LOG_FILE="${LOG_DIR}/validate_${TIMESTAMP}.log"

log() {
    echo "[$( date '+%Y-%m-%d %H:%M:%S' )] $*" | tee -a "${LOG_FILE}"
}

log "Repo root  : ${REPO_ROOT}"
log "Script dir : ${SCRIPT_DIR}"
log "Log file   : ${LOG_FILE}"

# ---------------------------------------------------------------------------
# Step 1: probe the GPU
# ---------------------------------------------------------------------------

log "--- Step 1/5: GPU probe ---"
if ! command -v nvidia-smi &>/dev/null; then
    log "ERROR: nvidia-smi not found; cannot validate on CPU-only host"
    exit 1
fi

GPU_INFO="$( nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 1 )"
log "Detected GPU: ${GPU_INFO}"

COMPUTE_CAP="$( echo "${GPU_INFO}" | awk -F', ' '{print $2}' )"
if [[ "${COMPUTE_CAP}" != "8.9" ]]; then
    log "WARNING: detected compute capability ${COMPUTE_CAP} (expected 8.9 for RTX 4090)."
    log "         The kernels will attempt to build but may not execute."
    log "         Set HKUST_V9_DISABLE_CUDA=1 to force the Triton-only path."
fi

# ---------------------------------------------------------------------------
# Step 2: toolchain check
# ---------------------------------------------------------------------------

log "--- Step 2/5: toolchain check ---"
if ! command -v nvcc &>/dev/null; then
    log "ERROR: nvcc not on PATH; required for torch.utils.cpp_extension.load"
    exit 1
fi
NVCC_VER="$( nvcc --version | tail -n 1 )"
log "nvcc       : ${NVCC_VER}"

PY_TORCH_INFO="$( python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1 || true )"
log "torch      : ${PY_TORCH_INFO}"

# ---------------------------------------------------------------------------
# Step 3: prime JIT build cache
# ---------------------------------------------------------------------------

log "--- Step 3/5: prime JIT build cache ---"
export HKUST_V9_CUDA_VERBOSE=1
pushd "${REPO_ROOT}" >/dev/null
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import logging, sys, time
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("prime_jit")
t0 = time.time()
try:
    from kernel.cuda_kernel import ops  # triggers build
    log.info("CUDA extension loaded OK in %.1f s", time.time() - t0)
    for name in ("activation_quant", "dense_gemm", "sparse_gemm",
                 "fused_dense_sparse"):
        fn = getattr(ops, f"{name}_cuda", None)
        log.info("  %-22s : %s", name, "OK" if callable(fn) else "MISSING")
except Exception as e:
    log.error("CUDA extension failed to build: %s", e, exc_info=True)
    sys.exit(2)
PY
popd >/dev/null

# ---------------------------------------------------------------------------
# Step 4: parity tests
# ---------------------------------------------------------------------------

log "--- Step 4/5: pytest parity suite ---"
pushd "${REPO_ROOT}" >/dev/null
# -x: stop at first failure so the log is focused on the first regression.
# -rA: show short summary for all outcomes.
# --tb=short: concise traceback format.
python -m pytest kernel/cuda_kernel/tests/test_parity.py \
    -x -rA --tb=short --color=no 2>&1 | tee -a "${LOG_FILE}"
PYTEST_EXIT=${PIPESTATUS[0]}
popd >/dev/null

if [[ ${PYTEST_EXIT} -ne 0 ]]; then
    log "ERROR: pytest failed with exit code ${PYTEST_EXIT}; see ${LOG_FILE}"
    exit ${PYTEST_EXIT}
fi

# ---------------------------------------------------------------------------
# Step 5: backend status report
# ---------------------------------------------------------------------------

log "--- Step 5/5: backend status ---"
pushd "${REPO_ROOT}" >/dev/null
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("status")
from kernel.backend import get_backend_status, set_backend_policy
log.info("auto policy status  : %s", json.dumps(get_backend_status(), indent=2, default=str))
set_backend_policy("cuda")
log.info("cuda-forced status  : %s", json.dumps(get_backend_status(), indent=2, default=str))
set_backend_policy("auto")
PY
popd >/dev/null

log "=== validate_cuda.sh completed successfully ==="
