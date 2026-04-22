#!/usr/bin/env bash
# run_nsys_sweep.sh
#
# Profile V9 + cuBLAS FP16 with Nsight Systems across 3 representative
# workloads (decode / mid / prefill).  For each run we:
#   1. launch driver under nsys with --capture-range=cudaProfilerApi
#      so ONLY the measured region lands in the report
#   2. export GPU-kernel, NVTX-range and CUDA-API stats as CSV
#   3. summarise into a single markdown file
#
# All paths are derived from BASH_SOURCE to stay CWD-independent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
DRIVER_PY="${SCRIPT_DIR}/profile_nvtx_driver.py"
mkdir -p "${RESULTS_DIR}"

TS=$(date +%Y%m%d_%H%M%S)
SUMMARY_MD="${RESULTS_DIR}/nsys_summary_${TS}.md"
LOG_FILE="${RESULTS_DIR}/nsys_sweep_${TS}.log"

# route all stderr/stdout to both console (INFO level) and log file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date +%T)] INFO  nsys profiling sweep started (ts=${TS})"
echo "[$(date +%T)] INFO  results dir: ${RESULTS_DIR}"
nsys --version | head -1
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

# Workloads: (tag, bs, d_out, d_in, hp_ratio, iters)
WORKLOADS=(
    "decode    1     11008 4096 0.00 50"
    "mid       64    11008 4096 0.00 50"
    "prefill   2048  11008 4096 0.00 30"
    "mid_hp5   64    11008 4096 0.05 50"
)

declare -a REPORT_TAGS=()
for wl in "${WORKLOADS[@]}"; do
    # shellcheck disable=SC2086
    set -- $wl
    TAG=$1 ; BS=$2 ; DOUT=$3 ; DIN=$4 ; HP=$5 ; ITERS=$6
    REP_BASE="${RESULTS_DIR}/nsys_${TS}_${TAG}"
    echo
    echo "[$(date +%T)] INFO  === workload ${TAG}: bs=${BS} d_out=${DOUT} d_in=${DIN} hp=${HP} iters=${ITERS} ==="

    ZIP_PROFILE_BS=${BS} \
    ZIP_PROFILE_DOUT=${DOUT} \
    ZIP_PROFILE_DIN=${DIN} \
    ZIP_PROFILE_HP=${HP} \
    ZIP_PROFILE_ITERS=${ITERS} \
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=false \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --sample=none \
        --cpuctxsw=none \
        --force-overwrite=true \
        --output="${REP_BASE}" \
        python "${DRIVER_PY}"

    # Export stats. Available reports: cuda_gpu_kern_sum, nvtx_sum, nvtx_gpu_proj_sum,
    # cuda_api_sum. nvtx_gpu_proj_sum attributes GPU time to NVTX ranges --
    # exactly what we need.
    nsys stats \
        --report cuda_gpu_kern_sum \
        --report nvtx_gpu_proj_sum \
        --report cuda_api_sum \
        --format csv \
        --output "${REP_BASE}" \
        "${REP_BASE}.nsys-rep" >/dev/null

    REPORT_TAGS+=("${TAG}|${REP_BASE}")
done

echo
echo "[$(date +%T)] INFO  building markdown summary -> ${SUMMARY_MD}"

python "${SCRIPT_DIR}/summarize_nsys.py" \
    --tags "$(IFS=';'; echo "${REPORT_TAGS[*]}")" \
    --output "${SUMMARY_MD}" \
    --ts "${TS}"

echo "[$(date +%T)] INFO  done. see ${SUMMARY_MD}"
