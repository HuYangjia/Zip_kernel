#!/bin/bash
# Launch r66 140-shape bench (C.1+C.2+C.3+C.4 cumulative).
source /root/miniconda3/etc/profile.d/conda.sh
conda activate zip
cd /root
export PYTHONPATH=/root
exec python -m kernel.cuda_kernel.benchmarks.bench_qwen3_shapes \
    --full \
    --ts 1 32 128 512 \
    --out-root /root/Zip_kernel/cuda_kernel/logs/r66_path_c/bench
