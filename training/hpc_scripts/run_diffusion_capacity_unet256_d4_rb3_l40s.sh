#!/bin/bash
#BSUB -J diffusion_capacity_unet256_d4_rb3
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_capacity_unet256_d4_rb3.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_capacity_unet256_d4_rb3.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00  # 48 hours (large model with more capacity needs more time)
#BSUB -q gpul40s

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
CONFIG_FILE="${BASE_DIR}/experiments/diffusion/ablation/capacity_unet256_d4_rb3.yaml"
PYTHON_SCRIPT="${BASE_DIR}/training/train_diffusion.py"
LOG_DIR="${BASE_DIR}/training/hpc_scripts/logs"

mkdir -p "${LOG_DIR}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

echo "=========================================="
echo "Diffusion Capacity Ablation: UNet 256 (d4, rb3, td256)"
echo "Config: ${CONFIG_FILE}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "Improvements:"
echo "  - num_res_blocks: 3 (was 2)"
echo "  - time_dim: 256 (was 128)"
echo "  - num_steps: 500 (same as baseline)"
echo "Note: Larger model - slower training expected"
echo "=========================================="

cd "${BASE_DIR}"
python -m training.train_diffusion "${CONFIG_FILE}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Training COMPLETE - SUCCESS"
    echo "End: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training FAILED with exit code: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

