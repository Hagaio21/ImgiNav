#!/bin/bash
#BSUB -J diffusion_capacity_unet128_d5_rb3
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_capacity_unet128_d5_rb3.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/diffusion_capacity_unet128_d5_rb3.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=32000]"
#BSUB -gpu "num=1"
#BSUB -W 48:00  # 48 hours (deeper model with more resblocks needs more time)
#BSUB -q gpul40s

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
CONFIG_FILE="${BASE_DIR}/experiments/diffusion/ablation/capacity_unet128_d5_rb3.yaml"
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
echo "Diffusion Capacity Ablation: UNet 128 (d5, rb3)"
echo "Config: ${CONFIG_FILE}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "Configuration:"
echo "  - base_channels: 128"
echo "  - depth: 5"
echo "  - num_res_blocks: 3"
echo "  - time_dim: 128"
echo "Note: Deeper model with more resblocks - slower training expected"
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

