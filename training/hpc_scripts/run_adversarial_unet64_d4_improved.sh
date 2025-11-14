#!/bin/bash
#BSUB -J adversarial_unet64_d4_improved
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/adversarial_unet64_d4_improved.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/adversarial_unet64_d4_improved.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 72:00
#BSUB -q gpul40s

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
CONFIG_FILE="${BASE_DIR}/experiments/diffusion/adversarial/adversarial_unet64_d4_improved.yaml"
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
echo "Adversarial Training: UNet 64 (d4, improved)"
echo "Config: ${CONFIG_FILE}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"
python -m training.train_adversarial "${CONFIG_FILE}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training COMPLETE - SUCCESS"
    echo "End: $(date)"
else
    echo "Training FAILED - Exit code: ${EXIT_CODE}"
    exit $EXIT_CODE
fi

