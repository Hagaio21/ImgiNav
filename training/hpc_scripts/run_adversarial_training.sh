#!/bin/bash
#BSUB -J adversarial_training
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/adversarial_training.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/adversarial_training.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"  # 16GB per CPU = 128GB total
#BSUB -gpu "num=1"
#BSUB -W 72:00  # 72 hours (adversarial training with multiple iterations can take a long time)
#BSUB -q gpuv100

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
CONFIG_FILE="${BASE_DIR}/experiments/diffusion/adversarial/adversarial_training.yaml"
PYTHON_SCRIPT="${BASE_DIR}/training/train_adversarial.py"
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
echo "Adversarial Training Pipeline"
echo "Config: ${CONFIG_FILE}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "=========================================="
echo ""
echo "Pipeline Overview:"
echo "  This pipeline iteratively:"
echo "    1. Generates fake latents from current diffusion model"
echo "    2. Trains discriminator on real vs fake latents"
echo "    3. Fine-tunes diffusion with discriminator loss (gradient reconnection)"
echo "    4. Repeats for configurable number of iterations"
echo ""
echo "Note: Each iteration trains a NEW discriminator from scratch."
echo "      To resume from previous discriminator, modify the config."
echo "=========================================="

cd "${BASE_DIR}"
python -m training.train_adversarial "${CONFIG_FILE}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Adversarial Training COMPLETE - SUCCESS"
    echo "End: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Adversarial Training FAILED"
    echo "Exit code: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

