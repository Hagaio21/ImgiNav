#!/bin/bash
#BSUB -J stage2_discriminator_unet64_d4
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/stage2_discriminator_unet64_d4.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/stage2_discriminator_unet64_d4.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 72:00  # 72 hours (adversarial training takes longer - 3 iterations × ~24h each)
#BSUB -q gpuv100

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
CONFIG_FILE="${BASE_DIR}/experiments/diffusion/stage2_discriminator/stage2_discriminator_unet64_d4.yaml"
PYTHON_SCRIPT="${BASE_DIR}/training/train_stage2_discriminator.py"
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
echo "Stage 2 Discriminator Training: UNet 64 (d4)"
echo "Config: ${CONFIG_FILE}"
echo "Working directory: ${BASE_DIR}"
echo "Start: $(date)"
echo "Adversarial Training Configuration:"
echo "  - Iterations: 3 (fixed)"
echo "  - Samples per iteration: 5000"
echo "  - Discriminator batch size: 512 (for faster convergence)"
echo "  - Discriminator max steps: 50000 (trains until convergence)"
echo "  - Diffusion max steps: 100000 (trains until convergence)"
echo "  - Early stopping patience: 10 evaluations"
echo "  - Early stopping min delta: 0.0001"
echo "  - Generation batch size: 32"
echo "  - Generation steps: 100 DDIM"
echo "=========================================="

cd "${BASE_DIR}"
python -m training.train_stage2_discriminator "${CONFIG_FILE}" \
    --num_iterations 3 \
    --num_samples 5000 \
    --discriminator_batch_size 512 \
    --discriminator_max_steps 50000 \
    --diffusion_max_steps 100000 \
    --discriminator_eval_interval 500 \
    --diffusion_eval_interval 1000 \
    --diffusion_sample_interval 5000 \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.0001 \
    --generation_batch_size 32 \
    --generation_steps 100

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Training COMPLETE - SUCCESS"
    echo "End: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training FAILED"
    echo "Exit code: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

