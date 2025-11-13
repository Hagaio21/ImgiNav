#!/bin/bash
#BSUB -J stage2_discriminator_test
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/stage2_discriminator_test.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/training/hpc_scripts/logs/stage2_discriminator_test.%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=2000]"  # 2GB per CPU = 8GB total (actual usage ~3GB)
#BSUB -gpu "num=1"
#BSUB -W 2:00  # 2 hours for quick test
#BSUB -q gpuv100

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
CONFIG_FILE="${BASE_DIR}/experiments/diffusion/stage2_discriminator/stage2_discriminator_test.yaml"
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
echo "Stage 2 Discriminator Training: TEST RUN"
echo "Config: ${CONFIG_FILE}"
echo "Start: $(date)"
echo "=========================================="

cd "${BASE_DIR}"
python -m training.train_stage2_discriminator "${CONFIG_FILE}" \
    --num_iterations 1 \
    --num_samples 64 \
    --discriminator_batch_size 32 \
    --discriminator_max_steps 2 \
    --diffusion_max_steps 2 \
    --discriminator_eval_interval 1 \
    --diffusion_eval_interval 1 \
    --diffusion_sample_interval 1 \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.0001 \
    --generation_batch_size 32 \
    --generation_steps 50

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Test Training COMPLETE - SUCCESS"
    echo "End: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "Test Training FAILED"
    echo "Exit code: ${EXIT_CODE}"
    echo "End: $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi

