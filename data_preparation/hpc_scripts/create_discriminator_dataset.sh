#!/bin/bash
#BSUB -J create_discriminator_dataset
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_discriminator_dataset.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_discriminator_dataset.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpuv100

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/scripts/create_discriminator_dataset.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Configuration - UPDATE THESE PATHS
MANIFEST="/work3/s233249/ImgiNav/datasets/augmented/manifest.csv"
AUTOENCODER_CHECKPOINT="/work3/s233249/ImgiNav/experiments/phase1/phase1_6_AE_normalized/phase1_6_AE_normalized_checkpoint_best.pt"
DIFFUSION_CHECKPOINT="/work3/s233249/ImgiNav/experiments/diffusion/stage2/stage2_unet128_d4/diffusion_stage2_unet128_d4_checkpoint_best.pt"
DIFFUSION_CONFIG="${BASE_DIR}/experiments/diffusion/stage2/stage2_unet128_d4.yaml"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/discriminator_dataset"

# Parameters
NUM_SAMPLES=5000
BATCH_SIZE=32
NUM_STEPS=100
SEED=42

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: Manifest not found: ${MANIFEST}" >&2
    exit 1
fi

if [ ! -f "${AUTOENCODER_CHECKPOINT}" ]; then
    echo "ERROR: Autoencoder checkpoint not found: ${AUTOENCODER_CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "${DIFFUSION_CHECKPOINT}" ]; then
    echo "ERROR: Diffusion checkpoint not found: ${DIFFUSION_CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "${DIFFUSION_CONFIG}" ]; then
    echo "ERROR: Diffusion config not found: ${DIFFUSION_CONFIG}" >&2
    exit 1
fi

module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X
export MKL_INTERFACE_LAYER=LP64

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

cd "${BASE_DIR}"

echo "========================================="
echo "Creating Discriminator Dataset"
echo "========================================="
echo "Manifest: ${MANIFEST}"
echo "Autoencoder: ${AUTOENCODER_CHECKPOINT}"
echo "Diffusion model: ${DIFFUSION_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "Samples: ${NUM_SAMPLES} real + ${NUM_SAMPLES} bad"
echo ""

python "${SCRIPT_PATH}" \
    --manifest "${MANIFEST}" \
    --autoencoder_checkpoint "${AUTOENCODER_CHECKPOINT}" \
    --diffusion_checkpoint "${DIFFUSION_CHECKPOINT}" \
    --diffusion_config "${DIFFUSION_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --num_steps "${NUM_STEPS}" \
    --device "cuda" \
    --seed "${SEED}"

echo ""
echo "========================================="
echo "Discriminator dataset creation completed"
echo "========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Next step: Train discriminator using:"
echo "  python training/train_discriminator.py \\"
echo "    --real_latents ${OUTPUT_DIR}/real_latents_all.pt \\"
echo "    --fake_latents ${OUTPUT_DIR}/bad_latents_all.pt \\"
echo "    --output_dir /path/to/discriminator_output"

