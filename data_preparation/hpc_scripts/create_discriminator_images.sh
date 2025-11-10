#!/bin/bash
#BSUB -J create_discriminator_images
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_discriminator_images.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_discriminator_images.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpuv100

set -euo pipefail

BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/data_preparation/create_discriminator_images.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Configuration - UPDATE THESE PATHS
MANIFEST="/work3/s233249/ImgiNav/datasets/layouts.csv"
DIFFUSION_CHECKPOINT="/work3/s233249/ImgiNav/experiments/diffusion/stage2/stage2_unet128_d4/diffusion_stage2_unet128_d4_checkpoint_best.pt"
DIFFUSION_CONFIG="${BASE_DIR}/experiments/diffusion/stage2/stage2_unet128_d4.yaml"
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/discriminator"

# Parameters
NUM_REAL=5000
NUM_GENERATED=5000
BATCH_SIZE=8
NUM_STEPS=50
SEED=42
LAYOUT_COLUMN="layout_path"

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
echo "Creating Discriminator Dataset (Images)"
echo "========================================="
echo "Manifest: ${MANIFEST}"
echo "Diffusion model: ${DIFFUSION_CHECKPOINT}"
echo "Diffusion config: ${DIFFUSION_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Real images: ${NUM_REAL}"
echo "Generated images: ${NUM_GENERATED}"
echo "Batch size: ${BATCH_SIZE}"
echo "DDIM steps: ${NUM_STEPS}"
echo ""

python "${SCRIPT_PATH}" \
    --manifest "${MANIFEST}" \
    --diffusion_checkpoint "${DIFFUSION_CHECKPOINT}" \
    --diffusion_config "${DIFFUSION_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_real "${NUM_REAL}" \
    --num_generated "${NUM_GENERATED}" \
    --batch_size "${BATCH_SIZE}" \
    --num_steps "${NUM_STEPS}" \
    --device "cuda" \
    --seed "${SEED}" \
    --layout_column "${LAYOUT_COLUMN}"

echo ""
echo "========================================="
echo "Discriminator dataset creation completed"
echo "========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Directory structure:"
echo "  ${OUTPUT_DIR}/real/          (${NUM_REAL} real images)"
echo "  ${OUTPUT_DIR}/generated/     (${NUM_GENERATED} generated images)"
echo "  ${OUTPUT_DIR}/manifest.csv   (manifest with valid/invalid labels)"
echo ""
echo "Manifest columns:"
echo "  - image_path: Path to image"
echo "  - label: 1 for real/valid, 0 for generated/invalid"
echo "  - is_valid: Boolean flag"
echo "  - type: 'real' or 'generated'"

