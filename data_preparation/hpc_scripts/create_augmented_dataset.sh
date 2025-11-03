#!/bin/bash
#BSUB -J create_augmented_dataset
#BSUB -o /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_augmented_dataset.%J.out
#BSUB -e /work3/s233249/ImgiNav/ImgiNav/data_preparation/hpc_scripts/logs/create_augmented_dataset.%J.err
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -gpu "num=1"
#BSUB -W 12:00
#BSUB -q gpuv100

set -euo pipefail
export MKL_INTERFACE_LAYER=LP64

# ----------------------------------------------------------------------
# Configuration
BASE_DIR="/work3/s233249/ImgiNav/ImgiNav"
SCRIPT_PATH="${BASE_DIR}/data_preparation/create_augmented_dataset.py"
LOG_DIR="${BASE_DIR}/data_preparation/hpc_scripts/logs"

# Autoencoder config and checkpoint (needed to encode images to latents)
AUTOENCODER_CONFIG="/work3/s233249/ImgiNav/ImgiNav/experiments/autoencoders/phase1/phase1_5_AE_final.yaml"
AUTOENCODER_CHECKPOINT="/work3/s233249/ImgiNav/experiments/phase1/phase1_5_AE_final/phase1_5_AE_final_checkpoint_best.pt"

# Input manifest (original dataset)
INPUT_MANIFEST="/work3/s233249/ImgiNav/datasets/layouts_latents.csv"

# Output directory for augmented dataset
OUTPUT_DIR="/work3/s233249/ImgiNav/datasets/augmented"

# Augmentation parameters
USE_MIRROR_ROTATION=true  # Use mirror + 90-degree rotations (7 variants per sample)

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo "Creating Augmented Dataset"
echo "========================================="
echo "Script: ${SCRIPT_PATH}"
echo "Autoencoder config: ${AUTOENCODER_CONFIG}"
echo "Autoencoder checkpoint: ${AUTOENCODER_CHECKPOINT}"
echo "Input manifest: ${INPUT_MANIFEST}"
echo "Output directory: ${OUTPUT_DIR}"
if [ "${USE_MIRROR_ROTATION}" = "true" ]; then
    echo "Augmentation method: Mirror + 90-degree rotations (7 variants per sample)"
    echo "  - Original + 3 rotations (90°, 180°, 270°)"
    echo "  - Mirror + 3 rotations (mirrored 90°, 180°, 270°)"
else
    echo "Augmentation method: Legacy random augmentation (3 variants per sample, deprecated)"
fi
echo ""

# Check required files
if [ ! -f "${SCRIPT_PATH}" ]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

if [ ! -f "${AUTOENCODER_CONFIG}" ]; then
    echo "ERROR: Autoencoder config not found: ${AUTOENCODER_CONFIG}" >&2
    exit 1
fi

if [ ! -f "${AUTOENCODER_CHECKPOINT}" ]; then
    echo "ERROR: Autoencoder checkpoint not found: ${AUTOENCODER_CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "${INPUT_MANIFEST}" ]; then
    echo "ERROR: Input manifest not found: ${INPUT_MANIFEST}" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# Conda environment
module load cuda/11.8
module load cudnn/v8.6.0.163-prod-cuda-11.X

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate imginav || conda activate scenefactor || exit 1
fi

# ----------------------------------------------------------------------
# Run augmentation
cd "${BASE_DIR}"

echo "Starting augmentation process..."
echo "Note: This will filter out white images (>95% whitish pixels) and empty images"
echo ""

# Build command with optional --use-mirror-rotation flag
CMD="python \"${SCRIPT_PATH}\" \
    --autoencoder-config \"${AUTOENCODER_CONFIG}\" \
    --autoencoder-checkpoint \"${AUTOENCODER_CHECKPOINT}\" \
    --dataset-manifest \"${INPUT_MANIFEST}\" \
    --output-dir \"${OUTPUT_DIR}\" \
    --batch-size 32 \
    --num-workers 8 \
    --device cuda \
    --overwrite"

if [ "${USE_MIRROR_ROTATION}" = "true" ]; then
    CMD="${CMD} --use-mirror-rotation"
fi

eval ${CMD}

echo ""
echo "========================================="
echo "Augmented dataset creation completed"
echo "========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Manifest: ${OUTPUT_DIR}/manifest.csv"
echo ""
echo "You can now update your training configs to use:"
echo "  manifest: ${OUTPUT_DIR}/manifest.csv"

